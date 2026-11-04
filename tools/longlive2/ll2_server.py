"""LongLive 2.0 live server (runs on the GPU box): websocket in/out, just-in-time generation at a fixed playout rate.

  /root/ll2/bin/python ll2_server.py [--port 8765] [--height 576 --width 1024 --ckpt S2 --sage --block 8 ...]
Reach it through an SSH tunnel (cloud/client/ll2_client.py does it). Protocol (one client at a time):
  client -> server  JSON text messages:
     {"type": "prompt", "text": "...", "cut": false}      new prompt (cut = hard scene cut)
     {"type": "set", "mode": "t2v"|"v2v", "strength": 0.7, "person": true}
     {"type": "image", "data": "<base64 png/jpg>", "prompt": "..."}   restart the stream from a reference image
     {"type": "reset", "prompt": "..."}                   restart (text only)
     {"type": "start", "prompt": ..., "mode": ..., "strength": ..., "image": optional base64}
  client -> server  binary: b"F" + uint32 input id + JPEG webcam frame (v2v), b"M" + JPEG person mask (white = person; optional,
                    computed on the client with MediaPipe; used when "person" is on)
  server -> client  binary: 16-byte header (uint32 frame seq, prompt version, chunk, source input id or 0xFFFFFFFF)
                    + JPEG; the input id lets the client pair each output frame with the webcam frame it came from
  server -> client  JSON {"type": "stats", ...} once per chunk
Generation runs one chunk ahead of the playout clock (fps), never more: every queued frame is latency.
"""

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8765)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--height", type=int, default=576)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--ckpt", default="S2")
p.add_argument("--steps", type=int, default=None)
p.add_argument("--block", type=int, default=8, help="latent frames per chunk (8 native; 4 halves latency)")
p.add_argument("--window", type=int, default=32)
p.add_argument("--kvq", action="store_true", help="NVFP4 KV cache (less memory; bf16 KV is faster on a 5090)")
p.add_argument("--sage", action="store_true")
p.add_argument("--no-compile", action="store_true")
p.add_argument("--clean-every", type=int, default=1)
p.add_argument("--fps", type=float, default=24.0)
p.add_argument("--ahead", type=int, default=34, help="unplayed frames kept at the client (in flight + queued) before the "
               "next chunk starts: must cover one chunk's generation (~24 frames at 24 fps) + jitter")
p.add_argument("--jpeg", type=int, default=90)
p.add_argument("--recache-frames", type=int, default=16, help="latent frames rebuilt under a new prompt (0 = off)")
p.add_argument("--prompt", default="a cinematic shot of a cozy living room, warm light")
a = p.parse_args()

from ll2_stream import LL2Engine  # noqa: E402

eng = LL2Engine(height=a.height, width=a.width, ckpt=a.ckpt, steps=a.steps, kv_quant=a.kvq, local_attn=a.window,
                block=a.block, sage=a.sage, compile=not a.no_compile, clean_every=a.clean_every,
                recache_frames=max(1, a.recache_frames))
FPC = 4 * a.block  # video frames per chunk


def warmup():
    t = time.time()
    eng.reset(prompt=a.prompt)
    for _ in range(6):  # compile + a full KV window
        eng.step()
    blank = [np.zeros((a.height, a.width, 3), np.uint8)] * FPC
    eng.reset(prompt=a.prompt)
    for _ in range(3):
        eng.step(input_frames=blank, strength=0.7)
    print(f"[server] warm-up {time.time() - t:.0f}s, last chunk {eng.stats}", flush=True)


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.prompt, self.cut, self.version = a.prompt, False, 0
        self.mode, self.strength, self.person = "t2v", 0.7, False
        self.reset_req = ("text", None)  # (kind, image)
        self.inputs = deque(maxlen=4 * FPC)  # latest webcam frames (RGB)
        self.mask = None  # latest person mask (uint8 HxW)
        self.out = None  # asyncio queue (set by the connection)
        self.loop = None
        self.running = False
        self.t0 = None
        self.sent = 0
        self.recache_info = None
        self.consumed = None
        self.last_fid = 0  # frames the client has played or dropped (reported by the client)


S = State()


def person_map():
    """Per-latent-pixel strength map from the client's person mask: LL2_PERSON_KEEP on the person (kept, so arms and
    hands enter the scene), the configured strength elsewhere (regenerated)."""
    m = S.mask
    if m is None:
        return None
    m = cv2.resize(m, (a.width // 16, a.height // 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    keep = float(os.environ.get("LL2_PERSON_KEEP", "0.6"))
    return np.where(m > 0.5, keep, S.strength).astype(np.float32)


def gen_loop():
    period = 1.0 / a.fps
    while True:
        if not S.running:
            time.sleep(0.02)
            continue
        with S.lock:
            reset, S.reset_req = S.reset_req, None
            prompt, cut, version = S.prompt, S.cut, S.version
            S.cut = False
            mode, strength, person = S.mode, S.strength, S.person
            inputs = list(S.inputs)
        if reset is not None:
            eng.reset(prompt=prompt, image=reset[1])
            S.t0, S.sent, S.consumed, S.last_fid = time.perf_counter(), 0, None, 0
        # pace: keep at most `ahead` chunks of frames in front of the playout clock
        while S.running and S.t0 is not None:
            played = S.consumed if S.consumed is not None else (time.perf_counter() - S.t0) / period
            if S.sent - played <= a.ahead or S.version != version:  # a change: generate now (client snaps)
                break
            time.sleep(0.003)
        with S.lock:  # pick up a change that arrived while waiting
            prompt, version = S.prompt, S.version
            cut = cut or S.cut
            S.cut = False
            mode, strength, person = S.mode, S.strength, S.person
        if prompt != eng.prompt or cut:
            eng.set_prompt(prompt, cut=cut, recache=a.recache_frames > 0)
        kw, in_ids = {}, None
        if mode == "v2v" and inputs:
            new = [x for x in inputs if x[0] > S.last_fid]  # frames not used yet (keeps the encoder continuous)
            old = [x for x in inputs if x[0] <= S.last_fid]
            sel = (old + new)[-FPC:] if len(new) < FPC else new[-FPC:]
            sel = [sel[0]] * (FPC - len(sel)) + sel
            S.last_fid = sel[-1][0]
            in_ids = [i for i, _ in sel]
            kw = {"input_frames": [f for _, f in sel], "strength": strength}
            if person:
                m = person_map()
                if m is not None:
                    kw["strength_map"] = m
        t = time.perf_counter()
        frames = eng.step(**kw, abort=lambda: S.version != version or S.reset_req is not None)
        dt = time.perf_counter() - t
        if frames is None:  # a change arrived mid-chunk: regenerate this position with the new input
            continue
        played = (time.perf_counter() - S.t0) / period
        if S.sent < played - 2:  # we fell behind the clock: resync (avoids an ever-growing debt)
            S.t0 = time.perf_counter() - S.sent * period
        ENC_Q.put((frames, S.sent, version, eng.chunk, in_ids))  # JPEG encoding + sending in its own thread
        S.sent += len(frames)
        if eng.chunk % 5 == 0:
            print(f"[server] chunk {eng.chunk}: mode {mode}, inputs {len(inputs)}, v2v used {eng.stats.get('v2v')}, "
                  f"person map {'strength_map' in kw}, prompt v{version}: {prompt[:50]}", flush=True)
        stats = {"type": "stats", "chunk": eng.chunk, "gen_ms": round(1000 * dt), "dit_ms": round(eng.stats.get("dit_ms", 0)),
                 "frames": len(frames), "gen_fps": round(len(frames) / dt, 1), "mode": mode, "version": version,
                 "ahead_frames": round(S.sent - (time.perf_counter() - S.t0) / period)}
        if S.recache_info is not None:
            stats.update(S.recache_info)
        stats["recache_ms"] = eng.stats.get("recache_ms")
        ENC_Q.put(json.dumps(stats))


import queue  # noqa: E402
ENC_Q = queue.Queue()


def enc_loop():
    while True:
        item = ENC_Q.get()
        if isinstance(item, str):
            msgs = [item]
        else:
            frames, seq0, version, chunk, in_ids = item
            msgs = []
            for i, f in enumerate(frames):
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, a.jpeg])
                src = in_ids[min(i, len(in_ids) - 1)] if in_ids else 0xFFFFFFFF
                msgs.append(struct.pack("<IIII", seq0 + i, version, chunk, src) + jpg.tobytes())
        if S.loop is not None and S.out is not None:
            for m in msgs:
                S.loop.call_soon_threadsafe(S.out.put_nowait, m)


def decode_image(b64):
    data = np.frombuffer(base64.b64decode(b64.split(",")[-1]), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


async def handler(ws):
    import websockets
    S.loop, S.out = asyncio.get_running_loop(), asyncio.Queue()
    print("[server] client connected", flush=True)

    async def sender():
        while True:
            m = await S.out.get()
            await ws.send(m)

    st = asyncio.create_task(sender())
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                kind, body = msg[:1], np.frombuffer(msg[1:], np.uint8)
                if kind == b"M":
                    S.mask = cv2.imdecode(body, cv2.IMREAD_GRAYSCALE)
                else:
                    fid = struct.unpack("<I", msg[1:5])[0]
                    img = cv2.imdecode(np.frombuffer(msg[5:], np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with S.lock:
                            S.inputs.append((fid, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                continue
            j = json.loads(msg)
            t = j.get("type")
            with S.lock:
                if t in ("start", "reset", "image"):
                    S.prompt = j.get("prompt") or S.prompt
                    S.mode = j.get("mode", S.mode)
                    S.strength = float(j.get("strength", S.strength))
                    S.person = bool(j.get("person", S.person))
                    img = j.get("image") or j.get("data")
                    S.reset_req = ("image" if img else "text", decode_image(img) if img else None)
                    S.version += 1
                    S.running = True
                elif t == "prompt":
                    S.prompt, S.cut = j["text"], bool(j.get("cut"))
                    S.version += 1
                elif t == "played":
                    S.consumed = int(j["n"])
                    continue
                elif t == "set":
                    S.mode = j.get("mode", S.mode)
                    S.strength = float(j.get("strength", S.strength))
                    S.person = bool(j.get("person", S.person))
            if t == "played":
                continue
            print(f"[server] {t}: {json.dumps({k: v for k, v in j.items() if k not in ('image', 'data')})[:200]}", flush=True)
    except websockets.ConnectionClosed:
        pass
    finally:
        S.running, S.out = False, None
        st.cancel()
        print("[server] client disconnected", flush=True)


async def main():
    import websockets
    async with websockets.serve(handler, a.host, a.port, max_size=2 ** 25, ping_interval=20):
        print(f"[server] ready on ws://{a.host}:{a.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    warmup()
    threading.Thread(target=gen_loop, daemon=True).start()
    threading.Thread(target=enc_loop, daemon=True).start()
    asyncio.run(main())
