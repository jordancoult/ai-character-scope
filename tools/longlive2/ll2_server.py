"""LongLive 2.0 live server (runs on the GPU box): websocket in/out, just-in-time generation at a fixed playout rate.

  /root/ll2/bin/python ll2_server.py [--port 8765] [--height 576 --width 1024 --ckpt S2 --sage --block 8 ...]
Reach it through an SSH tunnel (cloud/client/ll2_client.py does it). Protocol (one client at a time):
  client -> server  JSON text messages:
     {"type": "prompt", "text": "...", "cut": false}      new prompt (cut = hard scene cut)
     {"type": "set", "mode": "t2v"|"v2v", "strength": 0.7, "person": true}
     {"type": "image", "data": "<base64 png/jpg>", "prompt": "..."}   restart the stream from a reference image
     {"type": "reset", "prompt": "..."}                   restart (text only)
     {"type": "start", "prompt": ..., "mode": ..., "strength": ..., "image": optional base64}
  client -> server  binary: b"F" + JPEG webcam frame (v2v), b"M" + JPEG person mask (white = person; optional,
                    computed on the client with MediaPipe; used when "person" is on)
  server -> client  binary: 12-byte header (uint32 frame seq, uint32 prompt version, uint32 chunk) + JPEG
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
p.add_argument("--ahead", type=float, default=1.0, help="chunks generated ahead of playout (latency vs. underruns)")
p.add_argument("--jpeg", type=int, default=90)
p.add_argument("--prompt", default="a cinematic shot of a cozy living room, warm light")
a = p.parse_args()

from ll2_stream import LL2Engine  # noqa: E402

eng = LL2Engine(height=a.height, width=a.width, ckpt=a.ckpt, steps=a.steps, kv_quant=a.kvq, local_attn=a.window,
                block=a.block, sage=a.sage, compile=not a.no_compile, clean_every=a.clean_every)
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


S = State()


def person_map():
    """Per-latent-pixel strength map from the client's person mask: LL2_PERSON_KEEP on the person (kept, so arms and
    hands enter the scene), the configured strength elsewhere (regenerated)."""
    m = S.mask
    if m is None:
        return None
    m = cv2.resize(m, (a.width // 16, a.height // 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    keep = float(os.environ.get("LL2_PERSON_KEEP", "0.35"))
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
            S.t0, S.sent = time.perf_counter(), 0
        # pace: keep at most `ahead` chunks of frames in front of the playout clock
        while S.running and S.t0 is not None:
            played = (time.perf_counter() - S.t0) / period
            if S.sent - played <= a.ahead * FPC:
                break
            time.sleep(0.003)
        if prompt != eng.prompt or cut:
            eng.set_prompt(prompt, cut=cut)
        kw = {}
        if mode == "v2v" and inputs:
            frames = inputs[-FPC:]
            frames = [frames[0]] * (FPC - len(frames)) + frames
            kw = {"input_frames": frames, "strength": strength}
            if person:
                m = person_map()
                if m is not None:
                    kw["strength_map"] = m
        t = time.perf_counter()
        frames = eng.step(**kw)
        dt = time.perf_counter() - t
        played = (time.perf_counter() - S.t0) / period
        if S.sent < played - 2:  # we fell behind the clock: resync (avoids an ever-growing debt)
            S.t0 = time.perf_counter() - S.sent * period
        payload = []
        for f in frames:
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, a.jpeg])
            payload.append(struct.pack("<III", S.sent, version, eng.chunk) + jpg.tobytes())
            S.sent += 1
        stats = {"type": "stats", "chunk": eng.chunk, "gen_ms": round(1000 * dt), "dit_ms": round(eng.stats.get("dit_ms", 0)),
                 "frames": len(frames), "gen_fps": round(len(frames) / dt, 1), "mode": mode, "version": version,
                 "ahead_frames": round(S.sent - (time.perf_counter() - S.t0) / period)}
        if S.loop is not None and S.out is not None:
            for m in payload:
                S.loop.call_soon_threadsafe(S.out.put_nowait, m)
            S.loop.call_soon_threadsafe(S.out.put_nowait, json.dumps(stats))


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
                    img = cv2.imdecode(body, cv2.IMREAD_COLOR)
                    if img is not None:
                        with S.lock:
                            S.inputs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
                elif t == "set":
                    S.mode = j.get("mode", S.mode)
                    S.strength = float(j.get("strength", S.strength))
                    S.person = bool(j.get("person", S.person))
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
    asyncio.run(main())
