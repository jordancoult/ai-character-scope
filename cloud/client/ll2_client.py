"""Client for the LongLive 2.0 live server (tools/longlive2/ll2_server.py): input -> GPU box -> 24 fps playout.

  cloud/facefix/.venv/bin/python cloud/client/ll2_client.py --ssh "ssh -p PORT root@HOST" [--mode t2v|v2v]
      [--input /dev/video0 | clip.mp4] [--image ref.png] [--prompt "..."] [--strength 0.7] [--person]
      [--output ffplay | out.mp4] [--seconds 0] [--script "8:a new prompt;16:/cut a different scene;20:/strength 0.5"]
Type while it runs (+ Enter):
  any text           new prompt (smooth transition)         /cut TEXT      hard scene cut to TEXT
  /strength 0.6      v2v noise level (0 keep input .. 1 new) /mode t2v|v2v  switch input mode
  /person on|off     keep yourself (low noise) + regenerate the background (v2v; uses MediaPipe on the laptop)
  /image PATH        restart from a reference image          /reset         restart (text)          /q quit
Without --ssh it connects to --url directly (e.g. an existing tunnel).
"""

import argparse
import asyncio
import base64
import json
import os
import shlex
import struct
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--url", default="ws://127.0.0.1:8765")
p.add_argument("--ssh", default=None, help="ssh command to the GPU box; opens a tunnel to its port 8765")
p.add_argument("--ssh-key", default=os.path.expanduser("~/personal/ai-livestream/.secrets/runpod_ed25519"))
p.add_argument("--local-port", type=int, default=8765)
p.add_argument("--mode", default="t2v", choices=["t2v", "v2v"])
p.add_argument("--input", default=None, help="v2v source: /dev/videoN or a video file (looped)")
p.add_argument("--image", default=None, help="reference image: the stream starts from it (I2V)")
p.add_argument("--prompt", default="a cinematic shot of a cozy living room, warm light")
p.add_argument("--strength", type=float, default=0.7)
p.add_argument("--person", action="store_true", help="v2v: keep the person (low noise), regenerate the rest")
p.add_argument("--size", default="1024x576", help="input frames are cropped/resized to this (match the server)")
p.add_argument("--fps", type=float, default=24.0)
p.add_argument("--buffer", type=int, default=6, help="frames buffered before playout starts")
p.add_argument("--output", default="ffplay", help="ffplay, an .mp4 path, or 'ffplay,out.mp4'")
p.add_argument("--seconds", type=float, default=0)
p.add_argument("--script", default="", help="timed commands 'T:cmd;T:cmd' (seconds from start; same syntax as typing)")
p.add_argument("--side-by-side", action="store_true", help="recording: input | output (v2v)")
a = p.parse_args()
W, H = (int(v) for v in a.size.split("x"))


def open_tunnel():
    if not a.ssh:
        return None
    parts = shlex.split(a.ssh)
    cmd = parts[:1] + ["-i", a.ssh_key, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                       "-o", "LogLevel=ERROR", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                       "-N", "-L", f"{a.local_port}:127.0.0.1:8765"] + parts[1:]
    t = subprocess.Popen(cmd)
    time.sleep(3)
    if t.poll() is not None:
        raise SystemExit("[client] ssh tunnel failed")
    return t


class Input:
    """Latest input frame (webcam or looped file at its own fps), center-cropped to WxH."""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(int(src[10:]) if src.startswith("/dev/video") and src[10:].isdigit() else src)
        self.is_file = not src.startswith("/dev/video")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame, self.stop = None, False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            h, w = f.shape[:2]
            s = max(W / w, H / h)
            f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            y0, x0 = (f.shape[0] - H) // 2, (f.shape[1] - W) // 2
            self.frame = f[y0:y0 + H, x0:x0 + W]
            if self.is_file:
                time.sleep(1.0 / self.fps)


class Segmenter:
    def __init__(self):
        import mediapipe as mp
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def __call__(self, bgr):
        m = self.seg.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).segmentation_mask
        return (np.clip(m, 0, 1) * 255).astype(np.uint8)


async def main():
    import websockets
    tunnel = open_tunnel()
    inp = Input(a.input) if a.input else None
    seg = Segmenter() if a.person else None
    state = {"mode": a.mode, "person": a.person, "version": 0, "sent_t": {}}
    q = deque()  # (frame_bgr, version)
    stop = threading.Event()
    sinks = []
    outs = [o for o in a.output.split(",") if o]
    ow = W * 2 if (a.side_by_side and inp) else W

    def ffmpeg_sink(path):
        return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{H}",
                                 "-r", str(a.fps), "-i", "-", "-vf", "scale='min(1920,iw)':-2", "-c:v", "libx264",
                                 "-crf", "18", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
                                 "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for o in outs:
        if o == "ffplay":
            sinks.append(subprocess.Popen(["ffplay", "-v", "error", "-window_title", "LongLive 2.0", "-f", "rawvideo",
                                           "-pixel_format", "bgr24", "-video_size", f"{ow}x{H}", "-framerate",
                                           str(a.fps), "-fflags", "nobuffer", "-flags", "low_delay", "-"],
                                          stdin=subprocess.PIPE))
        else:
            sinks.append(ffmpeg_sink(o))
    stats = {"played": 0, "repeated": 0, "latency": []}
    async with websockets.connect(a.url, max_size=2 ** 25, ping_interval=20) as ws:
        start = {"type": "start", "prompt": a.prompt, "mode": a.mode, "strength": a.strength, "person": a.person}
        if a.image:
            start["image"] = base64.b64encode(open(a.image, "rb").read()).decode()
        await ws.send(json.dumps(start))
        state["version"] = 1
        state["sent_t"][1] = time.time()
        t_start = time.time()
        loop = asyncio.get_running_loop()

        async def command(line):
            line = line.strip()
            if not line:
                return
            if line in ("/q", "/quit"):
                stop.set()
                return
            msg = None
            if line.startswith("/cut "):
                msg = {"type": "prompt", "text": line[5:], "cut": True}
            elif line.startswith("/strength "):
                msg = {"type": "set", "strength": float(line.split()[1])}
            elif line.startswith("/mode "):
                state["mode"] = line.split()[1]
                msg = {"type": "set", "mode": state["mode"]}
            elif line.startswith("/person "):
                state["person"] = line.split()[1] == "on"
                msg = {"type": "set", "person": state["person"]}
            elif line.startswith("/image "):
                msg = {"type": "image", "data": base64.b64encode(open(os.path.expanduser(line[7:]), "rb").read()).decode()}
            elif line.startswith("/reset"):
                msg = {"type": "reset"}
            elif not line.startswith("/"):
                msg = {"type": "prompt", "text": line, "cut": False}
            if msg is None:
                print(f"[client] unknown command {line!r}", flush=True)
                return
            if msg["type"] in ("prompt", "image", "reset"):
                state["version"] += 1
                state["sent_t"][state["version"]] = time.time()
            await ws.send(json.dumps(msg))
            print(f"[client] -> {line[:100]}", flush=True)

        async def stdin_reader():
            if not sys.stdin or not sys.stdin.isatty():
                return
            lq = asyncio.Queue()
            threading.Thread(target=lambda: [loop.call_soon_threadsafe(lq.put_nowait, l) for l in sys.stdin],
                             daemon=True).start()
            print("[client] type a prompt + Enter; /cut TEXT, /strength X, /mode t2v|v2v, /person on|off, "
                  "/image PATH, /reset, /q", flush=True)
            while not stop.is_set():
                try:
                    await command(await asyncio.wait_for(lq.get(), 0.5))
                except asyncio.TimeoutError:
                    pass

        async def scripted():
            for item in [x for x in a.script.split(";") if x.strip()]:
                t, cmd = item.split(":", 1)
                await asyncio.sleep(max(0, t_start + float(t) - time.time()))
                if stop.is_set():
                    return
                await command(cmd)

        async def sender():
            period = 1.0 / a.fps
            nxt = time.time()
            while not stop.is_set():
                if inp is not None and inp.frame is not None and state["mode"] == "v2v":
                    f = inp.frame
                    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    await ws.send(b"F" + jpg.tobytes())
                    if state["person"] and seg is not None:
                        ok, mj = cv2.imencode(".jpg", seg(f), [cv2.IMWRITE_JPEG_QUALITY, 80])
                        await ws.send(b"M" + mj.tobytes())
                nxt += period
                await asyncio.sleep(max(0, nxt - time.time()))

        async def receiver():
            async for m in ws:
                if isinstance(m, bytes):
                    seq, ver, chunk = struct.unpack("<III", m[:12])
                    img = cv2.imdecode(np.frombuffer(m[12:], np.uint8), cv2.IMREAD_COLOR)
                    q.append((img, ver))
                else:
                    j = json.loads(m)
                    if j.get("type") == "stats" and j["chunk"] % 5 == 0:
                        print(f"[client] server chunk {j['chunk']}: gen {j['gen_ms']} ms ({j['gen_fps']} fps), "
                              f"ahead {j['ahead_frames']} frames, buffer {len(q)}", flush=True)

        async def player():
            period = 1.0 / a.fps
            while len(q) < a.buffer and not stop.is_set():
                await asyncio.sleep(0.005)
            nxt = time.time()
            last, shown_ver = None, 0
            while not stop.is_set():
                if q:
                    last, ver = q.popleft()
                    stats["played"] += 1
                    if ver > shown_ver:
                        if ver in state["sent_t"]:
                            lat = time.time() - state["sent_t"][ver]
                            if ver > 1:
                                stats["latency"].append(lat)
                                print(f"[client] change v{ver} visible after {lat:.2f} s", flush=True)
                        shown_ver = ver
                else:
                    stats["repeated"] += 1
                if last is not None:
                    fr = last
                    if a.side_by_side and inp is not None and inp.frame is not None:
                        fr = np.hstack([inp.frame, last])
                    for s in sinks:
                        try:
                            s.stdin.write(fr.tobytes())
                        except BrokenPipeError:
                            stop.set()
                if a.seconds and time.time() - t_start > a.seconds:
                    stop.set()
                nxt += period
                await asyncio.sleep(max(0, nxt - time.time()))
            await ws.close()

        await asyncio.gather(receiver(), player(), sender(), stdin_reader(), scripted(), return_exceptions=True)
    for s in sinks:
        try:
            s.stdin.close()
            s.wait(timeout=10)
        except Exception:
            s.kill()
    if tunnel:
        tunnel.terminate()
    lat = stats["latency"]
    print(f"[client] RESULT played {stats['played']} repeated {stats['repeated']}" +
          (f", change latency median {sorted(lat)[len(lat) // 2]:.2f} s (n={len(lat)})" if lat else ""), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
