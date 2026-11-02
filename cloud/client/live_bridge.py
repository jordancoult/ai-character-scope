"""Live bridge: camera (or a clip as a fake camera) -> JoyAI server on the GPU pod -> smooth playout to a
virtual camera (v4l2loopback, e.g. /dev/video10 for OBS) or an mp4 file.

The server returns edited frames in bursts of one chunk (8 frames), so playout runs from a small jitter
buffer at a steady rate: it starts once `--buffer` frames are queued, repeats the last frame on underrun and
drops the oldest frames when the queue grows past `--max-buffer` (keeps latency bounded).

  python live_bridge.py --url ws://127.0.0.1:18080/ws --input /dev/video0 --output /dev/video10 \
      --ref assets/char_front.png --prompt "..." --portrait
  python live_bridge.py --input assets/webcam_5941016.mp4 --output test.mp4 --seconds 30 ...   # no camera
"""

import argparse
import asyncio
import base64
import collections
import json
import subprocess
import time

import av
import cv2
import numpy as np
import websockets

p = argparse.ArgumentParser()
p.add_argument("--url", default="ws://127.0.0.1:18080/ws")
p.add_argument("--input", default="/dev/video0", help="v4l2 device or a video file (played in real time, looped)")
p.add_argument("--input-size", default="1280x720", help="capture size for v4l2 devices")
p.add_argument("--output", default="/dev/video10", help="v4l2loopback device or .mp4 file")
p.add_argument("--out-size", default=None, help="output WxH after resize (default: generation size)")
p.add_argument("--ref", required=True)
p.add_argument("--prompt", required=True)
p.add_argument("--portrait", action="store_true", help="480x840 (9:16) instead of 840x480")
p.add_argument("--fps", type=float, default=30, help="camera send rate")
p.add_argument("--play-fps", type=float, default=25, help="playout rate (set near the server's output fps)")
p.add_argument("--buffer", type=int, default=6, help="frames queued before playout starts; also the queue target")
p.add_argument("--max-buffer", type=int, default=20, help="drop oldest frames above this queue length")
p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Ctrl+C)")
p.add_argument("--crf", type=int, default=20, help="uplink H.264 quality")
p.add_argument("--extra", default="{}", help="extra JSON for the start message")
args = p.parse_args()

W, H = (480, 840) if args.portrait else (840, 480)


def crop_resize(frame, w, h):
    fh, fw = frame.shape[:2]
    s = max(w / fw, h / fh)
    frame = cv2.resize(frame, (round(fw * s), round(fh * s)), interpolation=cv2.INTER_AREA)
    y, x = (frame.shape[0] - h) // 2, (frame.shape[1] - w) // 2
    return frame[y:y + h, x:x + w]


class Source:
    def __init__(self, spec):
        self.live = spec.startswith("/dev/video")
        self.cap = cv2.VideoCapture(spec, cv2.CAP_V4L2 if self.live else cv2.CAP_ANY)
        if self.live:
            iw, ih = map(int, args.input_size.split("x"))
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, iw)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ih)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.t0 = time.time()
        self.i = 0

    def read(self):
        if self.live:
            ok, f = self.cap.read()
            return f if ok else None
        want = int((time.time() - self.t0) * self.src_fps)  # real-time pacing for files
        f = None
        while self.i <= want:
            ok, f = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = self.cap.read()
            self.i += 1
        return f


class Sink:
    def __init__(self, path, w, h):
        ow, oh = map(int, args.out_size.split("x")) if args.out_size else (w, h)
        vf = f"scale={ow}:{oh}:flags=lanczos" if (ow, oh) != (w, h) else "null"
        if path.startswith("/dev/video"):
            out = ["-vf", vf + ",format=yuv420p", "-f", "v4l2", path]
        else:
            out = ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", path]
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w}x{h}", "-r", str(args.play_fps), "-i", "-", *out],
                                  stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(frame.tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


async def main():
    with open(args.ref, "rb") as f:
        ref = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    start = {"type": "start", "prompt": args.prompt, "ref_image": ref, "width": W, "height": H, "fps": args.fps,
             "num_inference_steps": 2, "input_codec": "h264", "output_codec": "h264", "detection_mode": "off",
             "use_pe": False, "kv_reset_frames": 0,
             # freeze-on-static skips "static" frames; it also drops subtle mouth motion (lip-sync test 2026-11-02)
             "freeze_kv_on_static": False, **json.loads(args.extra)}
    src = Source(args.input)
    queue = collections.deque()          # (frame, t_capture_ms)
    stats = collections.Counter()
    lat = []
    stop = asyncio.Event()
    t_end = time.time() + args.seconds if args.seconds else None

    async with websockets.connect(args.url, max_size=64 * 2**20, ping_interval=20) as ws:
        await ws.send(json.dumps(start))
        while True:
            m = json.loads(await ws.recv())
            if m.get("type") in ("started", "error"):
                break
        if m["type"] == "error":
            print("[bridge] server error:", m)
            return
        h, w = m["height"], m["width"]
        print(f"[bridge] session started {w}x{h}; {args.input} -> {args.output}")
        enc = av.CodecContext.create("libx264", "w")
        enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
        enc.options = {"preset": "veryfast", "tune": "zerolatency", "crf": str(args.crf), "g": "16",
                       "x264-params": "scenecut=0:repeat-headers=1:colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv"}
        enc.open()
        dec = av.CodecContext.create("h264", "r")
        sink = Sink(args.output, w, h)
        received = 0

        async def sender():
            seq, t0 = 0, time.time()
            while not stop.is_set():
                if t_end and time.time() > t_end:
                    stop.set()
                    break
                frame = await asyncio.to_thread(src.read) if src.live else src.read()
                if frame is None:
                    await asyncio.sleep(0.005)
                    continue
                frame = crop_resize(frame, w, h)
                vf = av.VideoFrame.from_ndarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), format="rgb24")
                vf = vf.reformat(format="yuv420p", dst_colorspace="itu709")
                data = b"".join(bytes(pk) for pk in enc.encode(vf))
                seq += 1
                await ws.send(json.dumps({"type": "frame_meta", "seq": seq, "t_capture_ms": time.time() * 1000}))
                await ws.send(data)
                await asyncio.sleep(max(0.0, t0 + seq / args.fps - time.time()))

        async def receiver():
            nonlocal received
            meta = None
            while not stop.is_set():
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if isinstance(m, str):
                    j = json.loads(m)
                    if j.get("type") == "output_frame":
                        meta = j
                    elif j.get("type") in ("error", "flow_drop", "session_reset", "no_person"):
                        stats[j["type"]] += 1
                    continue
                try:
                    frames = dec.decode(av.Packet(m))
                except av.error.InvalidDataError:
                    continue
                received += 1
                await ws.send(json.dumps({"type": "ack", "recv": received}))
                for fr in frames:
                    queue.append((fr.to_ndarray(format="bgr24"), (meta or {}).get("t_capture_ms")))
                while len(queue) > args.max_buffer:
                    queue.popleft()
                    stats["dropped"] += 1

        async def pinger():
            while not stop.is_set():
                await ws.send(json.dumps({"type": "ping", "t": time.time() * 1000, "recv": received}))
                await asyncio.sleep(1.0)

        async def player():
            last = None
            started = False
            next_t = None
            n = 0
            while not stop.is_set():
                if not started:
                    if len(queue) >= args.buffer:
                        started, next_t = True, time.time()
                    else:
                        await asyncio.sleep(0.005)
                        continue
                if queue:
                    last, tcap = queue.popleft()
                    if tcap:
                        lat.append(time.time() * 1000 - tcap)
                    stats["played"] += 1
                else:
                    stats["repeated"] += 1
                if last is not None:
                    sink.write(last)
                n += 1
                # adaptive rate: keep the queue near the target (one chunk) so latency doesn't creep up
                err = len(queue) - args.buffer
                speed = 1.0 + max(-0.25, min(0.25, 0.05 * err))
                next_t = max(next_t + 1.0 / (args.play_fps * speed), time.time() - 0.1)
                await asyncio.sleep(max(0.0, next_t - time.time()))
                if n % int(args.play_fps * 10) == 0:
                    l = sorted(lat[-250:])
                    print(f"[bridge] played {stats['played']} repeated {stats['repeated']} dropped {stats['dropped']} "
                          f"queue {len(queue)} glass-to-glass median {l[len(l) // 2]:.0f} ms" if l else "", flush=True)

        try:
            await asyncio.gather(sender(), receiver(), pinger(), player())
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            stop.set()
            try:
                await ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            sink.close()
    l = sorted(lat[50:] or lat)
    res = {**stats, "glass_to_glass_median_ms": round(l[len(l) // 2]) if l else None,
           "glass_to_glass_p95_ms": round(l[int(len(l) * 0.95)]) if l else None}
    print("[bridge] RESULT", json.dumps(res))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
