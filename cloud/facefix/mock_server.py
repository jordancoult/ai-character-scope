"""Offline stand-in for the JoyAI server, to test live_bridge.py (+ --facefix) without a GPU pod.

Replays a recorded JoyAI output that is frame-aligned to its input (joyai_client --stats JSON with source_seq):
when the bridge sends input seq s, the recorded output frames made from input s are queued and sent back in
8-frame chunks every ~0.3 s after ~0.46 s (the 5090's rhythm), as H.264 with output_frame metadata. Feed the bridge
the same input clip as sent (in_*.mp4) so seq numbers line up.

  .venv/bin/python mock_server.py --output OUT.mp4 --stats OUT.json --port 18090
  python live_bridge.py --url ws://127.0.0.1:18090/ws --input IN.mp4 ...
"""
import argparse
import asyncio
import json
import time

import av
import cv2
import websockets

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--stats", required=True)
p.add_argument("--port", type=int, default=18090)
p.add_argument("--delay", type=float, default=0.46, help="seconds from input frame to its output chunk")
p.add_argument("--chunk", type=int, default=8)
a = p.parse_args()

cap, frames = cv2.VideoCapture(a.output), []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
seq_of = json.load(open(a.stats))["source_seq"][:len(frames)]
by_seq = {}
for i, s in enumerate(seq_of):
    if s:
        by_seq.setdefault(s, []).append(i)
H, W = frames[0].shape[:2]
print(f"[mock] {len(frames)} recorded frames {W}x{H}, inputs {min(by_seq)}..{max(by_seq)}", flush=True)


async def handle(ws):
    start = json.loads(await ws.recv())
    assert start.get("type") == "start"
    await ws.send(json.dumps({"type": "started", "width": W, "height": H}))
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = W, H, "yuv420p"
    enc.options = {"preset": "veryfast", "tune": "zerolatency", "crf": "20", "g": "16",
                   "x264-params": "repeat-headers=1"}
    enc.open()
    pending, meta, stop = [], {}, asyncio.Event()

    async def rx():
        try:
            async for m in ws:
                if isinstance(m, str):
                    j = json.loads(m)
                    if j.get("type") == "frame_meta":
                        meta[j["seq"]] = j["t_capture_ms"]
                        for i in by_seq.get(j["seq"], []):
                            pending.append((time.time() + a.delay, i, j["seq"], j["t_capture_ms"]))
                    elif j.get("type") == "stop":
                        break
        finally:
            stop.set()

    async def tx():
        while not stop.is_set():
            await asyncio.sleep(0.3)
            due = [x for x in pending[:a.chunk] if x[0] <= time.time()]
            if len(due) < a.chunk and not (pending and time.time() - pending[0][0] > 0.5):
                continue
            del pending[:len(due)]
            for _, i, s, tcap in due:
                vf = av.VideoFrame.from_ndarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), format="rgb24")
                data = b"".join(bytes(pk) for pk in enc.encode(vf.reformat(format="yuv420p")))
                await ws.send(json.dumps({"type": "output_frame", "source_seq": s, "t_capture_ms": tcap}))
                await ws.send(data)

    await asyncio.gather(rx(), tx(), return_exceptions=True)


async def main():
    async with websockets.serve(handle, "127.0.0.1", a.port, max_size=64 * 2**20):
        await asyncio.Future()

asyncio.run(main())
