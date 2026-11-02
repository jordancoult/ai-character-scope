"""Simulated-webcam client for the JoyAI-Video-Edit streaming server.

Plays a video file as if it were a live camera (real-time pacing, JPEG uplink), sends a reference
image + instruction, records what comes back, and reports output fps and glass-to-glass latency
(frame capture time on this machine -> edited frame received on this machine).

  python joyai_client.py --url ws://127.0.0.1:8080/ws --video clip.mp4 --ref char.png \
      --prompt "Replace the person ..." --seconds 40 --out out.mp4
"""

import argparse
import asyncio
import base64
import json
import statistics
import time

import cv2
import numpy as np
import websockets

p = argparse.ArgumentParser()
p.add_argument("--url", default="ws://127.0.0.1:8080/ws")
p.add_argument("--video", required=True)
p.add_argument("--ref", default=None, help="reference image (character)")
p.add_argument("--prompt", required=True)
p.add_argument("--fps", type=float, default=24)
p.add_argument("--width", type=int, default=840)
p.add_argument("--height", type=int, default=480)
p.add_argument("--seconds", type=float, default=40)
p.add_argument("--steps", type=int, default=2)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--quality", type=int, default=90, help="uplink JPEG quality")
p.add_argument("--out", default="out.mp4", help="edited output video")
p.add_argument("--out-input", default=None, help="also save the frames that were sent")
p.add_argument("--stats", default=None, help="write a JSON summary here")
p.add_argument("--extra", default="{}", help="extra JSON fields for the start message")
p.add_argument("--codec", default="mjpeg", choices=["mjpeg", "h264"],
               help="wire codec both ways; h264 is what the official web client uses (~10x less bandwidth)")
p.add_argument("--h264-crf", type=int, default=20, help="uplink H.264 quality")
args = p.parse_args()


def crop_resize(frame, w, h):
    fh, fw = frame.shape[:2]
    scale = max(w / fw, h / fh)
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x, y = (nw - w) // 2, (nh - h) // 2
    return frame[y:y + h, x:x + w]


async def main():
    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start = {"type": "start", "prompt": args.prompt, "width": args.width, "height": args.height,
             "fps": args.fps, "num_inference_steps": args.steps, "seed": args.seed,
             "input_codec": args.codec, "output_codec": args.codec, "detection_mode": "off",
             "use_pe": False, "output_quality": 92}
    if args.ref:
        with open(args.ref, "rb") as f:
            start["ref_image"] = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    start.update(json.loads(args.extra))

    recv_frames, lat_ms, recv_t, events = [], [], [], []
    src_seq = []  # input frame number (1-based, as sent) each output frame was made from
    sent_frames = []
    t_started = None
    async with websockets.connect(args.url, max_size=64 * 2**20, ping_interval=20) as ws:
        await ws.send(json.dumps(start))
        # wait for session start
        while True:
            msg = json.loads(await ws.recv())
            events.append(msg.get("type"))
            if msg.get("type") in ("started", "error"):
                print("[client]", msg)
                break
        if msg.get("type") == "error":
            return
        H, W = msg["height"], msg["width"]
        stop = asyncio.Event()
        enc = dec = None
        if args.codec == "h264":
            import av
            enc = av.CodecContext.create("libx264", "w")
            enc.width, enc.height, enc.pix_fmt = W, H, "yuv420p"
            enc.options = {"preset": "veryfast", "tune": "zerolatency", "crf": str(args.h264_crf), "g": "16",
                           "x264-params": "scenecut=0:repeat-headers=1:colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv"}
            enc.open()
            dec = av.CodecContext.create("h264", "r")

        def encode_frame(bgr):
            if enc is None:
                ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                return jpg.tobytes()
            import av
            vf = av.VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), format="rgb24")
            vf = vf.reformat(format="yuv420p", dst_colorspace="itu709")  # server assumes a BT.709 uplink (Chrome)
            return b"".join(bytes(pk) for pk in enc.encode(vf))

        def decode_frame(data):
            if dec is None:
                return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            import av
            try:
                fr = dec.decode(av.Packet(data))
            except av.error.InvalidDataError:
                return None
            return fr[-1].to_ndarray(format="bgr24") if fr else None

        async def sender():
            nonlocal t_started
            t_started = time.time()
            seq, src_i = 0, 0
            period = 1.0 / args.fps
            while not stop.is_set() and time.time() - t_started < args.seconds:
                # pick the source frame for "now" (loop the clip), like a live camera
                want = int((time.time() - t_started) * src_fps)
                frame = None
                while src_i <= want:
                    ok, frame = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    src_i += 1
                if frame is None:
                    await asyncio.sleep(0.001)
                    continue
                frame = crop_resize(frame, W, H)
                wire = encode_frame(frame)
                seq += 1
                now_ms = time.time() * 1000.0
                await ws.send(json.dumps({"type": "frame_meta", "seq": seq, "t_capture_ms": now_ms}))
                await ws.send(wire)
                if args.out_input:
                    sent_frames.append(frame)
                next_t = t_started + seq * period
                await asyncio.sleep(max(0.0, next_t - time.time()))
            await asyncio.sleep(3.0)  # drain
            stop.set()

        async def receiver():
            pending = None
            while not stop.is_set():
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if isinstance(m, str):
                    j = json.loads(m)
                    t = j.get("type")
                    if t == "output_frame":
                        pending = j
                    elif t not in ("chunk_start", "chunk_done", "pong"):
                        events.append(t)
                        if t in ("error", "no_person", "session_reset", "flow_drop"):
                            print("[client]", j)
                else:
                    now = time.time()
                    img = decode_frame(m)
                    if img is not None:
                        recv_frames.append(img)
                        recv_t.append(now)
                        src_seq.append((pending or {}).get("source_seq"))
                        if pending and pending.get("t_capture_ms"):
                            lat_ms.append(now * 1000.0 - float(pending["t_capture_ms"]))
                    # ack every frame (as the official web client does): the server's flow control
                    # treats un-acked frames as network backlog and drops output chunks
                    await ws.send(json.dumps({"type": "ack", "recv": len(recv_frames)}))

        async def pinger():
            while not stop.is_set():
                await ws.send(json.dumps({"type": "ping", "t": time.time() * 1000.0, "recv": len(recv_frames)}))
                await asyncio.sleep(1.0)

        await asyncio.gather(sender(), receiver(), pinger())
        try:
            await ws.send(json.dumps({"type": "stop"}))
        except Exception:
            pass

    # summary
    res = {"frames_out": len(recv_frames), "events": sorted(set(events))}
    if len(recv_t) > 48:
        warm = recv_t[24:]
        res["fps_out"] = round((len(warm) - 1) / (warm[-1] - warm[0]), 2)
        gaps = sorted(np.diff(warm) * 1000)
        res["gap_p95_ms"] = round(gaps[int(len(gaps) * 0.95)], 1)
        res["gap_max_ms"] = round(gaps[-1], 1)
    if lat_ms:
        l = sorted(lat_ms[24:] or lat_ms)
        res["latency_median_ms"] = round(statistics.median(l), 0)
        res["latency_p95_ms"] = round(l[int(len(l) * 0.95)], 0)
    print("[client] RESULT", json.dumps(res))
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump({**res, "args": vars(args), "source_seq": src_seq}, f, indent=1)
    if recv_frames:
        h, w = recv_frames[0].shape[:2]
        vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for f in recv_frames:
            vw.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
        vw.release()
    if args.out_input and sent_frames:
        h, w = sent_frames[0].shape[:2]
        vw = cv2.VideoWriter(args.out_input, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for f in sent_frames:
            vw.write(f)
        vw.release()


asyncio.run(main())
