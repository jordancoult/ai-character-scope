"""Live bridge: camera (or a clip as a fake camera) -> JoyAI server on the GPU pod -> smooth playout to a
virtual camera (v4l2loopback, e.g. /dev/video10 for OBS) or an mp4 file.

The server returns edited frames in bursts of one chunk (8 frames), so playout runs from a small jitter
buffer at a steady rate: it starts once `--buffer` frames are queued, repeats the last frame on underrun and
drops the oldest frames when the queue grows past `--max-buffer` (keeps latency bounded).

  python live_bridge.py --url ws://127.0.0.1:18080/ws --input /dev/video0 --output /dev/video10 \
      --ref assets/char_front.png --prompt "..." --portrait
  python live_bridge.py --input assets/webcam_5941016.mp4 --output test.mp4 --seconds 30 ...   # no camera
  python live_bridge.py ... --output ffplay                    # watch in an ffplay window

Optional face pass on the local GPU (off by default; run with cloud/facefix/.venv/bin/python, see
cloud/lpfast/README.md): --facefix rel (cartoon refs like G-Star) | abs (human refs). LivePortrait (TensorRT, ~34 ms
per frame on the RTX 5080 Laptop) re-renders the generated face so its mouth and eyes follow the real face; pose and
identity stay JoyAI's. Each sent frame is analysed when it is sent (cached by seq); each received frame is paired
with its input via the server's source_seq and rendered in a worker thread; playout uses the refined frame if it is
ready by its playout time, else holds the previous refined frame, else the raw JoyAI frame.
"""

import argparse
import asyncio
import base64
import collections
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
import websockets

p = argparse.ArgumentParser()
p.add_argument("--url", default="ws://127.0.0.1:18080/ws")
p.add_argument("--input", default="/dev/video0", help="v4l2 device, a video file, an http(s) URL to a video, or 'web' "
               "(prompts you to paste a URL at start) "
               "(downloaded once to cloud/client/.cache; played in real time, looped)")
p.add_argument("--input-size", default="1280x720", help="capture size for v4l2 devices")
p.add_argument("--output", default="/dev/video10", help="v4l2loopback device, .mp4 file or 'ffplay' (a window)")
p.add_argument("--out-size", default=None, help="output WxH after resize (default: generation size)")
p.add_argument("--ref", default=None, help="reference image (file or http(s) URL); omit for a prompt-only edit")
p.add_argument("--keep-download", action="store_true", help="keep URL downloads in cloud/client/.cache for reuse "
               "(default: delete them when the run ends)")
p.add_argument("--prompt", required=True)
p.add_argument("--portrait", action="store_true", help="480x840 (9:16) instead of 840x480")
p.add_argument("--fps", type=float, default=26, help="camera send rate (JoyAI on an RTX 5090 outputs ~25-26 fps; "
               "frames above that are dropped by the server anyway)")
p.add_argument("--play-fps", type=float, default=25, help="playout rate (set near the server's output fps)")
p.add_argument("--buffer", type=int, default=6, help="frames queued before playout starts; also the queue target")
p.add_argument("--max-buffer", type=int, default=20, help="drop oldest frames above this queue length")
p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Ctrl+C)")
p.add_argument("--crf", type=int, default=20, help="uplink H.264 quality")
p.add_argument("--extra", default="{}", help="extra JSON for the start message")
p.add_argument("--facefix-preset", default=None, help="saved face-pass preset from cloud/lpfast/presets.json (e.g. gstar, "
               "girl); implies --facefix <its mode>; --facefix-gains override it. Live: '/preset <name>'")
p.add_argument("--facefix", default="off", choices=["off", "rel", "abs"],
               help="LivePortrait face pass on the local GPU (cloud/lpfast, TensorRT): the generated face's mouth and "
                    "eyes follow the real face. rel = keep the character's own mouth shape and add the speaker's "
                    "motion (cartoon refs, e.g. G-Star); abs = copy the speaker's mouth shape (human refs)")
p.add_argument("--facefix-gains", default="", help="per-region motion gains, e.g. 'lip=1.3,eye=1,brow=-1' (regions "
               "lip, eye, brow, brow2, rest; 1 = the speaker's motion as is, 0.5 = half, 0 = off, negative = "
               "mirrored). Defaults: rel lip=1.3 eye=1 brow=-1 (G-Star); abs lip/eye/brow/brow2/rest=1. "
               "Live: type '/gain lip 1.5', '/gain all 0.8' or '/gain' (show) + Enter")
p.add_argument("--facefix-wait", type=float, default=15, help="ms the player waits for a late refined frame")
p.add_argument("--facefix-hold", type=int, default=2, help="on a miss, repeat the last refined frame if it is at "
               "most this many frames old (instead of popping to the raw frame)")
p.add_argument("--facefix-extra", default="", help="more facefix.py flags, e.g. '--yaw-hi 45 --no-hands'")
p.add_argument("--review", default=None, help="also write input | JoyAI | played (facefix) as an mp4 (<=1920 wide)")
p.add_argument("--save-input", default=None, help="write the frames as sent (generation size) to this mp4 and "
               "<name>.json with source_seq per played frame (for client/mouth_sync.py)")
args = p.parse_args()
if args.facefix_preset:  # a preset implies its mode
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lpfast"))
    import json as _json
    _pre = _json.loads((Path(__file__).resolve().parents[1] / "lpfast" / "presets.json").read_text())
    if args.facefix_preset not in _pre:
        raise SystemExit(f"[bridge] no face preset {args.facefix_preset!r}; have {sorted(_pre)}")
    if args.facefix == "off":
        args.facefix = _pre[args.facefix_preset].get("mode", "rel")

W, H = (480, 840) if args.portrait else (840, 480)


FETCHED = []  # downloads made by this run (deleted at exit unless --keep-download)


def fetch(spec):
    """http(s) URL -> local file in cloud/client/.cache (keyed by URL; reused if kept); anything else unchanged."""
    if not spec or not spec.startswith(("http://", "https://")):
        return spec
    import hashlib
    import urllib.parse
    import urllib.request
    from pathlib import Path
    cache = Path(__file__).resolve().parent / ".cache"
    cache.mkdir(exist_ok=True)
    ext = Path(urllib.parse.urlparse(spec).path).suffix or ".bin"
    dst = cache / (hashlib.sha1(spec.encode()).hexdigest()[:16] + ext)
    if not dst.exists() or dst.stat().st_size == 0:
        print(f"[bridge] downloading {spec[:90]}...", flush=True)
        req = urllib.request.Request(spec, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(str(dst) + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        Path(str(dst) + ".part").rename(dst)
        print(f"[bridge] saved {dst.stat().st_size / 1e6:.1f} MB -> {dst}", flush=True)
        FETCHED.append(dst)
    return str(dst)


def cleanup_downloads():
    if args.keep_download:
        return
    for f in FETCHED:
        try:
            f.unlink()
        except OSError:
            pass


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
        if (oh > ow) != (h > w):  # keep the output in the session's orientation (portrait vs landscape)
            print(f"[bridge] --out-size {ow}x{oh} doesn't match the {w}x{h} session; using {oh}x{ow}", flush=True)
            ow, oh = oh, ow
        vf = f"scale={ow}:{oh}:flags=lanczos" if (ow, oh) != (w, h) else "null"
        if path == "ffplay":
            self.p = subprocess.Popen(["ffplay", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
                                       "-video_size", f"{w}x{h}", "-framerate", str(args.play_fps), "-fflags",
                                       "nobuffer", "-flags", "low_delay", "-window_title", "ai-livestream", "-autoexit",
                                       "-vf", vf, "-i", "-"], stdin=subprocess.PIPE)
            return
        if path.startswith("/dev/video"):
            out = ["-vf", vf + ",format=yuv420p", "-f", "v4l2", path]
        else:
            out = ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                   "-profile:v", "high", "-level", "4.1", "-movflags", "+faststart", path]
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w}x{h}", "-r", str(args.play_fps), "-i", "-", *out],
                                  stdin=subprocess.PIPE)

    def write(self, frame):
        try:
            self.p.stdin.write(frame.tobytes())
        except BrokenPipeError:            # ffplay window closed
            pass

    def close(self):
        try:
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        self.p.wait()


class FileWriter:
    """Background mp4 writer (review / saved input) so encoding never blocks playout."""

    def __init__(self, path, w, h, fps, max_w=1920):
        vf = f"scale={max_w}:-2:flags=area" if w > max_w else "null"
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-vf", vf, "-c:v", "libx264",
                                   "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-profile:v", "high",
                                   "-level", "4.1", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        self.q = collections.deque()
        self.ev = threading.Event()
        self.done = False
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def write(self, frame):
        self.q.append(frame)
        self.ev.set()

    def run(self):
        while True:
            self.ev.wait(0.1)
            self.ev.clear()
            while self.q:
                self.p.stdin.write(np.ascontiguousarray(self.q.popleft()).tobytes())
            if self.done and not self.q:
                break

    def close(self):
        self.done = True
        self.ev.set()
        self.t.join()
        self.p.stdin.close()
        self.p.wait()


def parse_gains(text):
    """'lip=1.3,eye=1' -> {'lip': 1.3, 'eye': 1.0}"""
    out = {}
    for part in text.replace(" ", "").split(","):
        if part:
            k, v = part.split("=")
            out[k] = float(v)
    return out


class FaceFixer:
    """cloud/lpfast LivePortrait pass (TensorRT 11, laptop GPU) behind two single-thread pools: driver analysis
    when a frame is sent (off the critical path) and the render when the JoyAI frame arrives (in order).
    Never blocks the event loop. The playout jitter buffer hides the ~34 ms render."""

    def __init__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lpfast"))
        import gc
        from lp_pass import LPPass  # lazy: needs cloud/facefix/.venv (torch, TensorRT 11) + engines (lpfast/README)
        self.pool_in, self.pool_gpu = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
        self.inputs = collections.OrderedDict()          # seq -> future of the driver analysis
        self.stats = collections.Counter()
        self.last_info = None
        self.ms = collections.deque(maxlen=250)
        t = time.time()
        from lp_pass import load_presets
        pre = load_presets().get(args.facefix_preset, {}) if args.facefix_preset else {}
        gains = dict(pre.get("gains", {})) | parse_gains(args.facefix_gains)
        self.lp = self.pool_gpu.submit(lambda: LPPass(mode=args.facefix, gains=gains)).result()
        self.drv = self.pool_in.submit(self.lp.driver).result()
        gc.freeze()                                       # long-lived objects out of the GC: fewer pauses
        print(f"[bridge] facefix {args.facefix} (LivePortrait TensorRT, gains {self.lp.gains}) ready "
              f"({time.time() - t:.0f} s warm-up)", flush=True)

    def add_input(self, seq, frame):
        self.inputs[seq] = self.pool_in.submit(self.drv.analyze, frame)
        while len(self.inputs) > 300:
            self.inputs.popitem(last=False)

    def submit(self, seq, out):
        fd = self.inputs.get(seq)
        if fd is None:
            self.stats["no_input"] += 1
            return None
        job = {"cancel": False}

        def render():
            if job["cancel"]:
                self.stats["skipped_late"] += 1
                return None
            t0 = time.perf_counter()
            res, info = self.lp.render(out, fd.result())
            self.ms.append(1000 * (time.perf_counter() - t0))
            job["info"] = {k: info.get(k) for k in ("ok", "w")}
            return res
        job["fut"] = self.pool_gpu.submit(render)
        return job

    def take(self, job):
        """Refined frame if ready now, else None (and the job is cancelled)."""
        if job is None:
            return None
        f = job["fut"]
        self.last_info = None
        if f.done() and f.exception() is None and f.result() is not None:
            self.stats["hit"] += 1
            self.last_info = job.get("info")
            return f.result()
        if f.done() and f.exception() is not None:
            self.stats["error"] += 1
            print("[bridge] facefix error:", repr(f.exception()), flush=True)
        job["cancel"] = True
        self.stats["miss"] += 1
        return None


async def main():
    if args.input == "web":  # ask for the URL interactively (paste + Enter)
        args.input = input("[bridge] paste the video URL and press Enter: ").strip().strip('"').strip("'")
        if not args.input.startswith(("http://", "https://")):
            raise SystemExit(f"[bridge] not an http(s) URL: {args.input!r}")
    args.input, args.ref = fetch(args.input), fetch(args.ref)
    ref = None
    if args.ref:
        with open(args.ref, "rb") as f:
            ref = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    start = {"type": "start", "prompt": args.prompt, "width": W, "height": H, "fps": args.fps,
             "num_inference_steps": 2, "input_codec": "h264", "output_codec": "h264", "detection_mode": "off",
             "use_pe": False, "kv_reset_frames": 0,
             # freeze-on-static skips "static" frames; it also drops subtle mouth motion (lip-sync test 2026-11-02)
             "freeze_kv_on_static": False, **json.loads(args.extra)}
    if ref:
        start["ref_image"] = ref
    src = Source(args.input)
    ff = FaceFixer() if args.facefix != "off" else None
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
        def new_enc():
            e = av.CodecContext.create("libx264", "w")
            e.width, e.height, e.pix_fmt = w, h, "yuv420p"
            e.options = {"preset": "veryfast", "tune": "zerolatency", "crf": str(args.crf), "g": "16",
                         "x264-params": "scenecut=0:repeat-headers=1:colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv"}
            e.open()
            return e
        # mutable session state: a prompt change restarts the server session (fresh encoder/decoder, ack counter)
        sess = {"enc": new_enc(), "dec": av.CodecContext.create("h264", "r"), "new_enc": False}
        sink = Sink(args.output, w, h)
        review = FileWriter(args.review, 3 * w, h, args.play_fps) if args.review else None
        saved_in = FileWriter(args.save_input, w, h, args.fps) if args.save_input else None
        played_seq, played_ff = [], []
        sent_cache = collections.OrderedDict()           # seq -> frame as sent (for --review)
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
                if ff is not None:
                    ff.add_input(seq + 1, frame)
                if review is not None:
                    sent_cache[seq + 1] = frame
                    while len(sent_cache) > 300:
                        sent_cache.popitem(last=False)
                if saved_in is not None:
                    saved_in.write(frame)
                vf = av.VideoFrame.from_ndarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), format="rgb24")
                vf = vf.reformat(format="yuv420p", dst_colorspace="itu709")
                if sess["new_enc"]:  # session restarted: begin a new GOP so the server's fresh decoder can start
                    sess["enc"], sess["new_enc"] = new_enc(), False
                data = b"".join(bytes(pk) for pk in sess["enc"].encode(vf))
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
                    elif j.get("type") == "started":  # after a prompt change / reset
                        received = 0
                        sess["dec"] = av.CodecContext.create("h264", "r")
                        print("[bridge] session restarted", flush=True)
                    elif j.get("type") in ("error", "flow_drop", "session_reset", "no_person"):
                        stats[j["type"]] += 1
                    continue
                try:
                    frames = sess["dec"].decode(av.Packet(m))
                except av.error.InvalidDataError:
                    continue
                received += 1
                await ws.send(json.dumps({"type": "ack", "recv": received}))
                for fr in frames:
                    img = fr.to_ndarray(format="bgr24")
                    sseq = (meta or {}).get("source_seq")
                    job = ff.submit(sseq, img) if ff is not None and sseq else None
                    queue.append((img, (meta or {}).get("t_capture_ms"), job, sseq))
                while len(queue) > args.max_buffer:
                    old = queue.popleft()
                    if old[2] is not None:
                        old[2]["cancel"] = True
                    stats["dropped"] += 1

        async def commands():
            """Type a new prompt + Enter to change it live; /r = reset memory (same prompt); /q = quit."""
            if not sys.stdin or not sys.stdin.isatty():
                return
            loop = asyncio.get_running_loop()
            q = asyncio.Queue()
            import threading

            def reader():
                for line in sys.stdin:
                    loop.call_soon_threadsafe(q.put_nowait, line.rstrip("\n"))
            threading.Thread(target=reader, daemon=True).start()
            print("[bridge] type a new prompt + Enter to change it; /r = reset memory; /q = quit"
                  + ("; /gain <lip|eye|brow|all> <value> = face motion gain" if ff is not None else ""), flush=True)
            while not stop.is_set():
                try:
                    line = (await asyncio.wait_for(q.get(), timeout=0.5)).strip()
                except asyncio.TimeoutError:
                    continue
                if not line:
                    continue
                if line in ("/q", "/quit"):
                    stop.set()
                    break
                if line.startswith("/preset") and ff is not None:   # live face-pass preset switch
                    parts = line.split()
                    try:
                        if len(parts) == 2:
                            ff.lp.apply_preset(parts[1])
                        print(f"[bridge] face preset mode {ff.lp.mode}, gains {ff.lp.gains}", flush=True)
                    except ValueError as e:
                        print(f"[bridge] {e}", flush=True)
                    continue
                if line.startswith("/gain"):                    # live face-pass gain: /gain [region|all] [value]
                    if ff is None:
                        print("[bridge] /gain needs --facefix rel|abs", flush=True)
                        continue
                    parts = line.split()
                    try:
                        if len(parts) == 3:
                            ff.lp.set_gain(**{parts[1]: float(parts[2])})
                        elif len(parts) != 1:
                            raise ValueError("usage: /gain <lip|eye|brow|brow2|rest|all> <value>")
                        print(f"[bridge] face gains {ff.lp.gains}", flush=True)
                    except ValueError as e:
                        print(f"[bridge] {e}", flush=True)
                    continue
                if line not in ("/r", "/reset"):
                    start["prompt"] = line
                    print(f"[bridge] new prompt: {line[:100]}", flush=True)
                else:
                    print("[bridge] resetting memory", flush=True)
                sess["new_enc"] = True
                await ws.send(json.dumps(start))

        async def pinger():
            while not stop.is_set():
                await ws.send(json.dumps({"type": "ping", "t": time.time() * 1000, "recv": received}))
                await asyncio.sleep(1.0)

        async def player():
            last, last_raw, last_seq = None, None, None
            last_ref, last_ref_n = None, -99
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
                    last_raw, tcap, job, last_seq = queue.popleft()
                    last = last_raw
                    if ff is not None:
                        if job is not None and not job["fut"].done():     # give the GPU a few ms
                            t_dead = time.time() + args.facefix_wait / 1000
                            while not job["fut"].done() and time.time() < t_dead:
                                await asyncio.sleep(0.002)
                        last = ff.take(job)
                        if last is not None:
                            last_ref, last_ref_n = last, n
                        elif last_ref is not None and n - last_ref_n <= args.facefix_hold:
                            last = last_ref                    # hold the previous refined frame: a raw frame
                            ff.stats["held"] += 1              # between refined ones pops the mouth
                        else:
                            last = last_raw
                    if tcap:
                        lat.append(time.time() * 1000 - tcap)
                    stats["played"] += 1
                else:
                    stats["repeated"] += 1
                if last is not None:
                    sink.write(last)
                    played_seq.append(last_seq)
                    played_ff.append(ff.last_info if ff is not None else None)   # None = raw frame (miss/off)
                    if review is not None:
                        inp = sent_cache.get(last_seq)
                        review.write(np.hstack([inp if inp is not None else np.zeros_like(last), last_raw, last]))
                n += 1
                # adaptive rate: keep the queue near the target (one chunk) so latency doesn't creep up
                err = len(queue) - args.buffer
                speed = 1.0 + max(-0.25, min(0.25, 0.05 * err))
                next_t = max(next_t + 1.0 / (args.play_fps * speed), time.time() - 0.1)
                await asyncio.sleep(max(0.0, next_t - time.time()))
                if n % int(args.play_fps * 10) == 0:
                    l = sorted(lat[-250:])
                    fx = ""
                    if ff is not None:
                        tot = ff.stats["hit"] + ff.stats["miss"]
                        ms = sorted(ff.ms)
                        fx = (f" facefix hit {ff.stats['hit']}/{tot}" +
                              (f" render {ms[len(ms) // 2]:.0f} ms" if ms else ""))
                    print(f"[bridge] played {stats['played']} repeated {stats['repeated']} dropped {stats['dropped']} "
                          f"queue {len(queue)} glass-to-glass median {l[len(l) // 2]:.0f} ms{fx}" if l else "",
                          flush=True)

        try:
            await asyncio.gather(sender(), receiver(), pinger(), player(), commands())
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            stop.set()
            try:
                await ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            sink.close()
            for wr in (review, saved_in):
                if wr is not None:
                    wr.close()
            if args.save_input:
                with open(str(Path(args.save_input).with_suffix(".json")), "w") as f:
                    json.dump({"source_seq": played_seq, "facefix": played_ff}, f)
    l = sorted(lat[50:] or lat)
    res = {**stats, "glass_to_glass_median_ms": round(l[len(l) // 2]) if l else None,
           "glass_to_glass_p95_ms": round(l[int(len(l) * 0.95)]) if l else None}
    if ff is not None:
        tot = ff.stats["hit"] + ff.stats["miss"]
        ms = sorted(ff.ms)
        res["facefix"] = {**ff.stats, "hit_rate": round(ff.stats["hit"] / tot, 3) if tot else None,
                          "render_ms_median": round(ms[len(ms) // 2], 1) if ms else None}
    print("[bridge] RESULT", json.dumps(res))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_downloads()
