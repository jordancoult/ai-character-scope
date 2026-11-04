"""Live local test of the face pass: a recorded JoyAI output (looped) is animated by your webcam face, at 25 fps.

  cloud/facefix/.venv/bin/python cloud/lpfast/live_test.py \
      [--source cloud/pipelines/joyai/experiments/2026-11-02-joyai-5090/arms.mp4] \
      [--camera /dev/video0 | --driver VIDEO (e.g. the recording's own input in_arms.mp4, paired by motion DTW)] \
      [--mode rel] [--gains lip=1.3,eye=1,brow=-1] [--seconds 0] [--save out.mp4] [--layout 3|1]

Window (ffplay): webcam (9:16 crop, as the bridge sends it) | JoyAI raw | animated. While running, type in this
terminal + Enter: `/gain lip 1.5`, `/gain all 0.8`, `/gain` (show), `q` (quit). Simulates the end of the live
pipeline; the source frames are a recording, so their head motion won't match yours (only the face is driven).
"""

import argparse
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEFAULT_SRC = HERE.parents[1] / "assets" / "g-star-screenshot-no-hat.png"  # the G-Star test image (2026-11-04)


class Camera:
    """Newest webcam frame, cropped to the source aspect and resized (like live_bridge --portrait)."""

    def __init__(self, dev, w, h):
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.w, self.h, self.frame, self.n, self.stop = w, h, None, 0, False
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open camera {dev}")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            f = cv2.flip(f, 1)  # mirror, like a selfie view
            H, W = f.shape[:2]
            cw = min(W, int(H * self.w / self.h))
            ch = min(H, int(W * self.h / self.w))
            x0, y0 = (W - cw) // 2, (H - ch) // 2
            self.frame = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (self.w, self.h), interpolation=cv2.INTER_AREA)
            self.n += 1

    def close(self):
        self.stop = True
        time.sleep(0.05)
        self.cap.release()


def label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(DEFAULT_SRC), help="video (looped) or image (e.g. assets/char_front.png)")
    p.add_argument("--crop", default=None, help="crop the source, fractions x0,y0,x1,y1 (e.g. 0,0,1,0.5 = upper half)")
    p.add_argument("--pose", default=None, choices=["on", "off"], help="transfer head pose too (default: on for an "
                   "image source, off for a video, whose head already moves)")
    p.add_argument("--camera", default="/dev/video0")
    p.add_argument("--driver", default=None, help="driver video instead of the webcam: 'auto' = the source's "
                   "in_<name>.mp4; paired with the source frame by frame (source_seq JSON if present, else DTW)")
    p.add_argument("--preset", default=None, help="saved preset from cloud/lpfast/presets.json (e.g. gstar, girl): "
                   "sets mode, gains and pose; --mode/--gains/--pose given explicitly override it")
    p.add_argument("--mode", default=None, choices=["rel", "abs"])
    p.add_argument("--gains", default="")
    p.add_argument("--limits", default="", help="soft limits per region, e.g. lip=0.21,eye=0.1 (max; knee = 2/3) or "
                   "lip=0.14:0.21 (knee:max), lip=off; defaults calibrated on G-Star (lp_pass.DEFAULT_LIMITS)")
    p.add_argument("--no-visibility", action="store_true", help="don't run the occlusion / turned-away gates")
    p.add_argument("--smooth", type=float, default=0.0, help="driver expression smoothing 0..0.9 (higher = calmer, "
                   "laggier)")
    p.add_argument("--neutral-ema", type=float, default=0.01, help="rel: how fast your 'neutral face' baseline adapts "
                   "per frame (0.01 = ~4 s; 0 = keep the first frame)")
    p.add_argument("--base-ema", type=float, default=0.08, help="rel: how fast the character's resting expression "
                   "follows the source (still image: irrelevant)")
    p.add_argument("--fps", type=float, default=25)
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--save", default=None, help="also write the window content to this mp4")
    p.add_argument("--layout", type=int, default=3, choices=[1, 3], help="3 = webcam | raw | animated, 1 = animated")
    a = p.parse_args()
    from lp_pass import LPPass
    is_image = Path(a.source).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    if is_image:
        src = [cv2.imread(a.source, cv2.IMREAD_COLOR)]
    else:
        cap = cv2.VideoCapture(a.source)
        src = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            src.append(f)
    if a.crop:
        x0, y0, x1, y1 = (float(v) for v in a.crop.split(","))
        h0, w0 = src[0].shape[:2]
        src = [f[int(y0 * h0):int(y1 * h0), int(x0 * w0):int(x1 * w0)] for f in src]
    src = [f[:f.shape[0] // 2 * 2, :f.shape[1] // 2 * 2] for f in src]  # even size for the video encoders
    H, W = src[0].shape[:2]
    from lp_pass import load_presets, save_preset
    pre = load_presets().get(a.preset, {}) if a.preset else {}
    if a.preset and not pre:
        raise SystemExit(f"no preset {a.preset!r}; have {sorted(load_presets())}")
    a.mode = a.mode or pre.get("mode", "rel")
    gains = dict(pre.get("gains", {})) | {k: float(v) for k, v in (x.split("=") for x in a.gains.split(",") if x)}
    pose = (a.pose == "on") if a.pose else bool(pre.get("pose", is_image))
    def parse_limits(text):
        out = {}
        for part in text.replace(" ", "").split(","):
            if not part:
                continue
            k, v = part.split("=")
            out[k] = None if v in ("off", "0", "none") else (tuple(float(x) for x in v.split(":")) if ":" in v else float(v))
        return out
    limits = {k: (tuple(v) if v else None) for k, v in pre.get("limits", {}).items()} | parse_limits(a.limits)
    lp = LPPass(mode=a.mode, gains=gains, pose=pose, drv_smooth=a.smooth, neutral_ema=a.neutral_ema,
                base_ema=a.base_ema, limits=limits, visibility=not a.no_visibility)
    if pre.get("gates"):
        lp.set_gate(**{k: v for k, v in pre["gates"].items() if not (k in ("hands", "visibility") and a.no_visibility)})
    print(f"[live] source {Path(a.source).name} {W}x{H}{' (image)' if is_image else ''}, head pose transfer "
          f"{'on' if pose else 'off'}", flush=True)
    drv = lp.driver()
    cam, dfr, pair = None, None, None
    if a.driver:
        from run_offline import align_dtw, read
        dpath = Path(a.source).with_name("in_" + Path(a.source).name) if a.driver == "auto" else Path(a.driver)
        dfr = read(dpath)
        sj = Path(a.source).with_suffix(".json")
        import json
        seq = json.load(open(sj)).get("source_seq") if sj.exists() else None
        pair = [s - 1 for s in seq[:len(src)]] if seq else align_dtw(dfr, src).tolist()
        print(f"[live] driver {dpath.name}: {len(dfr)} frames, pairing {'source_seq' if seq else 'DTW'}", flush=True)
        dfr = [cv2.resize(f, (W, H)) if f.shape[:2] != (H, W) else f for f in dfr]
    else:
        cam = Camera(a.camera, W, H)
        t0 = time.time()
        while cam.frame is None and time.time() - t0 < 5:
            time.sleep(0.05)
        if cam.frame is None:
            raise SystemExit("no frames from the camera")
    get_driver = (lambda k: dfr[pair[k % len(src)]]) if dfr is not None else (lambda k: cam.frame)
    ow = W * a.layout
    play = subprocess.Popen(["ffplay", "-v", "error", "-window_title", "lpfast live test", "-f", "rawvideo",
                             "-pixel_format", "bgr24", "-video_size", f"{ow}x{H}", "-framerate", str(a.fps),
                             "-fflags", "nobuffer", "-flags", "low_delay", "-"], stdin=subprocess.PIPE)
    rec = None
    if a.save:
        rec = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                f"{ow}x{H}", "-r", str(a.fps), "-i", "-", "-vf", "scale='min(1920,iw)':-2",
                                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-profile:v", "high",
                                "-level", "4.1", "-movflags", "+faststart", a.save], stdin=subprocess.PIPE)
    quit_ = threading.Event()

    def commands():
        print(f"[live] gains {lp.gains}. Type '/gain <lip|eye|brow|brow2|rest|all> <v>', '/gain', '/reset' "
              "(re-take your neutral face), '/limit [region max|off]', '/gate [yaw|pitch a b | hands on|off | hold n]', "
              "'/preset [name]', '/save <name>' or 'q' + Enter",
              flush=True)
        for line in sys.stdin:
            line = line.strip()
            if line in ("q", "/q", "quit"):
                quit_.set()
                return
            if line.startswith("/preset"):
                parts = line.split()
                try:
                    if len(parts) == 2:
                        lp.apply_preset(parts[1])
                    print(f"[live] presets {sorted(load_presets())}; now mode {lp.mode}, pose "
                          f"{'on' if lp.pose else 'off'}, gains {lp.gains}", flush=True)
                except ValueError as e:
                    print(f"[live] {e}", flush=True)
                continue
            if line.startswith("/limit"):
                parts = line.split()
                try:
                    if len(parts) == 3:
                        v = parts[2]
                        lp.set_limit(**{parts[1]: None if v in ("off", "0") else (
                            tuple(float(x) for x in v.split(":")) if ":" in v else float(v))})
                    elif len(parts) != 1:
                        raise ValueError("usage: /limit <region> <max | knee:max | off>")
                    print(f"[live] limits {lp.limits}", flush=True)
                except ValueError as e:
                    print(f"[live] {e}", flush=True)
                continue
            if line.startswith("/gate"):
                parts = line.split()
                try:
                    if len(parts) >= 3:
                        k = {"yaw": "drv_yaw", "pitch": "drv_pitch"}.get(parts[1], parts[1])
                        v = parts[2:]
                        val = (v[0] in ("on", "1", "true")) if k in ("visibility", "hands") else (int(v[0]) if k in ("cover_hold", "hold") else (float(v[0]), float(v[1])))
                        lp.set_gate(**{("cover_hold" if k == "hold" else k): val})
                    elif len(parts) != 1:
                        raise ValueError("usage: /gate yaw|pitch|src_yaw|src_pitch <start> <off> | cover|score <full> <off> | "
                                         "visibility on|off | hold <frames>")
                    print(f"[live] gates {lp.gates()}", flush=True)
                except (ValueError, IndexError) as e:
                    print(f"[live] {e}", flush=True)
                continue
            if line.startswith("/save"):
                parts = line.split()
                if len(parts) != 2:
                    print("[live] usage: /save <name>", flush=True)
                    continue
                save_preset(parts[1], lp.mode, lp.gains, lp.pose, note=f"saved from live_test {time.strftime('%F %T')}",
                            limits=lp.limits, gates=lp.gates())
                print(f"[live] saved preset {parts[1]!r}: mode {lp.mode}, pose {'on' if lp.pose else 'off'}, "
                      f"gains {lp.gains} -> {Path(__file__).resolve().parent / 'presets.json'}", flush=True)
                continue
            if line in ("/reset", "/r"):  # take your current face / head position as the new neutral
                lp.neutral_in = lp.drv0 = None
                if drv is not None:
                    drv.reset()
                print("[live] neutral face + head position reset to the next frame", flush=True)
                continue
            if line.startswith("/gain"):
                parts = line.split()
                try:
                    if len(parts) == 3:
                        lp.set_gain(**{parts[1]: float(parts[2])})
                    print(f"[live] gains {lp.gains}", flush=True)
                except ValueError as e:
                    print(f"[live] {e}", flush=True)
    threading.Thread(target=commands, daemon=True).start()
    from concurrent.futures import ThreadPoolExecutor
    import queue
    pool = ThreadPoolExecutor(1)                  # driver analysis of the next frame overlaps the current render
    outq = queue.Queue(maxsize=4)                 # display / recording writes off the render loop

    def writer():
        while True:
            fr = outq.get()
            if fr is None:
                return
            try:
                play.stdin.write(fr.tobytes())
                if rec:
                    rec.stdin.write(fr.tobytes())
            except BrokenPipeError:
                quit_.set()
                return
    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    period, k, t_next, t_start = 1.0 / a.fps, 0, time.perf_counter(), time.time()
    ms, late = deque(maxlen=100), 0
    c = get_driver(0)
    fut = pool.submit(drv.analyze, c)
    try:
        while not quit_.is_set() and (not a.seconds or time.time() - t_start < a.seconds):
            o = src[k % len(src)]
            d = fut.result()
            c_now = c
            c = get_driver(k + 1)
            fut = pool.submit(drv.analyze, c)
            t = time.perf_counter()
            r, info = lp.render(o, d)
            ms.append((time.perf_counter() - t) * 1000)
            frame = np.hstack([label(c_now.copy(), "driver"), label(o.copy(), "source"),
                               label(r.copy(), f"{lp.mode} w={info['w']:.2f}" + (f" yaw {info.get('drv_yaw')} mouth {info.get('mouth_vis')} conf {info.get('face_score')}" if 'drv_yaw' in info else ""))]) if a.layout == 3 else r
            try:
                outq.put_nowait(frame)
            except queue.Full:
                pass                              # display behind: drop this frame rather than stall
            k += 1
            if k % 125 == 0:
                print(f"[live] {k} frames, render {np.median(ms):.0f} ms median / {max(ms):.0f} max, "
                      f"late ticks {late}", flush=True)
            t_next += period
            dt = t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                late += 1
                t_next = time.perf_counter()
        t_loop = time.time() - t_start
    finally:
        t_loop = locals().get("t_loop", time.time() - t_start)
        outq.put(None)
        wt.join(timeout=3)
        if cam:
            cam.close()
        for proc in (play, rec):
            if proc:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print(f"[live] done: {k} frames in {t_loop:.1f} s = {k / t_loop:.1f} fps; late ticks {late}", flush=True)


if __name__ == "__main__":
    main()
