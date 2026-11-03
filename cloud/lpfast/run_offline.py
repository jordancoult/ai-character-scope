"""Replay a recorded JoyAI run through the LivePortrait pass (exact source_seq pairing) and write review media.

  cloud/facefix/.venv/bin/python cloud/lpfast/run_offline.py --out G_gstar.mp4 --inp in_G_gstar.mp4 \
      --stats G_gstar.json --dst cloud/lpfast/experiments/<name> [--mode rel] [--max 300]

Writes <dst>/lp.mp4 (result frames, same count/order as the JoyAI output: score with cloud/client/mouth_sync.py
--input <inp> --output lp.mp4 --stats <stats>), <dst>/review.mp4 (input | JoyAI | LP; <=1920 wide, h264 high@4.1,
faststart) and <dst>/timing.json.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def read(path, n=0):
    cap, fr = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok or (n and len(fr) >= n):
            return fr
        fr.append(f)


class Writer:
    def __init__(self, path, w, h, fps):
        vf = "scale='min(1920,iw)':-2"
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                   f"{w}x{h}", "-r", str(fps), "-i", "-", "-vf", vf, "-c:v", "libx264", "-crf", "18",
                                   "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
                                   "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--mode", default="rel", choices=["rel", "abs"])
    p.add_argument("--regions", default="lip,eyes")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--eye-gain", type=float, default=1.0)
    p.add_argument("--base-ema", type=float, default=0.08)
    p.add_argument("--drv-smooth", type=float, default=0.0)
    p.add_argument("--profile", action="store_true", help="sync per stage (stage_ms), slower")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--fps", type=float, default=25)
    a = p.parse_args()
    from lp_pass import LPPass
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    outs = read(a.out, a.max)
    ins = read(a.inp)
    seq = json.load(open(a.stats))["source_seq"][:len(outs)]
    lp = LPPass(mode=a.mode, regions=tuple(a.regions.split(",")), gain=a.gain, eye_gain=a.eye_gain,
                base_ema=a.base_ema, drv_smooth=a.drv_smooth)
    drv = lp.driver()
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)  # live: the driver frame is analysed when it is sent, in parallel
    H, W = outs[0].shape[:2]
    w_lp, w_rv = Writer(dst / "lp.mp4", W, H, a.fps), Writer(dst / "review.mp4", 3 * W, H, a.fps)
    times, oks = [], 0
    pick = lambda s: ins[min(len(ins) - 1, max(0, s - 1))]  # noqa: E731  output k was made from input seq[k]-1
    fut = pool.submit(drv.analyze, pick(seq[0]))
    t_start = time.perf_counter()
    for k, (o, s) in enumerate(zip(outs, seq)):
        i = pick(s)
        d = fut.result()
        if k + 1 < len(seq):
            fut = pool.submit(drv.analyze, pick(seq[k + 1]))
        t0 = time.perf_counter()
        r, info = lp.render(o, d, profile=a.profile)
        times.append((time.perf_counter() - t0) * 1000)
        oks += info["ok"]
        w_lp.write(r)
        w_rv.write(np.hstack([label(i, f"input {s}"), label(o, "JoyAI"), label(r, f"LP {a.mode} w={info['w']:.2f}")]))
    wall = time.perf_counter() - t_start
    w_lp.close()
    w_rv.close()
    t = np.array(times[10:]) if len(times) > 20 else np.array(times)
    res = {"frames": len(outs), "faces_ok": oks / len(outs), "ms_mean": float(t.mean()), "ms_p50": float(np.median(t)),
           "ms_p95": float(np.percentile(t, 95)), "ms_max": float(t.max()),
           "detections_out": lp.src.detections, "detections_in": drv.face.detections,
           "wall_fps_incl_io": len(outs) / wall, "stage_ms": {k: round(v, 2) for k, v in lp.timing.items()},
           "args": vars(a)}
    (dst / "timing.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "args"}))


if __name__ == "__main__":
    main()
