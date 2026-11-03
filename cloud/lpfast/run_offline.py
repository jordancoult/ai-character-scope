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


def motion_feats(frames, g=8, sz=64):
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (sz, sz), interpolation=cv2.INTER_AREA).astype(np.float32)
             for f in frames]
    feats = [np.zeros(g * g, np.float32)]
    for x, y in zip(small, small[1:]):
        feats.append(np.abs(y - x).reshape(g, sz // g, g, sz // g).mean((1, 3)).ravel())
    f = np.array(feats)
    f -= f.mean(1, keepdims=True)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-6)


def align_dtw(in_frames, out_frames, max_step=60, max_start=90, jump_cost=0.01):
    """(from cloud/facefix) input index per output frame: monotonic DTW over grid motion-energy features."""
    cost = 1.0 - motion_feats(out_frames) @ motion_feats(in_frames).T
    n_o, n_i = cost.shape
    D = np.full((n_o, n_i), 1e9)
    B = np.zeros((n_o, n_i), np.int32)
    D[0, :max_start] = cost[0, :max_start]
    for j in range(1, n_o):
        best, arg = np.full(n_i, 1e9), np.zeros(n_i, np.int32)
        for st in range(1, max_step + 1):
            cand = np.full(n_i, 1e9)
            cand[st:] = D[j - 1, :-st] + jump_cost * max(0, st - 2)
            m = cand < best
            best[m], arg[m] = cand[m], st
        D[j], B[j] = best + cost[j], arg
    i = int(np.argmin(D[-1]))
    idx = [i]
    for j in range(n_o - 1, 0, -1):
        i -= B[j, i]
        idx.append(i)
    return np.array(idx[::-1])


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--stats", default=None, help="joyai_client --stats JSON with source_seq (exact pairing); "
                   "without it the pairing is estimated by DTW on motion")
    p.add_argument("--dst", required=True)
    p.add_argument("--mode", default="rel", choices=["rel", "abs"])
    p.add_argument("--gains", default="", help="per-region gains, e.g. lip=1,eye=1,brow=1,rest=0 (defaults: "
                   "lip/eye/brow 1, rest 0)")
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
    seq = (json.load(open(a.stats)).get("source_seq") if a.stats else None) or None
    if seq:
        seq = seq[:len(outs)]
    else:
        seq = [int(i) + 1 for i in align_dtw(ins, outs)]
        print("pairing: DTW")
    gains = {k: float(v) for k, v in (x.split("=") for x in a.gains.split(",") if x)}
    lp = LPPass(mode=a.mode, gains=gains, base_ema=a.base_ema, drv_smooth=a.drv_smooth)
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
        if times[-1] > 50 and k > 5:
            print(f"slow frame {k}: {times[-1]:.0f} ms, det_out {lp.src.detections}, det_in {drv.face.detections}")
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
