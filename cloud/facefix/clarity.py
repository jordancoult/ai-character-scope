"""Sharpness of the face in a 3-panel review video (input | JoyAI | refined, from facefix.py --sbs or
live_bridge.py --review): Laplacian variance and high-frequency share (var of gray - blur(1.5 px) / var of gray) for
the whole face, the eye+brow region and the mouth region (fixed boxes from MediaPipe on the JoyAI panel, median of
the first frames), JoyAI vs refined; plus the max abs difference JoyAI vs refined in the eye region (0 = untouched).

  .venv/bin/python clarity.py --video sbs.mp4 [--skip 50 --frames 350]
"""
import argparse
import json

import cv2
import numpy as np

from facefix import Landmarker

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
p.add_argument("--skip", type=int, default=50)
p.add_argument("--frames", type=int, default=350)
p.add_argument("--track", action="store_true", help="per-frame boxes (moving face) instead of fixed ones")
a = p.parse_args()

cap = cv2.VideoCapture(a.video)
frames, i = [], 0
while len(frames) < a.frames:
    ok, f = cap.read()
    if not ok:
        break
    if i >= a.skip:
        frames.append(f)
    i += 1
W = frames[0].shape[1] // 3
raw = [f[:, W:2 * W] for f in frames]
ref = [f[:, 2 * W:] for f in frames]
lm = Landmarker(1)
EYES = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 133, 159, 145, 362, 263, 386, 374]
MOUTH = [61, 291, 0, 17, 13, 14, 152, 2]


def boxes(pts):
    def bb(idx, m):
        q = pts[idx]
        lo, hi = q.min(0), q.max(0)
        d = (hi - lo) * m
        return np.r_[lo - d, hi + d].astype(int)
    lo, hi = pts.min(0), pts.max(0)
    return {"face": np.r_[lo, hi].astype(int), "eyes": bb(EYES, 0.1), "mouth": bb(MOUTH, 0.1)}


def crop(f, b):
    return cv2.cvtColor(f[max(b[1], 0):b[3], max(b[0], 0):b[2]], cv2.COLOR_BGR2GRAY).astype(np.float32)


def sharp(g):
    lap = cv2.Laplacian(g, cv2.CV_32F).var()
    hf = (g - cv2.GaussianBlur(g, (0, 0), 1.5)).var() / (g.var() + 1e-6)
    return lap, hf


fixed = None
if not a.track:
    pts = [x.pts for x in (lm(f) for f in raw[:60]) if x is not None]
    fixed = boxes(np.median(np.array(pts), 0))
acc = {k: {"raw": [], "refined": []} for k in ("face", "eyes", "mouth")}
eye_diff = []
for r, o in zip(raw, ref):
    if fixed is None:
        x = lm(r)
        if x is None:
            continue
        bx = boxes(x.pts)
    else:
        bx = fixed
    for k, b in bx.items():
        acc[k]["raw"].append(sharp(crop(r, b)))
        acc[k]["refined"].append(sharp(crop(o, b)))
    b = bx["eyes"]
    eye_diff.append(np.abs(r[b[1]:b[3], b[0]:b[2]].astype(int) - o[b[1]:b[3], b[0]:b[2]].astype(int)).max())
res = {}
for k, v in acc.items():
    R, O = np.array(v["raw"]), np.array(v["refined"])
    res[k] = {"lap_raw": round(float(R[:, 0].mean()), 1), "lap_refined": round(float(O[:, 0].mean()), 1),
              "lap_ratio": round(float(O[:, 0].mean() / R[:, 0].mean()), 3),
              "hf_raw": round(float(R[:, 1].mean()), 4), "hf_refined": round(float(O[:, 1].mean()), 4)}
res["eye_region_max_abs_diff"] = int(np.max(eye_diff)) if eye_diff else None
print(json.dumps(res))
