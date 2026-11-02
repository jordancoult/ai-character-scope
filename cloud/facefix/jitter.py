"""Temporal jitter / periodicity of the face region in a live_bridge --review video (input | JoyAI | played).

For the raw JoyAI panel and the played (facefix) panel: mean abs difference between consecutive frames inside a
fixed face box and mouth box (from MediaPipe on the JoyAI panel; static input assumed), MediaPipe lip-landmark
jitter (std of the lip points over time / face size), lip gap series, and the dominant period of each series
(autocorrelation peak; 8 frames = JoyAI's chunk length). Also checks the input panel is static and how the
source_seq steps.

  .venv/bin/python jitter.py --review review_rel.mp4 --json in_rel.json [--plot plot.png]
"""
import argparse
import json

import cv2
import numpy as np

from facefix import Landmarker

p = argparse.ArgumentParser()
p.add_argument("--review", required=True)
p.add_argument("--json", default=None, help="--save-input JSON (source_seq, facefix weights)")
p.add_argument("--skip", type=int, default=50, help="skip the first frames (session start)")
p.add_argument("--plot", default=None)
a = p.parse_args()

cap = cv2.VideoCapture(a.review)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
frames = frames[a.skip:]
W = frames[0].shape[1] // 3
panels = {k: [f[:, i * W:(i + 1) * W] for f in frames] for i, k in enumerate(["input", "raw", "played"])}

# fixed boxes from the JoyAI panel
lm = Landmarker(1)
pts = [x.pts for x in (lm(f) for f in panels["raw"][:60]) if x is not None]
P = np.median(np.array(pts), 0)
lo, hi = P.min(0), P.max(0)
face_box = np.r_[lo, hi].astype(int)
mouth = P[[61, 291, 0, 17]]
mc, mw = mouth.mean(0), np.linalg.norm(mouth[1] - mouth[0])
mouth_box = np.r_[mc - np.array([0.9, 0.7]) * mw, mc + np.array([0.9, 0.8]) * mw].astype(int)
fsize = float((hi - lo).max())


def crop(f, b):
    return f[max(b[1], 0):b[3], max(b[0], 0):b[2]].astype(np.float32)


def diffs(seq, b):
    return np.array([np.abs(crop(x, b) - crop(y, b)).mean() for x, y in zip(seq, seq[1:])])


def period(x, lo=2, hi=40):
    """Dominant period: autocorrelation of the high-passed series (minus a 1 s moving average); the highest local
    maximum after the first zero crossing. Returns (lag, autocorr) or (None, 0)."""
    x = np.nan_to_num(np.asarray(x, float))
    if len(x) < 3 * hi:
        return None, 0.0
    x = x - np.convolve(x, np.ones(25) / 25, "same")
    if x.std() < 1e-9:
        return None, 0.0
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:] / (x.var() * np.arange(len(x), 0, -1))
    z = np.argmax(ac[1:] < 0) + 1                       # first zero crossing
    cand = [k for k in range(max(lo, z), hi) if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not cand:
        return None, 0.0
    k = max(cand, key=lambda k: ac[k])
    return k, round(float(ac[k]), 3)


lip_idx = [13, 14, 78, 308, 61, 291, 0, 17]
res = {"frames": len(frames), "face_size_px": round(fsize), "face_box": face_box.tolist(), "mouth_box": mouth_box.tolist()}
series = {}
for k in ("input", "raw", "played"):
    dm, df = diffs(panels[k], mouth_box), diffs(panels[k], face_box)
    lms, gap = [], []
    lm2 = Landmarker(1)
    for f in panels[k]:
        x = lm2(f)
        if x is None:
            lms.append(np.full((len(lip_idx), 2), np.nan))
            gap.append(np.nan)
        else:
            lms.append(x.pts[lip_idx])
            gap.append(abs(x.pts[14, 1] - x.pts[13, 1]) / fsize)
    L = np.array(lms)
    jit = float(np.nanmean(np.nanstd(L, 0)) / fsize)
    step = np.nanmean(np.linalg.norm(np.diff(L, axis=0), axis=2), 1) / fsize     # per-frame lip motion
    series[k] = {"mouth_diff": dm, "face_diff": df, "lip_step": step, "lip_gap": np.array(gap)}
    res[k] = {"mouth_diff_mean": round(float(dm.mean()), 2), "face_diff_mean": round(float(df.mean()), 2),
              "lip_jitter_std_pct": round(100 * jit, 2), "lip_step_mean_pct": round(100 * float(np.nanmean(step)), 3),
              "lip_gap_std_pct": round(100 * float(np.nanstd(gap)), 3),
              "period_mouth_diff": period(dm), "period_lip_gap": period(np.array(gap) - np.nanmean(gap)),
              "zero_diff_frames": int((dm < 0.05).sum())}
if a.json:
    j = json.load(open(a.json))
    s = np.array([x if x is not None else -1 for x in j["source_seq"]])[a.skip:]
    st = np.diff(s)
    res["source_seq_steps"] = {str(k): int(v) for k, v in zip(*np.unique(st, return_counts=True))}
    ff = j.get("facefix") or []
    if ff:
        w = np.array([(x or {}).get("weight", np.nan) if x else np.nan for x in ff], float)[a.skip:]
        res["facefix_weight"] = {"mean": round(float(np.nanmean(w)), 3), "std": round(float(np.nanstd(w)), 3),
                                 "miss_frames": int(np.isnan(w).sum()), "period": period(np.nan_to_num(w, nan=0))}
print(json.dumps(res, indent=1))

if a.plot:
    H, Wp = 600, 1200
    img = np.full((H, Wp, 3), 255, np.uint8)
    rows = [("raw mouth diff", series["raw"]["mouth_diff"], (0, 0, 200)),
            ("played mouth diff", series["played"]["mouth_diff"], (0, 150, 0)),
            ("raw lip gap", series["raw"]["lip_gap"], (0, 0, 200)),
            ("played lip gap", series["played"]["lip_gap"], (0, 150, 0))]
    n = min(300, len(series["raw"]["mouth_diff"]))
    for r, (name, y, col) in enumerate(rows):
        y = np.nan_to_num(np.asarray(y[:n], float))
        top, hgt = r * H // 4 + 20, H // 4 - 30
        sc = (y.max() - y.min()) or 1
        ptsl = np.array([[int(i * Wp / n), int(top + hgt - (v - y.min()) / sc * hgt)] for i, v in enumerate(y)])
        cv2.polylines(img, [ptsl], False, col, 1, cv2.LINE_AA)
        for x in range(0, n, 8):
            cv2.line(img, (int(x * Wp / n), top), (int(x * Wp / n), top + hgt), (225, 225, 225), 1)
        cv2.putText(img, name + " (grid = 8 frames)", (5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(a.plot, img)
