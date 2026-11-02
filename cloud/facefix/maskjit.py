"""Mask-geometry stability from a facefix.py --stats JSON: frame-to-frame jumps of the mouth-mask centre and size,
for the geometry actually used (input-anchored tracker) and for the raw per-frame measurement on the generated face
(what the unstabilised version used), in % of the face size; plus tracker counts.

  .venv/bin/python maskjit.py run.json
"""
import json
import sys

import numpy as np

r = json.load(open(sys.argv[1]))
fi = r["frames_info"]
out = {}
for key in ("meas_geom", "mask_geom"):
    c, hw, fs = [], [], []
    for f in fi:
        g = f.get(key)
        if g is None or f.get("fsize") is None:
            c.append([np.nan, np.nan]); hw.append(np.nan); fs.append(np.nan)
        else:
            c.append(g[:2]); hw.append(g[2]); fs.append(f["fsize"])
    c, hw, fs = np.array(c), np.array(hw), np.array(fs)
    step = np.linalg.norm(np.diff(c, axis=0), axis=1) / fs[1:] * 100
    if "idx" in r:                      # offline DTW pairing can jump the input by many frames: exclude those
        di = np.diff(np.array(r["idx"]))
        step[di > 3] = np.nan
    dsz = np.abs(np.diff(hw)) / hw[1:] * 100
    ok = ~np.isnan(step)
    out[key] = {"frames": int(ok.sum()), "center_step_mean_pct": round(float(np.nanmean(step)), 2),
                "center_step_std_pct": round(float(np.nanstd(step)), 2),
                "center_step_max_pct": round(float(np.nanmax(step)), 1) if ok.any() else None,
                "jumps_over_8pct": int(np.nansum(step > 8)),
                "size_step_mean_pct": round(float(np.nanmean(dsz)), 2), "size_step_max_pct":
                round(float(np.nanmax(dsz)), 1) if ok.any() else None}
out["tracker"] = {"mask": r.get("track_mask"), "rect": r.get("track_rect")}
w = np.array([f["weight"] for f in fi])
out["weight"] = {"mean": round(float(w.mean()), 3), "frames_below_0.5": int((w < 0.5).sum())}
print(json.dumps(out))
