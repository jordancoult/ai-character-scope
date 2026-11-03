"""Score eval_infer.py outputs on the laptop (CPU, MediaPipe): mouth sync of each variant vs the driver.

  cloud/.venv/bin/python cloud/pipelines/joyai/training/mouth_sync_eval.py EVAL_DIR [EVAL_DIR ...]

Per input folder (input.mp4 + <variant>.mp4): lip-gap series (inner-lip gap / face height, as in
cloud/client/mouth_sync.py) for the driver and every variant ->
  corr      Pearson correlation at lag 0 (1 = perfect sync, ~0 = mouth ignores the driver)
  best      best correlation over lags -4..+4 frames, and that lag (positive = output late)
  range     output lip-gap std / driver lip-gap std (mouth motion amplitude; Kling targets are ~0.5-1)
  face      fraction of frames where a face is detected on the output (G-Star detection ~0.8-1.0)
  drift     mean abs pixel difference to the base output (0-255; how far the LoRA moved the picture)
Writes EVAL_DIR/mouth_sync.json and prints a table.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data"))
from common import read_frames  # noqa: E402
from vision import Vision, best_lag, corr  # noqa: E402


def series(vis, frames, rois=None):
    gap, _, box, r = vis.faces(frames, fallback_rois=rois)
    return gap, r


def main():
    vis = Vision(use_pose=True)
    for ed in map(Path, sys.argv[1:]):
        res = {}
        for d in sorted(p for p in ed.iterdir() if (p / "input.mp4").exists()):
            fin = read_frames(d / "input.mp4")
            gi, rois = series(vis, fin)
            out = {"driver_face": float(np.mean(~np.isnan(gi)))}
            base = read_frames(d / "base.mp4") if (d / "base.mp4").exists() else None
            for v in sorted(d.glob("*.mp4")):
                if v.stem in ("input", "side_by_side"):
                    continue
                fo = read_frames(v)
                n = min(len(fin), len(fo))
                go, _ = series(vis, fo[:n], rois[:n])
                c = corr(gi[:n], go)
                b = best_lag(gi[:n], go, max_lag=4)
                rng = float(np.nanstd(go) / (np.nanstd(gi[:n]) or 1e-6)) if np.isfinite(go).any() else None
                drift = None
                if base is not None:
                    m = min(n, len(base))
                    drift = float(np.mean([np.abs(fo[i].astype(np.int16) - base[i].astype(np.int16)).mean()
                                           for i in range(0, m, 4)]))
                out[v.stem] = {"corr": None if c is None else round(c, 3), "best": b,
                               "range": None if rng is None else round(rng, 3),
                               "face": round(float(np.mean(~np.isnan(go))), 3),
                               "drift": None if drift is None else round(drift, 2)}
            res[d.name] = out
        (ed / "mouth_sync.json").write_text(json.dumps(res, indent=1))
        print(f"\n{ed}")
        print(f"{'input':28s} {'variant':14s} {'corr':>6s} {'best(lag)':>11s} {'range':>6s} {'face':>5s} {'drift':>6s}")
        for name, out in res.items():
            for k, v in out.items():
                if isinstance(v, dict):
                    print(f"{name:28s} {k:14s} {str(v['corr']):>6s} {str(v['best'][0]) + '(' + str(v['best'][1]) + ')':>11s}"
                          f" {str(v['range']):>6s} {v['face']:5.2f} {str(v['drift']):>6s}")
    vis.close()


if __name__ == "__main__":
    main()
