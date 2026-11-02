"""Mouth-sync metric: how well the output's mouth follows the input's mouth.

Pairs every output frame with the input frame it was made from (`source_seq` saved by joyai_client.py --stats),
measures mouth opening in both with MediaPipe FaceLandmarker (blendshape `jawOpen` + inner-lip gap / face height),
and reports the Pearson correlation of the two series (1 = perfect sync, ~0 = mouth ignores the input), the
output's mouth-motion range relative to the input's, and face detection rates (G-Star may not be detected).

  python mouth_sync.py --input in.mp4 --output out.mp4 --stats run.json [--plot sync.png]
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="frames as sent (joyai_client --out-input)")
p.add_argument("--output", required=True, help="edited frames (joyai_client --out)")
p.add_argument("--stats", required=True, help="joyai_client --stats JSON (holds source_seq)")
p.add_argument("--plot", default=None)
p.add_argument("--model", default=str(Path(__file__).parent / "models" / "face_landmarker.task"))
args = p.parse_args()

det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=args.model), output_face_blendshapes=True, num_faces=1,
    min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))


def read_all(path):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


def mouth(frame):
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        return np.nan, np.nan
    lm = r.face_landmarks[0]
    face_h = abs(lm[152].y - lm[10].y) or 1e-6           # chin - forehead
    gap = abs(lm[14].y - lm[13].y) / face_h               # inner lower lip - inner upper lip
    jaw = next((b.score for b in r.face_blendshapes[0] if b.category_name == "jawOpen"), np.nan)
    return jaw, gap


ins, outs = read_all(args.input), read_all(args.output)
seq = json.load(open(args.stats)).get("source_seq") or []
pairs = [(i, s - 1) for i, s in enumerate(seq[:len(outs)]) if s and 0 < s <= len(ins)]
if not pairs:  # fallback: assume 1:1 order
    pairs = [(i, i) for i in range(min(len(ins), len(outs)))]
res = np.array([(*mouth(ins[j]), *mouth(outs[i])) for i, j in pairs])  # in_jaw, in_gap, out_jaw, out_gap


def corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 and a[m].std() > 1e-6 and b[m].std() > 1e-6 else None


def best_lag(a, b, max_lag=16):
    # output frames may react late (chunked generation): correlation of out[t] with in[t - lag]
    best = (None, 0)
    for lag in range(0, max_lag + 1):
        x, y = a[:len(a) - lag] if lag else a, b[lag:]
        c = corr(x, y)
        if c is not None and (best[0] is None or c > best[0]):
            best = (round(c, 3), lag)
    return best


report = {
    "pairs": len(pairs),
    "face_detected_in": round(float(np.mean(~np.isnan(res[:, 0]))), 3),
    "face_detected_out": round(float(np.mean(~np.isnan(res[:, 2]))), 3),
    "jaw_corr": corr(res[:, 0], res[:, 2]),
    "lipgap_corr": corr(res[:, 1], res[:, 3]),
    "lipgap_best_lag": best_lag(res[:, 1], res[:, 3]),   # (corr, lag in output frames)
    "motion_range_ratio": round(float(np.nanstd(res[:, 3]) / (np.nanstd(res[:, 1]) or 1e-6)), 3),
}
print("MOUTH_SYNC", json.dumps(report))
if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.plot(res[:, 1], label="input lip gap")
    plt.plot(res[:, 3], label="output lip gap")
    plt.legend(); plt.title(f"lip-gap corr {report['lipgap_corr']}"); plt.tight_layout(); plt.savefig(args.plot)
