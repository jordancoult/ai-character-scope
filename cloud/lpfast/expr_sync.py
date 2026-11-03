"""Expression-transfer metric: how well mouth, eyes and brows of the output follow the driver (MediaPipe FaceMesh).

  cloud/facefix/.venv/bin/python cloud/lpfast/expr_sync.py --input IN.mp4 --output OUT.mp4 [--stats OUT.json]
      [--label NAME]
Pairing: output frame k <- input frame source_seq[k]-1 (stats JSON), else frame k. Per frame, normalised by face
height (forehead 10 - chin 152):
  lip   inner-lip gap (13-14)
  eye   mean eyelid opening (159-145, 386-374)
  brow  mean brow height above the upper eyelid (105-159, 334-386)
Prints Pearson correlation driver vs output and the motion range ratio (std out / std in) per channel.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).resolve().parents[1] / "client" / "models" / "face_landmarker.task"


def series(path, det, idx=None):
    cap, out, k = cv2.VideoCapture(str(path)), [], 0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    sel = idx if idx is not None else range(len(frames))
    for i in sel:
        f = frames[min(len(frames) - 1, max(0, i))]
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if not r.face_landmarks:
            out.append((np.nan,) * 3)
            continue
        p = np.array([(q.x, q.y) for q in r.face_landmarks[0]])
        h = abs(p[152, 1] - p[10, 1]) + 1e-6
        lip = abs(p[14, 1] - p[13, 1]) / h
        eye = (abs(p[145, 1] - p[159, 1]) + abs(p[374, 1] - p[386, 1])) / 2 / h
        brow = (abs(p[159, 1] - p[105, 1]) + abs(p[386, 1] - p[334, 1])) / 2 / h
        out.append((lip, eye, brow))
    return np.array(out)


def corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 10 or a[m].std() < 1e-6 or b[m].std() < 1e-6:
        return None
    return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stats", default=None)
    p.add_argument("--label", default="")
    a = p.parse_args()
    det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
    so = series(a.output, det)
    seq = json.load(open(a.stats)).get("source_seq") if a.stats else None
    idx = [s - 1 for s in seq[:len(so)]] if seq else list(range(len(so)))
    si = series(a.input, det, idx)
    res = {"label": a.label, "frames": len(so)}
    for c, name in enumerate(("lip", "eye", "brow")):
        res[name] = corr(si[:, c], so[:, c])
        ok = ~np.isnan(si[:, c]) & ~np.isnan(so[:, c])
        res[name + "_range"] = round(float(so[ok, c].std() / (si[ok, c].std() + 1e-9)), 2) if ok.sum() > 10 else None
    print("EXPR_SYNC", json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
