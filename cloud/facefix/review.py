"""Contact sheet for a facefix run: rows of the zoom video (input | JoyAI | refined, with the gate overlay), picked
evenly plus at the gate events (lowest weight, largest |yaw|, hand occlusion), labelled with frame number.

  .venv/bin/python review.py --stats run.json --video zoom_run.mp4 --out sheet.jpg [--rows 8]
"""
import argparse
import json

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--stats", required=True)
p.add_argument("--video", required=True)
p.add_argument("--out", required=True)
p.add_argument("--rows", type=int, default=8)
p.add_argument("--by-jaw", action="store_true", help="pick the frames with the widest open input mouth (spread out)")
p.add_argument("--by-yaw", action="store_true", help="pick the frames nearest |yaw| = 0, 10, ... 70 deg instead")
p.add_argument("--width", type=int, default=720, help="row width in the sheet")
a = p.parse_args()

fi = json.load(open(a.stats))["frames_info"]
n = len(fi)
w = np.array([f["weight"] for f in fi])
yaw = np.array([abs(f.get("yaw") or 0) for f in fi])
hand = np.array([1 - f["gates"].get("hand", 1) for f in fi])
pick = {int(i): "even" for i in np.linspace(10, n - 1, max(2, a.rows // 2)).astype(int)}
events = [("max |yaw|", int(np.argmax(yaw))), ("hand", int(np.argmax(hand)) if hand.max() > 0 else None),
          ("min weight", int(np.argmin(w[10:]) + 10))]
dw = np.abs(np.diff(w))
if dw.max() > 0.05:
    events.append(("fastest fade", int(np.argmax(dw)) + 1))
for name, i in events:
    if i is not None and len(pick) < a.rows:
        pick[i] = name
if a.by_jaw:
    jaw = np.nan_to_num(np.array([f.get("jaw_in") if f.get("jaw_in") is not None else np.nan for f in fi], float))
    pick = {}
    for i in np.argsort(-jaw):
        if all(abs(i - k) > 15 for k in pick):
            pick[int(i)] = f"jaw {jaw[i]:.2f}"
        if len(pick) >= a.rows:
            break
if a.by_yaw:
    pick = {}
    for b in range(0, 80, 10):
        i = int(np.argmin(np.abs(yaw - b) + (yaw == 0) * 99))
        if abs(yaw[i] - b) < 4:
            pick[i] = f"|yaw|~{b}"
cap = cv2.VideoCapture(a.video)
rows = []
for i in sorted(pick):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    if not ok:
        continue
    f = cv2.resize(f, (a.width, round(f.shape[0] * a.width / f.shape[1])), interpolation=cv2.INTER_AREA)
    info = fi[i]
    label = f"#{i} {pick[i]}  w={info['weight']:.2f} src={info['src']} yaw={info.get('yaw')} pitch={info.get('pitch')}"
    cv2.putText(f, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(f, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    rows.append(f)
cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("wrote", a.out, len(rows), "rows")
