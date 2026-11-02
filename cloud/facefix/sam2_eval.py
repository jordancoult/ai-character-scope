"""Offline evaluation of SAM 2.1 (tiny) video tracking of G-Star's mouth: box prompt from MediaPipe lips on the
first confident frame, then propagation; reports ms/frame, mask centre/area stability and writes a debug video.

  .venv/bin/python sam2_eval.py --video OUT.mp4 --frames 300 --out dbg.mp4 [--image-size 512]
"""
import argparse, json, os, shutil, tempfile, time
import cv2, numpy as np, torch
from facefix import Landmarker, WEIGHTS

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
p.add_argument("--frames", type=int, default=300)
p.add_argument("--out", default=None)
p.add_argument("--image-size", type=int, default=1024)
p.add_argument("--crop", type=int, default=0, help="track on a fixed square crop of this size around the face")
a = p.parse_args()

cap = cv2.VideoCapture(a.video); frames = []
while len(frames) < a.frames:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
lm = Landmarker(1)
k0, f0 = next((i, x) for i, x in ((i, lm(f)) for i, f in enumerate(frames)) if x is not None)
P = f0.pts
ox = oy = 0
if a.crop:                                     # fixed crop around the first face
    c = P.mean(0).astype(int); h = a.crop // 2
    ox, oy = max(c[0] - h, 0), max(c[1] - h, 0)
    frames = [f[oy:oy + a.crop, ox:ox + a.crop] for f in frames]
L = P[[61, 291, 0, 17, 13, 14]] - (ox, oy)
lo, hi = L.min(0), L.max(0); d = (hi - lo) * 0.15
box = np.r_[lo - d, hi + d]
tmp = tempfile.mkdtemp()
for i, f in enumerate(frames[k0:]):
    cv2.imwrite(f"{tmp}/{i:05d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
from sam2.build_sam import build_sam2_video_predictor
over = [f"++model.image_size={a.image_size}"] if a.image_size != 1024 else []
pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", str(WEIGHTS / "sam2/sam2.1_hiera_tiny.pt"),
                                  device="cuda", hydra_overrides_extra=over)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    st = pred.init_state(video_path=tmp, offload_video_to_cpu=True)
    pred.add_new_points_or_box(st, frame_idx=0, obj_id=1, box=box)
    masks, ts = [], []
    t = time.perf_counter()
    for fi, ids, logits in pred.propagate_in_video(st):
        torch.cuda.synchronize(); now = time.perf_counter(); ts.append(now - t); t = now
        masks.append((logits[0, 0] > 0).cpu().numpy())
shutil.rmtree(tmp)
cs, ar = [], []
for m in masks:
    ys, xs = np.nonzero(m)
    cs.append([xs.mean(), ys.mean()] if len(xs) else [np.nan, np.nan]); ar.append(len(xs))
cs, ar = np.array(cs), np.array(ar, float)
fs = float(np.ptp(P, 0).max())
step = np.linalg.norm(np.diff(cs, axis=0), axis=1) / fs * 100
res = {"frames": len(masks), "ms_per_frame_median": round(1000 * float(np.median(ts[5:])), 1),
       "image_size": a.image_size, "crop": a.crop, "lost_frames": int((ar == 0).sum()),
       "center_step_std_pct": round(float(np.nanstd(step)), 2), "center_step_max_pct": round(float(np.nanmax(step)), 1),
       "area_rel_std": round(float(np.nanstd(ar) / np.nanmean(ar)), 3)}
print(json.dumps(res))
if a.out:
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter("/tmp/sam2dbg.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25, (w, h))
    for f, m in zip(frames[k0:], masks):
        g = f.copy()
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(g, cnt, -1, (0, 255, 0), 2)
        wr.write(g)
    wr.release()
    os.system(f"ffmpeg -loglevel error -y -i /tmp/sam2dbg.avi -c:v libx264 -preset veryfast -crf 21 -profile:v high "
              f"-level 4.1 -pix_fmt yuv420p -movflags +faststart {a.out}")
