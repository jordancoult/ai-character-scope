"""Compare upscalers on a generated clip: GPU time per frame + a zoomed side-by-side image.

Usage: python tools/upscale_eval.py bench/cd320.mp4 out.jpg [t_seconds]
"""

import sys
import time

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from upscalers import Upscaler  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
t_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
with av.open(src) as c:
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
fps = 30
frame = frames[min(int(t_sec * fps), len(frames) - 1)]
x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
batch = x.repeat(4, 1, 1, 1)  # a 4-frame chunk, as in the stream

tiles = []
for method in ("bicubic", "bicubic-sharp", "animevideov3"):
    up = Upscaler(method)
    for _ in range(5):
        up(batch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        up(batch)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 / 4 * 1000
    y = (up(x)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # 1280x720
    print(f"{method:14s} {ms:6.2f} ms/frame -> {y.shape[1]}x{y.shape[0]}")
    # zoom into the face/upper body (center-top 360x360 crop of the 720x1280 frame)
    crop = Image.fromarray(y).crop((0, 0, 720, 900))
    ImageDraw.Draw(crop).text((8, 8), f"{method} {ms:.1f}ms", fill=(255, 255, 255))
    tiles.append(crop)

sheet = Image.new("RGB", (720 * len(tiles), 900))
for i, tile in enumerate(tiles):
    sheet.paste(tile, (720 * i, 0))
sheet.save(out, quality=92)
print("saved", out)
