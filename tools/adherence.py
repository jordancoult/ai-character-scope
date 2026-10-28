"""Structure adherence: does the generated character follow the input's pose/framing?

Runs Video-Depth-Anything on the input clip and on each output clip, aligns frames by index,
and reports the mean Pearson correlation of the (downsampled) depth maps. ~1 = same silhouette
and layout as the input; ~0 = unrelated composition. Also reports a frame-to-frame flicker
score for the output (mean abs difference between consecutive frames, lower = steadier).

Usage: python tools/adherence.py input.mp4 out1.mp4 [out2.mp4 ...]
"""

import sys

import av
import numpy as np
import torch
import torch.nn.functional as F

from scope.core.pipelines.video_depth_anything.gpu_depth import GpuStreamingDepth


def frames(path, size):
    with av.open(path) as c:
        out = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    x = torch.from_numpy(np.stack(out)).cuda().permute(0, 3, 1, 2).float() / 255
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def depth_of(x):
    model = GpuStreamingDepth(frames_per_chunk=4)
    ds = []
    for i in range(0, x.shape[0] - 3, 4):
        chunk = (x[i : i + 4] * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0).to(torch.bfloat16)
        d = model(chunk)[0, 0].float()  # [T, H, W]
        ds.append(F.avg_pool2d(d.unsqueeze(1), 8).squeeze(1))
    return torch.cat(ds)


@torch.no_grad()
def main():
    inp, outs = sys.argv[1], sys.argv[2:]
    size = (576, 320)
    d_in = depth_of(frames(inp, size))
    for path in outs:
        x = frames(path, size)
        d_out = depth_of(x)
        n = min(len(d_in), len(d_out))
        a, b = d_in[:n].flatten(1), d_out[:n].flatten(1)
        a = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)
        b = (b - b.mean(1, keepdim=True)) / (b.std(1, keepdim=True) + 1e-6)
        corr = (a * b).mean(1)
        flicker = (x[1:] - x[:-1]).abs().mean().item()
        print(f"{path:32s} depth-corr {corr.mean():.3f} (min {corr.min():.2f})  flicker {flicker:.4f}  frames {n}")


main()
