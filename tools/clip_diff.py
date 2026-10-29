"""Per-frame PSNR between two clips of the same run setup (how far an optimization drifts from a reference).

Usage: python tools/clip_diff.py ref.mp4 other.mp4 [...]
"""
import sys

import av
import numpy as np


def frames(p):
    with av.open(p) as c:
        return [f.to_ndarray(format="rgb24").astype(np.float32) for f in c.decode(video=0)]


ref = frames(sys.argv[1])
for p in sys.argv[2:]:
    x = frames(p)
    n = min(len(ref), len(x))
    psnr = [10 * np.log10(255**2 / max(np.mean((ref[i] - x[i]) ** 2), 1e-6)) for i in range(n)]
    q = n // 4
    print(f"{p[-28:]:28s} PSNR vs ref: first quarter {np.mean(psnr[:q]):5.1f} dB, last quarter {np.mean(psnr[-q:]):5.1f} dB, min {min(psnr):5.1f}")
