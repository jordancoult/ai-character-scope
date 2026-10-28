"""Read the virtual camera like OBS/ffplay would and timestamp flash frames (monotonic clock).

Pair with `scope_bridge.py --latency-probe` (which prints recv=<monotonic>) to measure the
bridge -> v4l2loopback -> consumer segment.  Usage: python tools/v4l2_flash_reader.py /dev/video10 SECONDS
"""
import sys
import time

import av

dev, seconds = sys.argv[1], float(sys.argv[2])
container = av.open(dev, format="v4l2", options={"input_format": "yuv420p"}, timeout=10)
stream = container.streams.video[0]
base, prev, end = None, None, time.monotonic() + seconds
for frame in container.decode(stream):
    now = time.monotonic()
    if now > end:
        break
    luma = float(frame.to_ndarray(format="gray")[::8, ::8].mean())
    if base is not None and prev is not None and luma > base + 60 and prev <= base + 60:
        print(f"[reader] flash t={now:.4f}", flush=True)
    base = luma if base is None else 0.95 * base + 0.05 * min(luma, base + 20)
    prev = luma
