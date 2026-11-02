"""Micro-benchmark of the LivePortrait parts on the GPU (random crops): motion extractor x2, appearance F,
warping W, decoder G. Usage: .venv/bin/python bench_lp.py [--compile] [--channels-last]"""
import sys, time, torch, numpy as np
sys.argv += []
from facefix import Retargeter
torch.backends.cudnn.benchmark = True
rt = Retargeter("human", "lip", compile="--compile" in sys.argv)
w = rt.w
a = np.random.randint(0, 255, (256, 256, 3), np.uint8)
def tm(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return 1000 * (time.perf_counter() - t) / n
x = torch.rand(2, 3, 256, 256, device="cuda")
kp = w.get_kp_info(x); f = w.extract_feature_3d(x[1:]); xs = w.transform_keypoint({k: v[1:] for k, v in kp.items()})
print("M x2", round(tm(lambda: w.get_kp_info(x)), 1))
print("F", round(tm(lambda: w.extract_feature_3d(x[1:])), 1))
def wg():
    with torch.no_grad(), w.inference_ctx():
        return w.warping_module(f, kp_source=xs, kp_driving=xs)
print("W", round(tm(wg), 1))
o = wg()["out"]
def g():
    with torch.no_grad(), w.inference_ctx():
        return w.spade_generator(feature=o)
print("G", round(tm(g), 1))
print("full", round(tm(lambda: rt(a, a)), 1))
