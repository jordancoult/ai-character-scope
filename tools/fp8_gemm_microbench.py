"""Is fp8 (torch._scaled_mm) faster than bf16 for Wan-1.3B linear shapes on this GPU?

Times: bf16 F.linear vs fp8 rowwise-scaled _scaled_mm, both with and without the
activation-quantization cost (amax + scale + cast, fused via torch.compile).
"""
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


@torch.compile(dynamic=False)
def quant_rowwise(x):
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
    scale = amax / E4M3_MAX
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def bench(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


print(torch.cuda.get_device_name(0), torch.__version__)
for M in (720, 1440, 2880, 5760):
    for K, N in ((1536, 1536), (1536, 8960), (8960, 1536)):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        b = torch.zeros(N, device=dev, dtype=torch.bfloat16)
        wq, ws = quant_rowwise(w)  # per-output-channel weight scales [N,1]
        xq, xs = quant_rowwise(x)

        def fp8_mm():
            return torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t(), bias=b, out_dtype=torch.bfloat16)

        def fp8_full():
            q, s = quant_rowwise(x)
            return torch._scaled_mm(q, wq.t(), scale_a=s, scale_b=ws.t(), bias=b, out_dtype=torch.bfloat16)

        try:
            ref = F.linear(x, w, b)
            err = ((fp8_full() - ref).float().norm() / ref.float().norm()).item()
            t_bf, t_mm, t_full = bench(lambda: F.linear(x, w, b)), bench(fp8_mm), bench(fp8_full)
            print(f"M={M:5d} K={K:5d} N={N:5d}  bf16 {t_bf:6.3f}ms  fp8-gemm {t_mm:6.3f}ms ({t_bf / t_mm:4.2f}x)"
                  f"  fp8+quant {t_full:6.3f}ms ({t_bf / t_full:4.2f}x)  rel.err {err:.4f}")
        except Exception as e:
            print(f"M={M} K={K} N={N}: fp8 failed: {type(e).__name__}: {str(e)[:150]}")
