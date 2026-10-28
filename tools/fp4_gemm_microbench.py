"""Does this GPU/torch support NVFP4 / MXFP4 torch._scaled_mm, and is it faster than fp8?

Shapes = Wan-1.3B FFN in a fused 2-frame call at 576x320 (M=1440).
"""
import time

import torch

dev = "cuda"
print(torch.cuda.get_device_name(0), torch.__version__, "fp4 dtype:", hasattr(torch, "float4_e2m1fn_x2"))


def bench(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


# e2m1 code -> value table, for quantizing
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=dev)


def quant_nvfp4(x, block=16):
    """x [M, K] -> packed fp4 [M, K/2] (uint8 view), scales [M, K/block] in e4m3."""
    m, k = x.shape
    xb = x.float().view(m, k // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 6.0).to(torch.float8_e4m3fn)
    xs = xb / scale.float()
    mag = xs.abs().unsqueeze(-1)
    code = (mag - E2M1).abs().argmin(-1)  # nearest magnitude code 0..7
    code = code | ((xs < 0).to(torch.long) << 3)
    code = code.view(m, k).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, scale.view(m, k // block)


def to_blocked(s):
    """cuBLAS block-scaling layout: tiles of 128 rows x 4 cols, swizzled (torchao's to_blocked)."""
    rows, cols = s.shape
    n_rb, n_cb = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(n_rb * 128, n_cb * 4, dtype=s.dtype, device=s.device)
    padded[:rows, :cols] = s
    blocks = padded.view(n_rb, 128, n_cb, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


for M, K, N in ((1440, 1536, 8960), (1440, 8960, 1536), (1440, 1536, 1536)):
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    ref = x.float() @ w.float().t()
    t_bf = bench(lambda: x @ w.t())
    # fp8 rowwise reference
    xs8 = x.abs().amax(-1, keepdim=True).float() / 448
    ws8 = w.abs().amax(-1, keepdim=True).float() / 448
    x8 = (x.float() / xs8).to(torch.float8_e4m3fn)
    w8 = (w.float() / ws8).to(torch.float8_e4m3fn)
    t_8 = bench(lambda: torch._scaled_mm(x8, w8.t(), scale_a=xs8, scale_b=ws8.t(), out_dtype=torch.bfloat16))
    e8 = ((torch._scaled_mm(x8, w8.t(), scale_a=xs8, scale_b=ws8.t(), out_dtype=torch.bfloat16).float() - ref).norm() / ref.norm()).item()
    line = f"M={M} K={K:5d} N={N:5d}  bf16 {t_bf:.3f}ms  fp8 {t_8:.3f}ms (err {e8:.3f})"
    try:
        xq, xsc = quant_nvfp4(x)
        wq, wsc = quant_nvfp4(w)
        a = xq.view(torch.float4_e2m1fn_x2)
        b = wq.view(torch.float4_e2m1fn_x2)
        sa, sb = to_blocked(xsc), to_blocked(wsc)
        out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        e4 = ((out.float() - ref).norm() / ref.norm()).item()
        t_4 = bench(lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        line += f"  nvfp4 {t_4:.3f}ms ({t_8 / t_4:.2f}x vs fp8, err {e4:.3f})"
    except Exception as ex:
        line += f"  nvfp4 FAILED: {type(ex).__name__}: {str(ex)[:160]}"
    print(line)
