import sys, os, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "longlive2", ".sage2", "pkg"))
import sageattention as sa
print("sageattention from", sa.__file__)
import flash_attn
torch.manual_seed(0)
B, H, D = 1, 24, 128
Lq, Lk = 8 * 576, 32 * 576
q, k, v = (torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16) for L in (Lq, Lk, Lk))
ref = F.scaled_dot_product_attention(*(t.float().transpose(1, 2) for t in (q[:, :1024], k, v))).transpose(1, 2)
def bench(name, fn, reps=20):
    try:
        out = fn(); torch.cuda.synchronize()
        for _ in range(3): fn()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(reps): fn()
        torch.cuda.synchronize(); ms = (time.time() - t) / reps * 1000
        o = out[:, :1024].float()
        print(f"{name:34s} {ms:7.2f} ms   cos {F.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item():.6f}  max err {(o - ref).abs().max().item():.4f}")
    except Exception as e:
        print(f"{name:34s} FAILED: {type(e).__name__}: {str(e)[:120]}")
bench("flash_attn 2 (current)", lambda: flash_attn.flash_attn_func(q, k, v))
bench("sage2 auto (sageattn)", lambda: sa.sageattn(q, k, v, tensor_layout="NHD", is_causal=False))
for fn_name in ("sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton"):
    f = getattr(sa, fn_name, None)
    if f:
        bench(f"sage2 {fn_name}", lambda f=f: f(q, k, v, tensor_layout="NHD", is_causal=False))
f = getattr(sa, "sageattn_qk_int8_pv_fp8_cuda", None)
if f:
    bench("sage2 int8/fp8 pv_accum fp32+fp32", lambda: f(q, k, v, tensor_layout="NHD", is_causal=False, pv_accum_dtype="fp32+fp32"))
