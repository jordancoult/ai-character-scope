"""Time the DiT patch-embedding Conv3d (kernel=stride=(1,2,2)) on this GPU, cuDNN vs a matmul rewrite."""
import torch, time
torch.manual_seed(0)
conv = torch.nn.Conv3d(16, 1536, (1, 2, 2), stride=(1, 2, 2)).cuda().to(torch.bfloat16)

def as_linear(x):
    # (B,C,T,H,W) -> patches (B,T,H/2,W/2,C*2*2) @ W^T -> (B,1536,T,H/2,W/2)
    B, C, T, H, W = x.shape
    p = x.view(B, C, T, H // 2, 2, W // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(B, T, H // 2, W // 2, C * 4)
    y = torch.nn.functional.linear(p, conv.weight.view(1536, -1), conv.bias)
    return y.permute(0, 4, 1, 2, 3).contiguous()

def bench(fn, x, n=20):
    for _ in range(3): fn(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn(x)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

for T in (1, 2, 4):
    x = torch.randn(1, 16, T, 64, 64, device="cuda", dtype=torch.bfloat16)
    ref, alt = conv(x), as_linear(x)
    print(f"T={T}: cudnn conv3d {bench(conv, x):7.2f} ms | matmul {bench(as_linear, x):6.3f} ms | max diff {(ref - alt).abs().max().item():.4f}")
torch.backends.cudnn.benchmark = True
x = torch.randn(1, 16, 1, 64, 64, device="cuda", dtype=torch.bfloat16)
print(f"cudnn.benchmark=True T=1: {bench(conv, x):.2f} ms")
