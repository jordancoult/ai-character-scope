"""Score generated clips for realism, cleanliness, stability and pose adherence.

Usage: python tools/quality_metrics.py input.mp4 out1.mp4 [out2.mp4 ...]
(out*.mp4 frame-aligned with input.mp4, e.g. from bench_scope.py --max-seconds ... --out)

Metrics (per output clip):
  realism   CLIP no-reference score (CLIP-IQA style, antonym prompt pairs), 0..1, higher = more
            "real photo"/"clean sharp" vs "AI-generated with artifacts"/"blurry noisy distorted"
  spurious  output change between frames 250 ms apart, measured where the INPUT barely moved
            (invented motion / oscillation); lower is better
  flicker   high-frequency change between consecutive frames (shimmer, sparkles); lower is better
  adhere    depth-map correlation with the input (follows your pose/framing); higher is better
  drift     adherence in the first third minus the last third (long-term drift); lower is better
  score     weighted total (realism counts most)
"""

import sys

import av
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"
LAG = 8  # frames ~ 267 ms at 30 fps


def load(path, size=(576, 320)):
    """-> uint8 CPU tensor [N, 3, H, W] (kept off the GPU; batches are moved as needed)."""
    out = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            if (f.height, f.width) != size:
                f = f.reformat(width=size[1], height=size[0])
            out.append(torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1))
    return torch.stack(out)


def batches(x, n=128):
    for i in range(0, len(x), n):
        yield x[i : i + n].to(DEV).float() / 255


def gray_small(x):
    gs = []
    for b in batches(x):
        g = (0.299 * b[:, 0] + 0.587 * b[:, 1] + 0.114 * b[:, 2]).unsqueeze(1)
        gs.append(F.avg_pool2d(g, 4))  # 144x80
    return torch.cat(gs)


def motion(x):
    g = F.avg_pool2d(gray_small(x), 3, 1, 1)  # blur so sensor noise doesn't count as motion
    return (g[LAG:] - g[:-LAG]).abs().mean(dim=(1, 2, 3))


def flicker(x):
    g = gray_small(x)
    hp = g - F.avg_pool2d(g, 5, 1, 2)
    return (hp[1:] - hp[:-1]).abs().mean().item()


class Realism:
    PAIRS = [
        ("a real photo of a person", "an AI-generated image with artifacts"),
        ("a clean, sharp, high quality photo", "a blurry, noisy, distorted image"),
        ("a natural, realistic face", "a warped, melted, unrealistic face"),
    ]

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor

        name = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        texts = [t for pair in self.PAIRS for t in pair]
        tok = self.proc(text=texts, return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            t = self.model.get_text_features(**tok)
        self.text = F.normalize(t.float(), dim=-1)

    @torch.no_grad()
    def __call__(self, x):
        idx = torch.linspace(0, x.shape[0] - 1, 24).long()  # 24 frames across the clip
        imgs = x[idx].permute(0, 2, 3, 1).numpy()
        inp = self.proc(images=list(imgs), return_tensors="pt").to(DEV)
        f = F.normalize(self.model.get_image_features(pixel_values=inp["pixel_values"].half()).float(), dim=-1)
        logits = 100 * f @ self.text.T  # [N, 2*pairs]
        probs = logits.view(len(idx), len(self.PAIRS), 2).softmax(-1)[..., 0]  # P(positive) per pair
        return probs.mean().item()


def adherence_series(x_in, x_out):
    from scope.core.pipelines.video_depth_anything.gpu_depth import GpuStreamingDepth

    def depth(x):
        m, ds = GpuStreamingDepth(frames_per_chunk=4), []
        for i in range(0, x.shape[0] - 3, 4):
            chunk = (x[i : i + 4].to(DEV).float() / 255 * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0).to(torch.bfloat16)
            ds.append(F.avg_pool2d(m(chunk)[0, 0].float().unsqueeze(1), 8).squeeze(1))
        return torch.cat(ds)

    a, b = depth(x_in), depth(x_out)
    n = min(len(a), len(b))
    a, b = a[:n].flatten(1), b[:n].flatten(1)
    a = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)
    b = (b - b.mean(1, keepdim=True)) / (b.std(1, keepdim=True) + 1e-6)
    return (a * b).mean(1)


@torch.no_grad()
def main():
    inp, outs = sys.argv[1], sys.argv[2:]
    x_in = load(inp)
    m_in = motion(x_in)
    realism = Realism()
    rows = []
    for path in outs:
        x = load(path)
        n = min(len(x), len(x_in))
        x, xi = x[:n], x_in[:n]
        m_out, m_i = motion(x), m_in[: n - LAG]
        static = m_i < torch.quantile(m_i, 0.4)  # moments where you barely moved
        spur = m_out[static].mean().item()
        adh = adherence_series(xi, x)
        third = len(adh) // 3
        drift = (adh[:third].mean() - adh[-third:].mean()).item()
        r = realism(x)
        fl = flicker(x)
        score = 3.0 * r - 8.0 * spur - 10.0 * fl + 1.0 * adh.mean().item() - 1.0 * max(0.0, drift)
        rows.append((score, path, r, spur, fl, adh.mean().item(), drift))
    print(f"{'clip':34s} {'score':>6s} {'realism':>8s} {'spurious':>9s} {'flicker':>8s} {'adhere':>7s} {'drift':>6s}")
    for score, path, r, spur, fl, adh, drift in sorted(rows, reverse=True):
        print(f"{path[-34:]:34s} {score:6.3f} {r:8.3f} {spur:9.4f} {fl:8.4f} {adh:7.3f} {drift:6.3f}")
    print(f"(input reference: spurious {m_in[m_in < torch.quantile(m_in, 0.4)].mean():.4f}, flicker {flicker(x_in):.4f})")


main()
