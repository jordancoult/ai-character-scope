"""Real-time GPU upscalers to 720x1280: plain resize vs Real-ESRGAN compact (SRVGGNetCompact).

Used by tools/scope_bridge.py (--upscale) and tools/upscale_eval.py.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "upscalers"
COMPACT_MODELS = {
    # name: (file, num_conv)
    "animevideov3": ("realesr-animevideov3.pth", 16),
    "general-x4v3": ("realesr-general-x4v3.pth", 32),
}


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN's compact VGG-style SR net (same layout/keys as basicsr's implementation)."""

    def __init__(self, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body += [nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1)]
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class Upscaler:
    """frames [T, 3, h, w] float in [0, 1] on GPU -> [T, 3, H, W] in [0, 1]."""

    def __init__(self, method="bicubic", size=(1280, 720), device="cuda"):
        self.method, self.size = method, size
        self.sharpen = 0.6  # bicubic-sharp unsharp-mask amount
        self.net = None
        if method in COMPACT_MODELS:
            file, num_conv = COMPACT_MODELS[method]
            net = SRVGGNetCompact(num_conv=num_conv)
            state = torch.load(MODELS_DIR / file, map_location="cpu", weights_only=True)
            net.load_state_dict(state.get("params", state))
            self.net = net.to(device).half().eval().to(memory_format=torch.channels_last)

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if self.method == "none":
            return frames
        if self.net is not None:
            x = self.net(frames.half().contiguous(memory_format=torch.channels_last))
            # x4 output is larger than the target; antialiased downscale to 720x1280
            return F.interpolate(x.float(), size=self.size, mode="bilinear", antialias=True).clamp(0, 1)
        mode = "bilinear" if self.method == "bilinear" else "bicubic"
        y = F.interpolate(frames.float(), size=self.size, mode=mode, align_corners=False)
        if self.method == "bicubic-sharp":
            # unsharp mask: y + amount * (y - blur(y)); ~0.3 ms/frame on GPU
            blur = F.avg_pool2d(F.pad(y, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
            y = y + self.sharpen * (y - blur)
        return y.clamp(0, 1)
