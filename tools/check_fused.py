"""Numerical check of the local KV-cache patches vs. the stock flow (same seed, same inputs).

Usage: check_fused.py video n_chunks steps [A_flags] [B_flags]
  flags: comma list of "fused", "ring" (default A = stock, B = fused)
Prints per-chunk differences of the decoded frames. A correct patch differs only by float
noise (reordered sums); a cache-slot bug shows up as large errors right away.
"""

import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline
from scope.core.pipelines.video import load_video
from scope.core.pipelines.streamdiffusionv2.modules import causal_model
from scope.core.pipelines.wan2_1.blocks import denoise

video_path, n_chunks = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
steps = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "750").split(",")]
flags_a = set(filter(None, (sys.argv[4] if len(sys.argv) > 4 else "").split(",")))
flags_b = set(filter(None, (sys.argv[5] if len(sys.argv) > 5 else "fused").split(",")))
pipe_dir = Path(sys.modules[StreamDiffusionV2Pipeline.__module__].__file__).parent
h, w = 576, 320
config = OmegaConf.create({
    "model_dir": str(get_models_dir()),
    "generator_path": str(get_model_file_path("StreamDiffusionV2/wan_causal_dmd_v2v/model.pt")),
    "text_encoder_path": str(get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")),
    "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
    "model_config": OmegaConf.load(pipe_dir / "model.yaml"),
    "height": h, "width": w, "vae_type": "tae",
})
video = load_video(video_path, resize_hw=(h, w)).unsqueeze(0).to("cuda", torch.bfloat16)
prompts = [{"text": "a pink cartoon character in a black cap", "weight": 100}]


def run(flags):
    fused = "fused" in flags
    causal_model.RING_KV = "ring" in flags
    denoise.FUSED_KV = fused
    os.environ["SCOPE_FUSED_KV"] = "1" if fused else "0"  # read by CleanKVCacheBlock
    torch.manual_seed(0)
    pipe = StreamDiffusionV2Pipeline(config, device=torch.device("cuda"), dtype=torch.bfloat16)
    outs, s, e = [], 0, 5
    for i in range(n_chunks):
        torch.manual_seed(1000 + i)
        out = pipe(video=video[:, :, s:e], prompts=prompts, denoising_step_list=steps,
                   noise_scale=0.75, noise_controller=False, base_seed=42)["video"]
        outs.append(out.float().cpu())
        s, e = e, e + 4
    del pipe
    torch.cuda.empty_cache()
    return outs


print('A =', sorted(flags_a) or ['stock'], ' B =', sorted(flags_b))
ref, fus = run(flags_a), run(flags_b)
for i, (a, b) in enumerate(zip(ref, fus)):
    d = (a - b).abs()
    print(f"chunk {i}: mean|diff| {d.mean():.4f}  max {d.max():.3f}  (pixel range 0..1)")
