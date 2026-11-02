"""Numerical check of the fused clean-KV pass for the LongLive family vs. the stock flow.

Usage: check_fused_ll.py [pipeline_id] [n_chunks] [steps]   e.g. reward-forcing 8 750,400
Same seed and input clip for both runs; prints per-chunk differences of the decoded frames.
A correct patch differs by float noise only; a cache-slot bug shows large errors right away.
Env overrides (SCOPE_RF_GENERATOR, ...) apply as in start_scope.sh.
"""

import os
import sys

import torch

from scope.core.pipelines.video import load_video
from scope.core.pipelines.wan2_1.blocks import denoise
from scope.server.pipeline_manager import PipelineManager

pid = sys.argv[1] if len(sys.argv) > 1 else "longlive"
n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 8
steps = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "750,400").split(",")]
h, w = 352, 640
video = load_video("assets/webcam_7643311.mp4", resize_hw=(h, w)).unsqueeze(0).to("cuda", torch.bfloat16)
prompts = [{"text": "a man in a wooden cabin, photorealistic", "weight": 100}]


def run(mode):
    fused = "fused" in mode
    from scope.core.pipelines.reward_forcing.modules import causal_model as rf_cm
    rf_cm.ROPE_CACHE = "rope" in mode  # Reward Forcing: cached RoPE keys
    denoise.FUSED_KV = fused
    os.environ["SCOPE_FUSED_KV"] = "1" if fused else "0"  # read by CleanKVCacheBlock
    torch.manual_seed(0)
    pipe = PipelineManager()._load_pipeline_implementation(
        pid, {"height": h, "width": w, "vae_type": "lighttae", "vace_enabled": False, "base_seed": 42})
    text = os.environ.get("CHECK_TEXT") == "1"  # pure generation: no input clip
    req = pipe.prepare(video=not text)
    chunk = req.input_size if req else 12
    outs = []
    for i in range(n_chunks):
        torch.manual_seed(1000 + i)
        kw = {} if text else {"video": video[:, :, i * chunk:(i + 1) * chunk], "noise_scale": 0.9}
        kw["clean_kv_every"] = int(os.environ.get("CHECK_EVERY", "1"))
        out = pipe(**kw, prompts=prompts, denoising_step_list=steps,
                   noise_controller=False)["video"]
        outs.append(torch.as_tensor(out).float().cpu() / (255.0 if out.dtype == torch.uint8 else 1.0))
    del pipe
    torch.cuda.empty_cache()
    return outs


a_mode, b_mode = (sys.argv[4].split(",") if len(sys.argv) > 4 else ["stock", "fused"])
print("A =", a_mode, " B =", b_mode)
ref, fus = run(a_mode), run(b_mode)  # modes: stock, fused, rope, fused+rope
for i, (a, b) in enumerate(zip(ref, fus)):
    d = (a - b).abs()
    print(f"chunk {i}: mean|diff| {d.mean():.4f}  max {d.max():.3f}  (pixel range 0..1)")
