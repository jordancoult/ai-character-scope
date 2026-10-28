"""Generate a stand-in character reference image with Scope's StreamDiffusionV2 in text mode.

Usage: scope/.venv/bin/python tools/gen_character_ref.py "prompt" out.png [H W]
"""

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

prompt, out = sys.argv[1], sys.argv[2]
h, w = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (512, 288)
PIPE_DIR = Path(sys.modules[StreamDiffusionV2Pipeline.__module__].__file__).parent
config = OmegaConf.create({
    "model_dir": str(get_models_dir()),
    "generator_path": str(get_model_file_path("StreamDiffusionV2/wan_causal_dmd_v2v/model.pt")),
    "text_encoder_path": str(get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")),
    "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
    "model_config": OmegaConf.load(PIPE_DIR / "model.yaml"),
    "height": h, "width": w, "vae_type": "tae",
})
pipe = StreamDiffusionV2Pipeline(config, device=torch.device("cuda"), dtype=torch.bfloat16)
frames = []
for _ in range(6):
    out_video = pipe(prompts=[{"text": prompt, "weight": 100}], denoising_step_list=[1000, 750, 500, 250])["video"]
    frames.append(out_video[-1])
img = frames[-1].float().cpu().numpy()
img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
Image.fromarray(img).save(out)
print("saved", out, img.shape)
