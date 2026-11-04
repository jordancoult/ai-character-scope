"""ai-livestream: apply a JoyAI LoRA adapter at runtime, next to the (FP8 or bf16) base linears.

Why not merge: the G-Star LoRA delta is ~3e-4 of the weight RMS. Merged into bf16 only ~30% of it survives, and the
5090's FP8 weight quantization turns it into noise (cloud/docs/gstar-lora-finetune.md). So the base weights stay
exactly as served and every adapted linear gets a forward hook  y += (x @ A^T) @ (s*B)^T  in bf16 (rank 32: ~1-2%
extra FLOPs, 336 MB VRAM for all 320 modules).

Off unless JOYAI_LORA is set (bypass by default):
  JOYAI_LORA=/workspace/lora/r3_step50.pt   adapter file from train_lora.py (upstream generator_lora format)
  JOYAI_LORA_SCALE=1.0                      strength (0.5-0.7 if fast motion blurs)
  JOYAI_LORA_STREAMS=img,txt                which streams to adapt
  JOYAI_LORA_MODULES=attn_qkv,attn_proj,mlp.net.0.proj,mlp.net.2   which linears (dropping the MLPs is much cheaper)
Installed by serverpatch/patch_server.py right after load_dit() in joyomni_streaming.py.
"""
import os
import time

import torch

_ALIAS = {"attn_qkv": "attn_qkv", "attn_proj": "attn_proj", "mlp.net.0.proj": "mlp_up", "mlp.net.2": "mlp_down"}


def _hook(mod, inp, out):
    x = inp[0]
    return out + ((x.to(mod._lora_A.dtype) @ mod._lora_A.t()) @ mod._lora_B.t()).to(out.dtype)


def maybe_install(dit):
    path = os.environ.get("JOYAI_LORA", "").strip()
    if not path:
        return dit
    scale = float(os.environ.get("JOYAI_LORA_SCALE", "1.0"))
    streams = [s.strip() for s in os.environ.get("JOYAI_LORA_STREAMS", "img,txt").split(",") if s.strip()]
    modules = [s.strip() for s in os.environ.get("JOYAI_LORA_MODULES", ",".join(_ALIAS)).split(",") if s.strip()]
    from xvideo.models.dit.dit import _fp8_stream_enabled, _maybe_install_fp8_stream
    t0 = time.time()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    lora, lc = ck["generator_lora"], ck["generator_lora_config"]
    factor = scale * lc["lora_alpha"] / lc["r"]
    device = dit.img_in.weight.device
    blocks = dit.double_blocks
    for block in blocks:  # idempotent; on the low-VRAM path the FP8 twins already exist
        for s in ("img", "txt"):
            _maybe_install_fp8_stream(block, s)
    n_fp8 = n_bf16 = 0
    for m in sorted({k.split(".lora_")[0] for k in lora}):
        _, idx, rest = m.split(".", 2)  # double_blocks.<i>.<stream>_<name>
        stream, name = rest.split("_", 1)
        if stream not in streams or name not in modules:
            continue
        block = blocks[int(idx)]
        if _fp8_stream_enabled(block, stream):
            target = getattr(block, f"_fp8_{stream}_{_ALIAS[name]}")
            n_fp8 += 1
        else:
            target = block.get_submodule(f"{stream}_{name}")
            n_bf16 += 1
        A = lora[f"{m}.lora_A.weight"].to(device=device, dtype=torch.bfloat16)
        B = (lora[f"{m}.lora_B.weight"].float() * factor).to(device=device, dtype=torch.bfloat16)
        target.register_buffer("_lora_A", A, persistent=False)
        target.register_buffer("_lora_B", B, persistent=False)
        target.register_forward_hook(_hook)
    print(f"#####[runtime_lora] {path}: step {ck.get('step')}, r={lc['r']} alpha={lc['lora_alpha']} scale={scale}, "
          f"streams={streams} modules={modules}: {n_fp8} FP8 + {n_bf16} bf16 linears hooked in {time.time() - t0:.1f}s", flush=True)
    return dit
