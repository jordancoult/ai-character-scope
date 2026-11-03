"""Merge a lip-sync LoRA adapter into the 1101_face++ backbone -> a full .pth the serving code loads unchanged.

  python merge_lora.py --base <joyai_video_edit_dit_1101_face++.pth> --adapter adapter_stepN.pt \
      --out joyai_video_edit_dit_1101_face++_gstar_lipsync.pth [--scale 1.0]

The server has no LoRA support (deploy/xvideo load_dit does torch.load -> strict load_state_dict into the 894-key
layout), so we write W' = W + scale * (alpha / r) * B @ A for every adapted Linear (fp32 math, stored bf16),
same keys and dtypes as the base. Serve it with JOYOMNI_DIT_CKPT=<out> (cloud/pipelines/joyai/run_server.sh).
--scale < 1 blends the LoRA strength (e.g. 0.5) without retraining.
Memory: the base is memory-mapped; peak RAM ~ one copy of the backbone (~33 GB).
"""

import argparse
import time

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    a = p.parse_args()
    t0 = time.time()
    base = torch.load(a.base, map_location="cpu", mmap=True, weights_only=True)
    if "model" in base and isinstance(base["model"], dict):
        base = base["model"]
    ck = torch.load(a.adapter, map_location="cpu", weights_only=False)
    lora, lc = ck["generator_lora"], ck["generator_lora_config"]
    factor = a.scale * lc["lora_alpha"] / lc["r"]
    mods = sorted({k.split(".lora_")[0] for k in lora})
    out = dict(base)
    for m in mods:
        A = lora[f"{m}.lora_A.weight"].float()
        B = lora[f"{m}.lora_B.weight"].float()
        key = f"{m}.weight"
        if key not in base:
            raise KeyError(f"{key} not in base checkpoint")
        W = base[key]
        out[key] = (W.float() + factor * (B @ A)).to(W.dtype)
    print(f"merged {len(mods)} modules (r={lc['r']}, alpha={lc['lora_alpha']}, scale={a.scale}) "
          f"from step {ck.get('step')} in {time.time() - t0:.0f}s; writing {a.out}")
    torch.save(out, a.out)
    print(f"done in {time.time() - t0:.0f}s; {len(out)} tensors")


if __name__ == "__main__":
    main()
