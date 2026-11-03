"""Inference eval: base 1101_face++ vs LoRA adapters on held-out driver videos, through upstream's streaming
pipeline (KV cache, 2 steps, causal 1-latent chunks, window 3 + sink, max_temporal_ids=8, ref image prefilled),
i.e. the same math as the live server in bf16 (no FP8 / CUDA graphs).

  python eval_infer.py --adapters base runs/r1/adapter_step00350.pt ... --inputs drv1.mp4 drv2.mp4 \
      --out /workspace/eval/r1 [--frames 193] [--prompt long|short]

For every input: <out>/<input>/<variant>.mp4 (480x840, 24 fps), input.mp4 (the exact frames fed), and
side_by_side.mp4 (input | variant ... ; <=1920 wide, h264 high@4.1, faststart). Same noise seed for every variant.
Mouth-sync is computed afterwards on the laptop (mouth_sync_eval.py) since MediaPipe pins numpy<2.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lora as T  # noqa: E402  (sets up the upstream import path)


def load_frames(path, n, w=480, h=840, fps=24, start=0.0):
    """Decode, resample to 24 fps, scale to cover w x h and center-crop (what the live bridge does in portrait)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path), "-vf",
           f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    k = (len(arr) - 1) // 8 * 8 + 1
    return arr[:k]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapters", nargs="+", required=True, help="'base' and/or adapter .pt files")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=193)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--prompt", default="long", choices=list(T.LIVE_PROMPTS))
    p.add_argument("--base-ckpt", default=f"{T.CK}/dit/joyai_video_edit_dit_1101_face++.pth")
    p.add_argument("--ref-image", default=f"{T.W}/data_raw/char_front.png")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)  # noqa: E731
    device = torch.device("cuda")
    from src.models import load_pipeline
    from PIL import Image
    cfg = T.make_cfg(32, 32, "attn_mlp")
    # rank/targets of the model's adapter slot must match the adapters: read them from the first adapter file
    first = next((x for x in a.adapters if x != "base"), None)
    if first:
        ck0 = torch.load(first, map_location="cpu", weights_only=False)
        lc = ck0["generator_lora_config"]
        cfg.dmd_lora_rank, cfg.dmd_lora_alpha = lc["r"], lc["lora_alpha"]
        # exact module paths from the adapter keys (resolve_lora_target_modules accepts full names)
        cfg.dmd_lora_target_modules = sorted({k.split(".lora_")[0] for k in ck0["generator_lora"]})
        del ck0
    model, _ = T.load_model(cfg, a.base_ckpt, device, log, checkpointing=False)
    pipeline = load_pipeline(cfg, model, device)
    ref = T.ref_pixels(a.ref_image).to(device, pipeline.vae.dtype)
    ref_lat = pipeline._sample_vae_latents(ref, enable_denormalization=cfg.enable_denormalization)[:, :, :1]
    prompt = f"<|im_start|>user\n<image>\n{T.LIVE_PROMPTS[a.prompt]}<|im_end|>\n"
    summary = {}
    for inp in a.inputs:
        name = Path(inp).stem
        d = out / name
        d.mkdir(exist_ok=True)
        frames = load_frames(inp, a.frames, start=a.start)
        T.write_mp4(list(frames), d / "input.mp4")
        px = T.to_pixels(frames)  # 1 c t h w in [-1, 1]
        results = {}
        for ad in a.adapters:
            tag = "base" if ad == "base" else Path(ad).stem.replace("adapter_", "")
            if ad == "base":
                model.disable_adapters()
            else:
                model.enable_adapters()
                T.load_adapter_weights(model, ad)
                import src.train_t2v_dmd as D
                D.activate_lora_adapter(model, "generator")
            gen = torch.Generator(device="cuda").manual_seed(a.seed)
            t0 = time.time()
            with torch.no_grad():
                vid = pipeline(
                    prompt=[prompt], height=frames.shape[1], width=frames.shape[2], num_frames=len(frames),
                    images=[Image.fromarray(frames[0])], reference_visual_content=px.to(device, torch.bfloat16),
                    num_inference_steps=2, guidance_scale=1.0, generator=gen, output_type="pt",
                    max_sequence_length=cfg.text_token_max_length, enable_denormalization=cfg.enable_denormalization,
                    use_vit=True, ref_image_latent=[ref_lat.to(torch.bfloat16)], max_temporal_ids=8,
                    use_relative_temporal_ids=False,
                ).videos  # b n t c h w in [0, 1]
            v = (vid[0, 0].permute(0, 2, 3, 1).float().numpy() * 255).round().astype(np.uint8)
            results[tag] = v
            T.write_mp4(list(v), d / f"{tag}.mp4")
            log(f"{name} {tag}: {len(v)} frames in {time.time() - t0:.1f}s")
        model.enable_adapters()
        n = min([len(frames)] + [len(v) for v in results.values()])
        tiles = [frames[:n]] + [results[k][:n] for k in results]
        T.write_mp4([np.hstack([t[i] for t in tiles]) for i in range(n)], d / "side_by_side.mp4")
        summary[name] = {"frames": n, "variants": list(results)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    log("EVAL_DONE")


if __name__ == "__main__":
    main()
