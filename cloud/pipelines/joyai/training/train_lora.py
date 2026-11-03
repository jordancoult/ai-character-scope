"""Supervised lip-sync LoRA for JoyAI-Video-Edit 1101_face++ (G-Star), trained on serving-exact rollouts.

What one step does (see README.md / PLAN.md for the reasoning):
  1. pick a cached clip (driver latents, face-aligned Kling target latents, G-Star ref latent, prompt embeddings
     of the driver's first frame + the live prompt, per-latent face/mouth masks);
  2. BASE rollout, no grad: upstream `generate_dmd_latents` with the LoRA disabled = exactly the live model
     (2 steps, causal 1-latent chunks, window 3 + sink, history at t=0, max_temporal_ids=8 absolute);
  3. LoRA rollout with the SAME noise and exit steps; each chunk's exit forward keeps gradients and is
     back-propagated right away (history is detached, so chunks are independent: memory = one forward);
  4. loss per chunk (on the 24-px latent grid, mean over channels):
        face:   (gen - kling)^2 * (w_face * face + w_mouth * mouth) * valid
        anchor: (gen - base)^2  * w_anchor * (1 - face)
     -> mouth/face follow Kling; background, body and identity stay where the live model already is.

Usage (on the training box; paths default to setup_train.sh's layout):
  python train_lora.py --data /workspace/data --out /workspace/runs/r1 [--smoke]
Outputs: <out>/adapter_step<N>.pt (LoRA only, upstream `generator_lora` format), <out>/log.jsonl,
         <out>/previews/step<N>_<key>.mp4 (driver | base | LoRA | aligned Kling).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

W = os.environ.get("JOYAI_DATA", "/workspace")
UP = os.environ.get("JOYAI_TRAINING", f"{W}/JoyAI-Video-Edit/training")
CK = os.environ.get("JOYAI_CKPT", f"{W}/ckpt")
os.environ.setdefault("JOYAI_VAE", f"{CK}/vae")
os.environ.setdefault("JOYAI_TEXT_ENCODER", f"{CK}/MiMo-VL-7B-RL-2508")
sys.path.insert(0, UP)

LIVE_PROMPTS = {
    # committed cloud/pipelines/joyai/prompts/gstar.txt
    "long": "Replace the person with the pink cartoon starfish creature from the reference image: a tall cone-shaped "
            "pink head with no hair, sleepy half-closed eyes, a wide smile, bumpy pink skin, a black baseball cap with "
            "a red G-Star logo, and a thick gold chain on a bare pink chest. Keep exactly the original person's facial "
            "expressions, mouth movements, lip shapes and speech, eye blinks, head pose, gestures and hand movements.",
    # the user's current (shorter) live prompt
    "short": "Replace the person with the pink cartoon starfish creature from the reference image: a tall cone-shaped "
             "pink head with no hair, sleepy half-closed eyes, a wide smile, bumpy pink skin, a black baseball cap "
             "with a red \"G-Star\" logo, and a thick gold chain.",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=f"{W}/data")
    p.add_argument("--out", default=f"{W}/runs/lipsync")
    p.add_argument("--base-ckpt", default=f"{CK}/dit/joyai_video_edit_dit_1101_face++.pth")
    p.add_argument("--ref-image", default=None, help="default: <data>/char_front.png")
    p.add_argument("--val-drivers", nargs="*", default=["rilango_oficial1"])
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--targets", default="attn_mlp", choices=["attn_mlp", "attn"],
                   help="attn_mlp: img/txt qkv, proj, mlp (upstream default); attn: qkv + proj only")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--w-face", type=float, default=1.0)
    p.add_argument("--w-mouth", type=float, default=4.0)
    p.add_argument("--w-anchor", type=float, default=0.0, help="weight of (gen - base)^2 outside the face mask "
                   "(0 = off, and the base rollout is skipped)")
    p.add_argument("--w-frame", type=float, default=1.0, help="weight of the full-frame loss vs the Kling target")
    p.add_argument("--p-exit-last", type=float, default=0.7, help="probability that a chunk exits at the last step")
    p.add_argument("--grad-window", type=int, default=1, help="chunks per segmented backward")
    p.add_argument("--latents", type=int, default=0, help="crop clips to this many latent frames (0 = all)")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--preview-keys", type=int, default=2, help="val clips rendered to mp4 at each validation")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smoke", action="store_true", help="2 clips, 5 steps, val+save at the end")
    p.add_argument("--max-hours", type=float, default=6.0, help="stop (and save) after this wall time")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------- setup
def make_cfg(rank, alpha, targets):
    from configs.stage2_distillation import Stage2DistillationConfig
    cfg = Stage2DistillationConfig()
    # serving settings (deploy/xvideo: 2 steps, max_temporal_ids=8 absolute, window 3 + sink from the model config)
    cfg.dmd_generator_num_inference_steps = 2
    cfg.max_temporal_ids = 8
    cfg.dmd_rollout_use_relative_temporal_ids = False
    cfg.dmd_use_lora = True
    cfg.dmd_lora_rank, cfg.dmd_lora_alpha = rank, alpha
    if targets == "attn":
        cfg.dmd_lora_target_modules = ["img_attn_qkv", "img_attn_proj", "txt_attn_qkv", "txt_attn_proj"]
    cfg.enable_activation_checkpointing = True
    cfg.cfg_rate = 0.0
    cfg.use_vit = True
    return cfg


def load_model(cfg, base_ckpt, device, log, checkpointing=True):
    import src.train_t2v_dmd as D
    from src.models import _read_dit_checkpoint, _strip_common_state_prefixes
    from src.utils.activation_checkpointing import apply_activation_checkpointing
    t0 = time.time()
    model = D._build_base_dit(cfg, device)
    sd = _strip_common_state_prefixes(_read_dit_checkpoint(base_ckpt, "pt"))
    missing, unexpected = model.load_state_dict(sd, strict=True)
    del sd
    model.requires_grad_(False)
    lcfg = D.build_lora_config(cfg, model)
    model.add_adapter(lcfg, adapter_name="generator")
    D.activate_lora_adapter(model, "generator")
    n_lora = 0
    for name, prm in model.named_parameters():
        if "lora_" in name:
            prm.data = prm.data.float()  # fp32 adapter weights (tiny); bf16 base
            prm.requires_grad_(True)
            n_lora += prm.numel()
        else:
            prm.requires_grad_(False)
    if checkpointing:
        model = apply_activation_checkpointing(model, skip_interval=1)
    model.eval()
    log(f"model loaded in {time.time() - t0:.0f}s, LoRA params {n_lora / 1e6:.1f}M, "
        f"targets {len(lcfg.target_modules)} modules, r={lcfg.r} alpha={lcfg.lora_alpha}")
    return model, lcfg


def lora_state(model):
    """LoRA weights in peft's saved-key format (no adapter name, no checkpoint-wrapper segment)."""
    out = {}
    for name, prm in model.named_parameters():
        if "lora_" in name and ".generator." in name:
            key = name.replace("_checkpoint_wrapped_module.", "").replace(".generator.", ".")
            out[key] = prm.detach().to("cpu", torch.bfloat16).clone()
    assert out, "no LoRA parameters found"
    return out


def save_adapter(model, lcfg, path, step, extra=None):
    tm = lcfg.target_modules
    obj = {
        "step": step, "dmd_use_lora": True, "generator_adapter_name": "generator",
        "generator_lora": lora_state(model),
        "generator_lora_config": {"r": lcfg.r, "lora_alpha": lcfg.lora_alpha,
                                  "target_modules": tm if isinstance(tm, str) else sorted(tm), "bias": lcfg.bias},
        "base": "jdopensource/JoyAI-Video-Edit dit/joyai_video_edit_dit_1101_face++.pth",
    }
    if extra:
        obj.update(extra)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_adapter_weights(model, path):
    from peft.utils import set_peft_model_state_dict
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["generator_lora"]
    # the model is wrapped by activation checkpointing; map clean keys onto the wrapped names
    params = dict(model.named_parameters())
    names = {n.replace("_checkpoint_wrapped_module.", ""): n for n in params}
    with torch.no_grad():
        n = 0
        for k, v in sd.items():
            # peft state keys omit the adapter name: "...lora_A.weight" -> "...lora_A.generator.weight"
            kk = k.replace(".lora_A.weight", ".lora_A.generator.weight").replace(".lora_B.weight", ".lora_B.generator.weight")
            prm = params[names[kk]] if kk in names else None
            if prm is None:
                raise KeyError(f"adapter key {k} not found in model")
            prm.copy_(v.to(prm.dtype))
            n += 1
    return ck


# ----------------------------------------------------------------------------------------------- data
def read_video(path):
    import decord
    vr = decord.VideoReader(str(path))
    arr = vr.get_batch(list(range(len(vr)))).asnumpy()  # T H W C uint8
    return arr


def to_pixels(arr):
    """uint8 [T,H,W,C] -> float [-1,1] [1, C, T, H, W] (clips are already 480x840, the 24-px bucket size)."""
    from src.dataset.webdataset import resize_crop_normalize
    from einops import rearrange
    x = torch.from_numpy(arr).permute(0, 3, 1, 2)
    H, W_ = x.shape[-2:]
    x = resize_crop_normalize(x, (H // 24 * 24, W_ // 24 * 24))
    return rearrange(x, "t c h w -> 1 c t h w")


def ref_pixels(path):
    from PIL import Image
    from einops import rearrange
    from src.dataset.webdataset import to_rgb_on_white, resize_crop_normalize
    from src.dataset.bucket_util import reference_size
    img = Image.open(path)
    px = torch.from_numpy(np.array(to_rgb_on_white(img)))
    h, w = px.shape[:2]
    th, tw = reference_size(h, w)
    px = rearrange(px, "h w c -> 1 c h w")
    px = resize_crop_normalize(px, (th, tw))
    return rearrange(px, "1 c h w -> 1 c 1 h w")


@torch.no_grad()
def build_cache(pipeline, cfg, data_dir, rows, ref_path, device, log, gen, keep_px=False):
    """Encode every clip once: VAE latents (as in prepare_inputs / serving), MiMo prompt embeds per live prompt."""
    from PIL import Image
    ref = ref_pixels(ref_path).to(device, pipeline.vae.dtype)
    ref_lat = pipeline._sample_vae_latents(ref, enable_denormalization=cfg.enable_denormalization)[:, :, :1]
    cache = []
    t0 = time.time()
    for i, r in enumerate(rows):
        cd = Path(data_dir) / "clips"
        src = read_video(cd / f"{r['key']}.src.mp4")
        tgt = read_video(cd / f"{r['key']}.tgt.mp4")
        n = min(len(src), len(tgt))
        n = (n - 1) // 8 * 8 + 1
        src, tgt = src[:n], tgt[:n]
        ps, pt = to_pixels(src).to(device, pipeline.vae.dtype), to_pixels(tgt).to(device, pipeline.vae.dtype)
        ls = pipeline._sample_vae_latents(ps, enable_denormalization=cfg.enable_denormalization, generator=gen)
        lt = pipeline._sample_vae_latents(pt, enable_denormalization=cfg.enable_denormalization, generator=gen)
        anchor = Image.fromarray(src[0])  # serving: the first camera frame, at generation size
        embeds = {}
        for name, text in LIVE_PROMPTS.items():
            e, m = pipeline.encode_prompt_multiple_images(
                prompt=[f"<|im_start|>user\n<image>\n{text}<|im_end|>\n"], images=[anchor], device=device,
                max_sequence_length=cfg.text_token_max_length)
            embeds[name] = (e.to(torch.bfloat16), m)
        mk = np.load(cd / f"{r['key']}.masks.npz")
        T = ls.shape[2]
        face = torch.from_numpy(mk["face"].astype(np.float32))[:T]
        mouth = torch.from_numpy(mk["mouth"].astype(np.float32))[:T]
        valid = torch.from_numpy(mk["valid"].astype(np.float32))[:T]
        assert face.shape[1:] == ls.shape[-2:], (face.shape, ls.shape)
        cache.append({"key": r["key"], "driver": r["driver"], "src": ls.float().cpu(), "tgt": lt.float().cpu(),
                      "embeds": {k: (v[0].cpu(), v[1].cpu()) for k, v in embeds.items()},
                      "face": face, "mouth": mouth, "valid": valid, "src_px": src if keep_px else None})
        if i % 10 == 0:
            log(f"cached {i + 1}/{len(rows)} ({r['key']}, T_lat={T}, txt={embeds['long'][0].shape[1]}) "
                f"{time.time() - t0:.0f}s")
    return cache, ref_lat.float()


# ----------------------------------------------------------------------------------------------- rollout + loss
class ExitPlan:
    """Replaces upstream's random exit index so the base and LoRA rollouts use the same exits."""

    def __init__(self):
        self.seq, self.i = [], 0

    def set(self, seq):
        self.seq, self.i = list(seq), 0

    def __call__(self, num_timesteps):
        e = self.seq[self.i] if self.i < len(self.seq) else num_timesteps - 1
        self.i += 1
        return min(e, num_timesteps - 1)


def rollout(D, cfg, pipeline, model, item, ref_lat, prompt, seed, exits, plan, device, *, lora, grad_fn=None):
    src = item["src"].to(device)
    tgt_shape = item["tgt"].to(device)
    e, m = item["embeds"][prompt]
    inputs = {"task_type": "v2v", "ref_video_latent": src, "ref_image_latent": [ref_lat.to(device)]}
    gen = torch.Generator(device="cuda").manual_seed(seed)
    plan.set(exits)
    T = src.shape[2]
    if lora:
        cfg.dmd_use_lora = True
        D.activate_lora_adapter(model, "generator")
        out, _, _ = D.generate_dmd_latents(
            cfg, pipeline, model, tgt_shape, e.to(device), m.to(device), inputs, gen, requires_grad=grad_fn is not None,
            rollout_num_chunks_override=T if grad_fn is not None else None, group_backward_fn=grad_fn)
        return out
    # base: adapters off; generate_dmd_latents would re-enable them when dmd_use_lora is set, so unset it
    cfg.dmd_use_lora = False
    model.disable_adapters()
    try:
        with torch.no_grad():
            out, _, _ = D.generate_dmd_latents(cfg, pipeline, model, tgt_shape, e.to(device), m.to(device), inputs, gen,
                                               requires_grad=False)
    finally:
        model.enable_adapters()
        D.activate_lora_adapter(model, "generator")
        cfg.dmd_use_lora = True
    return out.detach()


def chunk_loss(gen, k0, item, base, a, device):
    """gen: [1,C,t,h,w] chunk(s) starting at latent k0. Returns (loss, parts dict)."""
    t = gen.shape[2]
    tgt = item["tgt"][:, :, k0:k0 + t].to(device)
    bas = base[:, :, k0:k0 + t].to(device)
    face = item["face"][k0:k0 + t].to(device)[None]
    mouth = item["mouth"][k0:k0 + t].to(device)[None]
    valid = item["valid"][k0:k0 + t].to(device)[None, :, None, None]
    g = gen.float()
    e_t = ((g - tgt) ** 2).mean(1)  # [1,t,h,w]
    e_b = ((g - bas) ** 2).mean(1)
    cells = e_t.numel()
    l_face = (e_t * a.w_face * face * valid).sum() / cells
    l_mouth = (e_t * a.w_mouth * mouth * valid).sum() / cells
    l_anchor = (e_b * a.w_anchor * (1 - face)).sum() / cells if a.w_anchor else e_t.new_zeros(())
    l_frame = (e_t * a.w_frame * valid).sum() / cells
    with torch.no_grad():
        fm = (face * valid).sum().clamp_min(1e-6)
        mm = (mouth * valid).sum().clamp_min(1e-6)
        parts = {"face_mse": float((e_t * face * valid).sum() / fm), "mouth_mse": float((e_t * mouth * valid).sum() / mm),
                 "anchor_mse": float((e_b * (1 - face)).sum() / (1 - face).sum().clamp_min(1e-6))}
    parts["frame_mse"] = float(e_t.mean())
    return l_face + l_mouth + l_anchor + l_frame, parts


def crop_item(item, T):
    if not T or item["src"].shape[2] <= T:
        return item
    # keep latent 0 semantics simple: crop from the start (latent 0 = a single frame, like a stream start)
    out = dict(item)
    for k in ("src", "tgt"):
        out[k] = item[k][:, :, :T]
    for k in ("face", "mouth", "valid"):
        out[k] = item[k][:T]
    return out


@torch.no_grad()
def decode(pipeline, cfg, lat):
    px, _ = pipeline._decode_chunk(lat.to(torch.bfloat16)[:, None], None, cfg.enable_denormalization)
    px = (px[0, 0].float() / 2 + 0.5).clamp(0, 1)  # c t h w
    return (px.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)


def write_mp4(frames, path, fps=24):
    import subprocess
    H, W_ = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_}x{H}",
                          "-r", str(fps), "-i", "-", "-vf", "scale='min(1920,iw)':-2", "-c:v", "libx264", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-movflags", "+faststart",
                          str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


@torch.no_grad()
def validate(D, cfg, pipeline, model, val, ref_lat, plan, device, a, step, out, base_cache):
    """Full 2-step rollouts (every chunk exits at the last step), fixed noise; face/mouth MSE vs Kling, anchor MSE."""
    res = {"face_mse": [], "mouth_mse": [], "anchor_mse": [], "frame_mse": [], "base_face_mse": [], "base_mouth_mse": [],
           "base_frame_mse": []}
    for vi, item in enumerate(val):
        T = item["src"].shape[2]
        exits = [1] * T
        seed = 777 + vi
        if item["key"] not in base_cache:
            base_cache[item["key"]] = rollout(D, cfg, pipeline, model, item, ref_lat, "long", seed, exits, plan, device,
                                              lora=False).cpu()
        base = base_cache[item["key"]]
        gen = rollout(D, cfg, pipeline, model, item, ref_lat, "long", seed, exits, plan, device, lora=True).detach()
        _, parts = chunk_loss(gen, 0, item, base, a, device)
        _, bparts = chunk_loss(base.to(device), 0, item, base, a, device)
        for k in ("face_mse", "mouth_mse", "anchor_mse", "frame_mse"):
            res[k].append(parts[k])
        for k in ("face_mse", "mouth_mse", "frame_mse"):
            res["base_" + k].append(bparts[k])
        if vi < a.preview_keys:
            fb, fl, ft = decode(pipeline, cfg, base.to(device)), decode(pipeline, cfg, gen), decode(pipeline, cfg, item["tgt"].to(device))
            n = min(len(fb), len(fl), len(ft), len(item["src_px"]))
            frames = [np.hstack([item["src_px"][i], fb[i], fl[i], ft[i]]) for i in range(n)]
            (out / "previews").mkdir(exist_ok=True)
            write_mp4(frames, out / "previews" / f"step{step:05d}_{item['key']}.mp4")
    return {k: float(np.mean(v)) for k, v in res.items() if v}


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    import src.train_t2v_dmd as D
    from src.models import load_pipeline
    plan = ExitPlan()
    D._sample_exit_index = plan  # deterministic, shared exits for base and LoRA rollouts

    rows = [json.loads(l) for l in open(Path(a.data) / "metadata.jsonl")]
    train_rows = [r for r in rows if r["driver"] not in a.val_drivers]
    val_rows = [r for r in rows if r["driver"] in a.val_drivers]
    if a.smoke:
        train_rows, val_rows = train_rows[:2], (val_rows or train_rows)[:1]
        a.steps, a.save_every, a.val_every, a.warmup, a.preview_keys = 5, 5, 5, 1, 1
    log(f"args {vars(a)}")
    log(f"train clips {len(train_rows)} ({len({r['pair_id'] for r in train_rows})} pairs), val clips {len(val_rows)}")
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)

    cfg = make_cfg(a.rank, a.alpha, a.targets)
    model, lcfg = load_model(cfg, a.base_ckpt, device, log)
    pipeline = load_pipeline(cfg, model, device)
    log(f"pipeline loaded; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    ref_path = a.ref_image or str(Path(a.data) / "char_front.png")
    train, ref_lat = build_cache(pipeline, cfg, a.data, train_rows, ref_path, device, log, gen)
    val, _ = build_cache(pipeline, cfg, a.data, val_rows, ref_path, device, log, gen, keep_px=True)
    # free the 7B text encoder (prompt embeddings are cached)
    pipeline.text_encoder = None
    torch.cuda.empty_cache()
    log(f"cache done; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))) * 0.9 + 0.1))
    base_val_cache = {}
    t_start = time.time()
    logj = open(out / "log.jsonl", "a")
    v = validate(D, cfg, pipeline, model, val, ref_lat, plan, device, a, 0, out, base_val_cache)
    log(f"val step 0: {v}")
    logj.write(json.dumps({"step": 0, "val": v}) + "\n")
    logj.flush()
    rng = random.Random(a.seed)
    for step in range(1, a.steps + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "face_mse": 0.0, "mouth_mse": 0.0, "anchor_mse": 0.0, "frame_mse": 0.0}
        nparts = 0
        for _ in range(a.accum):
            item = crop_item(rng.choice(train), a.latents)
            prompt = rng.choice(list(LIVE_PROMPTS))
            T = item["src"].shape[2]
            exits = [1 if rng.random() < a.p_exit_last else 0 for _ in range(T)]
            seed = rng.randrange(1 << 30)
            if a.w_anchor:
                base = rollout(D, cfg, pipeline, model, item, ref_lat, prompt, seed, exits, plan, device, lora=False)
            else:
                base = item["tgt"]  # unused (anchor weight 0)
            state = {"k": 0}

            def grad_fn(group_latents, overrides):
                nonlocal nparts
                k0 = state["k"]
                loss, parts = chunk_loss(group_latents, k0, item, base, a, device)
                t = group_latents.shape[2]
                (loss * t / T / a.accum).backward()
                state["k"] += t
                agg["loss"] += float(loss.detach()) * t / T / a.accum
                for kk in ("face_mse", "mouth_mse", "anchor_mse", "frame_mse"):
                    agg[kk] += parts[kk]
                nparts += 1

            cfg.dmd_grad_window_chunks = a.grad_window
            rollout(D, cfg, pipeline, model, item, ref_lat, prompt, seed, exits, plan, device, lora=True, grad_fn=grad_fn)
            assert state["k"] == T, (state["k"], T)
        gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        if torch.isfinite(gn):
            opt.step()
        sched.step()
        dt = time.time() - t0
        rec = {"step": step, "loss": agg["loss"], "grad_norm": float(gn), "lr": sched.get_last_lr()[0], "sec": round(dt, 1),
               **{k: agg[k] / max(1, nparts) for k in ("face_mse", "mouth_mse", "anchor_mse", "frame_mse")},
               "mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        logj.write(json.dumps(rec) + "\n")
        logj.flush()
        if step % 5 == 0 or step <= 3 or a.smoke:
            log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        over_time = (time.time() - t_start) / 3600 > a.max_hours
        if step % a.save_every == 0 or step == a.steps or over_time:
            save_adapter(model, lcfg, out / f"adapter_step{step:05d}.pt", step, {"args": vars(a)})
            log(f"saved adapter_step{step:05d}.pt")
        if step % a.val_every == 0 or step == a.steps or over_time:
            v = validate(D, cfg, pipeline, model, val, ref_lat, plan, device, a, step, out, base_val_cache)
            log(f"val step {step}: {v}")
            logj.write(json.dumps({"step": step, "val": v}) + "\n")
            logj.flush()
        if over_time:
            log("max-hours reached, stopping")
            break
    log("TRAIN_DONE")


if __name__ == "__main__":
    main()
