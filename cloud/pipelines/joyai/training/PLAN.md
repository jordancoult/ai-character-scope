# Fine-tuning JoyAI for lip-sync (RV2V, G-Star): plan

Goal: the live RV2V edit (G-Star ref + instruction) should keep the driver's mouth/lip motion. Data: ~2–3 min of
paired clips (driver video → G-Star video from Kling, lip-synced; several drivers; ~40–60 clips of 3–5 s). The result
must still run unchanged in our real-time server (2 steps, causal 8-frame chunks, `1101_face++` layout).

Baseline to beat (`../experiments/2026-11-02-mouth/FINDINGS.md`): lip-gap correlation 0.03 (G-Star), 0.25
(human ref); the output mouth moves 0.3–0.6× as much as the input. Measure with `cloud/client/mouth_sync.py`.

All references below are to the upstream clone `cloud/third_party/joyai` at `6c4e7a2` (2026-11-02). Upstream
force-pushed its history: the old local HEAD is kept as branch `old-main-pre-force-push`.

## TL;DR
- **No Stage1 (AR-SFT) checkpoint is public.** The released 2-step files are full, merged backbones, and the
  stock Stage2 (DMD) trainer needs Stage1 as its real-score teacher. So "run their Stage2 on our data" isn't
  available as shipped, and even with a teacher, DMD never looks at the target video (see 2a).
- **Recommended: path (c).** Train a LoRA on `1101_face++` directly, with a supervised x0 loss against the Kling
  target, on self-forcing rollouts (2 steps, random exit step, causal chunks, generated history). Build it from
  their own DMD rollout code: about 100 lines on top of `train_t2v_dmd.py`. Then merge the LoRA into a full `.pth`;
  the server loads it unchanged.
- Hardware: **1× H200 141 GB** (simplest), or 1× H100 80 GB with the text encoder moved off the GPU (or 2× H100
  with FSDP). ~10–18 s per micro-step, ~1000–1500 micro-steps → **4–7 GPU-hours, about $15–30 per run**, plus ~1 h of
  setup and downloads.

## 1. What the checkpoints are
| File (HF `jdopensource/JoyAI-Video-Edit/dit/`) | Size | What it is (checked by reading the pickle header over HTTP range requests) |
|---|---|---|
| `pretrain_editing_bid_version.pth` | 32,527,642,399 B | Flat `OrderedDict`, 894 bf16 tensors, no prefix, no LoRA keys. The bidirectional base (T2I/T2V/I2I/V2V), the Stage1 starting point (`training/README.md` "We provide a bidirectional version…") |
| `joyai_video_edit_dit_1101_face++.pth` | 32,527,642,399 B (byte-identical size to `bid`) | Same 894-key layout: a **full 16.26B backbone with the generator LoRA already fused in**. Keys: `img_in.*`, `condition_embedder.*`, `proj_out.*`, `double_blocks.{0..39}.{img,txt}_{mod.modulate_table, attn_qkv, attn_q_norm, attn_k_norm, attn_proj, mlp.net.0.proj, mlp.net.2}` |
| `joyai_video_edit_dit_0811.pth` | 32,527,662,903 B | Same layout (merged) |
| `joyai_video_edit_dit_0804.pth` | 32,527,711,955 B | Same tensors, saved with module metadata; the zip root is named `streaming_dit_step2k` (a step-2k streaming/distillation run) |

- "Merged" follows from the training code: when given a Stage2 file, `load_dit` builds the Stage1 base and calls
  `_merge_generator_lora` → `fuse_lora` + `unload_lora` (`training/src/models/__init__.py:136-161, 164-175, 211-212`).
  The released files contain no `generator_lora` key, so they are that fused result. The rank of the fused LoRA can't
  be recovered.
- The serving default `--max-temporal-ids 8` (`deploy/xvideo/serving/serve_joyomni_streaming.py:2213`, also the
  warm-up at `joyomni_streaming.py:427`) is the **LHAD long-tuning** setting (`configs/stage2_distillation_long-tuning.py:
  max_temporal_ids=8, dmd_rollout_use_relative_temporal_ids=False`). Short Stage2 uses 2
  (`configs/stage2_distillation.py:365`). So the released models are most likely Stage1 → Stage2 → LHAD, merged.
- **Stage2 output format** (`train_t2v_dmd.py:1276-1346`, `save_dmd_checkpoint`): `outputs/.../global_stepN/step_N.pth` =
  `{"step", "dmd_use_lora": True, "generator_adapter_name": "generator", "fake_score_adapter_name": "fake_score",
  "generator_lora": {peft keys}, "generator_lora_config": {r, lora_alpha, target_modules, bias}, "fake_score_lora": {...},
  + optimizer/scheduler states}` and `resilient_dmd_controller.pt`. README: "Stage2 / LHAD LoRA checkpoints do not
  include the full backbone." Default LoRA: rank 64, alpha 64, targets `img/txt_attn_qkv, attn_proj, mlp.net.0.proj,
  mlp.net.2` in all 40 double blocks (`configs/stage2_distillation.py:333-353`) ≈ 336 M params per adapter.
- **Stage1 checkpoint: not public.** The HF repo has only the four DiT files and the VAE. The Diffusers repo
  `jdopensource/JoyAI-Video-Edit-Diffusers` is the 0811 model converted, and there are no other JD repos. The configs
  expect `JOYAI_STAGE1_CKPT=checkpoints/stage1.pth` (`configs/stage2_distillation.py:326`).

### Can 1101_face++ be the Stage2 starting point?
- **As the generator init: yes** (through `dmd_generator_ckpt` in non-LoRA mode, `train_t2v_dmd.py:391-398`, or as
  `dmd_base_ckpt` in LoRA mode).
- **As teacher/critic: no.** In LoRA mode, one shared base serves as the generator (base + `generator` LoRA), the fake
  score (base + `fake_score` LoRA) **and the real score (base with adapters disabled)**
  (`train_t2v_dmd.py:262-338, 879`). With base = 1101, the "teacher" would be the 2-step generator itself, evaluated at
  arbitrary σ∈[0.02, 0.98] with CFG 2.5 (`compute_dmd_gradient`, `:850-928`). A distilled generator has no meaningful
  score there. Non-LoRA mode allows a separate `dmd_real_score_ckpt`, but there's nothing suitable to put in it: the
  teacher must be a causal **SFT** model scored with the teacher-forcing mask (`score_flow_prediction`, `:782-824`,
  "Score all target chunks in one forward with the SFT mask"). The bidirectional base was never trained with that mask.
  Non-LoRA mode would also hold 3 full 16B copies plus 2 full optimizers (≫ 8×H100).

## 2. Candidate paths (ranked)

### (c) RECOMMENDED: supervised LoRA on the 2-step generator, on self-forcing rollouts
Idea: keep the inference procedure exactly (2 steps, chunk size 1 latent = 8 frames, window 3 + sink, history at
t=0, source/text conditioning only on the active chunk, ref image latent). Swap the DMD gradient for a regression
to the Kling target latent of the same chunk.

Everything needed already exists in `train_t2v_dmd.py`:
- `prepare_inputs` (from `train_t2v.py:327-411`) encodes source + target with the VAE, the ref image with the VAE
  (`ref_image_latent`), and the prompt with MiMo-VL using the first source frame as `<image>`. This matches serving
  (`joyomni_streaming.py:685-699`: `"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n"` + the anchor frame; the ref
  image goes in only as a VAE latent through a prefilled KV, `:665-678`).
- `generate_dmd_latents(..., requires_grad=True)` (`:1106-1273`) runs the causal 2-step rollout with a random exit step
  per chunk (`:1205`). Only the exit forward keeps gradients, and the history is the generator's own detached output
  (self forcing: no exposure bias). It returns `generated_latents` with gradients when `dmd_rollout_num_chunks=-1`.
  The target is already in the loop as `target_latents = latents[:, -1]` (`:1623`).
- `load_lora_base_dit` (`:262-338`) with `dmd_base_ckpt=1101_face++.pth` gives frozen base + `generator` adapter.
  `save_dmd_checkpoint` writes the standard LoRA file.

New trainer `cloud/pipelines/joyai/training/train_lipsync_lora.py` (a copy of `main()` of `train_t2v_dmd.py`):
- generator-only loop: drop the fake-score update and the controller (the controller is disabled for RV2V anyway:
  `:1218` skips it when a ref image is present);
- `loss = mean(w * (gen - target)^2)` per chunk, with `w = 1 + λ_mouth · mouth_mask` on the 24-px latent grid (from
  MediaPipe on the target frames, precomputed per clip; λ_mouth ≈ 4–8);
- optional anchor against drift: run the same rollout with `lora_adapter_context(model, disable_adapters=True)` and
  the same noise RNG state (no grad), then add `λ_anchor · mean((1-mouth_mask)·(gen - gen_base)^2)`. This keeps
  identity, background and style where the Kling target is just "a different plausible render". Costs ~+30 % time;
- config: `dmd_use_lora=True, dmd_base_ckpt=<1101_face++>, dmd_lora_rank=32 (alpha 32), dmd_generator_learning_rate=1e-5
  (warm-up 50), dmd_generator_num_inference_steps=2, max_temporal_ids=8, dmd_rollout_use_relative_temporal_ids=False
  (serving settings), dmd_rollout_num_chunks=-1, image/i2i/t2v probs 0 and multiple_videos 1, TRAIN_RESOLUTION=(480, 840),
  frame_buckets=[49, 57, …, 113], bs_mvid=1, gradient_accumulation_steps=2–4, checkpoint_interval=250, cfg_rate=0`.

Risks:
- **Regression to the mean** (blur), mainly at exit step 0 (pure noise → x0). The mapping is close to deterministic
  given the source + ref, and the mouth is strongly determined by the source, so this should be mild. Watch the
  texture of the cone head and the background. Mitigations: anchor loss, low LR, early stopping; weight exit step 1
  more.
- **Breaking few-step behaviour**: low, because training runs the same 2-step schedule (t = 1000 and ≈838: shift 5.159
  on `linspace(1,0,3)`, `scheduling_flow_match_discrete.py:70-90`). A small LoRA rank keeps it close.
- **Overfitting** on 40–60 clips / few drivers (memorized backgrounds or Kling's look). Hold out one driver; eval every
  250 steps.
- **FP8 serving**: the 5090 low-VRAM path quantizes the merged weights to FP8, and a small LoRA delta can partly
  vanish. A/B the merged checkpoint on the 5090 (`mouth_sync.py`) and on the PRO 6000 (bf16).
- **Resolution ceiling**: the latent grid is 24 px (480×840 → 20×35 tokens). In a typical webcam framing the mouth is
  only ~2×3 latent cells. Close-up framing helps; at wide framing the gain may stay small.

### (a) Their Stage2 Resilient-DMD on our data: NOT feasible as shipped
- Needs a Stage1 causal-SFT teacher (not public, see §1). With 1101 as the shared base the teacher is meaningless.
- **Even with a teacher, DMD ignores our targets**: the generator loss uses only (source, ref, prompt) and the
  teacher's score. `target_latents` serves only as a shape template (`generate_dmd_latents(latent_shape_like=…)`,
  `:1106-1145`). Lip-sync would come from the teacher alone, and no teacher we have does lip-sync. Paired Kling data
  helps DMD only if a teacher is first fine-tuned on it, which leads to (b).
- Data/config for it would be the same as (c) (`multiple_videos` shards; with a ref image the Resilient controller is
  off, `:1218`).

### (b) Short Stage1 from the bidirectional base, then Stage2: not worth it
- Stage1 is full fine-tuning of all 16B params (`train_t2v.py:461-480`, `load_optimizer` = AdamW over every
  `requires_grad` param). Only `dit_train_modules` (`models/__init__.py:52-80`) restricts it, and there's no LoRA option.
  Memory: ~16B × (2+2+4) B ≈ 130 GB + activations → ≥ 4×H100 / 2×H200 with FSDP.
- Turning a bidirectional model into a causal AR editor (teacher → resampling forcing, `resampling_forcing_warmup_steps
  = 7000`, `configs/stage1_ar.py:154-158`) takes their large mixed T2I/I2I/T2V/V2V corpus. 50 clips would produce a
  teacher that knows only our clips, and then thousands of DMD steps on top. Expect worse general quality than 1101 at
  10–50× the cost of (c).

### (d) Other options in the code
- **LHAD long-tuning** (`configs/stage2_distillation_long-tuning.py`): the same DMD machinery with longer rollouts
  (`dmd_rollout_num_chunks=91`, ping-pong source traversal `_ref_content_chunk_id`, `:1077-1103`, grouped backward
  `dmd_grad_window_chunks`). It needs the same teacher, so it's out as such. **Reusable in (c)**: set
  `dmd_rollout_num_chunks=[8,16]` + `dmd_grad_window_chunks=4` so rollouts go past the 8-position ceiling like live
  streaming does. For that, gather the target with the same ping-pong `_gather_ref_chunks` (a Kling clip played in
  reverse is still lip-synced to the reversed driver). Phase 2 of (c), only if drift appears.
- **Stage1 trainer loss on 1101 + LoRA** (plain flow matching at random σ with teacher-forced history,
  `forward_and_compute_loss`, `train_t2v.py:414-458`): simpler, but it trains σ values the model never uses and
  pushes it back towards a multi-step flow. It will likely degrade 2-step quality. If used at all, restrict σ to
  {1.0, 0.838} with `teacher_forcing_noise_max_sigma` ~0.003. (c) is the principled version of this.

## 3. Data format the trainer expects (V2V + reference image)
Loader: `training/src/dataset/webdataset.py` (HF `datasets` "webdataset", streaming). Manifest + TAR shards.

**Manifest** `data/rv2v.json` (pass as `train_multiple_videos_data_files`, or `JOYAI_V2V_DATA`), read at `:232-238`:
```json
{"resolved_files": ["/abs/path/rv2v_000.tar", "/abs/path/rv2v_001.tar"], "num_samples": 52}
```
- `ensure_divisible_shards=True` (the Stage2 default) cuts shards to a multiple of world size (`:251-254`). Use ≥ 1
  shard per GPU, or set it False.

**One sample** = files sharing a key inside a TAR (same as ToyData `iv2v.tar`; features declared at `:201-212`):
```
<key>.0.mp4   source/driver video (what the camera would see), H.264
<key>.1.mp4   target video (Kling G-Star output), H.264
<key>.0.png   reference image (assets/char_front.png, the image served live; .0.jpg also accepted)
<key>.json    {"caption": "<instruction prompt>"}
```
Rules that matter (from the code):
- **Mode detection**: `0.mp4` present → `multiple_videos` (`:489-490`). Videos are `0.mp4, 1.mp4, …` in order; the
  **last one is the target**, the earlier ones are the source (`:343-371`, `train_t2v.py:356`).
- **The ref image is used only if** the JSON is exactly `{"caption": ...}` (no other keys) **or** has
  `"metadata": {"ref_image": <anything non-null>}` (`:505-514`). Keep the JSON caption-only, or add
  `{"caption": ..., "metadata": {"ref_image": "char_front.png", "driver": "...", "clip": "..."}}`.
- Caption key `caption` (`train_multiple_videos_caption_keys=["caption"]`). The trainer wraps it as
  `<|im_start|>user\n<image>\n{caption}<|im_end|>\n` with the first source frame as the image (`train_t2v.py:379-391`,
  `:65-68`), the same as serving. **Use our live prompt text verbatim** (`../prompts/gstar.txt`). ToyData captions
  say "the … shown in Image 1"; our prompt's "from the reference image" works live, keep it.
- **Frame alignment is our job**: source and target are sampled with the same `start_ratio`/`frame_rand`, but the
  indices come from each video's own length and fps (`_sample_video_with_bucket`, `:308-341`). **Both files must have
  identical fps and frame count, frame-accurate**, or the lips go off by a few frames. Write both at **24 fps**
  (`fps=24` in the config; higher fps is subsampled per reader) with the same N frames, N = 8k+1 (49, 57, …, 113;
  `VAE_TEMPORAL_FACTOR=8`, `config.py:14, 193`). Clips longer than the largest bucket get a random window, which is fine.
- Resolution: anything. Each frame is resized to cover the bucket and center-cropped, the same crop for source and
  target (`bucket_util.py:50-60`). With `TRAIN_RESOLUTION=(480, 840)` the buckets include 480×840 and 840×480
  (`config.py:15-25, 175-183`). Export source and target at the same aspect, ideally 840×480 or 480×840 to match live.
  The ref image is resized to a ~768² area on a 24-px grid (`bucket_util.py:34-47`); alpha is composited on white
  (`:85-95`).
- Mouth mask (for the (c) loss): sidecar `<key>.mouth.npy` [T_latent, 20, 35] is NOT read by the stock loader. Either
  extend `_process_item` to return it, or precompute it per key into a separate `.npz` that the trainer loads by
  `key`.

Shard writer: plain `tarfile` with names `f"{key}.0.mp4"` etc.; keys must not contain dots. 1 shard per ~25 samples.

## 4. Hardware, time, cost
Model facts: DiT 16.26B params = 32.5 GB bf16. MiMo-VL-7B text encoder ≈ 16 GB bf16; `load_pipeline` puts it on the
GPU of every rank (`models/__init__.py:105-121`). VAE 1.5 GB. Tokens at 480×840: 700 per latent frame (8 frames).
A rollout forward is ~4 chunks of window + active source + ~1k ref + ~0.5k text ≈ 5k tokens. A full-clip score
forward (DMD only) is ≈ 16k tokens.

| Path | Memory (480×840, 49–89 frames, act. checkpointing on) | Minimum | Comfortable |
|---|---|---|---|
| (c) LoRA r32, generator-only | 32.5 (DiT) + 16 (MiMo) + 1.5 (VAE) + ~2 (LoRA + Adam) + ~10–20 activations (graph kept across the rollout chunks; grouped backward caps it) ≈ 65–75 GB | 1× H100 80 GB if MiMo is moved to the CPU after encoding (a few lines) or the prompt embeddings are precomputed | **1× H200 141 GB** |
| (a) Stage2 LoRA DMD (if a teacher existed) | same shared base + 2 adapters + 4 score forwards on 16k tokens ≈ 80–95 GB | 2× H100 (`hsdp_shard_dim=2`) | 1× H200 |
| (b) Stage1 full FT | ~130 GB states + activations | 4× H100 / 2× H200 FSDP2 | 8× H100 |

- FSDP2 applies only when world size > 1 (`fsdp_load.py:72-77`); `cpu_offload` only then too. On 1 GPU nothing is
  sharded.
- Kernels: flex-attention / SDPA / flash-attn with torch 2.9.1 cu128 (`transformer.py:24-82`, `deploy/requirements.txt`).
  Hopper (sm_90) and B200 (sm_100) are fine for training. Our sm_120 serving wheels aren't needed for training.
  An RTX PRO 6000 96 GB would fit (c) but is ~2–3× slower than an H100 in dense bf16.
- Step time (c), estimated: per chunk ≈ 0.33 s per no-grad forward and ≈ 1.3 s per grad forward + recompute +
  backward on an H100/H200. 49 frames (7 chunks) ≈ 10 s, 89 frames (11 chunks) ≈ 16 s, +1–2 s for VAE/MiMo encoding,
  +30 % with the anchor loss. B200 ≈ 2× faster.
- Steps: ~50 clips, batch 1, accumulation 2–4. Train **1000–1500 micro-steps (≈ 20–30 epochs)**, checkpoint every
  250, pick by held-out `mouth_sync` corr + visual check. ≈ **4–7 h**.
- Cost (rough Nov-2026 ranges, check live prices before renting): H200 ≈ $2.5–4/h (Vast at the low end, RunPod at the
  high end) → **$15–30 per run**. H100 80 GB ≈ $1.8–2.7/h → $10–20 per run, but slower to set up because of the memory
  patch. B200 ≈ $4–6/h, about the same total. Add ~1 h for setup + downloads (32.5 GB DiT + ~16 GB MiMo-VL + VAE;
  disk ≥ 150 GB). Budget **~$50** for a first run plus one retry.

## 5. Serving the result
- `deploy/xvideo` has **no LoRA/PEFT support** (no `peft`/`lora` anywhere in `deploy/`). `load_dit`
  (`deploy/xvideo/models/models.py:149-196`) does `torch.load` → optional `"model"` unwrap → prefix strip →
  `load_state_dict(strict=True)` into the 894-key layout.
- So **merge offline** into a full `.pth` with the same 894 keys, bf16, flat dict (like 1101). Script
  `merge_lora.py`: mmap-load the base, for each `generator_lora` pair `X.lora_A.weight` [r, in] / `X.lora_B.weight`
  [out, r] do `W[X.weight] += (alpha/r) · B @ A` (in fp32, cast back to bf16), then `torch.save`. Strip a possible
  `_checkpoint_wrapped_module.` segment from the LoRA keys: activation checkpointing wraps the blocks
  (`utils/activation_checkpointing.py`). Alternatively run upstream's `load_dit` with `dit_ckpt=<step.pth>,
  dmd_base_ckpt=<1101>` (it fuses through `_merge_generator_lora`), then `torch.save(model.state_dict())` without FSDP.
  Needs ~70 GB RAM; the mmap/tensor-wise script needs ~35 GB.
- Shipping: keep only the LoRA file (~0.7 GB at r=32 for both adapters; strip `fake_score_lora` and optimizer
  state) in the private HF repo `JordanCo/ai-livestream-artifacts/joyai/lora/`, and merge on the GPU machine in
  `setup.sh weights` → `JOYAI_DIT=joyai_lipsync_<tag>.pth` (`run_server.sh` already takes `JOYAI_DIT`). Default stays
  `1101_face++` (bypass by default).

## 6. Step by step
1. **Data** (local, `training/data/` scripts, in progress): align each driver/Kling pair (audio cross-correlation,
   `align.py`), cut to 3–5 s, re-encode both at 24 fps with the same frame count 8k+1, the same aspect (840×480 or
   480×840), H.264 CRF ~16. Hold out one driver (~10 %). Write `rv2v_train_*.tar`, `rv2v_val_*.tar` + manifests
   (schema §3), and `mouth_mask.npz` (MediaPipe lips bbox on the target → 24-px latent grid, dilated by 1 cell).
   Sanity: decode one sample through `ImageVideoWebDataset._process_item` locally (CPU) and check alignment on a
   side-by-side.
2. **Code** (local, no GPU): `cloud/pipelines/joyai/training/train_lipsync_lora.py` (§2c), config
   `configs/lipsync_lora.py` subclassing `configs/stage2_distillation.py`, `merge_lora.py`, `setup_train.sh` (clone
   upstream at a pinned commit, venv with `deploy/requirements.txt` + `training/requirements.txt`, `hf download`
   `dit/joyai_video_edit_dit_1101_face++.pth`, `vae/`, `XiaomiMiMo/MiMo-VL-7B-RL-2508`).
3. **Smoke test** on the GPU box (state the hourly price first): `max_train_steps=20`, 2 clips. Check the loss
   decreases, memory, step time; then decode one rollout to mp4 (the decoder is in the pipeline) and confirm it
   matches the live output at step 0 (LoRA B=0 → identical to 1101).
4. **Train**: `PYTHONPATH=<upstream>/training torchrun --nproc_per_node 1 -m train_lipsync_lora --config
   configs/lipsync_lora.py`; `nvidia-smi` + log tail with a hard deadline; checkpoints every 250.
5. **Offline eval** per checkpoint on the held-out driver and `assets/test_face_landscape.mp4`: upstream
   `evaluation/benchmark` streaming inference (`MAX_TEMPORAL_IDS=8`, 2 steps) with the LoRA checkpoint + `--dmd_base_ckpt`.
   Run `mouth_sync.py` (target: G-Star corr ≫ 0.03, motion ratio → ~1) and look at contact sheets (identity, blur).
6. **Merge + live test** on a 5090: `merge_lora.py` → `JOYAI_DIT=...` → our live bridge, `mouth_sync` A/B against 1101
   (FP8 path), 1–2 min stability run. Upload the LoRA to the artifacts repo; the merged file doesn't need uploading.
7. Write `experiments/<date>-lipsync-lora/FINDINGS.md`.

## 7. Risks (summary)
- No lip-sync-capable teacher → DMD paths are out; (c) is supervised and may soften textures.
- 24-px latent grid limits mouth detail at wide framing.
- Kling targets carry Kling's look; the model may drift towards it (fine for G-Star, check the human refs aren't
  hurt: they use the same model, so keep the 1101 file for human refs if needed).
- FP8 quantization on the 5090 may weaken a small LoRA delta.
- Overfitting to few drivers; temporal flicker if rollouts are shorter than live streams (use the phase-2
  long rollouts).
- Training env drift: upstream changes fast (it was force-pushed today); pin the commit.

## 8. Open questions
- How tightly is Kling lip-synced to the driver frames (not just the audio)? Check `mouth_sync` corr between driver and
  Kling output; if it's < ~0.7, the targets themselves cap what we can learn.
- Do the Kling clips keep the driver's framing/camera (pixel-aligned) or reframe? (c) assumes spatial
  correspondence; reframed targets would need cropping to match.
- Does the gain hold on human references, or should the LoRA be G-Star-only (merged file selected per character)?
- Precompute MiMo prompt embeddings (the anchor frame changes per clip) to fit on an H100 80 GB, or just rent an H200?
- Is a mouth-weighted loss enough, or do we need a lip-reading/landmark perceptual loss (decode the mouth crop
  through the VAE)? That costs VAE-decode memory per step.
