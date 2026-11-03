# G-Star lip-sync LoRA for JoyAI-Video-Edit (training, merge, eval)

Goal: the live JoyAI G-Star edit should follow the driver's mouth / face motion. The base `1101_face++` model keeps
G-Star's grin almost static (lip-gap correlation with the driver ≈ 0 or negative). We fine-tune a LoRA on paired
driver → Kling-motion-control G-Star clips. Background and the rejected alternatives: `PLAN.md`.

| What | Where |
|---|---|
| Dataset (private) | HF `JordanCo/gstar-lipsync-pairs` (81 clips, 14 pairs; README there). Built by `data/build_pairs.py` + `data/face_align.py --raw` |
| Trainer | `train_lora.py` (+ `setup_train.sh` for the GPU box) |
| Merge to a servable `.pth` | `merge_lora.py` |
| Eval | `eval_infer.py` (GPU box) + `mouth_sync_eval.py` (laptop) |
| Results | **`FINDINGS.md`: first attempt gave no lip-sync gain** (expression only; longer training breaks the face texture); adapters + eval media in HF `JordanCo/joyai-gstar-lora` (private) |

## Method (what `train_lora.py` does)
- **Serving-exact rollouts.** Each step reruns the live inference procedure with upstream's own
  `generate_dmd_latents` (`training/src/train_t2v_dmd.py`): 2 denoising steps, causal 1-latent (8-frame) chunks,
  window 3 + sink, clean history at t=0, `max_temporal_ids=8` absolute, source and text conditioning on the active
  chunk only, G-Star reference image as a VAE latent, MiMo-VL prompt with the driver's first frame as `<image>`.
  Self-forcing: the history is the model's own (detached) output, so there's no train/test mismatch.
- **Exit step per chunk** (upstream's random-exit trick): 70% of chunks back-propagate through the final step's
  x0, 30% through the first step's. Only the exit forward keeps gradients.
- **Segmented backward.** History is detached, so the chunks are independent: each chunk's loss is
  back-propagated right away (`group_backward_fn`, 1 chunk per group). Memory ≈ one forward with activation
  checkpointing: 50.7 GB peak on an H200 for 13-latent (97-frame) clips at 480×840.
- **Loss** (24-px latent grid, mean over channels), selectable by flags:
  - `--w-frame`: full-frame MSE vs the target. With raw Kling clips this **collapses** (run r1, see FINDINGS).
  - `--w-face`, `--w-mouth`: MSE on the target's face / mouth cells (masks from `face_align.py`). With
    face-aligned targets (Kling's face warped onto the driver's face; the dataset itself stays as generated).
  - `--w-anchor`, `--w-anchor-face`: MSE to the frozen base model's output (same noise, no-grad rollout) outside
    and inside the face, keeping background/body/texture where the live model is.
  - Best setting so far (r3): face 1, mouth 4, anchor 1, anchor-face 0.5, `--p-exit-last 1.0`, lr 2e-5.
- **LoRA**: rank 32, alpha 32 on `img/txt_attn_qkv`, `attn_proj`, `mlp.net.0.proj`, `mlp.net.2` in all 40 blocks
  (167.8 M params, fp32), AdamW (r3: lr 2e-5, warm-up 10), cosine to 10%, grad clip 1, 2 clips per step.
- **Cached conditioning**: the VAE latents and MiMo prompt embeddings (both live prompts: the committed long one and
  the user's short one, sampled 50/50) are computed once, then the 7B text encoder is freed.
- Validation every N steps on the held-out driver (`rilango_oficial1`, p15): face/mouth/frame MSE vs Kling with full
  2-step rollouts and fixed noise, plus preview mp4s `driver | base | LoRA | Kling`.

## Run it
```bash
# 1. rent a Hopper/Blackwell-datacenter GPU with >= 80 GB (H200 141 GB used: $3.60/h on Vast), >= 200 GB disk
# 2. copy this folder to /workspace/train, the HF token to ~/.cache/huggingface/token, then:
bash /workspace/train/setup_train.sh all          # venv (py3.10, torch 2.9.1 cu128), weights (~50 GB), dataset
# 3. smoke test (2 clips, 5 steps, ~8 min incl. loading)
/root/tvenv/bin/python train_lora.py --smoke --data /workspace/data --out /workspace/runs/smoke
# 4. train (~21 s per clip sample on an H200)
/root/tvenv/bin/python train_lora.py --data /workspace/data --out /workspace/runs/r1 --steps 350 --accum 2 --lr 1e-4
# 5. eval: base vs adapters through the streaming (KV-cache) pipeline, then score mouth sync on the laptop
/root/tvenv/bin/python eval_infer.py --adapters base runs/r1/adapter_step00350.pt --inputs a.mp4 b.mp4 --out eval/r1
cloud/.venv/bin/python cloud/pipelines/joyai/training/mouth_sync_eval.py <local copy of eval/r1>
# 6. merge for the live server (needs ~35 GB RAM), then JOYOMNI_DIT_CKPT=<out> bash run_server.sh
python merge_lora.py --base .../joyai_video_edit_dit_1101_face++.pth --adapter adapter_step00350.pt --out gstar.pth
```
`setup_train.sh` gotchas: upstream's `training/requirements.txt` pins `torchdata==0.11.0`, which isn't on the cu128
index (needs `--index-strategy unsafe-best-match`); MediaPipe pins numpy<2, so it isn't installed on the box.
