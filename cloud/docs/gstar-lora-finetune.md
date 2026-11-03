# G-Star LoRA fine-tune of JoyAI-Video-Edit: process, results, how to use, recommendations

First attempt, 2026-11-02/03. Code: `cloud/pipelines/joyai/training/` (README = commands, FINDINGS = run details,
PLAN = the original analysis). Everything below is private on HF:
- dataset `JordanCo/gstar-lipsync-pairs`;
- adapters, logs, previews, eval videos and code `JordanCo/joyai-gstar-lora`.

## TL;DR
- **Goal:** G-Star's mouth/face follows the driver in the live JoyAI edit (base `1101_face++` keeps a fixed grin).
- **Mouth:** no real gain. Lip-gap correlation with the driver is unchanged within noise on 3 held-out clips; the
  mouth moves more, but not in time with speech.
- **What improved (owner's verdict from the eval videos):** the **character detail and skin quality, and the hat**,
  best at **25 and 50 steps** (`r2_step25`, `r3_step50`). The eyes also open and follow the driver's. Possible
  downside: some **added blur during fast movement**. Longer training (≥ 75–100 steps) breaks the face into a
  hatched texture.
- **Using it live needs a server change:** merging the LoRA into the weights loses most of it (bf16) or all of it
  (FP8, the 5090 path). Apply it at runtime instead (see "How to use").
- **For the mouth**, the real-time face pass `lpfast` (FasterLivePortrait on the laptop 5080, `--facefix rel`;
  G-Star lip-gap 0.03 → 0.81) is the working route today. A sensible live stack: **LoRA for character detail +
  lpfast for the mouth**.
- **Cost:** ≈ $52, H200 NVL 13.5 h on Vast. About 8 h of that was idle while waiting on a question; see lessons.

## 1. Process
### Data (`training/data/`)
1. **Source:** 16 pairs from the owner's sheet (`gstar_pairs.csv`): a TikTok driver (people dancing / lip-syncing to
   music) and the same performance rendered as G-Star by **Kling motion control v3**. Downloaded unwatermarked via the
   Zyla "TikTok Media Grabber" API (`fetch.py`).
2. **Time alignment by audio** (same song in both): waveform cross-correlation, onset-envelope fallback; trimmed to
   the overlap. Mirrored inputs (`flip`) and the bad output start of p12 (`skip_output_s`) are handled.
   `build_pairs.py` also normalises to 480×840 portrait, 24 fps, identical frame counts.
3. **Gates and lag fix:** MediaPipe lip-gap correlation driver ↔ Kling (0.45–0.91: the targets really are lip-synced).
   A residual lag of ±1–2 frames is corrected on 6 pairs. Dropped: p01 (tiny faces), p08 (misaligned).
4. **Clips:** 81 clips of 97 frames (13 VAE latents), stride 48, from 14 pairs. Held out: driver rilango_oficial1 (p15).
5. **Kling targets are not pixel-aligned** with the drivers: Kling swaps the background, zooms in (face 1.1–1.9×) and
   re-frames the body. The dataset keeps them exactly as generated (the owner's rule). For training,
   `face_align.py` (without `--raw`) derives targets that warp Kling's face onto the driver's face (similarity
   transform from face landmarks) plus face/mouth masks.

### Trainer (`training/train_lora.py`)
- **Serving-exact self-forcing rollouts** using upstream's own `generate_dmd_latents`: 2 steps, causal 8-frame
  chunks, window 3 + sink, `max_temporal_ids=8`, G-Star ref latent, MiMo-VL prompt with the driver's first frame.
  The history is the model's own detached output, so there's no train/test mismatch.
- **Segmented backward:** one chunk at a time, since the history is detached. 50.7 GB peak at 480×840 on an H200,
  ~21–30 s per 4 s clip.
- **Losses** (latent grid): face + 4× mouth vs the (aligned) Kling target; outside the face (and, in r3, inside it
  at 0.5) an **anchor** to the frozen base model's own output with the same noise.
- **LoRA:** rank 32 / alpha 32 on attention + MLP of all 40 blocks (168 M params), AdamW, cosine schedule. Both live
  prompts (the long committed one and the short one) are sampled.
- **Eval** (`eval_infer.py`): base vs adapters through upstream's streaming pipeline with the KV cache (serving
  math, bf16, LoRA unmerged). Scored on the laptop with `mouth_sync_eval.py`.

### Runs
| run | setup | outcome |
|---|---|---|
| r1 | raw Kling targets, full-frame loss, lr 1e-4 | collapsed to brown blur by step 50 (unpredictable background/zoom → MSE averages) |
| r2 | face-aligned face/mouth loss + anchor outside the face, lr 5e-5 | step 25 clean; hatched face from step 50 |
| r3 | + anchor inside the face (0.5), final-step exit only, lr 2e-5 | step 50 clean; step 100 broken in streaming inference |

## 2. Results
Lip-gap correlation with the driver (1 = perfect; noise ≈ ±0.15):

| input | base | r2 s25 | r3 s50 | r3 s100 |
|---|---|---|---|---|
| p15 held-out driver | −0.01 | −0.01 | −0.12 | 0.06 |
| talking head | 0.26 | 0.10 | −0.02 | −0.15 |
| webcam clip | −0.03 | −0.31 | −0.33 | −0.04 |

- **By eye** (`eval/final/<input>/side_by_side.mp4` on HF: input | base | r2 s25 | r3 s50 | r3 s100):
  - **at s25/s50:** sharper, more detailed skin; a cleaner, better-rendered hat; eyes that open with the driver's;
    livelier smiles; possibly more blur in fast motion;
  - **at s100:** broken (hatching, colour blow-out).
- **Why the mouth didn't learn:**
  - per-pixel MSE toward Kling's differently drawn face only learns an average;
  - at 480×840 the mouth covers ~2×6 latent cells, so lip opening is a sub-cell change;
  - only 76 training clips.
- **Why the texture breaks:** the model drifts off its distilled 2-step manifold once the MSE pull outweighs the
  anchor. Validation MSE keeps falling while the picture breaks, so **stop by eye, not by val MSE**.

## 3. How to use (next time the 5090 server runs)
### Important: don't just merge it
- **The LoRA is a tiny change:** about 0.03% of the weights' magnitude.
- **Merged into bf16:** only about 30% of its effect survives, plus about 3× as much rounding noise.
- **On the 5090:** the low-VRAM path then quantizes the weights to **FP8**, and what's left is effectively noise
  (cosine 0.07 to the intended change; measured with the real adapter on synthetic weights).
- So a merged `.pth` on the 5090 will most likely **not** show what the eval videos show.
- `merge_lora.py` is still correct (verified on a synthetic base) and fine for an unquantized bf16 server, with the
  caveat above.

### Runtime LoRA (recommended; small server patch, not written yet)
- **What:** keep the base weights as they are (FP8 on the 5090) and add `y += scale · (x @ Aᵀ) @ Bᵀ` in bf16 next
  to each of the 320 adapted linears. Rank 32 is about 1–2% extra FLOPs, plus 336 MB of VRAM: tight but likely OK on
  32 GB, otherwise use the attention-only subset.
- **Where:** a `cloud/pipelines/joyai/serverpatch/` hook that wraps the DiT's `img/txt_attn_qkv`, `attn_proj`,
  `mlp.net.0.proj` and `mlp.net.2` after loading.
- **Settings:** off by default (`JOYAI_LORA=<adapter.pt>`, `JOYAI_LORA_SCALE=1.0`). The adapter keys are
  `double_blocks.N.<module>.lora_{A,B}.weight`; scale = alpha/r = 1.
- **Check after the patch:** run `cloud/client/mouth_sync.py` plus a side-by-side recording on the same clip with
  and without `JOYAI_LORA`. Look at skin and hat detail and at fast-motion blur.

### Quick test without the patch
- **Hardware:** an RTX PRO 6000 (96 GB, `JOYOMNI_LOW_VRAM=0`).
- **Steps:** merge with `merge_lora.py`, then set `JOYOMNI_DIT_CKPT=<merged.pth>` for `run_server.sh`.
- **Caveat:** if that server also uses FP8 weights, the same loss applies. Check the startup log for the FP8 setting.

### Which adapter
- **`r3_step50`** first: the most stable run; best detail by the owner's check.
- **`r2_step25`** as the alternative.
- **Never `r3_step100`.**
- If fast-motion blur bothers you, try scale 0.5–0.7.

## 4. Recommendations for a successor
1. **Detail LoRA:** the 25–50-step adapters improve character detail. To make that robust:
   - add the runtime-LoRA server path (above) and A/B it live first;
   - if it holds up, retrain on purpose for detail: anchor everywhere except the face/hat/chain, more and faster-motion
     clips (to address the blur), stop at the first artefact (decode previews every 10 steps).
2. **Mouth:** don't repeat pixel-MSE. Options, by expected value:
   - **(a) Use lpfast** (already real-time on the laptop) for the mouth; the LoRA only for character detail.
   - **(b) Perceptual mouth loss:** decode the mouth crop (VAE decode with grad on a small crop) and match lip
     landmarks or lip-reading features of driver vs output, not Kling's pixels. Kling then only shows "what's
     possible", and the driver defines the target opening.
   - **(c) DMD-style distribution matching** with a teacher fine-tuned on the pairs (the upstream Stage-2 machinery;
     needs a teacher, which isn't public: see PLAN §2a).
   - **(d) Closer framing** at train and test time: the mouth must span more 24-px latent cells.
3. **More G-Star data is available:** the owner can supply at least 2× the current pairs. Add them as rows to
   `training/data/gstar_pairs.csv` and re-run `build_pairs.py` → `face_align.py`. Prefer pairs with fast moves (to
   fight the motion blur) and close-up talking faces (for the mouth). More data should let the detail LoRA train
   longer before the texture breaks; for the mouth it helps only together with 2(b)/2(d).
4. **A general expression/mouth LoRA** (any character) only makes sense after 2(b) or 2(d) shows a mouth gain on
   G-Star. Data then: 20–40 min of pairs, 30+ characters, 50+ drivers who talk (not only music).
5. **Data hygiene that mattered:**
   - align by audio, then check the lip-gap lag (fixed ±1–2 frames on 6 of 14 pairs);
   - look at contact sheets: Kling re-frames;
   - keep a held-out driver.
6. **Ops lessons:**
   - never leave a rented GPU idle waiting for an answer (this cost ~$30);
   - check uptime with `vastai show instances`;
   - pre-stage all eval inputs before training ends, so the box can be destroyed right after;
   - avoid `pgrep -f`/`pkill -f` patterns that match your own ssh command line.

## 5. Reproduce
`cloud/pipelines/joyai/training/README.md`: `setup_train.sh` (H200, py3.10, torch 2.9.1 cu128, weights + dataset from
HF), smoke test (`--smoke`, ~8 min), training (r3 flags in FINDINGS.md), `eval_infer.py`, `mouth_sync_eval.py`.
About 1 h of setup + 30 s per clip-sample on an H200 ($3.76/h).
