# Research: open models where the character's face and mouth follow the user by design (2026-11-02)

**Question:** which open model can replace JoyAI-Video-Edit so that one reference image (G-Star) plus a live webcam
gives the character the user's pose, expressions and lip movements, in real time (25–30 fps, < 2 s)?
JoyAI gets the body right at 25–27 fps on a 5090, but its mouth does not follow the speaker, and the LivePortrait
paste (`cloud/facefix/`) looks janky.

Research only: papers, READMEs and code read from `/tmp` clones. Nothing was run.
**P** = published by the authors · **E** = my estimate (method under the tables) · **C** = read from the code.

## TL;DR
- **Try EditaLive first** ([GVCLab/EditaLive](https://github.com/GVCLab/EditaLive), Apache-2.0, weights out since
  2026-09-14). It is **the only released, streaming, full-body model whose face is driven by design.** It takes
  Wan2.2-Animate-14B and makes it causal with 2-step LoRAs. Every 12-frame chunk gets the webcam's ViTPose skeleton
  plus a **512×512 face crop**, which an implicit motion encoder turns into features fed to the DiT through a face
  adapter. That is the Wan-Animate face path, built for expression and lip reenactment. It ships `edit_webcam.py`.
- **Speed: not 25 fps on one 5090.** **P:** 14.5 fps (16.4 with Flash-VAED) at 384×672 on 1× H100, 0.83 s per
  chunk. **E:** ~6–10 fps on a 5090; ~25–35 fps on a B200 at 384×672. Latency is ~1.5–2.5 s on the machine (12-frame
  chunks), which is at or above our 2 s bar.
- **The better models have no weights:** LiveAnimate (code "coming soon") and Wan-Animate-2-Lite (paper only).
  Both are also Wan-Animate-based and run on 2–4 H100s.
- **Face-only or offline models** (PersonaLive, LivePortrait, X-Portrait 2, HunyuanPortrait, Animate-X,
  UniAnimate-DiT, StableAnimator) cannot replace JoyAI. PersonaLive's paper says it fails on cartoon faces.

## Candidates
| Model | Code / weights | Streaming | Face/mouth driven by | Published speed (P) | Cartoon reference | Hands |
|---|---|---|---|---|---|---|
| **EditaLive** ([paper 2608.27123](https://arxiv.org/abs/2608.27123), [repo](https://github.com/GVCLab/EditaLive), [HF](https://huggingface.co/huaichang/EditaLive)) | **yes**: inference + 3 LoRAs + Flash-VAED on Wan2.2-Animate-14B; no training code | **yes**, causal, 2 steps [1000, 250], 12-frame chunks (first chunk 9), sink + rolling KV, 75 % sparse attention (VSA) | Wan-Animate path (C): ViTPose skeleton plus a 512² face crop → LIA-style motion encoder (`motion_dim=20`) → FaceAdapter cross-attention every 5 blocks. Implicit, no landmarks | **14.47 fps / 0.829 s per chunk, 1× H100, 384×672**; 16.4 fps with Flash-VAED; `torch.compile` adds ~1.3×. VRAM with FP8 at 384×672: 31.4 GiB, or 17.5 GiB with `--low_vram weights` (only ~2 % slower) | No examples. Built for "human or human-like" subjects. Inherits Wan-Animate's cross-identity animation. Trained mainly as an *editor* (reference = edited first frame); swapping in a foreign reference is shown only as a side capability | Body skeleton only. Fingers are a stated limitation |
| **LiveAnimate** ([2608.11745](https://arxiv.org/abs/2608.11745), [page](https://liveanimate.github.io/)) | **no**: [repo](https://github.com/liveanimate/LiveAnimate) says "coming soon" | yes, 3 steps, 12-frame blocks, PR-Sink KV | 133 whole-body ViTPose keypoints, including face; Wan2.2-Animate-14B + LoRA | 12.41 fps (1× H100), **19.63 fps (2× H100)**, 22.13 fps (4× H100); 0.61 s per block on 2× H100; 480×480 | Humans only in the paper | 21 keypoints per hand |
| **Wan-Animate-2-Lite** ([2608.06009](https://arxiv.org/abs/2608.06009), [repo](https://github.com/Wan-Video/Wan-Animate-2)) | **no**: only Base and Distillation (10-step, offline) are released; the paper promises Base weights only | yes, 3 steps, 8-frame chunks | Driving video consumed directly (no extractor), so expressions transfer very well | **24 fps at 400×720 on 4× H100** (1 GPU VAE encode, 2 GPUs DiT, 1 GPU VAE decode) | **Yes**: the paper shows cartoon characters, robots and animals, and trains on anthropomorphic cartoon animals | Paper claims intricate hand motion |
| **PersonaLive** ([2512.11253](https://arxiv.org/abs/2512.11253), [repo](https://github.com/GVCLab/PersonaLive)) | yes, webcam UI, TensorRT | yes, 4 steps, 4-frame micro-chunks | X-NeMo face motion embedding + LivePortrait 3D implicit keypoints (head pose) | 15.8 fps / 0.25 s, 1× H100, 512² (~20 fps with TinyVAE). Users report ~5 fps on a 4090/5080 without TensorRT | **Bad**: the paper says it fails on cartoons and animals (eyes and mouth distort) | none (portrait only) |
| FasterLivePortrait / LivePortrait (+ animal model) | yes | per frame | implicit keypoints | 30+ fps on a 3090 with TensorRT ([repo](https://github.com/warmshao/FasterLivePortrait)) | animal model exists; cartoons hit or miss | face only. What `facefix/` already uses |
| X-Portrait 2 / X-NeMo, X-UniMotion | X-NeMo yes ([mirror](https://gitee.com/ByteDance/x-nemo-inference)); X-UniMotion: paper only | no (offline) | implicit motion latents (X-UniMotion: face, body, 2 hand tokens) | none | stylized portraits (X-Portrait) | X-UniMotion only, no code |
| Follow-Your-Emoji-Faster ([2509.16630](https://arxiv.org/abs/2509.16630)) | code unclear | no | expression-aware landmarks | "2.6× lossless speed-up", not real time | yes (freestyle portraits) | face only |
| Animate-X, UniAnimate-DiT, StableAnimator, HunyuanPortrait | yes | no (offline; minutes per clip) | pose (+ implicit face for HunyuanPortrait) | — | Animate-X is built for anthropomorphic characters | body models: pose only |
| SVEET ([2609.24788](https://arxiv.org/abs/2609.24788)), StreamDiffusionV2 | yes | yes | none (vid2vid editing) | SVEET 15 fps on 1× H100 | no reference-identity drive | — |
| MirageLSD | closed API | — | — | — | — | — |
| StreamAvatar, LiveAvatar, SoulX-LiveAct | partly | yes | **audio** (talking avatars), body not from the webcam | 20 fps on 2–5 H100/H800s | — | — |

## EditaLive per GPU (384×672 unless noted; the webcam UI also offers 480×832)
| GPU | fps | Latency on the machine | Notes |
|---|---|---|---|
| 1× H100 | **P 14.5 (16.4 Flash-VAED)**; E ~19–21 with compile + FP8 | P 0.83 s per chunk; E ~1.6–2 s total with chunk capture | tested hardware; the VSA kernel is built for sm_90a |
| 2× H100 (`--second-device`) | E ~20–24 | E ~1.3–1.7 s | pose extraction and VAE run on the 2nd card (built-in option) |
| RTX 5090 (32 GB) | E ~6–10 | E ~2.5–3.5 s | needs `--low_vram weights` (17.5 GiB). sm_120: flash-attn 2.7.2 has no wheel, so build it or use SageAttention; the FastVideo kernel may not build, so use `--vsa_backend sdpa` or flash. Our JoyAI tricks (FP8 with fp16 accumulate, SageAttention) could add ~1.5× |
| RTX PRO 6000 (96 GB) | E ~7–11 | E ~2.5–3 s | no low-VRAM mode needed; same sm_120 caveats |
| B200 | E ~25–35 (480×832: ~18–25) | E ~1–1.4 s | sm_100: the VSA kernel is 9.0a-only, so it falls back to flash or SDPA; flash-attn needs an sm_100 build. Unverified |

**How the estimates were made:** dense BF16/FP8 tensor throughput relative to H100 (5090 ≈ 0.2–0.45×, depending on
the FP8 accumulate mode; B200 ≈ 2–2.3×), applied to the published H100 number. The 5090 range is checked against
JoyAI, a 16B 2-step DiT that does 215 ms per 8-frame chunk at 840×480 on a 5090 with heavy tuning. EditaLive's
stock code is less tuned.

Wan-Animate models are trained at 16 fps. Plan to generate at ~16 fps and interpolate (RIFE) to 30 fps, at the cost
of about one extra frame of latency.

## Recommendation
1. **Quality gate (cheapest, no code changes): offline EditaLive on the RTX 5090 we already know.**
   `edit_streaming.py --video assets/webcam_*.mp4 --ref_image assets/char_front.png --fp8 --low_vram weights
   --vsa_backend sdpa --short_edge 384 --long_edge 672` (portrait follows the reference). Use a talking clip and
   judge lips, expressions, G-Star's cone head and the hands against JoyAI. ~1–2 h of setup (flash-attn for sm_120).
   If lip sync or identity on G-Star is poor, stop here: Wan-Animate's face path is the mechanism everything else on
   this list shares.
2. **Speed: 1× H100 first** (published numbers, known kernels): expect 15–20 fps and ~1.5–2 s. Then try a **B200**
   for ≥ 25 fps (estimated, sm_100 risk). A 5090 alone will not reach real time with EditaLive.
3. Webcam patches needed (C): `webcam/service.py` accepts only "camera" reference frames (v1; a custom reference is
   on their TODO list), and `webcam/config.py` is landscape-only. Both are small changes, since the pipeline's
   `stream_start(ref_image, …)` already takes any RGB image. Or write our own `live_bridge` adapter around
   `stream_start`.
4. Watch for weights from LiveAnimate and Wan-Animate-2-Lite. Lite has the best published cartoon handling and
   expression transfer, but it needs 4× H100 for 24 fps.

## Risks
- **Body proportions:** G-Star has a cone head and short, thick arms. The webcam path has no pose retargeting
  (offline has `--retarget`), so a human skeleton can stretch him into a human silhouette. JoyAI avoided this by
  editing instead of skeleton-driving.
- **Editor bias:** EditaLive was tuned to keep the source person's identity and apply edits. With a foreign
  reference, faces may pull back toward the webcam person ([issue #2](https://github.com/GVCLab/EditaLive/issues/2)
  reports identity changes in the other direction under style edits).
- **Fingers are not tracked** (body skeleton only), so hands may be no better than JoyAI's.
- **Latency:** 12-frame chunks plus 0.8 s of compute already make ~1.5–2 s on an H100, above JoyAI's 0.46 s.
- **Immature project:** the repo is ~2 months old (98 stars), the kernels are built from source, it was tested only
  on an H100, and there are no Blackwell reports yet.
- **Fallback if nothing passes:** keep JoyAI and replace the LivePortrait paste with an audio-driven lip-sync model
  on the generated face. MuseTalk-class models claim 30 fps on a V100 (not verified on cartoons).
