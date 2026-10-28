# AI Character Scope — Quickstart

Real-time webcam → your character (VACE reference image) → vertical 9:16 720×1280 → OBS, on an ASUS ROG Zephyrus G16 GU605CW
(RTX 5080 Laptop 16 GB, Blackwell `sm_120`). Everything lives in `~/personal/ai-character-scope/`.

## Tested hardware

| | |
|---|---|
| Laptop | ASUS ROG Zephyrus G16 **GU605CW**, AC power, ASUS profile *Performance*, GPU mode *Hybrid* |
| GPU | NVIDIA GeForce RTX 5080 Laptop, 16 GB, Blackwell `sm_120` (100 W, up to 130 W) |
| CPU | Intel Core Ultra 9 285H (6 P + 8 E + 2 LP-E cores); GPU work pinned to E-cores 6–13 |
| RAM | 64 GB |
| OS / driver | Ubuntu 26.04, kernel 7.0, NVIDIA 580.178 (CUDA 13.0 driver) |
| Stack | PyTorch 2.9.1 + cu128, cuDNN 9.17, Triton 3.5.1, flash-attn 2.8.3, SageAttention 2 |

## Summary of improvements

| | Before | Now |
|---|---|---|
| Live speed | ~3.6 FPS (Scope as shipped) | **~29 FPS** (input-limited at 30 fps camera) |
| Model speed (576×320) | 3.6 FPS | **38.7 FPS** (~10×) |
| Output | square, low-res | **vertical 9:16, 720×1280** |
| Character | none | **your character** from a reference image |
| Control | none | **slider 30 / 24 / 20 / 15** presets, instant switching |

Speed, step by step: TAE fix (3.6→25) · fp8 GEMMs (→29.8) · no CPU↔GPU syncs (→31.5) · compiled blocks (→33.8) ·
fused cache-refresh pass (→37.4) · compiled TAE (→38.7).

Bugs fixed: Scope stuck slow after "video as guide" mode · fp8 slow path at 544×304 · standalone demo out of memory
and broken GPU libraries · bridge sending full-size camera video (CPU heat).

Built: webcam→character→virtual camera bridge, slider page, delayed mic, thermal watchdog, GPU depth guidance.

Rejected (measured): FP4 (fast, loses your pose) · shorter attention window (big face instead of your body) ·
AI upscalers (−20–30 % FPS) · Balanced profile (20 FPS) · P-cores (105 °C).

Still to do: real webcam + OBS test · cone-shaped head · optional GPU clock lock for cooler streams.

## TL;DR

```bash
cd ~/personal/ai-character-scope

# Live: webcam -> character -> /dev/video10 "AI_Scope_Cam" (720x1280) + delayed mic "AI_Character_Mic"
CHARACTER_REF=assets/char_front.png \
PROMPT="a pink cone-headed starfish man with sleepy eyes, wearing a black baseball cap with a red logo and a thick gold chain, 3D Pixar style render" \
./start_scope.sh --bridge
# FPS / quality slider: http://127.0.0.1:8090   (first ~30 s after start are slow: torch.compile warm-up)

./start_scope.sh            # Scope server + web UI only -> http://localhost:8000
./start_sdv2_demo.sh        # standalone StreamDiffusionV2 demo (prompt-only) -> http://localhost:7860
```

Keep the laptop **plugged in and on the ASUS `Performance` profile** (Balanced throttles the CPU too: 29 → 20 FPS).

## The slider (presets)

All presets generate at **576×320 (9:16)** and are upscaled on the GPU to **720×1280** (bicubic + sharpen,
0.4 ms/frame), so switching is instant — no model reload.

| Slider | Mode | Live FPS (measured, 30 fps input) | Notes |
|---|---|---|---|
| **30** | fast: camera latents + character ref, 1 step | **29.0–29.2** (input-limited; pipeline 30.0) | default |
| **24** | fast, 2 denoising steps | 24.6–25.0 | cleanest image, best pose adherence |
| **20** | depth-guided: + depth→VACE, 1 step | ~24 | steadier torso/clothes; preloaded at start |
| **15** | depth-guided, 2 steps | ~15 (model-only 16.8) | |

The bridge: `tools/scope_bridge.py` (crops the camera to 9:16 and downscales *before* WebRTC, GPU upscale,
GPU RGB→YUV). Options: `--preset`, `--upscale none|bicubic|bicubic-sharp|animevideov3|general-x4v3`,
`--noise-scale` (0.75), `--vace-scale`, `--ref a.png [b.png]`.

## How fast is it? (RTX 5080 Laptop, model-only, 576×320, character ref, 1 step)

| Step | FPS | What changed |
|---|---|---|
| Scope as shipped (full Wan VAE, 512² 2-step) | 3.6 | — |
| TAE encode+decode (patch) | 25.0 | at 576×320, 1 step |
| fp8 row-wise GEMMs (`SCOPE_FP8_ROWWISE`) | 29.8 | own `_scaled_mm` kernel path; torchao's fp8 was *slower* |
| no CPU↔GPU syncs on KV indices | 31.5 | host-mirrored cache indices |
| `torch.compile` of blocks minus self-attention (`SCOPE_COMPILE_BLOCKS`) | 33.8 | |
| **fused clean-KV refresh** (`SCOPE_FUSED_KV`) | 37.4 | one DiT pass per chunk instead of two |
| compiled TAE (`SCOPE_COMPILE_TAE`) | **38.7** | ← current default stack |

Live overhead (WebRTC + bridge) is ~10–15 %, so ~33 FPS capacity for a 30 fps camera.
Peak VRAM ~10 GB. 2-step: 25.4 · depth 1-step: 26.7 · 640×352 1-step: ~34 (kv6, est.).

### Tried and rejected (with numbers)

| Idea | Result | Verdict |
|---|---|---|
| NVFP4 FFN (`SCOPE_FP4_FFN=1`, own Triton quant kernel) | 47 FPS with kv3, but pose adherence 0.38 → **0.1**, ghosting | opt-in only |
| 3-frame KV window (`SCOPE_KV_FRAMES=3`) | +10 % FPS, crisper, but adherence 0.38 → 0.30 (draws a big face instead of following you) | off |
| Skip clean-KV pass entirely | 58 FPS but mushy faces | rejected (fused version keeps quality) |
| Ring-buffer KV cache (`SCOPE_RING_KV=1`) | numerically equivalent, no measurable gain | off |
| Real-ESRGAN upscalers | animevideov3 10 ms/frame, general-x4v3 17.5 ms → -20–30 % FPS for a modest sharpness gain | `--upscale` option |
| P-cores / all cores for the CPU side | CPU hits 105 °C in seconds | E-cores 6–13 only |
| ASUS Balanced profile | GPU 88 W instead of ~100 W, but 29 → 20 FPS | stay on Performance |
| fp8 torchao, cudnn.benchmark, smaller sizes (CPU-bound) | no gain | — |

"Adherence" = `tools/adherence.py`: correlation of depth maps between input and output (does the character
occupy your silhouette/pose?), averaged over the clip. Visual checks: `tools/contact_sheet.sh`.

## Character reference tips

- Use a **single clean view** (`assets/char_front.png` = front view cropped from `assets/character_sheet.png`);
  the whole 4-panel sheet as reference adds glitch streaks. Front+face (`--ref a b`) ≈ front alone.
- The prompt matters as much as the image: describe what should appear (`black baseball cap` — "trucker cap" became a cowboy hat).
- Known limit: the cone-shaped head mostly comes out round under the cap (model/VACE limit at this size).

## OBS Studio

`sudo apt install obs-studio`, then:
- **Video:** *Video Capture Device (V4L2)* → **AI_Scope_Cam** (`/dev/video10`, 720×1280).
- **Audio:** *Audio Input Capture (PulseAudio)* → **AI_Character_Mic** = default mic delayed by `AUDIO_DELAY_MS`
  (default 350 ms, an estimate — clap-test and adjust, or use OBS *Sync Offset*). Mute the raw mic in OBS.

Not yet tested end-to-end with the real webcam/OBS — all measurements used a 9:16 crop of a boxing clip.

## Thermals

- `tools/thermal_watchdog.sh GPU_C CPU_C STRIKES` (3 s samples) kills GPU compute processes when over limit:
  `start_scope.sh` uses the **live** profile `90 103 10` (~30 s sustained); benchmarks use strict `87 100 2`.
  The GPU's own target is 87 °C (max operating ~88, slowdown ~90); Intel throttles at 105 °C.
- At the 30 preset: GPU ~100 W, reaches ~84–87 °C after a few minutes; CPU 85–95 °C.
- Optional (needs sudo, your call): lock GPU clocks lower for better perf/W now that there's ~25 % headroom,
  e.g. `sudo nvidia-smi -lgc 1500,1900` (undo: `sudo nvidia-smi -rgc`).
- Cooling pad / hard surface for long streams. GPU work is pinned to E-cores 6–13; compile workers capped.

## Local patches (not upstream) — all opt-in via env vars set in `start_scope.sh`

Scope (`scope/src/scope/core/pipelines/…`):
1. `streamdiffusionv2/pipeline.py` — honor `vae_type="tae"`; clear stale `vace_input_frames/masks`;
   env-gated fp8 / FP4 / compile / KV-window / TAE-compile / depth preload.
2. `wan2_1/fp8_rowwise.py` — fp8 row-wise linear (static per-channel weights, compiled per-row act quant, pads M to 16).
3. `wan2_1/fp4_linear.py` + `wan2_1/nvfp4_triton.py` — NVFP4 linear with a fused Triton activation quantizer (experimental).
4. `streamdiffusionv2/modules/causal_model.py` — host-mirrored KV indices (no syncs); fused-prefix-aware cache
   eviction and split attention; optional ring buffer.
5. `wan2_1/blocks/denoise.py`, `clean_kv_cache.py` — fused clean-KV refresh (`SCOPE_FUSED_KV`).
6. `wan2_1/vace/models/causal_vace_model.py` — VACE hints for fused calls (new-chunk slice, zero-padded prefix).
7. `streamdiffusionv2/depth_vace_block.py`, `video_depth_anything/gpu_depth.py` — all-GPU depth → VACE.
StreamDiffusionV2: T5 CPU offload (`SDV2_TEXT_ENCODER_OFFLOAD`), torch 2.9.1/cu128 + cuDNN 9.17 overrides.

Correctness checks: `tools/check_fused.py` compares patched vs stock outputs chunk by chunk (fused: 0.16 %
mean diff at the first fused chunk with exact attention; ring: bit-identical until the cache fills).

## Layout

```
env.sh              start_scope.sh      start_sdv2_demo.sh      README_QUICKSTART.md
tools/  scope_bridge.py  upscalers.py  bench_scope.py  sweep_scope.sh  contact_sheet.sh  adherence.py
        check_fused.py  thermal_watchdog.sh  gen_character_ref.py  *_microbench.py
assets/ character_sheet.png  char_front.png  char_face.png  portrait_clip.mp4 (+ long loops)  ref_char*.png
models/ Wan2.1-T2V-1.3B  WanVideo_comfy  Autoencoders  StreamDiffusionV2  Video-Depth-Anything-Small  upscalers  sdv2/
scope/  StreamDiffusionV2/  bench/ (results.jsonl + sample videos)
```
Disk ~46 GB. Nothing outside the workspace (HF/torch/uv/npm/Scope caches all redirected by `env.sh`).
