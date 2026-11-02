# AI Character Scope — Quickstart

Real-time webcam → your character (VACE reference image) → vertical 9:16 720×1280 → OBS, on an ASUS ROG Zephyrus G16 GU605CW
(RTX 5080 Laptop 16 GB, Blackwell `sm_120`). Everything lives in the repo folder (clone it anywhere; paths below are relative to it).

## Tested hardware

| | |
|---|---|
| Laptop | ASUS ROG Zephyrus G16 **GU605CW**, AC power, ASUS profile *Performance*, GPU mode *Hybrid* |
| GPU | NVIDIA GeForce RTX 5080 Laptop, 16 GB, Blackwell `sm_120` (100 W, up to 130 W) |
| CPU | Intel Core Ultra 9 285H (6 P + 8 E + 2 LP-E cores); GPU work pinned to E-cores 6–13 |
| RAM | 64 GB |
| OS / driver | Ubuntu 26.04, kernel 7.0, NVIDIA 580.178 (CUDA 13.0 driver) |
| Stack | PyTorch 2.9.1 + cu128, cuDNN 9.17, Triton 3.5.1, flash-attn 2.8.3, SageAttention 2 |

## Full local setup

Linux, NVIDIA driver ≥ 570 (CUDA 12.8-capable), Node.js + npm, ffmpeg. Clone anywhere, under any folder name:
every script derives the repo root from its own location (`env.sh` exports it as `$AICS_ROOT`). Where each
component comes from: [README.md → Components](README.md#components-and-where-they-come-from).

```bash
git clone git@github.com:jordancoult/ai-character-scope.git   # branch main; optional: add a target folder, any name works
cd ai-character-scope
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$PWD/bin UV_NO_MODIFY_PATH=1 sh
source env.sh

# Patched forks (branch blackwell-realtime). Remote "fork" = ours; "origin" = upstream (read-only, never push)
git clone -o fork -b blackwell-realtime git@github.com:jordancoult/scope.git scope
git -C scope remote add origin https://github.com/daydreamlive/scope.git
git clone -o fork -b blackwell-realtime git@github.com:jordancoult/StreamDiffusionV2.git StreamDiffusionV2
git -C StreamDiffusionV2 remote add origin https://github.com/chenfengxu714/StreamDiffusionV2.git
(cd scope && uv sync && cd frontend && npm ci && npm run build)

# Virtual camera (once)
sudo apt install v4l2loopback-dkms v4l-utils pulseaudio-utils ffmpeg
echo 'options v4l2loopback devices=1 video_nr=10 card_label="AI_Scope_Cam" exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf && sudo modprobe v4l2loopback
```

Run: `CHARACTER_REF=path/to/character.png PROMPT="describe the character" ./start_scope.sh --bridge`,
then open the slider at http://127.0.0.1:8090 and add **AI_Scope_Cam** + **AI_Character_Mic** in OBS.
Character art, test clips and model weights are not in this repo (`assets/`, `models/` are git-ignored).

### Models

All into `./models` (gitignored). Scope also fetches its `daydreamlive/*` artifacts by itself when a pipeline
loads; the commands below make it explicit and cover the extras.

```bash
HF="uvx --from huggingface_hub[cli] hf download"
# Shared by every Scope Wan pipeline + StreamDiffusionV2 (~15 GB)
$HF daydreamlive/Wan2.1-T2V-1.3B config.json Wan2.1_VAE.pth --local-dir models/Wan2.1-T2V-1.3B
$HF daydreamlive/Wan2.1-T2V-1.3B --include 'google/*' --local-dir models/Wan2.1-T2V-1.3B
$HF daydreamlive/WanVideo_comfy config.json umt5-xxl-enc-fp8_e4m3fn.safetensors Wan2_1-VACE_module_1_3B_bf16.safetensors --local-dir models/WanVideo_comfy
$HF daydreamlive/Autoencoders config.json taew2_1.pth lighttaew2_1.pth lightvaew2_1.pth --local-dir models/Autoencoders
$HF daydreamlive/StreamDiffusionV2 config.json wan_causal_dmd_v2v/model.pt --local-dir models/StreamDiffusionV2
$HF daydreamlive/Video-Depth-Anything-Small config.json video_depth_anything_vits.pth --local-dir models/Video-Depth-Anything-Small
ln -f models/Autoencoders/taew2_1.pth models/taew2_1.pth
ln -sfn StreamDiffusionV2/wan_causal_dmd_v2v models/wan_causal_dmd_v2v

# LongLive family (MODEL=longlive / reward-forcing / memflow)
$HF daydreamlive/LongLive-1.3B config.json models/longlive_base.pt models/lora.pt --local-dir models/LongLive-1.3B
$HF daydreamlive/Reward-Forcing-T2V-1.3B config.json rewardforcing.pt --local-dir models/Reward-Forcing-T2V-1.3B
$HF daydreamlive/MemFlow config.json base.pt lora.pt --local-dir models/MemFlow
$HF zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt causal-forcing++/framewise-2step.pt --local-dir models/Causal-Forcing
$HF MeiGen-AI/OPSD-V checkpoints/opsdv_longlive_lora.pt --local-dir models/OPSD-V

# Standalone StreamDiffusionV2 demo (full Wan2.1 1.3B; its T5 is reused by LongLive 2.0)
$HF Wan-AI/Wan2.1-T2V-1.3B --local-dir models/sdv2/Wan2.1-T2V-1.3B
mkdir -p StreamDiffusionV2/ckpts StreamDiffusionV2/wan_models
ln -sfn ../../models/taew2_1.pth StreamDiffusionV2/ckpts/taew2_1.pth
ln -sfn ../../models/StreamDiffusionV2/wan_causal_dmd_v2v StreamDiffusionV2/ckpts/wan_causal_dmd_v2v
ln -sfn ../../models/sdv2/Wan2.1-T2V-1.3B StreamDiffusionV2/wan_models/Wan2.1-T2V-1.3B

# Optional: style LoRAs (--lora path:scale) and Real-ESRGAN upscalers (bridge --upscale)
$HF Alissonerdx/UltraWanComfy ultrawan_1k_comfy.safetensors --local-dir models/lora/ultrawan
$HF AlekseyCalvin/HSToricColorT2VWan1.3B_v2_rank128LoRA_bySilverAgePoets HSTwan13LoRA2_000003750.safetensors --local-dir models/lora/hstoric
$HF wookiekim/Wan2.1-T2V-1.3B-SOLACE adapter_model.safetensors adapter_config.json --local-dir models/lora/solace
mkdir -p models/upscalers && for f in realesr-animevideov3.pth realesr-general-x4v3.pth; do
  curl -sL -o models/upscalers/$f https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/$f; done
```

LongLive 2.0 (`models/LongLive-2.0-5B/`, plus the `longlive2/` clone): see
[tools/longlive2/README.md](tools/longlive2/README.md). The optional Matrix-Game decoder used in its tests:
`$HF Skywork/Matrix-Game-3.0 MG-LightVAE_v2.pth --local-dir models/LongLive-2.0-5B/Matrix-Game-3.0`.

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

Just want to start it? See [START_HERE.md](START_HERE.md).

```bash
cd ai-character-scope   # your clone (any folder name/location)

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

## Quality sweep (realism first)

`tools/quality_metrics.py` scores clips rendered offline from a real webcam recording (`bench_scope.py
--max-seconds 40 --out`): **realism** (CLIP no-reference score, weighted most), **invented motion** (output
change 250 ms apart where the input was still), **flicker** (frame-to-frame fine detail), **adherence**
(depth correlation with the input) and **drift**. Blonde reference, 576×320, model-only FPS:

| Config | Realism | Invented motion | Flicker | Adherence | FPS |
|---|---|---|---|---|---|
| 1 step, random noise (old 30 preset) | 0.53 | 0.024 | 0.0034 | 0.34 | 40 |
| 2 steps, random noise | 0.77 | 0.027 | 0.0039 | 0.26 | 26 |
| 2 steps, **fixed noise** | 0.73 | 0.018 | 0.0018 | 0.32 | 26 |
| **depth + 2 steps + fixed noise** | 0.69 | 0.019 | 0.0030 | **0.47** | 18 |
| **depth + 2 steps + random noise** | **0.75** | 0.021 | 0.0039 | 0.46 | 18 |
| 3 steps / 9-frame history / noise 0.6 / weaker ref | no better | | | | |
| *your camera (reference)* | — | *0.0015* | *0.0017* | — | — |

Findings: fixed noise halves flicker at 2 steps (but leaves a haze at 1 step); depth guidance is the only thing
that stops head-turn overshoot (adherence 0.32 → 0.47); invented motion stays ~10× the camera's in every config
(feedback through the attention cache). A cache reset visibly cleans up accumulated degradation — the controls
page has a **Refresh character** button. The reference dropdown has **(none - prompt only)**: same speed (the ref is encoded once into the cache at reset, not per chunk), the prompt alone sets the look, and in a quick test head turns followed the input more closely. A **Noise pattern** dropdown (fixed by default; random / auto, or `NOISE_MODE=`) overrides
the preset's fixed-noise choice live.

## Root cause of the sparkles: noise-level offset (fixed)

In video mode Scope mixes the camera latents with noise at the slider level `s`, but tells the model the first
step is `1000·s − 100`, i.e. less noise than is really there. The model removes only what it was told about;
the rest stays as residual noise. It shows up as colored sparkles, 8×8-pixel blocks (one latent cell each)
morphing on their own, and a pulse once per chunk when the noise is re-drawn. `SCOPE_NOISE_T_OFFSET=0` (default
in `start_scope.sh`) tells the true level. Scored on the webcam clip (StreamDiffusionV2):

| Preset | Noise | Offset 100 (stock): realism / flicker / adhere | Offset 0: realism / flicker / adhere |
|---|---|---|---|
| 30 (1 step) | random | 0.50 / 0.0053 / 0.26 | **0.75 / 0.0035 / 0.41** |
| 30 (1 step) | fixed | 0.39 / 0.0016 / 0.36 | **0.74** / 0.0022 / **0.40** |
| 20 (depth, 2 steps) | random | 0.72 / 0.0060 / 0.48 | **0.78 / 0.0045** / 0.49 |
| 20 (depth, 2 steps) | fixed | 0.69 / 0.0040 / 0.49 | **0.71 / 0.0037** / 0.47 |

Same FPS. The 1-step "fixed noise leaves a haze" finding above was this offset, not fixed noise.

## Experimental: LongLive (`MODEL=longlive`)

`MODEL=longlive ./start_scope.sh --bridge` swaps in NVIDIA's LongLive (Scope ships the pipeline; weights in
`models/LongLive-1.3B`, ~8.5 GB). It is trained for long streams (frame sink + KV recache) and keeps a bigger
memory (12 latent frames, 3 per chunk). Same presets and controls page; depth presets fall back to no depth.

- Its built-in LoRA is merged and the model converted to fp8 row-wise at load (bf16 + LoRA + VACE doesn't fit
  in 16 GB; torchao fp8 fails on it). ~21 FPS on the 24 preset, 12.9 GB VRAM. lighttae VAE (it expects
  normalized latents). Fused KV is off by default for the LongLive family; since 2026-11-01 it works for
  LongLive and Reward Forcing with `SCOPE_FUSED_KV=1` (bit-identical output, see the model status below).
- Defaults to **random noise** (fixed noise piles up into confetti) and noise 0.9 (at 0.75 it barely changes the camera).
- **Status:** clean with the noise-offset fix above (the sparkles were that offset). Fixed noise still piles up
  into artifacts on LongLive, so it stays on random noise.

## Model status (2026-11-01): what runs locally and how fast

All at 640×352 landscape on the desk stock clip, live through Scope + bridge, feedback off. Details and videos
are in `experiments/`.

| Model (`MODEL=`) | Setting | fps, depth off | fps, depth on | Notes |
|---|---|---|---|---|
| LongLive (`longlive`) | 1 step | 21.3 | 17.4 | stutter fix: 157 → 27 stutters/min |
| LongLive | 3 steps, clean pass every 2nd | 13.4 | 10.1 (11.4 VACE hint reuse) | sharper |
| Causal Forcing chunk-wise (`reward-forcing` + `SCOPE_RF_GENERATOR`) | 2 steps, fused + RoPE cache + clean every 2nd | **22.6** | 15.1 | best 2-step speed |
| Causal Forcing chunk-wise | 1 step, fused + RoPE cache | **~26** | 17 | |
| Causal Forcing++ frame-wise | 2 steps | | 9.4 | background glitches; skip |
| LongLive 2.0 (`longlive2`, 5B NVFP4) | 2 steps, 1024×576, clean every 2nd | 13–15 in Scope | — | prompt-only (no camera); about 15 fps standalone |
| OPSD-V LoRA (`SCOPE_LONGLIVE_LORA`) | on LongLive | same fps | | swaps LongLive's LoRA |

Fast 2-step Causal Forcing (`experiments/2026-11-01-fused-ll/FINDINGS.md`):
```bash
MODEL=reward-forcing SCOPE_FUSED_KV=1 SCOPE_RF_ROPE_CACHE=1 \
SCOPE_RF_GENERATOR=$PWD/models/Causal-Forcing/chunkwise/causal_forcing.pt SCOPE_RF_GENERATOR_KEY=generator \
ORIENTATION=landscape BRIDGE_ARGS="--size 352 640" ./start_scope.sh --bridge
# then on the page: Denoiser → Steps 2, Clean pass every 2nd chunk; depth off for 20+ fps
```

Limits found: cost is linear in model passes per chunk (about 230 ms per pass at 640×352). Depth/VACE adds 25–45 %.
Reference-image identity is weak in the streaming 1.3B models. The next step is a rented cloud GPU
(`experiments/2026-11-01-cloud-gpu-research.md`); the work lives in `cloud/`, separate from this local setup.

Experiments index: `2026-10-31-dream-variants` (dream presets) · `2026-11-01-longlive2` (LongLive 2.0) ·
`2026-11-01-ll1-speed` (smoothness, 3-step cost) · `2026-11-01-fused-ll` (fused clean pass, RoPE cache).

## Latency (camera frame in → character frame out, measured)

`tools/latency_test.sh` streams `assets/latency_flash.mp4` (white flash every 5 s) and times each flash through the
whole loop (`scope_bridge.py --latency-probe`). Stock Scope kept a full 30-frame output queue (it paces at exactly the
production rate, so a backlog never drains):

| Output cushion (`SCOPE_OUTPUT_TARGET_FRAMES`) | Latency | Stutters (gap > 66 ms) |
|---|---|---|
| stock (30-frame queue) | 1344 ms | — |
| 6 | 569 ms | 3 / 1735 |
| **3 (default)** | **462 ms** | 3 / 1734 |
| 1 | 413 ms | 10 / 1737 |

Breakdown at the default: input transport 36 ms · inside Scope 357 ms (chunk fill + model + cushion) · output
transport 77 ms. Plus camera capture and the viewer (OBS) on top. Most of the rest is structural: the model works on
4-frame chunks.

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
8. `longlive/`, `reward_forcing/modules/causal_model.py` — fused-prefix attention (pre-eviction prefix keys);
   Reward Forcing cached RoPE keys + fp32 RoPE (`SCOPE_RF_ROPE_CACHE`); generator/LoRA swaps (`SCOPE_RF_GENERATOR*`,
   `SCOPE_LONGLIVE_LORA`); clean pass every Nth chunk (`clean_kv_every`), attention switch, VACE hint reuse.
9. `longlive2/` — LongLive 2.0 as a Scope text-to-video pipeline (local port).
10. `server/frame_processor.py`, `process.py` — input upload on a side stream; CPU uint8 output per chunk.
StreamDiffusionV2: T5 CPU offload (`SDV2_TEXT_ENCODER_OFFLOAD`), torch 2.9.1/cu128 + cuDNN 9.17 overrides.

Correctness checks: `tools/check_fused.py` compares patched vs stock outputs chunk by chunk (fused: 0.16 %
mean diff at the first fused chunk with exact attention; ring: bit-identical until the cache fills).
`tools/check_fused_ll.py` does the same for LongLive / Reward Forcing (fused: bit-identical, text and video mode).

## Layout

```
env.sh              start_scope.sh      start_sdv2_demo.sh      README_QUICKSTART.md
tools/  scope_bridge.py  upscalers.py  bench_scope.py  sweep_scope.sh  contact_sheet.sh  adherence.py
        check_fused.py  thermal_watchdog.sh  gen_character_ref.py  *_microbench.py
assets/ character_sheet.png  char_front.png  char_face.png  portrait_clip.mp4 (+ long loops)  ref_char*.png
models/ Wan2.1-T2V-1.3B  WanVideo_comfy  Autoencoders  StreamDiffusionV2  Video-Depth-Anything-Small  upscalers  sdv2/
        LongLive-1.3B  LongLive-2.0-5B  Reward-Forcing-T2V-1.3B  MemFlow  Causal-Forcing  OPSD-V  lora/
scope/  StreamDiffusionV2/  longlive2/  bench/ (results.jsonl + sample videos)  cloud/ (cloud pipelines)
```
Disk ~46 GB. Nothing outside the workspace (HF/torch/uv/npm/Scope caches all redirected by `env.sh`).
