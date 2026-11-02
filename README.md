# ai-character-scope

Real-time webcam → stylized character (from one reference image) → vertical 720×1280 virtual camera for OBS,
built on [Daydream Scope](https://github.com/daydreamlive/scope) + StreamDiffusionV2 and tuned for an
RTX 5080 Laptop GPU (Blackwell, `sm_120`, 16 GB). ~29 FPS live (input-limited), 38.7 FPS model-only at 576×320
(stock Scope: 3.6 FPS). Includes a 30/24/20/15 FPS quality slider, delayed-mic audio sync and a thermal watchdog.

## Project map (start here)
| Read | For |
|---|---|
| [AGENTS.md](AGENTS.md) | The whole picture in one page: goal, local vs cloud, rules, secrets, how to pick up the work |
| [README_QUICKSTART.md](README_QUICKSTART.md) | **Local realtime** (laptop GPU, Scope): setup, presets, model status table, benchmarks, every patch |
| [START_HERE.md](START_HERE.md) | Local: just start it (commands for the live webcam → character → OBS path) |
| [cloud/README.md](cloud/README.md) | **Cloud realtime pipelines** (rented RTX 5090): pipeline index (character swap first), quickstart, live bridge to OBS |
| [cloud/pipelines/joyai/README.md](cloud/pipelines/joyai/README.md) | JoyAI-Video-Edit: recipe, numbers (25–27 fps, ~1 s), limits |
| [experiments/](experiments/) · [cloud/pipelines/*/experiments/](cloud/pipelines/joyai/experiments/) | Dated test runs with FINDINGS.md: local model tests / cloud tests |
| [cloud/docs/lessons.md](cloud/docs/lessons.md) | Mistakes that cost time once |

Repos: this one (`jordancoult/ai-character-scope`, everything) + `jordancoult/scope` (Scope fork with the local
model patches, checked out at `scope/`). Large files live outside git: local weights in `models/`, cloud build
artifacts in the private HF repo `JordanCo/ai-livestream-artifacts`.

**Results, benchmarks, presets, every patch and what was tried/rejected: [README_QUICKSTART.md](README_QUICKSTART.md).**

## Tested hardware

| | |
|---|---|
| Laptop | ASUS ROG Zephyrus G16 **GU605CW**, AC power, ASUS profile *Performance*, GPU mode *Hybrid* |
| GPU | NVIDIA GeForce RTX 5080 Laptop, 16 GB, Blackwell `sm_120` (100 W, up to 130 W) |
| CPU | Intel Core Ultra 9 285H (6 P + 8 E + 2 LP-E cores); GPU work pinned to E-cores 6–13 |
| RAM | 64 GB |
| OS / driver | Ubuntu 26.04, kernel 7.0, NVIDIA 580.178 (CUDA 13.0 driver) |
| Stack | PyTorch 2.9.1 + cu128, cuDNN 9.17, Triton 3.5.1, flash-attn 2.8.3, SageAttention 2 |

## Setup (Linux, NVIDIA driver ≥ 570, CUDA 12.8-capable)

Scripts assume the workspace lives at `~/personal/ai-character-scope` (edit `env.sh` / scripts to move it).

```bash
git clone git@github.com:jordancoult/ai-character-scope.git ~/personal/ai-character-scope
cd ~/personal/ai-character-scope
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$PWD/bin UV_NO_MODIFY_PATH=1 sh
source env.sh

# Patched forks (branch blackwell-realtime)
git clone -b blackwell-realtime https://github.com/jordancoult/scope.git scope
git clone -b blackwell-realtime https://github.com/jordancoult/StreamDiffusionV2.git StreamDiffusionV2
(cd scope && uv sync && cd frontend && npm ci && npm run build)

# Models for Scope's streamdiffusionv2 pipeline (~15 GB, into ./models)
HF="uvx --from huggingface_hub[cli] hf download"
$HF daydreamlive/Wan2.1-T2V-1.3B config.json Wan2.1_VAE.pth --local-dir models/Wan2.1-T2V-1.3B
$HF daydreamlive/Wan2.1-T2V-1.3B --include 'google/*' --local-dir models/Wan2.1-T2V-1.3B
$HF daydreamlive/WanVideo_comfy config.json umt5-xxl-enc-fp8_e4m3fn.safetensors Wan2_1-VACE_module_1_3B_bf16.safetensors --local-dir models/WanVideo_comfy
$HF daydreamlive/Autoencoders config.json taew2_1.pth lighttaew2_1.pth lightvaew2_1.pth --local-dir models/Autoencoders
$HF daydreamlive/StreamDiffusionV2 config.json wan_causal_dmd_v2v/model.pt --local-dir models/StreamDiffusionV2
$HF daydreamlive/Video-Depth-Anything-Small config.json video_depth_anything_vits.pth --local-dir models/Video-Depth-Anything-Small

# Virtual camera (once)
sudo apt install v4l2loopback-dkms v4l-utils pulseaudio-utils
echo 'options v4l2loopback devices=1 video_nr=10 card_label="AI_Scope_Cam" exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf && sudo modprobe v4l2loopback
```

Run: `CHARACTER_REF=path/to/character.png PROMPT="describe the character" ./start_scope.sh --bridge`,
then open the slider at http://127.0.0.1:8090 and add **AI_Scope_Cam** + **AI_Character_Mic** in OBS.

Character art, test clips and model weights are not in this repo (`assets/`, `models/` are git-ignored).
Scope is licensed CC BY-NC-SA 4.0; the fork keeps that license.
