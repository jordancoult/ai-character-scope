# ai-character-scope: realtime AI video pipelines for livestreaming

Webcam in → AI-transformed video out → OBS virtual camera (`/dev/video10`) → TikTok etc. Several pipelines share
one bar: **25–30 fps** and **< 2 s** input → output latency. The first pipeline is **character swap**: the person
on the webcam becomes a character taken from one reference image. Pipelines run locally on a laptop GPU (Daydream
Scope fork) or on a rented Blackwell GPU (`cloud/`).

## Pipelines
| Where | Pipeline | Task | Speed | Docs |
|---|---|---|---|---|
| Local: RTX 5080 Laptop 16 GB, Scope fork | **StreamDiffusionV2** + VACE | character swap from a reference image (or prompt only), 576×320 portrait upscaled to 720×1280 | **~29 fps live** (input-limited; 38.7 fps model-only) | [README_QUICKSTART.md](README_QUICKSTART.md) |
| Local | LongLive / Causal Forcing / Reward Forcing / MemFlow | camera restyle and prompt-driven "dream" worlds, 640×352 | 17–26 fps (depth on/off, 1–2 steps) | [Model status](README_QUICKSTART.md#model-status-2026-11-01-what-runs-locally-and-how-fast) |
| Local | LongLive 2.0 (5B NVFP4) | prompt-only world generation (no camera), 1024×576 | 13–15 fps in Scope (~15 standalone) | [tools/longlive2/README.md](tools/longlive2/README.md) |
| Cloud: RTX 5090 | **JoyAI-Video-Edit** | character swap (also instruction edits), 840×480 / 480×840 | **25–27 fps**, 0.46 s on the GPU machine, ~1 s glass-to-glass | [cloud/pipelines/joyai/README.md](cloud/pipelines/joyai/README.md) |

Local reference identity is weak in the streaming 1.3B models; JoyAI on a 5090 is the one that meets the goal.

## Project map (start here)
| Read | For |
|---|---|
| [AGENTS.md](AGENTS.md) | The whole picture in one page: goal, local vs cloud, rules, secrets, how to pick up the work |
| [README_QUICKSTART.md](README_QUICKSTART.md) | **Local** (laptop GPU, Scope): full setup, tested hardware, presets, model status, benchmarks, every patch |
| [START_HERE.md](START_HERE.md) | Local: just start it (webcam → character → OBS commands) |
| [cloud/README.md](cloud/README.md) | **Cloud** (rented RTX 5090): pipeline index, quickstart, live bridge to OBS |
| [cloud/pipelines/joyai/README.md](cloud/pipelines/joyai/README.md) | JoyAI-Video-Edit: recipe, numbers, limits |
| [cloud/infra/providers.md](cloud/infra/providers.md) · [cloud/infra/artifacts.md](cloud/infra/artifacts.md) | How to rent a GPU machine · the private HF artifacts repo |
| [tools/longlive2/README.md](tools/longlive2/README.md) | LongLive 2.0 standalone setup |
| [experiments/](experiments/) · [cloud/pipelines/joyai/experiments/](cloud/pipelines/joyai/experiments/) | Dated test runs with FINDINGS.md: local / cloud |
| [cloud/docs/lessons.md](cloud/docs/lessons.md) | Mistakes that cost time once |

## Components and where they come from
Everything outside git (weights, clones, envs, secrets) is gitignored; paths are relative to the repo root.

**Code**
| Component | Source | Goes to | Used by | How to get it |
|---|---|---|---|---|
| This repo | `github.com/jordancoult/ai-character-scope`, default branch **`main`** (pre-merge tip tagged `main-pre-merge-2026-11-02`) | repo root (any folder name) | everything | `git clone git@github.com:jordancoult/ai-character-scope.git` |
| Scope fork | `github.com/jordancoult/scope`, branch **`blackwell-realtime`** (pre-merge tip tagged `blackwell-realtime-pre-merge-2026-11-02`) | `scope/`, remote **`fork`**; `origin` = upstream `daydreamlive/scope` (never push there) | all local pipelines | [Local setup](README_QUICKSTART.md#full-local-setup) |
| StreamDiffusionV2 fork | `github.com/jordancoult/StreamDiffusionV2`, branch `blackwell-realtime` (upstream `chenfengxu714/StreamDiffusionV2`) | `StreamDiffusionV2/` (own `.venv`; `ckpts/`, `wan_models/` symlink into `models/`) | standalone demo `start_sdv2_demo.sh` | [Local setup](README_QUICKSTART.md#full-local-setup) |
| NVlabs LongLive | `github.com/NVlabs/LongLive` + `tools/longlive2/skip_clean.patch`; FourOverSix + sentencepiece in `longlive2/.deps` | `longlive2/` | LongLive 2.0 (standalone and Scope `longlive2`) | [tools/longlive2/README.md](tools/longlive2/README.md) |
| JoyAI-Video-Edit | `github.com/jd-opensource/JoyAI-Video-Edit` (upstream server, unchanged) | GPU machine `/workspace/JoyAI-Video-Edit` | cloud `joyai` | cloned by `cloud/pipelines/joyai/setup.sh` |
| SageAttention @ `d1a57a5` · cutlass @ `dcf215af` | `github.com/thu-ml/SageAttention` · `github.com/NVIDIA/cutlass` | GPU machine, built into wheels (sm_120) | cloud `joyai` | `setup.sh env` (or prebuilt wheels from the artifacts repo) |
| Cloud client venv | PyPI: `websockets opencv-python-headless numpy av vastai huggingface_hub` | `cloud/.venv` | `cloud/client/*`, `cloud/infra/vast.sh` | `bin/uv venv cloud/.venv --python 3.12 && bin/uv pip install --python cloud/.venv/bin/python websockets opencv-python-headless numpy av vastai huggingface_hub` |

**Local weights** (`models/`, ~60 GB+; Scope also auto-downloads its `daydreamlive/*` artifacts when a pipeline loads)
| Folder | Source (Hugging Face unless noted) | Used by |
|---|---|---|
| `Wan2.1-T2V-1.3B/` | `daydreamlive/Wan2.1-T2V-1.3B` (`config.json`, `Wan2.1_VAE.pth`, `google/`) | all Scope Wan pipelines |
| `WanVideo_comfy/` | `daydreamlive/WanVideo_comfy` (umt5-xxl fp8 encoder, VACE 1.3B module) | all Scope Wan pipelines, VACE |
| `Autoencoders/` (+ `taew2_1.pth` hardlink at `models/`) | `daydreamlive/Autoencoders` (`taew2_1`, `lighttaew2_1`, `lightvaew2_1`) | Scope VAEs; SDv2 demo TAEHV |
| `StreamDiffusionV2/` (+ `wan_causal_dmd_v2v` symlink) | `daydreamlive/StreamDiffusionV2` (`wan_causal_dmd_v2v/model.pt`) | StreamDiffusionV2 (Scope and demo) |
| `Video-Depth-Anything-Small/` | `daydreamlive/Video-Depth-Anything-Small` | depth presets (20/15), depth → VACE |
| `LongLive-1.3B/` | `daydreamlive/LongLive-1.3B` (`models/longlive_base.pt`, `models/lora.pt`) | `MODEL=longlive` |
| `Reward-Forcing-T2V-1.3B/` | `daydreamlive/Reward-Forcing-T2V-1.3B` (`rewardforcing.pt`) | `MODEL=reward-forcing` |
| `MemFlow/` | `daydreamlive/MemFlow` (`base.pt`, `lora.pt`) | MemFlow pipeline |
| `Causal-Forcing/` | `zhuhz22/Causal-Forcing` (`chunkwise/causal_forcing.pt`, `causal-forcing++/framewise-2step.pt`) | `SCOPE_RF_GENERATOR` on reward-forcing |
| `OPSD-V/` | `MeiGen-AI/OPSD-V` (`checkpoints/opsdv_longlive_lora.pt`) | `SCOPE_LONGLIVE_LORA` |
| `LongLive-2.0-5B/` | `Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2` (`model_4o6.pt`); `Wan2.2-TI2V-5B/` from `Wan-AI/Wan2.2-TI2V-5B` (VAE, tokenizer); `fastvae/` from GitHub `madebyollin/taehv` (`taew2_2.pth`, `taehv.py`); `Matrix-Game-3.0/` from `Skywork/Matrix-Game-3.0` (`MG-LightVAE_v2.pth`, decoder test) | LongLive 2.0 |
| `sdv2/Wan2.1-T2V-1.3B/` | `Wan-AI/Wan2.1-T2V-1.3B` (full: DiT, bf16 T5, VAE, tokenizer) | SDv2 demo (`StreamDiffusionV2/wan_models`); its T5 is symlinked into LongLive 2.0 |
| `lora/` | `Alissonerdx/UltraWanComfy` (`ultrawan/`), `AlekseyCalvin/HSToricColorT2VWan1.3B_v2_rank128LoRA_bySilverAgePoets` (`hstoric/`), `wookiekim/Wan2.1-T2V-1.3B-SOLACE` (`solace/`) | optional style LoRAs (`--lora path:scale`) |
| `upscalers/` | GitHub release `xinntao/Real-ESRGAN` v0.2.5.0 (`realesr-animevideov3.pth`, `realesr-general-x4v3.pth`) | bridge `--upscale` option |

Download commands: [README_QUICKSTART.md → Models](README_QUICKSTART.md#models).

**Cloud weights and artifacts** (pulled on the GPU machine by `setup.sh`)
| Component | Source | Used by |
|---|---|---|
| JoyAI DiT (`dit/1101_face++`) + VAE | HF `jdopensource/JoyAI-Video-Edit` | `joyai` |
| Condition encoder | HF `XiaomiMiMo/MiMo-VL-7B-RL-2508` | `joyai` |
| Prebuilt kernel wheels + compile cache | **private** HF `JordanCo/ai-livestream-artifacts` ([artifacts.md](cloud/infra/artifacts.md)) | `joyai` (`setup.sh artifacts`, skips ~22 min) |

**Tools and system packages**
| Tool | Source | Goes to | Used by |
|---|---|---|---|
| uv / uvx | `astral.sh/uv/install.sh` | `bin/` | all Python envs |
| cloudflared | GitHub `cloudflare/cloudflared` releases (downloaded by `tools/remote_tunnel.sh`) | `bin/` | remote access to the local UI |
| v4l2loopback | apt `v4l2loopback-dkms v4l-utils` | kernel module, `/dev/video10` "AI_Scope_Cam" | virtual camera for OBS (local and cloud bridges) |
| ffmpeg, PulseAudio utils | apt `ffmpeg pulseaudio-utils` | system | previews, review videos, delayed mic |
| Node.js + npm | system (tested Node 22) | system | building the Scope frontend |
| OBS Studio | apt `obs-studio` | system | streaming |

**Secrets** (`.secrets/`, gitignored, never print): `runpod_ed25519` / `runpod_ed25519.pub` (SSH key for GPU
machines), `hf_token` (HF token; read for the private artifacts, write to publish), `vast_api_key` (Vast.ai).

## Quick start
**Local** (Linux, NVIDIA driver ≥ 570): full steps in [README_QUICKSTART.md → Full local setup](README_QUICKSTART.md#full-local-setup).
```bash
git clone git@github.com:jordancoult/ai-character-scope.git && cd ai-character-scope
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$PWD/bin UV_NO_MODIFY_PATH=1 sh && source env.sh
git clone -o fork -b blackwell-realtime git@github.com:jordancoult/scope.git scope
(cd scope && uv sync && cd frontend && npm ci && npm run build)
# models: README_QUICKSTART.md#models · virtual camera: README_QUICKSTART.md#full-local-setup
CHARACTER_REF=assets/char_front.png PROMPT="describe the character" ./start_scope.sh --bridge   # slider: http://127.0.0.1:8090
```

**Cloud** ([cloud/README.md](cloud/README.md#quickstart-any-provider-1220-min-to-first-frame)): rent an RTX 5090
(sm_120, ≥ 32 GB, ≥ 120 GB disk, SSH + port 8080), put its `HOST PORT` in `cloud/infra/.pod`, copy
`cloud/pipelines/joyai/*.sh` and the HF token over, run `setup.sh all` then `run_server.sh`, tunnel port 8080 and
run `cloud/client/live_bridge.py` into `/dev/video10`. Terminate the machine when done.

## Repos, storage, license
- Git: this repo (everything we write) + the two forks above. Never push to `scope/`'s `origin` (upstream Daydream).
- Outside git: local weights in `models/`, clones in `scope/`, `StreamDiffusionV2/`, `longlive2/`, media in
  `assets/` (character art, test clips; not distributed), cloud build artifacts in the private HF repo
  `JordanCo/ai-livestream-artifacts`. All caches stay inside the repo folder (`env.sh`).
- Licenses: Scope is CC BY-NC-SA 4.0 (the fork keeps it); JoyAI-Video-Edit is Apache-2.0. Model weights carry
  their own licenses (see each HF repo).
