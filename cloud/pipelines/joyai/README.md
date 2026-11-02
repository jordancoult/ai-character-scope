# Pipeline: JoyAI-Video-Edit

[JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit) (JD, Apache-2.0): a 16B MM-DiT streaming
video editor with a MiMo-VL-7B condition encoder and a causal VAE. It edits live frames in 8-frame chunks
from an instruction plus an optional reference image (RV2V). We use the upstream server
(`deploy/xvideo/serving/serve_joyomni_streaming.py`, FastAPI WebSocket on :8080) unchanged and drive it with our
own clients.

## Requirements
- **GPU: Blackwell sm_120 with ≥ 32 GB**: RTX 5090 (target), RTX PRO 6000, RTX PRO 4500 (works at ~half speed).
  `setup.sh` checks it. The settings are picked per GPU in `run_server.sh` (5090: SageAttention + fp16-accumulate
  FP8 + low-VRAM layout; PRO 6000: cuDNN attention, full-rate accumulate, normal layout).
- Host RAM ≥ 48 GB (the low-VRAM layout keeps the text encoder on the CPU). Disk ≥ 110 GB free under `$JOYAI_DATA`.
- NVIDIA driver for CUDA ≥ 12.8; `setup.sh` installs the CUDA 12.8 compiler with apt (Ubuntu 22.04/24.04).

## Files
| File | What |
|---|---|
| `setup.sh` | `artifacts` (private HF wheels + compile cache) · `env` (CUDA compiler, venv, kernels) · `weights` (public HF) · `all` |
| `run_server.sh` | Starts the server: per-GPU settings, face++ checkpoint, compile cache restored from `$JOYAI_DATA/cache/<GPU>.tar` |
| `save_cache.sh` | After the first warm start on a new GPU model: packs the compile cache into one tar |
| `publish_artifacts.sh` | Uploads wheels + cache tars to `JordanCo/ai-livestream-artifacts` (needs a write token) |
| `prompts/` | Working prompts (G-Star, human) |
| `experiments/` | Dated runs with FINDINGS.md (videos are gitignored) |

## Recipe (what works)
- **Reference:** a full-body character image on a plain background (`assets/char_front.png`). An upper-body
  crop turned G-Star into a realistic human torso.
- **Prompt:** describe the character explicitly (shape, face, skin, clothing, accessories), plus "Keep the pose,
  gestures and hand movements" (`prompts/gstar.txt`). The generic "replace with the character from the reference
  image" only transfers accessories. The built-in prompt enhancer (`use_pe`) made identity worse, so it's off.
- **Checkpoint `1101_face++`** (released 2026-11-01) over `0811`: clearly better human identity and motion.
- **Session options** (start message): `detection_mode: "off"`, `use_pe: false`, `kv_reset_frames: 0` (no drift
  in 2 min; the default 600 resets every 25 s without a visible jump), H.264 both ways.

## Numbers (RTX 5090, 840×480 or 480×840, 2 steps)
| | |
|---|---|
| Output fps | 25–27 (30 fps camera in); per chunk: DiT 215 ms, VAE enc 26 ms, dec 46 ms → ~26.5 fps ceiling |
| Latency on the GPU machine | 0.46 s median, 0.58 s p95 |
| Glass-to-glass, laptop → EU → laptop, smooth playout | 1.03 s median, 1.07 s p95 (2-min run, 0 drops) |
| RTX PRO 4500 (half a 5090) | ~10–11 fps, 1.15 s on the machine |

Details: `experiments/2026-11-02-joyai-first/FINDINGS.md` (prompt, ref and checkpoint tests on the PRO 4500),
`experiments/2026-11-02-joyai-5090/FINDINGS.md` (5090 speed, codecs, portrait, live bridge, stability).

## Known limits / next
- Hands stay human-shaped (pink) on G-Star; the head comes out rounder than the reference's tall cone.
- 840×480 is the 5090's real-time limit (720p needs a B200: sm_100, not supported by our kernels yet).
- First start on a new GPU model compiles ~12 min; publish its cache tar afterwards.
