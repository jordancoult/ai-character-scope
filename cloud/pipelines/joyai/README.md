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
| `serverpatch/` | Explicit patches applied at start (`patch_server.py`): no YOLO body gate outside full-body detection; **runtime LoRA** (`runtime_lora.py`, off unless `JOYAI_LORA` is set) |
| `prompts/` | Working prompts (G-Star, human) |
| `webui/` | Our additions to the upstream browser demo, injected at start (`patch_webui.sh`): a **memory panel** with a **Reset (R)** key/button (fresh temporal memory without restarting the camera), a since-reset timer and the auto-reset interval (default 0 = never, for drift tests; upstream hides it at 600 frames ≈ 25 s) |
| `experiments/` | Dated runs with FINDINGS.md (videos are gitignored) |

## G-Star detail LoRA (optional, off by default)
Adapters from `training/` (better G-Star skin/hat detail; mouth unchanged; `cloud/docs/gstar-lora-finetune.md`).
Applied at runtime next to the FP8/bf16 linears (merging would lose it):
```bash
/root/venv/bin/hf download JordanCo/joyai-gstar-lora adapters/r3_step50.pt --local-dir /workspace/lora   # private
JOYAI_LORA=/workspace/lora/adapters/r3_step50.pt JOYAI_LORA_SCALE=1.0 bash run_server.sh
```
The log shows `#####[runtime_lora] ... 320 FP8 + 0 bf16 linears hooked`. Speed cost on the 5090 (same box,
2026-11-04): base 0.319 s per 8-frame chunk (25 fps) · full LoRA 0.38 s (~21 fps) · `JOYAI_LORA_MODULES=attn_qkv,attn_proj`
0.352 s (22.7 fps). The hooks re-read each layer's bf16 input next to the memory-bound FP8 GEMM; a fused kernel would
be needed to make it free. Run the bridge with `--fps`/`--play-fps` near the server rate. About 65% of the LoRA's
change survives the bf16 add (vs ~30% merged into bf16, ~7% merged into FP8).

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
