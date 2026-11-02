# Private build artifacts (Hugging Face)

Repo: **`JordanCo/live-character-artifacts`** (private, model type). Holds what we build ourselves and isn't on any
public hub, so a fresh machine on any provider can skip the kernel build (~10 min) and the first compile (~12 min).

```
joyai/wheels/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl   # SageAttention @ d1a57a5 + upstream CUDA-graph stream patch, sm_120
joyai/wheels/joyomni_ops-0.1.0-cp312-cp312-linux_x86_64.whl     # JoyAI FP8 GEMM + fused kernels (cutlass dcf215af), sm_120a
joyai/cache/NVIDIA_GeForce_RTX_5090.tar                        # torchinductor/triton/nv caches for torch 2.9.1+cu128, 840x480 both orientations
```
Built against: torch 2.9.1+cu128, Python 3.12, CUDA 12.8. Rebuild if any of those change.

- **Pull** (automatic): `setup.sh artifacts` (needs an HF token in `~/.cache/huggingface/token` on the machine).
- **Publish/refresh**: on the GPU machine after a warm start: `bash save_cache.sh && bash publish_artifacts.sh`
  (needs a write token). Add one cache tar per GPU model (e.g. `NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.tar`).
- Token: `.secrets/hf_token` locally; copy it to the machine as `/root/.cache/huggingface/token` (never print it).
