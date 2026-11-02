#!/bin/bash
# Start the JoyAI-Video-Edit streaming server on the pod (after setup_pod.sh env + weights).
# RTX 5090 / 32 GB defaults from the upstream deployment guide: SageAttention, fp16-accumulate
# FP8 GEMMs, low-VRAM layout (text encoder offloaded). Override any JOYOMNI_* env var.
set -euo pipefail
W=/workspace
J=$W/JoyAI-Video-Edit
source /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
export JOYOMNI_CACHE_ROOT=${JOYOMNI_CACHE_ROOT:-$W/cache/$GPU}
export JOYOMNI_SAGE_ATTN=${JOYOMNI_SAGE_ATTN:-1}
export JOYOMNI_FP8_FAST_ACCUM=${JOYOMNI_FP8_FAST_ACCUM:-1}
export JOYOMNI_LOW_VRAM=${JOYOMNI_LOW_VRAM:-1}
export JOYOMNI_RECORD_DIR=${JOYOMNI_RECORD_DIR:-$W/recordings}
export HF_HOME=$W/hf_cache
exec bash $J/deploy/run_server.sh "$@"
