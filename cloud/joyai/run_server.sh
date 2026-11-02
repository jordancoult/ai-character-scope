#!/bin/bash
# Start the JoyAI-Video-Edit streaming server on the pod (after setup_pod.sh env + weights).
# RTX 5090 / 32 GB defaults from the upstream deployment guide: SageAttention, fp16-accumulate
# FP8 GEMMs, low-VRAM layout (text encoder offloaded). Override any JOYOMNI_* env var.
set -euo pipefail
W=/workspace
J=${JOYAI_DIR:-$W/JoyAI-Video-Edit}   # code (the Docker image uses /opt/JoyAI-Video-Edit)
export JOYOMNI_CKPT_ROOT=${JOYOMNI_CKPT_ROOT:-$W/JoyAI-Video-Edit/deploy/deps/checkpoints}   # weights on the volume
source /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
# Compile cache: work on fast local disk, seeded from the volume; save_cache.sh copies it back once warm
# (compiling straight onto the network volume took ~60 min on the first start).
VC=$W/cache/$GPU
LC=/root/jcache/$GPU
if [ -z "${JOYOMNI_CACHE_ROOT:-}" ]; then
  mkdir -p "$LC"
  if [ ! -e "$LC/.seeded" ]; then
    if [ -f "$VC.tar" ]; then tar -xf "$VC.tar" -C "$LC"        # fast path: one sequential read
    elif [ -d "$VC" ]; then cp -a "$VC/." "$LC/"; fi            # legacy directory cache (slow)
    touch "$LC/.seeded"
  fi
  export JOYOMNI_CACHE_ROOT=$LC
fi
export JOYOMNI_SAGE_ATTN=${JOYOMNI_SAGE_ATTN:-1}
export JOYOMNI_FP8_FAST_ACCUM=${JOYOMNI_FP8_FAST_ACCUM:-1}
export JOYOMNI_LOW_VRAM=${JOYOMNI_LOW_VRAM:-1}
export JOYOMNI_RECORD_DIR=${JOYOMNI_RECORD_DIR:-$W/recordings}
export HF_HOME=$W/hf_cache
# face++ checkpoint (2026-11-01): better human identity + motion following than 0811 in our tests
export JOYOMNI_DIT_CKPT=${JOYOMNI_DIT_CKPT:-$JOYOMNI_CKPT_ROOT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth}
exec bash $J/deploy/run_server.sh "$@"
