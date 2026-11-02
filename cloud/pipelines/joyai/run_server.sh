#!/bin/bash
# Start the JoyAI-Video-Edit streaming server on the GPU machine (after setup.sh all).
# Settings are picked per GPU (RTX 5090 vs RTX PRO 6000) below; override any JOYOMNI_* env var.
set -euo pipefail
W=${JOYAI_DATA:-/workspace}   # code, weights, caches (volume or local disk)
J=${JOYAI_DIR:-$W/JoyAI-Video-Edit}   # code (the Docker image uses /opt/JoyAI-Video-Edit)
export JOYOMNI_CKPT_ROOT=${JOYOMNI_CKPT_ROOT:-$W/JoyAI-Video-Edit/deploy/deps/checkpoints}   # weights on the volume
source /root/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
# Compile cache: work on fast local disk, seeded from $W/cache/<GPU>.tar (artifacts or save_cache.sh)
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
# Per-GPU defaults (upstream DEPLOYMENT.md): GeForce runs fp32-accumulate MMA at half rate, so the RTX 5090 uses
# SageAttention + fp16-accumulate FP8; 32 GB cards need the low-VRAM layout. The RTX PRO 6000 (96 GB) is faster on
# plain cuDNN attention with full-rate accumulation and the normal layout. Any JOYOMNI_* env var overrides.
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${MEM:-0}" -gt 60000 ]; then   # RTX PRO 6000 class
  export JOYOMNI_SAGE_ATTN=${JOYOMNI_SAGE_ATTN:-0} JOYOMNI_FP8_FAST_ACCUM=${JOYOMNI_FP8_FAST_ACCUM:-0} JOYOMNI_LOW_VRAM=${JOYOMNI_LOW_VRAM:-0}
else                                 # RTX 5090 / 32 GB class
  export JOYOMNI_SAGE_ATTN=${JOYOMNI_SAGE_ATTN:-1} JOYOMNI_FP8_FAST_ACCUM=${JOYOMNI_FP8_FAST_ACCUM:-1} JOYOMNI_LOW_VRAM=${JOYOMNI_LOW_VRAM:-1}
fi
echo "[run_server] $GPU: sage=$JOYOMNI_SAGE_ATTN fast_accum=$JOYOMNI_FP8_FAST_ACCUM low_vram=$JOYOMNI_LOW_VRAM"
export JOYOMNI_RECORD_DIR=${JOYOMNI_RECORD_DIR:-$W/recordings}
export HF_HOME=$W/hf_cache
# face++ checkpoint (2026-11-01): better human identity + motion following than 0811 in our tests
export JOYOMNI_DIT_CKPT=${JOYOMNI_DIT_CKPT:-$JOYOMNI_CKPT_ROOT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth}
# our web-UI additions (memory panel: R-key reset, auto-reset setting); upstream file kept as index.html.orig
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/webui/patch_webui.sh" ] && JOYAI_DATA=$W bash "$HERE/webui/patch_webui.sh" || true
exec bash $J/deploy/run_server.sh "$@"
