#!/bin/bash
# JoyAI-Video-Edit: provider-independent setup on any Linux GPU machine (RTX 5090 / sm_120 Blackwell tested).
# Needs: Ubuntu 22.04/24.04 with an NVIDIA driver for CUDA >= 12.8, Python 3.12 + uv (the runpod/pytorch images
# have both; otherwise `pip install uv`), ~110 GB free under $JOYAI_DATA, root (apt) for the CUDA compiler.
#
#   JOYAI_DATA       where code, weights and caches live (default /workspace: a persistent volume if the provider
#                    gives one, otherwise just local disk)
#   ARTIFACTS_REPO   private Hugging Face repo with our prebuilt kernel wheels + warm compile caches
#                    (default JordanCo/ai-livestream-artifacts); needs an HF token in ~/.cache/huggingface/token
#   JOYAI_DIT        DiT checkpoint file (default joyai_video_edit_dit_1101_face++.pth)
#
# Usage: bash setup.sh env|weights|artifacts|all      (idempotent; `all` = artifacts, env, weights)
set -euo pipefail

# --- hardware requirement: consumer/workstation Blackwell (sm_120) with >= 32 GB, e.g. RTX 5090 or RTX PRO 6000 ---
# (kernel wheels and compile caches are built for sm_120; B200/B300 are sm_100 and need their own build + FA4)
if [ "${JOYAI_SKIP_GPU_CHECK:-0}" != 1 ]; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
  NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  if [ "$CAP" != "12.0" ] || [ "${MEM:-0}" -lt 30000 ]; then
    echo "UNSUPPORTED GPU: $NAME (compute $CAP, ${MEM} MiB). Needs Blackwell sm_120 with >= 32 GB:"
    echo "  RTX 5090 (target), RTX PRO 6000, RTX PRO 4500 (half speed). Set JOYAI_SKIP_GPU_CHECK=1 to override."
    exit 3
  fi
  echo "GPU ok: $NAME (sm_${CAP/./}, ${MEM} MiB)"
fi

W=${JOYAI_DATA:-/workspace}
STEP=${1:-all}
ARTIFACTS_REPO=${ARTIFACTS_REPO:-JordanCo/ai-livestream-artifacts}
export HF_HOME=${HF_HOME:-$W/hf_cache} UV_CACHE_DIR=/root/uv_cache UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=1
# keep the user's HF login visible even though HF_HOME moved
[ -f /root/.cache/huggingface/token ] && [ ! -f "$HF_HOME/token" ] && mkdir -p "$HF_HOME" && cp /root/.cache/huggingface/token "$HF_HOME/token"
mkdir -p "$W/logs" "$W/wheels" "$W/cache"
[ -d "$W/JoyAI-Video-Edit" ] || git clone -q https://github.com/jd-opensource/JoyAI-Video-Edit.git "$W/JoyAI-Video-Edit"
J=$W/JoyAI-Video-Edit
CK=$J/deploy/deps/checkpoints
mkdir -p "$CK"

hfcli() {
  [ -x /root/venv/bin/hf ] || { [ -x /root/venv/bin/python ] || uv venv -q /root/venv --python 3.12; uv pip install -q --python /root/venv/bin/python "huggingface_hub[hf_transfer]"; }
  /root/venv/bin/hf "$@"
}

artifacts() {
  # prebuilt kernel wheels + compile-cache tars (per GPU model) from our private repo; skipped if unreachable
  if hfcli download "$ARTIFACTS_REPO" --repo-type model --include "joyai/*" --local-dir "$W/artifacts" >"$W/logs/artifacts.log" 2>&1; then
    cp -n "$W"/artifacts/joyai/wheels/*.whl "$W/wheels/" 2>/dev/null || true
    for t in "$W"/artifacts/joyai/cache/*.tar; do [ -f "$t" ] && cp -n "$t" "$W/cache/"; done
    ls "$W/wheels" "$W/cache"
    echo ARTIFACTS_DONE
  else
    echo "ARTIFACTS_SKIPPED (no token or repo; kernels will be built from source, first start compiles ~12 min)"
  fi
}

weights() {
  DIT=${JOYAI_DIT:-joyai_video_edit_dit_1101_face++.pth}
  hfcli download jdopensource/JoyAI-Video-Edit "dit/$DIT" vae/config.json vae/diffusion_pytorch_model.safetensors \
    --repo-type model --local-dir "$CK/JoyAI-Video-Edit"
  hfcli download XiaomiMiMo/MiMo-VL-7B-RL-2508 --repo-type model --local-dir "$CK/MiMo-VL-7B-RL-2508"
  [ -s "$CK/face_detection_yunet_2023mar.onnx" ] || curl -sL -o "$CK/face_detection_yunet_2023mar.onnx" \
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  du -sh "$CK"/*
  echo WEIGHTS_DONE
}

envsetup() {
  # CUDA 12.8 compiler + headers (Blackwell needs nvcc >= 12.8); most images ship the runtime only
  if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
    if ! apt-cache policy cuda-nvcc-12-8 2>/dev/null | grep -q Candidate; then
      . /etc/os-release; REPO=ubuntu${VERSION_ID//./}
      curl -sL -o /tmp/kr.deb "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb"
      dpkg -i /tmp/kr.deb >/dev/null
    fi
    apt-get update -qq >/dev/null
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8 \
      libcublas-dev-12-8 libcusparse-dev-12-8 libcusolver-dev-12-8 cuda-profiler-api-12-8 cuda-nvtx-12-8 \
      libcurand-dev-12-8 >/dev/null
  fi
  export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
  nvcc --version | grep release
  [ -x /root/venv/bin/python ] || uv venv -q /root/venv --python 3.12
  source /root/venv/bin/activate
  uv pip install -q -r "$J/deploy/requirements.txt" pip setuptools wheel "huggingface_hub[hf_transfer]"
  export TORCH_CUDA_ARCH_LIST=12.0
  # kernel wheels: from artifacts if present, else build once (cached in $W/wheels)
  if ! ls "$W"/wheels/sageattention-*.whl >/dev/null 2>&1; then
    [ -d "$W/build/SageAttention" ] || git clone -q https://github.com/thu-ml/SageAttention.git "$W/build/SageAttention"
    rm -rf /root/build && mkdir -p /root/build && cp -r "$W/build/SageAttention" /root/build/
    ( cd /root/build/SageAttention && git checkout -q d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 && \
      git apply "$J/deploy/sageattention-cudagraph-stream.patch" && \
      EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=$(nproc) python setup.py bdist_wheel >"$W/logs/sage_build.log" 2>&1 && \
      cp dist/*.whl "$W/wheels/" )
  fi
  if ! ls "$W"/wheels/joyomni_ops-*.whl >/dev/null 2>&1; then
    [ -d "$W/build/cutlass" ] || git clone -q https://github.com/NVIDIA/cutlass.git "$W/build/cutlass"
    git -C "$W/build/cutlass" checkout -q dcf215af
    mkdir -p /root/build && cp -r "$J/deploy/joyomni_ops" /root/build/
    ( cd /root/build && JOYOMNI_OPS_CUDA_ARCHS=120a JOYOMNI_OPS_CUTLASS_DIR="$W/build/cutlass" MAX_JOBS=$(nproc) \
      python -m pip wheel --no-build-isolation --no-deps -w "$W/wheels" ./joyomni_ops >"$W/logs/ops_build.log" 2>&1 )
  fi
  uv pip install -q --no-deps "$W"/wheels/sageattention-*.whl "$W"/wheels/joyomni_ops-*.whl
  python -c "import torch, sageattention, joyomni_ops; print('torch', torch.__version__, '| sage ok | joyomni_ops fp8', joyomni_ops.has_fp8())"
  echo ENV_DONE
}

case $STEP in
  artifacts) artifacts ;;
  weights) weights ;;
  env) envsetup ;;
  all) artifacts; envsetup; weights ;;
  *) echo "usage: setup.sh env|weights|artifacts|all"; exit 2 ;;
esac
