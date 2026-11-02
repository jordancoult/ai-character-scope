#!/bin/bash
# JoyAI-Video-Edit on a RunPod pod (image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, RTX 5090 /
# any sm_120 Blackwell). Layout:
#   /workspace (network volume, persistent): repo, weights (~51 GB), built kernel wheels, compile caches
#   /root/venv (container disk, fast): Python env, rebuilt per pod in a few minutes from cached wheels
# Usage: bash setup_pod.sh env|weights|all      (idempotent)
set -euo pipefail
W=/workspace
STEP=${1:-all}
export HF_HOME=$W/hf_cache UV_CACHE_DIR=/root/uv_cache UV_LINK_MODE=copy
mkdir -p $W/logs $W/wheels
[ -d $W/JoyAI-Video-Edit ] || git clone -q https://github.com/jd-opensource/JoyAI-Video-Edit.git $W/JoyAI-Video-Edit
J=$W/JoyAI-Video-Edit
CK=$J/deploy/deps/checkpoints
mkdir -p $CK

weights() {
  [ -x /root/venv/bin/python ] || uv venv -q /root/venv --python 3.12
  uv pip install -q --python /root/venv/bin/python "huggingface_hub[hf_transfer]"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  DIT=${JOYAI_DIT:-joyai_video_edit_dit_1101_face++.pth}
  /root/venv/bin/hf download jdopensource/JoyAI-Video-Edit "dit/$DIT" vae/config.json \
    vae/diffusion_pytorch_model.safetensors --repo-type model --local-dir $CK/JoyAI-Video-Edit
  /root/venv/bin/hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 --repo-type model --local-dir $CK/MiMo-VL-7B-RL-2508
  [ -s $CK/face_detection_yunet_2023mar.onnx ] || curl -sL -o $CK/face_detection_yunet_2023mar.onnx \
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  du -sh $CK/*
  echo WEIGHTS_DONE
}

envsetup() {
  # CUDA 12.8 compiler + headers (Blackwell needs nvcc >= 12.8); the image ships the runtime only
  if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
    if ! apt-cache policy cuda-nvcc-12-8 2>/dev/null | grep -q Candidate; then
      curl -sL -o /tmp/kr.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
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
  uv pip install -q -r $J/deploy/requirements.txt pip setuptools wheel "huggingface_hub[hf_transfer]"
  export TORCH_CUDA_ARCH_LIST=12.0
  # kernel wheels: build once, cache on the volume
  if ! ls $W/wheels/sageattention-*.whl >/dev/null 2>&1; then
    [ -d $W/build/SageAttention ] || git clone -q https://github.com/thu-ml/SageAttention.git $W/build/SageAttention
    rm -rf /root/build && mkdir -p /root/build && cp -r $W/build/SageAttention /root/build/
    ( cd /root/build/SageAttention && git checkout -q d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 && \
      git apply $J/deploy/sageattention-cudagraph-stream.patch && \
      EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=$(nproc) python setup.py bdist_wheel >$W/logs/sage_build.log 2>&1 && \
      cp dist/*.whl $W/wheels/ )
  fi
  if ! ls $W/wheels/joyomni_ops-*.whl >/dev/null 2>&1; then
    [ -d $W/build/cutlass ] || git clone -q https://github.com/NVIDIA/cutlass.git $W/build/cutlass
    git -C $W/build/cutlass checkout -q dcf215af
    mkdir -p /root/build && cp -r $J/deploy/joyomni_ops /root/build/
    ( cd /root/build && JOYOMNI_OPS_CUDA_ARCHS=120a JOYOMNI_OPS_CUTLASS_DIR=$W/build/cutlass MAX_JOBS=$(nproc) \
      python -m pip wheel --no-build-isolation --no-deps -w $W/wheels ./joyomni_ops >$W/logs/ops_build.log 2>&1 )
  fi
  uv pip install -q --no-deps $W/wheels/sageattention-*.whl $W/wheels/joyomni_ops-*.whl
  python -c "import torch, sageattention, joyomni_ops; print('torch', torch.__version__, '| sage ok | joyomni_ops fp8', joyomni_ops.has_fp8())"
  echo ENV_DONE
}

case $STEP in
  weights) weights ;;
  env) envsetup ;;
  all) envsetup; weights ;;
esac
