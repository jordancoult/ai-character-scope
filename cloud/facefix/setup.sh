#!/bin/bash
# Face post-process env on the laptop (RTX 5080 Laptop, sm_120): venv + LivePortrait clone (patched) + weights.
# Run from the repo root: bash cloud/facefix/setup.sh   (~8 GB venv, ~2.3 GB weights)
set -euo pipefail
cd "$(dirname "$0")"
UV=../../bin/uv
[ -x .venv/bin/python ] || $UV venv .venv --python 3.12
$UV pip install --python .venv/bin/python torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
$UV pip install --python .venv/bin/python numpy opencv-python-headless av mediapipe==0.10.21 spandrel \
    spandrel_extra_arches pyyaml rich scipy huggingface_hub

mkdir -p third_party weights/restore
if [ ! -d third_party/LivePortrait ]; then
  git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git third_party/LivePortrait
  # channels-last conv weights (3x faster conv3d on sm_120) need reshape instead of view in these modules
  sed -i 's/\.view(/.reshape(/g' third_party/LivePortrait/src/modules/{dense_motion,warping_network,appearance_feature_extractor}.py
fi
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("KwaiVGI/LivePortrait", local_dir="weights/liveportrait", allow_patterns=[
    "liveportrait/base_models/*", "liveportrait/retargeting_models/*",
    "liveportrait_animals/base_models_v1.1/*"])
EOF
dl() { [ -s "$2" ] || curl -sSL -o "$2" "$1"; }
dl https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task weights/hand_landmarker.task
FL=../client/models/face_landmarker.task
mkdir -p ../client/models
dl https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task $FL
# optional face restoration (GFPGAN: Apache-2.0; CodeFormer: S-Lab NON-COMMERCIAL; RestoreFormer: Apache-2.0)
if [ "${RESTORE:-0}" = 1 ]; then
  dl https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth weights/restore/GFPGANv1.4.pth
  dl https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth weights/restore/RestoreFormer.pth
  dl https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth weights/restore/codeformer.pth
fi
echo "facefix setup done"
