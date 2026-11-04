#!/bin/bash
# LongLive 2.0 live server on a rented RTX 5090 (sm_120), from scratch: ~10 min (mostly downloads, ~20 GB).
# Run ON the GPU box as root, from a copy of this folder (tools/longlive2) at /workspace/tools/longlive2:
#   HF_TOKEN=... bash /workspace/tools/longlive2/setup_5090.sh        # token: private kernel packages (optional)
# Then:  cd /workspace && LL2_ROOT=/workspace /root/ll2/bin/python tools/longlive2/ll2_server.py --ckpt S2 --kvq
# Layout: /workspace/longlive2 (NVlabs code, pinned + skip_clean.patch), /workspace/models/LongLive-2.0-5B/
#         {S2,S4}/model_4o6.pt, Wan2.2-TI2V-5B/ (VAE, T5, tokenizer), fastvae/ (TAEW2.2); venv /root/ll2 (py3.12).
set -euo pipefail
W=/workspace
LL2_COMMIT=${LL2_COMMIT:-2426358}
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH=/root/.local/bin:$PATH UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=1
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; }
[ -x /root/hfvenv/bin/hf ] || { uv venv -q /root/hfvenv --python 3.12; uv pip install -q --python /root/hfvenv/bin/python "huggingface_hub[hf_transfer]"; }
H=/root/hfvenv/bin/hf

# --- code
[ -d $W/longlive2 ] || git clone -q https://github.com/NVlabs/LongLive.git $W/longlive2
if [ -d $W/longlive2/.git ]; then  # a copied (non-git) tree is used as is
  git -C $W/longlive2 checkout -q "$LL2_COMMIT"
  git -C $W/longlive2 apply --check "$HERE/skip_clean.patch" 2>/dev/null && git -C $W/longlive2 apply "$HERE/skip_clean.patch" || true
fi

# --- weights (downloads run in the background while the venv installs)
M=$W/models/LongLive-2.0-5B
(
  $H download Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2 model_4o6.pt --local-dir $M/S2
  $H download Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4 model_4o6.pt --local-dir $M/S4
  $H download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth config.json configuration.json google/umt5-xxl/special_tokens_map.json \
      google/umt5-xxl/spiece.model google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json --local-dir $M/Wan2.2-TI2V-5B
  $H download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth --local-dir $M/Wan2.2-TI2V-5B
  mkdir -p $M/fastvae
  curl -sL -o $M/fastvae/taew2_2.pth https://github.com/madebyollin/taehv/raw/main/taew2_2.pth
  curl -sL -o $M/fastvae/taehv.py https://raw.githubusercontent.com/madebyollin/taehv/main/taehv.py
  echo WEIGHTS_DONE
) > $W/weights.log 2>&1 &
WP=$!
mkdir -p $W/longlive2/LongLive2.0/wan_models
ln -sfn $M/Wan2.2-TI2V-5B $W/longlive2/LongLive2.0/wan_models/Wan2.2-TI2V-5B

# --- venv
[ -x /root/ll2/bin/python ] || uv venv -q /root/ll2 --python 3.12
uv pip install -q --python /root/ll2/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
  diffusers==0.31.0 "transformers>=4.49,<5" accelerate omegaconf einops easydict ftfy imageio imageio-ffmpeg av \
  opencv-python-headless scikit-image peft safetensors websockets numpy pillow huggingface_hub
SP=$(/root/ll2/bin/python -c "import site; print(site.getsitepackages()[0])")
# sm_120 kernels (flash-attn 2.8.3, SageAttention 2.2, FourOverSix NVFP4) prebuilt for py3.12 / torch 2.9.1 cu128
K=$W/kernels
$H download JordanCo/ai-livestream-artifacts --repo-type model --include "longlive2/sm120_py312_torch291cu128/*" --local-dir $K
for t in $K/longlive2/sm120_py312_torch291cu128/*.tgz; do tar -xzf "$t" -C "$SP"; done
command -v ffmpeg >/dev/null || { apt-get update -qq >/dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null; }
/root/ll2/bin/python -c "import torch, flash_attn, sageattention, fouroversix; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
wait $WP; tail -1 $W/weights.log
mkdir -p $W/tools && [ -e $W/tools/longlive2 ] || ln -sfn "$HERE" $W/tools/longlive2
echo SETUP_DONE
