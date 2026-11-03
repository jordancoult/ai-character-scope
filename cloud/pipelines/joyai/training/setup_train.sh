#!/bin/bash
# Training box setup for the JoyAI G-Star lip-sync LoRA (Hopper H100/H200 or B200; not the sm_120 serving box).
# Runs on the GPU machine as root, from a copy of cloud/pipelines/joyai/training at $W/train (rsync'd by you).
#   bash setup_train.sh [all|env|weights|data]
# Needs an HF token in ~/.cache/huggingface/token (for the private dataset repo).
# Layout:  $W/JoyAI-Video-Edit (upstream, pinned)   $W/ckpt/{dit,vae,MiMo-VL-7B-RL-2508}   $W/data (dataset)
#          /root/tvenv (python 3.10 venv: upstream deploy + training requirements)
set -euo pipefail
W=${JOYAI_DATA:-/workspace}
STEP=${1:-all}
UPSTREAM_COMMIT=${UPSTREAM_COMMIT:-6c4e7a2}
DATASET_REPO=${DATASET_REPO:-JordanCo/gstar-lipsync-pairs}
export HF_HOME=${HF_HOME:-$W/hf_cache} UV_CACHE_DIR=/root/uv_cache UV_LINK_MODE=copy HF_HUB_ENABLE_HF_TRANSFER=1
[ -f /root/.cache/huggingface/token ] && [ ! -f "$HF_HOME/token" ] && mkdir -p "$HF_HOME" && cp /root/.cache/huggingface/token "$HF_HOME/token"
mkdir -p "$W/logs" "$W/ckpt"
J=$W/JoyAI-Video-Edit
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=/root/.local/bin:$PATH; }
export PATH=/root/.local/bin:$PATH

hfcli() {
  [ -x /root/hfvenv/bin/hf ] || { uv venv -q /root/hfvenv --python 3.12; uv pip install -q --python /root/hfvenv/bin/python "huggingface_hub[hf_transfer]"; }
  /root/hfvenv/bin/hf "$@"
}

envsetup() {
  [ -d "$J" ] || git clone -q https://github.com/jd-opensource/JoyAI-Video-Edit.git "$J"
  git -C "$J" fetch -q origin && git -C "$J" checkout -q "$UPSTREAM_COMMIT"
  [ -x /root/tvenv/bin/python ] || uv venv -q /root/tvenv --python 3.10
  uv pip install -q --index-strategy unsafe-best-match --python /root/tvenv/bin/python -r "$J/deploy/requirements.txt" -r "$J/training/requirements.txt" \
    "huggingface_hub[hf_transfer]" scipy
  # ffmpeg for decoding/encoding review videos
  command -v ffmpeg >/dev/null || { apt-get update -qq >/dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null; }
  /root/tvenv/bin/python -c "import torch, peft, diffusers, transformers, decord; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
  echo ENV_DONE
}

weights() {
  hfcli download jdopensource/JoyAI-Video-Edit dit/joyai_video_edit_dit_1101_face++.pth vae/config.json \
    vae/diffusion_pytorch_model.safetensors --repo-type model --local-dir "$W/ckpt"
  hfcli download XiaomiMiMo/MiMo-VL-7B-RL-2508 --repo-type model --local-dir "$W/ckpt/MiMo-VL-7B-RL-2508"
  du -sh "$W"/ckpt/*
  echo WEIGHTS_DONE
}

data() {
  hfcli download "$DATASET_REPO" --repo-type dataset --local-dir "$W/data"
  ls "$W/data" | head
  echo DATA_DONE
}

case "$STEP" in
  env) envsetup ;;
  weights) weights ;;
  data) data ;;
  all) weights > "$W/logs/weights.log" 2>&1 &
       WP=$!
       envsetup
       wait $WP && tail -2 "$W/logs/weights.log"
       data ;;
  *) sed -n 2,7p "$0"; exit 1 ;;
esac
