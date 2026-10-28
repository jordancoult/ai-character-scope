#!/bin/bash
# Launch the standalone StreamDiffusionV2 demo web UI on http://localhost:7860 (512x512, TAEHV decode).
# Prompt-only restyling (no character reference image) - use start_scope.sh for VACE characters.
#
# Env overrides: STEP=1|2 (1 = faster, default), PORT=7860
set -u
W="$HOME/personal/ai-character-scope"
source "$W/env.sh"
cd "$W/StreamDiffusionV2"

export PATH="$PWD/.venv/bin:$PATH"
export USE_TAEHV=1
# Local patch in models/wan/wan_wrapper.py: keep the 11 GB T5 encoder in CPU RAM and stream it to
# the GPU per block, otherwise the demo does not fit in 16 GB.
export SDV2_TEXT_ENCODER_OFFLOAD=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export GPU_IDS=0
export STEP="${STEP:-1}"
export PORT="${PORT:-7860}"
export HOST=127.0.0.1

if ! pgrep -f "tools/thermal_watchdog.sh" >/dev/null; then
  nohup "$W/tools/thermal_watchdog.sh" >/dev/null 2>&1 &
  echo "[sdv2] thermal watchdog started (log: .tmp/thermal.log)"
fi

echo "[sdv2] GPU: $(nvidia-smi --query-gpu=name,temperature.gpu --format=csv,noheader); steps=$STEP"
echo "[sdv2] UI will be at http://localhost:$PORT once the model has loaded (~1-2 min)."
exec taskset -c 6-13 bash demo/start.sh
