#!/bin/bash
# Start the Daydream Scope server (patched StreamDiffusionV2 + TAE) for live character streaming.
#
#   ./start_scope.sh                 # server + web UI only (use the browser, webcam via WebRTC)
#   ./start_scope.sh --bridge        # also run the headless bridge: /dev/video0 -> Scope -> /dev/video10
#
# Env overrides (bridge mode):
#   CHARACTER_REF="a.png [b.png]"         reference image(s) for VACE (omit for prompt-only)
#   PROMPT="..."                          text prompt describing the character
#   PRESET=30|24|20|15                    starting FPS preset (change live on the slider page)
#   MODEL=streamdiffusionv2|longlive      model (default streamdiffusionv2; longlive = experimental, no depth presets)
#   NOISE_MODE=fixed|random|auto          per-chunk noise pattern (default fixed; auto = preset choice; also on the page)
#   CAMERA=/dev/video0  CAMERA_SIZE=1280x720
#   AUDIO_DELAY_MS=350                    delay applied to the AI_Character_Mic virtual mic
#   NO_AUDIO=1                            skip creating the virtual mic
set -u
W="$HOME/personal/ai-character-scope"
source "$W/env.sh"
cd "$W"

BRIDGE=0
[ "${1:-}" = "--bridge" ] && BRIDGE=1
PORT="${PORT:-8000}"
AUDIO_DELAY_MS="${AUDIO_DELAY_MS:-350}"
LOG_DIR="$W/.tmp"; mkdir -p "$LOG_DIR"
PIDS=(); AUDIO_MODULES=()

cleanup() {
  echo; echo "[scope] shutting down..."
  for ((i=${#PIDS[@]}-1; i>=0; i--)); do kill "${PIDS[$i]}" 2>/dev/null; sleep 1; done  # bridge first, then server
  for ((i=${#AUDIO_MODULES[@]}-1; i>=0; i--)); do pactl unload-module "${AUDIO_MODULES[$i]}" 2>/dev/null; done
  wait 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# --- GPU / power sanity checks (ROG Zephyrus) ---
echo "[scope] GPU: $(nvidia-smi --query-gpu=name,memory.total,temperature.gpu,power.limit --format=csv,noheader)"
command -v supergfxctl >/dev/null && echo "[scope] GPU mode: $(supergfxctl -g 2>/dev/null)"
if command -v asusctl >/dev/null; then
  PROFILE=$(asusctl profile get 2>/dev/null | awk -F': ' '/Active profile/{print $2}')
  echo "[scope] ASUS profile: $PROFILE"
  [ "$PROFILE" != "Performance" ] && echo "[scope] WARNING: not in Performance profile (on battery?). Expect lower FPS."
fi
grep -q 1 /sys/class/power_supply/A*/online 2>/dev/null || echo "[scope] WARNING: running on battery."

# --- Thermal watchdog, live profile: kills GPU jobs if GPU>=90C (past its 87C target, in the
# driver slowdown zone) or CPU>=103C, sustained ~30s ---
# pidfile check (pgrep -f could match any shell whose command line mentions the script)
if ! kill -0 "$(cat "$W/.tmp/thermal_watchdog.pid" 2>/dev/null)" 2>/dev/null; then
  nohup "$W/tools/thermal_watchdog.sh" 90 103 10 >/dev/null 2>&1 &
  echo "[scope] thermal watchdog started (log: .tmp/thermal.log)"
fi

# --- Virtual camera ---
if [ ! -e /dev/video10 ]; then
  echo "[scope] /dev/video10 missing; loading v4l2loopback (needs sudo)..."
  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="AI_Scope_Cam" exclusive_caps=1 \
    || echo "[scope] WARNING: could not load v4l2loopback; bridge output to /dev/video10 will fail."
fi

# --- Latency-matched virtual microphone: real mic -> delay -> AI_Character_Mic ---
if [ "${NO_AUDIO:-0}" != 1 ] && command -v pactl >/dev/null; then
  if pactl list short sources | grep -q "AI_Character_Mic"; then
    echo "[scope] AI_Character_Mic already exists (left over?); not recreating."
  else
    AUDIO_MODULES+=("$(pactl load-module module-null-sink sink_name=ai_character_mic_sink sink_properties=device.description=AI_Character_Mic_Sink)")
    AUDIO_MODULES+=("$(pactl load-module module-loopback source=@DEFAULT_SOURCE@ sink=ai_character_mic_sink latency_msec="$AUDIO_DELAY_MS" source_dont_move=true sink_dont_move=true)")
    AUDIO_MODULES+=("$(pactl load-module module-remap-source master=ai_character_mic_sink.monitor source_name=AI_Character_Mic source_properties=device.description=AI_Character_Mic)")
    echo "[scope] virtual mic AI_Character_Mic = default mic delayed ${AUDIO_DELAY_MS}ms"
  fi
fi

# --- Blackwell speedups (local Scope patches): fp8 row-wise GEMMs + compiled blocks ---
export SCOPE_FP8_ROWWISE="${SCOPE_FP8_ROWWISE:-1}"
export SCOPE_COMPILE_BLOCKS="${SCOPE_COMPILE_BLOCKS:-1}"
# Fold the clean-KV refresh into the next chunk's forward (one DiT pass per chunk instead of two)
export SCOPE_FUSED_KV="${SCOPE_FUSED_KV:-1}"
# LongLive shares the denoise/clean-KV blocks but its model has no fused-prefix support
[ "${MODEL:-streamdiffusionv2}" = longlive ] && export SCOPE_FUSED_KV=0
export SCOPE_COMPILE_TAE="${SCOPE_COMPILE_TAE:-1}"
export SCOPE_PRELOAD_DEPTH="${SCOPE_PRELOAD_DEPTH:-1}"
# Latency: keep only a 3-frame cushion in the output queue (1.34 s -> 0.46 s, no added stutter)
export SCOPE_OUTPUT_TARGET_FRAMES="${SCOPE_OUTPUT_TARGET_FRAMES:-3}"
export SCOPE_SINK_QUEUE_MAX="${SCOPE_SINK_QUEUE_MAX:-12}"
# Input side: keep at most 2 chunks of camera frames, dropping the oldest when the model is slower
export SCOPE_INPUT_QUEUE_MAX="${SCOPE_INPUT_QUEUE_MAX:-8}"
# SCOPE_KV_FRAMES=3 is ~10% faster but follows the camera framing less (see README); stock 6 kept.

# --- Refuse to start next to a stray server (the bridge would silently attach to it) ---
if ss -ltn | grep -q ":$PORT "; then
  echo "[scope] ERROR: port $PORT is already in use (another Scope server still running?)."
  echo "[scope]        Stop it first:  fuser -k -TERM $PORT/tcp"
  exit 1
fi

# --- Scope server (pinned to E-cores 6-13 to keep CPU package temps down) ---
taskset -c 6-13 scope/.venv/bin/daydream-scope --host 127.0.0.1 --port "$PORT" --no-browser \
  > "$LOG_DIR/scope-server.log" 2>&1 &
PIDS+=($!)
echo -n "[scope] starting server"
for _ in $(seq 1 90); do
  curl -sf "localhost:$PORT/api/v1/hardware/info" >/dev/null && break
  kill -0 "${PIDS[0]}" 2>/dev/null || { echo; echo "[scope] server died; see .tmp/scope-server.log"; exit 1; }
  echo -n "."; sleep 2
done
echo
echo "[scope] Web UI:  http://localhost:$PORT"
echo "[scope] Recommended UI settings: StreamDiffusionV2, VAE=tae, 576x320 (portrait), 1 denoising step."

if [ "$BRIDGE" = 1 ]; then
  REF_ARGS=(); [ -n "${CHARACTER_REF:-}" ] && read -r -a REFS <<<"$CHARACTER_REF" && REF_ARGS=(--ref "${REFS[@]}")
  taskset -c 6-13 scope/.venv/bin/python tools/scope_bridge.py \
    --server "http://127.0.0.1:$PORT" --input "${CAMERA:-/dev/video0}" --input-size "${CAMERA_SIZE:-1280x720}" \
    --output /dev/video10 --preset "${PRESET:-30}" ${NOISE_MODE:+--noise-mode "$NOISE_MODE"} \
    --pipeline "${MODEL:-streamdiffusionv2}" \
    --prompt "${PROMPT:-a stylized 3D animated character, Pixar style render}" \
    "${REF_ARGS[@]}" &
  PIDS+=($!)
  echo "[scope] bridge: ${CAMERA:-/dev/video0} -> Scope (576x320) -> /dev/video10 (AI_Scope_Cam, 720x1280)"
  echo "[scope] live controls (FPS, noise, reference, prompt): http://127.0.0.1:8090   (first ~30 s after start are slow while torch.compile warms up)"
fi

echo "[scope] Ctrl+C to stop."
wait "${PIDS[0]}"
