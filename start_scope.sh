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
#   MODEL=streamdiffusionv2|longlive|reward-forcing|memflow|longlive2   model (longlive2 = LongLive 2.0, prompt-only, 1024x576)
#   NOISE_MODE=fixed|random|auto          per-chunk noise pattern (default fixed; auto = preset choice; also on the page)
#   CAMERA=/dev/video0  CAMERA_SIZE=1280x720
#   AUDIO_DELAY_MS=350                    delay applied to the AI_Character_Mic virtual mic
#   NO_AUDIO=1                            skip creating the virtual mic
#   ORIENTATION=landscape                 16:9 (576x320 -> 1280x720) instead of 9:16 portrait
#   INPUT_IMAGE=assets/x.png              no camera: animate this still image (also its reference unless CHARACTER_REF set)
#   BRIDGE_ARGS="--size 640 352 --seed 7"  extra scope_bridge.py arguments (generation size, VAE, seed)
#   REMOTE_TOKEN=...                      require this token on the control page (all routes); unset = no auth.
#                                         Read by scope_bridge.py from the environment (kept off the command line).
#
# Remote testing from a phone (output viewer + controls over a free Cloudflare quick tunnel):
#   terminal 1:  tools/remote_tunnel.sh          # prints a token and, once up, https://....trycloudflare.com/remote?token=...
#   terminal 2:  REMOTE_TOKEN=<that token> ./start_scope.sh --bridge
#   The tunnel refuses to run (or stops) if the bridge answers without a token. Ctrl+C stops the tunnel.
#   Output preview without a tunnel: http://127.0.0.1:8090/remote (also /out.jpg, /out.mjpg).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # repo root (this script's folder)
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
# LongLive / Reward Forcing support it too, but off unless asked (SCOPE_FUSED_KV=1); MemFlow never
# (its clean pass also writes the memory bank).
case "${MODEL:-streamdiffusionv2}" in
  longlive|reward-forcing) export SCOPE_FUSED_KV="${SCOPE_FUSED_KV:-0}" ;;
  memflow) export SCOPE_FUSED_KV=0 ;;
  *) export SCOPE_FUSED_KV="${SCOPE_FUSED_KV:-1}" ;;
esac
export SCOPE_COMPILE_TAE="${SCOPE_COMPILE_TAE:-1}"
# Tell the model the true noise level (stock Scope says 100 less): removes residual-noise sparkles,
# 1-step realism 0.50 -> 0.75; LongLive is unusable without it. 100 = stock behavior.
export SCOPE_NOISE_T_OFFSET="${SCOPE_NOISE_T_OFFSET:-0}"
export SCOPE_PRELOAD_DEPTH="${SCOPE_PRELOAD_DEPTH:-1}"
# Latency: keep only a small cushion in the output queue (1.34 s -> 0.46 s on StreamDiffusionV2).
# LongLive-family models deliver 12-frame bursts (one chunk), so their cushion must cover a whole
# chunk: with 3, every burst counted as backlog and was played up to 45% fast, then the output
# stalled until the next chunk (a hitch every ~0.6 s). The output queue also holds two chunks there.
case "${MODEL:-streamdiffusionv2}" in
  longlive|reward-forcing|memflow) OUT_TARGET_DEFAULT=14; SINK_MAX_DEFAULT=24 ;;
  longlive2) OUT_TARGET_DEFAULT=36; SINK_MAX_DEFAULT=72  # 32-frame chunks
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" ;;  # 5B model near 16 GB
  *) OUT_TARGET_DEFAULT=3; SINK_MAX_DEFAULT=12 ;;
esac
export SCOPE_OUTPUT_TARGET_FRAMES="${SCOPE_OUTPUT_TARGET_FRAMES:-$OUT_TARGET_DEFAULT}"
export SCOPE_SINK_QUEUE_MAX="${SCOPE_SINK_QUEUE_MAX:-$SINK_MAX_DEFAULT}"
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
  EXTRA_ARGS=(); INPUT="${CAMERA:-/dev/video0}"
  [ "${ORIENTATION:-portrait}" = landscape ] && EXTRA_ARGS+=(--landscape)
  if [ -n "${INPUT_IMAGE:-}" ]; then INPUT="$INPUT_IMAGE"; CHARACTER_REF="${CHARACTER_REF:-$INPUT_IMAGE}"; fi
  REF_ARGS=(); [ -n "${CHARACTER_REF:-}" ] && read -r -a REFS <<<"$CHARACTER_REF" && REF_ARGS=(--ref "${REFS[@]}")
  taskset -c 6-13 scope/.venv/bin/python tools/scope_bridge.py \
    --server "http://127.0.0.1:$PORT" --input "$INPUT" --input-size "${CAMERA_SIZE:-1280x720}" \
    --output /dev/video10 --preset "${PRESET:-30}" ${NOISE_MODE:+--noise-mode "$NOISE_MODE"} \
    --pipeline "${MODEL:-streamdiffusionv2}" \
    --prompt "${PROMPT:-a stylized 3D animated character, Pixar style render}" \
    "${REF_ARGS[@]}" "${EXTRA_ARGS[@]}" ${BRIDGE_ARGS:-} &
  PIDS+=($!)
  echo "[scope] bridge: $INPUT -> Scope (${ORIENTATION:-portrait}) -> /dev/video10 (AI_Scope_Cam)"
  echo "[scope] live controls (FPS, noise, reference, prompt): http://127.0.0.1:8090   (first ~30 s after start are slow while torch.compile warms up)"
  [ -n "${REMOTE_TOKEN:-}" ] && echo "[scope] remote token set: control page needs ?token=... (phone viewer: /remote; tunnel: tools/remote_tunnel.sh)"
fi

echo "[scope] Ctrl+C to stop."
if [ "$BRIDGE" = 1 ]; then
  wait "${PIDS[-1]}"; RC=$?
  # Exit code 75 = "Restart with these settings" on the graph page: relaunch with the new launch env
  if [ "$RC" = 75 ] && [ -f "$LOG_DIR/restart.env" ]; then
    echo "[scope] restarting with new launch settings..."
    cleanup; trap - EXIT
    for _ in $(seq 30); do ss -ltn | grep -q ":$PORT " || break; sleep 1; done
    set -a; source "$LOG_DIR/restart.env"; set +a; rm -f "$LOG_DIR/restart.env"
    exec "$0" --bridge
  fi
fi
wait "${PIDS[0]}"
