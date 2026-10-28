#!/bin/bash
# End-to-end latency test: fresh Scope server + bridge streaming assets/latency_flash.mp4
# (white flash every 5 s). Prints per-flash latency, summary, output FPS and peak temps.
# Usage: tools/latency_test.sh [tag] [extra bridge args...]   (~2 min incl. startup)
W="$HOME/personal/ai-character-scope"; source "$W/env.sh"; cd "$W"
TAG="${1:-run}"; shift
LOG=".tmp/latency_$TAG.log"
# Precise cleanup: whatever holds the Scope port (name-matching could hit the caller's shell)
fuser -k -TERM 8000/tcp >/dev/null 2>&1; sleep 2
NO_AUDIO=1 ./start_scope.sh > ".tmp/start_scope_$TAG.log" 2>&1 &
for _ in $(seq 1 90); do curl -sf localhost:8000/api/v1/hardware/info >/dev/null && break; sleep 2; done
T0=$(date +%T)
taskset -c 6-13 scope/.venv/bin/python tools/scope_bridge.py --input assets/latency_flash.mp4 --output /dev/video10 \
  --ref assets/char_face.png --prompt "a pink starfish man in a black baseball cap and a gold chain, head and shoulders portrait, 3D Pixar style" \
  --latency-probe --duration 70 --control-port 0 "$@" > "$LOG" 2>&1
# Stop by port: the server exits, start_scope.sh's wait returns and its cleanup runs.
fuser -k -TERM 8000/tcp >/dev/null 2>&1; sleep 4
echo "== $TAG ($*)"
grep -E "^\[latency\]" "$LOG" | awk '{print $2}' | tr '\n' ' '; echo
grep -E "summary|\[smooth\]" "$LOG"
echo "fps: $(grep -E 'output .* fps' "$LOG" | awk '{print $3}' | tail -8 | tr '\n' ' ')"
awk -v t="$T0" '$1>=t' .tmp/thermal.log | sed -n 's/.*gpu=\([0-9]*\)C.*cpu_max=\([0-9]*\)C/\1 \2/p' |
  awk '{if($1>g)g=$1; if($2>c)c=$2} END {print "peak gpu " g "C, cpu " c "C"}'
grep -m2 -E "Traceback|Error" "$LOG"
