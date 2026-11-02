#!/bin/bash
# One bridge run against a running server: run.sh TAG STEPS DEPTH(on|off) [SECONDS]
cd "$(dirname "$0")/../.."
TAG=$1; STEPS=$2; DEPTH=$3; SECS=${4:-45}; EVERY=${5:-2}
OUT=experiments/2026-11-01-fused-ll
scope/.venv/bin/python tools/scope_bridge.py --server http://127.0.0.1:8000 --input assets/webcam_7643311.mp4 \
  --output $OUT/$TAG.mp4 --pipeline reward-forcing --landscape --size 352 640 --out-size 352 640 --upscale none \
  --prompt "a man in a wooden cabin, photorealistic" --noise-scale 0.9 --duration $((SECS + 10)) --latency-probe \
  --control-port 8091 > $OUT/$TAG.log 2>&1 &
BP=$!
for _ in $(seq 120); do grep -q "live controls" $OUT/$TAG.log 2>/dev/null && break; sleep 1; done; sleep 2
curl -s -X POST localhost:8091/params -H 'content-type: application/json' -d "{\"steps\": $STEPS, \"depth\": \"$DEPTH\", \"clean_kv_every\": $EVERY}" >/dev/null
( sleep $((SECS + 45)); kill $BP 2>/dev/null ) & G=$!
wait $BP; kill $G 2>/dev/null
grep "output .* fps" $OUT/$TAG.log | tail -4 | tr '\n' ' '; echo; grep "\[smooth\]" $OUT/$TAG.log
