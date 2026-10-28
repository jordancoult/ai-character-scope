#!/bin/bash
# Logs GPU/CPU temps every 3s. If either is over its limit for 2 consecutive
# samples, SIGTERMs every process currently using the GPU for compute
# (whatever launched it), then SIGKILLs survivors after 10s.
# Usage: tools/thermal_watchdog.sh [gpu_limit_C=87] [cpu_limit_C=100] [strikes=2]
#   strict (unattended benchmarks): 87 100 2   -> ~6 s over limit
#   live   (streaming):             90 103 10  -> ~30 s over limit. The GPU targets 87 C by design
#                                                 (max operating ~88 C, slowdown ~90 C); Intel throttles at 105 C,
#                                                 so brief CPU spikes to 100-105 are within spec
GPU_LIMIT="${1:-87}"; CPU_LIMIT="${2:-100}"; MAX_STRIKES="${3:-2}"
LOG="$HOME/personal/ai-character-scope/.tmp/thermal.log"
strikes=0
while true; do
  read -r GT GP GC <<<"$(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits | tr -d ',')"
  CT=$(( $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1) / 1000 ))
  echo "$(date +%T) gpu=${GT}C ${GP}W ${GC}MHz cpu_max=${CT}C" >> "$LOG"
  if [ "${GT%.*}" -ge "$GPU_LIMIT" ] || [ "$CT" -ge "$CPU_LIMIT" ]; then
    strikes=$((strikes + 1))
  else
    strikes=0
  fi
  if [ "$strikes" -ge "$MAX_STRIKES" ]; then
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
    echo "$(date +%T) OVER LIMIT (gpu ${GT}/${GPU_LIMIT} cpu ${CT}/${CPU_LIMIT}) - TERM gpu pids: ${PIDS:-none}" >> "$LOG"
    [ -n "$PIDS" ] && kill $PIDS 2>/dev/null
    sleep 10
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
    [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null && echo "$(date +%T) KILL -9 $PIDS" >> "$LOG"
    strikes=0
  fi
  sleep 3
done
