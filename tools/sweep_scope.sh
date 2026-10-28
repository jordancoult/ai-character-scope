#!/bin/bash
# Runs bench_scope.py over a set of configs, cooling down between runs.
# Results append to bench/results.jsonl
W="$HOME/personal/ai-character-scope"; source "$W/env.sh"; cd "$W"
cool() { while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -gt 65 ] || [ $(( $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1) / 1000 )) -gt 80 ]; do sleep 5; done; }
while read -r tag rest; do
  [ -z "$tag" ] && continue
  cool
  echo "== $tag: $rest"
  eval "args=($rest)"
  taskset -c ${CPUS:-6-13} scope/.venv/bin/python tools/bench_scope.py --tag "$tag" "${args[@]}" > ".tmp/sweep-$tag.log" 2>&1
  code=$?
  line=$(grep -h '^RESULT ' ".tmp/sweep-$tag.log" | sed 's/^RESULT //')
  if [ -n "$line" ]; then echo "$line" | tee -a bench/results.jsonl; else echo "{\"tag\":\"$tag\",\"error\":\"exit $code\"}" | tee -a bench/results.jsonl; tail -3 ".tmp/sweep-$tag.log"; fi
done
