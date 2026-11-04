#!/bin/bash
# Companion to vast_watchdog.sh (runs ON the instance): keeps the idle heartbeat fresh while the laptop is connected,
# i.e. while any SSH session is established (our self-healing tunnel to the server is one). When the laptop goes away
# (closed, asleep, offline), the tunnel drops, the heartbeat ages and the watchdog stops the instance after IDLE_MIN.
#   nohup bash vast_heartbeat.sh > /dev/null 2>&1 &
HB=/workspace/.heartbeat
while true; do
  if [ "$(ss -Htn state established '( sport = :22 )' 2>/dev/null | wc -l)" -gt 0 ]; then touch "$HB"; fi
  sleep 60
done
