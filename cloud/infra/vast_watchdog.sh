#!/bin/bash
# Self-stop watchdog for a rented Vast.ai instance (runs ON the instance, so it works even if the controlling
# laptop session freezes, sleeps or loses network). Stops the instance (GPU billing ends, disk kept) when
#   - no heartbeat for IDLE_MIN minutes (anything that touches $HB counts; our ssh helper touches it on every call), or
#   - the hard deadline DEADLINE_H hours after start is reached.
#   nohup bash vast_watchdog.sh [IDLE_MIN=45] [DEADLINE_H=8] > /workspace/watchdog.log 2>&1 &
# Uses the instance's own CONTAINER_ID / CONTAINER_API_KEY (Vast sets them in the container environment).
IDLE_MIN=${1:-45}
DEADLINE_H=${2:-8}
HB=/workspace/.heartbeat
touch "$HB"
START=$(date +%s)
DEADLINE=$((START + DEADLINE_H * 3600))
# the keys are in PID 1's environment even when this runs from an ssh session
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | sed 's/^/export /')"
[ -n "$CONTAINER_ID" ] && [ -n "$CONTAINER_API_KEY" ] || { echo "$(date -u +%T) no CONTAINER_ID/CONTAINER_API_KEY: cannot self-stop"; exit 1; }
echo "$(date -u +%T) watchdog: instance $CONTAINER_ID, idle limit ${IDLE_MIN} min, hard deadline $(date -u -d @$DEADLINE +%T) UTC"
stop_now() {
    echo "$(date -u +%T) STOPPING instance $CONTAINER_ID: $1"
    for i in 1 2 3; do
        code=$(curl -s -o /tmp/wd_resp -w '%{http_code}' -X PUT "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" \
            -H "Authorization: Bearer $CONTAINER_API_KEY" -H 'Content-Type: application/json' -d '{"state": "stopped"}')
        echo "$(date -u +%T) stop request -> HTTP $code $(head -c 200 /tmp/wd_resp)"
        [ "$code" = 200 ] && exit 0
        sleep 20
    done
}
while true; do
    now=$(date +%s)
    age=$(( now - $(stat -c %Y "$HB") ))
    [ "$now" -ge "$DEADLINE" ] && stop_now "hard deadline (${DEADLINE_H} h)"
    [ "$age" -ge $((IDLE_MIN * 60)) ] && stop_now "no heartbeat for $((age / 60)) min"
    sleep 60
done
