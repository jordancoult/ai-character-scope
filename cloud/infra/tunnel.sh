#!/bin/bash
# Self-healing SSH tunnel: localhost:18080 -> GPU machine :8080 (reconnects whenever it drops).
#   cloud/infra/tunnel.sh start | stop | status       (address from cloud/infra/.pod)
D="$(cd "$(dirname "$0")" && pwd)"; R="$D/../.."
PIDF="$R/.tmp/tunnel.pid"; mkdir -p "$R/.tmp"
case "${1:-start}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "tunnel already running (pid $(cat "$PIDF"))"; exit 0; fi
    setsid bash -c '
      while true; do
        read -r H P < "'"$D"'/.pod"
        ssh -N -i "'"$R"'/.secrets/runpod_ed25519" -o UserKnownHostsFile="'"$R"'/.secrets/known_hosts" \
            -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -L 18080:127.0.0.1:8080 -p "$P" "root@$H" 2>/dev/null
        sleep 2
      done' > /dev/null 2>&1 < /dev/null &
    echo $! > "$PIDF"; sleep 3
    ss -ltn | grep -q ":18080 " && echo "tunnel up (localhost:18080, supervisor pid $(cat "$PIDF"))" || echo "tunnel starting (supervisor pid $(cat "$PIDF"))" ;;
  stop)
    [ -f "$PIDF" ] && kill -- -"$(cat "$PIDF")" 2>/dev/null; rm -f "$PIDF"; echo "tunnel stopped" ;;
  status)
    [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && echo "supervisor running (pid $(cat "$PIDF"))" || echo "supervisor not running"
    ss -ltn | grep -q ":18080 " && echo "port 18080 listening" || echo "port 18080 not listening" ;;
esac
