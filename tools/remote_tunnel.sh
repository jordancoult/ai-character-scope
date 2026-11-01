#!/bin/bash
# Public HTTPS link to the bridge control page (http://127.0.0.1:8090) for testing from a phone.
#
#   tools/remote_tunnel.sh                     # new random token each run
#   REMOTE_TOKEN=mytoken tools/remote_tunnel.sh
#
# Uses a free Cloudflare "quick tunnel" (no account, no API key): cloudflared is downloaded into
# ~/personal/ai-character-scope/bin/ on first use (no sudo). The bridge must be started with the SAME
# token (REMOTE_TOKEN=... ./start_scope.sh --bridge); the script refuses to expose a bridge that answers
# without a token, and stops the tunnel if one shows up later. Ctrl+C stops the tunnel.
# Env: CONTROL_PORT=8090 (bridge --control-port), CLOUDFLARED=path (use another binary).
set -u
W="$HOME/personal/ai-character-scope"
PORT="${CONTROL_PORT:-8090}"
BIN="${CLOUDFLARED:-$W/bin/cloudflared}"
LOG_DIR="$W/.tmp"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cloudflared.log"

if [ ! -x "$BIN" ]; then
  echo "[tunnel] downloading cloudflared (linux-amd64) -> $BIN"
  mkdir -p "$(dirname "$BIN")"
  curl -fL --progress-bar -o "$BIN.part" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    || { echo "[tunnel] download failed"; rm -f "$BIN.part"; exit 1; }
  chmod +x "$BIN.part" && mv "$BIN.part" "$BIN"
fi
echo "[tunnel] $("$BIN" --version 2>&1 | head -1)"

if [ -z "${REMOTE_TOKEN:-}" ]; then
  REMOTE_TOKEN="$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)"
  echo "[tunnel] generated a new token"
fi
(umask 077; echo "$REMOTE_TOKEN" > "$LOG_DIR/remote_token")

# 000 = nothing listening, 401 = protected (good), anything else = open bridge or not the bridge (refuse)
status() { curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:$PORT/state" 2>/dev/null; }
case "$(status)" in
  401) echo "[tunnel] bridge on :$PORT requires a token: good (make sure it is this run's token)" ;;
  000) echo "[tunnel] nothing on :$PORT yet. Start the bridge in another terminal with this token:"
       echo
       echo "    cd ~/personal/ai-character-scope && REMOTE_TOKEN=$REMOTE_TOKEN ./start_scope.sh --bridge"
       echo "    (or: REMOTE_TOKEN=\$(cat .tmp/remote_token) ./start_scope.sh --bridge)"
       echo ;;
  200) echo "[tunnel] REFUSING: the bridge on :$PORT answers WITHOUT a token, so anyone with the link could control it."
       echo "[tunnel] Restart it with:  REMOTE_TOKEN=$REMOTE_TOKEN ./start_scope.sh --bridge"
       exit 1 ;;
  *)   echo "[tunnel] REFUSING: :$PORT answered HTTP $(status) to /state; that doesn't look like the bridge."; exit 1 ;;
esac

: > "$LOG"
"$BIN" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$LOG" 2>&1 &
CF=$!
trap 'echo; echo "[tunnel] stopping"; kill "$CF" 2>/dev/null; wait "$CF" 2>/dev/null; exit 0' INT TERM
trap 'kill "$CF" 2>/dev/null' EXIT

echo -n "[tunnel] starting cloudflared quick tunnel (log: .tmp/cloudflared.log)"
URL=""
for _ in $(seq 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1)"
  [ -n "$URL" ] && break
  kill -0 "$CF" 2>/dev/null || break
  echo -n "."; sleep 1
done
echo
if [ -z "$URL" ]; then
  echo "[tunnel] cloudflared did not report a URL; last log lines:"; tail -n 15 "$LOG"; exit 1
fi

echo
echo "  Phone link (output viewer):  $URL/remote?token=$REMOTE_TOKEN"
echo "  Full control page:           $URL/?token=$REMOTE_TOKEN"
echo
echo "[tunnel] the link can take ~10-30 s to start working. Anyone with it can control the session. Ctrl+C to stop."

# Keep running; stop the tunnel if the bridge ever answers without a token (e.g. restarted without one).
while kill -0 "$CF" 2>/dev/null; do
  if [ "$(status)" = 200 ]; then
    echo "[tunnel] bridge on :$PORT now answers WITHOUT a token: stopping the tunnel."
    kill "$CF" 2>/dev/null; wait "$CF" 2>/dev/null; exit 1
  fi
  sleep 5
done
wait "$CF"; RC=$?
# Ctrl+C reaches cloudflared too (same process group), so it usually exits before the trap runs
if [ "$RC" = 0 ] || [ "$RC" = 130 ]; then echo; echo "[tunnel] stopped"; exit 0; fi
echo "[tunnel] cloudflared exited ($RC); last log lines:"; tail -n 10 "$LOG"
exit 1
