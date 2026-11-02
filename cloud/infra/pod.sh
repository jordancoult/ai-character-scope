#!/bin/bash
# ssh/scp helper for the current GPU machine (any provider). Address in cloud/infra/.pod ("HOST PORT"), key in .secrets/ at the repo root.
D="$(cd "$(dirname "$0")" && pwd)"
read -r HOST PORT < "$D/.pod"
KEY="$D/../../.secrets/runpod_ed25519"
OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$D/../../.secrets/known_hosts" -o ServerAliveInterval=30 -o ConnectTimeout=15)
case "$1" in
  put) shift; scp -q -P "$PORT" "${OPTS[@]}" -r "${@:1:$#-1}" "root@$HOST:${@: -1}" ;;
  get) shift; scp -q -P "$PORT" "${OPTS[@]}" -r "root@$HOST:$1" "$2" ;;
  *) ssh -p "$PORT" "${OPTS[@]}" "root@$HOST" "$@" ;;
esac
