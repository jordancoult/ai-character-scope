#!/usr/bin/env bash
# Switch the workspace and the scope repo together (they must stay paired).
#   ./switch_branch.sh main          the one active setup (scope: blackwell-realtime)
#   ./switch_branch.sh <name>        any other branch name: same name in both repos (for experiments)
# Old pre-merge states (2026-11-02) are tags: main-pre-merge-2026-11-02 / blackwell-realtime-pre-merge-2026-11-02.
# Stop the server first (Ctrl+C in the start_scope.sh terminal).
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  main) ws=main; sc=blackwell-realtime ;;
  "") echo "usage: $0 main|<branch>   (now: workspace=$(git branch --show-current), scope=$(git -C scope branch --show-current))"; exit 1 ;;
  *) ws=$1; sc=$1 ;;
esac
if ss -ltn | grep -q ":8000 "; then echo "Scope is still running on port 8000: stop it first."; exit 1; fi
git checkout -q "$ws" && git -C scope checkout -q "$sc"
echo "workspace=$(git branch --show-current)  scope=$(git -C scope branch --show-current)"
