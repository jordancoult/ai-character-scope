#!/usr/bin/env bash
# Switch the workspace and the scope repo together.
#   ./switch_branch.sh main         stable setup (scope: blackwell-realtime)
#   ./switch_branch.sh node-graph   test branch: node-graph control page + attention steering
# Stop the server first (Ctrl+C in the start_scope.sh terminal).
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  main) ws=main; sc=blackwell-realtime ;;
  node-graph) ws=node-graph; sc=node-graph ;;
  *) echo "usage: $0 main|node-graph   (now: workspace=$(git branch --show-current), scope=$(git -C scope branch --show-current))"; exit 1 ;;
esac
if ss -ltn | grep -q ":8000 "; then echo "Scope is still running on port 8000: stop it first."; exit 1; fi
git checkout -q "$ws" && git -C scope checkout -q "$sc"
echo "workspace=$(git branch --show-current)  scope=$(git -C scope branch --show-current)"
