#!/bin/bash
# Upload our build artifacts (kernel wheels + per-GPU compile-cache tars) to the private Hugging Face repo, so
# setup.sh on any provider can skip the kernel build and the first ~12-min compile.
# Run on the GPU machine after setup.sh env + a first server start + save_cache.sh. Needs a WRITE token in
# ~/.cache/huggingface/token (or HF_TOKEN). Layout in the repo:  joyai/wheels/*.whl, joyai/cache/<GPU>.tar
set -euo pipefail
W=${JOYAI_DATA:-/workspace}
REPO=${ARTIFACTS_REPO:-JordanCo/live-character-artifacts}
export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/root/venv/bin/hf
$HF repo create "$REPO" --private --repo-type model >/dev/null 2>&1 || true   # no-op if it exists
$HF upload "$REPO" "$W/wheels" joyai/wheels --repo-type model --include "*.whl" --commit-message "joyai kernel wheels"
for t in "$W"/cache/*.tar; do
  [ -f "$t" ] && $HF upload "$REPO" "$t" "joyai/cache/$(basename "$t")" --repo-type model --commit-message "joyai compile cache $(basename "$t" .tar)"
done
echo PUBLISH_DONE
