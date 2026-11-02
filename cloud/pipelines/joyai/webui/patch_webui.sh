#!/bin/bash
# Inject our memory panel (R-key / button reset + auto-reset setting) into the upstream JoyAI web UI.
# Idempotent; the server re-reads index.html per page load, so a browser reload picks it up (no restart).
set -euo pipefail
W=${JOYAI_DATA:-/workspace}
H="$(cd "$(dirname "$0")" && pwd)"
IDX=$W/JoyAI-Video-Edit/deploy/static/index.html
PANEL=${1:-$H/memory_panel.html}
[ -f "$IDX.orig" ] || cp "$IDX" "$IDX.orig"
python3 - "$IDX.orig" "$PANEL" "$IDX" <<'PY'
import sys
orig, panel, out = sys.argv[1:]
html = open(orig, encoding="utf-8").read()
inj = open(panel, encoding="utf-8").read()
assert html.count("</body>") == 1, "unexpected index.html layout"
open(out, "w", encoding="utf-8").write(html.replace("</body>", inj + "\n</body>"))
print("patched", out)
PY
