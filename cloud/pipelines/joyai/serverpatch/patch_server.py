"""Small, explicit patches to the upstream JoyAI server, applied at start by run_server.sh (idempotent; keeps a
.orig copy). Each patch must match exactly once or the script fails loudly (upstream changed: re-check).

1. person detector only for full-body detection: upstream also runs YOLOv8n in the default "near" mode as a
   mid-session body-presence gate whenever yolov8n.onnx exists (black-holds the output when no body is seen).
   We want the detector used only when the user turns full-body detection on.
"""
import shutil
import sys
from pathlib import Path

srv = Path(sys.argv[1]) / "deploy/xvideo/serving/serve_joyomni_streaming.py"
orig = srv.with_suffix(".py.orig")
if not orig.exists():
    shutil.copy(srv, orig)
src = orig.read_text()

PATCHES = [
    (
        """                            else:
                                gate_state["person_last"] = _person_present(
                                    frame, onnx_path=args.person_detector_onnx, conf=float(args.person_gate_conf))""",
        """                            else:
                                # ai-livestream patch: no YOLO body gate outside full-body detection
                                gate_state["person_last"] = True""",
    ),
]
for old, new in PATCHES:
    n = src.count(old)
    if n != 1:
        sys.exit(f"patch_server: expected 1 match, found {n}; upstream changed")
    src = src.replace(old, new)
srv.write_text(src)
print("patched", srv)
