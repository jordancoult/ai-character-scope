"""Small, explicit patches to the upstream JoyAI server, applied at start by run_server.sh (idempotent; keeps a
.orig copy). Each patch must match exactly once or the script fails loudly (upstream changed: re-check).

1. person detector only for full-body detection: upstream also runs YOLOv8n in the default "near" mode as a
   mid-session body-presence gate whenever yolov8n.onnx exists (black-holds the output when no body is seen).
   We want the detector used only when the user turns full-body detection on.
2. runtime LoRA (runtime_lora.py): installed right after load_dit(); a no-op unless JOYAI_LORA is set.
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

# 2. runtime LoRA: copy the module next to the DiT code and call it after load_dit()
here = Path(__file__).resolve().parent
shutil.copy(here / "runtime_lora.py", Path(sys.argv[1]) / "deploy/xvideo/models/runtime_lora.py")
stream = Path(sys.argv[1]) / "deploy/xvideo/serving/joyomni_streaming.py"
stream_orig = stream.with_suffix(".py.orig")
if not stream_orig.exists():
    shutil.copy(stream, stream_orig)
ssrc = stream_orig.read_text()
old = "        dit = load_dit(cfg, device=device_obj)\n"
if ssrc.count(old) != 1:
    sys.exit(f"patch_server: expected 1 load_dit() call in {stream}, found {ssrc.count(old)}; upstream changed")
ssrc = ssrc.replace(old, old + "        from xvideo.models.runtime_lora import maybe_install as _ail_lora  # ai-livestream patch\n"
                    "        dit = _ail_lora(dit)\n")
stream.write_text(ssrc)
print("patched", stream)
