"""Shared helpers: ffmpeg/ffprobe wrappers, logging, small utils."""

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
log = logging.getLogger("pairs")


def setup_logging(log_file=None, verbose=False):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def run(cmd, **kw):
    """Run a command; raise with stderr on failure."""
    r = subprocess.run([str(c) for c in cmd], capture_output=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}): {r.stderr.decode(errors='replace')[-800:]}")
    return r


def probe(path):
    """Return {'duration', 'width', 'height', 'fps', 'has_audio', 'rotation'} for a media file."""
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path])
    info = json.loads(r.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video" and s.get("disposition", {}).get(
        "attached_pic", 0) == 0), None)
    if v is None:
        raise RuntimeError(f"no video stream in {path}")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(v.get("tags", {}).get("rotate", rot))
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90:  # ffmpeg autorotates on decode
        w, h = h, w
    dur = float(info["format"].get("duration") or v.get("duration") or 0)
    return {"duration": dur, "width": w, "height": h, "fps": round(fps, 3),
            "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]), "rotation": rot}


def read_audio(path, sr=16000):
    """Mono float32 audio at `sr` (empty array if the file has no audio)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                        "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return np.zeros(0, np.float32)
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0


def read_frames(path, every=1, max_frames=None):
    """Decode frames (BGR uint8) with OpenCV; keep every `every`-th frame."""
    import cv2
    cap, out, i = cv2.VideoCapture(str(path)), [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            out.append(f)
            if max_frames and len(out) >= max_frames:
                break
        i += 1
    cap.release()
    return out


def sha(s, n=12):
    return hashlib.sha1(str(s).encode()).hexdigest()[:n]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def rnd(x, n=3):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except TypeError:
        pass
    return round(float(x), n)
