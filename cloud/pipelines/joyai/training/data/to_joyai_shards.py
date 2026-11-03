#!/usr/bin/env python3
"""Convert build_pairs.py output (manifest.jsonl + paired clips) into JoyAI RV2V webdataset TAR shards.

Schema from ../PLAN.md section 3 (upstream training/src/dataset/webdataset.py):
  <key>.0.mp4   source / driver clip (H.264, audio stripped)
  <key>.1.mp4   target / Kling G-Star clip (same fps + frame count, frame-accurate)
  <key>.0.png   reference image (assets/char_front.png, the image served live)
  <key>.json    {"caption": <instruction>, "metadata": {"ref_image": "char_front.png", "driver": ..., "clip": ...}}
  keys contain no dots; ~25 samples per shard.
Manifest per split: {"resolved_files": ["/abs/rv2v_train_000.tar", ...], "num_samples": N}
Optional mouth_mask_<split>.npz: key -> uint8 [T_latent, H/24, W/24], lips bbox on the target (fallback: the
aligned source) per 8-frame latent group (frame 0 alone, then 1-8, 9-16, ...), dilated by one cell.

  cloud/.venv/bin/python to_joyai_shards.py [--processed DIR] [--out DIR] [--val-drivers d1 d2 | --val-frac 0.1]
      [--status pass] [--path-prefix /workspace/rv2v] [--mouth-mask]
"""

import argparse
import io
import json
import random
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, REPO, read_frames  # noqa: E402

GRID = 24          # latent grid in pixels (VAE spatial 8 x patch 3)
TFACTOR = 8        # VAE temporal factor


def video_bytes(path):
    """The clip's video stream only (stream copy, audio dropped), as mp4 bytes."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4") as t:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-an",
                        "-movflags", "+faststart", t.name], check=True)
        return Path(t.name).read_bytes()


def add(tar, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 0
    tar.addfile(ti, io.BytesIO(data))


def mouth_mask(vis, out_clip, in_clip, frames):
    """uint8 [T_latent, H/GRID, W/GRID] mouth mask from the target clip (fallback: aligned source)."""
    fo, fi = read_frames(out_clip), read_frames(in_clip)
    H, W = fo[0].shape[:2]
    gh, gw = H // GRID, W // GRID
    boxes = []
    for a, b in zip(fo, fi):
        bx = None
        pts_in, _, roi_in = vis.landmarks_any(b)
        if roi_in is None and pts_in is not None:
            side = max(64.0, 2.2 * float(np.ptp(pts_in[:, 1])))
            cx, cy = pts_in.mean(0)
            roi_in = (cx - side / 2, cy - side / 2, side)
        pts, _, _ = vis.landmarks_any(a, roi_in)  # target first; the aligned source's head ROI as a hint
        if pts is None:
            pts = pts_in
        if pts is not None:
            lips = pts[[61, 291, 0, 17, 13, 14]]
            bx = (*lips.min(0), *lips.max(0))
        boxes.append(bx)
    # fill gaps with the nearest detected box
    idx = [i for i, b in enumerate(boxes) if b is not None]
    if not idx:
        return None
    for i, b in enumerate(boxes):
        if b is None:
            boxes[i] = boxes[min(idx, key=lambda j: abs(j - i))]
    T = (frames - 1) // TFACTOR + 1
    m = np.zeros((T, gh, gw), np.uint8)
    for t in range(T):
        group = [0] if t == 0 else range(1 + (t - 1) * TFACTOR, min(frames, 1 + t * TFACTOR))
        for i in group:
            x0, y0, x1, y1 = boxes[i]
            c0, r0 = int(x0 // GRID) - 1, int(y0 // GRID) - 1          # dilate by one cell
            c1, r1 = int(x1 // GRID) + 1, int(y1 // GRID) + 1
            m[t, max(0, r0):min(gh, r1 + 1), max(0, c0):min(gw, c1 + 1)] = 1
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", default=str(HERE / "processed"))
    p.add_argument("--out", default=None, help="default: <processed>/shards")
    p.add_argument("--ref", default=str(REPO / "assets" / "char_front.png"))
    p.add_argument("--status", nargs="+", default=["pass"], help="clip statuses to include")
    p.add_argument("--val-drivers", nargs="*", default=None, help="drivers held out for validation")
    p.add_argument("--val-frac", type=float, default=0.1, help="else hold out ~this fraction of drivers")
    p.add_argument("--per-shard", type=int, default=25)
    p.add_argument("--path-prefix", default=None, help="write manifest paths under this dir (GPU machine)")
    p.add_argument("--caption", default=None, help="override the instruction stored in the manifest")
    p.add_argument("--mouth-mask", action="store_true", help="also write mouth_mask_<split>.npz (MediaPipe)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    proc = Path(a.processed)
    out = Path(a.out or proc / "shards")
    out.mkdir(parents=True, exist_ok=True)
    clips = [json.loads(x) for x in (proc / "manifest.jsonl").read_text().splitlines() if x.strip()]
    clips = [c for c in clips if c.get("status") in a.status]
    if not clips:
        sys.exit("no clips with status " + ",".join(a.status))
    ref_png = Path(a.ref).read_bytes()
    drivers = sorted({c.get("driver") or c["pair_id"] for c in clips})
    if a.val_drivers is not None:
        val = set(a.val_drivers)
    else:
        rng = random.Random(a.seed)
        k = int(round(len(drivers) * a.val_frac)) if len(drivers) > 1 else 0
        val = set(rng.sample(drivers, k))
    splits = {"train": [c for c in clips if (c.get("driver") or c["pair_id"]) not in val],
              "val": [c for c in clips if (c.get("driver") or c["pair_id"]) in val]}
    vis = None
    if a.mouth_mask:
        from vision import Vision
        vis = Vision()
    for split, items in splits.items():
        if not items:
            continue
        files, masks = [], {}
        for si in range(0, len(items), a.per_shard):
            tpath = out / f"rv2v_{split}_{si // a.per_shard:03d}.tar"
            with tarfile.open(tpath, "w") as tar:
                for c in items[si:si + a.per_shard]:
                    key = c["clip_id"].replace(".", "_")
                    add(tar, f"{key}.0.mp4", video_bytes(proc / c["input"]))
                    add(tar, f"{key}.1.mp4", video_bytes(proc / c["output"]))
                    add(tar, f"{key}.0.png", ref_png)
                    meta = {"caption": a.caption or c.get("instruction") or "",
                            "metadata": {"ref_image": Path(a.ref).name, "driver": c.get("driver"),
                                         "clip": c["clip_id"], "fps": c["fps"], "frames": c["frames"],
                                         "offset_s": c.get("offset_s"), "sync": c.get("sync")}}
                    add(tar, f"{key}.json", json.dumps(meta).encode())
                    if vis is not None:
                        m = mouth_mask(vis, proc / c["output"], proc / c["input"], c["frames"])
                        if m is not None:
                            masks[key] = m
            files.append(tpath)
        resolved = [str(Path(a.path_prefix) / f.name) if a.path_prefix else str(f.resolve()) for f in files]
        (out / f"rv2v_{split}.json").write_text(json.dumps({"resolved_files": resolved, "num_samples": len(items)},
                                                           indent=1))
        if masks:
            np.savez_compressed(out / f"mouth_mask_{split}.npz", **masks)
        print(f"{split}: {len(items)} samples, {len(files)} shards -> {out / f'rv2v_{split}.json'}"
              + (f", {len(masks)} mouth masks" if vis else ""))
    if vis is not None:
        vis.close()
    print("val drivers:", sorted(val))


if __name__ == "__main__":
    main()
