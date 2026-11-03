"""Face-aligned training clips for the JoyAI lip-sync LoRA (stage after build_pairs.py).

Why: Kling motion-control outputs are NOT pixel-aligned with the driver. Kling replaces the background, zooms in
(face 1.1-1.9x larger) and re-frames the body. A plain per-pixel loss would teach JoyAI to zoom and swap
backgrounds. So per frame we warp the Kling output with a similarity transform that puts its face where the
driver's face is (MediaPipe landmarks on both), and write per-latent loss masks:

  face   fraction of each 24-px latent cell inside the (warped) target face hull  -> supervised by the Kling target
  mouth  fraction inside the (warped) target lip region (dilated)                  -> extra weight
  valid  per latent frame: both faces were found (or interpolated over short gaps)

Outside the face, the trainer anchors the LoRA output to the frozen base model's own output (keeps background,
body, identity as today).

Also fixes the per-pair residual lag: if the lip-gap correlation is clearly better at lag L (|L| <= 2 frames),
the output is shifted by L frames before cutting.

Input: processed/pairs/<pair>/norm_input.mp4, norm_output.mp4, analysis.json (from build_pairs.py).
Output (--out, default processed/aligned):
  clips/<key>.src.mp4  driver, 480x840 24 fps, N frames (8k+1)
  clips/<key>.tgt.mp4  face-aligned Kling output, same size / frames
  clips/<key>.masks.npz  face, mouth float16 [T_lat, H/24, W/24], valid uint8 [T_lat]
  clips/<key>.review.jpg  driver | aligned target with the face/mouth mask outline (a few frames)
  metadata.jsonl, summary.md
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import log, setup_logging, read_frames, run  # noqa: E402

# Similarity-fit points: outer/inner eye corners, nose tip, mouth corners, upper/lower inner lip, chin.
FIT_IDX = [33, 133, 362, 263, 1, 61, 291, 13, 14, 152]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
CELL = 24
VAE_T = 8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default=str(HERE / "processed" / "pairs"))
    p.add_argument("--out", default=str(HERE / "processed" / "aligned"))
    p.add_argument("--exclude", nargs="*", default=["p01_elgomez__", "p08_kevvo"],
                   help="pairs to drop (p01: lip corr 0.21 on tiny faces; p08: misaligned, no face in input)")
    p.add_argument("--min-sync", type=float, default=0.4, help="drop pairs whose best-lag lip-gap corr is lower")
    p.add_argument("--frames", type=int, default=97, help="clip length (8k+1)")
    p.add_argument("--stride", type=int, default=48, help="clip start stride (overlap = augmentation)")
    p.add_argument("--max-gap", type=int, default=6, help="interpolate the transform over at most this many "
                   "frames without a face; longer gaps are invalid")
    p.add_argument("--smooth", type=float, default=1.5, help="temporal Gaussian sigma (frames) on the transform")
    p.add_argument("--face-dilate", type=float, default=0.12, help="face hull dilation, fraction of face height")
    p.add_argument("--mouth-dilate", type=float, default=0.10, help="lip hull dilation, fraction of face height")
    p.add_argument("--min-lag-gain", type=float, default=0.02)
    p.add_argument("--only", nargs="*")
    p.add_argument("--raw", action="store_true", help="no spatial warp: the target is the Kling output exactly as "
                   "generated (time-aligned + trimmed only); masks = the target's own face/lips; valid = all frames")
    return p.parse_args(argv)


def landmarks_all(vis, frames, fallback=None):
    pts = []
    for i, f in enumerate(frames):
        fb = fallback[i] if fallback is not None else None
        p, _, _ = vis.landmarks_any(f, fb)
        pts.append(p)
    return pts


def roi_from_pts(p, H, W):
    if p is None:
        return None
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    side = max(64.0, 2.2 * (y1 - y0))
    return ((x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2, side)


def fit_similarity(src_pts, dst_pts):
    """Similarity (scale, rot, tx, ty) mapping src -> dst, least squares (Umeyama, no reflection)."""
    s, d = src_pts[FIT_IDX], dst_pts[FIT_IDX]
    M, _ = cv2.estimateAffinePartial2D(s, d, method=cv2.LMEDS)
    if M is None:
        return None
    a, b = M[0, 0], M[1, 0]
    return np.array([np.hypot(a, b), np.arctan2(b, a), M[0, 2], M[1, 2]])


def params_to_M(q):
    sc, th, tx, ty = q
    c, s = sc * np.cos(th), sc * np.sin(th)
    return np.array([[c, -s, tx], [s, c, ty]], np.float64)


def smooth_params(P, valid, max_gap, sigma):
    """Interpolate gaps <= max_gap, Gaussian-smooth; returns (P, ok mask)."""
    n = len(P)
    idx = np.arange(n)
    ok = valid.copy()
    if valid.sum() < 2:
        return P, np.zeros(n, bool)
    Q = P.copy()
    for k in range(4):
        Q[:, k] = np.interp(idx, idx[valid], P[valid, k])
    # gap lengths
    good = valid.copy()
    i = 0
    while i < n:
        if not valid[i]:
            j = i
            while j < n and not valid[j]:
                j += 1
            edge = i == 0 or j == n
            if (j - i) <= max_gap and not edge:
                good[i:j] = True
            i = j
        else:
            i += 1
    if sigma > 0:
        r = int(3 * sigma + 0.5)
        ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        ker /= ker.sum()
        for k in range(4):
            x = np.pad(Q[:, k], r, mode="edge")
            Q[:, k] = np.convolve(x, ker, mode="valid")
    return Q, good


def hull_mask(pts, idx, H, W, dilate_px):
    m = np.zeros((H, W), np.uint8)
    if pts is None:
        return m
    hull = cv2.convexHull(pts[idx].astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(m, hull, 255)
    if dilate_px > 0:
        k = int(max(1, round(dilate_px)))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return m


def to_latent(masks, valid):
    """[N,H,W] uint8 masks (frames) -> [T_lat, H/24, W/24] fraction; latent 0 = frame 0, k = frames 8k-7..8k."""
    N, H, W = masks.shape
    T = (N - 1) // VAE_T + 1
    cells = masks.reshape(N, H // CELL, CELL, W // CELL, CELL).mean(axis=(2, 4)) / 255.0
    out = np.zeros((T, H // CELL, W // CELL), np.float32)
    v = np.zeros(T, np.uint8)
    for k in range(T):
        fr = [0] if k == 0 else list(range(8 * k - 7, 8 * k + 1))
        out[k] = cells[fr].mean(0)
        v[k] = int(all(valid[f] for f in fr))
    return out, v


def write_mp4(frames, path, fps=24):
    H, W = frames[0].shape[:2]
    import subprocess
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "14", "-preset", "slow",
                          "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-movflags", "+faststart",
                          str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def process_pair(pdir, vis, a, out):
    an = json.loads((pdir / "analysis.json").read_text())
    pid = an["pair_id"]
    v = an.get("vision") or {}
    lag0 = v.get("lipgap_corr")
    best = v.get("lipgap_best_lag") or [None, 0]
    lag = 0
    if best[0] is not None and lag0 is not None and abs(best[1]) <= 2 and best[0] - lag0 >= a.min_lag_gain:
        lag = int(best[1])
    sync = best[0] if lag else lag0
    if sync is None or sync < a.min_sync:
        log.info("[%s] skip: sync %s < %s", pid, sync, a.min_sync)
        return None, {"pair_id": pid, "status": "skip_sync", "sync": sync}
    fi = read_frames(pdir / "norm_input.mp4")
    fo = read_frames(pdir / "norm_output.mp4")
    n = min(len(fi), len(fo))
    # positive lag = output late: in[t] <-> out[t + lag]
    if lag > 0:
        fi, fo = fi[:n - lag], fo[lag:n]
    elif lag < 0:
        fi, fo = fi[-lag:n], fo[:n + lag]
    else:
        fi, fo = fi[:n], fo[:n]
    n = len(fi)
    H, W = fi[0].shape[:2]
    assert H % CELL == 0 and W % CELL == 0, (H, W)
    log.info("[%s] %d frames, lag %+d (sync %.3f), landmarks...", pid, n, lag, sync)
    pin = landmarks_all(vis, fi)
    rois = [roi_from_pts(p, H, W) for p in pin]
    pout = landmarks_all(vis, fo, rois)
    P = np.zeros((n, 4))
    ok = np.zeros(n, bool)
    for t in range(n):
        if pin[t] is not None and pout[t] is not None:
            q = fit_similarity(pout[t], pin[t])
            if q is not None and 0.3 < q[0] < 2.0:
                P[t], ok[t] = q, True
    face_rate = float(ok.mean())
    P, good = smooth_params(P, ok, a.max_gap, a.smooth)
    if a.raw:  # identity transform; mouth/face masks where the Kling face is detected
        P[:] = (1.0, 0.0, 0.0, 0.0)
        good = np.array([p is not None for p in pout])
    # interpolated output landmarks for frames where the output face is missing but the transform is good
    warped, face_m, mouth_m = [], [], []
    for t in range(n):
        M = params_to_M(P[t])
        warped.append(cv2.warpAffine(fo[t], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT))
        src_pts = pout[t]
        if good[t] and src_pts is not None:
            wp = src_pts @ M[:, :2].T + M[:, 2]
            fh = np.ptp(wp[:, 1])
            face_m.append(hull_mask(wp, FACE_OVAL, H, W, a.face_dilate * fh))
            mouth_m.append(hull_mask(wp, LIPS_OUTER, H, W, a.mouth_dilate * fh))
        elif good[t] and pin[t] is not None and not a.raw:  # output face missing (short gap): use the driver's face region
            fh = np.ptp(pin[t][:, 1])
            face_m.append(hull_mask(pin[t], FACE_OVAL, H, W, a.face_dilate * fh))
            mouth_m.append(hull_mask(pin[t], LIPS_OUTER, H, W, a.mouth_dilate * fh))
        else:
            good[t] = bool(a.raw) and False
            face_m.append(np.zeros((H, W), np.uint8))
            mouth_m.append(np.zeros((H, W), np.uint8))
    face_m, mouth_m = np.stack(face_m), np.stack(mouth_m)
    rows = []
    cdir = out / "clips"
    cdir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, n - a.frames + 1, a.stride))
    for ci, s in enumerate(starts):
        e = s + a.frames
        fv, lv = to_latent(face_m[s:e], good[s:e])
        mv, _ = to_latent(mouth_m[s:e], good[s:e])
        if a.raw:
            lv[:] = 1  # every frame is a valid target ("golden"); masks only add mouth/face weight
        if lv.mean() < 0.6:
            log.info("[%s] clip %d: only %.0f%% valid latents, dropped", pid, ci, 100 * lv.mean())
            continue
        key = f"{pid.replace('.', '_')}_c{ci:02d}"
        write_mp4(fi[s:e], cdir / f"{key}.src.mp4")
        write_mp4(warped[s:e], cdir / f"{key}.tgt.mp4")
        np.savez_compressed(cdir / f"{key}.masks.npz", face=fv.astype(np.float16), mouth=mv.astype(np.float16),
                            valid=lv)
        # review: 4 frames, driver | aligned target with mask outlines
        tiles = []
        for t in np.linspace(s, e - 1, 4).astype(int):
            tgt = warped[t].copy()
            for m, col in ((face_m[t], (0, 255, 0)), (mouth_m[t], (0, 0, 255))):
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tgt, cs, -1, col, 2)
            blend = cv2.addWeighted(fi[t], 0.5, warped[t], 0.5, 0)
            tiles.append(np.hstack([fi[t], tgt, blend]))
        rv = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        cv2.imwrite(str(cdir / f"{key}.review.jpg"), cv2.resize(rv, None, fx=0.5, fy=0.5),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append({"key": key, "pair_id": pid, "driver": None, "start_frame": s, "frames": a.frames,
                     "width": W, "height": H, "fps": 24, "lag_frames": lag, "pair_sync": sync,
                     "face_frac": float(fv.mean()), "mouth_frac": float(mv.mean()),
                     "valid_latents": float(lv.mean())})
    scale = float(np.median(P[good, 0])) if good.any() else None
    summ = {"pair_id": pid, "status": "ok", "frames": n, "lag": lag, "sync": sync, "face_pair_rate": face_rate,
            "good_rate": float(good.mean()), "median_scale_out_to_in": scale, "clips": len(rows)}
    log.info("[%s] %s", pid, summ)
    return rows, summ


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "face_align.log")
    from vision import Vision
    vis = Vision(use_pose=True)
    csv_drivers = {}
    import csv
    with open(HERE / "gstar_pairs.csv") as f:
        for r in csv.DictReader(f):
            csv_drivers[r["pair_id"]] = r.get("driver") or r["pair_id"]
    all_rows, summs = [], []
    for pdir in sorted(Path(a.pairs).iterdir()):
        if not (pdir / "analysis.json").exists() or pdir.name in a.exclude:
            continue
        if a.only and pdir.name not in a.only:
            continue
        try:
            rows, summ = process_pair(pdir, vis, a, out)
        except Exception as e:  # keep going
            log.exception("[%s] failed", pdir.name)
            rows, summ = None, {"pair_id": pdir.name, "status": f"error: {e}"}
        summs.append(summ)
        for r in rows or []:
            r["driver"] = csv_drivers.get(r["pair_id"], r["pair_id"])
            all_rows.append(r)
    vis.close()
    with open(out / "metadata.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    lines = ["# Face-aligned clips", "", f"{len(all_rows)} clips from {sum(s['status'] == 'ok' for s in summs)} "
             "pairs.", "", "| pair | status | frames | lag | sync | face pair rate | good rate | scale out->in | clips |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summs:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            s["pair_id"], s["status"], s.get("frames"), s.get("lag"), s.get("sync"),
            None if s.get("face_pair_rate") is None else round(s["face_pair_rate"], 3),
            None if s.get("good_rate") is None else round(s["good_rate"], 3),
            None if s.get("median_scale_out_to_in") is None else round(s["median_scale_out_to_in"], 3),
            s.get("clips")))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
