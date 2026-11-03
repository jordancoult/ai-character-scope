#!/usr/bin/env python3
"""Build aligned input->output training pairs from a CSV of (driver video, Kling G-Star render) links.

  cloud/.venv/bin/python cloud/pipelines/joyai/training/data/build_pairs.py pairs.csv \
      [--out DIR] [--width 480 --height 840 --fps 30] [--frames 49 --stride 49] [--crop cover|pad]

Per pair: fetch (Zyla for TikTok pages, direct URLs, local paths) -> audio alignment (motion fallback) ->
overlap span -> normalize both to WxH @ fps CFR -> MediaPipe gates (person, G-Star face, mouth sync, framing) ->
fixed-length paired clips + manifest.jsonl -> review side-by-side mp4 + contact sheet + summary.csv/.md.

Idempotent and resumable: downloads are cached in raw/<pair_id>/, every stage stores its result in
<out>/pairs/<pair_id>/analysis.json keyed by its parameters; rerunning only does missing / changed work.
Failures are logged and the run continues. See README.md.
"""

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, REPO, log, probe, read_audio, read_frames, rnd, run, setup_logging, sha, write_json  # noqa: E402
import align  # noqa: E402
import fetch  # noqa: E402
import review  # noqa: E402

DEFAULT_CAPTION_FILE = REPO / "cloud" / "pipelines" / "joyai" / "prompts" / "gstar.txt"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="CSV with columns pair_id,input_url,output_url[,notes][,driver][,flip][,skip_output_s]")
    p.add_argument("--raw", default=str(HERE / "raw"), help="download cache")
    p.add_argument("--out", default=str(HERE / "processed"), help="output root")
    p.add_argument("--only", nargs="*", help="process only these pair_ids")
    # normalization
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=840)
    p.add_argument("--fps", type=float, default=24.0, help="JoyAI trainer reads 24 fps (PLAN.md section 3)")
    p.add_argument("--crop", choices=["cover", "pad"], default="cover",
                   help="cover: scale to fill + centre crop; pad: fit + black bars (same policy for both videos)")
    p.add_argument("--match-framing", action="store_true",
                   help="crop the input to the output's framing when the output is zoomed in (face-box estimate)")
    p.add_argument("--crf", type=int, default=16)
    # clips
    p.add_argument("--frames", type=int, default=97,
                   help="frames per clip; JoyAI buckets are 8k+1 (49, 57, ..., 113); 97 = 4.0 s @ 24 fps")
    p.add_argument("--stride", type=int, default=None, help="frame stride between clip starts (default = --frames)")
    p.add_argument("--include-warn", action="store_true", help="also cut clips for pairs with gate warnings")
    p.add_argument("--no-clips", action="store_true")
    p.add_argument("--no-review", action="store_true")
    # gates
    p.add_argument("--min-duration", type=float, default=3.0)
    p.add_argument("--min-ratio", type=float, default=1.3, help="alignment peak / second peak")
    p.add_argument("--min-person", type=float, default=0.8, help="input frames with a face (or pose)")
    p.add_argument("--min-gstar", type=float, default=0.5, help="output frames with a detected face")
    p.add_argument("--min-sync", type=float, default=0.3, help="lip-gap correlation input vs output")
    p.add_argument("--vision-every", type=int, default=1, help="run face landmarks on every k-th frame")
    p.add_argument("--no-pose", action="store_true")
    # misc
    p.add_argument("--caption-file", default=str(DEFAULT_CAPTION_FILE),
                   help="edit instruction stored in the manifest ('' for none)")
    p.add_argument("--zyla-endpoint", default=None)
    p.add_argument("--force", nargs="*", default=None,
                   help="redo stages: fetch align vision clips review (no args = all)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    a.stride = a.stride or a.frames
    if (a.frames - 1) % 8:
        print(f"warning: --frames {a.frames} is not 8k+1 (JoyAI VAE temporal factor 8)", file=sys.stderr)
    if a.force is not None and not a.force:
        a.force = ["fetch", "align", "vision", "clips", "review"]
    a.force = set(a.force or [])
    return a


def read_csv(path):
    rows, seen = [], set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    for i, r in enumerate(csv.DictReader(lines)):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        pid = r.get("pair_id") or f"row{i + 1:04d}"
        pid = "".join(c if c.isalnum() or c in "-_." else "_" for c in pid)
        if not r.get("input_url") or not r.get("output_url"):
            log.warning("row %d (%s): missing input_url/output_url, skipped", i + 1, pid)
            continue
        if pid in seen:
            log.warning("duplicate pair_id %s, skipped", pid)
            continue
        seen.add(pid)
        flip = r.get("flip", "").lower() in ("1", "y", "yes", "true", "flip", "flipped", "hflip")
        try:
            skip = float(r.get("skip_output_s") or 0)
        except ValueError:
            log.warning("row %d (%s): bad skip_output_s %r, using 0", i + 1, pid, r.get("skip_output_s"))
            skip = 0.0
        rows.append({"pair_id": pid, "input_url": r["input_url"], "output_url": r["output_url"],
                     "notes": r.get("notes", ""), "driver": r.get("driver", ""), "flip": flip,
                     "skip_output_s": skip})
    return rows


# ---------------------------------------------------------------- normalization

def vf_chain(a, extra_crop=None, hflip=False):
    W, H = a.width, a.height
    f = [f"fps={a.fps}"] + (["hflip"] if hflip else [])  # hflip: input was mirrored before the output was made
    if a.crop == "cover":
        f += [f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={W}:{H}"]
    else:
        f += [f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos",
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black"]
    if extra_crop:  # (scale s, x0, y0) in normalized output coords
        s, x0, y0 = extra_crop
        f += [f"crop=iw/{s}:ih/{s}:{x0}*iw:{y0}*ih", f"scale={W}:{H}:flags=lanczos"]
    f += ["setsar=1", "format=yuv420p"]
    return ",".join(f)


def normalize(src, dst, start, nframes, a, extra_crop=None, hflip=False):
    """Trim [start, start + nframes/fps) to exactly nframes CFR frames at WxH; keep audio (AAC) if present."""
    dur = nframes / a.fps
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.4f}", "-i", src, "-t", f"{dur + 0.5:.4f}",
           "-vf", vf_chain(a, extra_crop, hflip), "-frames:v", str(nframes),
           "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-g", str(int(a.fps)),
           "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "44100", "-af", f"atrim=0:{dur:.4f}",
           "-movflags", "+faststart", dst]
    run(cmd)
    n = count_frames(dst)
    if n != nframes:
        log.warning("%s: %d frames, expected %d", dst, n, nframes)
    return n


def count_frames(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
             "stream=nb_read_packets", "-of", "csv=p=0", path])
    return int(r.stdout.decode().strip().split(",")[0] or 0)


def low_frames(path, fps, size=(64, 112)):
    """Decode at fps, small, for motion alignment."""
    w, h = size
    r = run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-"])
    arr = np.frombuffer(r.stdout, np.uint8)
    return list(arr.reshape(-1, h, w, 3))


def black_span(frames, fps, thresh=18, flat=4):
    """Seconds of leading / trailing frames that are near-black or flat (no picture)."""
    def bad(f):
        return f.mean() < thresh or f.std() < flat
    lead = next((i for i, f in enumerate(frames) if not bad(f)), len(frames))
    trail = next((i for i, f in enumerate(reversed(frames)) if not bad(f)), len(frames))
    lead, trail = (x + 1 if 0 < x < len(frames) else x for x in (lead, trail))  # 1-frame margin (fades)
    return (round(lead / fps, 3), round(trail / fps, 3))


# ---------------------------------------------------------------- stages

def stage_align(pdir, fin, fout, pin, pout, a, flip=False, skip_output_s=0.0):
    """Find offset, overlap, normalize both videos. Returns the 'align' dict."""
    res = {"probe_in": pin, "probe_out": pout}
    aud = None
    if pin["has_audio"] and pout["has_audio"]:
        aud = align.align_audio(read_audio(fin), read_audio(fout))
    res["audio"] = aud
    # motion: fallback without audio, cross-check with audio
    lf_in, lf_out = low_frames(fin, a.fps), low_frames(fout, a.fps)
    mot = align.align_motion(lf_in, lf_out, a.fps)
    res["motion"] = mot
    best = aud or mot
    if best is None:
        raise RuntimeError("alignment failed (no usable audio or motion signal)")
    res.update(offset_s=best["offset_s"], ratio=best["ratio"], peak=best["peak"], method=best["method"])
    in0, out0, dur = align.overlap(best["offset_s"], pin["duration"], pout["duration"])
    # drop leading/trailing black (or blank) frames of either video from the span
    bi, bo = black_span(lf_in, a.fps), black_span(lf_out, a.fps)
    off = best["offset_s"]
    s0 = max(out0, bo[0], bi[0] - off, skip_output_s)  # skip_output_s: user-marked bad start of the output
    s1 = min(out0 + dur, pout["duration"] - bo[1], pin["duration"] - bi[1] - off)
    res["black_trim"] = {"input": bi, "output": bo}
    in0, out0, dur = s0 + off, s0, max(0.0, s1 - s0)
    nframes = int(np.floor(dur * a.fps - 1e-6)) - 1  # 1-frame safety margin at the tail
    res.update(in_start=round(in0, 4), out_start=round(out0, 4), overlap_s=round(max(0, nframes) / a.fps, 3),
               frames=max(0, nframes))
    # spatial: aspect ratios
    ar_in, ar_out = pin["width"] / pin["height"], pout["width"] / pout["height"]
    res["aspect_in"], res["aspect_out"] = round(ar_in, 4), round(ar_out, 4)
    res["aspect_mismatch"] = abs(ar_in - ar_out) / ar_in > 0.02
    if nframes < 2:
        return res
    norm_in, norm_out = pdir / "norm_input.mp4", pdir / "norm_output.mp4"
    n_in = normalize(fin, norm_in, in0, nframes, a, hflip=flip)
    res["flip_input"], res["skip_output_s"] = bool(flip), float(skip_output_s)
    n_out = normalize(fout, norm_out, out0, nframes, a)
    res["frames"] = min(n_in, n_out)
    res["overlap_s"] = round(res["frames"] / a.fps, 3)
    res["extra_crop"] = None
    return res


def stage_vision(pdir, a, al, vis):
    from vision import best_lag, corr, framing
    k = max(1, a.vision_every)
    fi = read_frames(pdir / "norm_input.mp4", every=k)
    fo = read_frames(pdir / "norm_output.mp4", every=k)
    n = min(len(fi), len(fo))
    fi, fo = fi[:n], fo[:n]
    gi, ji, bi, rois = vis.faces(fi)
    go, jo, bo, _ = vis.faces(fo, fallback_rois=rois)
    samp = np.linspace(0, n - 1, min(n, 16)).astype(int)
    pose_in = vis.pose_rate([fi[i] for i in samp])
    pose_out = vis.pose_rate([fo[i] for i in samp])
    fr = framing(bi, bo)
    res = {
        "every": k, "n": n,
        "face_in": rnd(np.mean(~np.isnan(gi))), "face_out": rnd(np.mean(~np.isnan(go))),
        "pose_in": rnd(pose_in), "pose_out": rnd(pose_out),
        "lipgap_corr": rnd(corr(gi, go)), "jaw_corr": rnd(corr(ji, jo)),
        "lipgap_best_lag": best_lag(gi, go, max_lag=max(1, 8 // k)),
        "motion_range_ratio": rnd(np.nanstd(go) / (np.nanstd(gi) or 1e-6)) if np.isfinite(go).any() else None,
        "framing": fr,
        "series": {"gap_in": [rnd(x, 4) for x in gi], "gap_out": [rnd(x, 4) for x in go]},
    }
    return res, (bi, bo)


def maybe_match_framing(pdir, fin, a, al, fr):
    """If the output is zoomed in vs the input, crop the input to the same framing (re-normalize input)."""
    if not fr or fr["scale"] < 1.08:
        return None
    s = min(fr["scale"], 2.0)
    # face centres measured on the normalized frames: x0 = cx_in - cx_out / s (clamped)
    vis = json.loads((pdir / "analysis.json").read_text()).get("vision_boxes")
    cx_i, cy_i, cx_o, cy_o = vis
    x0 = float(np.clip(cx_i - cx_o / s, 0, 1 - 1 / s))
    y0 = float(np.clip(cy_i - cy_o / s, 0, 1 - 1 / s))
    crop = (round(s, 4), round(x0, 4), round(y0, 4))
    normalize(fin, pdir / "norm_input.mp4", al["in_start"], al["frames"], a, extra_crop=crop,
              hflip=al.get("flip_input", False))
    log.info("%s: matched framing, input cropped by %s", pdir.name, crop)
    return crop


def gates(al, vi, a):
    g = {}

    def put(name, value, ok, warn_only=False):
        g[name] = {"value": value, "status": "pass" if ok else ("warn" if warn_only else "fail")}

    put("duration", al.get("overlap_s"), (al.get("overlap_s") or 0) >= a.min_duration)
    put("offset_conf", al.get("ratio"), (al.get("ratio") or 0) >= a.min_ratio)
    aud, mot = al.get("audio"), al.get("motion")
    if aud and mot and mot["ratio"] >= a.min_ratio and abs(aud["offset_s"] - mot["offset_s"]) > 2.0 / a.fps:
        put("align_crosscheck", {"audio": (al.get("audio") or {}).get("offset_s"),
                                 "motion": (al.get("motion") or {}).get("offset_s")}, False, warn_only=True)
    if vi:
        person = max(vi["face_in"] or 0, vi["pose_in"] or 0)
        put("person_in", rnd(person), person >= a.min_person)
        put("gstar_out", vi["face_out"], (vi["face_out"] or 0) >= a.min_gstar, warn_only=True)
        sync = vi["lipgap_corr"]
        lag_sync = (vi["lipgap_best_lag"] or [None])[0]
        best = max([x for x in (sync, lag_sync) if x is not None], default=None)
        put("mouth_sync", sync, best is not None and best >= a.min_sync, warn_only=best is None)
        put("framing", vi["framing"], not al.get("aspect_mismatch") and not (
            vi["framing"] and not 0.8 <= vi["framing"]["scale"] <= 1.25), warn_only=True)
    st = [x["status"] for x in g.values()]
    status = "fail" if "fail" in st else ("warn" if "warn" in st else "pass")
    if not vi and status == "pass":
        status = "warn"
    return g, status


def stage_clips(pdir, a, al, vi, row, analysis, caption):
    cdir = pdir / "clips"
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    n, N, S = al["frames"], a.frames, a.stride
    entries = []
    gap_in = np.array(vi["series"]["gap_in"], float) if vi else None
    gap_out = np.array(vi["series"]["gap_out"], float) if vi else None
    from vision import corr
    for ci, s0 in enumerate(range(0, n - N + 1, S)):
        names = {}
        for side in ("input", "output"):
            src = pdir / f"norm_{side}.mp4"
            dst = cdir / f"{side}_{ci:04d}.mp4"
            t0, t1 = s0 / a.fps, (s0 + N) / a.fps
            run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex",
                 f"[0:v]trim=start_frame={s0}:end_frame={s0 + N},setpts=PTS-STARTPTS[v];"
                 f"[0:a]atrim={t0:.4f}:{t1:.4f},asetpts=PTS-STARTPTS[a]",
                 "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", str(a.crf), "-preset", "medium",
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", dst]
                if has_audio(src) else
                ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf",
                 f"trim=start_frame={s0}:end_frame={s0 + N},setpts=PTS-STARTPTS", "-c:v", "libx264",
                 "-crf", str(a.crf), "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst])
            names[side] = dst
        k = vi["every"] if vi else 1
        clip_sync = rnd(corr(gap_in[s0 // k:(s0 + N) // k], gap_out[s0 // k:(s0 + N) // k])) if vi else None
        entries.append({
            "pair_id": row["pair_id"], "clip_id": f"{row['pair_id']}_{ci:04d}",
            "input": str(names["input"].relative_to(Path(a.out))),
            "output": str(names["output"].relative_to(Path(a.out))),
            "fps": a.fps, "frames": N, "width": a.width, "height": a.height,
            "start_frame": s0, "input_time_s": round(al["in_start"] + s0 / a.fps, 4),
            "output_time_s": round(al["out_start"] + s0 / a.fps, 4),
            "offset_s": al["offset_s"], "align_method": al["method"], "confidence": al["ratio"],
            "sync": vi["lipgap_corr"] if vi else None, "clip_sync": clip_sync,
            "status": analysis["status"], "notes": row["notes"], "driver": row.get("driver") or row["pair_id"],
            "input_url": row["input_url"], "output_url": row["output_url"],
            "caption": "", "instruction": caption,
        })
    with open(cdir / "clips.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return entries


def has_audio(path):
    try:
        return probe(path)["has_audio"]
    except Exception:
        return False


# ---------------------------------------------------------------- driver

def process_pair(row, a, vis_holder, caption):
    pid = row["pair_id"]
    out = Path(a.out)
    pdir = out / "pairs" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    apath = pdir / "analysis.json"
    analysis = json.loads(apath.read_text()) if apath.exists() else {}
    analysis.update(pair_id=pid, input_url=row["input_url"], output_url=row["output_url"], notes=row["notes"])
    analysis.setdefault("errors", [])
    analysis["errors"] = []
    try:
        rdir = Path(a.raw) / pid
        fin, min_ = fetch.fetch(row["input_url"], rdir, "input", a.zyla_endpoint, force="fetch" in a.force)
        fout, mout = fetch.fetch(row["output_url"], rdir, "output", a.zyla_endpoint, force="fetch" in a.force)
        analysis["fetch"] = {"input": {k: min_.get(k) for k in ("method", "zyla_field", "fetched_at")},
                             "output": {k: mout.get(k) for k in ("method", "zyla_field", "fetched_at")}}

        akey = sha(json.dumps([min_.get("source"), min_.get("fetched_at"), mout.get("source"), mout.get("fetched_at"),
                               a.width, a.height, a.fps, a.crop, a.crf, a.match_framing, 4,
                               row.get("flip", False), row.get("skip_output_s", 0.0)]))
        norm_ok = (pdir / "norm_input.mp4").exists() and (pdir / "norm_output.mp4").exists()
        if analysis.get("align_key") != akey or not norm_ok or "align" in a.force:
            log.info("[%s] aligning", pid)
            analysis["align"] = stage_align(pdir, fin, fout, min_["probe"], mout["probe"], a,
                                            flip=row.get("flip", False), skip_output_s=row.get("skip_output_s", 0.0))
            analysis["align_key"] = akey
            analysis.pop("vision_key", None)
            write_json(apath, analysis)
        al = analysis["align"]
        log.info("[%s] offset %+.3fs (%s, ratio %.2f), overlap %.2fs", pid, al["offset_s"], al["method"],
                 al["ratio"], al["overlap_s"])

        vi = analysis.get("vision")
        if al["frames"] >= 10:
            vkey = sha(json.dumps([akey, a.vision_every, a.no_pose, 3]))
            if analysis.get("vision_key") != vkey or "vision" in a.force:
                if vis_holder[0] is None:
                    from vision import Vision
                    vis_holder[0] = Vision(use_pose=not a.no_pose)
                log.info("[%s] vision (%d frames)", pid, al["frames"])
                vi, (bi, bo) = stage_vision(pdir, a, al, vis_holder[0])
                m = ~np.isnan(bi[:, 2]) & ~np.isnan(bo[:, 2])
                analysis["vision_boxes"] = ([float(np.median(bi[m, 0])), float(np.median(bi[m, 1])),
                                             float(np.median(bo[m, 0])), float(np.median(bo[m, 1]))]
                                            if m.sum() >= 10 else None)
                analysis["vision"] = vi
                write_json(apath, analysis)
                if a.match_framing and vi["framing"] and analysis["vision_boxes"]:
                    crop = maybe_match_framing(pdir, fin, a, al, vi["framing"])
                    if crop:
                        al["extra_crop"] = crop
                        vi, _ = stage_vision(pdir, a, al, vis_holder[0])  # re-measure after cropping
                        vi["framing_before_match"] = analysis["vision"]["framing"]
                        analysis["vision"] = vi
                analysis["vision_key"] = vkey
                write_json(apath, analysis)
        else:
            vi = None
            analysis["vision"] = None

        analysis["gates"], analysis["status"] = gates(al, vi, a)
        log.info("[%s] status %s | sync %s | face in %s out %s", pid, analysis["status"],
                 vi and vi["lipgap_corr"], vi and vi["face_in"], vi and vi["face_out"])

        ckey = sha(json.dumps([analysis.get("vision_key"), akey, a.frames, a.stride, analysis["status"],
                               a.include_warn, row["notes"], row.get("driver"), caption, 1]))
        want = not a.no_clips and (analysis["status"] == "pass" or (a.include_warn and analysis["status"] == "warn"))
        cfile = pdir / "clips" / "clips.jsonl"
        if want:
            if analysis.get("clips_key") != ckey or not cfile.exists() or "clips" in a.force:
                log.info("[%s] cutting clips", pid)
                ents = stage_clips(pdir, a, al, vi, row, analysis, caption)
                analysis["clips_key"] = ckey
                analysis["n_clips"] = len(ents)
        else:
            if (pdir / "clips").exists():
                shutil.rmtree(pdir / "clips")
            analysis["n_clips"], analysis["clips_key"] = 0, None

        if not a.no_review:
            rkey = sha(json.dumps([akey, analysis.get("vision_key"), analysis["status"], 1]))
            if analysis.get("review_key") != rkey or "review" in a.force or not (pdir / "review.mp4").exists():
                review.pair_review(pdir, analysis, a)
                analysis["review_key"] = rkey
    except Exception as e:
        log.error("[%s] FAILED: %s", pid, e)
        log.debug(traceback.format_exc())
        analysis["errors"].append(str(e)[:1000])
        analysis["status"] = "error"
    analysis["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(apath, analysis)
    return analysis


def write_manifest(out, analyses):
    out = Path(out)
    n = 0
    with open(out / "manifest.jsonl.tmp", "w") as f:
        for an in analyses:
            cfile = out / "pairs" / an["pair_id"] / "clips" / "clips.jsonl"
            if an.get("n_clips") and cfile.exists():
                for ln in cfile.read_text().splitlines():
                    if ln.strip():
                        f.write(ln + "\n")
                        n += 1
    (out / "manifest.jsonl.tmp").replace(out / "manifest.jsonl")
    return n


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "build.log", a.verbose)
    caption = Path(a.caption_file).read_text().strip() if a.caption_file and Path(a.caption_file).exists() else ""
    rows = read_csv(a.csv)
    if a.only:
        rows = [r for r in rows if r["pair_id"] in set(a.only)]
    log.info("%d pairs from %s", len(rows), a.csv)
    vis_holder = [None]
    analyses = [process_pair(r, a, vis_holder, caption) for r in rows]
    # manifest/summary over every pair in the CSV that has an analysis (not only --only)
    all_rows = read_csv(a.csv)
    allan = []
    for r in all_rows:
        ap = out / "pairs" / r["pair_id"] / "analysis.json"
        if ap.exists():
            allan.append(json.loads(ap.read_text()))
    n = write_manifest(out, allan)
    review.summary(out, allan)
    st = {}
    for an in analyses:
        st[an.get("status")] = st.get(an.get("status"), 0) + 1
    if vis_holder[0] is not None:
        vis_holder[0].close()
    log.info("done: %s | %d clips in %s", st, n, out / "manifest.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
