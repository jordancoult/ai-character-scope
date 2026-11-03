"""Review outputs: per-pair side-by-side mp4 + contact sheet, and a summary table of all gates."""

import csv
from pathlib import Path

from common import log, run

GATES = ["duration", "offset_conf", "person_in", "gstar_out", "mouth_sync", "framing"]


def _label(an):
    al, vi = an.get("align") or {}, an.get("vision") or {}
    txt = (f"{an['pair_id']}  {an.get('status', '?').upper()}  off {al.get('offset_s', 0):+.3f}s "
           f"({al.get('method')}, r={al.get('ratio')})  sync {vi.get('lipgap_corr')}  "
           f"face {vi.get('face_in')}/{vi.get('face_out')}")
    return txt.replace(":", r"\:").replace("'", "").replace("%", r"\%")


def pair_review(pdir, an, a):
    """review.mp4: aligned input | output (normalized), h264 high@4.1, <=1920 wide, +faststart; review.jpg."""
    pdir = Path(pdir)
    ni, no = pdir / "norm_input.mp4", pdir / "norm_output.mp4"
    if not (ni.exists() and no.exists()):
        return
    mp4, jpg = pdir / "review.mp4", pdir / "review.jpg"
    fc = ("[0:v]setsar=1[a];[1:v]setsar=1[b];[a][b]hstack=2:shortest=1,"
          "scale='min(1920,iw)':-2:flags=lanczos,"
          f"drawtext=text='{_label(an)}':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.6,"
          "format=yuv420p[v]")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", ni, "-i", no, "-filter_complex", fc, "-map", "[v]",
           "-map", "0:a?", "-c:v", "libx264", "-crf", "22", "-preset", "medium", "-profile:v", "high",
           "-level", "4.1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
           "-shortest", mp4]
    try:
        run(cmd)
    except RuntimeError as e:  # drawtext needs fontconfig; retry without the label
        log.warning("review with label failed (%s); retrying without", str(e)[:120])
        cmd[cmd.index(fc)] = fc.replace(f"drawtext=text='{_label(an)}':x=8:y=8:fontsize=16:fontcolor=white:"
                                        "box=1:boxcolor=black@0.6,", "")
        run(cmd)
    # contact sheet: 8 evenly spaced aligned pairs, 4x2 grid, each tile = input|output side by side
    n = (an.get("align") or {}).get("frames") or 0
    if n:
        k = max(1, n // 8)
        run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-vf",
             f"select='not(mod(n\\,{k}))',scale=-2:360,tile=4x2:padding=4:color=white",
             "-frames:v", "1", "-fps_mode", "passthrough", "-q:v", "3", jpg])


def summary(out, analyses):
    out = Path(out)
    rows = []
    for an in analyses:
        al, vi, g = an.get("align") or {}, an.get("vision") or {}, an.get("gates") or {}
        fr = vi.get("framing") or {}
        rows.append({
            "pair_id": an["pair_id"], "status": an.get("status"), "clips": an.get("n_clips", 0),
            "offset_s": al.get("offset_s"), "method": al.get("method"), "conf_ratio": al.get("ratio"),
            "overlap_s": al.get("overlap_s"),
            "res_in": _res(al.get("probe_in")), "res_out": _res(al.get("probe_out")),
            "fps_in": (al.get("probe_in") or {}).get("fps"), "fps_out": (al.get("probe_out") or {}).get("fps"),
            "aspect_mismatch": al.get("aspect_mismatch"), "face_scale_out_vs_in": fr.get("scale"),
            "matched_crop": al.get("extra_crop"),
            "face_in": vi.get("face_in"), "pose_in": vi.get("pose_in"), "face_out": vi.get("face_out"),
            "lipgap_corr": vi.get("lipgap_corr"), "best_lag": vi.get("lipgap_best_lag"),
            "jaw_corr": vi.get("jaw_corr"),
            **{f"g_{k}": (g.get(k) or {}).get("status") for k in GATES},
            "errors": " | ".join(an.get("errors") or []), "notes": an.get("notes", ""),
        })
    if not rows:
        return
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["pair_id", "status", "clips", "offset_s", "method", "conf_ratio", "overlap_s", "res_in", "res_out",
            "face_scale_out_vs_in", "face_in", "face_out", "lipgap_corr", "best_lag", "errors"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]).replace("|", "/")[:80] for c in cols) + " |")
    tot = {}
    for r in rows:
        tot[r["status"]] = tot.get(r["status"], 0) + 1
    md = [f"# Pair summary\n\n{len(rows)} pairs: {tot}; {sum(r['clips'] for r in rows)} clips.\n"
          f"Per-pair review: pairs/<pair_id>/review.mp4 / review.jpg (left input, right output).\n"] + md
    (out / "summary.md").write_text("\n".join(md) + "\n")


def _res(p):
    return f"{p['width']}x{p['height']}" if p else None
