#!/usr/bin/env python3
"""Simulate input/output pairs from one real clip (no network, no API key) and write a CSV.

  make_test_pairs.py SRC.mp4 OUTDIR [--other OTHER.mp4] [--http-base http://127.0.0.1:PORT]

sim_shift   output = SRC from 1.30 s, 6 frames (0.24 s) of black/silence prepended, centre 80 % crop (zoom 1.25),
            576x1024 @ 25 fps, re-encoded audio              -> expected offset_s = +1.06, output zoomed in
sim_noaudio output = SRC from 2.00 s, no audio, 24 fps      -> motion fallback, expected offset_s = +2.00
sim_http    sim_shift's files served over HTTP (direct .mp4 URLs)  [only with --http-base]
sim_other   output = an unrelated clip                      -> must not pass
sim_missing input is a missing local file                   -> error, run continues
"""
import argparse
import csv
import subprocess
from pathlib import Path

EXPECTED = {"sim_shift": 1.06, "sim_noaudio": 2.00, "sim_http": 1.06}


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("outdir")
    p.add_argument("--other", default=None)
    p.add_argument("--http-base", default=None)
    a = p.parse_args()
    d = Path(a.outdir)
    d.mkdir(parents=True, exist_ok=True)
    src = Path(a.src).expanduser().resolve()
    shift = d / "sim_shift_out.mp4"
    if not shift.exists():
        # trim/atrim in one graph so audio and video share the exact cut (input seeking can shift A vs V)
        ff("-i", src, "-filter_complex",
           "[0:v]trim=start=1.30:duration=7.5,setpts=PTS-STARTPTS,crop=iw*0.8:ih*0.8,scale=576:1024,fps=25,"
           "tpad=start=6:color=black[v];"
           "[0:a]atrim=start=1.30:duration=7.5,asetpts=PTS-STARTPTS,adelay=240|240,volume=0.8[a]",
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "26", "-c:a", "aac", "-b:a", "96k", shift)
    noa = d / "sim_noaudio_out.mp4"
    if not noa.exists():
        ff("-i", src, "-an", "-vf", "trim=start=2.0:duration=7,setpts=PTS-STARTPTS,fps=24,scale=540:960",
           "-c:v", "libx264", "-crf", "24", noa)
    rows = [("sim_shift", str(src), str(shift), "simulated: shifted, cropped, 25 fps"),
            ("sim_noaudio", str(src), str(noa), "simulated: no audio, 24 fps")]
    if a.http_base:
        import shutil
        shutil.copyfile(src, d / "sim_src.mp4")
        rows.append(("sim_http", f"{a.http_base}/sim_src.mp4", f"{a.http_base}/{shift.name}", "direct URL"))
    if a.other:
        oth = d / "sim_other_out.mp4"
        if not oth.exists():
            ff("-ss", "5", "-i", Path(a.other).expanduser(), "-t", "8", "-vf", "scale=480:-2", "-c:v", "libx264",
               "-crf", "24", "-c:a", "aac", oth)
        rows.append(("sim_other", str(src), str(oth), "unrelated output"))
    rows.append(("sim_missing", str(d / "does_not_exist.mp4"), str(shift), "missing input"))
    with open(d / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "input_url", "output_url", "notes"])
        w.writerows(rows)
    print(d / "pairs.csv")


if __name__ == "__main__":
    main()
