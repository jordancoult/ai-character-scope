"""Offline test of the streaming engine (on the GPU box): timing per chunk + demo videos for every input mode.

  /root/ll2/bin/python ll2_stream_test.py OUTDIR [--ckpt S4 --sage --block 8] [--only t2v,i2v,v2v,person]
Writes OUTDIR/<test>.mp4 (and <test>_sbs.mp4 = input | output for v2v) and prints per-chunk ms / fps.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--ckpt", default="S4")
p.add_argument("--sage", action="store_true")
p.add_argument("--kvq", action="store_true")
p.add_argument("--block", type=int, default=8)
p.add_argument("--height", type=int, default=576)
p.add_argument("--width", type=int, default=1024)
p.add_argument("--only", default="t2v,i2v,v2v,person")
p.add_argument("--image", default=None)
p.add_argument("--video", default=None)
p.add_argument("--mask-video", default=None, help="person masks for the 'person' test (white = person)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
from ll2_stream import LL2Engine  # noqa: E402

eng = LL2Engine(height=a.height, width=a.width, ckpt=a.ckpt, kv_quant=a.kvq, block=a.block, sage=a.sage)
FPC = 4 * a.block


def write(path, frames, fps=24):
    import subprocess
    h, w = frames[0].shape[:2]
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(fps), "-i", "-", "-vf", "scale='min(1920,iw)':-2", "-c:v", "libx264", "-crf", "18",
                           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-movflags", "+faststart", path],
                          stdin=subprocess.PIPE)
    for f in frames:
        pr.stdin.write(np.ascontiguousarray(f).tobytes())
    pr.stdin.close()
    pr.wait()


def label(f, text):
    f = f.copy()
    cv2.putText(f, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(f, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return f


def read_video(path, n=None):
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok or (n and len(fr) >= n):
            return fr
        h, w = f.shape[:2]
        s = max(a.width / w, a.height / h)
        f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        y0, x0 = (f.shape[0] - a.height) // 2, (f.shape[1] - a.width) // 2
        fr.append(cv2.cvtColor(f[y0:y0 + a.height, x0:x0 + a.width], cv2.COLOR_BGR2RGB))


def run(name, schedule, n_chunks, inputs=None, masks=None, strength=0.7, image=None, first_prompt=None):
    """schedule: {chunk_index: (prompt, cut)}; inputs: list of RGB frames (v2v)."""
    eng.reset(prompt=first_prompt or schedule[0][0], image=image)
    out, sbs, times = [], [], []
    shown = schedule[0][0]
    for c in range(n_chunks):
        if c in schedule and c > 0:
            eng.set_prompt(*schedule[c])
            shown = schedule[c][0]
        kw = {}
        if inputs is not None:
            fr = [inputs[(c * FPC + i) % len(inputs)] for i in range(FPC)]
            kw = {"input_frames": fr, "strength": strength}
            if masks is not None:
                mk = masks[(c * FPC + FPC - 1) % len(masks)]
                m = cv2.resize(mk[..., 0], (a.width // 16, a.height // 16), interpolation=cv2.INTER_AREA) / 255.0
                kw["strength_map"] = np.where(m > 0.5, float(os.environ.get("LL2_PERSON_KEEP", "0.6")), strength)
        t = time.perf_counter()
        frames = eng.step(**kw)
        times.append(time.perf_counter() - t)
        if eng.stats.get("recache_ms") or eng.stats.get("cut"):
            print(f"[test] {name} chunk {c}: {'cut' if eng.stats.get('cut') else 'recache ' + str(round(eng.stats['recache_ms'])) + ' ms'}, "
                  f"chunk {1000 * times[-1]:.0f} ms", flush=True)
        out += [label(f, shown[:70]) for f in frames]
        if inputs is not None:
            sbs += [np.hstack([fr[i], label(f, shown[:60])]) for i, f in enumerate(frames)]
    t = np.array(times[2:])
    fps = FPC / t.mean()
    print(f"[test] {name}: {n_chunks} chunks, {1000 * t.mean():.0f} ms/chunk (min {1000 * t.min():.0f}, max "
          f"{1000 * t.max():.0f}) = {fps:.1f} fps, chunk = {FPC} frames = {FPC / 24:.2f} s", flush=True)
    write(os.path.join(a.out, f"{name}.mp4"), out)
    if sbs:
        write(os.path.join(a.out, f"{name}_sbs.mp4"), sbs)
    return fps


only = a.only.split(",")
n30 = int(30 * 24 / FPC)  # ~30 s
if "t2v" in only:
    sched = {0: ("a cinematic shot of a cozy wooden cabin interior, a fireplace, warm evening light", False),
             n30 // 4: ("the same cabin interior, snow starts falling outside the window", False),
             n30 // 2: ("the camera slowly moves forward toward the window", False),
             3 * n30 // 4: ("a sunny tropical beach with palm trees, turquoise water", True)}
    run("t2v_prompts", sched, n30)
if "i2v" in only and a.image:
    img = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
    run("i2v_person", {0: ("a young woman with long blonde hair smiles and talks to the camera, natural light", False),
                       n30 // 2: ("the young woman laughs and waves at the camera", False)}, n30, image=img)
if ("v2v" in only or "person" in only) and a.video:
    vid = read_video(a.video, 24 * 20)
    if "v2v" in only:
        for s in [float(x) for x in os.environ.get("LL2_V2V_STRENGTHS", "0.9,0.95,0.98").split(",")]:
            run(f"v2v_s{int(s * 100)}", {0: ("a cozy cabin at night, warm firelight, a man in a knitted sweater", False)},
                int(15 * 24 / FPC), inputs=vid, strength=s)
    if "person" in only and a.mask_video:
        masks = [cv2.resize(m, (a.width, a.height)) for m in read_video(a.mask_video, 24 * 20)]
        run("person_keep", {0: ("a man standing in a lush jungle next to a waterfall, sunlight", False)},
            int(15 * 24 / FPC), inputs=vid, masks=masks, strength=float(os.environ.get("LL2_PERSON_BG", "0.98")))
print("[test] DONE", flush=True)
