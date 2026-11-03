"""Max throughput of the LP pass with the 3 stages overlapped (driver analysis | G-Star analysis | warp+paste),
frames back to back (no 25 fps cap), on a recorded JoyAI run. No video writing."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_offline import read
from lp_pass import LPPass
E = Path(__file__).resolve().parents[1] / "pipelines/joyai/experiments/2026-11-02-mouth"
outs, ins = read(E / "G_gstar.mp4"), read(E / "in_G_gstar.mp4")
seq = json.load(open(E / "G_gstar.json"))["source_seq"][:len(outs)]
lp = LPPass(); drv = lp.driver()
pick = lambda s: ins[min(len(ins) - 1, max(0, s - 1))]
for mode in ("serial", "pipelined"):
    lp.src.lmk = drv.face.lmk = None
    t0 = time.perf_counter()
    if mode == "serial":
        for o, s in zip(outs, seq):
            lp.render(o, drv.analyze(pick(s)))
    else:
        pd, ps = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
        fd, fs = pd.submit(drv.analyze, pick(seq[0])), ps.submit(lp.analyze, outs[0])
        for k in range(len(outs)):
            d, s = fd.result(), fs.result()
            if k + 1 < len(outs):
                fd, fs = pd.submit(drv.analyze, pick(seq[k + 1])), ps.submit(lp.analyze, outs[k + 1])
            lp.render(outs[k], d, s=s)
    dt = time.perf_counter() - t0
    print(f"{mode}: {len(outs) / dt:.1f} fps ({dt / len(outs) * 1000:.1f} ms/frame)")
