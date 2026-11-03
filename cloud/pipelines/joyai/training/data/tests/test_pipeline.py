#!/usr/bin/env python3
"""End-to-end local test (no API key, no GPU): simulated pairs from one real clip, a local HTTP server for the
direct-URL path, a Zyla response-parsing unit test, then build_pairs twice (the second run must be a no-op).

  cloud/.venv/bin/python cloud/pipelines/joyai/training/data/tests/test_pipeline.py [SRC.mp4] [--other OTHER.mp4]
      [--work /tmp/pairs_test] [--keep]

SRC defaults to $PAIRS_TEST_SRC; it must be a clip of a person with audio (>= 9 s).
"""
import argparse
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import build_pairs  # noqa: E402
import fetch  # noqa: E402
from common import probe  # noqa: E402
from make_test_pairs import EXPECTED  # noqa: E402

FRAME = 1 / 24  # default --fps 24


def test_zyla_parse():
    # documented TikTok Media Grabber response shape (tikwm style); wmplay must never be chosen
    sample = {"code": 0, "msg": "success", "processed_time": 0.1, "data": {
        "id": "7237370304337628442", "title": "x", "cover": "https://x/cover.jpeg",
        "play": "https://v16m.tiktokcdn.com/abc/play.mp4", "wmplay": "https://v16m.tiktokcdn.com/abc/wm.mp4",
        "hdplay": "https://v16m.tiktokcdn.com/abc/7237370304337628442_original.mp4",
        "music": "https://sf16.tiktokcdn.com/obj/audio_mpeg.mp3",
        "music_info": {"play": "https://sf16.tiktokcdn.com/obj/music.mp3"}, "size": 1, "wm_size": 2, "hd_size": 3}}
    assert fetch.pick_video_url(sample) == (sample["data"]["hdplay"], "hdplay")
    del sample["data"]["hdplay"]
    assert fetch.pick_video_url(sample)[0] == sample["data"]["play"]
    assert fetch.pick_video_url({"code": 0, "data": {"play": "/video/media/play/123.mp4"}})[0].startswith("https://")
    assert fetch.pick_video_url({"data": {"wmplay": "https://x/wm.mp4"}}) == (None, None)
    try:
        fetch.pick_video_url({"code": -1, "msg": "Url parsing is failed"})
        raise AssertionError("expected FetchError")
    except fetch.FetchError:
        pass
    assert fetch.is_tiktok_page("https://www.tiktok.com/@user/video/7237370304337628442")
    assert fetch.is_tiktok_page("https://vm.tiktok.com/ZMabc/")
    assert not fetch.is_tiktok_page("https://v16m.tiktokcdn.com/x/y.mp4")
    print("ok zyla parse")


def serve(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", nargs="?", default=os.environ.get("PAIRS_TEST_SRC"))
    p.add_argument("--other", default=None, help="an unrelated clip with audio (adds a must-fail pair)")
    p.add_argument("--work", default="/tmp/pairs_test")
    p.add_argument("--keep", action="store_true")
    a = p.parse_args()
    if not a.src:
        sys.exit("give SRC.mp4 or set PAIRS_TEST_SRC")
    test_zyla_parse()

    work = Path(a.work)
    if work.exists() and not a.keep:
        shutil.rmtree(work)
    sim = work / "sim"
    sim.mkdir(parents=True, exist_ok=True)
    srv = serve(sim)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    cmd = [sys.executable, str(HERE / "make_test_pairs.py"), a.src, str(sim), "--http-base", base]
    if a.other:
        cmd += ["--other", a.other]
    subprocess.run(cmd, check=True)
    csv_path = sim / "pairs.csv"
    out, raw = work / "out", work / "raw"
    args = [str(csv_path), "--raw", str(raw), "--out", str(out), "--include-warn", "--match-framing",
            "--frames", "49", "--stride", "49"]

    t = time.time()
    assert build_pairs.main(args) == 0
    t1 = time.time() - t
    an = {d.name: json.loads((d / "analysis.json").read_text()) for d in (out / "pairs").iterdir()}

    for pid, exp in EXPECTED.items():
        al = an[pid]["align"]
        err = abs(al["offset_s"] - exp)
        print(f"{pid}: offset {al['offset_s']:+.4f} (expected {exp:+.2f}, err {err * 1000:.1f} ms) "
              f"method {al['method']} ratio {al['ratio']} status {an[pid]['status']} "
              f"sync {an[pid]['vision']['lipgap_corr']} lag {an[pid]['vision']['lipgap_best_lag']} "
              f"framing {an[pid]['vision']['framing']} crop {al.get('extra_crop')}")
        assert err < FRAME, f"{pid}: offset error {err:.3f}s"
        assert an[pid]["status"] in ("pass", "warn"), an[pid]["gates"]
        assert an[pid]["vision"]["lipgap_corr"] and an[pid]["vision"]["lipgap_corr"] > 0.6
        assert an[pid]["n_clips"] >= 1
    assert an["sim_shift"]["align"]["method"].startswith("audio")
    assert an["sim_noaudio"]["align"]["method"] == "motion"
    assert an["sim_shift"]["align"]["extra_crop"], "output is zoomed 1.25x; --match-framing should crop the input"
    assert an["sim_http"]["fetch"]["input"]["method"] == "direct"
    assert an["sim_missing"]["status"] == "error"
    if "sim_other" in an:
        assert an["sim_other"]["status"] == "fail", an["sim_other"]["gates"]

    # clips: exact frame count, size, fps; manifest consistent
    lines = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(lines) == sum(x.get("n_clips", 0) for x in an.values()) and lines
    for e in lines[:4]:
        for side in ("input", "output"):
            pth = out / e[side]
            pr = probe(pth)
            n = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                                    "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(pth)],
                                   capture_output=True, text=True).stdout.strip())
            assert (pr["width"], pr["height"]) == (480, 840) and n == 49 and abs(pr["fps"] - 24) < 0.01, (pth, pr, n)
    # review: preview-safe video + contact sheet + summary
    for pid in EXPECTED:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=profile,level,width", "-of", "json", str(out / "pairs" / pid / "review.mp4")],
                           capture_output=True, text=True)
        st = json.loads(r.stdout)["streams"][0]
        assert st["profile"] == "High" and st["level"] <= 41 and st["width"] <= 1920, st
        assert (out / "pairs" / pid / "review.jpg").exists()
    assert (out / "summary.csv").exists() and (out / "summary.md").exists()

    # JoyAI shards (PLAN.md section 3): key.0.mp4 / key.1.mp4 / key.0.png / key.json, manifests, mouth masks
    import tarfile
    import numpy as np
    subprocess.run([sys.executable, str(HERE.parent / "to_joyai_shards.py"), "--processed", str(out),
                    "--val-drivers", "sim_noaudio", "--mouth-mask", "--per-shard", "5"], check=True)
    sh = out / "shards"
    for split in ("train", "val"):
        man = json.loads((sh / f"rv2v_{split}.json").read_text())
        names = [n for f in man["resolved_files"] for n in tarfile.open(f).getnames()]
        keys = {n.split(".")[0] for n in names}
        assert len(keys) == man["num_samples"] and len(names) == 4 * len(keys), (split, names[:8])
        assert all(not k.count(".") for k in keys)
        masks = np.load(sh / f"mouth_mask_{split}.npz")
        assert set(masks) == keys and all(masks[k].shape == (7, 35, 20) and masks[k].any() for k in keys)
    with tarfile.open(json.loads((sh / "rv2v_train.json").read_text())["resolved_files"][0]) as t:
        js = [n for n in t.getnames() if n.endswith(".json")][0]
        meta = json.loads(t.extractfile(js).read())
        assert meta["caption"] and meta["metadata"]["ref_image"] == "char_front.png", meta
    print("ok shards")

    # resume: second run must not redo anything
    t = time.time()
    log_before = (out / "build.log").read_text()
    assert build_pairs.main(args) == 0
    t2 = time.time() - t
    new_log = (out / "build.log").read_text()[len(log_before):]
    assert "aligning" not in new_log and "vision (" not in new_log and "cutting clips" not in new_log, new_log
    print(f"first run {t1:.1f}s, resumed run {t2:.1f}s")
    print("ALL OK  ->", out / "summary.md")
    srv.shutdown()


if __name__ == "__main__":
    main()
