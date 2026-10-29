"""Benchmark / profile the LongLive pipeline directly (no server), same load path as the bridge.

Usage: python tools/bench_longlive.py [--steps 750 400] [--profile] [--block-timing] [--out out.mp4]
Env flags (SCOPE_FP8_ROWWISE, SCOPE_NOISE_T_OFFSET, ...) behave as in start_scope.sh.
"""

import argparse
import collections
import json
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="bench/webcam/input.mp4")
parser.add_argument("--size", type=int, nargs=2, default=[576, 320], metavar=("H", "W"))
parser.add_argument("--steps", type=int, nargs="+", default=[750, 400])
parser.add_argument("--noise-scale", type=float, default=0.9)
parser.add_argument("--vae", default="lighttae")
parser.add_argument("--ref", default="assets/blonde-gamer-girl.png")
parser.add_argument("--prompt", default="a young woman with long blonde hair and a white t-shirt, photorealistic")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--chunks", type=int, default=30, help="Timed chunks after warm-up")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--block-timing", action="store_true", help="GPU time per pipeline block (syncs)")
parser.add_argument("--out", default=None)
parser.add_argument("--tag", default="")
args = parser.parse_args()

from scope.core.pipelines.video import load_video  # noqa: E402
from scope.server.pipeline_manager import PipelineManager  # noqa: E402

h, w = args.size
t0 = time.time()
pipe = PipelineManager()._load_pipeline_implementation(
    "longlive", {"height": h, "width": w, "vae_type": args.vae, "vace_enabled": True, "base_seed": args.seed}
)
load_s = time.time() - t0

video = load_video(args.video, resize_hw=(h, w)).unsqueeze(0).to("cuda", torch.bfloat16)
req = pipe.prepare(video=True)
chunk = req.input_size if req else 12
prompts = [{"text": args.prompt, "weight": 100}]

block_gpu = collections.defaultdict(float)
if args.block_timing:
    for name, block in pipe.blocks.sub_blocks.items():
        orig = type(block).__call__

        def timed(self, components, state, _orig=orig, _name=name):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = _orig(self, components, state)
            torch.cuda.synchronize()
            block_gpu[_name] += time.perf_counter() - t
            return out

        type(block).__call__ = timed

writer = None
if args.out:
    import subprocess

    writer = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "16",
                               "-preset", "veryfast", "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)


def run(i, first=False):
    s = i * chunk % (video.shape[2] - chunk)
    kw = {"video": video[:, :, s:s + chunk], "prompts": prompts, "denoising_step_list": args.steps,
          "noise_scale": args.noise_scale, "noise_controller": False, "vace_use_input_video": False}
    if first and args.ref:
        kw["vace_ref_images"] = [args.ref]
    out = pipe(**kw)["video"]
    if writer is not None:
        u8 = (out.float().clamp(0, 1) * 255).to(torch.uint8)
        writer.stdin.write(u8.contiguous().cpu().numpy().tobytes())
    return out.shape[0]


warm = 4
for i in range(warm):
    run(i, first=i == 0)
block_gpu.clear()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
times, frames = [], 0
for i in range(warm, warm + args.chunks):
    t = time.perf_counter()
    frames += run(i)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t)

if writer is not None:
    writer.stdin.close()
    writer.wait()

total = sum(times)
res = {"tag": args.tag, "pipeline": "longlive", "steps": args.steps, "chunk_frames": chunk,
       "fps": round(frames / total, 2), "chunk_ms": round(1000 * total / len(times), 1),
       "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "load_s": round(load_s, 1)}
print("RESULT " + json.dumps(res))
if block_gpu:
    for k, v in sorted(block_gpu.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {1000 * v / len(times):7.1f} ms/chunk")

if args.profile:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(3):
            run(100 + i)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
