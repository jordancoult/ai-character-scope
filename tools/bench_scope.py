"""Benchmark Scope's StreamDiffusionV2 pipeline directly on the GPU.

Run with Scope's venv after `source env.sh`, e.g.:
    scope/.venv/bin/python tools/bench_scope.py --vae tae --steps 750 250 --size 512 512
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.enums import Quantization
from scope.core.pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline
from scope.core.pipelines.video import load_video

PIPE_DIR = Path(StreamDiffusionV2Pipeline.__module__.replace(".", "/")).parent
ASSET = Path(__file__).resolve().parents[1] / "scope/src" / PIPE_DIR / "assets/original.mp4"

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
parser.add_argument("--vae", default="tae", choices=["wan", "lightvae", "tae", "lighttae"])
parser.add_argument("--steps", type=int, nargs="+", default=[750, 250])
parser.add_argument("--fp8", action="store_true")
parser.add_argument("--vace-ref", nargs="+", default=None, help="Reference image path(s) (enables VACE)")
parser.add_argument("--chunk", type=int, default=4, help="Input frames per call")
parser.add_argument("--max-seconds", type=float, default=None, help="Single pass over the first N seconds; --out saves every frame")
parser.add_argument("--video", default=str(ASSET))
parser.add_argument("--out", default=None, help="Optional mp4 path for the output")
parser.add_argument("--tag", default="")
parser.add_argument("--seed", type=int, default=42, help="base_seed (noise)")
parser.add_argument("--prompt", default="a cel-shaded 3D cartoon bear walking on the grass, vibrant colors")
parser.add_argument("--compile", default="none", choices=["none", "blocks", "blocks-noattn"],
                    help="torch.compile each DiT/VACE transformer block (like Scope's Krea pipeline)")
parser.add_argument("--compile-mode", default="default",
                    help="torch.compile mode: default | max-autotune-no-cudagraphs | reduce-overhead")
parser.add_argument("--fp4-ffn", action="store_true", help="NVFP4 FFN linears in DiT blocks (applied before fp8)")
parser.add_argument("--fp4-skip", type=int, default=0, help="Keep the first/last N DiT blocks out of FP4")
parser.add_argument("--fp4-layers", default="0,2", help="FFN sublayer indices to convert (0=up, 2=down)")
parser.add_argument("--fp8-rowwise", action="store_true", help="Convert DiT/VACE linears to fp8 row-wise _scaled_mm")
parser.add_argument("--noise-scale", type=float, default=None)
parser.add_argument("--depth", action="store_true", help="With --vace-input-video: convert input to GPU depth maps first")
parser.add_argument("--depth-size", type=int, default=504, help="Depth model input long side (multiple of 14)")
parser.add_argument("--also-latent-init", action="store_true", help="With --vace-input-video: also pass the RGB input as latent init")
parser.add_argument("--vace-input-video", action="store_true", help="Feed the input as VACE structure conditioning (vace_input_frames) instead of latent init")
parser.add_argument("--noise-controller", choices=["on", "off"], default=None)
parser.add_argument("--vace-scale", type=float, default=1.0)
parser.add_argument("--cudnn-benchmark", action="store_true")
parser.add_argument("--kv-frames", type=int, default=None, help="Override local_attn_size (KV cache window, frames)")
parser.add_argument("--tae-compile", action="store_true", help="torch.compile the TAE streaming encode/decode")
parser.add_argument("--block-timing", action="store_true", help="CPU time per pipeline block (no syncs)")
parser.add_argument("--sync-debug", action="store_true", help="Report where CPU<->GPU syncs happen during 2 steady chunks")
parser.add_argument("--profile", action="store_true", help="Print torch profiler table for 3 steady-state chunks")
args = parser.parse_args()

h, w = args.size
config = OmegaConf.create(
    {
        "model_dir": str(get_models_dir()),
        "generator_path": str(get_model_file_path("StreamDiffusionV2/wan_causal_dmd_v2v/model.pt")),
        "text_encoder_path": str(get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")),
        "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
        "vace_path": str(get_model_file_path("WanVideo_comfy/Wan2_1-VACE_module_1_3B_bf16.safetensors"))
        if args.vace_ref or args.vace_input_video
        else None,
        "model_config": OmegaConf.load(ASSET.parents[1] / "model.yaml"),
        "height": h,
        "width": w,
        "vae_type": args.vae,
        "base_seed": args.seed,
    }
)

torch.backends.cudnn.benchmark = args.cudnn_benchmark
if args.kv_frames:
    config.model_config.local_attn_size = args.kv_frames
t0 = time.time()
pipeline = StreamDiffusionV2Pipeline(
    config,
    quantization=Quantization.FP8_E4M3FN if args.fp8 else None,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)
load_s = time.time() - t0

if args.fp4_ffn:
    from scope.core.pipelines.wan2_1.fp4_linear import convert_ffn_to_fp4

    print(f"fp4: converted {convert_ffn_to_fp4(pipeline.components.generator.model, skip_first=args.fp4_skip, skip_last=args.fp4_skip, layers=tuple(int(i) for i in args.fp4_layers.split(",")))} FFN linears")

if args.fp8_rowwise:
    from scope.core.pipelines.wan2_1.fp8_rowwise import convert_blocks_to_fp8_rowwise

    n = convert_blocks_to_fp8_rowwise(pipeline.components.generator.model)
    torch.cuda.empty_cache()
    print(f"fp8 rowwise: converted {n} linears")

if args.tae_compile:
    import scope.core.pipelines.wan2_1.vae.tae as tae_mod

    tae_mod._apply_model_parallel_streaming = torch.compile(
        tae_mod._apply_model_parallel_streaming, dynamic=False, mode=args.compile_mode
    )
    print("TAE streaming apply compiled")

if args.compile != "none":
    import torch._dynamo

    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 4096
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    if args.compile == "blocks-noattn":
        from scope.core.pipelines.streamdiffusionv2.modules import causal_model

        attn_cls = causal_model.CausalWanSelfAttention
        attn_cls.forward = torch.compiler.disable(attn_cls.forward)
    model = pipeline.components.generator.model
    compiled = 0
    for m in model.modules():
        if type(m).__name__.endswith("AttentionBlock"):
            m.compile(fullgraph=False, dynamic=None if args.compile == "blocks-noattn" else False, mode=args.compile_mode)
            compiled += 1
    print(f"compiled {compiled} blocks (mode={args.compile_mode})")

video = load_video(args.video, resize_hw=(h, w)).unsqueeze(0).to("cuda", torch.bfloat16)
num_frames = video.shape[2]
prompts = [{"text": args.prompt, "weight": 100}]

block_cpu = {}
if args.block_timing:
    import collections

    block_cpu = collections.defaultdict(float)
    for name, block in pipeline.blocks.sub_blocks.items():
        orig = block.__call__

        def timed(components, state, _orig=orig, _name=name):
            t = time.perf_counter()
            out = _orig(components, state)
            block_cpu[_name] += time.perf_counter() - t
            return out

        block.__call__ = timed
        type(block).__call__ = lambda self, components, state: self.__dict__["__call__"](components, state) if "__call__" in self.__dict__ else None

depth_model = None
depth_frames = []
if args.depth:
    from scope.core.pipelines.video_depth_anything.gpu_depth import GpuStreamingDepth

    depth_model = GpuStreamingDepth(input_long_side=args.depth_size)

torch.cuda.reset_peak_memory_stats()
outputs, timings = [], []
start_idx, end_idx = 0, args.chunk + 1
first = True
writer = None
if args.out and args.max_seconds:
    import subprocess

    writer = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "16",
                               "-preset", "veryfast", "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)
    num_frames = min(num_frames, int(args.max_seconds * 30))
# Loop the clip so we measure a few hundred frames of steady state (single pass with --max-seconds).
for _ in range(1 if args.max_seconds else 3):
    start_idx, end_idx = 0, args.chunk + 1 if first else args.chunk
    while end_idx <= num_frames:
        chunk = video[:, :, start_idx:end_idx]
        kwargs = {"prompts": prompts, "denoising_step_list": args.steps}
        if args.vace_input_video:
            if depth_model is not None:
                chunk = depth_model(chunk)
                if args.out and len(depth_frames) < 40:
                    depth_frames.append(((chunk[0].permute(1, 2, 3, 0).float() + 1) / 2).cpu())
            kwargs["vace_input_frames"] = chunk
            if args.also_latent_init:
                kwargs["video"] = video[:, :, start_idx:end_idx]
        else:
            kwargs["video"] = chunk
        if args.noise_scale is not None:
            kwargs["noise_scale"] = args.noise_scale
        if args.noise_controller is not None:
            kwargs["noise_controller"] = args.noise_controller == "on"
        if first and args.vace_ref:
            kwargs["vace_ref_images"] = list(args.vace_ref)
            kwargs["vace_context_scale"] = args.vace_scale
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = pipeline(**kwargs)["video"]
        torch.cuda.synchronize()
        timings.append((out.shape[0], time.perf_counter() - t))
        if writer is not None:
            frames_u8 = (out.float().clamp(0, 1) * 255).to(torch.uint8) if out.dtype != torch.uint8 else out
            writer.stdin.write(frames_u8.contiguous().cpu().numpy().tobytes())
        elif args.out and len(outputs) < 40:
            outputs.append(out.detach().cpu())
        first = False
        start_idx, end_idx = end_idx, end_idx + args.chunk

if args.profile:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                 with_stack=bool(os.environ.get("PROFILE_STACK"))) as prof:
        for i in range(3):
            s = 1 + args.chunk * (i + 1)
            pipeline(video=video[:, :, s:s + args.chunk], prompts=prompts, denoising_step_list=args.steps)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    if os.environ.get("PROFILE_STACK"):
        print("TOP BY STACK")
        print(prof.key_averages(group_by_stack_n=6).table(sort_by="self_cuda_time_total", row_limit=12))
    print("TOP OPS BY INPUT SHAPE")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=40))

if args.sync_debug:
    import collections
    import traceback
    import warnings

    sites = collections.Counter()

    def record(message, category, filename, lineno, file=None, line=None):
        frames = [f for f in traceback.extract_stack()[:-1] if "/scope/" in f.filename and "site-packages" not in f.filename]
        key = " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in frames[-3:][::-1])
        sites[key] += 1

    warnings.showwarning = record
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    for i in range(2):
        s0 = 1 + args.chunk * (i + 1)
        pipeline(video=video[:, :, s0:s0 + args.chunk], prompts=prompts, denoising_step_list=args.steps)
    torch.cuda.set_sync_debug_mode(0)
    print("SYNC SITES (2 chunks):")
    for k, n in sites.most_common(25):
        print(f"  {n:4d}  {k}")

steady = timings[max(3, len(timings) // 4):]  # drop warmup (incl. torch.compile recompiles)
frames = sum(n for n, _ in steady)
secs = sum(s for _, s in steady)
result = {
    "tag": args.tag,
    "size": f"{h}x{w}",
    "vae": args.vae,
    "steps": args.steps,
    "fp8": (f"rowwise+fp4ffn(skip{args.fp4_skip})" if args.fp4_ffn else "rowwise") if args.fp8_rowwise else args.fp8,
    "vace": ("depth+ref" if args.depth else "input+ref") if args.vace_input_video else bool(args.vace_ref),
    "noise": args.noise_scale,
    "vace_scale": args.vace_scale,
    "kv_frames": args.kv_frames or 6,
    "fixed_noise": os.environ.get("SCOPE_FIXED_NOISE") == "1",
    "fps": round(frames / secs, 2),
    "chunk_latency_ms": round(1000 * secs / len(steady), 1),
    "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    "load_s": round(load_s, 1),
    "frames_measured": frames,
}
print("RESULT " + json.dumps(result))
if block_cpu:
    total = sum(block_cpu.values())
    print(f"BLOCK CPU TIME per chunk (no sync; {len(timings)} chunks):")
    for k, v in sorted(block_cpu.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {1000 * v / len(timings):7.2f} ms")
    print(f"  {'(sum)':24s} {1000 * total / len(timings):7.2f} ms  vs wall {result['chunk_latency_ms']} ms")

if writer is not None:
    writer.stdin.close()
    writer.wait()
elif args.out:
    from diffusers.utils import export_to_video

    import av

    with av.open(args.video) as src:
        src_fps = float(src.streams.video[0].average_rate or 16)
    export_to_video(torch.concat(outputs).contiguous().numpy(), args.out, fps=round(src_fps))
    if depth_frames:
        export_to_video(torch.concat(depth_frames).contiguous().numpy(), args.out.replace(".mp4", "_depth.mp4"), fps=round(src_fps))
