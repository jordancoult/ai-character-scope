"""Headless Scope bridge: camera/file -> Scope (WebRTC) -> 720x1280 virtual camera, with an FPS slider.

Loads Scope's StreamDiffusionV2 at 576x320 (9:16) and streams the input to it over WebRTC. Output
frames are upscaled on the GPU to 720x1280 and written to a v4l2loopback device (OBS: "AI_Scope_Cam")
or an .mp4. A small control page (default http://127.0.0.1:8090) has a 30/24/20/15 FPS slider;
all presets share the resolution, so switching is live (no pipeline reload).

Presets (RTX 5080 Laptop; see README_QUICKSTART.md for measured live FPS):
  30  fast         - camera latents + character ref, 1 step
  24  fast 2-step  - same, 2 denoising steps (cleanest image)
  20  depth        - camera latents + depth->VACE + ref (tightest pose tracking)
  15  depth 2-step

Run with Scope's venv after `source env.sh` (start_scope.sh does this).
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import torch
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from av import VideoFrame

sys.path.insert(0, str(Path(__file__).parent))
from upscalers import Upscaler  # noqa: E402

PRESETS = {
    # Chosen by tools/quality_metrics.py sweeps on a real webcam clip (README: "Quality sweep").
    30: {"label": "30 fps - fast (1 step)", "depth_vace": False, "steps": [750], "fixed_noise": False,
         "upscale": "bicubic-sharp"},
    24: {"label": "~25 fps - clean (2 steps, fixed noise)", "depth_vace": False, "steps": [750, 400],
         "fixed_noise": True, "upscale": "bicubic-sharp"},
    20: {"label": "~16 fps - pose-tracked (depth, 2 steps, fixed noise)", "depth_vace": True, "steps": [750, 400],
         "fixed_noise": True, "upscale": "bicubic-sharp"},
    15: {"label": "~16 fps - pose-tracked, max realism (depth, 2 steps, random noise)", "depth_vace": True,
         "steps": [750, 400], "fixed_noise": False, "upscale": "bicubic-sharp"},
}

parser = argparse.ArgumentParser()
parser.add_argument("--server", default="http://127.0.0.1:8000")
parser.add_argument("--input", default="/dev/video0", help="v4l2 device or video file")
parser.add_argument("--input-size", default="1280x720", help="Capture size for v4l2 devices")
parser.add_argument("--output", default="/dev/video10", help="v4l2loopback device or .mp4 path")
parser.add_argument("--size", type=int, nargs=2, default=[576, 320], metavar=("H", "W"), help="Generation size")
parser.add_argument("--out-size", type=int, nargs=2, default=[1280, 720], metavar=("H", "W"), help="Output size after upscale")
parser.add_argument("--preset", type=int, default=30, choices=sorted(PRESETS))
parser.add_argument("--upscale", default=None, help="Override preset upscaler: none|bicubic|bicubic-sharp|animevideov3|general-x4v3")
parser.add_argument("--prompt", default="a stylized 3D animated character, Pixar style render")
parser.add_argument("--ref", nargs="+", default=None, help="Character reference image(s) (VACE)")
parser.add_argument("--vace-scale", type=float, default=1.0)
parser.add_argument("--noise-scale", type=float, default=None, help="Default 0.75 (0.9 for longlive, which follows the camera more)")
parser.add_argument("--pipeline", default="streamdiffusionv2", choices=["streamdiffusionv2", "longlive"],
                    help="longlive = experimental: trained for long streams (frame sink + KV recache), "
                         "3 latent frames per chunk, no depth presets, no Blackwell speed patches")
parser.add_argument("--vae", default=None, choices=["tae", "lighttae", "wan", "lightvae"],
                    help="Override the VAE (default: tae for streamdiffusionv2, lighttae for longlive)")
parser.add_argument("--seed", type=int, default=None, help="Base noise seed (load time; default: Scope's)")
parser.add_argument("--steps", type=int, nargs="+", default=None, help="Override every preset's denoising steps")
parser.add_argument("--noise-mode", default=None, choices=["auto", "fixed", "random"],
                    help="Per-chunk noise (default fixed; random for longlive, where fixed noise piles up as confetti): auto = the preset's choice, fixed = same noise every chunk, random = new noise")
parser.add_argument("--control-port", type=int, default=8090, help="Slider page port (0 = off)")
parser.add_argument("--latency-probe", action="store_true",
                    help="Measure end-to-end latency using bright flashes in the input (assets/latency_flash.mp4)")
parser.add_argument("--record-input", default=None, help="Also save the cropped camera frames sent to Scope (.mp4)")
parser.add_argument("--duration", type=float, default=0, help="Stop after N seconds (0 = run until Ctrl+C)")
parser.add_argument("--no-load", action="store_true", help="Skip pipeline load (already loaded)")
args = parser.parse_args()
if args.noise_scale is None:
    args.noise_scale = 0.9 if args.pipeline == "longlive" else 0.75
if args.noise_mode is None:
    args.noise_mode = "random" if args.pipeline == "longlive" else "fixed"

H, W = args.size
OH, OW = args.out_size
state = {"preset": args.preset, "fps": 0.0, "upscale": None, "noise": args.noise_scale,
         "noise_mode": args.noise_mode, "gpu_temp": None, "gpu_watts": None, "gpu_thermal_throttle": False,
         "cpu_temp": None,
         # Advanced overrides (None = use the preset): depth on/off, number of denoising steps,
         # the image fed in place of the camera right after a reset (seeds the KV-cache sink).
         "depth": None, "steps": None, "start": None, "vace_scale": args.vace_scale,
         # Attention steering: weight per KV-cache region (1 = stock); attn_on=False sends all 1s
         "attn": {"sink": 1.0, "recent": 1.0, "current": 1.0, "text": 1.0}, "attn_on": True,
         # Node graph extras: upscaler override, camera frames as VACE input
         "upscale_override": None, "vace_input": False,
         # sink_frames None = model default (3); noise_controller = Scope's motion-adaptive noise;
         # transition_steps = chunks to blend a new prompt over; zoom/y_offset = camera framing; sharpen = upscaler
         "sink_frames": None, "noise_controller": False, "transition_steps": 0, "zoom": 1.0, "y_offset": 0.0,
         "sharpen": 0.6,
         "prompt": args.prompt, "ref": [str(Path(r).resolve()) for r in args.ref] if args.ref else []}
ASSETS = Path(__file__).resolve().parents[1] / "assets"
SAVED = Path(__file__).resolve().parents[1] / "saved_settings"
GRAPH_PAGE = Path(__file__).resolve().parent / "graph_page.html"
UPSCALERS = ["bilinear", "bicubic", "bicubic-sharp", "animevideov3", "general-x4v3"]
probe = {"in_edges": [], "in_prev": 0.0, "out_prev": None, "out_base": None, "lat": [], "recv_t": []}


# Denoising timesteps for 1-5 steps (Scope replaces the first with the noise level)
STEP_LISTS = {1: [750], 2: [750, 400], 3: [750, 500, 250], 4: [750, 600, 400, 200], 5: [750, 600, 450, 300, 150]}
# Frames of the start image sent after a reset: covers the sink chunks (3 latent frames = 12 video
# frames on both pipelines) plus camera frames already queued when the reset lands.
START_HOLD_FRAMES = 24
inject = {"img": None, "left": 0}
sent_params: dict = {}  # everything sent to Scope so far, merged (for export)


def load_start_image(name: str):
    """Center-crop + resize an assets/ image to the generation size, as an RGB uint8 array."""
    from PIL import Image

    img = Image.open(ASSETS / Path(name).name).convert("RGB")
    target = W / H
    iw, ih = img.size
    if iw / ih > target:
        cw = round(ih * target)
        img = img.crop(((iw - cw) // 2, 0, (iw - cw) // 2 + cw, ih))
    else:
        ch = round(iw / target)
        img = img.crop((0, (ih - ch) // 2, iw, (ih - ch) // 2 + ch))
    import numpy as np

    return np.asarray(img.resize((W, H), Image.LANCZOS))


def effective_depth(preset: int) -> bool:
    if args.pipeline != "streamdiffusionv2":
        return False
    return PRESETS[preset]["depth_vace"] if state["depth"] is None else state["depth"]


def effective_steps(preset: int) -> list[int]:
    return STEP_LISTS[state["steps"]] if state["steps"] else (args.steps or PRESETS[preset]["steps"])


SINK_MAX = 10 if args.pipeline == "longlive" else 4  # must stay below the KV window (12 / 6 latent frames)
LAUNCH_ENV = ["MODEL", "PRESET", "NOISE_MODE", "CAMERA", "CAMERA_SIZE", "SCOPE_FUSED_KV", "SCOPE_COMPILE_BLOCKS",
              "SCOPE_COMPILE_TAE", "SCOPE_FP8_ROWWISE", "SCOPE_KV_FRAMES", "SCOPE_NOISE_T_OFFSET",
              "SCOPE_OUTPUT_TARGET_FRAMES", "SCOPE_SINK_QUEUE_MAX", "SCOPE_INPUT_QUEUE_MAX", "SCOPE_PRELOAD_DEPTH",
              "SCOPE_RING_KV", "SCOPE_SKIP_CLEAN_KV"]


def launch_settings() -> dict:
    """Settings that only change on restart: start_scope.sh env vars and bridge load arguments."""
    return {"env": {k: os.environ.get(k) for k in LAUNCH_ENV}, "model": args.pipeline, "size": [H, W],
            "vae": args.vae or ("lighttae" if args.pipeline == "longlive" else "tae"), "seed": args.seed,
            "input_size": args.input_size}


def attention_weights() -> dict:
    return dict(state["attn"]) if state["attn_on"] else {"sink": 1.0, "recent": 1.0, "current": 1.0, "text": 1.0}


def fixed_noise(preset: int) -> bool:
    mode = state["noise_mode"]
    return PRESETS[preset].get("fixed_noise", False) if mode == "auto" else mode == "fixed"


def preset_params(preset: int, with_ref: bool) -> dict:
    params = {
        "denoising_step_list": effective_steps(preset),
        # Scope maps noise_scale to the first timestep (1000 * noise - 100)
        "noise_scale": state["noise"],
        "noise_controller": state["noise_controller"],
        "sink_frames": 3 if state["sink_frames"] is None else state["sink_frames"],
        "fixed_noise": fixed_noise(preset),
        # depth works without a ref image (VACE is always loaded)
        "depth_vace": effective_depth(preset),
        "vace_context_scale": state["vace_scale"],
        "attention_weights": attention_weights(),
        "vace_use_input_video": state["vace_input"] and not effective_depth(preset),
        "reset_cache": True,
    }
    if with_ref and state["ref"]:
        params["vace_ref_images"] = state["ref"]
    return params


async def poll_temps() -> None:
    """GPU/CPU temperatures for the control page (same sources as tools/thermal_watchdog.sh)."""
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            out, _ = await proc.communicate()
            t, w, reasons = out.decode().split("\n")[0].split(",")
            state["gpu_temp"], state["gpu_watts"] = int(t), float(w)
            # 0x20 SW thermal slowdown, 0x40 HW thermal slowdown, 0x08 HW slowdown
            state["gpu_thermal_throttle"] = bool(int(reasons, 16) & 0x68)
        except Exception:
            state["gpu_temp"] = state["gpu_watts"] = None
            state["gpu_thermal_throttle"] = False
        # Hottest CPU core (Intel coretemp, TjMax 105 C on the Ultra 9 285H). The package sensor
        # updates ~1/s and jumps 92 <-> 105 while single cores sit at their limit.
        temps = []
        for h in Path("/sys/class/hwmon").glob("hwmon*"):
            try:
                if (h / "name").read_text().strip() == "coretemp":
                    temps += [int(f.read_text()) // 1000 for f in h.glob("temp*_input")]
            except (OSError, ValueError):
                pass
        state["cpu_temp"] = max(temps) if temps else None
        await asyncio.sleep(2)


async def load_pipeline(client: httpx.AsyncClient) -> None:
    # LongLive expects normalized latents: lighttae (tae is raw, only right for our StreamDiffusionV2 patch)
    vae = args.vae or ("lighttae" if args.pipeline == "longlive" else "tae")
    load_params = {"height": H, "width": W, "vae_type": vae, "vace_enabled": True}  # always: depth presets need VACE even without a ref
    if args.seed is not None:
        load_params["base_seed"] = args.seed
    r = await client.post("/api/v1/pipeline/load",
                          json={"pipeline_ids": [args.pipeline], "load_params": load_params})
    r.raise_for_status()
    print(f"[bridge] loading {args.pipeline} {load_params} ...", flush=True)
    while True:
        status = (await client.get("/api/v1/pipeline/status")).json()
        if status.get("status") == "loaded":
            break
        if status.get("status") == "error":
            raise RuntimeError(f"pipeline load failed: {status}")
        await asyncio.sleep(1)
    print("[bridge] pipeline loaded", flush=True)


class CropResizeTrack(MediaStreamTrack):
    """Center-crop the camera to the generation aspect (9:16) and downscale *before* WebRTC.

    Sending full 720p/1080p frames meant software VP8 encode (here) + decode (in Scope) of ~5x
    more pixels than the model uses, which was the main CPU heat source in live mode.
    """

    kind = "video"

    def __init__(self, source, width, height, record=None):
        super().__init__()
        self.source, self.width, self.height = source, width, height
        self.recorder = None
        if record:
            self.recorder = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                 "-s", f"{width}x{height}", "-r", "30", "-i", "-", "-c:v", "libx264",
                 "-preset", "ultrafast", "-pix_fmt", "yuv420p", record],
                stdin=subprocess.PIPE,
            )

    async def recv(self):
        frame = await self.source.recv()
        q = getattr(self.source, "_queue", None)  # aiortc MediaPlayer's internal (unbounded) queue
        if q is not None and args.latency_probe:
            self._n = getattr(self, "_n", 0) + 1
            if self._n % 150 == 0:
                print(f"[camera] frames waiting in player queue: {q.qsize()}", flush=True)
        fw, fh = frame.width, frame.height
        target = self.width / self.height
        zoom, yoff = state["zoom"], state["y_offset"]
        if abs(fw / fh - target) > 0.01 or zoom != 1.0 or yoff != 0.0:
            arr = frame.to_ndarray(format="rgb24")
            if fw / fh > target:  # too wide: crop width
                cw = round(fh * target)
                x0 = (fw - cw) // 2
                arr = arr[:, x0 : x0 + cw]
            else:  # too tall: crop height
                ch = round(fw / target)
                y0 = (fh - ch) // 2
                arr = arr[y0 : y0 + ch]
            if zoom != 1.0 or yoff != 0.0:  # live framing: zoom into the center, shift up/down
                ah, aw = arr.shape[:2]
                zh, zw = round(ah / zoom), round(aw / zoom)
                y0 = min(max(0, (ah - zh) // 2 + round(yoff * (ah - zh) / 2)), ah - zh)
                x0 = (aw - zw) // 2
                arr = arr[y0 : y0 + zh, x0 : x0 + zw]
            new = VideoFrame.from_ndarray(arr, format="rgb24")
        else:
            new = frame
        new = new.reformat(width=self.width, height=self.height, format="yuv420p")
        if inject["left"] > 0 and inject["img"] is not None:
            inject["left"] -= 1
            new = VideoFrame.from_ndarray(inject["img"], format="rgb24").reformat(format="yuv420p")
        new.pts, new.time_base = frame.pts, frame.time_base
        if args.latency_probe:
            luma = float(new.to_ndarray(format="gray")[::8, ::8].mean())
            if luma > 200 and probe["in_prev"] <= 200:
                probe["in_edges"].append(time.perf_counter())
            probe["in_prev"] = luma
        if self.recorder:
            self.recorder.stdin.write(new.to_ndarray(format="yuv420p").tobytes())  # unpadded planes
        return new


def open_output():
    if args.output.startswith("/dev/video"):
        target = ["-f", "v4l2", args.output]
    else:
        target = ["-c:v", "libx264", "-preset", "veryfast", args.output]
    # Frames arrive already converted to yuv420p on the GPU, so ffmpeg just copies bytes.
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
         "-s", f"{OW}x{OH}", "-r", "30", "-i", "-", "-pix_fmt", "yuv420p", *target],
        stdin=subprocess.PIPE,
    )


def rgb_to_yuv420p(x: torch.Tensor) -> bytes:
    """x: [3, H, W] float in [0, 1] on GPU -> BT.601 limited-range yuv420p bytes."""
    r, g, b = x[0], x[1], x[2]
    y = 16 + 65.481 * r + 128.553 * g + 24.966 * b
    u = 128 - 37.797 * r - 74.203 * g + 112.0 * b
    v = 128 + 112.0 * r - 93.786 * g - 18.214 * b
    uv = torch.stack([u, v]).unsqueeze(0)
    uv = torch.nn.functional.avg_pool2d(uv, 2)[0]
    planes = torch.cat([y.flatten(), uv[0].flatten(), uv[1].flatten()])
    return planes.round().clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes()


CONTROL_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Character Controls</title><style>
:root{--bg:#111;--fg:#eee;--muted:#999;--accent:#ff4d6d;--field:#1c1c1c}
body{background:var(--bg);color:var(--fg);font:16px system-ui,sans-serif;margin:0;padding:24px 16px;max-width:560px}
h1{font-size:20px;margin:0 0 16px} h2{font-size:15px;margin:22px 0 6px;color:var(--muted);font-weight:500}
input[type=range]{width:100%;accent-color:var(--accent)} .ticks{display:flex;justify-content:space-between;color:var(--muted);font-size:13px}
.big{font-size:40px;font-weight:600;margin:12px 0 2px}
.gauges{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:10px 0 4px} .g b{font-size:20px} .g small{color:var(--muted);font-size:13px}
.bar{position:relative;height:10px;border-radius:5px;margin:6px 0 3px;display:flex;overflow:visible}
.bar i{display:block;height:100%} .bar i:first-child{border-radius:5px 0 0 5px} .bar i:last-child{border-radius:0 5px 5px 0}
.mk{position:absolute;top:-4px;width:4px;height:18px;margin-left:-2px;background:var(--fg);border-radius:2px;box-shadow:0 0 0 2px var(--bg)}
.z0{color:#3fbf6f} .z1{color:#f0c03c} .z2{color:#ff4d4d} .muted{color:var(--muted);font-size:14px}
.att{margin-top:10px} .att span{font-size:14px} .att b{color:var(--accent)}
details{margin-top:18px;border:1px solid #333;border-radius:8px;padding:4px 14px 14px} summary{cursor:pointer;font-weight:600;padding:8px 0;color:var(--muted)}
textarea,select{width:100%;box-sizing:border-box;background:var(--field);color:var(--fg);border:1px solid #333;border-radius:6px;padding:8px;font:14px system-ui}
button{margin-top:8px;background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;font:600 14px system-ui;cursor:pointer}
</style></head><body><h1>AI character - live controls</h1>
<div class="big"><span id="fps">-</span> fps</div>
<div class="gauges"><div class="g" id="gg"></div><div class="g" id="cg"></div></div>
<div class="muted">Temperature zones: <span class="z0">efficient</span> &middot; <span class="z1">throttling (slows itself down, safe)</span> &middot; <span class="z2">damaging (avoid)</span></div><div class="muted" id="label"></div>
<h2>Quality vs FPS</h2>
<input id="s" type="range" min="0" max="3" step="1"><div class="ticks"><span>15</span><span>20</span><span>24</span><span>30</span></div>
<h2>Noise: <span id="nv"></span> &nbsp;<span class="muted">lower = follows you more, higher = more character</span></h2>
<input id="n" type="range" min="0.40" max="0.95" step="0.05">
<h2>Noise pattern</h2><select id="nm"><option value="auto">auto (preset default)</option><option value="fixed">fixed (same every chunk: less shimmer)</option><option value="random">random (new every chunk)</option></select>
<h2>Reference image</h2><select id="ref"></select>
<button id="rb">Refresh character (reset cache)</button>
<details id="adv"><summary>Advanced</summary>
<h2>Depth input (pose tracking)</h2><select id="dp"><option value="">preset default</option><option value="on">on</option><option value="off">off</option></select>
<h2>Diffusion steps <span class="muted">more = cleaner detail, slower</span></h2><select id="st"><option value="">preset default</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select>
<h2>Memory starts from <span class="muted">what the first frames after a reset (the sink) are made from</span></h2>
<select id="sf"></select>
<div class="muted">An image is shown to the model instead of the camera for ~0.8 s after each reset, then the camera takes over. The look it sets stays in memory until the next reset.</div>
<h2>Conditioning strength: <span id="vv"></span> <span class="muted">how hard the reference image and depth steer</span></h2>
<input id="vs" type="range" min="0" max="2" step="0.1">
<h2>Attention <label class="muted"><input type="checkbox" id="ao"> steering on</label></h2>
<div class="muted">How much each part of memory is looked at. 1.0 = normal, lower dampens, higher emphasises. Applies instantly, no reset.</div>
<div class="att"><span>Sink (start memory) <b id="av_sink"></b></span><input id="a_sink" type="range" min="0.1" max="3" step="0.05"></div>
<div class="att"><span>Recent frames <b id="av_recent"></b></span><input id="a_recent" type="range" min="0.1" max="3" step="0.05"></div>
<div class="att"><span>Current frame (camera) <b id="av_current"></b></span><input id="a_current" type="range" min="0.1" max="3" step="0.05"></div>
<button id="ar">Reset attention to 1.0</button>
</details>
<h2>Save settings</h2><div style="display:flex;gap:8px"><input id="svn" placeholder="name (optional)" style="flex:1;background:var(--field);color:var(--fg);border:1px solid #333;border-radius:6px;padding:8px;font:14px system-ui"><button id="sv" style="margin-top:0">Save</button></div><div class="muted" id="svm"></div>
<h2>Prompt</h2><textarea id="p" rows="4"></textarea><button id="pb">Apply prompt</button> <span class="muted">Enter applies, Shift+Enter for a new line</span>
<script>
// Zone edges from this laptop: GPU target/slowdown 87/89 C, shutdown 92 C (nvidia-smi -q -d TEMPERATURE);
// CPU TjMax 105 C (throttles from ~100), critical trip 110 C (thermal zone TCPU).
const ZC=['#3fbf6f','#f0c03c','#ff4d4d'], ZN=['efficient','throttling','damaging'];
const GZ={lo:40,hi:100,e:[87,92]}, CZ={lo:40,hi:115,e:[100,110]};
function gauge(id,name,v,z,extra,forceThrottle){const el=document.getElementById(id);
 if(v==null){el.innerHTML=`<small>${name}</small> <b>-</b>`;return;}
 let k=z.e.filter(e=>v>=e).length; if(forceThrottle&&k<1)k=1;
 const pts=[z.lo,...z.e,z.hi], pct=x=>100*(Math.min(Math.max(x,z.lo),z.hi)-z.lo)/(z.hi-z.lo);
 const segs=ZC.map((c,i)=>`<i style="width:${pct(pts[i+1])-pct(pts[i])}%;background:${c};opacity:${i==k?1:.35}"></i>`).join('');
 el.innerHTML=`<small>${name}</small> <b class="z${k}">${v}\u00b0C</b> <small class="z${k}">${ZN[k]}</small> <small>${extra}</small>
 <div class="bar">${segs}<span class="mk" style="left:${pct(v)}%"></span></div>`;}
const steps=[15,20,24,30], s=document.getElementById('s'), n=document.getElementById('n'), ref=document.getElementById('ref'), p=document.getElementById('p'), nm=document.getElementById('nm');
let loaded=false;
const post=(url,body)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
async function poll(){try{const r=await (await fetch('/state')).json();
 document.getElementById('fps').textContent=r.fps.toFixed(1);
 gauge('gg','GPU',r.gpu_temp,GZ,r.gpu_watts==null?'':Math.round(r.gpu_watts)+' W',r.gpu_thermal_throttle);
 gauge('cg','CPU',r.cpu_temp,CZ,'');
 document.getElementById('label').textContent=r.label+' | noise: '+(r.fixed_noise?'fixed':'random')+' | depth: '+(r.depth_on?'on':'off')+' | steps: '+r.n_steps+(r.start?' | starts from: '+r.start:'')+' | upscale: '+r.upscale;
 const dp=document.getElementById('dp'), st=document.getElementById('st'), sf=document.getElementById('sf'), vs=document.getElementById('vs');
 dp.disabled=!r.depth_available; if(!r.depth_available) dp.title='Depth is not available on LongLive';
 if(document.activeElement!==dp) dp.value=r.depth===null?'':(r.depth?'on':'off');
 if(document.activeElement!==st) st.value=r.steps||'';
 if(document.activeElement!==vs) vs.value=r.vace_scale; document.getElementById('vv').textContent=r.vace_scale.toFixed(1);
 if(!sf.options.length) sf.innerHTML='<option value="">camera (default)</option>'+r.refs.map(f=>`<option>${f}</option>`).join('');
 if(document.activeElement!==sf) sf.value=r.start||'';
 document.getElementById('ao').checked=r.attn_on;
 for(const k of ['sink','recent','current']){const el=document.getElementById('a_'+k); if(document.activeElement!==el) el.value=r.attn[k]; document.getElementById('av_'+k).textContent=r.attn[k].toFixed(2);}
 document.getElementById('nv').textContent=r.noise.toFixed(2);
 if(document.activeElement!==s) s.value=steps.indexOf(r.preset);
 if(document.activeElement!==n) n.value=r.noise;
 if(document.activeElement!==nm) nm.value=r.noise_mode;
 if(!loaded){p.value=r.prompt; ref.innerHTML='<option value="">(none - prompt only)</option>'+r.refs.map(f=>`<option ${r.ref.includes(f)?'selected':''}>${f}</option>`).join(''); loaded=true;}
}catch(e){} }
s.oninput=()=>post('/preset/'+steps[s.value]).then(poll);
n.onchange=()=>post('/params',{noise:parseFloat(n.value)}).then(poll);
nm.onchange=()=>post('/params',{noise_mode:nm.value}).then(poll);
ref.onchange=()=>post('/params',{ref:ref.value}).then(poll);
document.getElementById('pb').onclick=()=>post('/params',{prompt:p.value}).then(poll);
p.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('pb').click();}};
document.getElementById('dp').onchange=e=>post('/params',{depth:e.target.value}).then(poll);
document.getElementById('st').onchange=e=>post('/params',{steps:e.target.value}).then(poll);
document.getElementById('sf').onchange=e=>post('/params',{start:e.target.value}).then(poll);
document.getElementById('vs').onchange=e=>post('/params',{vace_scale:parseFloat(e.target.value)}).then(poll);
for(const k of ['sink','recent','current']){const el=document.getElementById('a_'+k);
 el.oninput=()=>{document.getElementById('av_'+k).textContent=parseFloat(el.value).toFixed(2);};
 el.onchange=()=>post('/params',{['attn_'+k]:parseFloat(el.value)}).then(poll);}
document.getElementById('ao').onchange=e=>post('/params',{attn_on:e.target.checked}).then(poll);
document.getElementById('ar').onclick=()=>post('/params',{attn_sink:1,attn_recent:1,attn_current:1}).then(poll);
const svn=document.getElementById('svn'); svn.onkeydown=e=>{if(e.key==='Enter')document.getElementById('sv').click();};
document.getElementById('sv').onclick=async()=>{const r=await (await post('/save',{name:svn.value})).json(); document.getElementById('svm').textContent=r.ok?'saved: '+r.path.split('/').slice(-2).join('/'):'save failed';};
document.getElementById('rb').onclick=()=>post('/refresh').then(poll);
setInterval(poll,1000); poll();
</script></body></html>"""


async def main() -> None:
    async with httpx.AsyncClient(base_url=args.server, timeout=600) as client:
        if not args.no_load:
            await load_pipeline(client)

        if args.input.startswith("/dev/video"):
            player = MediaPlayer(args.input, format="v4l2",
                                 options={"video_size": args.input_size, "framerate": "30", "input_format": "mjpeg"})
        else:
            player = MediaPlayer(args.input, loop=True)

        pc = RTCPeerConnection()
        channel = pc.createDataChannel("parameters", ordered=True)
        camera_track = CropResizeTrack(player.video, W, H, record=args.record_input)
        pc.addTransceiver(camera_track, direction="sendrecv")
        done = asyncio.Event()
        ffmpeg = open_output()
        upscalers: dict[str, Upscaler] = {}

        def current_upscaler() -> Upscaler:
            name = state["upscale_override"] or args.upscale or PRESETS[state["preset"]]["upscale"]
            if name not in upscalers:
                upscalers[name] = Upscaler(name, size=(OH, OW))
            state["upscale"] = name
            upscalers[name].sharpen = state["sharpen"]
            return upscalers[name]

        def send(update: dict) -> None:
            if channel.readyState != "open":
                return
            if update.get("reset_cache") and state["start"]:
                inject["img"], inject["left"] = load_start_image(state["start"]), START_HOLD_FRAMES
            sent_params.update(update)
            channel.send(json.dumps(update))

        def apply_preset(preset: int) -> None:
            state["preset"] = preset
            current_upscaler()
            send(preset_params(preset, with_ref=True))
            print(f"[bridge] preset -> {PRESETS[preset]['label']}", flush=True)

        @pc.on("track")
        def on_track(track):
            async def pump():
                count, t0 = 0, time.perf_counter()
                while True:
                    try:
                        frame = await track.recv()
                    except Exception:
                        break
                    img = frame.to_ndarray(format="rgb24")
                    if img.shape[:2] != (H, W):
                        continue  # skip pre-roll frames at a different size
                    if args.latency_probe:
                        now = time.perf_counter()
                        probe["recv_t"].append(now)
                        luma = float(img[::8, ::8].mean())
                        base = probe["out_base"]
                        probe["out_base"] = luma if base is None else 0.95 * base + 0.05 * min(luma, base + 20)
                        prev = probe["out_prev"]
                        # rising edge: much brighter than the running baseline, previous frame wasn't
                        if base is not None and prev is not None and luma > base + 60 and prev <= base + 60:
                            sent = [t for t in probe["in_edges"] if t < now]
                            if sent:
                                lat = now - sent[-1]
                                probe["lat"].append(lat)
                                print(f"[latency] {lat * 1000:6.0f} ms  sent={sent[-1]:.4f} recv={now:.4f}", flush=True)
                        probe["out_prev"] = luma
                    x = torch.from_numpy(img).cuda().permute(2, 0, 1).float().div(255).unsqueeze(0)
                    y = current_upscaler()(x)[0]
                    ffmpeg.stdin.write(rgb_to_yuv420p(y))
                    count += 1
                    dt = time.perf_counter() - t0
                    if dt >= 5:
                        state["fps"] = count / dt
                        print(f"[bridge] output {state['fps']:5.1f} fps ({PRESETS[state['preset']]['label']})", flush=True)
                        count, t0 = 0, time.perf_counter()
                done.set()

            asyncio.ensure_future(pump())

        @channel.on("open")
        def on_open():
            apply_preset(state["preset"])

        await pc.setLocalDescription(await pc.createOffer())
        params = {"pipeline_ids": [args.pipeline], "input_mode": "video",
                  "prompts": [{"text": state["prompt"], "weight": 1.0}], "manage_cache": True,
                  "vace_enabled": True, **preset_params(state["preset"], with_ref=True)}
        params.pop("reset_cache")
        sent_params.update(params)
        r = await client.post("/api/v1/webrtc/offer", json={
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "initialParameters": params})
        r.raise_for_status()
        answer = r.json()
        session_id = answer["sessionId"]
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
        print(f"[bridge] {args.input} -> Scope ({W}x{H}) -> {args.output} ({OW}x{OH})", flush=True)

        runner = None
        if args.control_port:
            asyncio.ensure_future(poll_temps())
            app = web.Application()
            # node-graph branch: the graph is the main page (read per request, so edits show on reload)
            app.router.add_get("/", lambda _: web.Response(text=GRAPH_PAGE.read_text(), content_type="text/html"))
            app.router.add_get("/classic", lambda _: web.Response(text=CONTROL_PAGE, content_type="text/html"))
            app.router.add_get("/state", lambda _: web.json_response(
                {"preset": state["preset"], "fps": state["fps"], "upscale": state["upscale"],
                 "gpu_temp": state["gpu_temp"], "gpu_watts": state["gpu_watts"], "cpu_temp": state["cpu_temp"],
                 "gpu_thermal_throttle": state["gpu_thermal_throttle"],
                 "label": ("[LongLive] " if args.pipeline == "longlive" else "") + PRESETS[state["preset"]]["label"], "noise": state["noise"], "noise_mode": state["noise_mode"],
                 "fixed_noise": fixed_noise(state["preset"]), "prompt": state["prompt"],
                 "depth": state["depth"], "depth_on": effective_depth(state["preset"]),
                 "depth_available": args.pipeline == "streamdiffusionv2",
                 "steps": state["steps"], "n_steps": len(effective_steps(state["preset"])),
                 "start": state["start"], "vace_scale": state["vace_scale"],
                 "attn": state["attn"], "attn_on": state["attn_on"],
                 "vace_input": state["vace_input"], "upscale_override": state["upscale_override"],
                 "upscalers": UPSCALERS, "pipeline": args.pipeline, "presets": {k: v["label"] for k, v in PRESETS.items()},
                 "noise_pattern": "fixed" if fixed_noise(state["preset"]) else "random",
                 "step_list": effective_steps(state["preset"]),
                 "sink_frames": state["sink_frames"], "sink_max": SINK_MAX, "noise_controller": state["noise_controller"],
                 "transition_steps": state["transition_steps"], "zoom": state["zoom"], "y_offset": state["y_offset"],
                 "sharpen": state["sharpen"], "launch": launch_settings(),
                 "ref": [Path(r).name for r in state["ref"]],
                 "refs": sorted(f.name for f in ASSETS.glob("*.png"))}))

            async def set_params(request):
                body = await request.json()
                update = {}
                if "noise" in body:
                    state["noise"] = max(0.3, min(0.98, float(body["noise"])))
                    update["noise_scale"] = state["noise"]
                if body.get("noise_mode") in ("auto", "fixed", "random"):
                    state["noise_mode"] = body["noise_mode"]
                    update["fixed_noise"] = fixed_noise(state["preset"])
                if "prompt" in body and body["prompt"].strip():
                    state["prompt"] = body["prompt"].strip()
                    target = [{"text": state["prompt"], "weight": 1.0}]
                    if state["transition_steps"]:
                        update["transition"] = {"target_prompts": target, "num_steps": state["transition_steps"],
                                                "temporal_interpolation_method": "linear"}
                    else:
                        update["prompts"] = target
                if body.get("ref") == "":  # "(none)": prompt only
                    state["ref"] = []
                    update = preset_params(state["preset"], with_ref=True)  # reset cache without a ref
                elif "ref" in body:
                    path = (ASSETS / Path(body["ref"]).name).resolve()
                    if path.is_file():
                        state["ref"] = [str(path)]
                        update = preset_params(state["preset"], with_ref=True)  # re-send ref + reset cache
                if "depth" in body:
                    state["depth"] = {"on": True, "off": False}.get(body["depth"])
                    if state["depth"]:
                        state["vace_input"] = False
                    update = preset_params(state["preset"], with_ref=True)  # VACE inputs change: reset
                if "steps" in body:
                    n = int(body["steps"] or 0)
                    state["steps"] = n if n in STEP_LISTS else None
                    update["denoising_step_list"] = effective_steps(state["preset"])
                if "vace_scale" in body:
                    state["vace_scale"] = max(0.0, min(2.0, float(body["vace_scale"])))
                    update["vace_context_scale"] = state["vace_scale"]
                if "upscale" in body:
                    state["upscale_override"] = body["upscale"] if body["upscale"] in UPSCALERS else None
                    current_upscaler()
                if "vace_input" in body:
                    state["vace_input"] = bool(body["vace_input"])
                    if state["vace_input"]:
                        state["depth"] = False  # one VACE video input at a time
                    update = preset_params(state["preset"], with_ref=True)
                if "sink_frames" in body:
                    v = body["sink_frames"]
                    state["sink_frames"] = None if v in ("", None) else max(0, min(SINK_MAX, int(v)))
                    update = preset_params(state["preset"], with_ref=True)  # new sink size needs a reset
                if "noise_controller" in body:
                    state["noise_controller"] = bool(body["noise_controller"])
                    update["noise_controller"] = state["noise_controller"]
                if "transition_steps" in body:
                    state["transition_steps"] = max(0, min(16, int(body["transition_steps"] or 0)))
                if "zoom" in body:
                    state["zoom"] = max(1.0, min(3.0, float(body["zoom"])))
                if "y_offset" in body:
                    state["y_offset"] = max(-1.0, min(1.0, float(body["y_offset"])))
                if "sharpen" in body:
                    state["sharpen"] = max(0.0, min(2.0, float(body["sharpen"])))
                    current_upscaler()
                for key in ("sink", "recent", "current", "text"):
                    if f"attn_{key}" in body:
                        state["attn"][key] = max(0.05, min(4.0, float(body[f"attn_{key}"])))
                        update["attention_weights"] = attention_weights()
                if "attn_on" in body:
                    state["attn_on"] = bool(body["attn_on"])
                    update["attention_weights"] = attention_weights()
                if "start" in body:
                    name = Path(body["start"] or "").name
                    state["start"] = name if name and (ASSETS / name).is_file() else None
                    update = preset_params(state["preset"], with_ref=True)  # re-seed the sink now
                if update:
                    send(update)
                print(f"[bridge] params -> {body}", flush=True)
                return web.json_response({"ok": True})

            app.router.add_post("/params", set_params)

            async def refresh(request):
                send(preset_params(state["preset"], with_ref=True))  # reset + ref (+ start image)
                print("[bridge] cache refresh", flush=True)
                return web.json_response({"ok": True})

            app.router.add_post("/refresh", refresh)

            async def save(request):
                body = await request.json() if request.can_read_body else {}
                name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(body.get("name", ""))).strip()
                pre = state["preset"]
                settings = {
                    "name": name, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": args.pipeline, "vae": args.vae, "preset": pre, "preset_label": PRESETS[pre]["label"],
                    "prompt": state["prompt"], "reference_image": [Path(r).name for r in state["ref"]],
                    "noise": state["noise"], "noise_mode": state["noise_mode"], "fixed_noise": fixed_noise(pre),
                    "depth": state["depth"], "depth_effective": effective_depth(pre),
                    "steps": state["steps"], "denoising_step_list": effective_steps(pre),
                    "start_image": state["start"], "conditioning_strength": state["vace_scale"],
                    "attention_steering_on": state["attn_on"], "attention_weights": state["attn"],
                    "camera_as_vace_input": state["vace_input"], "upscale_override": state["upscale_override"],
                    "graph_layout": body.get("layout"),
                    "upscale": state["upscale"], "size": [H, W], "out_size": [OH, OW],
                    "last_fps": round(state["fps"], 1),
                }
                SAVED.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                path = SAVED / (f"{name.replace(' ', '_')}-{stamp}.json" if name else f"settings-{stamp}.json")
                path.write_text(json.dumps(settings, indent=2) + "\n")
                print(f"[bridge] settings saved -> {path}", flush=True)
                return web.json_response({"ok": True, "path": str(path)})

            app.router.add_post("/save", save)

            async def export(request):
                body = await request.json() if request.can_read_body else {}
                git = {}
                for name, repo in (("workspace", ASSETS.parent), ("scope", ASSETS.parent / "scope")):
                    try:
                        git[name] = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h %D %s"],
                                                   capture_output=True, text=True, timeout=5).stdout.strip()
                    except Exception as e:
                        git[name] = f"unavailable: {e}"
                try:
                    server_status = (await client.get("/api/v1/pipeline/status")).json()
                except Exception as e:
                    server_status = f"unavailable: {e}"
                return web.json_response({
                    "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "bridge_state": {k: v for k, v in state.items()},
                    "effective": {"depth": effective_depth(state["preset"]), "denoising_step_list": effective_steps(state["preset"]),
                                  "fixed_noise": fixed_noise(state["preset"]), "attention_weights": attention_weights(),
                                  "upscaler": state["upscale"]},
                    "params_sent_to_scope": sent_params,
                    "launch": launch_settings(), "bridge_args": vars(args),
                    "scope_env": {k: v for k, v in os.environ.items() if k.startswith(("SCOPE_", "DAYDREAM_"))},
                    "preset_table": PRESETS, "step_lists": STEP_LISTS,
                    "server_pipeline_status": server_status, "git": git,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "graph_layout": body.get("layout"),
                })

            app.router.add_post("/export", export)

            async def set_preset(request):
                apply_preset(int(request.match_info["fps"]))
                return web.json_response({"ok": True})

            app.router.add_post("/preset/{fps:\\d+}", set_preset)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", args.control_port).start()
            print(f"[bridge] live controls: http://127.0.0.1:{args.control_port}", flush=True)

        try:
            if args.duration:
                await asyncio.wait_for(done.wait(), timeout=args.duration)
            else:
                await done.wait()
        except asyncio.TimeoutError:
            pass
        finally:
            try:
                await client.delete(f"/api/v1/webrtc/offer/{session_id}")
            except httpx.HTTPError:
                pass  # server already gone (e.g. start_scope.sh shutting down)
            if probe["lat"]:
                lat = sorted(probe["lat"])
                steady = lat if len(lat) < 4 else sorted(probe["lat"][2:])  # skip warm-up flashes
                print(f"[latency] summary: n={len(lat)} median(all) {lat[len(lat) // 2] * 1000:.0f} ms, "
                      f"median(after warm-up) {steady[len(steady) // 2] * 1000:.0f} ms, "
                      f"min {lat[0] * 1000:.0f} ms, max {lat[-1] * 1000:.0f} ms", flush=True)
            if len(probe["recv_t"]) > 300:
                gaps = [b - a for a, b in zip(probe["recv_t"][150:], probe["recv_t"][151:])]  # skip warm-up
                gaps_ms = sorted(g * 1000 for g in gaps)
                print(f"[smooth] frame gap median {gaps_ms[len(gaps_ms) // 2]:.1f} ms, p95 {gaps_ms[int(len(gaps_ms) * .95)]:.1f} ms, "
                      f"max {gaps_ms[-1]:.0f} ms, stutters(>66ms) {sum(g > 66 for g in gaps_ms)} / {len(gaps_ms)}", flush=True)
            if runner:
                await runner.cleanup()
            await pc.close()
            player.video.stop()
            if camera_track.recorder:
                camera_track.recorder.stdin.close()
                camera_track.recorder.wait()
            ffmpeg.stdin.close()
            ffmpeg.wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
