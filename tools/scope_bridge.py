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
parser.add_argument("--noise-scale", type=float, default=0.75)
parser.add_argument("--noise-mode", default="fixed", choices=["auto", "fixed", "random"],
                    help="Per-chunk noise (default fixed): auto = the preset's choice, fixed = same noise every chunk, random = new noise")
parser.add_argument("--control-port", type=int, default=8090, help="Slider page port (0 = off)")
parser.add_argument("--latency-probe", action="store_true",
                    help="Measure end-to-end latency using bright flashes in the input (assets/latency_flash.mp4)")
parser.add_argument("--record-input", default=None, help="Also save the cropped camera frames sent to Scope (.mp4)")
parser.add_argument("--duration", type=float, default=0, help="Stop after N seconds (0 = run until Ctrl+C)")
parser.add_argument("--no-load", action="store_true", help="Skip pipeline load (already loaded)")
args = parser.parse_args()

H, W = args.size
OH, OW = args.out_size
state = {"preset": args.preset, "fps": 0.0, "upscale": None, "noise": args.noise_scale,
         "noise_mode": args.noise_mode,
         "prompt": args.prompt, "ref": [str(Path(r).resolve()) for r in args.ref] if args.ref else []}
ASSETS = Path(__file__).resolve().parents[1] / "assets"
probe = {"in_edges": [], "in_prev": 0.0, "out_prev": None, "out_base": None, "lat": [], "recv_t": []}


def fixed_noise(preset: int) -> bool:
    mode = state["noise_mode"]
    return PRESETS[preset].get("fixed_noise", False) if mode == "auto" else mode == "fixed"


def preset_params(preset: int, with_ref: bool) -> dict:
    p = PRESETS[preset]
    params = {
        "denoising_step_list": p["steps"],
        # Scope maps noise_scale to the first timestep (1000 * noise - 100)
        "noise_scale": state["noise"],
        "noise_controller": False,
        "fixed_noise": fixed_noise(preset),
        "depth_vace": p["depth_vace"] and bool(state["ref"]),
        "vace_use_input_video": False,
        "reset_cache": True,
    }
    if with_ref and state["ref"]:
        params["vace_ref_images"] = state["ref"]
        params["vace_context_scale"] = args.vace_scale
    return params


async def load_pipeline(client: httpx.AsyncClient) -> None:
    load_params = {"height": H, "width": W, "vae_type": "tae", "vace_enabled": bool(args.ref)}
    r = await client.post("/api/v1/pipeline/load",
                          json={"pipeline_ids": ["streamdiffusionv2"], "load_params": load_params})
    r.raise_for_status()
    print(f"[bridge] loading streamdiffusionv2 {load_params} ...", flush=True)
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
        if abs(fw / fh - target) > 0.01:
            arr = frame.to_ndarray(format="rgb24")
            if fw / fh > target:  # too wide: crop width
                cw = round(fh * target)
                x0 = (fw - cw) // 2
                arr = arr[:, x0 : x0 + cw]
            else:  # too tall: crop height
                ch = round(fw / target)
                y0 = (fh - ch) // 2
                arr = arr[y0 : y0 + ch]
            new = VideoFrame.from_ndarray(arr, format="rgb24")
        else:
            new = frame
        new = new.reformat(width=self.width, height=self.height, format="yuv420p")
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
.big{font-size:40px;font-weight:600;margin:12px 0 2px} .muted{color:var(--muted);font-size:14px}
textarea,select{width:100%;box-sizing:border-box;background:var(--field);color:var(--fg);border:1px solid #333;border-radius:6px;padding:8px;font:14px system-ui}
button{margin-top:8px;background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;font:600 14px system-ui;cursor:pointer}
</style></head><body><h1>AI character - live controls</h1>
<div class="big"><span id="fps">-</span> fps</div><div class="muted" id="label"></div>
<h2>Quality vs FPS</h2>
<input id="s" type="range" min="0" max="3" step="1"><div class="ticks"><span>15</span><span>20</span><span>24</span><span>30</span></div>
<h2>Noise: <span id="nv"></span> &nbsp;<span class="muted">lower = follows you more, higher = more character</span></h2>
<input id="n" type="range" min="0.40" max="0.95" step="0.05">
<h2>Noise pattern</h2><select id="nm"><option value="auto">auto (preset default)</option><option value="fixed">fixed (same every chunk: less shimmer)</option><option value="random">random (new every chunk)</option></select>
<h2>Reference image</h2><select id="ref"></select>
<button id="rb">Refresh character (reset cache)</button>
<h2>Prompt</h2><textarea id="p" rows="4"></textarea><button id="pb">Apply prompt</button>
<script>
const steps=[15,20,24,30], s=document.getElementById('s'), n=document.getElementById('n'), ref=document.getElementById('ref'), p=document.getElementById('p'), nm=document.getElementById('nm');
let loaded=false;
const post=(url,body)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
async function poll(){try{const r=await (await fetch('/state')).json();
 document.getElementById('fps').textContent=r.fps.toFixed(1);
 document.getElementById('label').textContent=r.label+' | noise: '+(r.fixed_noise?'fixed':'random')+' | upscale: '+r.upscale;
 document.getElementById('nv').textContent=r.noise.toFixed(2);
 if(document.activeElement!==s) s.value=steps.indexOf(r.preset);
 if(document.activeElement!==n) n.value=r.noise;
 if(document.activeElement!==nm) nm.value=r.noise_mode;
 if(!loaded){p.value=r.prompt; ref.innerHTML=r.refs.map(f=>`<option ${r.ref.includes(f)?'selected':''}>${f}</option>`).join(''); loaded=true;}
}catch(e){} }
s.oninput=()=>post('/preset/'+steps[s.value]).then(poll);
n.onchange=()=>post('/params',{noise:parseFloat(n.value)}).then(poll);
nm.onchange=()=>post('/params',{noise_mode:nm.value}).then(poll);
ref.onchange=()=>post('/params',{ref:ref.value}).then(poll);
document.getElementById('pb').onclick=()=>post('/params',{prompt:p.value}).then(poll);
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
            name = args.upscale or PRESETS[state["preset"]]["upscale"]
            if name not in upscalers:
                upscalers[name] = Upscaler(name, size=(OH, OW))
            state["upscale"] = name
            return upscalers[name]

        def apply_preset(preset: int) -> None:
            state["preset"] = preset
            current_upscaler()
            if channel.readyState == "open":
                channel.send(json.dumps(preset_params(preset, with_ref=True)))
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
        params = {"pipeline_ids": ["streamdiffusionv2"], "input_mode": "video",
                  "prompts": [{"text": state["prompt"], "weight": 1.0}], "manage_cache": True,
                  "vace_enabled": bool(args.ref), **preset_params(state["preset"], with_ref=True)}
        params.pop("reset_cache")
        r = await client.post("/api/v1/webrtc/offer", json={
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "initialParameters": params})
        r.raise_for_status()
        answer = r.json()
        session_id = answer["sessionId"]
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
        print(f"[bridge] {args.input} -> Scope ({W}x{H}) -> {args.output} ({OW}x{OH})", flush=True)

        runner = None
        if args.control_port:
            app = web.Application()
            app.router.add_get("/", lambda _: web.Response(text=CONTROL_PAGE, content_type="text/html"))
            app.router.add_get("/state", lambda _: web.json_response(
                {"preset": state["preset"], "fps": state["fps"], "upscale": state["upscale"],
                 "label": PRESETS[state["preset"]]["label"], "noise": state["noise"], "noise_mode": state["noise_mode"],
                 "fixed_noise": fixed_noise(state["preset"]), "prompt": state["prompt"],
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
                    update["prompts"] = [{"text": state["prompt"], "weight": 1.0}]
                if "ref" in body:
                    path = (ASSETS / Path(body["ref"]).name).resolve()
                    if path.is_file():
                        state["ref"] = [str(path)]
                        update = preset_params(state["preset"], with_ref=True)  # re-send ref + reset cache
                if update and channel.readyState == "open":
                    channel.send(json.dumps(update))
                print(f"[bridge] params -> {body}", flush=True)
                return web.json_response({"ok": True})

            app.router.add_post("/params", set_params)

            async def refresh(request):
                if channel.readyState == "open":
                    channel.send(json.dumps(preset_params(state["preset"], with_ref=True)))  # reset + ref
                print("[bridge] cache refresh", flush=True)
                return web.json_response({"ok": True})

            app.router.add_post("/refresh", refresh)

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
