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
    30: {"label": "30 fps - fast", "depth_vace": False, "steps": [750], "upscale": "bicubic-sharp"},
    24: {"label": "24 fps - fast, 2-step (cleanest)", "depth_vace": False, "steps": [750, 400], "upscale": "bicubic-sharp"},
    20: {"label": "20 fps - depth-guided pose", "depth_vace": True, "steps": [750], "upscale": "bicubic-sharp"},
    15: {"label": "15 fps - depth-guided, 2-step", "depth_vace": True, "steps": [750, 400], "upscale": "bicubic-sharp"},
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
parser.add_argument("--control-port", type=int, default=8090, help="Slider page port (0 = off)")
parser.add_argument("--duration", type=float, default=0, help="Stop after N seconds (0 = run until Ctrl+C)")
parser.add_argument("--no-load", action="store_true", help="Skip pipeline load (already loaded)")
args = parser.parse_args()

H, W = args.size
OH, OW = args.out_size
state = {"preset": args.preset, "fps": 0.0, "upscale": None}


def preset_params(preset: int, with_ref: bool) -> dict:
    p = PRESETS[preset]
    steps = p["steps"]
    params = {
        "denoising_step_list": steps,
        # one-step SDv2: the timestep must match the noise actually added to the camera latents
        "noise_scale": args.noise_scale if len(steps) > 1 else steps[0] / 1000,
        "noise_controller": False,
        "depth_vace": p["depth_vace"] and bool(args.ref),
        "vace_use_input_video": False,
        "reset_cache": True,
    }
    if with_ref and args.ref:
        params["vace_ref_images"] = [str(Path(r).resolve()) for r in args.ref]
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

    def __init__(self, source, width, height):
        super().__init__()
        self.source, self.width, self.height = source, width, height

    async def recv(self):
        frame = await self.source.recv()
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
<title>AI Character FPS</title><style>
:root{--bg:#111;--fg:#eee;--muted:#999;--accent:#ff4d6d}
body{background:var(--bg);color:var(--fg);font:16px system-ui,sans-serif;margin:0;padding:24px 16px;max-width:520px}
h1{font-size:20px;margin:0 0 16px} input[type=range]{width:100%;accent-color:var(--accent)}
.ticks{display:flex;justify-content:space-between;color:var(--muted);font-size:13px}
.big{font-size:40px;font-weight:600;margin:16px 0 4px} .muted{color:var(--muted)}
</style></head><body><h1>AI character - quality vs FPS</h1>
<input id="s" type="range" min="0" max="3" step="1"><div class="ticks"><span>15</span><span>20</span><span>24</span><span>30</span></div>
<div class="big"><span id="fps">-</span> fps</div><div class="muted" id="label"></div>
<script>
const steps=[15,20,24,30], s=document.getElementById('s');
async function poll(){try{const r=await (await fetch('/state')).json();
 document.getElementById('fps').textContent=r.fps.toFixed(1);
 document.getElementById('label').textContent=r.label+' (upscale: '+r.upscale+')';
 if(document.activeElement!==s) s.value=steps.indexOf(r.preset);}catch(e){} }
s.oninput=()=>fetch('/preset/'+steps[s.value],{method:'POST'}).then(poll);
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
        pc.addTransceiver(CropResizeTrack(player.video, W, H), direction="sendrecv")
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
                  "prompts": [{"text": args.prompt, "weight": 1.0}], "manage_cache": True,
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
                 "label": PRESETS[state["preset"]]["label"]}))

            async def set_preset(request):
                apply_preset(int(request.match_info["fps"]))
                return web.json_response({"ok": True})

            app.router.add_post("/preset/{fps:\\d+}", set_preset)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", args.control_port).start()
            print(f"[bridge] FPS slider: http://127.0.0.1:{args.control_port}", flush=True)

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
            if runner:
                await runner.cleanup()
            await pc.close()
            player.video.stop()
            ffmpeg.stdin.close()
            ffmpeg.wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
