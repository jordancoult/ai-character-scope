"""Remote viewing for the bridge control server: output preview (JPEG/MJPEG), optional token auth, /remote page.

Used by scope_bridge.py. Kept free of torch/aiortc imports so it can be tested on its own.

- OutputPreview.publish(frame) is called from the bridge's output loop for every frame. It only stores a
  reference (no copy, no encode). JPEGs are made lazily when someone asks for /out.jpg or /out.mjpg,
  at most `max_fps` times a second, shared by all viewers, in a worker thread.
- With a token, every route needs `?token=...` (first visit; sets an HttpOnly cookie and redirects to the
  same URL without the token) or that cookie. Without a token nothing is installed except the routes.
"""

import asyncio
import hmac
import io
import time

from aiohttp import web

COOKIE = "scope_remote_token"


class OutputPreview:
    def __init__(self, max_width: int = 960, quality: int = 75, max_fps: float = 15.0):
        self.max_width, self.quality, self.min_interval = max_width, quality, 1.0 / max_fps
        self.frame = None  # latest output frame: torch [3, H, W] float 0..1, or numpy HxWx3 uint8
        self.seq = 0
        self._jpeg, self._jpeg_seq, self._jpeg_t = None, -1, 0.0
        self._lock = asyncio.Lock()

    def publish(self, frame) -> None:
        """Per-frame hot path: just keep a reference."""
        self.frame = frame
        self.seq += 1

    def _encode(self, frame) -> bytes:
        from PIL import Image

        if type(frame).__module__.startswith("torch"):  # [3, H, W] float on GPU
            import torch
            import torch.nn.functional as F

            x = frame.detach()
            h, w = x.shape[-2:]
            if w > self.max_width:
                x = F.interpolate(x[None].float(), size=(round(h * self.max_width / w), self.max_width),
                                  mode="bilinear", antialias=True)[0]
            arr = x.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray(arr)
        else:
            img = Image.fromarray(frame)
            if img.width > self.max_width:
                img = img.resize((self.max_width, round(img.height * self.max_width / img.width)), Image.BILINEAR)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=self.quality)
        return buf.getvalue()

    async def jpeg(self):
        """Latest output as JPEG bytes (cached; re-encoded at most max_fps times a second), or None."""
        async with self._lock:
            frame, seq = self.frame, self.seq
            if frame is None:
                return None
            fresh = seq == self._jpeg_seq or time.monotonic() - self._jpeg_t < self.min_interval
            if self._jpeg is None or not fresh:
                self._jpeg = await asyncio.get_running_loop().run_in_executor(None, self._encode, frame)
                self._jpeg_seq, self._jpeg_t = seq, time.monotonic()
            return self._jpeg


def token_middleware(token: str):
    @web.middleware
    async def check(request, handler):
        given = request.query.get("token")
        if given is not None and hmac.compare_digest(given.encode(), token.encode()):
            secure = request.headers.get("X-Forwarded-Proto", request.scheme) == "https"
            if request.method == "GET":  # drop the token from the address bar / history
                query = {k: v for k, v in request.query.items() if k != "token"}
                resp = web.HTTPFound(request.rel_url.with_query(query))
            else:
                resp = await handler(request)
            resp.set_cookie(COOKIE, token, max_age=7 * 24 * 3600, httponly=True, samesite="Lax", secure=secure)
            if isinstance(resp, web.HTTPException):
                raise resp
            return resp
        cookie = request.cookies.get(COOKIE, "")
        if cookie and hmac.compare_digest(cookie.encode(), token.encode()):
            return await handler(request)
        return web.Response(status=401, text="401 Unauthorized: open the link that includes ?token=...\n",
                            headers={"Cache-Control": "no-store"})

    return check


REMOTE_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>AI Character - remote</title><style>
body{margin:0;background:#0f0f12;color:#ececf0;font:15px system-ui,sans-serif;text-align:center}
#v{display:block;margin:0 auto;max-width:100%;max-height:82vh;background:#000}
.bar{padding:10px 12px;color:#9a9aa6;font-size:14px} .bar b{color:#ececf0}
select{background:#1a1a20;color:#ececf0;border:1px solid #34343f;border-radius:6px;padding:5px}
a.btn{display:inline-block;margin:6px 3px 16px;padding:10px 12px;border-radius:8px;background:#ff4d6d;color:#fff;
 text-decoration:none;font-weight:600} a.sec{background:#33333f}
</style></head><body>
<img id="v" alt="AI output (waiting for frames...)">
<div class="bar"><b id="fps">-</b> fps &middot; GPU <b id="gpu">-</b> &middot; CPU <b id="cpu">-</b> <span id="mode"></span></div>
<div class="bar">Stream <select id="fr"><option value="12">12 fps (~7 Mbit/s)</option><option value="6">6 fps</option>
<option value="3">3 fps (data saver)</option></select></div>
<a class="btn" href="/">Full control page</a><a class="btn sec" href="#" id="sw">Snapshot mode</a>
<script>
const v=document.getElementById('v'), fr=document.getElementById('fr'); let snap=false, timer=null;
fr.value=localStorage.remoteFps||'12'; fr.onchange=()=>{localStorage.remoteFps=fr.value; start();};
function stop(){ clearTimeout(timer); v.onerror=v.onload=null; v.removeAttribute('src'); }
function start(){ stop();
  if(!snap){ v.onerror=()=>{snap=true;start();}; v.onload=null; v.src='/out.mjpg?fps='+fr.value+'&t='+Date.now(); }
  else { v.onerror=()=>{timer=setTimeout(tick,1000);}; v.onload=()=>{timer=setTimeout(tick,1000/fr.value);}; tick(); }
  document.getElementById('sw').textContent=snap?'Stream mode':'Snapshot mode';
  document.getElementById('mode').textContent=snap?'(snapshots)':''; }
function tick(){ v.src='/out.jpg?t='+Date.now(); }
document.getElementById('sw').onclick=e=>{e.preventDefault(); snap=!snap; start();};
document.addEventListener('visibilitychange',()=>document.hidden?stop():start());  // no data while in the background
async function poll(){ try{ const s=await (await fetch('/state')).json();
  document.getElementById('fps').textContent=s.fps.toFixed(1);
  document.getElementById('gpu').textContent=s.gpu_temp==null?'-':s.gpu_temp+'\\u00b0C';
  document.getElementById('cpu').textContent=s.cpu_temp==null?'-':s.cpu_temp+'\\u00b0C'; }catch(e){} }
setInterval(poll,2000); poll(); start();
</script></body></html>"""


def setup(app: web.Application, preview: OutputPreview, token=None, stream_fps: float = 12.0) -> None:
    """Add /out.jpg, /out.mjpg and /remote; with a token, protect every route of `app`."""
    if token:
        app.middlewares.append(token_middleware(token))
    stopping = asyncio.Event()

    async def on_shutdown(_app):
        stopping.set()  # end open MJPEG streams so shutdown/restart doesn't wait for them

    app.on_shutdown.append(on_shutdown)

    async def out_jpg(_):
        data = await preview.jpeg()
        if data is None:
            return web.Response(status=204)
        return web.Response(body=data, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def out_mjpg(request):
        try:
            fps = max(1.0, min(15.0, float(request.query.get("fps", stream_fps))))
        except ValueError:
            fps = stream_fps
        resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                           "Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        await resp.prepare(request)
        last = None
        try:
            while not stopping.is_set():
                t0 = time.monotonic()
                data = await preview.jpeg()
                if data is not None and data is not last:
                    await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    last = data
                try:
                    await asyncio.wait_for(stopping.wait(), max(0.0, 1.0 / fps - (time.monotonic() - t0)))
                except asyncio.TimeoutError:
                    pass
        except ConnectionError:  # viewer went away
            pass
        return resp

    app.router.add_get("/out.jpg", out_jpg)
    app.router.add_get("/out.mjpg", out_mjpg)
    app.router.add_get("/remote", lambda _: web.Response(text=REMOTE_PAGE, content_type="text/html"))
