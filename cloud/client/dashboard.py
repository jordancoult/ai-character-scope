"""Local control dashboard for live_bridge.py (--dashboard PORT): a stdlib HTTP server in a thread + one HTML page.

The page posts JSON commands; the bridge applies them in its event loop (same handler as the terminal commands):
  {"prompt": str}                       change the prompt (restarts the server session: ~1-2 s gap, memory reset)
  {"reset": true}                       reset the model's memory, same prompt
  {"ref": "<assets path>"|"<data URL>"|"none"}   change the reference image (restart)
  {"orientation": "portrait"|"landscape"}       480x840 <-> 840x480 (restart, output window reopens)
  {"facefix": true|false}               face pass on/off (only if the bridge started with one)
  {"preset": name} / {"gains": {"lip": 2.6, ...}}  face-pass preset / region gains (no restart)
  {"fps": f, "play_fps": f, "buffer": n}           send rate / playout rate / jitter-buffer target (no restart)
  {"session": {...}}                    start-message options, e.g. kv_reset_frames, freeze_kv_on_static (restart)
  {"quit": true}
GET /status -> JSON (prompt, size, ref, rates, glass-to-glass, face pass), GET /assets -> reference images and prompts.
GET /stream/<input|joyai|final> -> MJPEG live view (camera as sent | JoyAI output | after the face pass), ~12 fps,
JPEG-encoded only while a view is open (closed views cost nothing). Binds 127.0.0.1 only.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def list_assets():
    imgs = sorted(str(p.relative_to(REPO)) for p in (REPO / "assets").glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    prompts = {}
    for p in sorted((REPO / "cloud/pipelines/joyai/prompts").glob("*.txt")):
        prompts[p.stem] = p.read_text().strip()
    presets = {}
    pf = REPO / "cloud/lpfast/presets.json"
    if pf.exists():
        presets = json.loads(pf.read_text())
    return {"images": imgs, "prompts": prompts, "presets": presets}


def start(port, submit, status, frames=None):
    """submit(dict) -> None (thread-safe hand-off to the bridge); status() -> dict;
    frames: dict name -> latest BGR frame (the bridge overwrites the entries; read-only here)."""
    frames = frames if frames is not None else {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/status":
                return self._send(200, json.dumps(status()))
            if self.path == "/assets":
                return self._send(200, json.dumps(list_assets()))
            if self.path.startswith("/stream/"):
                return self._stream(self.path[len("/stream/"):].split("?")[0])
            if self.path.startswith("/asset/"):
                p = (REPO / self.path[len("/asset/"):]).resolve()
                if REPO / "assets" in p.parents and p.is_file():
                    ctype = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                    return self._send(200, p.read_bytes(), ctype)
            self._send(404, '{"error": "not found"}')

        def _stream(self, name):
            import cv2
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while True:
                    f = frames.get(name)
                    if f is not None and f is not last:
                        last = f
                        ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                    time.sleep(1 / 12)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            if self.path != "/cmd":
                return self._send(404, '{"error": "not found"}')
            n = int(self.headers.get("Content-Length", 0))
            try:
                cmd = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, '{"error": "bad json"}')
            submit(cmd)
            self._send(200, '{"ok": true}')

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[bridge] dashboard on http://127.0.0.1:{port}", flush=True)
    return srv


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>ai-livestream</title>
<style>
body{font:14px system-ui,sans-serif;background:#111;color:#ddd;margin:0;padding:14px;max-width:1100px}
h1{font-size:17px;margin:0 0 10px} h2{font-size:14px;margin:0 0 8px;color:#9cf}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px} .card{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:12px}
textarea{width:100%;height:110px;background:#0b0b0b;color:#eee;border:1px solid #444;border-radius:4px;font:13px system-ui}
button{background:#2b5d9b;color:#fff;border:0;border-radius:4px;padding:6px 11px;margin:3px 3px 3px 0;cursor:pointer}
button.alt{background:#444} button.warn{background:#8a3b2b} button.on{background:#2e7d32}
select,input[type=number],input[type=text]{background:#0b0b0b;color:#eee;border:1px solid #444;border-radius:4px;padding:4px}
.row{display:flex;align-items:center;gap:8px;margin:5px 0} .row label{min-width:90px}
input[type=range]{flex:1} .val{min-width:42px;text-align:right;font-variant-numeric:tabular-nums}
#status{font:12px ui-monospace,monospace;white-space:pre-wrap;color:#bfb} .thumbs img{height:70px;margin:3px;border:2px solid transparent;border-radius:4px;cursor:pointer;background:#fff}
.thumbs img.sel{border-color:#4af} small{color:#888}
.views{display:flex;gap:10px;flex-wrap:wrap} .views figure{margin:0} .views img{max-height:430px;max-width:100%;border-radius:6px;background:#000}
.views figcaption{font-size:12px;color:#9cf;margin-top:3px}
</style></head><body>
<h1>ai-livestream &middot; live bridge dashboard</h1>
<div class="grid">
 <div class="card" style="grid-column:1/3"><h2>Live views</h2>
  <div class="row"><button id="v_input" class="alt" onclick="toggleView('input')">Webcam input</button>
   <button id="v_joyai" class="alt" onclick="toggleView('joyai')">JoyAI (before lpfast)</button>
   <button id="v_final" class="alt" onclick="toggleView('final')">Final (after lpfast)</button>
   <small>~12 fps previews; the output window / OBS feed is unaffected.</small></div>
  <div class="views" id="views"></div></div>
 <div class="card" style="grid-column:1/3"><h2>Prompt</h2>
  <textarea id="prompt"></textarea>
  <div class="row"><button onclick="send({prompt:val('prompt')})">Apply prompt (restarts session)</button>
   <button class="alt" onclick="send({reset:true})">Reset memory (R)</button>
   <span id="saved"></span></div>
  <div class="row"><small>Saved prompts:</small><span id="prompts"></span></div></div>
 <div class="card"><h2>Reference image</h2>
  <div class="thumbs" id="thumbs"></div>
  <div class="row"><input type="file" id="upload" accept="image/*"><button class="alt" onclick="send({ref:'none'})">No reference</button></div>
  <small>Changing the image restarts the session.</small></div>
 <div class="card"><h2>Format</h2>
  <div class="row"><button id="o_portrait" onclick="send({orientation:'portrait'})">Portrait 480x840</button>
   <button id="o_landscape" onclick="send({orientation:'landscape'})">Landscape 840x480</button></div>
  <h2 style="margin-top:12px">Playout</h2>
  <div class="row"><label>send fps</label><input type="number" id="fps" step="1" min="5" max="30" style="width:70px">
   <label>play fps</label><input type="number" id="play_fps" step="1" min="5" max="30" style="width:70px"></div>
  <div class="row"><label>buffer</label><input type="number" id="buffer" step="1" min="1" max="20" style="width:70px">
   <button class="alt" onclick="send({fps:+val('fps'),play_fps:+val('play_fps'),buffer:+val('buffer')})">Apply</button></div>
  <h2 style="margin-top:12px">Server session</h2>
  <div class="row"><label>kv_reset_frames</label><input type="number" id="kv" step="100" min="0" style="width:80px"><small>0 = never auto-reset</small></div>
  <div class="row"><label>freeze on static</label><input type="checkbox" id="freeze"><button class="alt" onclick="send({session:{kv_reset_frames:+val('kv'),freeze_kv_on_static:document.getElementById('freeze').checked}})">Apply (restart)</button></div></div>
 <div class="card"><h2>Face pass (lpfast)</h2>
  <div class="row"><button id="ff_btn" onclick="send({facefix:!S.facefix_on})">on/off</button>
   <select id="preset" onchange="send({preset:this.value})"></select><small id="ffmode"></small></div>
  <div id="gains"></div><small>Gains apply on the next frame. 1 = your motion as is, 0 = JoyAI's own, negative = mirrored.</small></div>
 <div class="card"><h2>Status</h2><div id="status">connecting...</div></div>
</div>
<script>
let S={}, A={}, gainTimer=null;
const VIEWS={input:'Webcam input',joyai:'JoyAI (before lpfast)',final:'Final (after lpfast)'};
const qv=new URLSearchParams(location.search).get('views');
const openViews=new Set(qv!==null?qv.split(',').filter(k=>k):JSON.parse(localStorage.getItem('views')||'["final"]'));
function renderViews(){const box=document.getElementById('views');
 for(const k of Object.keys(VIEWS)){const b=document.getElementById('v_'+k); b.className=openViews.has(k)?'on':'alt';
  let f=document.getElementById('f_'+k);
  if(openViews.has(k)&&!f){box.insertAdjacentHTML('beforeend',`<figure id="f_${k}" style="order:${Object.keys(VIEWS).indexOf(k)}"><img src="/stream/${k}?t=${Date.now()}"><figcaption>${VIEWS[k]}</figcaption></figure>`);}
  if(!openViews.has(k)&&f){f.querySelector('img').src='';f.remove();}}
 localStorage.setItem('views',JSON.stringify([...openViews]));}
function toggleView(k){openViews.has(k)?openViews.delete(k):openViews.add(k);renderViews();}
const val=id=>document.getElementById(id).value;
async function send(c){await fetch('/cmd',{method:'POST',body:JSON.stringify(c)});setTimeout(poll,300);}
function gainsUI(g){const box=document.getElementById('gains');
 for(const k of ['lip','eye','brow','brow2','rest']){let r=document.getElementById('g_'+k);
  if(!r){box.insertAdjacentHTML('beforeend',`<div class="row"><label>${k}</label><input type="range" id="g_${k}" min="-2" max="4" step="0.05"><span class="val" id="v_${k}"></span></div>`);
   r=document.getElementById('g_'+k); r.oninput=()=>{document.getElementById('v_'+k).textContent=(+r.value).toFixed(2);
    clearTimeout(gainTimer);gainTimer=setTimeout(()=>send({gains:{[k]:+r.value}}),120);};}
  if(document.activeElement!==r && g && k in g){r.value=g[k];document.getElementById('v_'+k).textContent=(+g[k]).toFixed(2);}}}
async function loadAssets(){A=await (await fetch('/assets')).json();
 document.getElementById('thumbs').innerHTML=A.images.map(p=>`<img src="/asset/${p}" title="${p}" data-p="${p}" onclick="send({ref:'${p}'})">`).join('');
 document.getElementById('prompts').innerHTML=Object.keys(A.prompts).map(k=>`<button class="alt" onclick="document.getElementById('prompt').value=A.prompts['${k}']">${k}</button>`).join('');
 document.getElementById('preset').innerHTML=Object.keys(A.presets).map(k=>`<option>${k}</option>`).join('');}
document.getElementById('upload').onchange=e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>send({ref:r.result,ref_name:f.name});r.readAsDataURL(f);};
async function poll(){try{S=await (await fetch('/status')).json();}catch(e){document.getElementById('status').textContent='bridge not reachable';return;}
 const p=document.getElementById('prompt'); if(!p.dataset.init){p.value=S.prompt;p.dataset.init=1;}
 for(const k of ['fps','play_fps','buffer']){const e=document.getElementById(k);if(document.activeElement!==e)e.value=S[k];}
 const kv=document.getElementById('kv'); if(document.activeElement!==kv) kv.value=S.session.kv_reset_frames;
 document.getElementById('freeze').checked=!!S.session.freeze_kv_on_static;
 document.getElementById('o_portrait').className=S.height>S.width?'on':'alt'; document.getElementById('o_landscape').className=S.height>S.width?'alt':'on';
 document.querySelectorAll('#thumbs img').forEach(i=>i.className=(i.dataset.p===S.ref)?'sel':'');
 const fb=document.getElementById('ff_btn'); if(S.facefix_available){fb.textContent='Face pass '+(S.facefix_on?'ON':'OFF');fb.className=S.facefix_on?'on':'alt';gainsUI(S.gains);
  document.getElementById('ffmode').textContent='mode '+S.facefix_mode;} else {fb.textContent='not started (run with --facefix-preset)';fb.disabled=true;}
 document.getElementById('status').textContent=
  `session   ${S.width}x${S.height}  ${S.state}\nref       ${S.ref||'none'}\nserver    ${S.server_fps} fps received\nplayout   ${S.play_fps} fps  queue ${S.queue}  played ${S.played}  repeated ${S.repeated}  dropped ${S.dropped}\nlatency   glass-to-glass median ${S.g2g_ms} ms\nface pass ${S.facefix_available?(S.facefix_on?'on':'off')+'  hit '+S.facefix_hit+'  render '+S.facefix_ms+' ms':'-'}\nprompt    ${S.prompt.slice(0,160)}${S.prompt.length>160?'...':''}\n${S.last_msg||''}`;}
loadAssets(); renderViews(); poll(); setInterval(poll,1000);
</script></body></html>"""
