"""Fetch videos: TikTok page URLs via the Zyla API (no watermark), direct media URLs, or local paths.

Zyla API Hub, "TikTok Media Grabber API" (listing 5312, documented at
https://zylalabs.com/api-marketplace/entertainment+%26+media/tiktok+media+grabber+api/5312):

  GET https://zylalabs.com/api/5312/tiktok+media+grabber+api/14542/get+video?url=<tiktok url>
  Authorization: Bearer <key>

Documented response (tikwm-style): {"code": 0, "msg": "success", "data": {"play": ..., "wmplay": ..., "hdplay": ...,
"music": ..., "size", "wm_size", "hd_size", ...}}. `wmplay` is the watermarked file; `hdplay` (HD original) and
`play` are watermark-free. Media URLs expire (x-expires), so they are downloaded immediately.
The endpoint can be overridden with --zyla-endpoint / ZYLA_ENDPOINT if the subscribed API differs.
"""

import os
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from common import REPO, log, probe, sha, write_json

ZYLA_ENDPOINT = "https://zylalabs.com/api/5312/tiktok+media+grabber+api/14542/get+video"
ZYLA_KEY_FILE = REPO / ".secrets" / "zyla_api_key"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
# Preference order of fields holding a watermark-free video URL (various Zyla TikTok APIs use different names).
NOWM_KEYS = ["hdplay", "hd_play", "nowm", "no_watermark", "nowatermark", "video_no_watermark", "play",
             "download_url", "video_url", "url"]
MEDIA_HOST_FALLBACK = "https://www.tikwm.com"  # relative media paths in tikwm-style responses


class FetchError(Exception):
    pass


def is_tiktok_page(url):
    host = urlparse(url).netloc.lower()
    return any(h in host for h in ("tiktok.com", "douyin.com")) and not urlparse(url).path.endswith(".mp4")


def zyla_key():
    k = os.environ.get("ZYLA_API_KEY")
    if k:
        return k.strip()
    if ZYLA_KEY_FILE.exists():
        return ZYLA_KEY_FILE.read_text().strip()
    return None


def pick_video_url(resp):
    """Find the best watermark-free video URL in a Zyla JSON response (returns (url, field) or (None, None))."""
    if isinstance(resp, dict) and resp.get("code") not in (None, 0, "0", 200):
        raise FetchError(f"API error code={resp.get('code')} msg={resp.get('msg')}")
    cands = []

    def walk(o, depth=0):
        if depth > 6:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if isinstance(v, str) and v.strip() and kl in NOWM_KEYS:
                    if "music" in kl or "audio" in kl or "cover" in kl:
                        continue
                    cands.append((NOWM_KEYS.index(kl), depth, v.strip(), k))
                elif isinstance(v, (dict, list)) and kl not in ("music_info", "author", "wmplay"):
                    walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, depth + 1)

    walk(resp)
    cands = [c for c in cands if not c[2].lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".mp3"))]
    if not cands:
        return None, None
    cands.sort(key=lambda c: (c[0], c[1]))
    url = cands[0][2]
    if url.startswith("/"):
        url = MEDIA_HOST_FALLBACK + url
    return url, cands[0][3]


def zyla_resolve(page_url, endpoint=None, retries=4, timeout=60):
    key = zyla_key()
    if not key:
        raise FetchError(f"Zyla key missing: put it in {ZYLA_KEY_FILE} (or ZYLA_API_KEY)")
    endpoint = endpoint or os.environ.get("ZYLA_ENDPOINT") or ZYLA_ENDPOINT
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(endpoint, params={"url": page_url}, timeout=timeout,
                             headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(2 ** attempt * 2)
                continue
            if r.status_code != 200:
                raise FetchError(f"Zyla HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            url, field = pick_video_url(data)
            if not url:
                raise FetchError(f"no video URL in Zyla response (keys: {list(_flat_keys(data))[:30]})")
            return url, field, data
        except (requests.RequestException, ValueError) as e:
            last = repr(e)
            time.sleep(2 ** attempt * 2)
    raise FetchError(f"Zyla failed after {retries} tries: {last}")


def _flat_keys(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield pre + str(k)
            yield from _flat_keys(v, pre + str(k) + ".")


def http_download(url, dest, timeout=120, referer=None):
    tmp = Path(str(dest) + ".part")
    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer
    with requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as r:
        if r.status_code != 200:
            raise FetchError(f"download HTTP {r.status_code} for {url[:120]}")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def fetch(src, dest_dir, name, zyla_endpoint=None, force=False):
    """Materialize `src` (TikTok page URL / direct URL / local path) as dest_dir/name.mp4.

    Cached: if the file exists with a matching .json sidecar for the same source, nothing is downloaded.
    Returns (path, meta). Raises FetchError.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest, side = dest_dir / f"{name}.mp4", dest_dir / f"{name}.json"
    if not force and dest.exists() and side.exists():
        import json
        meta = json.loads(side.read_text())
        if meta.get("source") == src:
            return dest, meta
    src = src.strip()
    meta = {"source": src, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if src.startswith(("http://", "https://")):
        if is_tiktok_page(src):
            url, field, _ = zyla_resolve(src, zyla_endpoint)
            meta.update(method="zyla", zyla_field=field, media_url_hash=sha(url))
            http_download(url, dest, referer="https://www.tiktok.com/")
        else:
            meta.update(method="direct")
            http_download(src, dest)
    else:
        p = Path(os.path.expanduser(src))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FetchError(f"local file not found: {p}")
        meta.update(method="local", local_path=str(p))
        shutil.copyfile(p, dest)
    try:
        meta["probe"] = probe(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise FetchError(f"downloaded file is not a video: {e}")
    write_json(side, meta)
    log.info("fetched %s -> %s (%s)", src[:90], dest, meta["method"])
    return dest, meta
