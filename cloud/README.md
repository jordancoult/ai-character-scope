# Cloud pipeline: live character swap on a rented GPU

Goal: webcam → character from a reference image (at least the upper body with arms), decent consistency
and quality, **25–30 fps sustained**, input-to-output latency **under 2 s**. Target hardware: the most
cost-efficient GPU that meets it, an **RTX 5090 (32 GB, sm_120)** on RunPod ($1.19/h secure, $0.69/h community).

This is separate from the local laptop setup (Scope, `start_scope.sh`), which is unchanged.

## Status (2026-11-02): goal met on one RTX 5090
Model: **JoyAI-Video-Edit** (JD, Apache-2.0, 16B streaming video editor), checkpoint `1101_face++`, 840×480 or
480×840, 2 steps. Details: `experiments/2026-11-02-joyai-5090/FINDINGS.md`, `experiments/2026-11-02-joyai-first/FINDINGS.md`.

| | Result |
|---|---|
| Output fps (RTX 5090) | **25–27 fps** (30 fps camera in) |
| Latency, pipeline on the pod | **0.46 s** median |
| Latency, laptop → EU pod → laptop, smooth playout | **1.03 s** median (US pod would be ~0.2–0.3 s less) |
| Stability | 2 min with no drift, 0 dropped / repeated frames |
| G-Star (non-human) | works with a **full-body ref + explicit description** in the prompt |
| Human reference | strong identity, gestures followed |

**Prompt recipe:** describe the character explicitly. Generic "replace with the character from the reference
image" only transfers accessories, and the built-in prompt enhancer made it worse.
```
Replace the person with the pink cartoon starfish creature from the reference image: a tall cone-shaped pink head
with no hair, sleepy half-closed eyes, a wide smile, bumpy pink skin, a black baseball cap with a red G-Star logo,
and a thick gold chain on a bare pink chest. Keep the pose, gestures and hand movements.
```

## Layout
| Path | What |
|---|---|
| `joyai/setup_pod.sh` | Pod setup: CUDA 12.8 compiler + Python env on local disk (`env`), weights to the volume (`weights`); kernel wheels built once and cached on the volume |
| `joyai/run_server.sh` | Starts the JoyAI server with the RTX 5090 settings and the face++ checkpoint; compile cache on local disk, seeded from the volume |
| `joyai/save_cache.sh` | Copies the warm compile cache back to the volume (run once after the first start on a new GPU type) |
| `client/live_bridge.py` | **The live path**: camera (or a clip) → pod over H.264 → adaptive jitter-buffer playout → v4l2loopback (OBS) or mp4 |
| `client/joyai_client.py` | Benchmark client: simulated webcam, records output, reports fps and per-frame latency |
| `client/sbs.sh` | input / output / reference review video + contact sheet |
| `pod.sh` | ssh/scp helper (address in `.pod`, key in `../.secrets/`, both gitignored) |
| `docker/Dockerfile` | **Draft, not built yet**: portable image with env + kernel wheels + warm cache baked in |
| `experiments/` | Runs, numbers, review videos, verdicts |

## RunPod resources
- Network volume `ai-character-cloud` (`jg2ckg7vlg`), 150 GB, **EU-RO-1**, mounted at `/workspace`: repo, weights
  (0811 + face++ DiT, VAE, MiMo-VL), kernel wheels, compile caches for the PRO 4500 and the RTX 5090. ~$10.50/month.
- Pods are created per session in EU-RO-1 (the volume's data center; network volumes need Secure Cloud) and
  terminated afterwards. RTX 5090 stock there comes and goes.

## Start a session (≈10 min with the warm cache)
1. Create a pod: RTX 5090, Secure Cloud, EU-RO-1, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, volume
   `jg2ckg7vlg` at `/workspace`, ports `22/tcp,8080/http`, env `PUBLIC_KEY` = `../.secrets/runpod_ed25519.pub`.
2. Put its SSH host and port in `cloud/.pod` (`HOST PORT`), then:
   ```bash
   cloud/pod.sh put cloud/joyai/*.sh /workspace/
   cloud/pod.sh 'bash /workspace/setup_pod.sh env'                       # ~5 min
   cloud/pod.sh 'cd /workspace && setsid nohup bash -c "bash run_server.sh; echo EXIT=\$?" > logs/server.log 2>&1 < /dev/null &'
   # ready when logs/server.log shows "Uvicorn running": ~4–20 min with the warm cache. The variable part is
   # reading the 32 GB DiT checkpoint from the network volume (cold reads were ~15 min once); ~12 min more on a new GPU type
   ssh -f -N -i .secrets/runpod_ed25519 -L 18080:127.0.0.1:8080 -p PORT root@HOST
   ```
3. Live to OBS (the virtual camera `/dev/video10` from the local setup):
   ```bash
   cloud/.venv/bin/python cloud/client/live_bridge.py --url ws://127.0.0.1:18080/ws \
     --input /dev/video0 --output /dev/video10 --portrait --out-size 720x1260 \
     --ref assets/char_front.png --prompt "Replace the person with the pink cartoon starfish creature ..."
   ```
   No camera: `--input assets/webcam_5941016.mp4 --output test.mp4 --seconds 30`.
4. Terminate the pod when done (the volume keeps everything).

## Next
- Package: build the Docker image (needs ~20 GB free disk locally) and push to a private registry → no volume needed,
  community-cloud 5090s ($0.69/h) in any region.
- A US data center for lower latency (no 5090 stock there tonight).
- Real-webcam test (`--input /dev/video0`) with the user; face/hand detail at 840×480 is the main quality limit.
