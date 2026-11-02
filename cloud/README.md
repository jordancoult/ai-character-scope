# Cloud pipelines: realtime AI livestream video on a rented Blackwell GPU

Webcam → AI-transformed video → OBS, rendered on a rented GPU. Each pipeline is a model + task (character swap,
restyle, world generation, ...) behind the same live bridge. Shared bar: 25–30 fps, < 2 s input → output. Start with `../AGENTS.md` for the repo-wide rules.

## Pipelines
| Pipeline | Task | Model | GPU | Result | Status |
|---|---|---|---|---|---|
| [`joyai`](pipelines/joyai/README.md) | character swap from a reference image (also instruction edits: restyle, background, ...) | JoyAI-Video-Edit (JD, Apache-2.0, 16B streaming editor, ckpt `1101_face++`) | RTX 5090 / RTX PRO 6000 (sm_120, ≥ 32 GB) | **25–27 fps** at 840×480 / 480×840, **0.46 s** on the GPU machine, **~1 s** laptop → EU → laptop | working |

Adding a pipeline: copy the shape of `pipelines/joyai/` (setup.sh with the GPU check, run_server.sh, README.md,
experiments/), keep it provider-independent, and add a row here.

## Layout
| Path | What |
|---|---|
| `pipelines/<name>/` | setup + launcher + recipe + experiments for one model |
| `client/live_bridge.py` | **The live path**: camera (or a clip) → GPU machine over H.264 WebSocket → adaptive jitter-buffer playout → v4l2loopback (`/dev/video10`, OBS) or mp4 |
| `client/joyai_client.py` | Benchmark client: simulated webcam, records output, reports fps + per-frame latency |
| `client/sbs.sh` | input / output / reference review video + contact sheet |
| `infra/pod.sh` | ssh/scp to the current GPU machine (address in `infra/.pod` as `HOST PORT`, gitignored) |
| `infra/providers.md` | How to rent a machine on RunPod / Vast.ai / others; stock and price notes |
| `infra/artifacts.md` | The private Hugging Face artifacts repo: contents, how to publish/refresh |
| `infra/docker/` | Draft Docker image (not built yet) |
| `docs/lessons.md` | Things that cost time once; read before changing infra |
| `third_party/` | Read-only upstream clones for reference (gitignored) |

## Quickstart (any provider, ~12–20 min to first frame)
1. Rent a machine: RTX 5090 or RTX PRO 6000, Ubuntu image with CUDA ≥ 12.8 and Python 3.12 (e.g.
   `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), ≥ 120 GB disk, SSH (`PUBLIC_KEY` = `.secrets/runpod_ed25519.pub`)
   and port 8080 exposed. See `infra/providers.md`.
2. Write `HOST PORT` into `cloud/infra/.pod`, then:
   ```bash
   cloud/infra/pod.sh put cloud/pipelines/joyai/*.sh cloud/pipelines/joyai/webui /workspace/
   cloud/infra/pod.sh put .secrets/hf_token /root/.cache/huggingface/token   # private artifacts (optional)
   cloud/infra/pod.sh 'cd /workspace && bash setup.sh all'                   # artifacts + env + weights
   cloud/infra/pod.sh 'cd /workspace && setsid nohup bash -c "bash run_server.sh; echo EXIT=\$?" > logs/server.log 2>&1 < /dev/null & disown'
   # ready when logs/server.log shows "Uvicorn running"
   ssh -f -N -i .secrets/runpod_ed25519 -L 18080:127.0.0.1:8080 -p PORT root@HOST
   ```
3. Live into OBS:
   ```bash
   cloud/.venv/bin/python cloud/client/live_bridge.py --url ws://127.0.0.1:18080/ws \
     --input /dev/video0 --output /dev/video10 --portrait --out-size 720x1260 \
     --ref assets/char_front.png --prompt "$(cat cloud/pipelines/joyai/prompts/gstar.txt)"
   ```
   Without a camera: `--input assets/webcam_5941016.mp4 --output test.mp4 --seconds 30`.
   Or the browser demo at http://127.0.0.1:18080 (webcam in the browser; our memory panel bottom-right: **R** resets
   the model's memory, auto-reset 0 = never).
4. Terminate the machine when done.

Local client env (once): `bin/uv venv cloud/.venv --python 3.12 && bin/uv pip install --python cloud/.venv/bin/python websockets opencv-python-headless numpy av`.
