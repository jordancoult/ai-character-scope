# Cloud pipeline: live character swap on a rented GPU

Goal: webcam → character from a reference image (at least the upper body with arms), decent consistency
and quality, **25–30 fps sustained**, input-to-output latency **under 2 s** on the GPU machine. Target
hardware is the most cost-efficient GPU that meets this; currently an **RTX 5090 (32 GB, sm_120)** on RunPod
(about $0.69–1.19/h). Everything here stays compatible with it.

This is separate from the local laptop setup (Scope, `start_scope.sh`), which is unchanged.

## Current candidate: JoyAI-Video-Edit
JD's open (Apache-2.0) 16B streaming video editor. Takes live frames plus an instruction and an optional
reference image (RV2V), and edits causally in 8-frame chunks. Published: 840×480 @ 24 fps on one RTX 5090.
Background: `../experiments/2026-11-01-cloud-gpu-research.md`.

## Layout
| Path | What |
|---|---|
| `joyai/setup_pod.sh` | One-time setup on a pod: CUDA 12.8 compiler, Python env on local disk, kernel wheels and weights on the network volume |
| `joyai/run_server.sh` | Starts the JoyAI server with the RTX 5090 settings |
| `client/joyai_client.py` | Simulated webcam: plays a clip in real time over the WebSocket, sends the reference + instruction, records the output, reports fps and latency |
| `pod.sh` | ssh/scp helper (address in `.pod`, key in `../.secrets/`, both gitignored) |
| `experiments/` | Test runs: settings, numbers, videos, verdicts |
| `third_party/` | Read-only clone of upstream repos for reference (gitignored) |

## RunPod resources
- Network volume `ai-character-cloud` (`jg2ckg7vlg`), 150 GB, EU-RO-1, mounted at `/workspace`. Holds the
  repo, weights, built wheels, compile caches. About $10.50/month.
- Pods are created per session in EU-RO-1 (the volume's data center) and terminated afterwards.

## Workflow
```bash
# on a fresh pod (same data center as the volume)
cloud/pod.sh put cloud/joyai/*.sh /workspace/
cloud/pod.sh 'bash /workspace/setup_pod.sh env'      # ~10 min first time, faster with cached wheels
cloud/pod.sh 'nohup bash /workspace/run_server.sh > /workspace/logs/server.log 2>&1 &'
# from the laptop: tunnel and simulated webcam
ssh -N -L 8080:127.0.0.1:8080 ...   # or the RunPod proxy URL
cloud/.venv/bin/python cloud/client/joyai_client.py --url ws://127.0.0.1:8080/ws \
  --video assets/webcam_5941016.mp4 --ref assets/char_face.png \
  --prompt "Replace the main subject in the video with the character from the reference image." --out out.mp4
```
