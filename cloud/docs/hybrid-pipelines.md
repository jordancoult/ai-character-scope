# Hybrid pipelines: one live pipeline across several GPU machines

A pipeline is a chain of **stages**. Each stage runs on a **host** (the local machine with the camera, or a rented
GPU machine), and stages talk over a **transport**. This page is the contract every pipeline in `cloud/pipelines/`
follows so that placements can change (a stage moves from the laptop to the cloud or back) without rewriting it.

## Vocabulary
| Term | Meaning |
|---|---|
| **Stage** | One processing step with a frame-in / frame-out contract, e.g. `capture`, `swap` (JoyAI), `facefix` (lip retarget), `playout` |
| **Host** | Where a stage runs: `local` (the streaming machine: camera, mic, display) or `remote:<name>` (a rented GPU machine) |
| **Transport** | How frames move between hosts: H.264 over WebSocket (JoyAI protocol) today; in-process between stages on the same host |
| **Placement** | The mapping stage → host for one run, chosen by command-line flags today |
| **Sidecar data** | Per-frame metadata carried alongside frames: `seq` (capture frame number), `t_capture_ms`, `source_seq` (which input frame an output frame was made from) |

## Rules
1. **Frames are identified by `seq`.** The capture stage numbers frames; every later stage passes the number on
   (JoyAI returns it as `source_seq`). A stage that needs the original camera frame (e.g. facefix needs the real
   face) looks it up by `seq` in a bounded cache on its own host; it never asks another host for it.
2. **Heavy, model-bound stages go remote; stages that need local data or add little GPU time stay local.** Local
   data = camera frames, mic audio, the display. Moving a stage changes bandwidth (raw frames must then cross the
   network) and GPU budget on the destination.
3. **No stage may block playout.** Late results are dropped and the previous stage's output plays instead (e.g.
   facefix miss → raw JoyAI frame). Playout runs from a jitter buffer with adaptive rate.
4. **Every stage reports**: per-frame time, hit/miss (for optional stages), and its host. The bridge prints one
   status line every 10 s and a JSON summary at exit.
5. **Optional stages default to off** (complete bypass) and are switched on by a flag.
6. **One process per host per pipeline.** Locally that's `cloud/client/live_bridge.py`; remotely it's the
   pipeline's server (`run_server.sh`).

## JoyAI character swap (the current hybrid pipeline)
```
 local (laptop, RTX 5080)                         remote (RTX 5090: Vast / RunPod)
 ┌──────────────────────────┐   H.264 + seq      ┌────────────────────────────────┐
 │ capture  /dev/video0     │ ─────────────────► │ swap: JoyAI-Video-Edit (16B)   │
 │  └─ cache frame + face   │                    │  8-frame chunks, 2 steps       │
 │     analysis by seq      │ ◄───────────────── │                                │
 │ facefix (optional)       │  H.264 + source_seq└────────────────────────────────┘
 │  LivePortrait lip pass   │
 │ playout (jitter buffer)  │ → ffplay window / /dev/video10 / mp4
 └──────────────────────────┘
```
| Stage | Host | GPU time / frame | Why there |
|---|---|---|---|
| capture | local | – | the camera is here |
| swap (JoyAI) | remote 5090 | ~37 ms (90–99 % of the card) | needs a 32 GB Blackwell card |
| facefix | local 5080 | ~25 ms (128 px) | needs the real face (local) + the remote GPU has no headroom |
| playout | local | – | the display is here |

Measured: 25–26 fps; ~1.0 s glass-to-glass without facefix (+~40 ms with it). Moving facefix to the remote 5090 is
estimated to drop the pipeline to ~18–20 fps (or ~23–24 fps with a TensorRT LivePortrait).

Run it: see `cloud/README.md` (Quickstart) and `cloud/facefix/README.md` (facefix flags).

## Adding or moving a stage
- **New stage on the local host:** implement it in `cloud/facefix/`-style module with `analyze_input` (at capture,
  optional) / `analyze` / `render`, run it from a worker pool in `live_bridge.py` behind a flag, obey rule 3.
- **New remote stage / new pipeline:** a folder `cloud/pipelines/<name>/` with `setup.sh` (GPU check, artifacts from
  HF), `run_server.sh`, `README.md`. Speak the JoyAI WebSocket protocol (start/frame_meta/frames/ack/ping →
  started/output_frame(source_seq)/frames) or add an adapter in the bridge.
- **Moving a stage remote:** the remote side then needs the inputs it used locally (e.g. the raw camera frame for
  facefix, already sent to JoyAI) and its GPU budget must fit; document the measured fps change in the pipeline README.

## Not done yet
- A declarative placement file (`pipeline.yaml`: stages, hosts, flags) instead of command-line flags.
- Remote-side optional stages (facefix on the GPU machine) as a server-side plugin.
- More than two hosts (e.g. a second GPU machine for facefix).
