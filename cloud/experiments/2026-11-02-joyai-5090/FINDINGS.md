# JoyAI-Video-Edit on an RTX 5090 (RunPod secure, EU-RO-1, $1.19/h) — 2026-11-02

Checkpoint `1101_face++`, 840×480, 2 steps, upstream RTX 5090 settings (SageAttention, fp16-accumulate FP8,
low-VRAM layout), G-Star full-body ref + explicit prompt, desk clip `webcam_5941016` as a simulated camera.

| Where the client runs | Wire codec | Input fps | Output fps | Latency median / p95 | Notes |
|---|---|---|---|---|---|
| on the pod | MJPEG | 24 | 22.6 | 0.48 / 0.60 s | input-rate bound |
| on the pod | MJPEG | 30 | **25.2** | **0.46 / 0.58 s** | |
| laptop → EU (SSH tunnel) | MJPEG | 24 | 19.7 | 0.71 / 5.8 s | bandwidth-bound: server flow-control drops |
| laptop → EU (SSH tunnel) | **H.264** | 30 | **25.5** | **0.69 / 0.78 s** | no drops; same look (`sbs_gstar_remote_h264.mp4`) |

Per chunk (8 frames): VAE encode 26 ms, DiT 215 ms, VAE decode 46 ms, JPEG 7 ms → 0.30 s per chunk → ~26.5 fps
ceiling at 840×480. The DiT is 70 % of it.

**Meets the goal at 840×480:** 25+ fps, latency far below 2 s (0.46 s on the pod, 0.69 s including a
transatlantic round trip), upper body with arms, character from a reference image.

Startup on a fresh 5090 pod: env ~5 min (cached kernel wheels); first compile ~12 min with the cache on local disk
(was ~60 min directly on the network volume). Cache saved to the volume for later starts.

## More 5090 runs (laptop → EU, H.264, 30 fps input)
| Run | Size | fps | Latency median / p95 | Look |
|---|---|---|---|---|
| `human` (blonde ref, `webcam_7643311`) | 840×480 | 23.4 | 0.69 / 0.89 s | very close identity; hand gestures followed well |
| `portrait_gstar` (boxer clip, fast side-on motion) | 480×840 | 25.4 | 0.68 / 0.78 s | body/chain transfer; face unstable in profile; a hard non-webcam clip |
| `portrait_desk` (desk clip cropped 9:16) | **480×840** | **25.6** | **0.68 / 0.78 s** | best G-Star so far: sleepy half-open eyes like the ref, cap, chain, gestures |

## Live bridge (`cloud/client/live_bridge.py`): camera/clip → pod → smooth playout → v4l2 / mp4
Server output arrives in 8-frame bursts, so playout runs from a jitter buffer with an adaptive rate (±25 %, gain
0.05 per frame of queue error, target 6 frames). Portrait G-Star, desk clip as camera, 40 s, laptop → EU:
| Playout | Dropped / repeated | Glass-to-glass median / p95 |
|---|---|---|
| fixed 25 fps, buffer 6 | 62 / 0 | 1.36 / 1.41 s (queue crept to 17) |
| adaptive, gain 0.02, target 8 | 0 / 0 | 1.18 / 1.22 s |
| **adaptive, gain 0.05, target 6** | **0 / 0** | **1.04 / 1.07 s** |
Glass-to-glass = capture on the laptop → frame shown by the player (network both ways + model + playout buffer).
