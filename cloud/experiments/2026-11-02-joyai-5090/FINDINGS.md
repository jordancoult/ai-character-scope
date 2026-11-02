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
