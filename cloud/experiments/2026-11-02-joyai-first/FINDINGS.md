# JoyAI-Video-Edit: first character-swap tests (2026-11-02)

Setup GPU: **RTX PRO 4500 Blackwell** (32 GB, sm_120, about half an RTX 5090's compute) in RunPod EU-RO-1, using the
upstream RTX 5090 settings (SageAttention, fp16-accumulate FP8, low-VRAM layout). 840×480, 2 steps, checkpoint 0811,
no prompt enhancement, detection gates off. Simulated webcam: `cloud/client/joyai_client.py` streams a stock clip in
real time, either from the laptop (SSH tunnel to Europe) or on the pod.

| Run | Ref | Prompt | fps | Latency median / p95 | Look |
|---|---|---|---|---|---|
| `gstar_face_5941016` | `char_face.png` | generic "replace the main subject with the character from the reference image" | 10.2 | 1.35 / 1.60 s (laptop) | Only the cap and pink colour transfer; human face stays |
| `pe_face` | `char_face.png` | generic + built-in prompt enhancer | 11.5 | 1.37 / 1.60 s | Worse: a different human in a pink polo |
| `explicit_face` | `char_face.png` | **explicit description** of G-Star | 11.5 | 1.37 / 1.61 s | **G-Star**: cone head, cap, chain, gestures followed; a bit human-bodied |
| `explicit_front` | `char_front.png` | explicit description | 9.6 | 1.37 / 1.61 s | **Best**: cartoon G-Star, smile, cap, chain; hands slightly human |
| `human_7643311` | `blonde-gamer-girl.png` | explicit description | 12.9 | 1.39 s | Identity very close to the ref, but she barely moves (gestures lost); only 89 frames came back, rerun |
| on-pod (same as `explicit_front`) | | | 11.5 | **1.15 / 1.41 s** (pod only) | the Europe round trip adds about 0.2 s |

Key learnings
- **Describe the character explicitly in the prompt** (shape, face, clothing, accessories). The generic RV2V
  prompt only transfers accessories. The built-in prompt enhancer (MiMo) hurt identity.
- A clean full-body reference on a white background works best for G-Star.
- Motion/gesture following is excellent for the cartoon target; identity is strong for the human target.
- Output arrives in 8-frame bursts (one chunk), so the player must pace it (gap p95 about 1.1 s at 10 fps).
- First server start on a new GPU type takes about 60 min (torch.compile + autotune for 49 shapes, written to the
  network volume). Cached for the next start on the same GPU type.

## Long runs (45 s, G-Star full-body ref, explicit prompt)
- `long_reset600` (server default: memory reset every 600 input frames = 25 s): 10.6 fps; no visible jump or
  blackout at the reset (`reset_window.jpg`).
- `long_noreset` (`kv_reset_frames: 0`): 11.2 fps; identity holds for all 45 s, no drift (`noreset_strip.jpg`).
- Head comes out rounder/shorter than the reference's tall cone.

## Per-chunk profile on the PRO 4500 (8 frames per chunk, `profile_timings`)
| Stage | Time |
|---|---|
| DiT (2 steps) | 0.51–0.56 s, caps this card at ~15 fps |
| VAE encode | 0.04–0.13 s |
| VAE decode | 0.08–0.20 s |
| JPEG encode (MJPEG downlink used by the test client) | 0.011 s / 0.356 s alternating, to investigate (H.264 path differs) |
| Server residence per chunk | 0.9–1.14 s |
The stages are pipelined, so throughput ≈ the slowest stage plus contention. An RTX 5090 has about 2× this card's
compute → DiT ~0.27 s per chunk → about 30 fps for the DiT alone [est].

## Checkpoint `1101_face++` (released 2026-11-01, undocumented) vs `0811`
- G-Star (`facepp_front`): about the same quality; the cap logo renders as a clean "G-Star". 11.6 fps.
- Human (`facepp_human`): **clearly better**. Identity is very close to the ref, and head tilts, expressions and
  hand-to-face gestures are followed (0811 gave a near-static person). 9.5 fps.
- **Adopted as the default** (`run_server.sh`, `setup_pod.sh`). Startup with the existing compile cache: ~4 min.
