# Lip-sync (mouth movement transfer) in JoyAI — 2026-11-02

User report: mouth movements don't transfer, even with a human reference. Measured with `cloud/client/mouth_sync.py`
(MediaPipe FaceLandmarker: inner-lip gap / face height and `jawOpen`, output frame paired with the exact input
frame it came from via `source_seq`; Pearson correlation; lag search up to 16 frames).

Talking-head clip `assets/test_face_landscape.mp4`, human ref `blonde-gamer-girl.png`, RTX 5090 (Vast, California),
840×480, 20 s, H.264, 30 fps in, laptop client through an SSH tunnel.

| Run | Change | lip-gap corr | best lag (corr, frames) | out/in mouth motion | fps |
|---|---|---|---|---|---|
| A | server defaults (freeze-on-static on) | 0.10 | 0.21 @ 10 | 0.33 | 25.1 |
| B | `freeze_kv_on_static: false` | 0.16 | | 0.37 | 25.0 |
| **C** | B + prompt "Keep exactly the original person's facial expressions, mouth movements, lip shapes and speech, eye blinks, …" | **0.25** | 0.25 @ 0 | 0.30 | 24.7 |
| D | C prompt, **no reference image** | 0.21 | | 0.34 | 25.3 |
| E | C + 3 denoising steps | 0.12 | | 0.31 | 19.1 |

Conclusion: JoyAI (2-step, ckpt face++) transfers almost no lip motion; the mouth moves ~1/3 as much as the input and
is barely correlated. Not a delay (lag search), not the reference (D), not steps (E). Adopted C (freeze off in the
live bridge, mouth sentence in `prompts/*.txt`) as a small free gain. Real fix: a face pass after generation
(lip/expression retargeting from the input face) — see `cloud/facefix/`.
