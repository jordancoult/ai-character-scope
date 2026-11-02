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
