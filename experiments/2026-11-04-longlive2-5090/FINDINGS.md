# LongLive 2.0 live on an RTX 5090: findings (2026-11-04)

Tool docs: `tools/longlive2/README.md`. Plan/state log: `PLAN.md`. Demos (gitignored, sent to the user): `demos/`.
Box: Vast RTX 5090 32 GB (Virginia, $0.605/h), torch 2.9.1 cu128, flash-attn 2.8.3 / SageAttention 2.2 / FourOverSix
prebuilt for sm_120.

## 1. Quality vs speed (offline benchmark, `ll2_generate.py`, ~21 s clips, end-to-end fps incl. TAE decode)
All: clean KV pass every chunk (best consistency), window 32 (stock), torch.compile, prompt "beautiful woman".

| Config | fps | VRAM | Look |
|---|---|---|---|
| 2-step, 1024×576, exact attn, NVFP4 KV | 31.7 | 11 GB | consistent; ghost hands once |
| 2-step, 1024×576, exact, bf16 KV | 33.3 | 21 GB | same |
| 2-step, 1024×576, SageAttention 2 | 44.5 | 11 GB | |
| 4-step, 1024×576, exact | 20.0 | 11 GB | too slow |
| **4-step, 1024×576, Sage2, bf16 KV** | 29.7 | 21 GB | richest; **smears faces in motion** (I2V wave) |
| 2-step, 1280×704 (native), exact | 18.5 | 14 GB | too slow |
| 2-step, 1280×704, Sage2 | 27.5–28.5 | 15–30 GB | a glass object grows, painterly walls: rejected |

**Chosen: 2-step, exact attention, NVFP4 KV, 1024×576, clean pass every chunk** (~33 fps live). It handles motion
best (I2V person waving stays clean, the 4-step+Sage version smears). bf16 KV doesn't fit next to the T5 on the GPU
(32 GB OOM); the T5 on the GPU is what makes prompt changes fast (~20 ms encode instead of seconds on the CPU).
4-latent chunks (16 frames, would halve latency) break the image (glitch objects, ribbons): the model needs 8.

## 2. Live (server on the box, client on the laptop over an SSH tunnel, 24 fps playout)
| Demo (`demos/`) | Result |
|---|---|
| `live_t2v_prompts.mp4` (text: 3 changes + a cut) | 33 fps generation; changes visible after 2.6 / 2.2 s, cut 1.3 s |
| `live_i2v_person.mp4` (start from `assets/blonde-gamer-girl.png`) | starts exactly from the photo; smiles, waves, puts on sunglasses on command, identity kept; the cut → a different woman on a beach |
| `live_v2v_person.mp4` (desk clip as webcam, side-by-side matched) | pose/hands follow; scene changes partially; change latency 1.8 s; input→output 3.1 s |
| `live_interaction_keep06.mp4` (**person kept at 0.6**, background 0.97) | **works**: the man + his exact gestures in front of a tropical beach at sunset; the switch to a spaceship lands partially (1.98 s) |
| `live_interaction.mp4` (person kept at 0.5, background 0.98) | person + gestures kept, but the background glitched into a text sign → default keep is now 0.6 |
| `live_directional.mp4` (camera-motion prompts every 7 s) | real forward/turn motion, imprecise; a hallucinated hand + gadget |
| `offline_*` | same engine, scripted on the box: v2v at strength 0.90/0.95/0.98, person-keep (clean jungle waterfall), I2V, prompt changes + clean cut |

Latency anatomy (8-latent chunk = 32 frames = 1.33 s): a change waits for the chunk in progress (aborted between
denoising passes), KV re-cache (~0.55 s for 16 latent frames), one chunk (~1.0 s), network + decode → 1.8–2.6 s
(smooth), 1.3 s (cut). Webcam input additionally needs a chunk's worth of frames (1.33 s) → ~3 s input→output.

## 3. What mattered (bugs found on the way)
- **KV re-cache on prompt change** is essential: without it the new prompt barely shows (snow / camera prompts did
  nothing). Re-caching the full 32-frame window costs ~1.1 s (playout stall); the last 16 frames are enough (~0.55 s).
- **Hard cut must clear the attention sink** (`sink_size 8`, `multi_shot_sink`): upstream's `_zero_kv_data` keeps it,
  and the old scene bleeds back (melting cabin under the beach).
- **V2V strength**: ≤ 0.85 keeps the input scene almost unchanged (office stays an office); 0.90–0.98 is the useful
  range (0.90: new room, exact pose; 0.98: new scene, pose roughly). The 2-step schedule only has two noise levels,
  so V2V uses its own few-step sampler (x0 = x − σv, re-noise) for a continuous strength.
- **Pacing on the client's consumption**, not a server clock: the clock version let the client buffer drift to 60–70
  frames (~2.5 s extra latency). Plus "snap" (drop stale frames after a change) and aborting the chunk in progress.
- Live side-by-side must pair each output frame with its own input frame (ids through the protocol), otherwise the
  latency makes v2v look unfaithful.
- torch.compile recompiles for each new KV length (first chunks, after a cut): warm-up covers it.

## 4. Stretch goals
- **Webcam interaction** (arms/hands in the generated scene): person-keep via a MediaPipe mask → per-pixel strength.
  Works offline (`offline_interaction_person_keep_sbs.mp4`) and live with keep 0.6 (`live_interaction_keep06.mp4`);
  keep 0.5 + background 0.98 glitched once. Next: send the mask aligned with the chunk's frames (now: latest mask).
- **Directional control**: no streamable action model for this backbone (SCOPE is a bidirectional 4 s clip model at
  30 steps; LongLive-Plug's few-step LoRA makes it 4 steps, still not causal). Camera-motion prompts give some motion.

## 5. Cost
Vast RTX 5090 at $0.605/h, ~3.5 h (03:41–~07:15 UTC) ≈ $2.1, plus ~$0.2 downloads.
