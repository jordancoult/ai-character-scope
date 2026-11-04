# LongLive 2.0 on an RTX 5090: plan, state, progress — DONE 2026-11-04 (instance destroyed)

## Goals (user, 2026-11-04 night; user asleep, autonomous)
1. Set up **LongLive 2.0** (NVlabs, Wan2.2-TI2V-5B, NVFP4) on a rented **RTX 5090**.
2. **Best quality the 5090 allows at ≥ 24 fps**, and the **lowest latency** for quick prompt changes and other
   input changes.
3. Input options ("plugins"):
   - **text, live prompt changes** (LongLive's KV-recache);
   - **reference image** → video of the person in it (LongLive 2.0 I2V path: image latent clamped as the first latent);
   - **webcam video-to-video**: the generated video morphs the user's environment (SDEdit-style: encode webcam chunk,
     noise it, denoise the rest of the steps with the prompt).
4. Stretch: **directional control** (forward/left/back/right live, scene otherwise unchanged): prompt-based camera
   motion, or an action-conditioned Wan2.2-5B world model (LongLive-Plug lists **SCOPE**, z2tong/SCOPE).
5. Stretch: **webcam interaction**: the user's arms/hands in the generated scene affect it (grab things, touch water);
   on LongLive 1 this worked with enough noise on the webcam input that everything but the person is regenerated →
   spatially varying noise (low on the person, high on the background).
6. **Document** as a toolbox tool in this repo (`tools/longlive2/` + README), upload reusable assets (compile caches,
   wheels) to HF `JordanCo/ai-livestream-artifacts`, **demos** for the user to watch in the morning.
7. **Stop the GPU when done**; a watchdog must stop the box if this session freezes.

## Infra state
- Vast instance **59051978**, RTX 5090 32 GB, Virginia, **$0.605/h**, created 2026-11-04 03:41 UTC.
  SSH: `.tmp/g5.sh '<cmd>'` (ssh3.vast.ai:20516, key `.secrets/runpod_ed25519`; every call touches the heartbeat).
- **Watchdog** (`cloud/infra/vast_watchdog.sh`, running on the box, log `/workspace/watchdog.log`): self-stops the
  instance after **45 min without heartbeat** or at the **hard deadline 11:44 UTC** (8 h). Uses CONTAINER_API_KEY.
- Box layout: venv `/root/ll2` (py3.12, torch 2.9.1 cu128, flash-attn 2.8.3 + SageAttention 2.2 + FourOverSix copied
  from the laptop's sm_120 builds), code `/workspace/longlive2` (+ skip_clean.patch), scripts `/workspace/tools/longlive2`,
  weights `/workspace/models/LongLive-2.0-5B/{S2,S4}/model_4o6.pt`, `Wan2.2-TI2V-5B/` (VAE, T5, tokenizer), `fastvae/taew2_2.pth`.
- At the end: destroy (or stop) the instance: `cloud/infra/vast.sh destroy 59051978`.

## Known from the laptop (experiments/2026-11-01-longlive2/FINDINGS.md, RTX 5080 laptop)
- 1024×576 is the minimum resolution the user accepts (960×544 artifacts). Native 1280×704 most stable.
- 2-step NVFP4 ckpt (S2), TAEW2.2 decoder (1300 fps vs 10 for the Wan VAE), NVFP4 KV needed on 16 GB.
- Clean-KV pass every chunk = best consistency; every 2nd ≈ as good; skipping → drift.
- SageAttention 2 OK end-to-end (v1 damages faces). Window 32 (stock) > 24 > 16 (blobs).
- Chunk = 8 latent frames = 32 video frames = 1.33 s → latency floor for changes; test 4-latent chunks.

## Findings so far (5090)
- Bench (`ll2_generate.py`, 1024×576 unless noted, clean pass every chunk, window 32, compiled, end-to-end fps incl.
  TAE decode): S2 exact NVFP4-KV 31.7 · S2 exact bf16-KV 33.3 · S2 Sage2 44.5 · S4 exact 20.0 · S4 Sage2 bf16-KV 29.7 ·
  S2 1280×704 exact 18.5 · S2 1280 Sage2 27.5 (NVFP4 KV) / 28.5 (bf16 KV, 29.5 GB). Visual: S4+Sage2 ≈ S2-exact best;
  1280+Sage2 grows a glass object (rejected); S2-exact showed ghost hands once.
- bf16 KV + T5 on GPU (11 GB) OOMs on 32 GB → NVFP4 KV with the T5 on the GPU (fast prompt encodes).
- Streaming engine `tools/longlive2/ll2_stream.py` works for T2V / I2V / V2V / person-keep (`ll2_stream_test.py`):
  S4+Sage2 steady 26 fps (T2V/I2V), V2V 26–32 fps. Compile recompiles on new KV lengths → warm up.
- Prompt switching needs KV-recache (implemented: ~860 ms one-off per switch) — without it the new prompt barely
  takes hold. Hard cut must also clear the attention sink (`sink_size 8`, `multi_shot_sink`) or the old scene bleeds in.
- V2V: strength ≤ 0.85 barely changes the scene; use 0.9–0.98 (0.98 ≈ new scene following the input's layout/motion).
- I2V: starts from the image correctly; with strong motion (waving) the face smears (S4+Sage2) — testing S2-exact.

## Steps / status
- [x] Rent 5090, watchdog, env, weights (03:41–03:55 UTC)
- [x] Benchmark quality options at 1024×576 / 1280×704 (bench script `/workspace/ll2_bench.sh`, log `/workspace/bench.log`,
      videos `/workspace/bench/`): A S2 KVQ, B S2 bf16 KV, C S2 Sage2, D S4 4-step, E S2 1280, F S2 1280 Sage2
- [x] Pick the best-quality config ≥ 24 fps (end-to-end incl. TAE decode)
- [x] Streaming engine `tools/longlive2/ll2_stream.py`: persistent model + KV, chunk loop, prompt switch (T5 on GPU,
      optional recache), I2V start, V2V (TAE encoder + noise strength), block size 8 vs 4 (latency)
- [x] Server (websocket) + client (`cloud/client/ll2_client.py`: webcam/file/image in, ffplay out, typed prompt + params)
- [x] Demos: T2V prompt changes, ref image → person video, webcam v2v morph (use assets/ clips, not the webcam)
- [x] Stretch (partial, see FINDINGS §4): directional prompts / SCOPE; spatial-noise interaction
- [x] Docs: `tools/longlive2/README.md` (toolbox entry), FINDINGS.md here; HF upload of compile caches if useful
- [x] Destroy the instance (destroyed 2026-11-04 ~06:05 UTC; 2.3 h at $0.753/h incl. storage ≈ $1.75); final report sent
