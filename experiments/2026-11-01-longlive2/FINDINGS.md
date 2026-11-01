# LongLive 2.0 (Wan2.2-TI2V-5B) on the RTX 5080 laptop: findings (2026-11-01)

## Update: the user's verdict
960×544 shows artifacts (random candles), so the model doesn't like that resolution. 1024×576 looked good. Treat **1024×576 as the minimum.**

At 1024×576, window 24, compiled (`clean_modes_1024.jpg`):
| Clean-KV pass | fps | Consistency at 20 s |
|---|---|---|
| every chunk | 12.7 | best |
| **every 2nd chunk** (`LL2_CLEAN_EVERY=2`) | **15.2** | **as good as every chunk**: same woman, same room |
| never (`LL2_SKIP_CLEAN=1`) | 18.7 | the background drifts (the wallpaper changes) |

- **SageAttention v1** (Triton, int8 QK): +15% (16.9 → 19.4 fps), but it **damages the image**: a different face and neon, oversaturated hair over time (`sage_cmp.jpg`). Rejected.
- **Profile at 1024×576:** attention 48%, FP4 GEMM 29%, elementwise ops and KV copies 22%.


These tests ran outside Scope. The script is `tools/longlive2/ll2_generate.py` and the setup is in `tools/longlive2/README.md`. Every test used a pure generator (no camera), the prompt "beautiful woman", the 2-step NVFP4 checkpoint (`model_4o6.pt`) and about 21 s clips of 128 latent frames (512 video frames). The fps figures are end-to-end: model plus decoder.

## First recipe (960×544, superseded by the update above)
| Setting | Value | Why |
|---|---|---|
| Steps | **2** (native) | 5 steps on this 2-step checkpoint falls apart into neon blocks. |
| Resolution | **960×544** (or 1024×576 at 17–19 fps) | Below ~1024×576 the model degrades after ~10 s: glitches at 640×352, a drifting extreme close-up at 832×480. Native 1280×704 is the most stable but runs at about 6 fps. |
| KV cache | **NVFP4** (`kv_quant: true`) | Cuts peak memory from 11.8 to 7.7 GB at 640×352. Above 640×352 nothing fits without it. |
| Memory window | **24** latent frames (stock 32) | 16 is faster (19.7 fps at 1024×576), but blobs appear by 20 s. 24 holds up. |
| Clean-KV pass | **skipped** (`LL2_SKIP_CLEAN=1`, `skip_clean.patch`) | One DiT pass fewer per chunk, +50% fps. Motion becomes more natural, and consistency drops a little (more drift; a glass grew over the face at native res). |
| Decoder | **TAEW2.2** (`taew2_2.pth`, 23 MB) | Decodes at 1,300 fps against 10 fps for the full Wan2.2 VAE, and looks almost the same (`vae_cmp.jpg`). |
| torch.compile | optional, `mode=default` | +9–11%. |

Measured at 2 steps, KV in NVFP4 and the TAE decoder unless noted:

| Resolution | Clean pass | Window | fps | Memory | At 20 s |
|---|---|---|---|---|---|
| 640×352 | on | 32 | 39–41 | 7.7 GB | glitches |
| 832×480 | on | 32 | 19.0 | 10.0 GB | drifts into an extreme close-up |
| 1280×704 (native) | on | 32 | 5.2 (5.6 with compile) | 14.3 GB | **solid** |
| 1280×704 | skipped | 32 | 7.8 | 14.2 GB | coherent, an object drifts in |
| 1024×576 | on | 32 | 10.1 | 11.0 GB | **solid** |
| 1024×576 | skipped | 32 | 14.9 | 11.0 GB | **holds**, more motion |
| 1024×576 | skipped | 24 | 16.9 (18.7 with compile) | 10.0 GB | **holds** |
| 1024×576 | skipped | 16 | 19.7 | 9.1 GB | blobs appear |
| 960×544 | skipped | 24 | 20.3 | 9.4 GB | the user saw artifacts (random candles) (`s960.jpg`) |

## Bugs and traps found on the way
- **The text encoder ran on the CPU every chunk.** The pipeline encodes one copy of the prompt per block, so a cache keyed on the whole list never hit. Caching per unique prompt took steady-state speed from 4.4 to 44 fps at 640×352.
- **The stock T5 loads in float32 (22 GB) and calls `.cuda()`.** The script builds a bf16 copy on the CPU instead.
- **Chunk-by-chunk VAE decoding** (`streaming_vae`) needs LightVAE; the stock Wan VAE crashes with it.
- **The full-resolution decode** has to be done in pieces and moved to the CPU, or the GPU runs out of memory.
- **FourOverSix installs real sm_120 kernels** even with `SKIP_CUDA_BUILD=1`, so no CUDA toolkit is needed.

## Not done yet
- **A Scope port** (live page and OBS output): LongLive 2.0 isn't a Scope pipeline. It has no VACE, so it's generator mode only.
- **Prompt switching:** LongLive 2.0 handles multi-shot prompts with a scene-cut prefix.
- **Latency:** the 8-latent chunk is 32 frames, roughly 1.5 s at 20 fps.

## LongLive 1 family on the same "beautiful woman" test (`lgl1_family_steps_grid.jpg`, 640×352)
- **3 steps with fast memory:**
  - LongLive 16.6 fps, MemFlow 16.0 fps, Reward Forcing 14.8 fps.
  - Looks as sharp as 5 steps (8.5–9.5 fps).
- **5 steps:** fast memory (skipping the clean-KV pass) gives +13–16% at no visible cost.
- **4-bit FFN:** +8–15%, but it brings artifacts (a striped background on MemFlow) and changes faces.
