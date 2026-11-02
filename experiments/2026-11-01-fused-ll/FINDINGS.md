# Faster 2-step LongLive / Causal Forcing (2026-11-01)

Desk stock clip, 640×352, Causal Forcing chunk-wise via the Reward Forcing pipeline, live through Scope + bridge.

| Setting | depth off | depth on |
|---|---|---|
| before: 2 steps, clean every 2nd | (~16) | 13.0 |
| 2 steps, fused + RoPE cache + clean every 2nd | **22.6** | 15.1 |
| 1 step, fused + RoPE cache | **~26** | 17 |

Standalone (`tools/bench_longlive.py --pipeline reward-forcing`, 2 steps, depth off): stock 16.5 → fused 16.2 →
+RoPE cache 17.7 → +float32 RoPE 18.6 → +clean every 2nd 22.9 fps.

## What changed (all off by default)
- `SCOPE_FUSED_KV=1` now works for LongLive and Reward Forcing: the clean-KV pass rides in the next chunk's first
  denoising forward. Bit-identical to stock (`tools/check_fused_ll.py`), text and video mode. It saves overhead only:
  the merged pass still processes the previous chunk, so on its own it's ~0–15%.
  - Prefix attends to the pre-eviction cache (what a real clean pass sees).
  - Reward Forcing's first video chunk overlaps the next by one frame, so that one chunk keeps a real clean pass.
  - Honors `clean_kv_every` (bit-identical to stock every-Nth).
- `SCOPE_RF_ROPE_CACHE=1` (Reward Forcing / Causal Forcing): cached keys stay RoPE-rotated at absolute frame positions
  instead of re-rotating the whole window in float64 in every layer; RoPE in float32. Same math; bf16-rounding-level
  differences (0.5–1 % mean pixel diff after the window rolls).

## Where the time goes now (2 steps, per chunk)
fp8 GEMMs ~215 ms, attention (Sage) ~130 ms, VAE ~50 ms, rest elementwise/copies. Depth/VACE adds ~45%.

Run it:
```
MODEL=reward-forcing SCOPE_FUSED_KV=1 SCOPE_RF_ROPE_CACHE=1 \
SCOPE_RF_GENERATOR=$PWD/models/Causal-Forcing/chunkwise/causal_forcing.pt SCOPE_RF_GENERATOR_KEY=generator \
ORIENTATION=landscape BRIDGE_ARGS="--size 352 640" ./start_scope.sh --bridge
```
then Denoiser: Steps 2, Clean pass every 2nd chunk.
