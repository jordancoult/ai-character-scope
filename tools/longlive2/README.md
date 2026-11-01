# LongLive 2.0 standalone setup (reproduce the 2026-11-01 tests)

Everything stays in the project folder, and Scope's environment is used without changing it. Results are in `experiments/2026-11-01-longlive2/FINDINGS.md`.

```bash
cd ~/personal/ai-character-scope
git clone --depth 1 https://github.com/NVlabs/LongLive.git longlive2            # gitignored
git -C longlive2 apply ../tools/longlive2/skip_clean.patch                       # LL2_SKIP_CLEAN switch
# FourOverSix (NVFP4 kernels; installs prebuilt sm_120 kernels) + sentencepiece into a side folder
SKIP_CUDA_BUILD=1 bin/uv pip install --target longlive2/.deps --python scope/.venv/bin/python \
    --no-build-isolation --no-deps ./longlive2/LongLive2.0/fouroversix
bin/uv pip install --target longlive2/.deps --python scope/.venv/bin/python --no-deps sentencepiece
# weights (~6 GB + 0.9 GB): 2-step NVFP4 generator, Wan2.2 VAE/tokenizer; the T5 is the Wan2.1 one (same hash)
scope/.venv/bin/hf download Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2 model_4o6.pt --local-dir models/LongLive-2.0-5B
scope/.venv/bin/hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth config.json configuration.json \
    google/umt5-xxl/special_tokens_map.json google/umt5-xxl/spiece.model google/umt5-xxl/tokenizer.json \
    google/umt5-xxl/tokenizer_config.json --local-dir models/LongLive-2.0-5B/Wan2.2-TI2V-5B
ln -sf ../../sdv2/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth models/LongLive-2.0-5B/Wan2.2-TI2V-5B/
mkdir -p models/LongLive-2.0-5B/fastvae && curl -sL -o models/LongLive-2.0-5B/fastvae/taew2_2.pth \
    https://github.com/madebyollin/taehv/raw/main/taew2_2.pth && curl -sL -o models/LongLive-2.0-5B/fastvae/taehv.py \
    https://raw.githubusercontent.com/madebyollin/taehv/main/taehv.py
mkdir -p longlive2/LongLive2.0/wan_models && ln -sfn ../../../models/LongLive-2.0-5B/Wan2.2-TI2V-5B longlive2/LongLive2.0/wan_models/Wan2.2-TI2V-5B
```

Recommended run (960×544, 2 steps, about 20 fps, roughly 21 s clip, saves the mp4 and the frame at 20 s):
```bash
LL2_LOCAL=24 LL2_SKIP_CLEAN=1 LL2_KVQ=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  scope/.venv/bin/python tools/longlive2/ll2_generate.py /tmp/out.mp4 2 16 544 960 "beautiful woman"
```
Switches:
- `LL2_KVQ=1`: NVFP4 KV cache.
- `LL2_LOCAL=N`: memory window in latent frames (stock 32).
- `LL2_SKIP_CLEAN=1`: drop the clean-KV pass.
- `LL2_COMPILE=1`: torch.compile; set `LL2_COMPILE_MODE` to change the mode.

Arguments: `OUT STEPS WARMUP_LATENTS H W [PROMPT]`.
