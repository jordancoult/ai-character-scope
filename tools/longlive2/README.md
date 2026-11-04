# LongLive 2.0 live (toolbox): real-time interactive video on one RTX 5090

NVlabs **LongLive 2.0** (Wan2.2-TI2V-5B, NVFP4, causal autoregressive) as a live tool: a server on a rented RTX 5090
streams 1024×576 video at a locked 24 fps to the laptop, and you steer it while it runs:

| Input | What it does |
|---|---|
| **Text prompt** (type it) | smooth change (KV re-cache: the new prompt takes hold within ~2 s, the scene evolves) or `/cut` = hard scene cut (~1.3 s) |
| **Reference image** (`--image` / `/image`) | the stream starts from your image (e.g. a person: they talk, laugh, wave, keep their identity) |
| **Webcam / video** (`--mode v2v`) | your video re-imagined with the prompt, keeping your pose/motion (`/strength 0.9–0.98`) |
| **Interaction** (`/person on`) | you stay (low noise), everything else becomes the prompt's scene: your arms and hands are in it |
| **Camera motion prompts** | "the camera moves forward / turns left ..." gives real camera motion, not precise (stretch goal) |

Measured 2026-11-04 (`experiments/2026-11-04-longlive2-5090/FINDINGS.md`, demos there): 33 fps generation (24 fps
locked playout, headroom for v2v and re-caches), prompt change visible after **1.8–2.6 s** (smooth) / **1.3 s** (cut),
webcam input → output **3.0 s**. The model's 8-latent chunk (32 frames = 1.33 s) is the latency floor: 4-latent chunks
were tested and break the image.

## Run
```bash
# 1. rent an RTX 5090 (Vast: cloud/infra/vast.sh search / create ID; ~$0.6/h), then on the box (as root):
#    copy this folder to /workspace/tools/longlive2 (rsync), then
HF_TOKEN=... bash /workspace/tools/longlive2/setup_5090.sh          # ~10 min; kernels from our HF artifacts
nohup bash cloud/infra/vast_watchdog.sh 45 8 &                      # self-stop if idle 45 min / after 8 h (copy it too)
cd /workspace && LL2_ROOT=/workspace /root/ll2/bin/python tools/longlive2/ll2_server.py --ckpt S2 --kvq   # ~20 s warm-up
# 2. on the laptop (opens the SSH tunnel itself; ffplay window; type to steer):
cloud/facefix/.venv/bin/python cloud/client/ll2_client.py --ssh "ssh -p PORT root@HOST" \
    --prompt "a cinematic shot of a cozy cabin, crackling fireplace"                      # text
    #  --image assets/blonde-gamer-girl.png --prompt "a young woman smiles and talks"     # reference image
    #  --mode v2v --input /dev/video0 --strength 0.95 --prompt "..."                      # webcam (or a clip)
    #  --mode v2v --input /dev/video0 --person --strength 0.98 --prompt "a man in a jungle" # interaction
```
While it runs, type + Enter: any text = new prompt · `/cut TEXT` hard cut · `/strength 0.95` · `/mode t2v|v2v` ·
`/person on|off` · `/image PATH` (restart from an image) · `/reset` · `/q`. Add `--output out.mp4` (or
`ffplay,out.mp4`) to record, `--side-by-side` (v2v: input | output, input matched frame by frame), `--script
"10:new prompt;20:/cut other scene"` for unattended demos. The client logs every change's visible latency.

Server switches (`ll2_server.py --help`): `--ckpt S2|S4` (2-step default; 4-step = richer but needs `--sage` to stay
above 24 fps, and SageAttention smears faces in motion), `--kvq` (NVFP4 KV: needed with the T5 on the GPU on 32 GB),
`--sage`, `--height/--width` (1024×576 is the quality/speed sweet spot; 1280×704 only reaches 24 fps with Sage and
grows artifacts), `--ahead 34` (unplayed frames kept at the client: covers one chunk's generation), `--recache-frames 16`
(0 = no re-cache on prompt change: faster switch, the new prompt barely takes hold), `--clean-every 1` (clean KV pass
every chunk: best consistency). Env `LL2_PERSON_KEEP=0.6`: noise level kept on the person in interaction mode.

## How it works
- `ll2_stream.py` (`LL2Engine`): NVlabs' pipeline with its per-chunk loop re-implemented so inputs can change between
  chunks; KV cache, attention sink and scheduler are upstream's. Prompt change = re-encode (T5 on the GPU, cached) +
  rebuild the KV of the last 16 latent frames under the new prompt; hard cut = clear the KV **including the attention
  sink** (else the old scene bleeds back). I2V = the image's TAE latent clamped as the first latent. V2V = the chunk's
  frames through the streaming TAE encoder, noised to `strength`, then few-step distilled sampling (x0 = x − σv,
  re-noise); a per-pixel strength map does the person-keep. Output through the streaming TAE decoder (seamless).
- `ll2_server.py`: websocket server, generation paced on the frames the client has consumed (not a server clock), a
  change aborts the chunk in progress, JPEG encoding off the generation thread. `cloud/client/ll2_client.py`: tunnel,
  input capture (+ MediaPipe person mask), 24 fps playout, drops stale frames after a change ("snap").
- `ll2_stream_test.py`: offline demos/timing of every mode on the box. `ll2_generate.py`: the original benchmark.
- Kernels: flash-attn 2.8.3, SageAttention 2.2, FourOverSix (NVFP4) prebuilt for sm_120 / py3.12 / torch 2.9.1 cu128 in
  HF `JordanCo/ai-livestream-artifacts/longlive2/sm120_py312_torch291cu128/`.

## Limits
- Smooth prompt changes are gradual (the scene evolves; subtle edits like "snow starts falling" barely show); a cut
  changes everything (an I2V person becomes someone else: use smooth changes to keep them).
- V2V keeps pose/layout well offline; live, the first seconds are weak and later prompt changes transform the scene
  less than a fresh start. Interaction mode works offline and live (person kept at 0.6: you + a new background), but
  a person keep of 0.5 with background 0.98 glitched once (text/sign artifacts); prompt changes in v2v land partially.
- No real directional/action control: SCOPE (Wan2.2-5B action model) is a bidirectional 4 s clip model, not streamable.

---

# LongLive 2.0 on the RTX 5080 laptop (2026-11-01 tests)
Results are in `experiments/2026-11-01-longlive2/FINDINGS.md`. The same scripts run on the laptop through Scope's venv:

```bash
cd ~/personal/ai-livestream   # repo root (your clone folder)
git clone --depth 1 https://github.com/NVlabs/LongLive.git longlive2            # gitignored
git -C longlive2 apply ../tools/longlive2/skip_clean.patch                       # LL2_SKIP_CLEAN switch
SKIP_CUDA_BUILD=1 bin/uv pip install --target longlive2/.deps --python scope/.venv/bin/python \
    --no-build-isolation --no-deps ./longlive2/LongLive2.0/fouroversix
bin/uv pip install --target longlive2/.deps --python scope/.venv/bin/python --no-deps sentencepiece
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
Recommended laptop run (1024×576, 2 steps, clean pass every 2nd chunk, compiled, about 15 fps):
```bash
LL2_LOCAL=24 LL2_CLEAN_EVERY=2 LL2_COMPILE=1 LL2_KVQ=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  scope/.venv/bin/python tools/longlive2/ll2_generate.py /tmp/out.mp4 2 16 576 1024 "beautiful woman"
```
`ll2_generate.py` switches: `LL2_KVQ=1` NVFP4 KV · `LL2_LOCAL=N` window · `LL2_SKIP_CLEAN=1` / `LL2_CLEAN_EVERY=N` ·
`LL2_COMPILE=1` · `LL2_SAGE=2` · `LL2_CKPT=path` (e.g. the S4 4-step checkpoint). Arguments: `OUT STEPS WARMUP_LATENTS H W [PROMPT]`.
