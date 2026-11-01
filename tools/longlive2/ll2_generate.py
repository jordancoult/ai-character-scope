"""LongLive 2.0 (Wan2.2-TI2V-5B, NVFP4 2-step ckpt) standalone test on the RTX 5080.
usage: ll2_run.py OUT.mp4 STEPS LATENT_FRAMES [H W] [prompt]
Runs from the LongLive2.0 repo dir (it uses relative wan_models/ paths). Text encoder stays on CPU."""
import os, sys, time
REPO = os.path.expanduser("~/personal/ai-character-scope/longlive2/LongLive2.0")
sys.path[:0] = [os.path.join(REPO, "..", ".deps"), REPO]
os.chdir(REPO)
import logging, warnings
import torch
from omegaconf import OmegaConf

# torch.compile can't fully compile the 4-bit layers / cache bookkeeping: harmless graph breaks and
# recompile-limit notices. Keep them out of the console (set LL2_VERBOSE=1 to see them).
if os.environ.get("LL2_VERBOSE") != "1":
    torch._logging.set_logs(dynamo=logging.ERROR, inductor=logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)

out, steps, nlat = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
H, W = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (352, 640)
prompt = sys.argv[6] if len(sys.argv) > 6 else "beautiful woman"

import wan_5b.modules.causal_model as cm
SAGE = os.environ.get("LL2_SAGE", "")
if SAGE:
    # LL2_SAGE=1: SageAttention v1 everywhere; =2: SageAttention 2 everywhere; =2first: SA2 on the first
    # denoising step only (later steps + clean pass keep exact flash-attention)
    if SAGE.startswith("2"):
        sys.path.insert(0, os.path.expanduser("~/personal/ai-character-scope/longlive2/.sage2/pkg"))
        for m in [m for m in sys.modules if m.startswith("sageattention")]:
            del sys.modules[m]
    from sageattention import sageattn
    _fa = cm.attention
    def _sage_attention(q, k, v, *a, **kw):
        if SAGE.endswith("first") and getattr(cm, "LL2_PASS_STEP", -1) != 0:
            return _fa(q, k, v, *a, **kw)
        return sageattn(q.contiguous(), k.contiguous(), v.contiguous(), tensor_layout="NHD", is_causal=False)
    cm.attention = _sage_attention
    print(f"[ll2] SageAttention mode {SAGE}", flush=True)
_orig = cm.CausalWanModel.from_pretrained
# The base Wan2.2 diffusion weights (20 GB) aren't downloaded: build from config.json only; the
# generator checkpoint supplies every weight afterwards.
cm.CausalWanModel.from_pretrained = classmethod(lambda cls, path, **kw: cls.from_config(cls.load_config(path), **kw))

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import prepare_single_prompt_inputs, save_video, setup_nvfp4_pipeline

cfg = OmegaConf.load("configs/nvfp4/inference_nvfp4.yaml")
cfg.checkpoints.generator_ckpt = os.path.expanduser("~/personal/ai-character-scope/models/LongLive-2.0-5B/model_4o6.pt")
cfg.model_quant_use_transformer_engine = False
cfg.model_quant_te_fallback_to_fouroversix = False
cfg.inference.kv_quant = os.environ.get("LL2_KVQ") == "1"
if os.environ.get("LL2_LOCAL"):
    cfg.model_kwargs.local_attn_size = int(os.environ["LL2_LOCAL"])
cfg.inference.vae_device = "cuda:0"
cfg.inference.streaming_vae = False
cfg.inference.sampling_steps = steps
cfg.torch_compile = os.environ.get("LL2_COMPILE") == "1"
if cfg.torch_compile:
    cfg.torch_compile_mode = os.environ.get("LL2_COMPILE_MODE", "default")
cfg.num_output_frames = nlat
cfg.data.image_or_video_shape = [1, nlat, 48, H // 16, W // 16]
config = normalize_config(cfg)
dev = torch.device("cuda")
torch.set_grad_enabled(False)

import utils.wan_5b_wrapper as ww
from wan_5b.modules.t5 import umt5_xxl
from wan_5b.modules.tokenizers import HuggingfaceTokenizer


class CpuT5(torch.nn.Module):
    """WanTextEncoder, but bf16 on the CPU (the stock one moves a float32 T5, 22 GB, to the GPU)."""
    def __init__(self):
        super().__init__()
        self.text_encoder = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16,
                                     device=torch.device("cpu")).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(torch.load("wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
                                                     map_location="cpu", weights_only=False))
        self.tokenizer = HuggingfaceTokenizer(name="wan_models/Wan2.2-TI2V-5B/google/umt5-xxl/", seq_len=512, clean="whitespace")

    device = property(lambda self: torch.device("cpu"))
    forward = ww.WanTextEncoder.forward


t0 = time.time()
pipe = CausalDiffusionInferencePipeline(config, device=dev, text_encoder=CpuT5())


class CpuTextEncoder(torch.nn.Module):
    """Keeps the 11 GB T5 on the CPU (hidden from .to()); caches embeddings per prompt list."""
    def __init__(self, enc):
        super().__init__()
        self._hidden = [enc]; self.cache = {}

    @property
    def enc(self):
        return self._hidden[0]

    def to(self, *a, **k):
        return self

    def forward(self, text_prompts):
        # The pipeline encodes one copy of the prompt per block: cache per unique prompt and stack.
        for t in set(text_prompts):
            if t not in self.cache:
                self.cache[t] = self.enc(text_prompts=[t])["prompt_embeds"].to(dev)
        return {"prompt_embeds": torch.cat([self.cache[t] for t in text_prompts], 0)}


pipe.text_encoder = CpuTextEncoder(pipe.text_encoder)
setup_nvfp4_pipeline(pipe, config, dev)
pipe.generator.model.eval().requires_grad_(False)
if os.environ.get("LL2_COMPILE") == "1":
    ok = pipe.generator.configure_torch_compile(mode=os.environ.get("LL2_COMPILE_MODE", "default"), dynamic=False, suppress_errors=True)
    print(f"[ll2] torch.compile enabled: {ok}", flush=True)
print(f"[ll2] loaded in {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

noise, prompts = prepare_single_prompt_inputs(config, prompt, dev)
pipe.text_encoder(prompts[0][:1])
def gen(n):
    nz = torch.randn([1, n, 48, H // 16, W // 16], device=dev, dtype=noise.dtype)
    pr = [[prompt] * (n // 8)]
    torch.cuda.synchronize(); t = time.time()
    pipe.inference(noise=nz, text_prompts=pr, return_latents=True)
    torch.cuda.synchronize(); return time.time() - t
sys.path.insert(0, os.path.expanduser("~/personal/ai-character-scope/models/LongLive-2.0-5B/fastvae"))
from taehv import TAEHV
import imageio, numpy as np
tae = TAEHV(os.path.expanduser("~/personal/ai-character-scope/models/LongLive-2.0-5B/fastvae/taew2_2.pth")).to(dev, torch.float16)
gen(64 if os.environ.get("LL2_COMPILE") == "1" else 16)  # warm-up (compiled: cover full-KV shapes)
nz = torch.randn([1, 128, 48, H // 16, W // 16], device=dev, dtype=noise.dtype, generator=torch.Generator(dev).manual_seed(1))
torch.cuda.synchronize(); t = time.time()
lat = pipe.inference(noise=nz, text_prompts=[[prompt] * 16], return_latents=True)
torch.cuda.synchronize(); tg = time.time() - t
parts = []
for i in range(0, lat.shape[1], 16):  # decode in pieces straight to CPU (full-res video doesn't fit on the GPU)
    parts.append(tae.decode_video(lat[:, i:i + 16].half(), parallel=False, show_progress_bar=False).cpu())
v = torch.cat(parts, 1)
torch.cuda.synchronize(); dt = time.time() - t
n = v.shape[1]
print(f"[res] DiT alone: {4 * lat.shape[1] / tg:.1f} fps", flush=True)
print(f"[res] {W}x{H}: {n} frames end-to-end (DiT + TAE decode) in {dt:.1f}s = {n / dt:.1f} fps, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
vid = (v[0].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
imageio.mimwrite(out, vid, fps=24, quality=8)
imageio.imwrite(out.replace(".mp4", "_20s.png"), vid[min(n - 1, 480)])
