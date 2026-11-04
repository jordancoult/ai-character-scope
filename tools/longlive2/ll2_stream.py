"""LongLive 2.0 streaming engine: persistent model + KV cache, chunk-by-chunk generation with live controls.

Wraps NVlabs' CausalDiffusionInferencePipeline (Wan2.2-TI2V-5B, NVFP4) and re-implements its per-chunk loop so the
inputs can change between chunks:
  - prompt: switch any time (T5 on the GPU, ~20 ms per new prompt, cached); `cut=True` = hard scene cut (LongLive's
    clean KV recache + pinned shot sink), otherwise a smooth transition;
  - reference image (I2V): the image's latent is clamped as the first latent of the first chunk;
  - input video (V2V): the chunk's input frames are encoded with the streaming TAE encoder, noised to `strength`
    (0 = keep the input, 1 = pure noise = text-to-video) and only the remaining denoising steps run;
  - spatial strength (`strength_map`, latent-resolution HxW in [0,1]): e.g. low on the person, high elsewhere.
Output frames are decoded with the streaming TAE decoder (TAEW2.2), seamless across chunks.

  eng = LL2Engine(height=576, width=1024)                # see __init__ for quality / speed switches
  eng.reset(prompt="a woman in a cafe", image=None)      # start a stream (optionally from a reference image)
  frames = eng.step()                                    # next chunk -> list of HxWx3 uint8 RGB frames
  eng.set_prompt("she starts to dance", cut=False)
  frames = eng.step(input_frames=webcam_frames, strength=0.7)   # V2V chunk (len = 4 * block frames)
Run from anywhere; set LL2_ROOT to the folder holding longlive2/ and models/ (default: this repo).
"""

import logging
import math
import os
import sys
import time
import warnings

ROOT = os.environ.get("LL2_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = os.path.join(ROOT, "longlive2", "LongLive2.0")
MODELS = os.path.join(ROOT, "models", "LongLive-2.0-5B")
sys.path[:0] = [os.path.join(REPO, "..", ".deps"), REPO, os.path.join(MODELS, "fastvae")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

if os.environ.get("LL2_VERBOSE") != "1":
    torch._logging.set_logs(dynamo=logging.ERROR, inductor=logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)


def _log(msg):
    print(f"[ll2] {msg}", flush=True)


class LL2Engine:
    def __init__(self, height=576, width=1024, ckpt="S2", steps=None, kv_quant=True, local_attn=32, block=8,
                 sage=False, compile=True, clean_every=1, t5_device="cuda", seed=0, recache_frames=16):
        """ckpt: S2 (2-step) | S4 (4-step) | path. block: latent frames per chunk (8 = native = 32 video frames;
        4 halves the reaction latency). clean_every: 1 = clean KV pass every chunk (best consistency), 2 = every
        2nd, 0 = never. sage: SageAttention 2 (faster, slightly less exact)."""
        cwd = os.getcwd()
        os.chdir(REPO)  # upstream uses relative wan_models/ paths
        try:
            self._build(height, width, ckpt, steps, kv_quant, local_attn, block, sage, compile, clean_every,
                        t5_device, seed)
            self.recache_frames = recache_frames
        finally:
            os.chdir(cwd)

    # ------------------------------------------------------------------ setup
    def _build(self, H, W, ckpt, steps, kv_quant, local_attn, block, sage, compile, clean_every, t5_device, seed):
        import wan_5b.modules.causal_model as cm
        self.cm = cm
        if sage:
            from sageattention import sageattn
            fa = cm.attention

            def _sage(q, k, v, *a, **kw):
                return sageattn(q.contiguous(), k.contiguous(), v.contiguous(), tensor_layout="NHD", is_causal=False)
            cm.attention = _sage
            self._fa = fa
        # the 20 GB base Wan2.2 weights aren't needed: build from config, the generator ckpt supplies every weight
        cm.CausalWanModel.from_pretrained = classmethod(
            lambda cls, path, **kw: cls.from_config(cls.load_config(path), **kw))
        from pipeline import CausalDiffusionInferencePipeline
        from utils.config import normalize_config
        from utils.inference_utils import setup_nvfp4_pipeline
        import utils.wan_5b_wrapper as ww
        from wan_5b.modules.t5 import umt5_xxl
        from wan_5b.modules.tokenizers import HuggingfaceTokenizer

        self.H, self.W, self.h, self.w = H, W, H // 16, W // 16
        if steps is None:
            steps = 4 if str(ckpt).upper() == "S4" else 2
        ck = os.path.join(MODELS, str(ckpt).upper(), "model_4o6.pt") if str(ckpt).upper() in ("S2", "S4") else ckpt
        cfg = OmegaConf.load("configs/nvfp4/inference_nvfp4.yaml")
        cfg.checkpoints.generator_ckpt = ck
        cfg.model_quant_use_transformer_engine = False
        cfg.model_quant_te_fallback_to_fouroversix = False
        cfg.inference.kv_quant = bool(kv_quant)
        cfg.model_kwargs.local_attn_size = int(local_attn)
        cfg.model_kwargs.num_frame_per_block = int(block)
        cfg.num_frame_per_block = int(block)
        cfg.inference.vae_device = "cuda:0"
        cfg.inference.streaming_vae = False
        cfg.inference.sampling_steps = int(steps)
        cfg.torch_compile = bool(compile)
        cfg.num_output_frames = 64
        cfg.data.image_or_video_shape = [1, 64, 48, self.h, self.w]
        self.config = normalize_config(cfg)
        self.dev = torch.device("cuda")
        torch.set_grad_enabled(False)
        t5dev = torch.device(t5_device)

        class T5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.text_encoder = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16,
                                             device=t5dev).eval().requires_grad_(False)
                self.text_encoder.load_state_dict(torch.load(
                    "wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth", map_location=t5dev,
                    weights_only=False))
                self.tokenizer = HuggingfaceTokenizer(name="wan_models/Wan2.2-TI2V-5B/google/umt5-xxl/", seq_len=512,
                                                      clean="whitespace")
            device = property(lambda self: t5dev)
            forward = ww.WanTextEncoder.forward

        t0 = time.time()
        self.pipe = CausalDiffusionInferencePipeline(self.config, device=self.dev, text_encoder=T5())
        enc = self.pipe.text_encoder
        dev = self.dev

        class CachedText(torch.nn.Module):
            """Hidden from .to() (keeps the T5 where it is); caches embeddings per prompt."""
            def __init__(self):
                super().__init__()
                self._e, self.cache = [enc], {}

            def to(self, *a, **k):
                return self

            def forward(self, text_prompts):
                for t in set(text_prompts):
                    if t not in self.cache:
                        self.cache[t] = self._e[0](text_prompts=[t])["prompt_embeds"].to(dev)
                return {"prompt_embeds": torch.cat([self.cache[t] for t in text_prompts], 0)}

        self.pipe.text_encoder = CachedText()
        setup_nvfp4_pipeline(self.pipe, self.config, dev)
        self.pipe.generator.model.eval().requires_grad_(False)
        if compile:
            ok = self.pipe.generator.configure_torch_compile(mode="default", dynamic=False, suppress_errors=True)
            _log(f"torch.compile: {ok}")
        from taehv import TAEHV, StreamingTAEHV
        self.tae = TAEHV(os.path.join(MODELS, "fastvae", "taew2_2.pth")).to(dev, torch.float16)
        self.dec = StreamingTAEHV(self.tae)
        self.enc = StreamingTAEHV(self.tae)
        self.steps, self.block, self.clean_every = steps, block, clean_every
        self.gen = torch.Generator(dev).manual_seed(seed)
        self.dtype = torch.bfloat16
        self.prompt, self.cut_pending, self.recache_pending = "", False, False
        self.hist = []  # (clean latents chunk, start frame) of the attention window, for KV-recache
        self.stats = {}
        _log(f"loaded {ck.split('/')[-2]} {W}x{H} steps {steps} block {block} kvq {kv_quant} window {local_attn} "
             f"sage {sage} compile {compile} in {time.time() - t0:.0f}s, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ------------------------------------------------------------------ stream control
    def set_prompt(self, prompt, cut=False, recache=True):
        """Next chunk uses `prompt`.
        cut=False: smooth change; with recache=True the KV cache of the recent window is rebuilt under the new prompt
        (LongLive's KV-recache: without it the new prompt barely takes hold, the old scene's KV dominates).
        cut=True: hard scene cut: the KV cache is cleared, a new shot starts (the video stream continues)."""
        if prompt != self.prompt or cut:
            self.cut_pending = self.cut_pending or bool(cut)
            self.recache_pending = self.recache_pending or (not cut and recache)
        self.prompt = prompt
        self.pipe.text_encoder([prompt])  # encode now (cached), not at the chunk

    def encode_image(self, rgb):
        """HxWx3 uint8 RGB -> 1 latent frame [1,1,48,h,w] (TAE encoder; resized to the stream size)."""
        import cv2
        im = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(im).to(self.dev).permute(2, 0, 1)[None, None].half() / 255
        return self.tae.encode_video(x, parallel=True, show_progress_bar=False).to(self.dtype)[:, :1]

    def reset(self, prompt=None, image=None):
        """Start a new stream (clears KV, decoder state). image: HxWx3 uint8 RGB reference (I2V) or None."""
        p = self.pipe
        if prompt is not None:
            self.set_prompt(prompt)
        self.cut_pending = self.recache_pending = False
        self.hist = []
        dit = p._dit_model
        if p.kv_cache_pos is None:
            p._initialize_kv_cache(batch_size=1, dtype=self.dtype, device=self.dev)
            p._initialize_crossattn_cache(batch_size=1, dtype=self.dtype, device=self.dev)
        else:
            for c in p.kv_cache_pos:
                c["global_end_index"] = torch.tensor([0], dtype=torch.long, device=self.dev)
                c["local_end_index"] = torch.tensor([0], dtype=torch.long, device=self.dev)
                c["pinned_start"].fill_(-1)
                c["pinned_len"].zero_()
        dit.local_attn_size = p.local_attn_size
        p._set_all_modules_max_attention_size(p.local_attn_size)
        if p.sink_size is not None:
            p._set_all_modules_sink_size(p.sink_size)
        p._set_all_modules_global_sink_size(p.global_sink_size)
        if p.inference_t_scale is not None:
            dit.t_scale = p.inference_t_scale
        if p._rope_method_override is not None:
            dit.rope_method = p._rope_method_override
        if p._original_seq_len_override is not None:
            dit.original_seq_len = p._original_seq_len_override
        dit.use_relative_rope = p.use_relative_rope
        dit.rope_temporal_offset = 0.0
        self.shot = 0
        self.start = 0          # current_start_frame (latent frames, RoPE position)
        self.chunk = 0
        self.image_latent = self.encode_image(image) if image is not None else None
        self.dec.reset()
        self.enc.reset()

    def _noise(self, n):
        return torch.randn([1, n, 48, self.h, self.w], device=self.dev, dtype=self.dtype, generator=self.gen)

    @torch.no_grad()
    def step(self, input_frames=None, strength=1.0, strength_map=None, decode=True, abort=None):
        """Generate the next chunk. input_frames: list/array of 4*block RGB uint8 frames (V2V) or None.
        strength: V2V noise level (0..1). strength_map: optional [h, w] tensor/array (latent grid) of per-pixel
        strengths (overrides strength). abort: optional callable checked between denoising passes; if it returns
        True the chunk is dropped (returns None; the next step regenerates the same position). Returns a list of RGB
        uint8 frames (or latents if decode=False)."""
        from pipeline.causal_diffusion_inference import _overwrite_i2v_context, _zero_i2v_context_timestep
        p, n, fsl = self.pipe, self.block, self.pipe.frame_seq_length
        t_all = time.perf_counter()
        cond = p.text_encoder([self.prompt])
        for b in range(p.num_transformer_blocks):
            p.crossattn_cache_pos[b]["is_init"] = False
        cut = self.cut_pending and self.chunk > 0
        recache = self.recache_pending and not cut and self.hist
        self.cut_pending = self.recache_pending = False
        if cut:  # new shot: forget the scene incl. the attention sink (KV + history); positions keep counting
            for c in p.kv_cache_pos:
                c["local_end_index"].fill_(0)
                c["global_end_index"].fill_(self.start * fsl)
                c["pinned_start"].fill_(-1)
                c["pinned_len"].zero_()
            self.hist = []
        if recache:  # rebuild the window's KV under the new prompt from the kept clean latents
            t_rc = time.perf_counter()
            # rebuild only the most recent `recache_frames` latent frames (the sink stays): halves the one-off cost
            k = max(1, -(-int(self.recache_frames) // n))
            hist = self.hist[-k:]
            p._zero_kv_data(p.kv_cache_pos, hist[0][1] * fsl)
            for lat, st in hist:
                for b in range(p.num_transformer_blocks):
                    p.crossattn_cache_pos[b]["is_init"] = False
                p.generator(noisy_image_or_video=lat, conditional_dict=cond,
                            timestep=torch.zeros([1, lat.shape[1]], device=self.dev, dtype=torch.float32),
                            kv_cache=p.kv_cache_pos, crossattn_cache=p.crossattn_cache_pos,
                            current_start=st * fsl, cache_start=st * fsl)
            for b in range(p.num_transformer_blocks):
                p.crossattn_cache_pos[b]["is_init"] = False
            self.stats["recache_ms"] = 1000 * (time.perf_counter() - t_rc)
        i2v = self.chunk == 0 and self.image_latent is not None
        noise = self._noise(n)
        sched = p._initialize_sample_scheduler(noise)
        timesteps = list(sched.timesteps)
        x_in = self._encode_frames(input_frames) if input_frames is not None else None
        if x_in is None:
            # text (and I2V first chunk): upstream scheduler (UniPC), exactly as NVlabs' pipeline
            latents = noise
            for t in timesteps:
                if abort is not None and abort():
                    return None
                ts = t * torch.ones([1, n], device=self.dev, dtype=torch.float32)
                if i2v:
                    latents = _overwrite_i2v_context(latents, self.image_latent, 1)
                    ts = _zero_i2v_context_timestep(ts, 1)
                flow, _ = p.generator(noisy_image_or_video=latents, conditional_dict=cond, timestep=ts,
                                      kv_cache=p.kv_cache_pos, crossattn_cache=p.crossattn_cache_pos,
                                      current_start=self.start * fsl, cache_start=self.start * fsl)
                latents = sched.step(flow, t, latents, return_dict=False)[0]
                if i2v:
                    latents = _overwrite_i2v_context(latents, self.image_latent, 1)
        else:
            # V2V: few-step distilled sampling from the input's noise level: x0 = x - sigma * v, then re-noise to the
            # next level. strength (or a per-pixel strength_map) = the noise level the input starts from.
            m = None if strength_map is None else \
                torch.as_tensor(strength_map, device=self.dev, dtype=torch.float32).view(1, 1, 1, self.h, self.w)
            s_top = float(strength) if m is None else float(m.max())
            s_top = min(max(s_top, 0.02), 1.0)
            sig = [s_top] + [float(t) / 1000.0 for t in timesteps if float(t) / 1000.0 < s_top - 0.02]
            x = (1 - sig[0]) * x_in + sig[0] * noise
            for i, sg in enumerate(sig):
                if abort is not None and abort():
                    return None
                if m is not None:  # pixels whose start level is below sg haven't started: keep the noised input
                    keep = (m < sg).to(self.dtype)
                    x = keep * ((1 - sg) * x_in + sg * self._noise(n)) + (1 - keep) * x
                ts = torch.full([1, n], sg * 1000.0, device=self.dev, dtype=torch.float32)
                flow, _ = p.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts,
                                      kv_cache=p.kv_cache_pos, crossattn_cache=p.crossattn_cache_pos,
                                      current_start=self.start * fsl, cache_start=self.start * fsl)
                x0 = x - sg * flow.to(x.dtype)
                if i + 1 < len(sig):
                    x = (1 - sig[i + 1]) * x0 + sig[i + 1] * self._noise(n)
            latents = x0
        clean = cut or (self.clean_every == 1) or (self.clean_every > 1 and self.chunk % self.clean_every == 0)
        if clean:
            p.generator(noisy_image_or_video=latents, conditional_dict=cond, timestep=ts * 0,
                        kv_cache=p.kv_cache_pos, crossattn_cache=p.crossattn_cache_pos,
                        current_start=self.start * fsl, cache_start=self.start * fsl)
        self.hist.append((latents.detach().clone(), self.start))
        keep = max(1, p.local_attn_size // n) if p.local_attn_size and p.local_attn_size > 0 else 4
        self.hist = self.hist[-keep:]
        self.start += n
        self.chunk += 1
        torch.cuda.synchronize()
        t_dit = time.perf_counter() - t_all
        if not decode:
            return latents
        frames = self._decode(latents)
        rc = self.stats.get("recache_ms") if recache else None
        self.stats = {"dit_ms": 1000 * t_dit, "total_ms": 1000 * (time.perf_counter() - t_all), "frames": len(frames),
                      "recache_ms": rc, "cut": bool(cut), "v2v": x_in is not None}
        return frames

    def _encode_frames(self, frames):
        import cv2
        lat = []
        for f in frames:
            if f.shape[:2] != (self.H, self.W):
                f = cv2.resize(f, (self.W, self.H), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(np.ascontiguousarray(f)).to(self.dev).permute(2, 0, 1)[None, None].half() / 255
            z = self.enc.encode(x)
            if z is not None:
                lat.append(z)
        while len(lat) < self.block and (z := self.enc.encode()) is not None:
            lat.append(z)
        if not lat:
            return None
        x = torch.cat(lat, 1).to(self.dtype)
        if x.shape[1] < self.block:  # startup: pad by repeating the last latent
            x = torch.cat([x] + [x[:, -1:]] * (self.block - x.shape[1]), 1)
        return x[:, -self.block:]

    def _decode(self, latents):
        out = []
        f = self.dec.decode(latents.half())
        while f is not None:
            out.append(f)
            f = self.dec.decode()
        if not out:
            return []
        v = torch.cat(out, 1)[0].clamp(0, 1)  # T C H W
        return list((v.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy())
