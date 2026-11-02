# Cloud GPU research: live webcam to reference-character at 25–30 fps (2026-11-01)

Labels used below. **[pub]** means a published number from the source linked. **[est]** means my own estimate. **[unverified]** means a secondary or aggregator source.

## TL;DR

- **Face-only swap at 30 fps is cheap and runs locally.** Deep-Live-Cam/inswapper and FasterLivePortrait (TensorRT, 30+ fps on a 3090 [pub]) fit on the 5080 laptop. No cloud GPU is needed.
- **Full-body character swap from a reference image at about 24 fps on one rented RTX 5090 is now possible with open weights.** The model is **JoyAI-Video-Edit** (JD, Apache-2.0, released Aug 2026). It edits webcam video causally and takes a reference image. Published numbers: **840×480 @ 24 fps on RTX 5090 or RTX PRO 6000; 720p @ 16 fps on PRO 6000; 720×1280 @ 30 fps on one B200** [pub].
- **No open model I found does a sustained 30 fps at 480p or higher on a GPU cheaper than a B200.** The options are a 5090 at 24 fps with RIFE interpolation up to 30, or a B200 at about $6–8/h.
- The current Scope Wan-1.3B pipelines (LongLive, SDv2, Reward/Causal Forcing + VACE) would run faster on a 5090, but **identity stays weak**. Streaming VACE reference-to-video is "severely degraded" vs batch ([arXiv 2602.14381](https://arxiv.org/html/2602.14381)). Renting a GPU buys frame rate here, not identity.

## 1. Candidate models / pipelines

| Model | Ref image native? | Webcam/streaming? | Published speed | Notes |
|---|---|---|---|---|
| **JoyAI-Video-Edit** ([repo](https://github.com/jd-opensource/JoyAI-Video-Edit), [paper](https://arxiv.org/html/2608.03974), [HF](https://huggingface.co/jdopensource/JoyAI-Video-Edit)) | Yes. IV2V/RV2V: the ref image is VAE-encoded and its tokens stay globally visible. The 0811 DiT checkpoint has "significantly stronger" RV2V | Yes. Causal 8-frame chunks; webcam demo on [HF Space](https://huggingface.co/spaces/wxDai/joyai-video-edit) | 5090: 840×480@24. PRO 6000: 480p@24 / 720p@16. B200: 720×1280@30. B200 per-chunk request→response 226 ms [pub] | 16B MM-DiT, 2-step SA-DMD, FP8 + SageAttn. At 480p24: ~21.5 GiB resident / 28 GiB peak, so **won't fit 16 GB**. Single GPU only (no multi-GPU path). Paper is about instruction/ref editing; I found no text describing whole-person swaps, so test it |
| **EditaLive** ([repo](https://github.com/GVCLab/EditaLive), [paper](https://arxiv.org/html/2608.27123v1)) | Yes. Wan-Animate based: ref image + pose skeleton + face crop. Cross-character editing is a stated task | Yes. `edit_webcam.py` + browser UI; remote GPU via SSH tunnel | 1×H100, 384×672: **14.47 fps, 16.4 with Flash-VAED**; 0.83 s inter-chunk latency; `torch.compile` gives about 1.3× [pub] | 17B (Wan2.2-Animate-14B + LoRAs). FP8 needs 14–38 GiB depending on offload. Custom ref image in the WebUI is still TODO (the CLI has `--ref_image`). Hands are weak. Research-only use. Closest to "become this character", but **too slow for 25–30 fps** without a B200 [est ~25 fps on B200] |
| LiveAnimate ([arXiv 2608.11745](https://arxiv.org/abs/2608.11745)) | Yes (ref + pose stream) | Streaming | 14B: 12.4 fps on 1×H100, 19.6 on 2×H100 [pub] | Code not confirmed |
| Wan-Animate-2 / -Lite ([repo](https://github.com/Wan-Video/Wan-Animate-2)) | Yes (character replace/animate) | Lite is causal/streaming | Lite "24 fps @ 400×720" [unverified, news only] | **Lite weights not released** as of the repo check; Base/Distill are offline (8×A800 for 720p) |
| Scope: LongLive / SDv2 / RewardForcing / MemFlow + VACE (Wan2.1-1.3B) | VACE ref path is weak in streaming | Yes (current setup) | LongLive 20.7 fps H100, 24.8 with FP8 [pub]. Streaming VACE 17–22 fps consumer GPUs; Scope author ~20 fps @ 368×640 on a 5090 [pub/Reddit] | Identity is the bottleneck, not fps |
| StreamDiffusionV2 ([arXiv](https://arxiv.org/html/2511.07399v2)) | No (text-prompt V2V) | Yes | 1.3B 4×H100: 42 fps @480p, 61.6 @512². 14B 4×H100: 39 @480p, 58 @512². **4×RTX 4090 (PCIe): ~16 @480p, ~24 @512².** 1×H100 512² e2e latency 357 ms mean [pub] | Multi-GPU scaling needs NVLink; PCIe consumer cards scale poorly |
| Krea Realtime 14B | No | V2V in Scope? (T2V listed) | 11 fps on B200 [pub] | Too slow |
| Self-Forcing 1.3B | No | — | 17 fps H100, ~10 fps 4090 [pub] | Superseded |
| StreamDiffusion SD-Turbo/SDXL + IP-Adapter + ControlNet (Daydream) | IP-Adapter = style/loose identity | Yes | SDXL 15–25 fps (vendor) [pub] | Flickery, weak identity |
| Face path: Deep-Live-Cam / FaceFusion (inswapper_128) | Yes, 1 photo | Yes | "Real-time"; no hard benchmark found | Face only; inswapper is non-commercial |
| Face path: FasterLivePortrait | Yes (animates the ref portrait with your expressions) | Yes | 30+ fps on a 3090 with TensorRT [pub] | Head/portrait only; static body |
| Closed reference: Decart Lucy 2.5 | Yes (whole-person swap) | WebRTC API | 30 fps, ~300 ms + network; **$0.02/s ≈ $72/h** ([mimiccam](https://github.com/bigrayszn/mimiccam), [Decart pricing](https://docs.platform.decart.ai/getting-started/pricing)) | Closed. Quality benchmark for comparison |

Also seen but not evaluated: SANA-Streaming (704×1280 @ 14.5 fps per the JoyAI paper), LiveEdit (arXiv 2606.26740), InteractiveAvatar (avatar generation, not V2V).

## 2. Expected performance by GPU

| GPU | JoyAI-Video-Edit | EditaLive | Scope Wan-1.3B (your pipeline) |
|---|---|---|---|
| RTX 5080 laptop 16 GB | Does not fit (28 GiB peak) | Only with heavy offload, <10 fps [est] | 17–23 fps @ 640×352 (measured) |
| RTX 5090 32 GB | **840×480 @ 24** [pub] | ~10–14 fps @ 384×672 [est] | ~28–38 @ 640×352, ~18–25 @ 832×480 [est] |
| RTX PRO 6000 96 GB | 480p @ 24, 720p @ 16 [pub] | ~12–15 [est] | Similar to 5090 [est] |
| L40S 48 GB | ~12–16 @ 480p [est] (slower than 5090) | <12 [est] | below 5090 [est] |
| H100 80 GB | Not published; ~20–28 @ 480p [est] | **14.5–16.4 @ 384×672**, ~20 with compile [pub/est] | LongLive 20.7 @ 832×480 (24.8 FP8) [pub] |
| H200 | ≈ H100, slightly faster [est] | ≈ H100 [est] | ≈ H100 [est] |
| B200 180 GB | **720×1280 @ 30** [pub] | ~25+ [est] | 40+ [est] |
| 4×H100 NVLink | n/a (single-GPU server) | n/a | SDv2 1.3B 42 fps @480p [pub] |
| 2–4× 5090/4090 PCIe | n/a | n/a | SDv2 4×4090 only ~16 fps @480p [pub]; poor scaling |

## 3. Prices (on-demand, per GPU-hour, checked 2026-11-01)

| GPU | RunPod community / secure ([pricing](https://www.runpod.io/pricing)) | Vast.ai [unverified, marketplace] | Others |
|---|---|---|---|
| RTX 5090 | $0.69 / $1.19 | ~$0.40–0.65 (median ~$0.53) | TensorDock ~$0.56–0.65 [unverified]; Salad $0.25–0.29 (batch priority, no inbound UDP; unsuitable) |
| RTX PRO 6000 | $1.69 / $2.49 | ~$1.15–1.55 | Modal $3.03 ([pricing](https://modal.com/pricing)) |
| L40S | $0.79 / $1.09 | — | Modal $1.95 |
| H100 SXM | $2.69 / $3.99 (PCIe $1.99 / $2.89) | ~$1.0–1.9 | Lambda $3.29 PCIe, $3.99–4.29 SXM ([pricing](https://lambda.ai/pricing)); Modal $3.95 |
| H200 | $3.59 / $5.29 | — | Modal $4.54 |
| B200 | $5.98 / $7.99 | — | Modal $6.25; Lambda $6.69–6.99 |

RunPod network volume costs $0.07/GB/mo. 100 GB of weights is about $7/mo.

## 4. Provider fit for a live WebRTC session

| Provider | UDP/WebRTC | Persistence | Billing / spin-up | Verdict |
|---|---|---|---|---|
| **RunPod** | **No UDP** ([docs](https://docs.runpod.io/pods/configuration/expose-ports)), so you need TURN over TCP/TLS. Scope already handles this: an HF token gives Cloudflare TURN (Scope README) | `/workspace` volume. Network volumes are tied to one datacenter (and I believe Secure Cloud only; verify) | Pods start in ~1–3 min with weights already on the volume; per-second/minute billing | **Best fit for Scope**. There is an official Scope RunPod template (image `daydreamlive/scope`, models in `/workspace/models`) |
| **Vast.ai** | **UDP ports supported** (`-p 8082:8082/udp`, randomized external port via `VAST_UDP_PORT_X`, max 64 ports) ([docs](https://docs.vast.ai/documentation/instances/connect/networking)) | Per-instance disk; no shared volumes across hosts | Cheapest; per-second; host quality varies | Good for cheap 5090 tests; TURN is still simplest |
| TensorDock | Full VM, open ports | VM disk | Per-second | Viable; check 5090 stock |
| Modal | Serverless; WebRTC example exists (FastRTC/`webrtc_yolo`) | Volumes | Per-second; cold starts; no 5090 | Pricier; fine for PRO 6000/B200 bursts |
| Lambda | Full VM | Filesystems | No 5090; H100/B200 only | Use only if you go B200 |
| Salad | No inbound UDP; interruptible | — | — | Not suitable |

**Scope cloud modes (verified in local repo `scope/` @ 72073126):**
1. The RunPod template is self-hosted. You run the Scope UI in a browser pointed at the pod, with TURN.
2. **"Remote Inference"** keeps the local Scope client and runs inference on a Livepeer-network runner hosted on fal.ai (`src/scope/cloud/livepeer_fal_app.py`, `docs/livepeer.md`). You can also deploy your own runner with `fal deploy` and set `SCOPE_CLOUD_APP_ID`/`LIVEPEER_WS_URL`. The older cloud-relay mode is being deprecated.

JoyAI and EditaLive are **not** Scope pipelines. You would use their own web demos (SSH tunnel / Gradio) or wrap them as a Scope plugin.

**Latency budget [est]:** RTT to a same-country datacenter is about 20–60 ms, plus ~30–60 ms each way for WebRTC encode/decode/jitter buffer, plus TURN relay hop ~10–40 ms, plus model buffering. JoyAI's 8-frame chunk at 24 fps is ~330 ms of capture, and compute is ~230–300 ms. **Expected glass-to-glass is ~0.7–1.0 s for JoyAI on a 5090 and ~1.0–1.3 s for EditaLive.** The <1 s target is borderline but plausible. Pick the RunPod/Vast region nearest you (US-central if you're in CST).

## 5. Recommendation

| Goal | Combo | Expected | Cost/h |
|---|---|---|---|
| **Primary test** | **JoyAI-Video-Edit (0811 DiT, RV2V with ref image) on 1× RTX 5090**, Vast.ai or RunPod community (or secure for a network volume) | 840×480 @ 24 fps [pub], ~0.8–1 s G2G [est]. Add RIFE ×1.25 to output 30 fps | $0.45–0.70 (secure $1.19) |
| 30 fps native 720p | JoyAI on 1× B200 (RunPod/Modal/Lambda) | 720×1280 @ 30 [pub] | $6–8 |
| Strongest "become this character" (pose-driven) | EditaLive on H100/B200 | 15–20 fps on H100 [pub/est]; B200 ~25 [est] | $2–8 |
| Keep current Scope stack | Scope RunPod template on a 5090 | ~25–35 fps @ 640×352 [est], same weak identity | $0.69–1.19 |
| Face only | Deep-Live-Cam or FasterLivePortrait **locally on the laptop** | ~30 fps | $0 |

**Main risks**
- JoyAI's whole-person reference swap quality is untested. The paper focuses on edits and ref-guided garment/object swaps. Run the HF Space with your webcam and character image first, for free, before renting.
- 24 fps is the published 5090 ceiling. A 30 fps sustained target needs interpolation or a B200.
- RunPod has no UDP, so TURN adds a hop and possibly some jitter. 5090 stock in the datacenter that holds your network volume can run out.
- These are new research repos (custom CUDA ops; sm_120 quirks such as no FA4). Setup time is likely a few hours.
- Licenses: JoyAI is Apache-2.0. EditaLive is for research only. inswapper is non-commercial. That matters for a monetized TikTok stream.
