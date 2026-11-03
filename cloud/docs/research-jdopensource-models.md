# Research: jdopensource models for real-time streaming + an "AI streamer" persona pipeline

Date: 2026-11-03. Sources: HF `jdopensource/*` model cards, GitHub `jd-opensource/JoyAI-VL-Interaction` (README,
`doc/architecture.md`, `doc/getting_started.md`, `services/webinfer/live_adapter.py`, `scripts/start_model.sh`).
Nothing below has been run by us yet, except JoyAI-Video-Edit.

## 1. The 23 jdopensource HF models, ranked for live streaming
7 families (the rest are quantized / format variants).

| Rank | Family (repos) | Real-time? | What it is | Use for us |
|---|---|---|---|---|
| 1 | **JoyAI-Video-Edit** (+ `-Diffusers`) | **yes** (in use) | 16B 2-step causal streaming V2V editor; our G-Star swap, ~25 fps on an RTX 5090 | The live video model. `-Diffusers` = 0811 for offline Diffusers use |
| 2 | **JoyAI-VL-Interaction** (+ `-Preview`, `-INT4`, `-INT8`, `-FP8`, `-NVFP4`) | **yes** | 8B vision-language model that watches a live stream (~1 fps) and decides every turn to speak / stay silent / delegate. Apache-2.0 | The "brain" of an AI streamer or AI call partner (§2, and `research-vl-interaction-call.md`) |
| 3 | **JoyAI-LLM-Flash** (+ `-Base`, `-FP8`, `-INT4`, `-INT8`, `-GGUF`) | yes (text) | Fast text LLM | Chat replies, persona scripting, the background "delegate" solver |
| 4 | **JoySafety**, **JSL-joysafety-v1/v2** | yes (classifier) | Safety / moderation models | Filter viewer chat before it reaches the persona or the live prompt |
| 5 | **JoyAI-Echo 1.5** | no (8-step DMD) | Joint audio+video generation on LTX-2.3 with a cross-shot memory bank for character look and voice. 23–46 GB + Gemma-3-12B. LTX-2 community license | Offline: pre-rendered G-Star segments with a consistent voice |
| 6 | **JoyAI-Image-Edit** (+ `-Diffusers`, `-ComfyUI`, `-Plus-Diffusers`, `-Plus-ComfyUI`) | no (~30 steps) | 16B instruction image editing; Plus takes several reference images | Make reference images (G-Star outfits, poses, a cleaner `char_front.png`) |

Bottom line: only Video-Edit (in use) and VL-Interaction are real-time streaming models.

## 2. JoyAI-VL-Interaction: how it actually works (from the code)
- **Stack**: 5 services:
  - main VLM on vLLM (port 7060, `--max-model-len 131072`, gpu util 0.9);
  - `webinfer` adapter, an OpenAI-compatible chat-completions API (port 8070) that keeps session state;
  - summary model `Qwen/Qwen3-VL-4B-Instruct` (port 8065);
  - optional ASR `Qwen3-ASR-1.7B` (8994) and TTS `Qwen3-TTS-12Hz-1.7B-CustomVoice` (8992, default voice `vivian`);
  - optional background agent wrapping the `codex` CLI (8079);
  - WebUI over WebRTC (8099).

  The default layout uses 3 GPUs (main, summary, ASR+TTS). "Tested on NVIDIA Hopper". Tuned configs for RTX 3090/5090
  are on their TODO list.
- **Input**: JPEG frames at **~1 fps** + text. The adapter treats the latest user text in the request as the
  "user query" (`use_prompt_as_query=True`). **ASR output is just injected as that text**, so **typed text works
  through the same path**: anything we put in the user message (viewer chat, a memory note) reaches the model.
- **Output**: exactly one action per turn (max 128–256 tokens):
  - `</silence>`;
  - `</response> text`;
  - `</response> note </delegation> question`: the question goes to the background agent, and the result is fed
    back later.
- **Persona**: the system prompt is replaceable. A `system` message in the request overrides it
  (`_extract_system_prompt_override`), or the `x-system-prompt-key` header picks a preset. The default prompt is a
  neutral "real-time video streaming assistant". A G-Star persona = our own system prompt that keeps their action
  format.
- **Proactivity switch**: `FORCE_SILENCE_BEFORE_QUERY=true` (default) keeps it silent until a user query exists.
  Set it false for a persona that talks on its own.
- **Memory**: see `research-vl-interaction-call.md` §2 (short answer: built in, about 1 h sessions).

## 3. Idea: VL-Interaction as the streamer's personality, with live lip-sync
```
viewer chat (TikTok Live API/scraper) ─┐  text
camera / scene frames (~1 fps) ────────┼─> VL-Interaction (G-Star system prompt) ─> </response> text
(optional: crop of the chat overlay) ──┘                                               │
                                                                       G-Star TTS (cloned voice) ─> audio
                                                                                         │
                                            audio-driven lip-sync ──> G-Star video ──> OBS
```
- **Personality: feasible.** Persona = our system prompt. Its trained "speak only when worth it" timing suits a
  streamer: it doesn't answer every message.
- **Chat as input: send text, don't OCR.** Pull chat from TikTok Live and put it in the user message. That's
  exact, and the frames stay free for the scene. Reading the overlay from frames would also work (it's a VLM), but
  the frames are only ~1 fps and small scrolling text is a weak spot. Batch chat lines per turn, e.g. "Chat (last
  5 s): @a: …, @b: …".
- **Voice**: Qwen3-TTS CustomVoice (bundled), or any low-latency TTS with a designed G-Star voice.
- **Lip-sync: the gap.** JoyAI-Video-Edit is video-driven and has no audio input, so TTS audio can't drive it
  directly. Options:
  - **A. audio → human talking-head → JoyAI**: a real-time audio-to-face model (MuseTalk-style lip inpainting, or
    audio-driven LivePortrait) renders a talking face, and JoyAI turns it into G-Star. Two models (+0.3–0.5 s). It
    also needs JoyAI to follow mouths, which is the lip-sync LoRA in `cloud/pipelines/joyai/training/`.
  - **B. audio → G-Star directly**: a real-time audio-driven lip-sync model on a looping G-Star clip, skipping JoyAI.
    Simplest and lowest latency. The risk is that these models are trained on human faces, so a cartoon mouth may
    fail.
  - **C. JoyAI-Echo**: synced G-Star audio+video, but offline only (pre-made segments).
- **Latency estimate**: ~1 s decision cadence + TTS (0.3–1 s) + lip-sync, so ~2–3 s from a chat message to G-Star
  speaking. Fine for chat.
- **First cheap test**: VL-Interaction INT4/NVFP4 + summary model on one rented 5090/H100 (or the JD Cloud API), G-Star
  system prompt, chat as text, TTS out, no video. If the personality is fun, add option B and check the cartoon mouth.

## Open questions
- Does INT4/NVFP4 plus the 4B summarizer fit on one 32 GB 5090 (vLLM `max-model-len` 131072 needs KV-cache room;
  lower it)?
- How well does it hold a strong persona while keeping the action format (it was RL-trained on the default prompt)?
- Chat language and slang (TikTok chat is mostly Spanish for our drivers); the TTS voice must match.
