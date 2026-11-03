# Research: a one-on-one "FaceTime call" with a character, powered by JoyAI-VL-Interaction

Date: 2026-11-03. Not built or tested yet; everything here comes from reading the code of
`github.com/jd-opensource/JoyAI-VL-Interaction` (`services/webinfer/live_adapter.py`, `memory_summarizer.py`,
`scripts/start_model.sh`, `doc/architecture.md`, `doc/getting_started.md`). See also
`research-jdopensource-models.md`.

## 1. The idea
A real person talks into their camera. Their frames and speech go to VL-Interaction, which plays the character on
the other end of the "call" (e.g. G-Star). Its replies become the character's voice and face.
```
caller camera ──frames ~1 fps──────┐
caller mic ──Qwen3-ASR──> text ────┼─> webinfer adapter ─> VL-Interaction (character system prompt)
(optional) memory notes ─> text ───┘                              │ </response> text | </silence>
                                                   TTS (character voice) ─> audio ─> caller's speaker
                                                              │
                                   audio-driven character video (lip-sync) ─> caller's screen
```

## 2. Can it hold a conversation? Horizon and memory
It's built for long streams, with a 3-level memory managed by the adapter:

| Level | What | Size / default |
|---|---|---|
| Short-term | Raw frames of the current chunk, plus this chunk's turns | `CHUNK` = 100–200 frames at ~1 fps ≈ **1.7–3.3 min** of video in context |
| Mid-term | Text summary of each finished chunk (Qwen3-VL-4B summarizer) | ≤ 4000 tokens (target 3000) |
| Long-term | Every `COMPRESS_EVERY_N_CHUNKS`=5 chunks, the summaries are compressed into long-term memory | ≤ 2000 tokens (target 1000), window 40 |
| Q&A history | Earlier user queries and the model's answers (`keep_qa_history=True`), in the system context | grows with the session |

- The main vLLM model runs with `--max-model-len 131072` (128k tokens), and the session timeout is
  `session_timeout_seconds` = 3600 (1 h).
- **Conclusion:** multi-minute to ~1 h conversations are supported out of the box. The last ~2–3 min are seen in
  full; older parts survive as summaries plus the Q&A log. The weak spot is exact recall of details from long ago,
  since summaries are lossy.

## 3. Extra memory via an LLM-managed text store: possible
- **VL-Interaction accepts text input.** The adapter takes the latest user text as the query
  (`use_prompt_as_query`). ASR is just one source of that text. A `system` message in the request replaces the
  system prompt (`_extract_system_prompt_override`).
- **Two injection points:**
  1. **System prompt** (persona + a "facts about the caller" block), rewritten whenever the store changes.
  2. **User text** (e.g. "[memory] Their dog is called Max" before the caller's words).
- **A small side LLM keeps the store.** For example JoyAI-LLM-Flash, or the bundled Qwen3-VL-4B summarizer, which is
  already running. After each exchange it extracts durable facts (names, preferences, promises, running jokes) into
  JSON/markdown, and the store is reloaded at the start of the next call. That gives memory across calls, which the
  built-in memory doesn't (it's per session).
- **Watch the context size.** Keep the injected block short (≤ 1–2k tokens) so the 1-fps frame stream keeps room.

## 4. Settings to change for a call (vs their defaults)
- **System prompt = character persona + their action format.** It must still emit `</response>` / `</silence>`.
  Drop delegation (`DEFAULT_SYSTEM_PROMPT_NO_DELEGATION`), or keep it for "let me look that up" moments.
- **Proactivity:**
  - **Replies:** `FORCE_SILENCE_BEFORE_QUERY=true` means it waits until the caller speaks. Every ASR utterance is a
    new query, so replies follow speech naturally.
  - **Proactive lines:** set it false, so it comments on what it sees (smiles, a dog walking in).
  - **Too chatty:** tune with the prompt and `MAIN_TEMPERATURE` (default 0.8).
- **Turn-taking:** the model decides per frame (~1 s). True barge-in (caller interrupts the character) needs the
  client to stop TTS playback when ASR detects speech; the adapter doesn't do that.
- **Latency per exchange:** about 0.5–1 s ASR endpointing + up to ~1 s decision + TTS first audio 0.3–0.5 s +
  lip-sync, so roughly **1.5–2.5 s**. OK for a casual call, slower than a human.

## 5. The character's face on the call
- The same gap as the streamer idea: our JoyAI pipeline is video-driven and has no audio input.
- **Option 1 (simplest):** an audio-driven real-time lip-sync model on a looping clip of the character. Check it
  on G-Star's cartoon mouth.
- **Option 2:** audio → talking human head → JoyAI-Video-Edit → character. Higher quality and identity, but two
  models; it relies on the lip-sync LoRA (`cloud/pipelines/joyai/training/`).
- **Idle behaviour:** while silent, the character should still blink and nod. Option 1 loops an idle clip; Option 2
  needs an idle driver.

## 6. Hardware
Their default layout uses 3 GPUs: 8B main model, 4B summarizer, ASR+TTS. Cheaper setups to try:
- **One 80 GB GPU:** INT8/FP8 main + summarizer + ASR + TTS, with `MAX_MODEL_LEN` lowered (32–64k is plenty at 1 fps
  with summaries).
- **One RTX 5090:** NVFP4/INT4 main + 4B summarizer at low memory utilization, ASR/TTS on a second small GPU, or the
  laptop for ASR. Unverified.
- **JD Cloud API** (JoyBuilder MaaS, since 2026-08-11): no GPU needed for a first personality test.

## 7. Plan for a first prototype (~1 day, < $10)
1. Rent an H100 80 GB. Run `install.sh --with-all`, `download-models.sh --all`, `run.sh minimal`. Start with their
   own WebUI (WebRTC, port 8099) and the default prompt.
2. Swap in a G-Star persona system prompt (no delegation). Talk to it for 10 min and check: does it stay in
   character, do replies come at the right times, does it recall what was said 5 min ago (mid-term memory)?
3. Add the memory store: a side LLM writes facts after the call, injected into the system prompt next call. Check
   that it remembers the caller across calls.
4. Then the face: audio-driven lip-sync on a G-Star loop (option 1), measuring latency glass-to-glass.

## Risks
- **Persona vs action format:** it was RL-trained on the default prompt; a strong persona could break the
  `</silence>` / `</response>` discipline. Fall back to: persona text inside the user query instead.
- **1 fps vision:** it misses fast gestures and fine expressions, which is fine for conversation.
- **Lossy summaries:** forgotten details after ~3 min. The text store mitigates this.
- **No native barge-in:** the client must handle interruptions.
