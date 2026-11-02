# AGENTS.md — start here (for Claude agents and humans picking up this repo)

**Scope:** realtime AI video pipelines for live streaming (webcam in → AI-transformed video out → OBS virtual camera
→ TikTok etc.). Several pipelines, several models; shared bar: **25–30 fps**, low latency (**< 2 s** input → output
on the GPU machine), decent quality and consistency.
**First pipeline (working):** character swap: the webcam person becomes a character from one reference image (at
least the upper body with arms). Others so far: prompt-driven "dream"/world generation on the laptop (LongLive family). Owner: jordancoult. Characters: "G-Star"
(`assets/char_front.png`, a pink cone-headed starfish man with a G-Star cap and gold chain) and human references.

## Two worlds in one repo
| | Where | Status |
|---|---|---|
| **Local laptop** (RTX 5080 Laptop 16 GB) | repo root: `start_scope.sh`, `tools/`, `scope/` (fork `jordancoult/scope`, branch `node-graph`), `saved_settings/` | Works, 17–26 fps at 640×352; weak reference identity. **Don't break it**: the user's presets and saves live here |
| **Cloud pipelines** (rented Blackwell GPU) | `cloud/` | **JoyAI-Video-Edit meets the goal on one RTX 5090** (25–27 fps, 0.46 s on the GPU machine, ~1 s glass-to-glass from the laptop to EU) |

Read next: `cloud/README.md` (cloud overview + pipeline index) · `README_QUICKSTART.md` (local setup, benchmarks,
patches) · `cloud/docs/lessons.md` (hard-won rules).

## Cloud design rules
- **Provider-independent.** A pipeline needs only a Linux machine with a supported GPU, SSH and one TCP port.
  Public weights come from Hugging Face; our build artifacts (kernel wheels, compile caches) come from the private
  HF repo `JordanCo/ai-livestream-artifacts`. Provider notes (how to rent) live in `cloud/infra/providers.md`.
- **GPU requirement (current pipelines): consumer/workstation Blackwell, sm_120, ≥ 32 GB**, i.e. RTX 5090 (target,
  best $/fps) or RTX PRO 6000. `setup.sh` checks this and refuses anything else.
- **One folder per pipeline** under `cloud/pipelines/<name>/`: `setup.sh`, `run_server.sh`, `README.md` (recipe,
  numbers), `experiments/<date>-<topic>/FINDINGS.md` (+ review videos, which are gitignored).
- Shared client code (live bridge to OBS, benchmark client, review tools) lives in `cloud/client/`.

## Working rules (from the user)
- Commit and push tested changes right away; noreply email `33331023+jordancoult@users.noreply.github.com`,
  user.name `jordancoult`; no Claude trailers. Never push to `origin` of `scope/` (that is upstream Daydream).
- New features default to a complete bypass (old behaviour unchanged unless switched on).
- Never turn on the webcam while the user is away; use the clips in `assets/` as a simulated camera.
- Sustained tests 1–2 min max. Judge quality by looking at outputs (contact sheets / videos), and send videos.
- Every wait on a background job needs a hard deadline and must end when the job ends (any exit code).
- Never use `pkill -f` / `pgrep -f` (kills your own shell); stop processes by PID. Port 8000 busy = the user's live
  local session; don't kill it.
- Secrets live in `.secrets/` (gitignored): `runpod_ed25519` (SSH key for GPU machines), `hf_token`. Never print them.
- Cloud spend: say the hourly price before creating anything; terminate machines when done; only touch resources you
  created.
