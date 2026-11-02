# facefix: lip sync (and optional face restoration) as a post-process on the laptop GPU

**Round 2 summary (current recommendation):** human refs → `--facefix swap` (HyperSwap face swap of the real face,
lower face pasted: real teeth/tongue/jaw, no warping, ~24 ms/frame); cartoon refs (G-Star) → `--facefix rel`
(LivePortrait; still the only option that keeps a cartoon face). Details in "Round 2" below.

JoyAI-Video-Edit barely moves the generated mouth with the speaker's (lip-gap correlation 0.25 for a human ref,
0.03 for G-Star, see `../pipelines/joyai/experiments/2026-11-02-mouth/`). facefix fixes that on the laptop's idle
RTX 5080 Laptop, between "frame received" and "frame played": **LivePortrait** re-renders the lower half of the
*generated* face with the lip keypoints of the *real* face, then pastes it back with a feathered mask. Identity,
eyes, hair and background stay JoyAI's.

Status: measured on recorded clips; wired into `live_bridge.py` behind `--facefix` (off by default, see "Live").

## Results (RTX 5080 Laptop 16 GB, torch 2.9.1+cu128)
Lip sync, scored with `cloud/client/mouth_sync.py` (exact `source_seq` pairing, MediaPipe lip gap):

| Clip | JoyAI only | facefix | Settings |
|---|---|---|---|
| `C_mouthprompt` (human ref, talking close-up) | 0.25 | **0.96** (0.975 at `--lp-size 256`) | `--lip-mode abs` |
| `G_gstar` (G-Star, same clip) | 0.03 | **0.70** (0.81 abs, but see below) | `--lip-mode rel` |
| `human` (webcam_7643311, DTW-aligned, MediaPipe jawOpen) | 0.34 | 0.67 | abs, 256 (first run) |

Speed (per output frame, steady state):

| | ms |
|---|---|
| LivePortrait at `--lp-size 128` (motion extractor ×2, appearance, warp, decoder, paste) | **25** |
| same at `--lp-size 256` | 55–60 |
| MediaPipe face on the output frame (CPU) | 9–12 |
| MediaPipe face + hands on the input frame (CPU; live: at capture time) | ~23 |
| **3-stage pipeline wall clock, lp 128** (input analysis / output analysis / GPU, threads) | **37.6 fps** (30 fps while also writing 3 review videos) |
| same, lp 256 | 16.5 fps: too slow without TensorRT |
| added latency | ~1 frame (render 25 ms; analysis overlaps the previous frame) |

So **lp 128 meets the 25 fps target with margin; lp 256 doesn't (yet)**. GPU memory: ~1 GB (nvidia-smi peak 991 MiB during a run).

### Quality verdicts (look at the sheets in `experiments/2026-11-02/`)
- **Human ref (`sheet_final_C.jpg`, `sheet_scen_human.jpg`)**: works. The mouth opens, shows teeth and closes with
  the speaker; identity kept (only the lower face is re-rendered, colour-matched). lp 128 renders the mouth at
  256 px, slightly softer than JoyAI's own texture on a big close-up (teeth a bit blobby); at normal webcam framing
  (face ~100–150 px) it isn't visible. lp 256 is sharper but too slow for now.
- **G-Star (`sheet_final_G.jpg`, `strip_consecutive_final_G.jpg`)**: works with `--lip-mode rel`: his wide grin is
  kept and opens to a big toothy mouth with the speech, stable across consecutive frames. `abs` mode (the human
  lip shape) scores higher (0.81) but squeezes a small human mouth into the grin, smeared, so not used. The
  LivePortrait **animal** model was worse (0.59, pink blobs). MediaPipe finds G-Star's face in 93–99 % of frames;
  the rest fall back to the input face box mapped onto the output (poses align), at 0.7 weight.
- No pops: all transitions are EMA fades (max weight change 0.3 per frame); see the "fastest fade" rows.

### Robustness per scenario (the gate overlay shows weight `w` and the gates that are < 1)
| Scenario | Clip | Result |
|---|---|---|
| Talking, small head motion | test_face_landscape (C, G) | **works**: weight ~1 all the time |
| Glasses | webcam_7643311, webcam_5941016 | **works** (landmarks fine through glasses) |
| Hand to glasses / face | webcam_7643311 | **degrades gracefully**: hand-near-mouth gate fades the effect out over ~4 frames, JoyAI frame shown, fades back in (conservative: also fires for hands at the eyes/cheek) |
| Gestures in front of the chest | webcam_5941016 (G-Star landscape + portrait) | **works**, brief fades when fingers come within ~0.5 face widths of the mouth (`sheet_scen_gstar_desk.jpg`, `..._portrait.jpg`) |
| Head turns | human clip (up to 23° yaw) | **works** up to the max yaw in our clips (23°); pose gate fades from 25° to 0 at 40° (yaw) / 20→35° (pitch) |
| Profile / side-on, fast motion, face leaving frame | portrait_clip (boxer) | **degrades gracefully**: 96 % of frames gated off (no input face in 78 %, yaw 50–67° in the rest), JoyAI passes through untouched (`sheet_scen_boxer.jpg`). Ungated, the mapped box lands on G-Star's cap/cone because JoyAI draws no face there in profile (`sheet_nogate_yaw.jpg`), so the gate is required |
| Multiple faces | none in our clips | handled in code (main = face nearest the previous pick among those ≥ 50 % of the largest, else the largest; output face must lie within 0.5 face sizes of the input face) but **untested** |
| Lighting changes | none in our clips | colour match per frame inside the mask; untested |

Not measured: the yaw range 25–45° on a human ref (no clip has it). The 25/40° thresholds are conservative.

### Face restoration ("mushy faces"): tested, not recommended for now
Same pipeline with `--region none --restore X` (FFHQ 5-point alignment from MediaPipe, 512 px, feathered paste,
same weight gate), on the human (webcam_7643311) and G-Star clips (`sheet_restore_human.jpg`,
`sheet_restore_gstar.jpg`; columns: JoyAI | GFPGAN 1.4 | CodeFormer w=0.7 | RestoreFormer | unsharp mask):

| | ms/frame | Human | G-Star |
|---|---|---|---|
| GFPGAN 1.4 (fp16 autocast; plain `.half()` overflows to a flat brown face) | 30 | sharper eyes/lips, slightly "beautified"/plastic skin | smooths skin texture, keeps the cartoon; not humanized |
| CodeFormer w=0.7 | 54 | mild, natural sharpening | nearly unchanged (good) |
| RestoreFormer (v1; RestoreFormer++ ckpt isn't loadable by spandrel) | 68 | sharper, more texture | **humanizes**: wrinkles, human teeth, odd forehead |
| unsharp mask on the face ROI | 2.4 | small gain, slight halos | small gain |

The gain at stream scale is modest and none of the networks fit next to the lip pass in the 40 ms budget
(25 + 30 ms). Use `--restore sharpen` if you want something, GFPGAN only with a human ref once TensorRT frees GPU
time. CodeFormer is non-commercial (below).

## Round 2 (after the first live test: "unstable, warped, artifacts with the mouth open")
Methods are modules with the same interface (`analyze_input` / `analyze` / `render`, gating and fades shared):
`facefix.py` (common + LivePortrait `FacePass`, `--method lp`), `swap.py` (`SwapPass`, `--method swap`).
Evidence in `experiments/2026-11-02-round2/` (sheets + preview-safe `review_*.mp4`: input | JoyAI | refined face
close-ups). Open-mouth stress test = the frames with the widest-open input mouth (`review.py --by-jaw`).

**Diagnosis of the LP artifacts:** they come from the *generator*, not the paste: with the mouth open LivePortrait
has to invent the mouth interior, and at `--lp-size 128` (256 px render) teeth become a white smear
(`sheet_lp_C_openmouth.jpg`). 256 is better but 58 ms/frame; TensorRT (FasterLivePortrait) is not possible on this
laptop right now: no CUDA compiler for its grid-sample-3D plugin and no TensorRT libraries (+1.5–3 GB; disk is at
99 %). Stability fixes added for all methods: One-Euro filter on the landmarks (`--euro-cutoff 1.5 --euro-beta 0.3`),
faster fades for swap, LAB colour/lighting match, `--mask mouth` (ellipse around the mouth only) for LP.

| Candidate | Open mouth | Head turn / hands / glasses | G-Star | Human identity (ArcFace cos to ref, JoyAI → refined)* | Lip-gap corr (C clip) | ms/frame (GPU thread) |
|---|---|---|---|---|---|---|
| LP abs, 128 (round 1) | teeth smeared | ok, gated | n/a | — | 0.96 | 26 |
| LP rel, 128, One-Euro (G-Star) | big grin opens with teeth; `mouth` mask ≈ `lower` | ok, gated | **works** (0.69; mouth mask 0.63) | — | — | 25 |
| **Swap (HyperSwap 256), lower-face mask** | **real teeth/tongue, natural** (`sheet_swap_lower_C_openmouth.jpg`) | works; hand/turn gates fade it; glasses NOT carried over | **unusable** (humanizes, purple blotches: `sheet_swap_restore_gstar.jpg`) | 0.525 → 0.563 | **0.95** | **24** |
| Swap, inner-face mask | same mouth; also real eyes/brows | carries the user's **glasses** onto the character (`sheet_swap_scen_human.jpg`); double mouth during fades | unusable | 0.525 → 0.748 | 0.95 | 24 |
| InSwapper 128 (instead of HyperSwap) | blurrier | same | unusable | 0.525 → 0.762 | 0.96 | 22 |
| + GFPGAN 1.4 on the swap | sharper, more "generic" face | same | — | 0.652 (identity drops) | 0.94 | 57 → too slow |
| + CodeFormer w=0.7 on the swap | similar | same | — | 0.634 | 0.94 | 83 → too slow |
| MuseTalk (audio-driven) | not tested: our clips have no audio track and the bridge has no audio path; ~4.5 GB weights (disk) | | | | | |

\* the cosine uses the same ArcFace that conditions the swapper, so it flatters the swap; compare rows, not absolutes.

**Verdict:** swap with the lower-face mask for human refs (real mouth interior, identity/eyes stay JoyAI's, no
glasses transfer, fast); LP `rel` stays for G-Star; restorers cost 30–60 ms and lower identity, so they're off.
Swap needs MediaPipe to find the real face (it does through glasses); fades with hand-over-mouth/turns as before.

Live (real server, 45 s, test_face_landscape, 30 fps in, 25 fps out), round-1 LP: human `abs` 1000 frames played,
0 dropped / 0 repeated, glass-to-glass 915 / 948 ms (median / p95), facefix in time 99.3 %, lip-gap 0.94; G-Star
`rel` 0 / 0, 930 / 1002 ms, 98.8 %, lip-gap 0.74 (a first G-Star run hit server flow drops: 293 repeats, 0.52).
Swap through the bridge (mock server): 25 ms render, in time 89 % in a 10 s run (start-up misses). A live swap run on
the real server is still pending: the server stopped accepting sessions during round 2.

## Licenses (for the record; the project is academic, so non-commercial models are allowed)
| Component | License | OK? |
|---|---|---|
| LivePortrait code + weights (KwaiVGI, HF `KwaiVGI/LivePortrait`) | MIT | yes |
| InsightFace detector that LivePortrait uses by default | **non-commercial** | **not used**: we crop from MediaPipe landmarks instead |
| MediaPipe FaceLandmarker / HandLandmarker | Apache-2.0 | yes |
| FasterLivePortrait (TensorRT port, warmshao) | MIT | yes (next step) |
| GFPGAN 1.4 (TencentARC) | Apache-2.0 (the "clean" arch we load has no NVIDIA-licensed ops) | yes |
| RestoreFormer / RestoreFormer++ (wzhouxiff) | Apache-2.0 | yes |
| CodeFormer (sczhou) | **S-Lab License 1.0, non-commercial** | **no** |
| LivePortrait animal model's X-Pose detector | non-commercial | not used (animal weights tested with our MediaPipe crop) |

## How it works (`facefix.py`)
`FacePass` per frame pair (input frame the output was made from, JoyAI output frame):
1. `analyze_input` (CPU): MediaPipe face (478 landmarks + head pose from the transformation matrix) and hand
   landmarks on the input. Live: run at capture time, cache by seq.
2. `analyze` (CPU, stateful): MediaPipe face on the output (track nearest the input face). Gates → target weight
   = input face found × pose (yaw/pitch/roll ramps, input and output face) × (1 − hand over mouth) × (0.7 if the
   box is mapped from the input). Weight follows with an EMA (in 0.25, out 0.4 per frame). LivePortrait-style crop
   rect (eye→lip axis, scale 2.0) from the output face, smoothed (less when moving fast, none on jumps).
3. `render` (GPU): motion extractor on both 256 crops; output keypoints with the lip keypoints `[6,12,14,17,19,20]`
   replaced (`abs`) or offset by the input's deviation from LivePortrait's neutral closed mouth (`rel`);
   stitching; warp + decode at `--lp-size`; colour-match; paste the lower-face mask × weight into the face ROI.
   Weight 0 → frame passes through untouched and the GPU is idle.

Speed fixes found on sm_120 (worth knowing for any LivePortrait use on Blackwell):
- cuDNN runs the warping module's fp16 conv3d (esp. the 7×7×7 head) on a slow `vol2col` path in NCDHW: **channels-last
  weights** cut the warp module 50 → 17 ms (needs `.view` → `.reshape` in three LivePortrait files; `setup.sh` does it).
- Decoder in fp16 + `torch.compile(max-autotune-no-cudagraphs)`: 35 → 20 ms. Compiling the warp module or the
  whole step made it slower. Batching frames does not help (compute-bound).
- `--lp-size 128` (appearance/warp/decoder on a 128 crop, renders 256): 4× less work; the motion extractor still
  sees 256.

## Usage
```bash
bash cloud/facefix/setup.sh                    # venv (~8 GB), LivePortrait clone + patch, weights; RESTORE=1 for restorers
cd cloud/facefix
E=../pipelines/joyai/experiments/2026-11-02-mouth
.venv/bin/python facefix.py --input $E/in_C_mouthprompt.mp4 --output $E/C_mouthprompt.mp4 \
    --seq-json $E/C_mouthprompt.json --out refined.mp4 --sbs sbs.mp4 --zoom zoom.mp4 --stats run.json \
    --lp-size 128 --compile --pipeline            # G-Star: add --lip-mode rel
../.venv/bin/python ../client/mouth_sync.py --input $E/in_C_mouthprompt.mp4 --output refined.mp4 --stats $E/C_mouthprompt.json
.venv/bin/python review.py --stats run.json --video zoom.mp4 --out sheet.jpg   # even + event frames
```
Without `--seq-json` the pairing comes from DTW over grid motion energy (`--align dtw`, roughly right; linear
scaling was clearly worse: jawOpen corr 0.21 vs 0.34 for JoyAI on the human clip). `bench_lp.py` times the
LivePortrait parts.

## Live: `live_bridge.py --facefix abs|rel`
Wired in, off by default. Run the bridge with this venv's python when the flag is on (it has torch, mediapipe and
websockets); with the flag off the bridge still runs from `cloud/.venv` (facefix is imported lazily).
```bash
cloud/facefix/.venv/bin/python cloud/client/live_bridge.py --url ws://127.0.0.1:18080/ws \
  --input /dev/video0 --output ffplay --portrait --out-size 720x1260 \
  --ref assets/char_front.png --prompt "$(cat cloud/pipelines/joyai/prompts/gstar.txt)" --facefix rel
# human ref: --ref assets/blonde-gamer-girl.png --prompt "$(cat cloud/pipelines/joyai/prompts/human_blonde.txt)" --facefix swap
```
Flags: `--facefix-size 128|256`, `--facefix-restore none|sharpen|gfpgan|codeformer`, `--facefix-extra '<facefix.py
flags>'` (e.g. `--yaw-hi 45 --no-hands`), `--review out.mp4` (input | JoyAI | played, ≤ 1920 wide),
`--save-input in.mp4` (+ `in.json` with source_seq per played frame, for `mouth_sync.py`). `--output ffplay` opens a
low-delay ffplay window ("ai-livestream"); `/dev/video10` and `.mp4` sinks work as before.

How: the sender caches every sent frame (generation size) by seq and starts `analyze_input` on it in a CPU thread;
the receiver pairs each decoded frame with its input through the server's `source_seq` and queues output analysis
(CPU thread, in order) + render (GPU thread); the player takes the refined frame if it is done by its playout
time, else plays the raw JoyAI frame and cancels the render. Warm-up (cuDNN autotune + compile) takes ~15 s before
the session starts.

Offline test of the whole bridge path: `mock_server.py` replays a recorded, frame-aligned JoyAI output in the
server's 8-frame chunk rhythm (`experiments/2026-11-02-live/`, review sheets `sheet_review_mock_*.jpg`):

| Run (20 s, 30 fps in, 25 fps playout) | facefix hit rate | render | lip-gap corr (mouth_sync on the played video) |
|---|---|---|---|
| human ref, `abs` | 97.8 % (misses: first frames after start) | 28 ms | **0.94** (JoyAI 0.25) |
| G-Star, `rel` | 98.5 % | 27 ms | **0.63** (JoyAI 0.03) |

0 dropped frames; the ~30 repeated frames come from the mock's chunk timing at start/end. Glass-to-glass in the
mock (~1.0–1.1 s) is not meaningful; see below for runs against the real server.

## Next
- **TensorRT via FasterLivePortrait** (MIT; grid-sample-3D plugin) to make `--facefix-size 256` real-time (sharper
  mouth on close-ups) and leave GPU time for GFPGAN/CodeFormer on human refs.
- Non-commercial options are allowed (academic project): InsightFace detection/landmarks for LivePortrait's own
  crop, CodeFormer (most natural restorer here, 54 ms), X-Pose for the animal model. Not tried with them yet.
- `--facefix-extra '--region lip+eyes'` for blinks (not evaluated); a clip with 25–45° head turns to tune the pose gate.
