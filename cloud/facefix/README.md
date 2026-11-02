# facefix: lip sync (and optional face restoration) as a post-process on the laptop GPU

JoyAI-Video-Edit barely moves the generated mouth with the speaker's (lip-gap correlation 0.25 for a human ref,
0.03 for G-Star, see `../pipelines/joyai/experiments/2026-11-02-mouth/`). facefix fixes that on the laptop's idle
RTX 5080 Laptop, between "frame received" and "frame played": **LivePortrait** re-renders the lower half of the
*generated* face with the lip keypoints of the *real* face, then pastes it back with a feathered mask. Identity,
eyes, hair and background stay JoyAI's.

Status: offline prototype, measured on recorded clips. Not wired into `live_bridge.py` yet (plan at the end).

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

## Licenses (the user streams on TikTok, i.e. commercial use)
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

## Integrating into `live_bridge.py` (recommendation)
1. Keep a dict `seq → (input frame 840×480, analyze_input future)` in the sender; run `analyze_input` in a CPU
   thread at capture (23 ms < 33 ms per frame; drop hands to every 2nd frame if the CPU is busy). Evict after ~3 s.
2. In the receiver, each decoded frame has `source_seq` in its `output_frame` meta: look up the input, run
   `FacePass.analyze` in a worker thread and `render` in another (or a 2-deep queue), and put the refined frame in
   the playout queue. Budget: 25 ms GPU + ~10 ms CPU per frame at lp 128 → ~38 fps, so 25 fps with ~35 % headroom;
   adds ~40 ms latency (glass-to-glass 1.03 → ~1.08 s). If a frame isn't done in time, play the raw frame
   (the weight EMA hides it).
3. Default off (`--facefix` flag, new features default to bypass); `--facefix-lip-mode rel` for cartoon refs,
   `abs` for human refs; watch GPU temperature (the 5080 Laptop sat at 64–74 °C in these runs).
4. Next for quality: **TensorRT via FasterLivePortrait** (MIT; their grid-sample-3D plugin) to make lp 256 real-time
   (sharper mouth on close-ups) and leave room for GFPGAN on human refs. Also worth trying: `--region lip+eyes` for
   blinks (not evaluated), and recording a clip with 25–45° head turns to tune the pose gate.
