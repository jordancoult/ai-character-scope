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

## G-Star "fuzzes around in a cycle" (static-input diagnosis, `experiments/2026-11-02-jitter/`)
Static input: one frame of webcam_7643311 with the mouth open (jawOpen 0.33, no hands), 9:16 crop, 30 s at
30 fps, through the real server (portrait 480×840, G-Star) and the bridge, 45 s per run. `jitter.py` measures the
face region of the review video (input | JoyAI | played): mean abs diff between consecutive frames in the mouth box,
MediaPipe lip-landmark motion and lip-gap std (% of face size), and the dominant period (autocorrelation of the
high-passed series; 8 = JoyAI chunk).

| Run | Mouth diff raw → played | Lip step raw → played | Lip-gap std raw → played | Period (raw) | facefix |
|---|---|---|---|---|---|
| `--facefix off` | 0.96 → 0.96 | 0.10 → 0.10 % | 0.70 → 0.68 % | 16 frames | — |
| `rel`, before | 1.18 → **1.93** | 0.12 → **0.22** % | 2.00 → 2.12 % | 16 | 95.8 % in time, **43 raw frames** |
| `rel`, after (live, anchor 0.05) | 1.39 → **1.49** | 0.165 → **0.154** % | 1.71 → 1.76 % | 16 | 99 % in time, 7 held, 3 raw |
| offline replay of the same raw JoyAI, anchor 0 / 0.05 / 0.01 / 0.003 | 1.36 → 1.47 / 1.43 / 1.44 / 1.46 | 0.17 → 0.18 / 0.14 / 0.14 / 0.14 | 1.73 → 2.04 / 1.66 / **1.41** / 1.29 | | |

Findings (`plot_*.png`: raw vs played series with an 8-frame grid; `review_compare.mp4` / `_zoom.mp4`: JoyAI raw |
facefix before | facefix after):
- **Ours, main cause: misses popped to the raw frame.** When a refined frame wasn't ready at its playout time the
  bridge played the raw JoyAI frame between refined ones; the mouth jumped (closed → open → closed) for one frame,
  ~1 per second (the dips in `plot_rel.png`). Fix: the player waits up to `--facefix-wait 15` ms, then repeats the
  previous refined frame (`--facefix-hold 2` frames) before ever falling back to raw.
- **Ours, second: `rel` was anchored to the current frame's mouth**, so JoyAI's own mouth wobble passed straight
  through and LivePortrait added its own on top. Fix: the base is a slow EMA of the output's mouth keypoints
  (`--rel-anchor 0.01`, about 4 s), driving keypoints smoothed (`--in-smooth 0.3`); with a static input the target
  mouth is now constant. Lip-gap wobble 2.04 → 1.41 % (below JoyAI's own 1.73 %); lip sync on the talking G-Star clip
  unchanged (0.70). The lower-face mask ramp is softer (0.36–0.58) to hide the seam under the eyes.
- Checked and fine: the input facefix sees is constant (input diff 0.01, source_seq steps 1 except the server's own
  input drops: 2s in ~6 %, never repeats); One-Euro on; blend weight 0.996–0.998 ± 0.02 (no cycling).
- **JoyAI itself has a cycle:** on a perfectly static input the raw output's mouth region changes by ~1–1.4 grey
  levels per frame with a dominant period of **16 frames** (two 8-frame chunks, autocorr 0.15–0.3), plus slow
  open/close drifts of the mouth over 1–2 s. facefix after the fixes follows it at ~1.07× (texture), and damps the
  mouth-shape part. To remove it at the source (server side, untested): `freeze_kv_on_static` (reuse the temporal
  KV when the input is static), blending the first frames of each chunk with the previous chunk's last frames
  (cross-fade at chunk boundaries), a fixed noise seed per session, a smaller `max_temporal_ids` / shorter temporal
  window, or one more denoising step (E_steps3 run: slower).

## Clarity: facefix blurred the face (eyes and brows too) — fixed (`experiments/2026-11-02-clarity/`)
Cause (ours): the old lower-face mask started near eye level in LivePortrait's crop, so the soft 128 px render
replaced eyes, brows and cheeks; plus bilinear resampling and no detail transfer. New defaults for LivePortrait:
- `--mask chin`: ellipse in frame pixels from 1.5× the mouth→subnasale distance above the mouth to below the chin
  (+25 % for the jaw drop), 1 mouth-width half-width, 5 px feather, rotated with the mouth line. Everything outside
  is copied back pixel-for-pixel: measured max abs difference in the eye/brow box = **0** on every frame (before
  encoding).
- `--detail sep`: frequency separation inside the mask: low band (σ 2 px) from the render, high band from JoyAI
  where both agree at low frequency (|Δ| < 20 grey levels); where they differ (open-mouth interior, moved lips) the
  render's band, boosted ×1.6 (`--render-sharpen 0.6`).
- One cubic warp from the render to the frame, colour match in frame space.

Laplacian-variance ratio refined / JoyAI raw (1 = as sharp as JoyAI; `clarity.py`), offline replay on the recorded
raw JoyAI output:

| Clip | Version | Whole face | Eyes + brows | Mouth | Lip sync (talking clip) | Mouth jitter / lip-gap wobble (static) |
|---|---|---|---|---|---|---|
| static open mouth (G-Star, portrait) | before (lower mask) | 0.55 | 0.43 | 0.50 | | 1.53 / 1.40 % (raw 1.25 / 1.74) |
| | chin mask only | 0.93 | 1.00 | 0.71 | | |
| | **chin + sep (default)** | **0.97** | **1.00** | **0.91** | | 1.41 / 1.26 % |
| talking G-Star (C clip) | before | 0.75 | 0.69 | 0.90 | 0.71 | |
| | **chin + sep (default)** | **1.18** | **1.01** | 1.63* | **0.72** | |

\* above 1 because the refined mouth is open with teeth where JoyAI's is closed (more edges), not ringing; see
`sheet_talking_zoom.jpg`. A tighter mask top (at the nose) cost lip sync (0.53–0.63); 1.5× restores it.
ms/frame unchanged (render 25–30 ms, 31–34 fps pipelined incl. writing). `--lp-size 256` would still be 58 ms.
Review videos: `review_static_zoom.mp4`, `review_talking_zoom.mp4` (JoyAI raw | before | after, zoomed) and the
`_full` versions. A live confirmation run didn't start (server busy).

## G-Star: jumping mask → wide box, input-anchored tracking (`experiments/2026-11-02-gstar-mask/`)
User report: the ellipse visibly jumped around on G-Star as MediaPipe lost him or locked onto another spot.
- `--mask-shape box` (default): rounded rectangle (superellipse p=5), half-width 1.2 mouth widths (G-Star's grin runs
  past the detected corners), top at 1.5× the mouth→nose distance, 5 px feather. Eyes/brows still pixel-identical
  (max abs diff 0 on every clip).
- **Tracking relative to the input face** (`RelTrack`, on by default; `--no-stabilize` = old per-frame behaviour):
  the crop rect and the mask geometry are stored relative to the real face (origin at the nose, scale = face size,
  angle = eye line; One-Euro-smoothed), so they move with the stable real head. A measurement on the generated face
  is rejected if it jumps > 8 % of the face size from the previous measurement or lands > 20 % from the state, or the
  size changes > 25 % (`--track-jump 0.08`); accepted ones are blended in with an adaptive EMA (0.15 → 0.6 on
  consistent motion); a new position that persists 10 frames is taken as real ("switch"). On a rejection the
  geometry is held and the effect fades out; if the generated face is merely not detected, the held,
  input-anchored geometry keeps the effect for up to 8 frames (`--track-hold`), then fades.
- The same tracker drives the LivePortrait crop (a jumping crop also jumped the render).
- The render is split into a GPU stage and a CPU compose stage (separate threads in the bridge), and the compose step
  got cheaper (polygon mask, subsampled colour stats, cv2 detail transfer): 36 → 30 ms per frame end to end.
- Fixed a crash when both faces vanish during a fade (bridge played raw frames on the exception).

Mask-centre motion per frame in % of face size (std / max / count of jumps > 8 % while visible), per-frame
measurement on the generated face (= old behaviour) vs tracked:

| Clip | Per-frame (old) | Tracked (new) | Tracker: rejected / switches / missing | Mean weight | Lip sync |
|---|---|---|---|---|---|
| talking G-Star (exact pairing) | 0.78 / 7.2 / 0 | 0.79 / **4.8** / 0 | 0 / 0 / 31 | 0.98 | **0.67** (0.72 unstabilised) |
| static open mouth | 0.54 / 11.1 / 1 | **0.30** / 8.7 (one start-up frame at w 0.5) / 0 visible | 2 / 0 / 6 | 0.99 | — |
| desk, portrait (DTW pairing) | 1.14 / 15.3 / 3 | **0.70 / 7.9 / 0** | 4 / 0 / 50 | 0.88 | — |
| desk, landscape (DTW pairing, gestures) | 1.91 / 14.0 / 14 | 1.95 / 24.7 / 14* | 54 / 2 / 133 | 0.79 | — |
| boxer (side-on) | — | — | effect off (weight 0), as before | 0.00 | — |

\* the desk-landscape steps follow the real head (fast gestures; the offline DTW pairing skips 2–3 input frames at
once); the large ones (16–25 %) happen while the effect is faded in from 0 (weight ≤ 0.25).
Live (real server, G-Star portrait, new `gstar.txt`, talking clip, 45 s): refined in time **97.6 %** (7 held, 21 raw),
render+compose 39 ms under load (it was 40.7 ms and 58 % in time before the split/optimisation), glass-to-glass
957 / 1269 ms, lip sync 0.60, 13 repeated / 0 dropped frames.
Review: `review_dbg_*.mp4` (face close-ups with the mask outline: orange = edge, green = 50 %), `review_clean_*.mp4`
(JoyAI | refined), `review_live.mp4`, `sheet_dbg_*.jpg`.

## Binding the mask to G-Star's mouth: tracker comparison (`experiments/2026-11-02-track/`)
User: "still janky… loses track easily". The mask was placed from MediaPipe landmarks on the generated face (via the
input-anchored filter), which drift and drop out on a cartoon. Tested trackers that follow G-Star's own pixels
(`--track flow|sam2|anchor`, `track.py`), offline on five G-Star recordings, then live:
- **flow (new default):** dense optical flow (OpenCV DIS, 160 px ROI around the mouth, CPU ~2–3 ms) propagates the
  mask geometry frame to frame with a RANSAC similarity fit (lip motion inside is rejected as outliers); MediaPipe
  only re-anchors it slowly when it agrees (drift) and is ignored when it disagrees unless that persists while the
  flow is unreliable; the mouth line is kept within 12° of the real face's eye line.
- **sam2:** SAM 2.1 tiny, streaming (camera-predictor fork `third_party/sam2rt`, 512 px, GPU preprocessing),
  box-prompted once from MediaPipe's lips, segments the actual grin (true shape), paste mask = segment grown around
  the lips and down for the jaw, feathered; keeps the connected component that overlaps the previous mouth, nothing
  above the eye–mouth midline; re-prompts on loss/drift. 9.4 ms alone, 15–19 ms next to the LivePortrait render.
- **anchor:** the previous version (MediaPipe geometry relative to the input face, jump rejection).
- LivePortrait's own implicit lip keypoints were checked on G-Star crops: they cluster at the mouth centre (no corner
  /extent information), so they can't size a mask. CoTracker3 online not tried: its sliding window adds ≥ 8 frames
  of latency, more than the whole render budget.

Mask-centre step per frame (% of face size): std / max / jumps > 8 %; track losses; mean effect weight:

| Clip | flow | sam2 | anchor |
|---|---|---|---|
| talking G-Star (lip sync) | **0.46 / 4.5 / 0**, 3 flow failures, w 0.98, **lip 0.65** | 0.92 / 5.1 / 0, 3 lost, lip 0.60 | 0.79 / 4.8 / 0, lip 0.66 |
| static open mouth | **0.08 / 1.7 / 0** | 0.31 / 4.9 / 0, 5 lost | 0.30 / 8.7 / 1 |
| desk portrait (hands near the face) | **0.57** / 13.1 / 1 | 0.85 / 15.2 / 1 | 0.70 / 7.9 / 0 |
| desk landscape (gestures, head motion) | **0.56 / 6.2 / 0**, 0 re-inits | 6.0 / 93 / 45 (locks onto hands; eye-region diff up to 47) | 1.95 / 24.7 / 14, 54 rejected |
| boxer (side-on) | effect off (w 0.00) | off | off |

Size changes per frame: flow ≤ 4 %, anchor ≤ 40 %, sam2 ≤ 47 %. Eyes/brows pixel-identical with flow on every clip.
SAM 2 fits the grin's true shape beautifully on the easy clips (`sheet_sam2_eval_desk_portrait.jpg`) but loses it
to hands and fragments on the hard one, and costs 15–19 ms; flow is the most stable everywhere at ~0 GPU cost, so
it is the default (`--track sam2` stays available).
Live (real server, G-Star portrait, `gstar.txt`, talking clip, 45 s, flow): refined in time **99.5 %** (4 held,
5 raw), 972 played / 1 repeated / 0 dropped, glass-to-glass 932 / 1403 ms, lip sync 0.53 (this G-Star renders a
smaller grin than the recording the offline numbers come from).
Review: `review_dbg_{flow,sam2}_*.mp4` (mask outline), `review_clean_flow_*.mp4` (JoyAI | refined),
`review_live_flow.mp4`, `sheet_*.jpg`.

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
