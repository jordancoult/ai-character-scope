# lpfast: real-time LivePortrait face pass on the laptop RTX 5080 (TensorRT 11, sm_120)

Last stage of the JoyAI pipeline, on the local GPU: the generated character's face (G-Star or any person) is
re-rendered so that its **mouth, eyes and brows follow the real face**. Head pose, framing, identity and everything
outside the face stay JoyAI's. It replaces the old `cloud/facefix` LivePortrait/HyperSwap pass (128 px, judged "not an
improvement"); same bridge flag, `--facefix rel|abs`, plus `--facefix-preset gstar|girl`.
It also animates a **still image** with the webcam (`live_test.py`, with head rotation/size transfer).

See **TUNING.md** for the full parameter sheet.

## Quick start (tuning with the webcam)
```bash
cd ~/personal/ai-livestream
# G-Star still image (upper half) driven by the webcam, your tuned preset; window = webcam | source | result
cloud/facefix/.venv/bin/python cloud/lpfast/live_test.py --source assets/char_front.png --crop 0,0,1,0.5 --preset gstar
# the same on a recorded JoyAI output, driven by the webcam (or --driver auto = the recording's own input)
cloud/facefix/.venv/bin/python cloud/lpfast/live_test.py --preset gstar
```
Type in the terminal + Enter while it runs:

| Command | Effect |
|---|---|
| `/gain lip 2.5` | region gain: `lip`, `eye`, `brow`, `brow2`, `rest`, or `all` (scales every region) |
| `/gain` | show the current gains |
| `/preset girl` / `/preset` | switch to a saved preset (mode, gains, pose) / list presets |
| `/save NAME` | save the current mode + gains + pose as preset NAME (`presets.json`; commit it to keep it) |
| `/limit lip 0.25` / `/limit lip 0.14:0.21` / `/limit lip off` / `/limit` | soft limit of a region's motion (max, or knee:max) / show |
| `/limit lipclose 0.01:0.025` | how far the lips may close past the character's resting mouth (stops them passing through each other) |
| `/euro 1.0 5` / `/euro off` / `/euro` / `/euro upper 0.5 3` / `/euro lower off` | smoothing of your expression (One-Euro), for both halves or only `upper` (brows, eyes) / `lower` (mouth, cheeks, jaw): lower first number = calmer when still, higher second = less lag on fast moves |
| `/stats on` / `/stats off` | periodic stats line on / off (always starts on, not saved) |
| `/gate yaw 25 40` / `/gate pitch 18 30` | your head turned / tilted too far: effect fades out from the first to the second angle |
| `/gate cover 0.55 0.35` | mouth covered (hand, mic, cup...): fades out as mouth visibility drops from 0.55 to 0.35 |
| `/gate score 0.65 0.5` | face confidence (turned away / profile): fades out from 0.65 down to 0.5 |
| `/gate hold 6` / `/gate visibility off` / `/gate` | frames to wait after a cover / disable the occlusion+confidence gates / show |
| `/reset` | re-take your neutral face and head reference (after moving / changing seat) |
| `q` | quit |

`live_test.py` flags: `--preset`, `--gains lip=..,eye=..` (on top of the preset), `--mode rel|abs`, `--pose on|off`
(head rotation + size from the driver; default on for an image source, off for a video), `--smooth 0..0.9` (calmer,
laggier driver), `--neutral-ema` (how fast your neutral face re-adapts, 0.01 ≈ 4 s, 0 = first frame),
`--crop x0,y0,x1,y1` (fractions), `--layout 1|3`, `--save out.mp4`, `--driver auto|VIDEO`, `--camera /dev/video0`,
`--limits lip=0.21,eye=0.10` (or `lip=0.14:0.21`, `lip=off`), `--no-visibility`. Default source: the G-Star test image
`assets/g-star-screenshot-no-hat.png`. `/save` stores limits and gates in the preset too.

### Limits and gates (2026-11-04)
- **Soft limits** (`lp_pass.DEFAULT_LIMITS`, per region `(knee, max)` on the size of the expression change after the
  gain): linear up to the knee, then tanh-compressed, never past max. Calibrated by sweeping a real driver's most
  extreme frame on G-Star: lips fine to 0.18, stretched at 0.24, torn ("double mouth") at 0.31 -> `lip (0.14, 0.21)`;
  eyes shut cleanly at 0.075, squashed at 0.12 -> `eye (0.07, 0.10)`; brows distort at 0.17 -> `brow (0.10, 0.14)`;
  `brow2`/`rest` uncalibrated guesses. With the limit, x3.5 / x5 lip motion renders as a big clean open mouth.
  **Closing** has its own limit (`lipclose (0.01, 0.025)`): the lip change is projected on a measured "mouth opens"
  direction (`lip_open_dir.npy`); pressing the lips tighter than your neutral face, times the gain, made G-Star's lips
  thin at 0.02 and pass through each other from 0.04.
- **Jitter**: output jitter grows with the gains (old 1.3/1/-1 -> the gstar preset: +40 % on the user's webcam
  footage; low light makes the webcam noisier too). A One-Euro filter on the driver expression (default
  `euro (1.0, 5)`) brings it back below the old level (-32 %); tune live with `/euro`, saved in presets.
- **Drop completely** (the effect fades out, JoyAI's frame plays): no face; your head yaw 25->40 deg or pitch (relative
  to your usual) 18->30 deg; the generated face's own pose 30->45 / 20->35 deg; **mouth covered**: a face-parsing net
  (BiSeNet, FP16 TensorRT, 0.6 ms) checks how much of your mouth's landmark region is lips / mouth (normal >= 0.66,
  covered 0, glasses held over the mouth 0.32); **turned away**: RetinaFace confidence (frontal 0.76-0.90, profile
  0.04-0.5). Face confidence does not drop for partial occlusion, hence the parser. After a cover the effect waits
  6 clear frames. Both checks run every 2nd frame in the driver thread (render stays at 36 ms mean / 39 p95).
- **No stuck states** (fixed 2026-11-05): after 8 bad driver frames in a row (mouth not visible, face confidence
  < 0.6, yaw > 40 deg) the driver face is re-detected (the landmark tracker could lock onto a hand / cheek after a
  cover or turn and keep the gates closed forever); the hold after a cover counts frames above the ramp's midpoint
  (hysteresis); `/reset` re-detects too. Tested: cover 40 frames -> off; back to full 8 frames (0.3 s) after it ends.
- Not covered yet: a hand over the **eyes only** (the parser doesn't label eyes/brows reliably at webcam scale).

### Presets (`presets.json`, tuned live by the user 2026-11-03; also the defaults of each mode)
| Preset | Mode | Pose | lip | eye | brow | brow2 | rest |
|---|---|---|---|---|---|---|---|
| `gstar` | rel | off | 2.65 (2.5–3 fine) | 2.5 | 1.5 | 0 | 0 |
| `girl` (human ref) | abs | off | 2.0 | 1.3 | 1.3 | 1 | 1 |

Bridge: `--facefix-preset gstar` (implies `--facefix rel`); live `/preset NAME`, `/gain ...` there too.

## Results (recorded 5090 runs replayed with exact `source_seq` pairing)
Correlation of the output with the driver (MediaPipe; `expr_sync.py`: lip gap, eyelid opening, brow height):

| Clip | | lips | eyes | brows |
|---|---|---|---|---|
| G-Star talking close-up (`G_gstar`) | JoyAI alone | 0.03 | 0.03 | −0.19 |
| | **lpfast rel, default gains** (lip 1.3, eye 1, brow −1) | **0.77** | **0.59** | **0.60** |
| Human ref, same driver (`C_mouthprompt`) | JoyAI alone | 0.25 | 0.28 | 0.57 |
| | **lpfast abs, full expression** (all regions 1) | **0.98** | **0.98** | **0.97** |

- Human `gestures/glasses` clip (DTW-paired): hands at the face edge stay intact; the effect fades on large tilts.
- G-Star boxer clip (profile, fast motion): JoyAI draws no face most of the time; lpfast passes those frames through.
- G-Star's brows on the JoyAI recording measured mirrored (brow gain +1: −0.54, −1: +0.58), but on the still image
  with the user's own face +1.5 looked right to the user (the `gstar` preset); trust the eye test. His brows sit
  under the cap brim, so the motion is small either way.
- The numbers above are from the earlier defaults (lip 1.3, eye 1, brow −1); the user later preferred much more
  motion (`gstar` preset).

### Gains (per region, live-tunable)
Regions = groups of LivePortrait's 21 implicit expression keypoints: `lip` (6, 12, 14, 17, 19, 20), `eye` (11, 13,
15, 16, 18), `brow` (1, 2), `brow2` (9 y/z), `rest` (3/4 y, 5/8 z: cheeks/jaw; with brow2 = LivePortrait's full
expression set). 1 = the speaker's motion as is, 0.5 = half, 0 = JoyAI's own expression, negative = mirrored.
Bridge: `--facefix-gains 'lip=2.65,eye=2.5,brow=1.5'`; while running type `/gain lip 1.5`, `/gain all 0.8`, `/gain`
(show) + Enter (applied from the next frame, no restart). In `abs` mode a gain blends towards (>1: beyond) the
driver's expression.

Review media (gitignored, regenerate with `run_offline.py`): `experiments/2026-11-03-first/*/review.mp4`
(input | JoyAI | lpfast).

## Speed (RTX 5080 Laptop, 16 GB, power-capped at ~108 W / ~1.7 GHz under load)
| Stage | ms |
|---|---|
| driver (webcam) analysis: 203-pt landmarks + motion | ~4, **off the critical path** (when the frame is sent, own thread/stream) |
| generated-face analysis: landmarks, 256 crop, motion + appearance | ~8 |
| warp + SPADE decoder, FP16 TensorRT, 512 px | **22–23** (the floor: FP16 convs, compute/power bound) |
| stitching + keypoint math + face-ROI paste (GPU) | ~1.5 |
| **render per output frame** | **34 mean, 36 p95** |
| capacity, frames back to back | **~31 fps** (24 % headroom over 25 fps) |
| added glass-to-glass latency through the bridge | **0** (993 vs 997 ms median, mock server): the render finishes while the frame waits in the playout buffer |

35–40 fps would need FP8/INT8 or a lower-resolution warp (quality loss), so it isn't done. Things tried that don't
help: TF32 flag, more FP16 (InstanceNorm / GridSample), TensorRT builder optimisation level 5, CUDA graphs,
overlapping the three stages (GPU already saturated: 29 fps vs 31 serial).

## How it works
- **FasterLivePortrait on Blackwell without its plugin.** FasterLivePortrait needs TensorRT 8 (no sm_120) and a
  compiled `grid_sample_3d` plugin (needs nvcc). TensorRT 11 supports ONNX `GridSample` on 5-D input natively, so
  `build_engines.py` rewrites the two custom `GridSample3D` nodes of `warping_spade-fix.onnx` into standard
  `GridSample` (opset 20; max abs error vs `torch.grid_sample` 5e-7) and converts the warp to FP16 with
  normalisation/softmax/grid-sample kept in FP32 (max output diff vs FP32 0.022, mean 0.001 on a 0..1 scale).
  `flp_compat.py` runs FasterLivePortrait's model wrappers on TensorRT 11 (name-based I/O, engine output order,
  stubbed `insightface`); the per-frame path uses `trt_runtime.Engine` directly.
- **Per frame** (`lp_pass.py`): track both faces with the 203-point landmark model from the previous frame
  (RetinaFace only when lost / on a jump; a detection frame passes JoyAI through so it never stacks with a render);
  LivePortrait motion of the driver crop and of the JoyAI crop; change only the lip (6) / eye (5) expression
  keypoints:
  - `rel` (cartoon refs): `exp = EMA(JoyAI exp) + gain · (driver exp − EMA(driver exp))` (slow EMAs ≈ the
    character's resting mouth, without JoyAI's chunk wobble, and the speaker's neutral face);
  - `abs` (human refs): the driver's lip/eye expression as is;
  then stitching, warp (512 px), paste with LivePortrait's soft face mask into the face ROI on the GPU.
- **Driver crop is jaw/brow-independent** (`Face._stable_crop`): LivePortrait sizes the crop on the landmark box,
  which includes chin and brows, so opening the mouth or raising the brows grew the box, the face looked smaller and
  the animated face shrank. The driver crop is now kept relative to the eyes (first frame's crop, scaled by the
  inter-ocular distance, rotated with the eye line); with pose transfer, head size follows the inter-ocular distance
  (LivePortrait's scale estimate moved with the jaw: corr −0.23). Head translation isn't transferred (stitching
  cancels it).
- **Still images**: `pose=True` adds the driver's head rotation (relative to its first frame) and size. Cartoon faces
  score low in RetinaFace at close framing, so detection retries at threshold 0.3; crops replicate the border (no
  black band where the crop leaves the image).
- **Gates**: no face on either side → JoyAI passes through; the effect fades (0.25 per frame) over 30–45° yaw /
  20–35° pitch of the generated face (smoothed), where a frontal re-render would warp profiles.

## Run
```bash
# once: TensorRT 11 + deps into the facefix venv (torch 2.9.1 cu128), ONNX models, engines (~1 min on the 5080)
bin/uv pip install --python cloud/facefix/.venv/bin/python "tensorrt-cu12>=10.9" onnxconverter-common \
    ffmpeg-python torchgeometry munch scikit-image
git clone --depth 1 https://github.com/warmshao/FasterLivePortrait.git cloud/lpfast/third_party/FasterLivePortrait
cloud/facefix/.venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('warmshao/FasterLivePortrait', \
    local_dir='cloud/lpfast/checkpoints', allow_patterns=['liveportrait_onnx/*.onnx'], ignore_patterns=['*warping_spade.onnx'])"
cloud/facefix/.venv/bin/python cloud/lpfast/build_engines.py    # all engines -> cloud/lpfast/checkpoints/trt
# live (G-Star): add --facefix-preset gstar to the usual bridge command, run with the facefix venv
cloud/facefix/.venv/bin/python cloud/client/live_bridge.py --url ws://127.0.0.1:18080/ws --input /dev/video0 \
    --output ffplay --portrait --out-size 720x1260 --ref assets/char_front.png \
    --prompt "$(cat cloud/pipelines/joyai/prompts/gstar.txt)" --facefix-preset gstar   # human refs: girl
# offline replay of a recorded run (input as sent + JoyAI output [+ stats JSON with source_seq, else DTW])
cloud/facefix/.venv/bin/python cloud/lpfast/run_offline.py --out OUT.mp4 --inp IN.mp4 --stats OUT.json \
    --dst cloud/lpfast/experiments/NAME --mode rel [--gains lip=2.65,eye=2.5,brow=1.5]
cloud/facefix/.venv/bin/python cloud/lpfast/expr_sync.py --input IN.mp4 --output cloud/lpfast/experiments/NAME/lp.mp4 --stats OUT.json
```
Disk: TensorRT 11 libs ~6.3 GB in the venv, ONNX ~0.7 GB, engines ~0.6 GB.

## Open / next
- Still image + large head turns: a pale smear at the cone's edge (disocclusion), a LivePortrait limit.
- Eye-only occlusion isn't detected (see Limits and gates); the mouth-cover gate is tested on synthetic covers and one
  real clip (glasses held over the mouth), not yet on your webcam.
- Not yet run against the real 5090 server (validated through the bridge with `cloud/facefix/mock_server.py`
  replaying a recorded 5090 run at its chunk rhythm).
