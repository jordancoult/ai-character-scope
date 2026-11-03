# lpfast: real-time LivePortrait face pass on the laptop RTX 5080 (TensorRT 11, sm_120)

Last stage of the JoyAI pipeline, on the local GPU: the generated character's face (G-Star or any person) is
re-rendered so that its **mouth and eyes follow the real face**. Head pose, framing, identity and everything outside
the face stay JoyAI's. It replaces the old `cloud/facefix` LivePortrait/HyperSwap pass (128 px, judged "not an
improvement"); same bridge flag, `--facefix rel|abs`.

## Results (recorded 5090 runs replayed with exact `source_seq` pairing; `cloud/client/mouth_sync.py`)
| Clip | JoyAI alone | lpfast | Mode |
|---|---|---|---|
| G-Star, talking close-up (`G_gstar`, landscape) | lip-gap corr **0.03**, motion range 0.62 | **0.79–0.81**, range 1.03 (lag 0) | `rel`, gain 2 |
| Human ref, same driver (`C_mouthprompt`) | 0.25, range 0.30 | **0.98**, range 0.91 (lag 0) | `abs` |
| Human ref, gestures/glasses/hand at face (`human`, DTW-paired) | | hands intact, effect fades on large head tilts | `abs` |
| G-Star boxer clip (profile, fast motion, face leaving frame) | JoyAI draws no face most of the time | passes JoyAI through untouched there (no artifacts) | `rel` |

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
# live (G-Star): add --facefix rel to the usual bridge command, run with the facefix venv
cloud/facefix/.venv/bin/python cloud/client/live_bridge.py --url ws://127.0.0.1:18080/ws --input /dev/video0 \
    --output ffplay --portrait --out-size 720x1260 --ref assets/char_front.png \
    --prompt "$(cat cloud/pipelines/joyai/prompts/gstar.txt)" --facefix rel        # human refs: --facefix abs
# offline replay of a recorded run (input as sent + JoyAI output [+ stats JSON with source_seq, else DTW])
cloud/facefix/.venv/bin/python cloud/lpfast/run_offline.py --out OUT.mp4 --inp IN.mp4 --stats OUT.json \
    --dst cloud/lpfast/experiments/NAME --mode rel --gain 2
```
Disk: TensorRT 11 libs ~6.3 GB in the venv, ONNX ~0.7 GB, engines ~0.6 GB.

## Open / next
- Eyes follow only partly on G-Star (his lids are drawn half-shut); `--eye-gain` in `run_offline.py` is untuned.
- A hand directly over the mouth isn't in any recording (hands at the face edge are fine); no hand gate yet.
- Not yet run against the real 5090 server (validated through the bridge with `cloud/facefix/mock_server.py`
  replaying a recorded 5090 run at its chunk rhythm).
