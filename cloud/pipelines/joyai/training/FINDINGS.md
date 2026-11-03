# G-Star lip-sync LoRA, first attempt (2026-11-02/03): FINDINGS

**Result: no lip-sync gain.** The best clean adapters (r2 step 25, r3 step 50) change G-Star's *expression*: the
eyes open and follow the driver, and the smile gets livelier. But the mouth still doesn't open and close with the
driver's speech (mouth-sync correlation unchanged, within noise). Longer training breaks the face texture. **Owner's verdict
from the eval videos:** mouth not much better, but **character detail, skin quality and the hat clearly improved** at
25 and 50 steps (best), with possibly some added blur in fast movement. Server default stays `1101_face++`; how to
try the adapters live (runtime LoRA, not a merge): `cloud/docs/gstar-lora-finetune.md`.

**Merging loses the effect:** the LoRA delta is ~3e-4 of the weight RMS. Merged into bf16 only ~30% of it survives
(cosine 0.3 to the intended change, ~3× noise); after the 5090's FP8 weight quantization it is noise (cosine 0.07).
Apply it at runtime instead.

Artifacts (private HF): dataset `JordanCo/gstar-lipsync-pairs`; adapters, logs, previews and eval videos
`JordanCo/joyai-gstar-lora` (`adapters/r2_step25.pt`, `adapters/r3_step50.pt`, `adapters/r3_step100.pt`,
`eval/final/<input>/side_by_side.mp4` = input | base | r2 s25 | r3 s50 | r3 s100).

## Data
16 Kling motion-control pairs from the owner's sheet, aligned by audio. 14 kept (p01: lip corr 0.21; p08: wrong
alignment), with lag fixes of ±1–2 frames on 6 pairs. 81 clips of 97 frames (480×840, 24 fps). Validation driver
rilango_oficial1 (p15, 5 clips). Driver ↔ Kling lip-gap correlation per pair 0.45–0.91: the targets really are
lip-synced.

**Kling targets aren't pixel-aligned with the drivers.** Kling replaces the background, zooms in (face 1.1–1.9×) and
re-frames the body. This decided everything below.

## Runs (1× H200 NVL, Vast, $3.76/h; ~21 s per 97-frame sample without anchor, ~30 s with)
| Run | Target / loss | LR, exits | Outcome |
|---|---|---|---|
| r1 | raw Kling clips (time-aligned only), full-frame MSE + 4× mouth | 1e-4, exit 0/1 | **Collapsed to brown mush by step 50** (val MSE fell 1.47→0.88: the blur is the MSE optimum when background/framing are unpredictable) |
| r2 | face-aligned targets (per-frame similarity warp of Kling's face onto the driver's), loss on face + 4× mouth only, outside the face anchored to the frozen base output | 5e-5, exit 0/1 | step 25 clean; **face texture breaks down (hatched grid) from step 50** although val face/mouth MSE kept falling (1.19→0.54) |
| r3 | as r2 + anchor inside the face too (0.5), train only the final-step exit | 2e-5, exit 1 | step 50 clean; step 100 clean in training previews but **broken in streaming inference** (hatching, colour blow-out) |

## Final eval (streaming pipeline with KV cache, bf16, 193 frames, same seed per variant)
Lip-gap correlation with the driver (1 = perfect sync; mouth_sync_eval.py):

| input | base | r2 s25 | r3 s50 | r3 s100 |
|---|---|---|---|---|
| p15 held-out driver | −0.01 | −0.01 | −0.12 | 0.06 |
| talking head (hdtf) | 0.26 | 0.10 | −0.02 | −0.15 |
| webcam clip | −0.03 | −0.31 | −0.33 | −0.04 |

- The metric's noise is about ±0.15: base on p15 scored −0.14 in an earlier run with the same seed. None of the changes is a
  real gain. Mouth *motion range* rises (e.g. webcam 0.65 → 1.1×), so the mouth moves more, but not in sync.
- By eye (talking head, `eval/final/test_face_hdtf/side_by_side.mp4`): r2 s25 and r3 s50 open G-Star's eyes with the
  driver's and give wider smiles on her laughs; the mouth doesn't open and close with speech. r3 s100 is broken.

## Why it didn't work (best understanding)
1. **Per-pixel regression to an uncertain target.** Kling's G-Star face is drawn differently from JoyAI's (other
   proportions, teeth, lips). Even face-aligned, the MSE-optimal output is an average of plausible faces, which
   decodes to the hatched texture once the LoRA moves far enough. Validation MSE can't see this (it keeps falling while
   the picture breaks), so **MSE is the wrong objective for a 2-step distilled generator.**
2. **Tiny mouth in latent space.** At 480×840 the latent cell is 24 px. A webcam-framed mouth covers ~2×3 cells and
   G-Star's grin ~3×6, so lip opening is a sub-cell change the latent loss barely sees.
3. **Little data per behaviour.** 76 training clips from 13 pairs; the signal (when the mouth opens) is a small part of
   each clip.

## What would likely work better (next steps, if pursued)
- **A perceptual/structural mouth signal instead of pixel MSE:** decode the mouth crop through the VAE and use a
  lip-landmark or lip-reading feature loss (e.g. a differentiable landmark net on the decoded crop), or a
  distribution-matching objective (DMD-style with a teacher fine-tuned on the pairs). This keeps the output on the
  model's manifold and only asks "is the mouth open the same amount".
- **Closer framing at train and test time** (face larger in the frame) so the mouth spans more latent cells.
- **Stop by visual check, not val MSE:** decode previews every 10–25 steps and stop at the first texture artefact.
- More pairs where the driver talks (not only music lip-sync) and has the face large in frame.

## Cost
H200 NVL 13.5 h × $3.76/h ≈ **$51**, plus ~$1 downloads and a few Zyla calls. About 8 h of that was the box sitting
idle while a question to the user was pending (r1 had been stopped). Lesson recorded: never leave a paid GPU idle.

## Reproduce
`README.md` (setup, smoke, train, eval, merge). r3 command:
`train_lora.py --data <face-aligned clips> --steps 200 --accum 2 --lr 2e-5 --warmup 10 --p-exit-last 1.0 --w-frame 0
--w-face 1 --w-mouth 4 --w-anchor 1 --w-anchor-face 0.5`; the face-aligned clips come from
`data/face_align.py` (without `--raw`) on the dataset's `pairs/*/norm_*.mp4`.
