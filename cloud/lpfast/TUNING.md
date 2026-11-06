# lpfast tuning sheet

Applies to `live_test.py` and the live bridge (`--facefix rel|abs` / `--facefix-preset`). Live commands are typed in
the terminal + Enter and take effect from the next frame. `/save NAME` stores everything marked **saved** in
`presets.json`; `/preset NAME` loads it. Defaults below are the current code defaults (2026-11-05).

## Expression gains (`/gain <region> <value>`, `--gains`, saved)
How much of your motion G-Star gets, per face region. 1 = your motion as is, 0.5 = half, 2 = double, 0 = off
(the character's own expression), negative = mirrored. `/gain all 0.8` scales every region at once.

| Region | What it moves | `gstar` preset | `girl` preset |
|---|---|---|---|
| `lip` | mouth opening, lip shape | 2.65 | 2.0 |
| `eye` | eyelids (blinks, wide eyes) | 2.5 | 1.3 |
| `brow` | eyebrows | 1.5 | 1.3 |
| `brow2` | extra keypoint near brows / upper lids | 0 | 1.0 |
| `rest` | cheeks, jaw | 0 | 1.0 |

## Mode and pose (`--mode`, `--pose`, saved)
| Parameter | Values | What it does |
|---|---|---|
| mode | `rel` (G-Star) / `abs` (human refs) | `rel`: the character's resting face + your motion (keeps his grin). `abs`: blends towards your actual expression shape (gain 1 = yours exactly) |
| pose | on / off | also turn and scale his head with yours (still images; off for JoyAI output, which already moves the head) |

## Motion limits (`/limit <region> <max | knee:max | off>`, `--limits`, saved)
Soft caps so extreme expressions don't tear the face: motion passes unchanged up to the knee, then is compressed
towards max and never exceeds it. A single value sets max (knee = 2/3 of it). Calibrated on G-Star.

| Limit | Default knee:max | Prevents |
|---|---|---|
| `lip` | 0.14:0.21 | torn / "double mouth" on wide opening (breaks from 0.24) |
| `lipclose` | 0.01:0.025 | lips passing through each other when you press them (merge from 0.04) |
| `eye` | 0.07:0.10 | squashed lids (from 0.12) |
| `brow` | 0.10:0.14 | forehead / eye distortion (from 0.17) |
| `brow2` | 0.05:0.08 | (guess, uncalibrated) |
| `rest` | 0.06:0.10 | (guess, uncalibrated) |

## Smoothing (`/euro <min_cutoff> <beta> | off`, `--euro`, saved)
One-Euro filter on your expression: smooths hard when you're still (webcam/low-light jitter, amplified by high gains),
little when you move fast.

| Parameter | Default | Lower | Higher |
|---|---|---|---|
| min_cutoff (Hz) | 1.0 | calmer when still, a bit laggier | more detail, more jitter |
| beta | 5 | more lag on fast moves | snappier on fast moves |

## Drop-out gates (`/gate ...`, saved)
The effect fades out completely (the plain JoyAI / source frame shows) when one of these trips.

| Gate | Command | Default | Meaning |
|---|---|---|---|
| your head yaw | `/gate yaw <start> <off>` | 25 40 (deg) | fades from 25 deg to off at 40 deg |
| your head pitch | `/gate pitch <start> <off>` | 18 30 (deg) | relative to your usual tilt |
| mouth covered | `/gate cover <full> <off>` | 0.55 0.35 | mouth visibility (normal >= 0.66, covered ~0): fades as it drops from 0.55 to 0.35 |
| turned away | `/gate score <full> <off>` | 0.65 0.5 | face confidence (frontal 0.76-0.90, profile < 0.5) |
| generated face yaw | `/gate src_yaw <start> <off>` | 30 45 (deg) | the character's own head (JoyAI output) |
| generated face pitch | `/gate src_pitch <start> <off>` | 20 35 (deg) | the character's own head |
| hold after a cover | `/gate hold <frames>` | 6 | clear frames before the effect returns |
| visibility checks | `/gate visibility on\|off` | on | the cover + confidence gates (off saves ~3 ms GPU per 2 frames) |

The result panel shows the live values: `w` (effect strength 0..1), `yaw`, `mouth` (visibility), `conf` (face
confidence), so you can see which gate is tripping.

## Other controls (live_test)
| Command / flag | Default | What it does |
|---|---|---|
| `/reset` | | re-take your neutral face, head reference and face detection (after moving seat, or if anything looks stuck) |
| `/gain` `/limit` `/gate` `/euro` `/preset` | | alone: show current values / list presets |
| `/save NAME` | | save mode, pose, gains, limits, gates, smoothing as preset NAME |
| `q` | | quit |
| `--source` | `assets/g-star-screenshot-no-hat.png` | image or video to animate |
| `--crop x0,y0,x1,y1` | none | crop the source (fractions) |
| `--driver auto\|VIDEO` | webcam | drive from a recording instead of the webcam |
| `--camera` | `/dev/video0` | webcam device |
| `--neutral-ema` | 0.01 | how fast your "neutral face" baseline re-adapts (0.01 ~ 4 s, 0 = keep the first frame) |
| `--base-ema` | 0.08 | how fast the character's resting expression follows a video source (irrelevant for images) |
| `--smooth` | 0 | old simple smoothing, only used if `--euro off` |
| `--no-visibility` | | don't run the cover / confidence gates |
| `--layout 1\|3` | 3 | result only / webcam - source - result |
| `--save out.mp4` | | record the window |
| `--fps` | 25 | output rate |

**Saved vs not saved.** `/save` stores mode, pose, gains, all limits, all gates (incl. hold and visibility) and the
smoothing (`euro`, including "off" — saved as `null`). Not saved (start-up flags only): `--neutral-ema`, `--base-ema`,
`--smooth`, `--source`/`--crop`/`--camera`/`--fps`/`--layout`. Loading a preset first resets limits and gains to the
defaults, then applies the preset, so a preset fully reproduces its settings.

Bridge: `--facefix-preset NAME` and live `/preset NAME` load the whole preset (limits, gates, smoothing) except
pose, which stays off (JoyAI output already moves the head). Bridge equivalents: `--facefix rel|abs`, `--facefix-preset NAME`, `--facefix-gains 'lip=..,eye=..'`; live `/gain`,
`/limit`, `/gate`, `/euro`, `/preset`.
