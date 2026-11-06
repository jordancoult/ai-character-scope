# lpfast tuning sheet

Applies to `live_test.py` and the live bridge (`--facefix rel|abs` / `--facefix-preset`). Live commands are typed in
the terminal + Enter and take effect from the next frame. `/save NAME` stores everything marked **saved** in
`presets.json`; `/preset NAME` loads it. Defaults below are the current code defaults (2026-11-05).

Ranges: **accepted** = what the code allows (anything outside is rejected with an error message and nothing
changes); **useful** = where it makes sense in practice. Limit values are in LivePortrait keypoint units, where the
whole face is about 1.0 wide.

## Expression gains (`/gain <region> <value>`, `--gains`, saved)
How strongly your face motion is copied onto the character, per face region. 1 = your motion as is, 0.5 = half,
2 = double, 0 = off (the character keeps his own expression there), negative = mirrored. `/gain all 0.8` multiplies
every region by 0.8. In `abs` mode a gain is a blend: 0 = the character's expression, 1 = yours exactly.

| Region | What it does | Accepted | Useful | `gstar` | `girl` |
|---|---|---|---|---|---|
| `lip` | copies your mouth: opening, closing, smile width, lip shape | any number | 0 to 3 (above 3 everything hits the lip limit) | 2.65 | 2.0 |
| `eye` | copies your eyelids: blinks, squints, wide eyes | any number | 0 to 3 | 2.5 | 1.3 |
| `brow` | copies your eyebrows raising / lowering | any number | -2 to 2 (negative = brows move the opposite way) | 1.5 | 1.3 |
| `brow2` | one extra keypoint between brows and upper lids; adds forehead / upper-lid movement | any number | 0 to 1.5 | 0 | 1.0 |
| `rest` | cheeks and jaw shape (not the mouth opening itself) | any number | 0 to 1.5 | 0 | 1.0 |

## Mode and pose (`--mode`, `--pose`, saved)
| Parameter | What it does | Accepted |
|---|---|---|
| mode | how your expression is applied. `rel`: the character's resting face + your *changes* from your neutral face (keeps his grin; for G-Star). `abs`: moves towards your actual expression shape (for human-looking references) | `rel` / `abs` |
| pose | also turns, tilts and scales the character's head with yours. For still images only; the bridge always keeps it off because JoyAI output already moves the head | `on` / `off` |

## Motion limits (`/limit <region> <max | knee:max | off>`, `--limits`, saved)
Soft caps so extreme expressions don't tear the face: motion passes unchanged up to the knee, then is compressed
towards max and never goes past it. A single value sets max (knee = 2/3 of it). `off` (or 0) = no limit.
Accepted for all: `0 <= knee < max`. "Breaks at" = where G-Star's image starts to tear without a limit.

| Limit | What it does | Default knee:max | Useful max | Breaks at |
|---|---|---|---|---|
| `lip` | caps how far the mouth can move overall (mainly how wide it opens); stops the torn / "double mouth" look | 0.14:0.21 | 0.12 to 0.24 | 0.24 |
| `lipclose` | caps how far the lips can press *past closed*; stops the lips sliding through each other when you press yours together. Smaller = lips stop sooner | 0.01:0.025 | 0.005 to 0.04 | 0.04 |
| `eye` | caps eyelid motion; stops squashed / smeared lids on hard blinks and squints | 0.07:0.10 | 0.05 to 0.12 | 0.12 |
| `brow` | caps eyebrow motion; stops forehead and eye distortion on big brow raises | 0.10:0.14 | 0.07 to 0.17 | 0.17 |
| `brow2` | caps the extra brow / upper-lid keypoint | 0.05:0.08 | 0.03 to 0.12 | not calibrated |
| `rest` | caps cheek / jaw shape changes | 0.06:0.10 | 0.03 to 0.12 | not calibrated |

**Finding your max with the meters.** In `live_test` the top of the result panel shows a live row for the mouth:
`lip` (mouth opening / lip shape), `close` (how far your lips press *past closed*) and `jaw` (the `rest` region:
cheeks / jaw). Each shows `now`, `pk` (the highest value in the last 3 s) and the current `max`, in exactly the units
of `/limit` (your motion x gain, before limiting), so: make the biggest mouth you want to allow, read `pk`, and set
that (or a little above) as the max. White = below the knee, yellow = being softened, red = at the max (clamped).
`jaw` only moves when the `rest` gain is above 0 (it's 0 in `gstar`). `/meters off` / `/meters on` hides / shows the
row (always starts on, not saved). Live_test only; the bridge has no overlay.

The knee is where squashing starts: about 2/3 of max is a good default. A knee close to max = a hard stop; a low
knee = softer, but big expressions look weaker.

## Smoothing (`/euro [upper|lower] <min_cutoff> <beta> | off`, `--euro`, saved)
A One-Euro filter on your expression before it's applied: it smooths hard while your face is still (removes webcam
and low-light jitter, which high gains amplify) and barely at all when you move fast, so lip sync stays quick.
It has two independent halves, split at nose level, so you can e.g. calm jittery eyes/brows a lot while keeping the
mouth snappy:

| Half | What it covers | Keypoints |
|---|---|---|
| `upper` | everything above the nose: eyebrows, eyelids, forehead, temples | 0, 1, 2, 8, 9, 10, 11-16 |
| `lower` | everything below the nose: mouth, lips, cheeks, jaw, chin | 3-7, 17-20 |

(Measured on G-Star, the girl reference, your webcam and a portrait clip: the same split on all of them. Two of
LivePortrait's "lip" keypoints, 12 and 14, actually sit on the eyes, so they are in `upper`.)

| Command | What it does |
|---|---|
| `/euro` | show both halves |
| `/euro 1 5` / `/euro off` | set / disable both halves |
| `/euro upper 0.5 3` | set only the upper half |
| `/euro lower off` | disable only the lower half (raw mouth motion) |
| `--euro 1,5` / `--euro upper=0.5:3,lower=1.5:8` / `--euro lower=off` | same at start-up (a half you leave out keeps the preset / default) |

Each half has the same two numbers:

| Parameter | What it does | Default | Accepted | Useful | Lower | Higher |
|---|---|---|---|---|---|---|
| min_cutoff (Hz) | how much smoothing when you're still | 1.0 | > 0 | 0.3 to 3 (mouth: 1 to 3, eyes/brows: 0.3 to 1.5) | calmer, slightly laggier | more detail, more jitter |
| beta | how quickly smoothing backs off when you move | 5 | >= 0 | 0 to 25 | more lag on fast moves (0 = constant smoothing) | snappier, more jitter while talking |

Presets save both halves (`"euro": {"upper": [1, 5], "lower": null}`, `null` = off). Older presets with a single
pair (`"euro": [5, 25]`) still load and apply it to both halves.

## Drop-out gates (`/gate ...`, saved)
These fade the effect out completely (the plain JoyAI / source frame shows) when your face can't be read well, so
you don't get a broken face. Each gate takes two numbers: where fading starts and where the effect is fully off.

| Gate | What it does | Command | Default | Accepted | Useful |
|---|---|---|---|---|---|
| your head yaw | fades out when you turn your head left / right too far | `/gate yaw <start> <off>` | 25 40 (deg) | start < off | start 15 to 40, off 30 to 60 |
| your head pitch | fades out when you look too far up / down, compared with your usual tilt (taken at start / `/reset`) | `/gate pitch <start> <off>` | 18 30 (deg) | start < off | start 10 to 30, off 20 to 45 |
| mouth covered | fades out when your mouth is hidden (hand, mic, cup). Uses mouth visibility: normal >= 0.66, covered ~0. A full value above ~0.65 can weaken the effect with your mouth uncovered | `/gate cover <full> <off>` | 0.55 0.35 | 0..1, full > off | full 0.4 to 0.65, off 0.1 to 0.5 |
| turned away | fades out when the face detector is unsure (face in profile or mostly hidden). Confidence: frontal 0.76-0.90, profile < 0.5. A full value above ~0.75 can weaken it when you face the camera | `/gate score <full> <off>` | 0.65 0.5 | 0..1, full > off | full 0.55 to 0.75, off 0.3 to 0.6 |
| generated face yaw | fades out when the *character* (JoyAI output) turns his head too far left / right | `/gate src_yaw <start> <off>` | 30 45 (deg) | start < off | start 20 to 45, off 35 to 60 |
| generated face pitch | same for the character looking too far up / down | `/gate src_pitch <start> <off>` | 20 35 (deg) | start < off | start 15 to 30, off 25 to 45 |
| hold after a cover | how many clear frames before the effect comes back after a cover, so it doesn't flicker | `/gate hold <frames>` | 6 | whole number >= 0 | 0 to 25 (25 = 1 s) |
| visibility checks | turns the mouth-covered and turned-away checks on / off (off saves ~3 ms GPU every 2 frames) | `/gate visibility on\|off` | on | on / off | |

The result panel shows the live values: `w` (effect strength 0..1), `yaw`, `mouth` (visibility), `conf` (face
confidence), so you can see which gate is tripping and pick thresholds from what you see.

## Other controls (live_test)
| Command / flag | What it does | Default | Accepted |
|---|---|---|---|
| `/reset` | re-takes your neutral face, head reference and face detection (after moving your seat, or if anything looks stuck) | | |
| `/gain` `/limit` `/gate` `/euro` `/preset` | typed alone: show the current values / list presets | | |
| `/save NAME` | saves mode, pose, gains, limits, gates and smoothing as preset NAME (overwrites it) | | any name without spaces |
| `/stats on\|off` | turns the periodic `[live] N frames, render ... ms, late ticks` line on / off (`/stats` alone shows the state). Always starts on, not saved in presets; the bridge has the same command for its `[bridge] played ...` line | on | on / off |
| `/meters on\|off` | shows / hides the lip / close / jaw meter row on the result (see Motion limits) | on | on / off |
| `q` | quits | | |
| `--source` | the image or video to animate | `assets/g-star-screenshot-no-hat.png` | image or video file |
| `--crop x0,y0,x1,y1` | crops the source first | none | fractions 0..1 |
| `--driver auto\|VIDEO` | drives from a recording instead of the webcam (with an image source it just plays through) | webcam | |
| `--camera` | which webcam to use | `/dev/video0` | device path |
| `--neutral-ema` | how fast your "neutral face" baseline re-adapts (0.01 ~ 4 s, 0.05 ~ 1 s, 0 = keep the first frame) | 0.01 | 0 to 1; useful 0 to 0.05 |
| `--base-ema` | how fast the character's resting expression follows a video source (no effect on images) | 0.08 | 0 to 1; useful 0.02 to 0.3 |
| `--smooth` | old simple smoothing (higher = calmer, laggier); only used with `--euro off` | 0 | 0 to 0.9 |
| `--no-visibility` | skips the mouth-covered / turned-away checks | | |
| `--layout 1\|3` | window shows the result only (1) or webcam - source - result (3) | 3 | 1 or 3 |
| `--save out.mp4` | records the window to a file | | file path |
| `--fps` | output frame rate | 25 | 1 to ~30 (the laptop manages ~31 at full quality) |

**Saved vs not saved.** `/save` stores mode, pose, gains, all limits, all gates (incl. hold and visibility) and the
smoothing (`euro`, including "off", saved as `null`). Not saved (start-up flags only): `--neutral-ema`, `--base-ema`,
`--smooth`, `--source`/`--crop`/`--camera`/`--fps`/`--layout`. Loading a preset first resets limits and gains to the
defaults, then applies the preset, so a preset fully reproduces its settings.

Bridge: `--facefix-preset NAME` and live `/preset NAME` load the whole preset (limits, gates, smoothing) except
pose, which stays off (JoyAI output already moves the head). Bridge equivalents: `--facefix rel|abs`,
`--facefix-preset NAME`, `--facefix-gains 'lip=..,eye=..'`; live `/gain`, `/limit`, `/gate`, `/euro`, `/preset`.

The HTML version is generated: after editing this file run `python3 cloud/lpfast/tuning_html.py`.
