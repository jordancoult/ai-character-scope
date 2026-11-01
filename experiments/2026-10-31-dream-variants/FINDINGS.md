# Dream-mode variants: findings (2026-10-31)

All on LongLive at 640x352 landscape, with stock test clips instead of the webcam: a boxer in a gym (body
motion) and a talking-head close-up (face). Each run took 50–80 s with cool-downs in between. No thermal
stops happened: the GPU peaked at 82°C and the CPU at 105°C in brief spikes. The side-by-side clips
(input | output) are in this folder. The new presets are in `saved_settings/claude_*.json`; open
them with **Load…** on the graph page.

## Three discoveries that change how the settings behave

1. **"Camera → VACE control" applies only when a stream starts.** Scope wires it at connect time.
   - When it's on, the camera only steers and the image starts from pure noise. On LongLive that
     gives confetti at every noise and step setting tested.
   - Your "many worlds" sessions were in the normal mode (the camera is the starting image), and
     that wire did nothing.
   - Recommendation: leave it off on LongLive. The page now says it applies on restart.
2. **LongLive kept a frozen pose after depth or camera-VACE was switched off.** StreamDiffusionV2
   already had the fix (clear stale VACE frames each chunk); LongLive didn't. Fixed now.
3. **The biggest lever is depth on LongLive** (newly ported). Depth carries your body's shape but not
   your appearance, so the noise can go to 1.0 (pure prompt world) and you stay in control.

## Winners

| Preset (Load…) | Settings | What you get | FPS |
|---|---|---|---|
| `claude_immersed_body_underwater_physics` | depth on, **body-only 0.25**, noise **1.0**, sink 3, 1 step, physics prompt | Your pose drives a character inside a fully generated world. Hair streams, bubbles burst off punches, waves crash around the body. Swap the prompt for "astronaut on the moon, moon dust kicked up by every movement" or "fire elemental … embers flying". | ~17 |
| `claude_realistic_face` | depth on, noise **0.9**, sink 3, 1 step | Most photoreal result, with mouth and expressions tracked closely and one consistent identity | ~17 |
| `claude_dream_world_with_body` | depth on, noise 1.0, sink 0 | A prompt world shaped by your depth, including the room layout, that keeps shifting like "many worlds" | ~17 |
| `claude_creature_body-only_face` | depth on, body-only **0.5**, noise 1.0, sink 3 | Face close-ups reinvented as a creature, with expressions still followed | ~17 |

## Webcam-style check (`w*.mp4` clips)

Two stock clips of a person at a desk, talking and gesturing at the camera: `assets/webcam_7643311.mp4`
and `assets/webcam_5941016.mp4`, both from Pexels.
- **Your "many worlds (interactive)" (noise 0.85, no depth)** mostly copies the room and suit with a
  watery tint at ~23.7 fps. With a full desk scene in view, 0.85 noise keeps the camera image dominant.
- **Immersed body** looks the most different: the person is underwater with floating hair, and hand
  gestures (hands to face, clasped hands) are tracked. The identity is reinvented, sometimes as a
  different gender, because only depth carries over.
- **Realistic face** is very photoreal, with gestures and face-touching followed well. But the room
  stays (noise 0.9 keeps the scene), so the prompt mostly changes the details.
- **Dream world with body** sits in the middle: an underwater set and a restyled person, with gestures followed.
- **Creature (body-only 0.5)** does little at desk framing, because the cutoff still keeps the whole upper body.
  Use it for close-ups only.
- Slow desk gestures show much less ghosting than the boxer's punches. Depth modes run at ~18 fps.

## Deep scene: you farther back, still doing your motions (`deep-scene/` clips)

These are new controls, all off by default, so the presets above behave as before.
- **Camera node → Place in scene** (size, left/right, up/down) shrinks your depth map into the frame.
  The surroundings become "unknown", so the model builds the world around a smaller you.
- **Depth node → Identity hint** (0–0.6): inside your outline, the latents are pulled toward your real
  camera colors and everywhere else stays pure noise. That keeps your clothes, glasses and hair
  while the world is invented. 0.3 works well.
- **Depth node → Auto body-only:** keeps the nearest fraction of the image (0.35 for a person at a desk)
  whatever the room depth is. It's reliable *without* placement; with placement, use the fixed cutoff.

What worked:
- **50% size + body-only cutoff 0.25 + identity hint 0.3 + a self-description in the prompt**
  (`r0_A_p50_desc_640`). This gives a consistent person (suit, pink shirt, glasses) floating
  deep in the underwater scene, with gestures followed, at ~18 fps. Without the self-description
  the face drifted to a different gender, because LongLive's reference image is weak. **The prompt is the
  main identity lever**: describe yourself in it.
- **At 768×432** (`--size 768 432`) the small subject is crisper and 40% size also works, but it runs
  at ~11 fps and uses 15 of the 16 GB of GPU memory. 832×480 runs out of memory.
- **Immersed + identity** (no placement, auto body-only 0.35 + hint 0.3 + description) gives the
  clearest identity: the bearded man in the checked shirt stays himself underwater.

What failed, and why:
- **Shrinking the camera image with noise below 1.0:** the gray border survives and becomes a wall
  with a TV-like window. Placement needs noise 1.0.
- **A mask that covers only the face and hands** (adaptive cutoff + placement, or 35% size) makes the model
  draw a giant close-up. The mask has to cover the whole upper body.
- **Pushing the placed depth into the far range** made close-ups worse, so it was reverted.
- Rooms with a desk or glass wall close to the camera can leave a box outline at cutoff 0.25. Raise
  the cutoff until it disappears.
- **Mouth detail at 50% size** is limited: the face is ~40 px tall at 640×352. Expressions read
  broadly, but lip-sync detail needs a bigger size (0.6+) or the 768 resolution.

## Dials and what they do

- **Noise** (now up to 1.0):
  - With depth on, 0.9 stays close to you (realism) and 1.0 gives a pure prompt world.
  - Without depth, 0.93 and above loses the body; for example, the boxer turns into fish.
- **Body-only depth** (Depth node):
  - 0 keeps the room's geometry, such as the gym ring.
  - 0.25–0.35 keeps the full body and the model invents the background.
  - 0.5 and above keeps only the nearest parts (head and hands), which is good for face close-ups.
- **Sink** 0 keeps the person and scene morphing ("many worlds"). Sink 3 keeps the same character
  for the whole run.
- **Keyframe memory** (Sink node, LongLive) copies a fresh frame into one sink slot every N
  seconds. The scene drifts gradually (city at night → a bar) instead of being pinned to the start
  (sink 3) or forgotten after ~3 s (sink 0). The effect is subtle, so it's a dial to explore live.
- **Steps:** 2 steps runs at ~12 fps and does **not** fix the ghosting on fast punches, which comes
  from fast motion inside each 12-frame chunk. 1 step is the better trade.
- **Prompts:** describing the physical interaction ("… swirling around every movement of the body")
  clearly makes the world react to motion.

## LoRAs (load time: `BRIDGE_ARGS="--lora path:scale"`, downloaded into `models/lora/`)

- `ultrawan/ultrawan_1k_comfy.safetensors:0.4` gives a polished 3D-game or animated look (strong
  style). It's a creative option, but not realism.
- `hstoric/HSTwan13LoRA2_000003750.safetensors:0.6` is slightly more photographic. The effect is subtle.
- `solace/` (PEFT format) wasn't tested.

## Not done / next ideas

- Prompt changes still wipe memory (`manage_cache: True`). LongLive's own recache would keep the
  scene through a prompt change; this is a one-line test.
- The ghosting on fast motion could improve with a smaller chunk, but LongLive is trained on 3-latent chunks.
- Real webcam check: all of the above used stock clips.
