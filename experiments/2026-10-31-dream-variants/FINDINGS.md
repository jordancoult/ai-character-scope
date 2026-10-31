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
