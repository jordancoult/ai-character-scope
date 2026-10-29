# Start here

Two terminals: one runs the AI, one shows you the result. Keep the laptop plugged in, on the ASUS **Performance** profile.

## Terminal 1: start the AI character

**My settings (what looks best so far):** preset **20** (pose-tracked with depth, ~16 FPS), **fixed** noise, **no reference image**, this prompt:

```bash
cd ~/personal/ai-character-scope
SCOPE_FUSED_KV=0 PRESET=20 PROMPT="swedish lady with blonde hair, black thin frame glasses" NO_AUDIO=1 ./start_scope.sh --bridge
```

Fixed noise is the default, and leaving out `CHARACTER_REF` means no reference image (depth still works).
To use a reference image instead, add `CHARACTER_REF=assets/blonde-gamer-girl.png` before `./start_scope.sh`.

Wait for `[bridge] output ... fps` lines. The first ~30 s are slow while it warms up.

## Terminal 2: watch the output (mirrored)

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 -sync ext -f v4l2 -vf hflip /dev/video10
```

`-vf hflip` mirrors only this preview. OBS and the stream get the unmirrored feed.

## Controls page

Open http://127.0.0.1:8090 to use:

- **Quality vs FPS:** 30 fast, 24 clean, 20 and 15 pose-tracked with depth (~16 FPS).
- **Noise slider:** lower follows you more; higher gives more character.
- **Noise pattern:** fixed (default, less shimmer), random, or auto.
- **Reference image:** any `assets/*.png`, or **(none - prompt only)**.
- **Refresh character:** resets the memory when the image starts degrading.
- **Prompt:** edit, then Apply.

## Stop

Press `Ctrl+C` in terminal 1. Close the ffplay window (or press `q`).

If it says the port is in use, an old server is still running:

```bash
fuser -k -TERM 8000/tcp
```

## Optional settings (put them before `./start_scope.sh`)

| Setting | What it does |
|---|---|
| `PRESET=24` | Starting FPS preset (30, 24, 20 or 15) |
| `NOISE_MODE=random` | Starting noise pattern (fixed, random or auto) |
| `CAMERA=/dev/video0` | Which webcam |
| `MODEL=longlive` | Experimental LongLive model (sparkle artifacts, see README_QUICKSTART.md) |
| `SCOPE_FUSED_KV=0` | Turns off one speed patch (what you've been running with; a bit slower, same look) |
| `NO_AUDIO=1` | Skip creating the delayed virtual mic |

Reference images go in `assets/`. Match their framing to how you sit in front of the camera.

More detail: `README_QUICKSTART.md`.
