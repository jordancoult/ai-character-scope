# Lessons (things that cost time once)

- **Waits need watchdogs.** A setup step crashed with "Aborted (core dumped)" and an until-loop grepping for known
  error words sat 42 min. Every wait: a hard deadline + exit when the job ends (write `EXIT=$?` markers) + progress.
- **Network volumes are slow for many small files.** A Python env install crawled; torch.compile writing ~60k cache
  files took ~60 min; copying the cache back took >12 min. Fix: env on local disk; compile cache as **one tar**
  (8 s to write). Reading the 32 GB checkpoint cold from the volume once took ~15 min.
- **Build extensions with the venv's Python.** `pip wheel` picked up the image's system pip/torch 2.8 → the wheel
  crashed on import under torch 2.9.1 (`std::length_error vector::reserve`). Use `python -m pip` inside the venv.
- **JoyAI flow control:** the client must `ack` every received frame (and `ping` every second). Acking every 24
  frames made the server think the network was backed up and drop chunks (5 fps instead of 10).
- **Use H.264 on the wire.** Near-lossless MJPEG (~16 Mbit/s each way) through a transatlantic SSH tunnel caused
  drops and 5.8 s p95 latency; H.264 gave 25.5 fps and 0.78 s p95.
- **Playout needs an adaptive jitter buffer.** Output arrives in 8-frame bursts; a fixed-rate player let the queue
  creep (1.36 s); adaptive rate (gain 0.05/frame, target 6) gives 1.03 s with 0 drops.
- **`ssh host 'cmd &'` hangs** if the background job keeps the session's stdout: use
  `setsid nohup bash -c "...; echo EXIT=\$?" > log 2>&1 < /dev/null & disown`.
- **Never `pkill -f` / `pgrep -f`** (also inside `ssh '...'`: it matches the remote shell running your own command).
- **GPU stock moves within minutes.** Data-center-pinned storage (RunPod network volume) means waiting for that DC;
  provider-independent setup (HF artifacts) removes the pin.
- **Prompting JoyAI for a character:** describe it explicitly; a full-body ref beats an upper-body crop; the built-in
  prompt enhancer hurt identity.

## Vast self-stop watchdog needs a heartbeat source (2026-11-04)
`vast_watchdog.sh` stops the instance after IDLE_MIN without a touch of `/workspace/.heartbeat`, and `pod.sh` doesn't
touch it. A live session (only the tunnel open) therefore got the 5090 stopped after 45 min, and the stopped instance
couldn't restart: its GPU had been rented out meanwhile ("Required resources are currently unavailable"). Run
`vast_heartbeat.sh` next to the watchdog: it touches the heartbeat while any SSH session (our tunnel) is established.
