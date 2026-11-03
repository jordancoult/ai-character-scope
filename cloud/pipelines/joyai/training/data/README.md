# Paired dataset builder: driver video → Kling G-Star video

This tool builds training pairs for the JoyAI lip-sync LoRA (`../PLAN.md`). Each pair is a TikTok of a driver person
lip-syncing (the **input**) and the same performance re-rendered as G-Star by Kling motion control (the **output**).
The tool downloads both videos, aligns them in time by their audio, normalizes them to one resolution and frame
rate, runs quality gates, and cuts frame-accurate paired clips. It also writes review media and converts the clips
to the JoyAI trainer's TAR shards.

Everything runs locally on the CPU. You don't need a GPU or a rented machine.

## Setup (once)
```bash
bin/uv pip install --python cloud/.venv/bin/python scipy requests mediapipe opencv-python-headless numpy
```
You also need `ffmpeg`/`ffprobe` on the PATH. MediaPipe models go in `cloud/client/models/`
(`face_landmarker.task`, plus `pose_landmarker_lite.task`, which downloads automatically on first use).

For TikTok page URLs, put the Zyla API key in `.secrets/zyla_api_key` (gitignored; the key is never printed).
`ZYLA_API_KEY` in the environment works too.

## CSV format
```csv
pair_id,input_url,output_url,notes,driver
mafe01,https://www.tiktok.com/@mafecardona.oficial/video/7681788876624629013,https://www.tiktok.com/@gstar/video/123...,first try,mafe
local01,~/Downloads/driver.mp4,https://cdn.example.com/kling_out.mp4,,jordan
```
- The header is required. `notes` and `driver` are optional. Blank lines and lines starting with `#` are ignored.
- `pair_id`: unique, and used as a folder name (characters other than `[A-Za-z0-9._-]` become `_`).
- `input_url` / `output_url` can each be one of three things:
  - **a TikTok page URL** (`tiktok.com/...`, `vm.tiktok.com/...`): downloaded unwatermarked through Zyla;
  - **a direct media URL**: any other http(s) URL, downloaded as is;
  - **a local path**: copied into the cache.
- `driver`: groups pairs by person for the train/val split (it defaults to `pair_id`).

### Zyla endpoint
The default is the Zyla API Hub "TikTok Media Grabber API" (listing 5312, which advertises "no watermark"):
```
GET https://zylalabs.com/api/5312/tiktok+media+grabber+api/14542/get+video?url=<tiktok url>
Authorization: Bearer <key>
```
The docs show a tikwm-style response: `{"code":0,"data":{"play","wmplay","hdplay","music",...}}`.
- The tool picks `hdplay`, then `play`. It never picks `wmplay`, which carries the watermark.
- Relative media paths get `https://www.tikwm.com` prepended.
- Media links expire, so the file downloads right after the API call.

To use a different Zyla TikTok API, pass `--zyla-endpoint URL` or set `ZYLA_ENDPOINT`. The parser also accepts
`nowm`, `no_watermark`, `video_url`, `download_url` and similar field names.

**Not verified against the live API yet (no key).** Check the first real response: `raw/<pair_id>/input.json`
records which field was used.

## Run
```bash
cloud/.venv/bin/python cloud/pipelines/joyai/training/data/build_pairs.py pairs.csv
#   --out DIR (default data/processed)  --raw DIR (default data/raw)  --only id1 id2
#   --width 480 --height 840 --fps 24 --crop cover|pad  --match-framing
#   --frames 97 --stride 97  --include-warn  --force [fetch align vision clips review]
cloud/.venv/bin/python cloud/pipelines/joyai/training/data/to_joyai_shards.py --mouth-mask \
    [--val-drivers mafe | --val-frac 0.1] [--path-prefix /workspace/rv2v]
```
**Idempotent and resumable.**
- Downloads are cached in `raw/<pair_id>/{input,output}.mp4`, with a `.json` sidecar holding the source and the
  probe.
- Each stage stores its result in `processed/pairs/<pair_id>/analysis.json` under a key built from its parameters.
- A rerun redoes only what is missing or changed. Re-running a fully processed CSV takes under a second.
- A failing pair (download error, missing file, no usable signal) is logged with `status=error`, and the run
  continues.

## Pipeline per pair
1. **Fetch** both videos (see the CSV section).
2. **Align in time.** The convention is `offset_s = t_input - t_output`.
   - Both files' audio is decoded to mono 16 kHz.
   - The tool first runs a raw-waveform normalized FFT cross-correlation over the whole clip. This is decisive when
     both files carry the same audio master; re-encoding doesn't matter. Method `audio_wave`.
   - If that isn't convincing, it cross-correlates onset envelopes (log-spectral flux at 100 Hz) and refines the
     result on the waveform within ±30 ms. Method `audio_envelope`.
   - **Confidence** = the correlation peak divided by the best competing peak more than 0.15 s away. Repetitive
     music scores about 1.
   - Without audio, the tool aligns on a frame-difference motion envelope (method `motion`, sub-frame refinement).
     When audio exists, the motion result serves as a cross-check, and a disagreement over 2 frames gives a warning.
   - Only the overlapping span is kept, minus leading or trailing black frames in either file.
   - Different input frame rates are handled by resampling both files to `--fps` CFR.
3. **Normalize** both videos to `--width`×`--height` (default portrait 480×840) at `--fps` (default **24**, the rate
   the JoyAI trainer reads), with the same crop policy for both:
   - `cover`: scale to fill, then crop the centre;
   - `pad`: fit inside the frame and add black bars.

   Both files get exactly the same frame count (H.264 CRF 16, AAC kept for review).
   **Spatial check.** The tool records the aspect ratios and the median face-box ratio between output and input on
   aligned frames (scale and centre shift).
   - With `--match-framing`, if the output is zoomed in (scale > 1.08), the input is cropped to the output's framing.
     This is off by default.
   - The scale is estimated from face-landmark height. That works for humans, but G-Star's cone head may bias it,
     so check `review.jpg` before trusting the crop.
4. **Gates** (thresholds are CLI flags):

   | gate | measure | default | on failure |
   |---|---|---|---|
   | `duration` | overlap length | ≥ 3 s | fail |
   | `offset_conf` | alignment peak ratio | ≥ 1.3 | fail |
   | `align_crosscheck` | audio vs motion offset | ≤ 2 frames | warn |
   | `person_in` | input frames with a face or pose (MediaPipe) | ≥ 0.8 | fail |
   | `gstar_out` | output frames with a detected face (works on G-Star about 93% of the time) | ≥ 0.5 | warn |
   | `mouth_sync` | lip-gap Pearson correlation, input vs output (as in `cloud/client/mouth_sync.py`), best of lag 0 and ±8 frames | ≥ 0.3 | fail |
   | `framing` | aspect mismatch, or face scale outside 0.8–1.25 | | warn |

   - Small faces in full-body shots are measured on a pose-based head crop upscaled to 256 px.
   - For the output, the aligned input's head region is used as a fallback.
   - `lipgap_best_lag` (in frames, positive = output late) shows any residual visual offset.
   - Status is `pass`, `warn`, `fail` or `error`. Clips are cut only for `pass` (add `--include-warn` to also cut
     `warn` pairs).
5. **Clips**: `--frames` frames each (default 97 = 4.0 s at 24 fps; JoyAI buckets are 8k+1: 49, 57, …, 113),
   starting every `--stride` frames.
6. **Review** for each pair:
   - `review.mp4`: aligned input | output side by side with a label. H.264 High@4.1, ≤ 1920 wide, `+faststart`,
     so it previews anywhere.
   - `review.jpg`: a contact sheet of 8 aligned pairs.

## Outputs (`processed/`, gitignored)
```
processed/
  manifest.jsonl          one line per clip: pair_id, clip_id, input, output (paths relative to processed/), fps,
                          frames, width, height, start_frame, input_time_s, output_time_s, offset_s, align_method,
                          confidence, sync (pair lip-gap corr), clip_sync, status, notes, driver, input_url,
                          output_url, caption (placeholder ""), instruction (from ../../prompts/gstar.txt)
  summary.csv / summary.md   all pairs and all gates
  build.log
  pairs/<pair_id>/analysis.json   all measurements (alignment candidates, gates, lip-gap series)
  pairs/<pair_id>/norm_input.mp4, norm_output.mp4   aligned, normalized full overlap
  pairs/<pair_id>/clips/input_0000.mp4, output_0000.mp4, ..., clips.jsonl
  pairs/<pair_id>/review.mp4, review.jpg
  shards/rv2v_{train,val}_000.tar, rv2v_{train,val}.json, mouth_mask_{train,val}.npz   (to_joyai_shards.py)
```
### Shards
`to_joyai_shards.py` writes the format in PLAN.md §3.
- Each sample has `<key>.0.mp4` (driver), `<key>.1.mp4` (Kling target), `<key>.0.png` (`assets/char_front.png`)
  and `<key>.json` = `{"caption": <live prompt>, "metadata": {"ref_image": ...}}`. Keys contain no dots.
- There are about 25 samples per shard. The manifest is `{"resolved_files": [...], "num_samples": N}`; use
  `--path-prefix` to write the paths as they will be on the GPU machine.
- The split holds out whole drivers.
- `--mouth-mask` adds `key -> uint8 [T_latent, H/24, W/24]`: the lips bounding box on the target (falling back to
  the source), per 8-frame latent group, dilated by one cell.

## Tests
```bash
cloud/.venv/bin/python cloud/pipelines/joyai/training/data/tests/test_pipeline.py SRC.mp4 [--other UNRELATED.mp4]
```
The test needs no network or API key. It builds simulated pairs from one real clip with audio:
- an output shifted by +1.06 s, zoomed 1.25×, at 25 fps, with re-encoded audio;
- an output with no audio at 24 fps;
- the same pair served over a local HTTP server;
- an unrelated clip;
- a missing file.

It then checks:
- alignment within one frame and the gate outcomes;
- `--match-framing`;
- clip frame counts, resolution and fps;
- that the review media is preview-safe;
- the shard layout and mouth masks;
- the Zyla response parsing (never `wmplay`);
- that a second run is a no-op.

Last run, with the source a 10 s 1080×1920 TikTok and AAC audio:
- **sim_shift**: offset error 0 ms (`audio_wave`), lip-gap corr 0.95;
- **sim_noaudio**: error 27 ms (`motion`), lip-gap corr 0.93;
- **unrelated output**: fail;
- **missing file**: error.

The first run takes about 75 s on CPU; the resumed run takes 0 s.
