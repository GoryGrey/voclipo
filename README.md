# VOCLIPO

**Turn your voice into content. Audio in. Clips out.**

Upload episode audio and cover art, get back a captioned video with an
animated background, a live waveform, a clip scorecard, and shareable quote
cards. Free while in beta. No accounts, no cloud, everything runs locally.

## What it makes

A 1080x1920 (vertical) or 1080x1080 (square) MP4 with:

* **Karaoke word captions** timed to the actual speech (faster-whisper word
  timestamps, not estimates). If you upload the script, captions use your
  exact wording; otherwise they use the transcription. Five caption styles
  (classic, neon, ember, ghost, big), and a standalone `.srt` file is also
  generated for upload to YouTube/Facebook/etc.
* **Animated background**: nebula, aurora, starfield, bokeh, matrix,
  synthwave grid, contour lines, dunes, or your own custom image.
* **Waveform**: neon line, neon glow, spectrum bars, spectrum curve,
  waterfall, mirror, tri-band, radial, or particles, each in six color
  themes (volt, ultraviolet, lime, sunset, mono, ice).
* A 2-second branded intro title card (optional, on by default).
* **Audio prep**: voice cleanup (highpass + loudness normalize), Dead-Air
  Killer (trims silence and tightens long pauses, captions stay in sync),
  and Clean Mode (auto-beeps profanity).
* **Clip scorecard**: honest heuristic signals about the episode's clip
  potential (hook speed, questions, quotable moments). Not a virality
  prediction.
* **Quote cards**: 1080x1080 stills of the top-scoring quotes, ready to
  post.
* **Series templates**: one-click starting points (Hot Take, Storytime,
  Interview, Hype, Clean Comedy), plus your own saved presets.

## Run it

```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5057
```

Needs `ffmpeg` on PATH and Python 3.10+.

## Beta setup

The beta is invite-only while capacity is limited:

1. Put one code per line in `beta_codes.txt` (already gitignored).
   Delete a line to revoke a code.
2. Usage counts are tracked in `beta_usage.json`.
3. Renders run **one at a time** through a FIFO queue, so a small box
   never melts. The job page shows queue position.
4. Audio over 20 minutes is rejected (`MAX_AUDIO_SECONDS` in app.py).
5. Finished/failed jobs older than 24 hours are deleted automatically.

No `beta_codes.txt` (or an empty one) means the gate is open, for local
dev.

## Monetization slots

Ad/affiliate HTML goes in the `ADS` dict in `app.py`:

* `ADS['index_banner']` renders below the upload form
* `ADS['job_banner']` renders on the finished job page

Set `ADS['enabled'] = True` once the code is pasted in. The slots are
marked with `AD SLOT` comments in the templates.

## Notes

* Long videos render in ~30s segments, never cutting mid-caption, then
  concatenate. Each segment encodes in under ~2 minutes on modest
  hardware.
* `ffmpeg` `showwaves` uses `draw=full` so quiet speech still renders a
  visible waveform. NumPy overlays use true-alpha `qtrle` MOVs.
* Everything is local and deterministic. No external models, no uploads
  to third-party services.
