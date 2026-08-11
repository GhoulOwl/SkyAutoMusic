# SkyAutoMusic

SkyAutoMusic converts local audio or MIDI files into 15-key Sky piano JSON scores, with playlist playback, preview, batch save and NetEase Cloud Music download support.

## V2 audio arrangement engine

Audio transcription is an arrangement pipeline, not a merger of detected instrument tracks:

1. Decode audio and run the bundled Demucs checkpoint as an internal two-stem separator (`vocals.wav` plus `accompaniment.wav`).
2. Estimate musical tempo, meter and a bar-smoothed sixteenth-note grid. Half/double tempo candidates are evaluated so an 80 BPM song is not exported as 160 BPM.
3. Extract a stable vocal melody from pYIN and Basic Pitch. For weak vocal material, fall back to a prominent instrumental line, then to chord-only accompaniment.
4. Infer global key, chords and bar-aligned sections.
5. Arrange melody, bass and harmony into the 15 natural-note keys. `auto` uses lighter verses and denser choruses; `simple`, `standard` and `full` are also available.

The supplied `htdemucs_6s` checkpoint is retained solely as a separation black box. The application does not expose six tracks, track switches, track fusion, or local LLM/AI score review.

### Controls

- Arrangement preset: `auto`, `simple`, `standard`, `full`
- Source key: auto or manual global key
- Melody octave: auto, -1, 0, +1
- Maximum polyphony: 2–5
- Optional BPM and meter correction (`4/4`, `3/4`, `6/8`)

Changing preset, key, octave or polyphony re-arranges the in-memory analysis; it does not run separation or pitch detection again. BPM/meter changes rerun audio analysis.

## Development

Install Python 3.10 dependencies:

```powershell
pip install -r requirements-lock.txt
python -B -m unittest discover -s tests -v
```

Fetch the bundled separation model before a release build:

```powershell
python scripts/fetch_separation_model.py
python scripts/smoke_separation.py
```

Run a private golden set with a JSON manifest containing `audio`, optional `id`, `key`, `bpm`, `meter`, and `preset` fields:

```powershell
python scripts/benchmark_transcription.py golden-manifest.json v2-report.json
```

## Packaging

The GitHub Actions build packages Demucs, Basic Pitch, ONNX Runtime and the checked separation checkpoint for offline use. It does not download or bundle an LLM runtime or extra AI model.
