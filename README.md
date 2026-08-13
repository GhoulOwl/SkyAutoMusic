# SkyAutoMusic

SkyAutoMusic converts local audio or MIDI files into 15-key Sky piano JSON scores, with playlist playback, preview, batch save and NetEase Cloud Music download support.

## Audio arrangement engines

Audio transcription is an arrangement pipeline, not a merger of detected instrument tracks:

1. Decode audio and run the bundled Demucs checkpoint as an internal two-stem separator (`vocals.wav` plus `accompaniment.wav`).
2. Estimate musical tempo, meter and a bar-smoothed sixteenth-note grid. Half/double tempo candidates are evaluated so an 80 BPM song is not exported as 160 BPM.
3. Extract a stable vocal melody from pYIN and Basic Pitch. For weak vocal material, fall back to a prominent instrumental line, then to chord-only accompaniment.
4. Infer global key, chords and bar-aligned sections.
5. Arrange melody, bass and harmony into the 15 natural-note keys. `auto` uses lighter verses and denser choruses; `simple`, `standard` and `full` are also available.

The supplied `htdemucs_6s` checkpoint is retained solely as a separation black box. The application does not expose six tracks, track switches, track fusion, or local LLM/AI score review.

### V3 high-quality mode

`高质量（MuScriptor）` is an optional whole-song transcription path. It emits
instrument-aware notes first, uses Beat This! beats/downbeats for timing, then
performs melody-first 15-key reduction. This avoids V2's source-separation and
chord-template chain when melody or rhythm fidelity matters.

- Windows CPU defaults to MuScriptor Small; CUDA and Apple Silicon default to Medium.
- On first use, accept MuScriptor's CC BY-NC model terms at Hugging Face and use
  a temporary Hugging Face token in the in-app setup dialog. Tokens are never
  written to config files or exported scores; model weights stay in the normal
  Hugging Face cache and are not bundled in releases.
- Each high-quality job requires confirming that you have the necessary rights
  to the input music and resulting score. The model is non-commercial and may
  not be appropriate for third-party online music without those rights.
- Drag across the V3 timeline to select bars, run `精修所选小节`, audition the
  candidate, then accept or discard it. Only the selected bars are replaced.

### Controls

- Arrangement preset: `auto`, `simple`, `standard`, `full`
- Source key: auto or manual global key
- Melody octave: auto, -1, 0, +1
- Maximum polyphony: 2–5
- Optional BPM and meter correction (`4/4`, `3/4`, `6/8`)

Changing preset, key, octave or polyphony re-arranges the in-memory analysis; it does not run separation or pitch detection again. BPM/meter changes rerun audio analysis.

## Development

Install Python 3.10 dependencies on Windows:

```powershell
pip install -r requirements-lock.txt
python -B -m unittest discover -s tests -v
```

### macOS (Apple Silicon)

The macOS source mode supports GUI, local/MIDI/NetEase transcription and
local score preview. Game-window control, injected keyboard playback and the
Windows click-through overlay are intentionally hidden on macOS.

Use an arm64 Python 3.10 distribution that includes Tk 8.6 or newer. The
macOS system Tcl/Tk 8.5 is too old for the current `ttk` UI and can leave most
widgets blank; the Python 3.12 environment supplied by some package managers
may also omit `_tkinter`.

```bash
python3.10 -c "import platform, tkinter; assert platform.machine() == 'arm64'; assert tkinter.Tcl().eval('info patchlevel').startswith('8.6'); print(tkinter.TkVersion)"
python3.10 -m venv .venv-macos
source .venv-macos/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-lock.txt
python scripts/fetch_separation_model.py
python -B -m unittest discover -s tests -v
python play_music_gui.py
```

macOS uses the system Keychain for NetEase cookies through `keyring`; cookie
contents are not written to `netease_auth.json`. The first online use may ask
for Keychain access. The supported target is macOS 13 or newer on Apple
Silicon.

Fetch the bundled separation model before a release build:

```powershell
python scripts/fetch_separation_model.py
python scripts/smoke_separation.py
```

Run a private golden set with a JSON manifest. Each record accepts `audio`, a
matching hand-authored Sky JSON in `reference`, optional `id`, `split`
(`tune` / `validation`), `key`, `bpm`, `meter`, `preset`, `engine`, and
`quality_model`. V3 records must set `rights_confirmed: true`. UTF-8 and
legacy UTF-16 Sky JSON are both supported. A hand score may cover only part
of a song: metrics automatically ignore generated notes outside its annotated
intervals. Use `reference_offset_ms` when that excerpt starts later in the
source, or `reference_windows_ms` for explicit source-time windows.

Keep the audio and hand score in ignored `golden_samples/`; use 10–20 songs
and reserve at least three as `validation`. The report contains 60 ms
key+onset F1, melody F1, onset-only F1, onset MAE, density, polyphony, elapsed
time and peak Python allocation. Run V2 and V3 manifests separately, then
compare validation medians before changing the app default.

```powershell
python scripts/benchmark_transcription.py golden-manifest.json v2-report.json
```

## Packaging

The GitHub Actions build packages the V2 runtime and the optional V3 code, but
never downloads or bundles MuScriptor weights. High-quality models are fetched
only after the user accepts their license.
