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

V8 reads Beat This! directly, so expressive tempo changes no longer fail a
whole-song fixed-tempo check. If beat tracking is unavailable, it preserves
MuScriptor's original note attacks instead of snapping them to a synthetic
120 BPM grid. Lead selection is phrase-aware: detected vocals lead while they
are active, short breaths remain silent, and a coherent BGM line can lead only
in a clear instrumental passage. `simple` stays melody-only; `standard` and
`auto` reuse supported bass, piano, guitar or string texture from the source
and keep accompaniment below one third of the melody. Inferred chords are a
last-resort option for `full` only.


- Windows CPU defaults to MuScriptor Small; CUDA and Apple Silicon default to Medium.
- When a Windows user explicitly selects a CUDA model, the setup dialog detects an NVIDIA GPU and can install the matching CUDA PyTorch runtime automatically. Restart the app once after that one-time runtime installation, then download the model.
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

### Playback fidelity and comparison

Game playback schedules key-down and key-up independently. Different keys can
overlap without delaying the next attack. The default press is 50 ms; repeated
physical keys release 5 ms before their next attack when the score allows it.
Pause, stop and seek clear pending releases, and an older playback session
cannot release keys belonging to a newly started session. Normal Windows input
batches chords through SendInput; the PyAutoGUI fallback omits its per-call
pause. The diagnostic tab shows the actual backend, failed sends, fallbacks and
lateness, and can export the current session. **A successful send acknowledges
the input API, not a sound produced by the game.**

The generation dialog offers `原曲`, `分离人声`, `纯旋律` and `完整编配`
comparison previews. Select bars first, or compare the bar containing the
playhead. Switching modes retains source time and pause state, including the
leading rest. Audio is prepared on a worker thread; vocal clips are separated
only when requested and cached locally for the dialog. A missing source file
disables source/vocal comparison while the rendered score remains available.

`稳定` retains the established lead selector. The optional `乐句优化` strategy
uses phrase-level dynamic programming and joint 15-key mapping. Its optional
`校验疑似人声片段` check ranks octave jumps, overlaps, dense attacks and
unexpected gaps; merged contexts are limited to 30 seconds each and 30% of a
song. Stem evidence can rank existing candidates or repair an octave only at
confidence >= 0.85 and union coverage >= 70%; it never creates a symbolic note.
Unresolved regions are marked for review. Separation failures retain the
model transcription, and cancellation propagates instead of becoming a
fallback. Manual octave choices also take precedence in the fast arranger.

The stable strategy remains the default until a fresh V8 comparison passes the
whole-set and validation gates **and** targeted vocal improvement is verified.
`TranscriptionOptions` exposes `fidelity_mode="stable"|"candidate"` and
`vocal_check="auto"|"off"`. Old drafts keep their saved notes on load;
re-arrangement explicitly applies the selected strategy. Raw symbolic notes
stay unchanged, and vocal evidence has its own version in drafts and caches.

Measure scheduling without injecting any real keyboard input:

```bash
python scripts/benchmark_playback.py --interval-ms 20 --count 100 --output playback-report.json
```

For a private candidate manifest, set each row's `fidelity_mode` to `candidate`
and `vocal_check` to `auto`. Capture the V8 baseline from commit
`a3e53ca623db5ed986a0784ce908173187e6d35b` first. After raw model caches exist:

```bash
python scripts/benchmark_transcription.py golden_samples/golden-manifest-fidelity-candidate.json golden_samples/fidelity-candidate-report.json --require-cached --summary-only
python scripts/verify_fidelity.py golden_samples/v8-fidelity-baseline-report.json golden_samples/fidelity-candidate-report.json --targeted golden_samples/fidelity-targeted-report.json --output golden_samples/fidelity-verdict.json
```

The gate allows a maximum 0.005 decrease in whole-set and validation onset/
selected-melody F1, requires the same sample IDs and verified targeted evidence,
and never treats an unmeasured game sound as a successful trigger. Reports
separate timing refresh and added vocal-check time from cached arrangement work.

### Model memory and drafts

MuScriptor, Demucs and Basic Pitch remain ready while a transcription task is active, then release their in-memory model caches after three idle minutes. The next task reloads only the model it needs; downloaded model weights are never removed.

Completed drafts are automatically stored in `Drafts/` next to the application and restored when the generation window is reopened. Local drafts retain their original source path; online drafts retain a managed copy of the downloaded audio. Right-click a draft to rename or delete it. Renaming changes the later exported score name and default filename, while deleting never removes a local source file or an already exported score.

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
source, or `reference_windows_ms` for explicit source-time windows. V3 rows
also report whole-song, front-half and back-half metrics. `top_voice` is the
highest rendered key at each onset; `selected_melody` is the arranger's actual
Viterbi-selected melody. The legacy `melody` field remains an alias of
`top_voice`.

Keep the audio and hand score in ignored `golden_samples/`; use 10–20 songs
and reserve at least three as `validation`. The report contains 60 ms
key+onset F1, top-voice/selected-melody F1, onset-only F1, onset MAE, density,
polyphony, elapsed time and peak Python allocation. Run V2 and V3 manifests separately, then
compare validation medians before changing the app default.

```powershell
python scripts/benchmark_transcription.py golden-manifest.json v2-report.json
```

For the historical V4 drum-only regression, retain its saved report and
verdict. V5 changes melody selection intentionally to prioritize detected
vocals, so compare a fresh V5 report with the V4 baseline instead. The V5 gate
allows at most a 0.005 F1 decrease in overall and validation onset/selected-
melody metrics, and rejects any per-song selected-melody recall drop.

```powershell
python scripts/benchmark_transcription.py golden_samples/golden-manifest-v3.json golden_samples/v5-vocal-golden-report.json
python scripts/verify_v5_vocal.py golden_samples/v3-v4-golden-report.json golden_samples/v5-vocal-golden-report.json --output golden_samples/v5-vocal-golden-verdict.json
```

## Packaging

The GitHub Actions build packages the V2 runtime and the optional V3 code, but
never downloads or bundles MuScriptor weights. High-quality models are fetched
only after the user accepts their license.
