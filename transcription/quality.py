"""Optional V3 whole-song transcription and melody-first Sky arrangement.

The module deliberately imports MuScriptor only at runtime. A normal V2
installation therefore remains usable offline and does not download model
weights just because the application starts.
"""
from __future__ import annotations

import bisect
import os
import threading
from collections import Counter, defaultdict
from dataclasses import replace
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .arranger import SKY_MIDI, _pitch_to_key, key_transpose
from .model_runtime import model_runtime
from .models import (
    CancelledError,
    Meter,
    NoteEvent,
    ProgressCallback,
    QualityAnalysisDraft,
    SymbolicNote,
    TranscriptionError,
    TranscriptionOptions,
)


_MODEL_LOCK = threading.Lock()
_MODELS: Dict[Tuple[str, str], object] = {}
_LEAD_PRIORITY = {
    "voice": 8, "synth_lead": 7, "violin": 6, "flutes": 6,
    "soprano_and_alto_sax": 5, "tenor_sax": 5, "clarinet": 5,
    "oboe": 5, "trumpet": 5, "acoustic_piano": 4, "electric_piano": 4,
}
_BASS_INSTRUMENTS = {"acoustic_bass", "electric_bass", "contrabass", "tuba", "bassoon"}


def release_quality_models() -> bool:
    """Drop cached MuScriptor instances after the shared idle timeout."""
    with _MODEL_LOCK:
        if not _MODELS:
            return False
        _MODELS.clear()
        return True


model_runtime.register("muscriptor", release_quality_models)


def choose_quality_device() -> str:
    """Choose the fastest local backend without making CUDA/MPS mandatory."""
    try:
        import torch
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return "cuda"
        mps = getattr(getattr(torch, "backends", object()), "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_quality_model(requested: str, device: Optional[str] = None) -> str:
    if requested in ("small", "medium"):
        return requested
    return "medium" if (device or choose_quality_device()) in ("cuda", "mps") else "small"


def _with_token(token: Optional[str]):
    """Small context manager that never writes a user token to disk/config."""
    class _Token:
        def __enter__(self):
            self.old = os.environ.get("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token

        def __exit__(self, *_exc):
            if token:
                if self.old is None:
                    os.environ.pop("HF_TOKEN", None)
                else:
                    os.environ["HF_TOKEN"] = self.old

    return _Token()


def _load_model(model_name: str, device: str, token: Optional[str] = None) -> object:
    key = (model_name, device)
    with _MODEL_LOCK:
        if key in _MODELS:
            return _MODELS[key]
        try:
            from muscriptor import TranscriptionModel
        except ImportError as exc:
            raise TranscriptionError(
                "高质量模式未安装。请安装 muscriptor==0.3.0 与 beat-this==1.1.0。"
            ) from exc
        try:
            with _with_token(token):
                model = TranscriptionModel.load_model(model_name, device=device)
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "401" in message or "403" in message:
                message = "请先在 Hugging Face 接受 MuScriptor 非商用许可，并提供临时访问 Token"
            raise TranscriptionError(f"高质量模型准备失败：{message}") from exc
        _MODELS[key] = model
        return model


def prepare_quality_model(
    requested_model: str = "auto", token: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[str, str]:
    with model_runtime.activity():
        device = choose_quality_device()
        model_name = resolve_quality_model(requested_model, device)
        if progress_cb:
            progress_cb("quality", 0.0, f"正在准备 MuScriptor {model_name} 模型")
        _load_model(model_name, device, token)
        if progress_cb:
            progress_cb("quality", 1.0, f"MuScriptor {model_name} 模型已就绪")
        return model_name, device


def _role(instrument: str) -> str:
    value = str(instrument or "").lower()
    if value == "drums":
        return "drums"
    if value in _BASS_INSTRUMENTS:
        return "bass"
    if value in _LEAD_PRIORITY:
        return "melody"
    if any(word in value for word in ("guitar", "piano", "organ", "string", "brass", "pad", "harp")):
        return "harmony"
    return "other"


def _extract_symbolic_events(model: object, path: str, cancel_event, progress_cb, beam_size: int = 1) -> List[SymbolicNote]:
    active: Dict[int, object] = {}
    notes: List[SymbolicNote] = []
    for event in model.transcribe(path, beam_size=beam_size):  # type: ignore[attr-defined]
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("转写已取消")
        name = type(event).__name__
        if name == "ProgressEvent":
            if progress_cb:
                total = max(1, int(getattr(event, "total", 1)))
                progress_cb("quality", min(.92, float(getattr(event, "completed", 0)) / total), "正在识别整曲音符与乐器")
        elif name == "NoteStartEvent":
            active[int(getattr(event, "index"))] = event
        elif name == "NoteEndEvent":
            start = getattr(event, "start_event", None) or active.pop(int(getattr(event, "start_event_index")), None)
            if start is None:
                continue
            active.pop(int(getattr(start, "index")), None)
            begin = int(round(float(getattr(start, "start_time")) * 1000))
            end = int(round(float(getattr(event, "end_time")) * 1000))
            if end > begin:
                instrument = str(getattr(start, "instrument", "other"))
                notes.append(SymbolicNote(begin, end, int(getattr(start, "pitch")), instrument, _role(instrument)))
    return sorted(notes, key=lambda item: (item.start_ms, item.midi_pitch, item.instrument))


def _timing_from_model(model: object, path: str, notes: Sequence[SymbolicNote], options: TranscriptionOptions) -> Tuple[List[int], List[int], List[int], float, Meter, float]:
    """Read Beat This!'s actual beat sequence instead of forcing a constant grid."""
    grid = None
    try:
        grid = model.detect_beat_grid_for(path, "best-effort")  # type: ignore[attr-defined]
    except Exception:
        grid = None
    beats = [int(round(float(value) * 1000)) for value in getattr(grid, "beats", [])] if grid is not None and getattr(grid, "beats", None) is not None else []
    if len(beats) < 2:
        duration = max((note.end_ms for note in notes), default=1000)
        bpm = float(options.bpm_override or 120.0)
        step = int(round(60000.0 / bpm))
        beats = list(range(0, duration + step, step))
        confidence = .2
    else:
        intervals = [right - left for left, right in zip(beats, beats[1:]) if right > left]
        bpm = float(options.bpm_override or (60000.0 / median(intervals)))
        confidence = .9
    count = int(getattr(grid, "beats_per_bar", 0) or 0) if grid is not None else 0
    if options.meter != "auto":
        meter: Meter = options.meter
        count = {"4/4": 4, "3/4": 3, "6/8": 6}[meter]
    elif count in (3, 4, 6):
        meter = f"{count}/4" if count != 6 else "6/8"  # type: ignore[assignment]
    else:
        meter, count = "4/4", 4
        confidence *= .65
    raw_downbeat = getattr(grid, "first_downbeat", None) if grid is not None else None
    first_downbeat = int(round(float(raw_downbeat) * 1000)) if raw_downbeat is not None else beats[0]
    anchor = min(range(len(beats)), key=lambda index: abs(beats[index] - first_downbeat))
    downbeats = beats[anchor::count]
    return beats, downbeats, list(downbeats), bpm, meter, confidence


def analyze_quality_audio(path: str, options: TranscriptionOptions, cancel_event=None, progress_cb: Optional[ProgressCallback] = None, token: Optional[str] = None, beam_size: int = 1) -> Tuple[QualityAnalysisDraft, List[NoteEvent]]:
    if not options.rights_confirmed:
        raise TranscriptionError("高质量模式需要确认：你拥有输入音频及生成乐谱所需的权利。")
    with model_runtime.activity():
        device = choose_quality_device()
        model_name = resolve_quality_model(options.quality_model, device)
        if progress_cb:
            progress_cb("quality", .01, f"正在加载 MuScriptor {model_name}")
        model = _load_model(model_name, device, token)
        symbolic = _extract_symbolic_events(model, path, cancel_event, progress_cb, beam_size=beam_size)
        if not symbolic:
            raise TranscriptionError("高质量模型未识别到可用音符")
        beats, downbeats, bars, bpm, meter, confidence = _timing_from_model(model, path, symbolic, options)
        duration = max(note.end_ms for note in symbolic) / 1000.0
        draft = QualityAnalysisDraft(duration, symbolic, beats, downbeats, bars, bpm, meter, confidence, model_name, device)
        events = [NoteEvent(note.start_ms, note.end_ms, note.midi_pitch, 1.0, f"quality:{note.instrument}") for note in symbolic if note.role != "drums"]
        if progress_cb:
            progress_cb("quality", 1.0, "整曲音符、节拍与重拍识别完成")
        return draft, events


def _local_grid(beats: Sequence[int], time_ms: int, subdivisions: int) -> List[int]:
    if len(beats) < 2:
        return [time_ms]
    index = max(0, min(len(beats) - 2, bisect.bisect_right(beats, time_ms) - 1))
    left, right = beats[index], beats[index + 1]
    return [int(round(left + (right - left) * unit / subdivisions)) for unit in range(subdivisions + 1)]


def _quantize_with_status(time_ms: int, draft: QualityAnalysisDraft) -> Tuple[int, bool]:
    if len(draft.beat_times_ms) < 2:
        return time_ms, False
    choices = [value for division in (2, 3, 4) for value in _local_grid(draft.beat_times_ms, time_ms, division)]
    candidate = min(choices, key=lambda value: (abs(value - time_ms), value))
    index = max(0, min(len(draft.beat_times_ms) - 2, bisect.bisect_right(draft.beat_times_ms, time_ms) - 1))
    beat = max(1, draft.beat_times_ms[index + 1] - draft.beat_times_ms[index])
    if abs(candidate - time_ms) <= min(70, int(round(beat * .18))):
        return candidate, True
    return time_ms, False


def _adaptive_quantize(time_ms: int, draft: QualityAnalysisDraft) -> int:
    """The original local 8th/triplet/16th quantizer, kept as the sound baseline."""
    return _quantize_with_status(time_ms, draft)[0]


def _lead_cost(note: SymbolicNote, previous: Optional[SymbolicNote]) -> float:
    value = 100.0 + 12.0 * _LEAD_PRIORITY.get(note.instrument, 0)
    value -= abs(note.midi_pitch - 74) * .6
    if previous:
        value -= abs(note.midi_pitch - previous.midi_pitch) * .55
        if note.instrument != previous.instrument:
            value -= 7.0
        if note.start_ms - previous.end_ms > 1400:
            value -= 3.0
    return value


def select_melody(notes: Sequence[SymbolicNote], beats: Sequence[int]) -> List[SymbolicNote]:
    """Viterbi-select one lead note per onset in two-bar-sized windows."""
    candidates = [note for note in notes if note.role == "melody" and 45 <= note.midi_pitch <= 100]
    if not candidates:
        candidates = [note for note in notes if note.role != "drums" and 54 <= note.midi_pitch <= 96]
    grouped: Dict[int, List[SymbolicNote]] = defaultdict(list)
    for note in candidates:
        grouped[_adaptive_key(note.start_ms, beats)].append(note)
    beats_per_window = 8
    selected: List[SymbolicNote] = []
    previous: Optional[SymbolicNote] = None
    windows: Dict[int, List[List[SymbolicNote]]] = defaultdict(list)
    for onset in sorted(grouped):
        beat_index = max(0, bisect.bisect_right(beats, onset) - 1) if beats else onset // 4000
        candidates_at_onset = sorted(grouped[onset], key=lambda note: (_LEAD_PRIORITY.get(note.instrument, 0), note.end_ms - note.start_ms), reverse=True)[:8]
        voice_candidates = [note for note in candidates_at_onset if note.instrument == "voice"]
        # A detected vocal onset is the song lead.  Other instruments at this
        # same onset remain available to the accompaniment pass below.
        if voice_candidates:
            candidates_at_onset = voice_candidates
        windows[beat_index // beats_per_window].append(candidates_at_onset)
    for window in (windows[index] for index in sorted(windows)):
        scores: List[List[float]] = []
        links: List[List[int]] = []
        for row, candidates_at_onset in enumerate(window):
            row_scores, row_links = [], []
            for current in candidates_at_onset:
                if row == 0:
                    row_scores.append(_lead_cost(current, previous))
                    row_links.append(-1)
                    continue
                choices = [scores[-1][old] + _lead_cost(current, window[row - 1][old]) for old in range(len(window[row - 1]))]
                best = max(range(len(choices)), key=choices.__getitem__)
                row_scores.append(choices[best])
                row_links.append(best)
            scores.append(row_scores)
            links.append(row_links)
        if not scores:
            continue
        index = max(range(len(scores[-1])), key=scores[-1].__getitem__)
        chosen: List[SymbolicNote] = []
        for row in range(len(window) - 1, -1, -1):
            chosen.append(window[row][index])
            index = links[row][index]
            if index < 0:
                break
        for current in reversed(chosen):
            # MuScriptor commonly lets sung-note tails overlap the following
            # syllable.  Preserve the newer vocal onset; retain the historical
            # guard for instrumental leads so piano-only arrangements are
            # unchanged.
            if previous and current.start_ms < previous.end_ms and current.midi_pitch != previous.midi_pitch and current.instrument != "voice":
                continue
            selected.append(current)
            previous = current
    return selected


def _adaptive_key(time_ms: int, beats: Sequence[int]) -> int:
    if len(beats) < 2:
        return time_ms
    return int(round(time_ms / 25.0) * 25)


def _shift_candidates(options: TranscriptionOptions) -> Iterable[int]:
    if options.source_key:
        return (key_transpose(options.source_key),)
    return range(-6, 6)


def _best_shift(melody: Sequence[SymbolicNote], options: TranscriptionOptions) -> int:
    if not melody:
        return 0
    best_shift, best_cost = 0, float("inf")
    for shift in _shift_candidates(options):
        previous_source = previous_sky = None
        cost = 0.0
        for note in melody:
            key, adjusted = _pitch_to_key(note.midi_pitch + shift, previous_source, previous_sky)
            cost += (2.5 if adjusted else 0.0) + abs(SKY_MIDI[key] - (note.midi_pitch + shift)) * .35
            if previous_sky is not None:
                source_delta, sky_delta = note.midi_pitch - previous_source, SKY_MIDI[key] - previous_sky
                if source_delta and sky_delta and (source_delta > 0) != (sky_delta > 0):
                    cost += 4.0
            previous_source, previous_sky = note.midi_pitch + shift, SKY_MIDI[key]
        if cost < best_cost:
            best_shift, best_cost = shift, cost
    return best_shift


def _key_for_shift(shift: int) -> str:
    for root, name in enumerate(("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")):
        if key_transpose(f"{name} major") == shift:
            return f"{name} major"
    return "C major"


def _map_quality_melody(
    melody: Sequence[SymbolicNote], draft: QualityAnalysisDraft, options: TranscriptionOptions, shift: int,
) -> List[Tuple[SymbolicNote, int, int]]:
    """Map the selected lead once for both rendering and benchmark scoring."""
    mapped: List[Tuple[SymbolicNote, int, int]] = []
    voice_positions = set()
    previous_source = previous_sky = None
    octave = 12 * (options.melody_octave_shift or 0)
    for note in melody:
        pitch = note.midi_pitch + shift + octave
        key, _ = _pitch_to_key(pitch, previous_source, previous_sky)
        time_ms = _adaptive_quantize(note.start_ms, draft)
        # Adjacent syllables can snap to the same 15-key onset.  Keep their
        # distinct attacks by falling back to the model's source time.  This
        # deliberately applies only to vocals; instrumental timing retains its
        # established quantization behavior.
        if note.instrument == "voice" and (time_ms, key) in voice_positions:
            source_position = (note.start_ms, key)
            if source_position not in voice_positions:
                time_ms = note.start_ms
        if note.instrument == "voice":
            voice_positions.add((time_ms, key))
        mapped.append((note, time_ms, key))
        previous_source, previous_sky = pitch, SKY_MIDI[key]
    return mapped


def arrange_quality_melody(draft: QualityAnalysisDraft, options: TranscriptionOptions, forced_shift: Optional[int] = None) -> List[Dict[str, object]]:
    """Return the exact selected/mapped melody path for evaluation only."""
    melody = select_melody(draft.symbolic_notes, draft.beat_times_ms)
    shift = _best_shift(melody, options) if forced_shift is None else forced_shift
    return [{"time": time_ms, "key": f"1Key{key}"} for _note, time_ms, key in _map_quality_melody(melody, draft, options, shift)]


def _diagnose_arrangement(draft: QualityAnalysisDraft, song_notes: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Read-only timing diagnostics. They never feed back into arranging."""
    duration = max([note.end_ms for note in draft.symbolic_notes] + [0])
    starts = sorted(set([0] + [value for value in draft.bar_starts_ms if 0 <= value < duration]))
    if not starts:
        starts = [0]
    ends = starts[1:] + [duration]
    diagnostics: List[Dict[str, object]] = []
    for start, end in zip(starts, ends):
        raw = [note for note in draft.symbolic_notes if start <= note.start_ms < end and note.role != "drums"]
        final_onsets = len({int(note["time"]) for note in song_notes if start <= int(note["time"]) < end})
        unsnapped = sum(not _quantize_with_status(note.start_ms, draft)[1] for note in raw)
        beats = max(1, sum(start <= beat < end for beat in draft.beat_times_ms))
        score = (2 if len(raw) >= 8 and unsnapped / max(1, len(raw)) > .30 else 0) + (1 if len(raw) / beats > 8 else 0)
        diagnostics.append({
            "startMs": start, "endMs": end, "rawNoteCount": len(raw), "finalOnsetCount": final_onsets,
            "unsnappedNoteCount": unsnapped, "suspicious": score >= 3,
        })
    return diagnostics


def arrange_quality_analysis(draft: QualityAnalysisDraft, options: TranscriptionOptions, forced_shift: Optional[int] = None) -> Tuple[List[Dict[str, object]], str, int, Dict[str, object]]:
    melody = select_melody(draft.symbolic_notes, draft.beat_times_ms)
    shift = _best_shift(melody, options) if forced_shift is None else forced_shift
    melody_ids = {(note.start_ms, note.end_ms, note.midi_pitch, note.instrument) for note in melody}
    by_time: Dict[int, Dict[int, Tuple[int, set[str]]]] = defaultdict(dict)
    previous_source = previous_sky = None

    def add(time_ms: int, pitch: int, priority: int, role: str) -> None:
        nonlocal previous_source, previous_sky
        index, _ = _pitch_to_key(pitch + shift, previous_source, previous_sky)
        old = by_time[time_ms].get(index)
        if old is None or priority > old[0]:
            by_time[time_ms][index] = (priority, {role})
        elif priority == old[0]:
            old[1].add(role)
        if role == "melody":
            previous_source, previous_sky = pitch + shift, SKY_MIDI[index]

    mapped_melody = _map_quality_melody(melody, draft, options, shift)
    for note, time_ms, index in mapped_melody:
        old = by_time[time_ms].get(index)
        if old is None or old[0] < 100:
            by_time[time_ms][index] = (100, {"melody"})
        elif old[0] == 100:
            old[1].add("melody")
    if mapped_melody:
        last_note, _last_time, last_index = mapped_melody[-1]
        previous_source = last_note.midi_pitch + shift + 12 * (options.melody_octave_shift or 0)
        previous_sky = SKY_MIDI[last_index]

    density = {"simple": 4, "standard": 2, "full": 1, "auto": 2}[options.arrangement_preset]
    # Keep the original candidate stream, including drum positions, through
    # sampling and max-polyphony selection. Removing drums earlier would free
    # slots and let previously discarded non-drum notes enter the output.
    accompaniment = [note for note in draft.symbolic_notes if note.role not in ("drums", "melody") or (note.start_ms, note.end_ms, note.midi_pitch, note.instrument) not in melody_ids]
    for ordinal, note in enumerate(accompaniment):
        if ordinal % density:
            continue
        time = _adaptive_quantize(note.start_ms, draft)
        add(time, note.midi_pitch, 40 if note.role == "bass" else 20, note.role)

    song_notes: List[Dict[str, object]] = []
    roles: Counter[str] = Counter()
    grouped: List[int] = []
    for time_ms in sorted(by_time):
        selected = sorted(by_time[time_ms].items(), key=lambda item: (-item[1][0], item[0]))[: options.max_polyphony]
        emitted = 0
        for key_index, (_priority, source_roles) in sorted(selected):
            # A shared key survives whenever a non-drum source produced it;
            # only entries derived exclusively from a drum source are removed.
            if source_roles == {"drums"}:
                continue
            song_notes.append({"time": int(time_ms), "key": f"1Key{key_index}"})
            emitted += 1
            role = next(role for role in source_roles if role != "drums")
            roles[role] += 1
        if emitted:
            grouped.append(time_ms)
    diagnostics = _diagnose_arrangement(draft, song_notes)
    stats: Dict[str, object] = {
        "arranged_note_count": len(song_notes), "onset_count": len(grouped),
        "chord_count": sum(sum(1 for note in song_notes if note["time"] == time) > 1 for time in grouped),
        "average_polyphony": round(len(song_notes) / max(1, len(grouped)), 3),
        "melody_note_count": roles["melody"], "qualitySymbolicCount": len(draft.symbolic_notes),
        "timingGridCount": len(draft.beat_times_ms), "timingConfidence": round(draft.timing_confidence, 3),
        "quantization": draft.quantization,
        "filteredDrumCount": sum(note.role == "drums" for note in draft.symbolic_notes),
        "unsnappedNoteCount": sum(not _quantize_with_status(note.start_ms, draft)[1] for note in draft.symbolic_notes if note.role != "drums"),
        "barDiagnostics": diagnostics,
        "suspiciousBarCount": sum(bool(item["suspicious"]) for item in diagnostics),
        "qualityArrangerVersion": 5,
    }
    return song_notes, options.source_key or _key_for_shift(shift), shift, stats
