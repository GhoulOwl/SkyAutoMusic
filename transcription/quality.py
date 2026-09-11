"""Optional V3 whole-song transcription and melody-first Sky arrangement.

The module deliberately imports MuScriptor only at runtime. A normal V2
installation therefore remains usable offline and does not download model
weights just because the application starts.
"""
from __future__ import annotations

import bisect
import math
import os
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .arranger import MAJOR_PROFILE, MINOR_PROFILE, SKY_MIDI, _pitch_to_key, key_transpose, parse_key
from .analysis import CHORD_INTERVALS
from .model_runtime import model_runtime
from .model_download import DownloadSource, download_model
from .models import (
    CancelledError,
    LeadSegment,
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
_BEAT_TRACKERS: Dict[str, object] = {}
_LEAD_PRIORITY = {
    "voice": 8, "synth_lead": 7, "violin": 6, "flutes": 6,
    "soprano_and_alto_sax": 5, "tenor_sax": 5, "clarinet": 5,
    "oboe": 5, "trumpet": 5, "acoustic_piano": 4, "electric_piano": 4,
}
_BASS_INSTRUMENTS = {"acoustic_bass", "electric_bass", "contrabass", "tuba", "bassoon"}
_LOW_TIMING_THRESHOLD = .5
_MIN_ACCOMPANIMENT_SCORE = 4.0


@dataclass(frozen=True)
class _AccompanimentCandidate:
    """A safe, mapped accompaniment event ready for bar budgeting."""

    note: SymbolicNote
    time_ms: int
    key_index: int
    bar_index: int
    beat_ms: int
    score: float
    role: str
    matched_melody_time: Optional[int] = None
    provenance: str = "source"
    anchor_confidence: float = 0.0
    local_interval_ms: int = 500

    @property
    def duration_ms(self) -> int:
        return self.note.end_ms - self.note.start_ms


@dataclass(frozen=True)
class _InferredChord:
    """A locally-supported diatonic chord attached to a locked melody onset."""

    melody_index: int
    time_ms: int
    melody_key: int
    root_pc: int
    quality: str
    confidence: float
    local_interval_ms: int


def release_quality_models() -> bool:
    """Drop cached MuScriptor instances after the shared idle timeout."""
    with _MODEL_LOCK:
        if not _MODELS and not _BEAT_TRACKERS:
            return False
        _MODELS.clear()
        _BEAT_TRACKERS.clear()
        return True


model_runtime.register("muscriptor", release_quality_models)


def choose_quality_device(preference: str = "auto") -> str:
    """Choose the fastest local backend without making CUDA/MPS mandatory."""
    try:
        import torch
        if preference != "cpu" and bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return "cuda"
        mps = getattr(getattr(torch, "backends", object()), "mps", None)
        if preference != "cpu" and mps is not None and bool(mps.is_available()):
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


def _load_model(model_name: str, device: str, token: Optional[str] = None,
                download_source: DownloadSource = "auto", progress_cb: Optional[ProgressCallback] = None) -> object:
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
            weights = download_model(model_name, download_source, token, lambda message: _notify(progress_cb, "quality", .02, message))
            model = TranscriptionModel.load_model(weights, device=device)
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "401" in message or "403" in message:
                message = "请先在 Hugging Face 接受 MuScriptor 非商用许可，并提供临时访问 Token"
            raise TranscriptionError(f"高质量模型准备失败：{message}") from exc
        _MODELS[key] = model
        return model


def prepare_quality_model(
    requested_model: str = "auto", token: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None, device_preference: str = "auto",
    download_source: DownloadSource = "auto",
) -> Tuple[str, str]:
    with model_runtime.activity():
        device = choose_quality_device(device_preference)
        if device_preference == "gpu" and device == "cpu":
            raise TranscriptionError("未检测到 CUDA 或 Apple 芯片加速设备，请选择 Small（CPU）。")
        model_name = resolve_quality_model(requested_model, device)
        if progress_cb:
            progress_cb("quality", 0.0, f"正在准备 MuScriptor {model_name} 模型")
        _load_model(model_name, device, token, download_source, progress_cb)
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


def _clean_times(values: Sequence[object]) -> List[int]:
    """Keep the strictly increasing tracker output; malformed points are not beats."""
    result: List[int] = []
    for value in values:
        try:
            point = int(round(float(value) * 1000.0))
        except (TypeError, ValueError):
            continue
        if point >= 0 and (not result or point > result[-1]):
            result.append(point)
    return result


def _beat_tracker(device: str) -> object:
    """Load Beat This! directly, bypassing MuScriptor's fixed-tempo gate."""
    # Beat This! is reliable on CPU across the supported desktop platforms;
    # MuScriptor can continue to use MPS/CUDA for the expensive note model.
    target = "cuda" if device == "cuda" else "cpu"
    with _MODEL_LOCK:
        tracker = _BEAT_TRACKERS.get(target)
        if tracker is not None:
            return tracker
        try:
            from beat_this.inference import File2Beats
            tracker = File2Beats(checkpoint_path="final0", device=target, dbn=False)
        except Exception as exc:
            raise TranscriptionError(f"Beat This! 节拍模型准备失败：{exc}") from exc
        _BEAT_TRACKERS[target] = tracker
        return tracker


def _infer_meter_from_downbeats(beats: Sequence[int], downbeats: Sequence[int]) -> Tuple[Meter, int, float]:
    if len(beats) < 2 or len(downbeats) < 2:
        return "4/4", 4, .0
    positions = [min(range(len(beats)), key=lambda index: abs(beats[index] - downbeat)) for downbeat in downbeats]
    counts = [right - left for left, right in zip(positions, positions[1:]) if 2 <= right - left <= 8]
    if not counts:
        return "4/4", 4, .0
    count = Counter(counts).most_common(1)[0][0]
    agreement = counts.count(count) / len(counts)
    if count not in (3, 4, 6):
        return "4/4", 4, agreement * .35
    return ("6/8" if count == 6 else f"{count}/4"), count, agreement  # type: ignore[return-value]


def _timing_from_model(model: object, path: str, notes: Sequence[SymbolicNote], options: TranscriptionOptions, device: str = "cpu") -> Tuple[List[int], List[int], List[int], float, Meter, float, str, Dict[str, object]]:
    """Read Beat This! directly and keep expressive, non-uniform beat timings.

    MuScriptor's helper deliberately rejects songs whose beats do not fit one
    global tempo. Pop recordings frequently have those small tempo movements,
    so rejecting them and snapping notes to 120 BPM was worse than retaining a
    low-confidence grid.  Only a tracker failure creates the synthetic grid.
    """
    diagnostics: Dict[str, object] = {}
    beats: List[int] = []
    downbeats: List[int] = []
    try:
        raw_beats, raw_downbeats = _beat_tracker(device)(path)  # type: ignore[operator]
        beats, downbeats = _clean_times(raw_beats), _clean_times(raw_downbeats)
        diagnostics["rawBeatCount"] = len(beats)
        diagnostics["rawDownbeatCount"] = len(downbeats)
    except Exception as exc:
        diagnostics["failure"] = str(exc)

    if len(beats) >= 2:
        intervals = [right - left for left, right in zip(beats, beats[1:]) if right > left]
        period = max(1.0, float(median(intervals)))
        deviations = [abs(value - period) for value in intervals]
        mad = float(median(deviations)) if deviations else period
        regularity = max(.0, min(1.0, 1.0 - mad / max(1.0, period * .35)))
        observed_bpm = 60000.0 / period
        if options.bpm_override is not None:
            scale = observed_bpm / float(options.bpm_override)
            origin = beats[0]
            beats = [int(round(origin + (point - origin) * scale)) for point in beats]
            downbeats = [int(round(origin + (point - origin) * scale)) for point in downbeats]
            bpm = float(options.bpm_override)
            diagnostics["bpmScale"] = round(scale, 6)
        else:
            bpm = observed_bpm
        inferred_meter, count, agreement = _infer_meter_from_downbeats(beats, downbeats)
        if options.meter != "auto":
            meter: Meter = options.meter
            count = {"4/4": 4, "3/4": 3, "6/8": 6}[meter]
        else:
            meter = inferred_meter
        if not downbeats:
            downbeats = beats[::count]
        else:
            anchor = min(range(len(beats)), key=lambda index: abs(beats[index] - downbeats[0]))
            downbeats = beats[anchor::count]
        confidence = max(.35, min(.95, .50 + .30 * regularity + .20 * agreement))
        diagnostics.update({
            "medianBeatMs": round(period, 2), "beatMadMs": round(mad, 2),
            "regularity": round(regularity, 3), "meterAgreement": round(agreement, 3),
        })
        return beats, downbeats, list(downbeats), bpm, meter, confidence, "beat_this_direct", diagnostics

    duration = max((note.end_ms for note in notes), default=1000)
    bpm = float(options.bpm_override or 120.0)
    step = int(round(60000.0 / bpm))
    beats = list(range(0, duration + step, step))
    meter = options.meter if options.meter != "auto" else "4/4"
    count = {"4/4": 4, "3/4": 3, "6/8": 6}[meter]
    downbeats = beats[::count]
    diagnostics["fallback"] = "fixed_grid_without_quantization"
    return beats, downbeats, list(downbeats), bpm, meter, .13, "fixed_grid_fallback", diagnostics


def analyze_quality_audio(path: str, options: TranscriptionOptions, cancel_event=None, progress_cb: Optional[ProgressCallback] = None, token: Optional[str] = None, beam_size: int = 1) -> Tuple[QualityAnalysisDraft, List[NoteEvent]]:
    if not options.rights_confirmed:
        raise TranscriptionError("高质量模式需要确认：你拥有输入音频及生成乐谱所需的权利。")
    with model_runtime.activity():
        device = choose_quality_device(options.quality_device)
        if options.quality_device == "gpu" and device == "cpu":
            raise TranscriptionError("未检测到 CUDA 或 Apple 芯片加速设备，请选择 Small（CPU）。")
        model_name = resolve_quality_model(options.quality_model, device)
        if progress_cb:
            progress_cb("quality", .01, f"正在加载 MuScriptor {model_name}")
        model = _load_model(model_name, device, token, options.quality_download_source, progress_cb)
        symbolic = _extract_symbolic_events(model, path, cancel_event, progress_cb, beam_size=beam_size)
        if not symbolic:
            raise TranscriptionError("高质量模型未识别到可用音符")
        beats, downbeats, bars, bpm, meter, confidence, timing_backend, timing_diagnostics = _timing_from_model(model, path, symbolic, options, device)
        duration = max(note.end_ms for note in symbolic) / 1000.0
        _selected, lead_segments = select_melody_with_segments(symbolic, beats)
        draft = QualityAnalysisDraft(
            duration, symbolic, beats, downbeats, bars, bpm, meter, confidence,
            model_name, device, timing_backend=timing_backend,
            timing_diagnostics=timing_diagnostics, lead_segments=lead_segments,
        )
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
    # A synthetic grid is useful for display but must never move detected notes.
    if len(draft.beat_times_ms) < 2 or draft.timing_backend == "fixed_grid_fallback" or draft.timing_confidence < _LOW_TIMING_THRESHOLD:
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


def _vocal_guard_ms(beats: Sequence[int]) -> int:
    if len(beats) < 2:
        return 1500
    intervals = [right - left for left, right in zip(beats, beats[1:]) if right > left]
    return max(400, int(round(2 * median(intervals))))


def _inside_or_near_vocal_phrase(time_ms: int, vocals: Sequence[SymbolicNote], guard_ms: int) -> bool:
    return any(note.start_ms - guard_ms <= time_ms <= note.end_ms + guard_ms for note in vocals)


def _lead_segments(selected: Sequence[SymbolicNote], duration_ms: int) -> List[LeadSegment]:
    if not selected:
        return [LeadSegment(0, max(1, duration_ms), "silence", None)]
    segments: List[LeadSegment] = []
    start, end = selected[0].start_ms, selected[0].end_ms
    source = "vocal" if selected[0].instrument == "voice" else "instrumental"
    for note in selected[1:]:
        current_source = "vocal" if note.instrument == "voice" else "instrumental"
        if current_source == source and note.start_ms - end <= 900:
            end = max(end, note.end_ms)
            continue
        segments.append(LeadSegment(start, max(start + 1, end), source, None))
        start, end, source = note.start_ms, note.end_ms, current_source
    segments.append(LeadSegment(start, max(start + 1, end), source, None))
    return segments


def select_melody_with_segments(notes: Sequence[SymbolicNote], beats: Sequence[int]) -> Tuple[List[SymbolicNote], List[LeadSegment]]:
    """Choose vocals by phrase and use BGM only in a clear non-vocal passage."""
    vocals = [note for note in notes if note.instrument == "voice" and 45 <= note.midi_pitch <= 100]
    instruments = [note for note in notes if note.instrument != "voice" and note.role != "drums" and 54 <= note.midi_pitch <= 96]
    guard_ms = _vocal_guard_ms(beats)
    grouped: Dict[int, List[SymbolicNote]] = defaultdict(list)
    for note in [*vocals, *instruments]:
        onset = _adaptive_key(note.start_ms, beats)
        if note.instrument == "voice" or not _inside_or_near_vocal_phrase(note.start_ms, vocals, guard_ms):
            grouped[onset].append(note)

    selected: List[SymbolicNote] = []
    previous: Optional[SymbolicNote] = None
    for onset in sorted(grouped):
        row = grouped[onset]
        voice = [note for note in row if note.instrument == "voice"]
        candidates = voice or sorted(row, key=lambda note: (_LEAD_PRIORITY.get(note.instrument, 0), note.end_ms - note.start_ms, -abs(note.midi_pitch - 74)), reverse=True)[:8]
        if not candidates:
            continue
        current = max(candidates, key=lambda note: _lead_cost(note, previous))
        if previous and current.instrument == previous.instrument and current.midi_pitch == previous.midi_pitch and current.start_ms - previous.start_ms < 55:
            if current.end_ms > previous.end_ms:
                selected[-1] = current
                previous = current
            continue
        selected.append(current)
        previous = current
    duration = max([note.end_ms for note in notes] + [1])
    return selected, _lead_segments(selected, duration)


def select_melody(notes: Sequence[SymbolicNote], beats: Sequence[int]) -> List[SymbolicNote]:
    """Compatibility wrapper for callers interested only in selected notes."""
    return select_melody_with_segments(notes, beats)[0]


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


def _phrase_octave_offsets(melody: Sequence[SymbolicNote], shift: int, options: TranscriptionOptions) -> List[int]:
    """Choose an octave per musical phrase instead of folding every note alone."""
    if options.melody_octave_shift is not None:
        return [12 * options.melody_octave_shift] * len(melody)
    offsets: List[int] = []
    index = 0
    while index < len(melody):
        end = index + 1
        while end < len(melody):
            previous, current = melody[end - 1], melody[end]
            if current.instrument != previous.instrument or current.start_ms - previous.end_ms > 1400:
                break
            end += 1
        phrase = melody[index:end]
        # The 15 keys cover C4--C6.  Centre a phrase there, then prefer the
        # option that needs the fewest folds and natural-note substitutions.
        def cost(offset: int) -> Tuple[float, int]:
            values = [note.midi_pitch + shift + offset for note in phrase]
            folds = sum(value < SKY_MIDI[0] or value > SKY_MIDI[-1] for value in values)
            centre = sum(abs(value - 72) for value in values) / max(1, len(values))
            return 12.0 * folds + centre, abs(offset)
        selected = min((-12, 0, 12), key=cost)
        offsets.extend([selected] * len(phrase))
        index = end
    return offsets


def _map_quality_melody(
    melody: Sequence[SymbolicNote], draft: QualityAnalysisDraft, options: TranscriptionOptions, shift: int,
) -> List[Tuple[SymbolicNote, int, int]]:
    """Map the selected lead once for both rendering and benchmark scoring."""
    mapped: List[Tuple[SymbolicNote, int, int]] = []
    voice_positions = set()
    previous_source = previous_sky = None
    offsets = _phrase_octave_offsets(melody, shift, options)
    for note, octave in zip(melody, offsets):
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
    # V9 drafts may contain experimental synthetic entries.  They were never
    # part of the V8 symbolic model and must not change a V8 re-arrangement.
    if any(note.source.startswith("vocal_evidence:") for note in draft.symbolic_notes):
        draft = replace(draft, symbolic_notes=[note for note in draft.symbolic_notes if not note.source.startswith("vocal_evidence:")])
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


def _note_id(note: SymbolicNote) -> Tuple[int, int, int, str]:
    return note.start_ms, note.end_ms, note.midi_pitch, note.instrument


def _clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def _local_beat_ms(draft: QualityAnalysisDraft, time_ms: int) -> int:
    beats = draft.beat_times_ms
    if len(beats) < 2:
        return max(1, int(round(60000.0 / max(1.0, draft.bpm))))
    index = max(0, min(len(beats) - 2, bisect.bisect_right(beats, time_ms) - 1))
    return max(1, beats[index + 1] - beats[index])


def _bar_ranges(draft: QualityAnalysisDraft) -> List[Tuple[int, int]]:
    duration = max([note.end_ms for note in draft.symbolic_notes] + [1])
    starts = sorted(set([0] + [value for value in draft.bar_starts_ms if 0 <= value < duration]))
    return list(zip(starts, starts[1:] + [duration])) or [(0, duration)]


def _bar_index(ranges: Sequence[Tuple[int, int]], time_ms: int) -> int:
    for index, (start, end) in enumerate(ranges):
        if start <= time_ms < end:
            return index
    return max(0, len(ranges) - 1)


def _nearest_beat_index(beats: Sequence[int], time_ms: int) -> Tuple[int, int]:
    if not beats:
        return 0, time_ms
    index = bisect.bisect_left(beats, time_ms)
    choices = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(beats)]
    best = min(choices or [0], key=lambda candidate: (abs(beats[candidate] - time_ms), candidate))
    return best, abs(beats[best] - time_ms)


def _beat_score(draft: QualityAnalysisDraft, time_ms: int, beat_ms: int) -> float:
    index, distance = _nearest_beat_index(draft.beat_times_ms, time_ms)
    if distance > min(90, int(round(beat_ms * .18))):
        return -2.0
    meter_count = {"4/4": 4, "3/4": 3, "6/8": 6}[draft.meter]
    downbeat_index, _ = _nearest_beat_index(draft.beat_times_ms, draft.downbeat_times_ms[0] if draft.downbeat_times_ms else 0)
    position = (index - downbeat_index) % meter_count
    if position == 0:
        return 6.0
    if (draft.meter == "4/4" and position == 2) or (draft.meter == "6/8" and position == 3):
        return 4.0
    return 1.0


def _candidate_rank(candidate: _AccompanimentCandidate) -> Tuple[int, float, int, int, int, int]:
    return (
        1 if candidate.provenance == "source" else 0,
        candidate.score,
        1 if candidate.role == "bass" else 0,
        candidate.duration_ms,
        -candidate.time_ms,
        -candidate.key_index,
    )


def _locked_melody(
    melody: Sequence[SymbolicNote], draft: QualityAnalysisDraft, options: TranscriptionOptions, shift: int,
) -> Tuple[List[Tuple[SymbolicNote, int, int]], Dict[int, set[int]], set[Tuple[int, int, int, str]]]:
    mapped = _map_quality_melody(melody, draft, options, shift)
    locked: Dict[int, set[int]] = defaultdict(set)
    for _note, time_ms, key_index in mapped:
        locked[time_ms].add(key_index)
    return mapped, locked, {_note_id(note) for note, _time, _key in mapped}


def _melody_distance_score(
    melody_by_bar: Dict[int, List[Tuple[int, int]]], bar: int, time_ms: int, key_index: int,
) -> float:
    melody = melody_by_bar.get(bar, [])
    if not melody:
        return 0.0
    _time, melody_key = min(melody, key=lambda item: (abs(item[0] - time_ms), item[0], item[1]))
    distance = melody_key - key_index
    if 3 <= distance <= 8:
        return 3.0
    if distance == 2 or 9 <= distance <= 10:
        return 1.0
    return -3.0


def _candidate_score(
    note: SymbolicNote, time_ms: int, key_index: int, bar: int, beat_ms: int,
    draft: QualityAnalysisDraft, melody_by_bar: Dict[int, List[Tuple[int, int]]], protected: bool,
) -> float:
    score = 4.0 if note.role == "bass" else 2.0
    score += min(4.0, 2.0 * (note.end_ms - note.start_ms) / max(1, beat_ms))
    score += _melody_distance_score(melody_by_bar, bar, time_ms, key_index)
    if not protected:
        score += _beat_score(draft, time_ms, beat_ms)
        score -= 4.0 * (1.0 - draft.timing_confidence)
    return score


def _safe_accompaniment_candidates(
    draft: QualityAnalysisDraft, shift: int,
    mapped_melody: Sequence[Tuple[SymbolicNote, int, int]], locked: Dict[int, set[int]],
    melody_ids: set[Tuple[int, int, int, str]], stats: Counter[str], protected: bool,
) -> List[_AccompanimentCandidate]:
    ranges = _bar_ranges(draft)
    melody_by_bar: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for _note, time_ms, key_index in mapped_melody:
        melody_by_bar[_bar_index(ranges, time_ms)].append((time_ms, key_index))
    raw_candidates: List[_AccompanimentCandidate] = []
    for note in draft.symbolic_notes:
        identity = _note_id(note)
        if identity in melody_ids:
            continue
        accompaniment_role = note.role
        if note.role == "melody":
            # Instrument labels are evidence, not a permanent rendering role.
            # Piano and similar texture can support a vocal lead even if the
            # transcriber labelled it "melody" on another phrase.
            if note.instrument in ("acoustic_piano", "electric_piano") or any(
                word in note.instrument for word in ("guitar", "string", "organ", "harp")
            ):
                accompaniment_role = "bass" if note.midi_pitch < 60 else "harmony"
                stats["reused_source_texture_count"] += 1
            else:
                stats["filtered_unselected_melody_count"] += 1
                continue
        if note.role == "drums":
            stats["filtered_drum_count"] += 1
            continue
        if note.role == "other":
            stats["filtered_other_count"] += 1
            continue
        if note.role not in ("bass", "harmony"):
            continue
        beat_ms = _local_beat_ms(draft, note.start_ms)
        if note.end_ms - note.start_ms < _clamp(beat_ms * .20, 80, 180):
            stats["filtered_short_accompaniment_count"] += 1
            continue
        key_index, _adjusted = _pitch_to_key(note.midi_pitch + shift)
        matched_time: Optional[int] = None
        if protected:
            matches = [item for item in mapped_melody if abs(item[0].start_ms - note.start_ms) <= 90]
            if not matches:
                stats["filtered_independent_protected_count"] += 1
                continue
            selected = min(matches, key=lambda item: (abs(item[0].start_ms - note.start_ms), item[1], item[2]))
            matched_time = selected[1]
            time_ms = matched_time
        else:
            time_ms = _adaptive_quantize(note.start_ms, draft)
        conflict_window = min(90, int(round(beat_ms * .18)))
        if any(
            abs(time_ms - melody_time) <= conflict_window and any(abs(key_index - melody_key) <= 1 for melody_key in melody_keys)
            for melody_time, melody_keys in locked.items()
        ):
            stats["filtered_melody_conflict_count"] += 1
            continue
        bar = _bar_index(ranges, time_ms)
        raw_candidates.append(_AccompanimentCandidate(
            note, time_ms, key_index, bar, beat_ms,
            _candidate_score(note, time_ms, key_index, bar, beat_ms, draft, melody_by_bar, protected),
            accompaniment_role, matched_time, "source", 0.0, beat_ms,
        ))

    # Same mapped key/time is always one playable event; choose deterministically.
    unique: Dict[Tuple[int, int], _AccompanimentCandidate] = {}
    for candidate in raw_candidates:
        key = candidate.time_ms, candidate.key_index
        current = unique.get(key)
        if current is None or _candidate_rank(candidate) > _candidate_rank(current):
            if current is not None:
                stats["filtered_duplicate_accompaniment_count"] += 1
            unique[key] = candidate
        else:
            stats["filtered_duplicate_accompaniment_count"] += 1

    # Dense re-articulations from one instrument are reduced before bar scoring.
    retained: List[_AccompanimentCandidate] = []
    grouped: Dict[Tuple[str, str], List[_AccompanimentCandidate]] = defaultdict(list)
    for candidate in unique.values():
        grouped[(candidate.role, candidate.note.instrument)].append(candidate)
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item.time_ms, item.key_index, item.note.end_ms))
        cluster: List[_AccompanimentCandidate] = []
        previous_time: Optional[int] = None
        for candidate in ordered:
            dense_window = _clamp(candidate.beat_ms * .25, 90, 220)
            if cluster and previous_time is not None and candidate.time_ms - previous_time >= dense_window:
                best = max(cluster, key=_candidate_rank)
                retained.append(best)
                stats["filtered_dense_accompaniment_count"] += len(cluster) - 1
                cluster = []
            cluster.append(candidate)
            previous_time = candidate.time_ms
        if cluster:
            best = max(cluster, key=_candidate_rank)
            retained.append(best)
            stats["filtered_dense_accompaniment_count"] += len(cluster) - 1

    # Reward stable bass and penalize crowded onsets only after direct filtering.
    bass_keys: Dict[int, set[int]] = defaultdict(set)
    crowd: Counter[int] = Counter(candidate.time_ms for candidate in retained)
    for candidate in retained:
        if candidate.role == "bass":
            bass_keys[candidate.bar_index].add(candidate.key_index)
    scored: List[_AccompanimentCandidate] = []
    for candidate in retained:
        score = candidate.score - min(6, 2 * max(0, crowd[candidate.time_ms] - 1))
        if candidate.role == "bass":
            score += 3.0 * sum(candidate.key_index in bass_keys.get(bar, set()) for bar in (candidate.bar_index - 1, candidate.bar_index + 1))
        if score < _MIN_ACCOMPANIMENT_SCORE:
            stats["filtered_low_score_accompaniment_count"] += 1
            continue
        scored.append(replace(candidate, score=score))
    return scored


def _profile_similarity(histogram: Sequence[float], profile: Sequence[float], root: int) -> float:
    target = [profile[(pitch - root) % 12] for pitch in range(12)]
    dot = sum(left * right for left, right in zip(histogram, target))
    norm = math.sqrt(sum(value * value for value in histogram) * sum(value * value for value in target))
    return dot / norm if norm else 0.0


def _infer_harmonic_key(
    mapped_melody: Sequence[Tuple[SymbolicNote, int, int]], draft: QualityAnalysisDraft, options: TranscriptionOptions,
) -> Tuple[int, str, float]:
    """Infer a source-space key without altering the V5 mapping shift."""
    if options.source_key:
        root, mode = parse_key(options.source_key)
        return root, mode, 1.0
    histogram = [0.0] * 12
    for note, _time, _key in mapped_melody:
        histogram[note.midi_pitch % 12] += max(.08, (note.end_ms - note.start_ms) / 1000.0)
    # Explicitly trusted accompaniment can stabilize a key, but unselected
    # melody/piano notes are deliberately not evidence: they never re-enter
    # the V6/V7 accompaniment path by another name.
    for note in draft.symbolic_notes:
        if note.role in ("bass", "harmony"):
            histogram[note.midi_pitch % 12] += .35 * max(.08, (note.end_ms - note.start_ms) / 1000.0)
    choices = [
        (_profile_similarity(histogram, profile, root), root, mode)
        for profile, mode in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor"))
        for root in range(12)
    ]
    choices.sort(reverse=True)
    if not choices or choices[0][0] <= 0:
        return 0, "major", 0.0
    best, second = choices[0], choices[1] if len(choices) > 1 else (0.0, 0, "major")
    confidence = max(.0, min(.99, .35 + .45 * best[0] + .20 * min(1.0, (best[0] - second[0]) * 12.0)))
    return best[1], best[2], confidence


def _diatonic_chords(root: int, mode: str) -> List[Tuple[int, str, Tuple[int, ...]]]:
    if mode == "minor":
        specs = ((0, "min"), (2, "dim"), (3, "maj"), (5, "min"), (7, "maj"), (8, "maj"), (10, "maj"))
    else:
        specs = ((0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min"), (11, "dim"))
    return [
        ((root + offset) % 12, quality, tuple((root + offset + interval) % 12 for interval in CHORD_INTERVALS[quality]))
        for offset, quality in specs
    ]


def _local_melody_interval(mapped_melody: Sequence[Tuple[SymbolicNote, int, int]], index: int) -> int:
    current = mapped_melody[index][0].start_ms
    values = [
        abs(current - mapped_melody[neighbor][0].start_ms)
        for neighbor in (index - 1, index + 1)
        if 0 <= neighbor < len(mapped_melody) and current != mapped_melody[neighbor][0].start_ms
    ]
    return _clamp(median(values) if values else 500, 80, 2000)


def _harmonic_evidence(
    index: int, mapped_melody: Sequence[Tuple[SymbolicNote, int, int]], draft: QualityAnalysisDraft,
) -> List[Tuple[int, float]]:
    note = mapped_melody[index][0]
    evidence: List[Tuple[int, float]] = []
    for offset, weight in ((0, 4.0), (-1, 1.6), (1, 1.6), (-2, .8), (2, .8)):
        neighbor = index + offset
        if 0 <= neighbor < len(mapped_melody):
            value = mapped_melody[neighbor][0]
            evidence.append((value.midi_pitch % 12, weight))
    for support in draft.symbolic_notes:
        if support.role not in ("bass", "harmony"):
            continue
        close = abs(support.start_ms - note.start_ms) <= 400
        active = support.start_ms <= note.start_ms <= support.end_ms
        if close or active:
            evidence.append((support.midi_pitch % 12, 2.0 if support.role == "bass" else 1.25))
    return evidence


def _chord_observation(current_pc: int, pcs: Sequence[int], evidence: Sequence[Tuple[int, float]]) -> Tuple[float, float]:
    pc_set = set(pcs)
    total = sum(weight for _pc, weight in evidence) or 1.0
    inside = sum(weight for pc, weight in evidence if pc in pc_set)
    outside = total - inside
    score = (5.0 if current_pc in pc_set else -5.0) + inside - .35 * outside
    return score, inside / total


def _chord_transition(previous: Tuple[int, str, Tuple[int, ...]], current: Tuple[int, str, Tuple[int, ...]]) -> float:
    previous_root, previous_quality, previous_pcs = previous
    root, quality, pcs = current
    if previous_root == root and previous_quality == quality:
        return 2.0
    common = len(set(previous_pcs) & set(pcs))
    motion = (root - previous_root) % 12
    return .75 * common + (.7 if motion in (5, 7) else 0.0) - (1.0 if common == 0 else 0.0)


def _infer_harmonic_anchors(
    mapped_melody: Sequence[Tuple[SymbolicNote, int, int]], draft: QualityAnalysisDraft, options: TranscriptionOptions,
) -> Tuple[List[_InferredChord], int, str, float, float]:
    """Infer stable diatonic chords on melody onsets; never invent timing."""
    if len(mapped_melody) < 4:
        return [], 0, options.source_key or "C major", 0.0, 0.0
    root, mode, key_confidence = _infer_harmonic_key(mapped_melody, draft, options)
    states = _diatonic_chords(root, mode)
    anchors: List[_InferredChord] = []
    inferred_count = 0
    index = 0
    while index < len(mapped_melody):
        end = index + 1
        while end < len(mapped_melody) and mapped_melody[end][0].start_ms - mapped_melody[end - 1][0].start_ms < 1400:
            end += 1
        local_scores: List[List[Tuple[float, float]]] = []
        scores: List[List[float]] = []
        links: List[List[int]] = []
        for position in range(index, end):
            note = mapped_melody[position][0]
            evidence = _harmonic_evidence(position, mapped_melody, draft)
            observations = [_chord_observation(note.midi_pitch % 12, state[2], evidence) for state in states]
            local_scores.append(observations)
            row_scores: List[float] = []
            row_links: List[int] = []
            for current, (observation, _coverage) in enumerate(observations):
                if not scores:
                    row_scores.append(observation)
                    row_links.append(-1)
                    continue
                alternatives = [scores[-1][previous] + _chord_transition(states[previous], states[current]) + observation for previous in range(len(states))]
                best = max(range(len(alternatives)), key=alternatives.__getitem__)
                row_scores.append(alternatives[best])
                row_links.append(best)
            scores.append(row_scores)
            links.append(row_links)
        state_index = max(range(len(scores[-1])), key=scores[-1].__getitem__)
        selected_states = [0] * len(scores)
        for row in range(len(scores) - 1, -1, -1):
            selected_states[row] = state_index
            state_index = links[row][state_index]
            if state_index < 0:
                break
        previous_state: Optional[Tuple[int, str]] = None
        for row, state_index in enumerate(selected_states):
            ordered = sorted((score for score, _coverage in local_scores[row]), reverse=True)
            best_observation, coverage = local_scores[row][state_index]
            margin = best_observation - (ordered[1] if len(ordered) > 1 else best_observation)
            confidence = .60 * coverage + .40 * max(.0, min(1.0, margin / 3.5))
            current_root, quality, _pcs = states[state_index]
            if confidence >= .45:
                melody_index = index + row
                note, time_ms, melody_key = mapped_melody[melody_index]
                anchors.append(_InferredChord(
                    melody_index, time_ms, melody_key, current_root, quality, confidence,
                    _local_melody_interval(mapped_melody, melody_index),
                ))
                if previous_state != (current_root, quality):
                    inferred_count += 1
                previous_state = current_root, quality
        index = end
    key_name = f"{('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')[root]} {mode}"
    average_confidence = sum(anchor.confidence for anchor in anchors) / max(1, len(anchors))
    return anchors, inferred_count, key_name, key_confidence, average_confidence


def _keys_for_pitch_class(pitch_class: int) -> List[int]:
    keys = set()
    for octave in (24, 36, 48, 60, 72, 84):
        key, _adjusted = _pitch_to_key(octave + pitch_class)
        keys.add(key)
    return sorted(keys)


def _generated_key_for_pc(
    pitch_class: int, melody_key: int, target_distance: int, occupied: set[int], previous_key: Optional[int] = None,
) -> Optional[int]:
    choices = [
        key for key in _keys_for_pitch_class(pitch_class)
        if key not in occupied and 2 <= melody_key - key <= 10
    ]
    if not choices:
        return None
    return min(
        choices,
        key=lambda key: (
            abs((melody_key - key) - target_distance),
            abs(key - previous_key) if previous_key is not None else 0,
            key,
        ),
    )


def _generated_accompaniment_candidates(
    anchors: Sequence[_InferredChord], mapped_melody: Sequence[Tuple[SymbolicNote, int, int]], draft: QualityAnalysisDraft,
    shift: int, stats: Counter[str],
) -> List[_AccompanimentCandidate]:
    ranges = _bar_ranges(draft)
    candidates: List[_AccompanimentCandidate] = []
    previous_harmony_key: Optional[int] = None
    for anchor in anchors:
        note, _time, _melody_key = mapped_melody[anchor.melody_index]
        interval_pcs = CHORD_INTERVALS[anchor.quality]
        transposed_root = (anchor.root_pc + shift) % 12
        occupied: set[int] = set()
        root_key = _generated_key_for_pc(transposed_root, anchor.melody_key, 7, occupied)
        end_ms = max(note.end_ms, note.start_ms + anchor.local_interval_ms)
        if root_key is None:
            stats["filtered_unsafe_generated_chord_count"] += 1
        else:
            occupied.add(root_key)
            virtual = SymbolicNote(note.start_ms, end_ms, 48 + transposed_root, "inferred_chord", "bass")
            candidates.append(_AccompanimentCandidate(
                virtual, anchor.time_ms, root_key, _bar_index(ranges, anchor.time_ms), anchor.local_interval_ms,
                3.0 + 4.0 * anchor.confidence, "bass", anchor.time_ms, "inferred", anchor.confidence,
                anchor.local_interval_ms,
            ))
        harmony_options: List[Tuple[Tuple[int, int, int], int, int]] = []
        for order, interval in enumerate(interval_pcs[1:], start=1):
            pitch_class = (anchor.root_pc + interval + shift) % 12
            key = _generated_key_for_pc(pitch_class, anchor.melody_key, 4, occupied, previous_harmony_key)
            if key is not None:
                harmony_options.append(((abs((anchor.melody_key - key) - 4), abs(key - previous_harmony_key) if previous_harmony_key is not None else 0, order), key, pitch_class))
        if not harmony_options:
            stats["filtered_unsafe_generated_chord_count"] += 1
            continue
        _rank, harmony_key, harmony_pc = min(harmony_options)
        previous_harmony_key = harmony_key
        virtual = SymbolicNote(note.start_ms, end_ms, 48 + harmony_pc, "inferred_chord", "harmony")
        candidates.append(_AccompanimentCandidate(
            virtual, anchor.time_ms, harmony_key, _bar_index(ranges, anchor.time_ms), anchor.local_interval_ms,
            2.5 + 4.0 * anchor.confidence, "harmony", anchor.time_ms, "inferred", anchor.confidence,
            anchor.local_interval_ms,
        ))
    return candidates


def _combine_accompaniment_candidates(
    source: Sequence[_AccompanimentCandidate], generated: Sequence[_AccompanimentCandidate], stats: Counter[str],
) -> List[_AccompanimentCandidate]:
    unique: Dict[Tuple[int, int], _AccompanimentCandidate] = {}
    for candidate in [*source, *generated]:
        identity = candidate.time_ms, candidate.key_index
        current = unique.get(identity)
        if current is None or _candidate_rank(candidate) > _candidate_rank(current):
            if current is not None and current.provenance == "inferred":
                stats["filtered_unsafe_generated_chord_count"] += 1
            unique[identity] = candidate
        elif candidate.provenance == "inferred":
            stats["filtered_unsafe_generated_chord_count"] += 1
    return list(unique.values())


def _accompaniment_budgets(melody_count: int, anchor_count: int, preset: str) -> Tuple[int, int]:
    support_ratio = anchor_count / max(1, melody_count)
    auto_budget = min(melody_count // 3, int(melody_count * (.10 + .23 * min(1.0, support_ratio))))
    budgets = {
        "simple": 0,
        "standard": min(melody_count // 4, max(0, auto_budget - 1)),
        "auto": auto_budget,
        "full": melody_count // 2,
    }
    return budgets[preset], auto_budget


def _candidate_cooldown(candidate: _AccompanimentCandidate, preset: str) -> int:
    multiplier, low, high = {
        "standard": (3.0, 600, 1600),
        "auto": (2.0, 400, 1200),
        "full": (1.0, 250, 800),
    }[preset]
    return _clamp(candidate.local_interval_ms * multiplier, low, high)


def _select_budgeted_accompaniment(
    candidates: Sequence[_AccompanimentCandidate], locked: Dict[int, set[int]], options: TranscriptionOptions,
    preset: str, budget: int,
) -> List[_AccompanimentCandidate]:
    if not candidates or budget <= 0 or preset == "simple":
        return []
    groups: Dict[int, List[_AccompanimentCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.time_ms].append(candidate)
    ordered_times = sorted(groups)
    for time_ms in ordered_times:
        groups[time_ms].sort(key=_candidate_rank, reverse=True)
    selected: List[_AccompanimentCandidate] = []
    selected_by_time: Counter[int] = Counter()
    selected_anchor_times: List[int] = []
    used: set[Tuple[int, int, str]] = set()

    def add_from_group(time_ms: int) -> None:
        if len(selected) >= budget:
            return
        group = groups[time_ms]
        confidence = max((candidate.anchor_confidence for candidate in group), default=0.0)
        per_anchor = 1 if preset == "standard" else (2 if preset == "full" or confidence >= .75 else 1)
        capacity = max(0, int(options.max_polyphony) - len(locked.get(time_ms, set())))
        if selected_by_time[time_ms] >= min(per_anchor, capacity):
            return
        first_at_time = selected_by_time[time_ms] == 0
        if first_at_time:
            candidate = group[0]
            cooldown = _candidate_cooldown(candidate, preset)
            if any(abs(time_ms - existing) < cooldown for existing in selected_anchor_times):
                return
        for candidate in group:
            identity = candidate.time_ms, candidate.key_index, candidate.provenance
            if identity in used or selected_by_time[time_ms] >= min(per_anchor, capacity) or len(selected) >= budget:
                continue
            used.add(identity)
            selected.append(candidate)
            selected_by_time[time_ms] += 1
            if first_at_time:
                selected_anchor_times.append(time_ms)
                first_at_time = False

    desired_groups = min(len(ordered_times), budget if preset != "full" else int(math.ceil(budget / 2.0)))
    if desired_groups:
        indexes = {
            int(round(position * (len(ordered_times) - 1) / max(1, desired_groups - 1)))
            for position in range(desired_groups)
        }
        for index in sorted(indexes):
            add_from_group(ordered_times[index])
    # Fill any remaining budget by score, retaining the same per-anchor and
    # cooldown constraints.  Stable ordering makes repeated arrangements exact.
    remaining_times = sorted(ordered_times, key=lambda time_ms: (_candidate_rank(groups[time_ms][0]), -time_ms), reverse=True)
    made_progress = True
    while len(selected) < budget and made_progress:
        before = len(selected)
        for time_ms in remaining_times:
            add_from_group(time_ms)
            if len(selected) >= budget:
                break
        made_progress = len(selected) > before
    return selected


def _note_role_id(time_ms: int, key_index: int) -> str:
    return f"{int(time_ms)}:1Key{int(key_index)}"


def arrange_quality_analysis_with_roles(
    draft: QualityAnalysisDraft, options: TranscriptionOptions, forced_shift: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, str], str, int, Dict[str, object]]:
    """V7 melody-locking arrangement with source-backed and inferred harmony."""
    if any(note.source.startswith("vocal_evidence:") for note in draft.symbolic_notes):
        draft = replace(draft, symbolic_notes=[note for note in draft.symbolic_notes if not note.source.startswith("vocal_evidence:")])
    melody, segments = select_melody_with_segments(draft.symbolic_notes, draft.beat_times_ms)
    draft.lead_segments = segments
    shift = _best_shift(melody, options) if forced_shift is None else forced_shift
    mapped_melody, locked, melody_ids = _locked_melody(melody, draft, options, shift)
    filtering: Counter[str] = Counter()
    protected = draft.timing_confidence < _LOW_TIMING_THRESHOLD or draft.timing_backend in ("fixed_120_fallback", "fixed_grid_fallback")
    source_candidates = _safe_accompaniment_candidates(
        draft, shift, mapped_melody, locked, melody_ids, filtering, protected,
    )
    melody_count = sum(len(keys) for keys in locked.values())
    anchors, inferred_chord_count, harmonic_key, harmonic_key_confidence, chord_inference_confidence = _infer_harmonic_anchors(
        mapped_melody, draft, options,
    )
    # Inferred chords are a last-resort texture for an explicitly full score.
    # Auto/standard must use evidence from the original accompaniment or remain
    # sparse, rather than filling every vocal onset with invented harmony.
    generated_candidates = _generated_accompaniment_candidates(anchors, mapped_melody, draft, shift, filtering) if options.arrangement_preset == "full" else []
    candidates = _combine_accompaniment_candidates(source_candidates, generated_candidates, filtering)
    budget, auto_budget = _accompaniment_budgets(melody_count, len(anchors), options.arrangement_preset)
    accompaniment = _select_budgeted_accompaniment(candidates, locked, options, options.arrangement_preset, budget)

    rendered: Dict[Tuple[int, int], str] = {}
    for time_ms, keys in locked.items():
        for key_index in keys:
            rendered[(time_ms, key_index)] = "melody"
    for candidate in accompaniment:
        rendered.setdefault((candidate.time_ms, candidate.key_index), candidate.role)

    song_notes = [
        {"time": time_ms, "key": f"1Key{key_index}"}
        for time_ms, key_index in sorted(rendered)
    ]
    note_roles = {
        _note_role_id(time_ms, key_index): role
        for (time_ms, key_index), role in rendered.items()
    }
    grouped = Counter(int(note["time"]) for note in song_notes)
    accompaniment_count = sum(role != "melody" for role in rendered.values())
    source_accompaniment_count = sum(candidate.provenance == "source" for candidate in accompaniment)
    generated_accompaniment_count = sum(candidate.provenance == "inferred" for candidate in accompaniment)
    diagnostics = _diagnose_arrangement(draft, song_notes)
    stats: Dict[str, object] = {
        "arranged_note_count": len(song_notes), "onset_count": len(grouped),
        "chord_count": sum(count > 1 for count in grouped.values()),
        "average_polyphony": round(len(song_notes) / max(1, len(grouped)), 3),
        "melody_note_count": melody_count,
        "accompaniment_note_count": accompaniment_count,
        "melody_note_ratio": round(melody_count / max(1, len(song_notes)), 3),
        "qualitySymbolicCount": len(draft.symbolic_notes),
        "timingGridCount": len(draft.beat_times_ms), "timingConfidence": round(draft.timing_confidence, 3),
        "timingBackend": draft.timing_backend,
        "timingDiagnostics": dict(draft.timing_diagnostics),
        "leadSegmentCount": len(draft.lead_segments),
        "vocalLeadSegmentCount": sum(segment.source == "vocal" for segment in draft.lead_segments),
        "instrumentalLeadSegmentCount": sum(segment.source == "instrumental" for segment in draft.lead_segments),
        "quantization": draft.quantization,
        "filteredDrumCount": filtering["filtered_drum_count"],
        "filtered_unselected_melody_count": filtering["filtered_unselected_melody_count"],
        "reused_source_texture_count": filtering["reused_source_texture_count"],
        "filtered_dense_accompaniment_count": filtering["filtered_dense_accompaniment_count"],
        "filtered_short_accompaniment_count": filtering["filtered_short_accompaniment_count"],
        "filtered_duplicate_accompaniment_count": filtering["filtered_duplicate_accompaniment_count"],
        "filtered_melody_conflict_count": filtering["filtered_melody_conflict_count"],
        "filtered_other_count": filtering["filtered_other_count"],
        "filtered_low_score_accompaniment_count": filtering["filtered_low_score_accompaniment_count"],
        "filtered_unsafe_generated_chord_count": filtering["filtered_unsafe_generated_chord_count"],
        "low_timing_protection": protected,
        "harmonic_key": harmonic_key,
        "harmonic_key_confidence": round(harmonic_key_confidence, 3),
        "inferred_chord_count": inferred_chord_count,
        "harmonic_anchor_count": len(anchors),
        "source_accompaniment_count": source_accompaniment_count,
        "generated_accompaniment_count": generated_accompaniment_count,
        "chord_inference_confidence": round(chord_inference_confidence, 3),
        "accompaniment_budget": budget,
        "accompaniment_budget_used": accompaniment_count,
        "auto_accompaniment_budget": auto_budget,
        "unsnappedNoteCount": sum(not _quantize_with_status(note.start_ms, draft)[1] for note in draft.symbolic_notes if note.role != "drums"),
        "barDiagnostics": diagnostics,
        "suspiciousBarCount": sum(bool(item["suspicious"]) for item in diagnostics),
        "qualityArrangerVersion": 8,
    }
    return song_notes, note_roles, options.source_key or _key_for_shift(shift), shift, stats


def arrange_quality_analysis(
    draft: QualityAnalysisDraft, options: TranscriptionOptions, forced_shift: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], str, int, Dict[str, object]]:
    """Compatibility wrapper for callers that do not need rendered-note roles."""
    song_notes, _roles, key, shift, stats = arrange_quality_analysis_with_roles(draft, options, forced_shift)
    return song_notes, key, shift, stats
