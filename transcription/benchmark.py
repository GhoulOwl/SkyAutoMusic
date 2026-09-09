"""Private golden-set benchmark for V2/V3 15-key arrangements."""
from __future__ import annotations

import bisect
import json
import hashlib
import importlib.metadata
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .models import LeadSegment, QualityAnalysisDraft, SymbolicNote, TranscriptionOptions, TranscriptionResult
from .pipeline import transcribe_draft
from .arranger import SKY_MIDI, _pitch_to_key
from .quality import _adaptive_quantize, _best_shift, arrange_quality_analysis_with_roles, arrange_quality_melody, select_melody


_V4_LEAD_PRIORITY = {
    "voice": 8, "synth_lead": 7, "violin": 6, "flutes": 6,
    "soprano_and_alto_sax": 5, "tenor_sax": 5, "clarinet": 5,
    "oboe": 5, "trumpet": 5, "acoustic_piano": 4, "electric_piano": 4,
}


def _read_score_payload(path: Path) -> Any:
    """Read exported Sky JSON from both modern UTF-8 and legacy UTF-16 tools."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"无法读取 Sky JSON（仅支持 UTF-8 / UTF-16）：{path}")


def _score_notes(path: Path) -> List[Dict[str, Any]]:
    payload = _read_score_payload(path)
    song = payload[0] if isinstance(payload, list) else payload
    return list(song.get("songNotes", []))


def _key_index(note: Dict[str, Any]) -> int:
    return int(str(note["key"])[4:])


def _pairs(reference: Sequence[Dict[str, Any]], estimate: Sequence[Dict[str, Any]], tolerance_ms: int, require_key: bool) -> List[Tuple[int, int]]:
    """One-to-one minimum-distance matches, equivalent in spirit to mir_eval."""
    choices = []
    for left, ref in enumerate(reference):
        for right, got in enumerate(estimate):
            delta = abs(int(ref["time"]) - int(got["time"]))
            if delta <= tolerance_ms and (not require_key or _key_index(ref) == _key_index(got)):
                choices.append((delta, left, right))
    used_ref, used_est, result = set(), set(), []
    for _delta, left, right in sorted(choices):
        if left not in used_ref and right not in used_est:
            used_ref.add(left); used_est.add(right); result.append((left, right))
    return result


def _f1(reference: Sequence[Dict[str, Any]], estimate: Sequence[Dict[str, Any]], tolerance_ms: int = 60, require_key: bool = True) -> Dict[str, float]:
    matches = _pairs(reference, estimate, tolerance_ms, require_key)
    precision = len(matches) / len(estimate) if estimate else (1.0 if not reference else 0.0)
    recall = len(matches) / len(reference) if reference else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    mae = median([abs(int(reference[left]["time"]) - int(estimate[right]["time"])) for left, right in matches]) if matches else None
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "matched": len(matches), "onset_mae_ms": mae}


def _melody(notes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for note in notes:
        time_ms = int(note["time"])
        if time_ms not in grouped or _key_index(note) > _key_index(grouped[time_ms]):
            grouped[time_ms] = note
    return [grouped[key] for key in sorted(grouped)]


def _coverage_windows(reference: Sequence[Dict[str, Any]], gap_ms: int = 5000) -> List[Tuple[int, int]]:
    """Infer annotated intervals, so an incomplete hand score is never a false negative."""
    onsets = sorted({int(note["time"]) for note in reference})
    if not onsets:
        return []
    windows: List[Tuple[int, int]] = []
    start = previous = onsets[0]
    for current in onsets[1:]:
        if current - previous > gap_ms:
            windows.append((start, previous))
            start = current
        previous = current
    windows.append((start, previous))
    return windows


def _in_windows(notes: Sequence[Dict[str, Any]], windows: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    if not windows:
        return list(notes)
    return [note for note in notes if any(start <= int(note["time"]) <= end for start, end in windows)]


def evaluate_15_key(
    reference: Sequence[Dict[str, Any]], estimate: Sequence[Dict[str, Any]],
    coverage_windows: Sequence[Tuple[int, int]] | None = None,
    selected_melody: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    windows = list(coverage_windows) if coverage_windows is not None else _coverage_windows(reference)
    if coverage_windows is not None and not windows:
        reference, estimate = [], []
        selected_melody = [] if selected_melody is not None else None
    else:
        reference = _in_windows(reference, windows)
        estimate = _in_windows(estimate, windows)
        selected_melody = _in_windows(selected_melody, windows) if selected_melody is not None else None
    top_voice = _f1(_melody(reference), _melody(estimate), require_key=True)
    selected = _f1(_melody(reference), selected_melody, require_key=True) if selected_melody is not None else top_voice
    return {
        "key_onset": _f1(reference, estimate, require_key=True),
        "onset": _f1(reference, estimate, require_key=False),
        "top_voice": top_voice,
        # Kept for dashboards written before the selected-melody metric existed.
        "melody": top_voice,
        "selected_melody": selected,
        "reference_note_count": len(reference), "estimate_note_count": len(estimate),
        "max_polyphony": max(CounterLike(estimate).values(), default=0),
        "coverage_windows_ms": [list(window) for window in windows],
    }


class CounterLike(dict):
    """Tiny dependency-free counter for score onsets."""
    def __init__(self, notes: Iterable[Dict[str, Any]]):
        super().__init__()
        for note in notes:
            time_ms = int(note["time"]); self[time_ms] = self.get(time_ms, 0) + 1


def _audio_fingerprint(path: Path) -> str:
    """Stable content identity; file names and mtimes are not cache identities."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _quality_model_version() -> str:
    try:
        return importlib.metadata.version("muscriptor")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cache_path(cache_root: Path, song_id: str, options: TranscriptionOptions, audio: Path) -> Path:
    engine = options.engine if options.engine != "auto" else "fast"
    model_spec = f"{options.quality_model}-{_quality_model_version()}"
    return cache_root / f"{song_id}-{engine}-{model_spec}-{_audio_fingerprint(audio)}.json"


def _save_quality_cache(path: Path, result: TranscriptionResult) -> None:
    if result.quality_analysis is None:
        return
    draft = result.quality_analysis
    payload = {
        "cacheVersion": 4,
        "modelName": draft.model_name,
        "modelVersion": _quality_model_version(),
        # Only the expensive symbolic analysis belongs in the cache.  Current
        # arrangement options are intentionally reapplied on every cache hit.
        "draft": {**asdict(draft), "symbolic_notes": [asdict(note) for note in draft.symbolic_notes]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_quality_cache(path: Path, options: TranscriptionOptions) -> TranscriptionResult | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cacheVersion") != 4 or payload.get("modelVersion") != _quality_model_version():
            return None
        value = payload["draft"]
        value["symbolic_notes"] = [SymbolicNote(**note) for note in value["symbolic_notes"]]
        value["lead_segments"] = [LeadSegment(**segment) for segment in value.get("lead_segments", [])]
        draft = QualityAnalysisDraft(**value)
        notes, note_roles, key, shift, stats = arrange_quality_analysis_with_roles(draft, options)
        return TranscriptionResult(
            [], notes, key, draft.bpm, stats, [], engine="arrangement_v3_quality",
            semitone_shift=shift, beat_times_ms=draft.beat_times_ms, options=options,
            quality_analysis=draft, note_roles=note_roles,
        )
    except Exception:
        return None


def _offset_score(notes: Sequence[Dict[str, Any]], offset_ms: int) -> List[Dict[str, Any]]:
    if not offset_ms:
        return list(notes)
    return [{**note, "time": int(note["time"]) + offset_ms} for note in notes]


def _split_windows(windows: Sequence[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
    """Return all/front/back score spans without crossing annotation gaps."""
    if not windows:
        return {"all": [], "front": [], "back": []}
    left, right = min(start for start, _end in windows), max(end for _start, end in windows)
    middle = (left + right) // 2

    def intersect(start: int, end: int) -> List[Tuple[int, int]]:
        return [(max(start, lo), min(end, hi)) for lo, hi in windows if max(start, lo) <= min(end, hi)]

    return {"all": list(windows), "front": intersect(left, middle), "back": intersect(middle + 1, right)}


def _select_melody_v4(notes: Sequence[SymbolicNote], beats: Sequence[int]) -> List[SymbolicNote]:
    """Frozen V4 lead selector used only as a benchmark baseline."""
    candidates = [note for note in notes if note.role == "melody" and 45 <= note.midi_pitch <= 100]
    if not candidates:
        candidates = [note for note in notes if note.role != "drums" and 54 <= note.midi_pitch <= 96]
    grouped: Dict[int, List[SymbolicNote]] = defaultdict(list)
    for note in candidates:
        onset = int(round(note.start_ms / 25.0) * 25) if len(beats) >= 2 else note.start_ms
        grouped[onset].append(note)
    windows: Dict[int, List[List[SymbolicNote]]] = defaultdict(list)
    for onset in sorted(grouped):
        beat_index = max(0, bisect.bisect_right(beats, onset) - 1) if beats else onset // 4000
        row = sorted(grouped[onset], key=lambda note: (_V4_LEAD_PRIORITY.get(note.instrument, 0), note.end_ms - note.start_ms), reverse=True)[:8]
        windows[beat_index // 8].append(row)
    selected: List[SymbolicNote] = []
    previous: SymbolicNote | None = None
    for window in (windows[index] for index in sorted(windows)):
        scores: List[List[float]] = []
        links: List[List[int]] = []
        for row_index, row in enumerate(window):
            row_scores: List[float] = []
            row_links: List[int] = []
            for current in row:
                value = 100.0 + 12.0 * _V4_LEAD_PRIORITY.get(current.instrument, 0) - abs(current.midi_pitch - 74) * .6
                if row_index == 0:
                    if previous:
                        value -= abs(current.midi_pitch - previous.midi_pitch) * .55
                        if current.instrument != previous.instrument:
                            value -= 7.0
                        if current.start_ms - previous.end_ms > 1400:
                            value -= 3.0
                    row_scores.append(value)
                    row_links.append(-1)
                    continue
                choices: List[float] = []
                for prior_index, prior in enumerate(window[row_index - 1]):
                    transition = 100.0 + 12.0 * _V4_LEAD_PRIORITY.get(current.instrument, 0) - abs(current.midi_pitch - 74) * .6
                    transition -= abs(current.midi_pitch - prior.midi_pitch) * .55
                    if current.instrument != prior.instrument:
                        transition -= 7.0
                    if current.start_ms - prior.end_ms > 1400:
                        transition -= 3.0
                    choices.append(scores[-1][prior_index] + transition)
                best = max(range(len(choices)), key=choices.__getitem__)
                row_scores.append(choices[best])
                row_links.append(best)
            scores.append(row_scores)
            links.append(row_links)
        if not scores:
            continue
        index = max(range(len(scores[-1])), key=scores[-1].__getitem__)
        chosen: List[SymbolicNote] = []
        for row_index in range(len(window) - 1, -1, -1):
            chosen.append(window[row_index][index])
            index = links[row_index][index]
            if index < 0:
                break
        for current in reversed(chosen):
            if previous and current.start_ms < previous.end_ms and current.midi_pitch != previous.midi_pitch:
                continue
            selected.append(current)
            previous = current
    return selected


def _original_v3_reference_arrangement(draft: QualityAnalysisDraft, options: TranscriptionOptions) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, str], str]]:
    """Historical drum baseline using the frozen V4 lead selector."""
    melody = _select_melody_v4(draft.symbolic_notes, draft.beat_times_ms)
    shift = _best_shift(melody, options)
    melody_ids = {(note.start_ms, note.end_ms, note.midi_pitch, note.instrument) for note in melody}
    by_time: Dict[int, Dict[int, Tuple[int, str]]] = defaultdict(dict)
    previous_source = previous_sky = None

    def add(time_ms: int, pitch: int, priority: int, role: str) -> None:
        nonlocal previous_source, previous_sky
        index, _adjusted = _pitch_to_key(pitch + shift, previous_source, previous_sky)
        old = by_time[time_ms].get(index)
        if old is None or priority > old[0]:
            by_time[time_ms][index] = (priority, role)
        if role == "melody":
            previous_source, previous_sky = pitch + shift, SKY_MIDI[index]

    for note in melody:
        add(_adaptive_quantize(note.start_ms, draft), note.midi_pitch + 12 * (options.melody_octave_shift or 0), 100, "melody")
    density = {"simple": 4, "standard": 2, "full": 1, "auto": 2}[options.arrangement_preset]
    # Historical predicate: it accidentally let drums through as accompaniment.
    accompaniment = [note for note in draft.symbolic_notes if note.role not in ("drums", "melody") or (note.start_ms, note.end_ms, note.midi_pitch, note.instrument) not in melody_ids]
    for ordinal, note in enumerate(accompaniment):
        if ordinal % density:
            continue
        add(_adaptive_quantize(note.start_ms, draft), note.midi_pitch, 40 if note.role == "bass" else 20, note.role)
    output: List[Dict[str, object]] = []
    output_roles: Dict[Tuple[int, str], str] = {}
    for time_ms in sorted(by_time):
        selected = sorted(by_time[time_ms].items(), key=lambda item: (-item[1][0], item[0]))[: options.max_polyphony]
        for key_index, (_priority, role) in sorted(selected):
            key = f"1Key{key_index}"
            output.append({"time": int(time_ms), "key": key})
            output_roles[(int(time_ms), key)] = role
    return output, output_roles


def _note_delta(before: Sequence[Dict[str, object]], after: Sequence[Dict[str, object]]) -> Tuple[int, int]:
    """Return removed and added rendered note counts, preserving duplicate notes."""
    key = lambda note: (int(note["time"]), str(note["key"]))
    old, new = Counter(map(key, before)), Counter(map(key, after))
    return sum((old - new).values()), sum((new - old).values())


def run_benchmark(manifest_path: Path, output_path: Path) -> Dict[str, Any]:
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    songs = manifest.get("songs", manifest if isinstance(manifest, list) else [])
    cache_root = output_path.parent / ".transcription-cache"
    rows: List[Dict[str, Any]] = []
    for song in songs:
        options = TranscriptionOptions(
            arrangement_preset=str(song.get("preset", "auto")), source_key=song.get("key") or None,
            bpm_override=song.get("bpm"), meter=str(song.get("meter", "auto")),
            engine=str(song.get("engine", "fast")), quality_model=str(song.get("quality_model", "auto")),
            rights_confirmed=bool(song.get("rights_confirmed", False)),
        )
        audio = Path(song["audio"]); audio = audio if audio.is_absolute() else manifest_path.parent / audio
        song_id = str(song.get("id", audio.stem)); cache = _cache_path(cache_root, song_id, options, audio)
        result = _load_quality_cache(cache, options) if options.engine == "quality" else None
        cache_hit = result is not None
        tracemalloc.start(); started = time.perf_counter()
        if result is None:
            result = transcribe_draft(str(audio), options)
            if options.engine == "quality": _save_quality_cache(cache, result)
        _current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        row: Dict[str, Any] = {
            "id": song_id, "engine": result.engine, "key": result.detected_key, "bpm": result.bpm,
            "split": str(song.get("split", "tune")),
            "noteCount": len(result.song_notes), "stats": result.stats, "warnings": result.warnings,
            "elapsed_sec": round(time.perf_counter() - started, 3), "peak_memory_mb": round(peak / 1024 / 1024, 2),
            "cache_hit": cache_hit,
        }
        reference = song.get("reference") or song.get("score")
        if reference:
            score = Path(reference); score = score if score.is_absolute() else manifest_path.parent / score
            hand_score = _offset_score(_score_notes(score), int(song.get("reference_offset_ms", 0)))
            raw_windows = song.get("reference_windows_ms")
            windows = [tuple(map(int, window)) for window in raw_windows] if raw_windows else _coverage_windows(hand_score)
            selected = arrange_quality_melody(result.quality_analysis, options, result.semitone_shift) if result.quality_analysis else None
            row["window_metrics"] = {
                name: evaluate_15_key(hand_score, result.song_notes, part, selected)
                for name, part in _split_windows(windows).items()
            }
            row["metrics"] = row["window_metrics"]["all"]
            if result.quality_analysis is not None:
                original_notes, original_roles = _original_v3_reference_arrangement(result.quality_analysis, options)
                row["original_v3_window_metrics"] = {
                    name: evaluate_15_key(hand_score, original_notes, part, selected)
                    for name, part in _split_windows(windows).items()
                }
                removed, added = _note_delta(original_notes, result.song_notes)
                note_key = lambda note: (int(note["time"]), str(note["key"]))
                removed_entries = Counter(map(note_key, original_notes)) - Counter(map(note_key, result.song_notes))
                non_drum_removed = sum(count for key, count in removed_entries.items() if original_roles.get(key) != "drums")
                row["v3_regression"] = {
                    "originalNoteCount": len(original_notes), "candidateNoteCount": len(result.song_notes),
                    "removedRenderedNoteCount": removed, "addedRenderedNoteCount": added,
                    "drumDerivedRemovedRenderedNoteCount": removed - non_drum_removed,
                    "nonDrumRemovedRenderedNoteCount": non_drum_removed,
                    "selectedMelodyIdentical": (
                        _select_melody_v4(result.quality_analysis.symbolic_notes, result.quality_analysis.beat_times_ms)
                        == select_melody(result.quality_analysis.symbolic_notes, result.quality_analysis.beat_times_ms)
                    ),
                    "filteredDrumCount": result.stats.get("filteredDrumCount", 0),
                }
        rows.append(row)
    metric_rows = [row["metrics"] for row in rows if "metrics" in row]
    summary: Dict[str, Any] = {"count": len(rows), "scored_count": len(metric_rows)}
    if metric_rows:
        for name, path in (("key_onset_f1", ("key_onset", "f1")), ("onset_f1", ("onset", "f1")), ("top_voice_f1", ("top_voice", "f1")), ("selected_melody_f1", ("selected_melody", "f1")), ("melody_f1", ("melody", "f1"))):
            summary[f"median_{name}"] = round(median(float(row[path[0]][path[1]]) for row in metric_rows), 4)
        validation = [row["metrics"] for row in rows if row.get("split") == "validation" and "metrics" in row]
        if validation:
            summary["validation_count"] = len(validation)
            for name, path in (("key_onset_f1", ("key_onset", "f1")), ("onset_f1", ("onset", "f1")), ("top_voice_f1", ("top_voice", "f1")), ("selected_melody_f1", ("selected_melody", "f1")), ("melody_f1", ("melody", "f1"))):
                summary[f"validation_median_{name}"] = round(median(float(row[path[0]][path[1]]) for row in validation), 4)
    payload = {"engine": "v3_benchmark", "summary": summary, "results": rows}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
