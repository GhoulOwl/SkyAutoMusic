"""Private golden-set benchmark for V2/V3 15-key arrangements."""
from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .models import QualityAnalysisDraft, SymbolicNote, TranscriptionOptions, TranscriptionResult
from .pipeline import transcribe_draft
from .quality import arrange_quality_analysis


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
) -> Dict[str, Any]:
    windows = list(coverage_windows) if coverage_windows is not None else _coverage_windows(reference)
    estimate = _in_windows(estimate, windows)
    return {
        "key_onset": _f1(reference, estimate, require_key=True),
        "onset": _f1(reference, estimate, require_key=False),
        "melody": _f1(_melody(reference), _melody(estimate), require_key=True),
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


def _cache_path(cache_root: Path, song_id: str, options: TranscriptionOptions) -> Path:
    engine = options.engine if options.engine != "auto" else "fast"
    return cache_root / f"{song_id}-{engine}-{options.quality_model}.json"


def _save_quality_cache(path: Path, result: TranscriptionResult) -> None:
    if result.quality_analysis is None:
        return
    draft = result.quality_analysis
    payload = {"draft": {**asdict(draft), "symbolic_notes": [asdict(note) for note in draft.symbolic_notes]}, "song_notes": result.song_notes, "detected_key": result.detected_key, "semitone_shift": result.semitone_shift, "stats": result.stats}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_quality_cache(path: Path, options: TranscriptionOptions) -> TranscriptionResult | None:
    if not path.is_file():
        return None


def _offset_score(notes: Sequence[Dict[str, Any]], offset_ms: int) -> List[Dict[str, Any]]:
    if not offset_ms:
        return list(notes)
    return [{**note, "time": int(note["time"]) + offset_ms} for note in notes]
    try:
        payload = json.loads(path.read_text(encoding="utf-8")); value = payload["draft"]
        value["symbolic_notes"] = [SymbolicNote(**note) for note in value["symbolic_notes"]]
        draft = QualityAnalysisDraft(**value)
        notes, key, shift, stats = arrange_quality_analysis(draft, options)
        return TranscriptionResult([], notes, key, draft.bpm, stats, [], engine="arrangement_v3_quality", semitone_shift=shift, beat_times_ms=draft.beat_times_ms, options=options, quality_analysis=draft)
    except Exception:
        return None


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
        song_id = str(song.get("id", audio.stem)); cache = _cache_path(cache_root, song_id, options)
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
            row["metrics"] = evaluate_15_key(hand_score, result.song_notes, windows)
        rows.append(row)
    metric_rows = [row["metrics"] for row in rows if "metrics" in row]
    summary: Dict[str, Any] = {"count": len(rows), "scored_count": len(metric_rows)}
    if metric_rows:
        for name, path in (("key_onset_f1", ("key_onset", "f1")), ("onset_f1", ("onset", "f1")), ("melody_f1", ("melody", "f1"))):
            summary[f"median_{name}"] = round(median(float(row[path[0]][path[1]]) for row in metric_rows), 4)
        validation = [row["metrics"] for row in rows if row.get("split") == "validation" and "metrics" in row]
        if validation:
            summary["validation_count"] = len(validation)
            for name, path in (("key_onset_f1", ("key_onset", "f1")), ("onset_f1", ("onset", "f1")), ("melody_f1", ("melody", "f1"))):
                summary[f"validation_median_{name}"] = round(median(float(row[path[0]][path[1]]) for row in validation), 4)
    payload = {"engine": "v3_benchmark", "summary": summary, "results": rows}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
