"""Persistent, versioned storage for transcription dialog drafts."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    AnalysisDraft,
    ChordSpan,
    LeadSegment,
    MelodyNote,
    NoteEvent,
    QualityAnalysisDraft,
    Section,
    SourceMetadata,
    SymbolicNote,
    TempoMap,
    TranscriptionOptions,
    TranscriptionResult,
    VocalEvidence,
)


DRAFT_SCHEMA_VERSION = 1
MANIFEST_NAME = "draft.json"


@dataclass
class DraftRecord:
    id: str
    name: str
    source_path: str
    result: TranscriptionResult
    source_kind: str = "external"
    source_filename: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def source_available(self) -> bool:
        return bool(self.source_path and os.path.isfile(self.source_path))


def _result_to_dict(result: TranscriptionResult) -> Dict[str, Any]:
    analysis = None
    if result.analysis is not None:
        value = result.analysis
        analysis = {
            "duration_sec": value.duration_sec,
            "tempo_map": {
                "bpm": value.tempo_map.bpm, "meter": value.tempo_map.meter,
                "beat_times_ms": list(value.tempo_map.beat_times_ms),
                "bar_starts_ms": list(value.tempo_map.bar_starts_ms),
                "grid_times_ms": list(value.tempo_map.grid_times_ms),
                "confidence": value.tempo_map.confidence,
            },
            "detected_key": value.detected_key, "key_confidence": value.key_confidence,
            "semitone_shift": value.semitone_shift,
            "melody": [item.__dict__ for item in value.melody],
            "chords": [item.__dict__ for item in value.chords],
            "sections": [item.__dict__ for item in value.sections],
            "lead_source": value.lead_source, "melody_confidence": value.melody_confidence,
            "harmony_confidence": value.harmony_confidence,
            "structure_confidence": value.structure_confidence,
            "used_mix_fallback": value.used_mix_fallback,
        }
    quality = None
    if result.quality_analysis is not None:
        value = result.quality_analysis
        quality = {
            "duration_sec": value.duration_sec,
            "symbolic_notes": [item.__dict__ for item in value.symbolic_notes],
            "beat_times_ms": list(value.beat_times_ms), "downbeat_times_ms": list(value.downbeat_times_ms),
            "bar_starts_ms": list(value.bar_starts_ms), "bpm": value.bpm, "meter": value.meter,
            "timing_confidence": value.timing_confidence, "model_name": value.model_name,
            "device": value.device, "timing_backend": value.timing_backend,
            "quantization": value.quantization,
            "refined_regions": [list(item) for item in value.refined_regions],
            "timing_diagnostics": dict(value.timing_diagnostics),
            "lead_segments": [item.__dict__ for item in value.lead_segments],
            "vocal_evidence": [item.__dict__ for item in value.vocal_evidence],
        }
    source = None
    if result.source is not None:
        source = {
            "platform": result.source.platform, "title": result.source.title,
            "artists": list(result.source.artists), "source_id": result.source.source_id,
            "webpage_url": result.source.webpage_url, "display_name": result.source.display_name,
        }
    return {
        "events": [item.__dict__ for item in result.events], "song_notes": list(result.song_notes),
        "detected_key": result.detected_key, "bpm": result.bpm, "stats": result.stats,
        "warnings": list(result.warnings), "engine": result.engine, "source_file": result.source_file,
        "semitone_shift": result.semitone_shift, "octave_shift": result.octave_shift,
        "beat_times_ms": list(result.beat_times_ms), "options": result.options.__dict__,
        "source": source, "analysis": analysis, "quality_analysis": quality,
        "separation_model": result.separation_model, "note_roles": dict(result.note_roles),
    }


def _result_from_dict(value: Dict[str, Any]) -> TranscriptionResult:
    source_value = value.get("source")
    source = None if source_value is None else SourceMetadata(
        platform=source_value.get("platform", "local"), title=str(source_value.get("title", "")),
        artists=tuple(source_value.get("artists", [])), source_id=str(source_value.get("source_id", "")),
        webpage_url=str(source_value.get("webpage_url", "")), display_name=str(source_value.get("display_name", "")),
    )
    analysis_value = value.get("analysis")
    analysis = None
    if analysis_value is not None:
        tempo = analysis_value["tempo_map"]
        analysis = AnalysisDraft(
            duration_sec=float(analysis_value["duration_sec"]),
            tempo_map=TempoMap(float(tempo["bpm"]), tempo["meter"], tuple(tempo["beat_times_ms"]), tuple(tempo["bar_starts_ms"]), tuple(tempo["grid_times_ms"]), float(tempo["confidence"])),
            detected_key=str(analysis_value["detected_key"]), key_confidence=float(analysis_value["key_confidence"]),
            semitone_shift=int(analysis_value["semitone_shift"]),
            melody=[MelodyNote(**item) for item in analysis_value.get("melody", [])],
            chords=[ChordSpan(**item) for item in analysis_value.get("chords", [])],
            sections=[Section(**item) for item in analysis_value.get("sections", [])],
            lead_source=analysis_value["lead_source"], melody_confidence=float(analysis_value["melody_confidence"]),
            harmony_confidence=float(analysis_value["harmony_confidence"]), structure_confidence=float(analysis_value["structure_confidence"]),
            used_mix_fallback=bool(analysis_value.get("used_mix_fallback", False)),
        )
    quality_value = value.get("quality_analysis")
    quality = None
    if quality_value is not None:
        quality = QualityAnalysisDraft(
            duration_sec=float(quality_value["duration_sec"]),
            symbolic_notes=[SymbolicNote(**item) for item in quality_value.get("symbolic_notes", [])],
            beat_times_ms=[int(item) for item in quality_value.get("beat_times_ms", [])],
            downbeat_times_ms=[int(item) for item in quality_value.get("downbeat_times_ms", [])],
            bar_starts_ms=[int(item) for item in quality_value.get("bar_starts_ms", [])],
            bpm=float(quality_value["bpm"]), meter=quality_value["meter"],
            timing_confidence=float(quality_value["timing_confidence"]), model_name=str(quality_value["model_name"]),
            device=str(quality_value["device"]), timing_backend=str(quality_value.get("timing_backend", "beat_this")),
            quantization=str(quality_value.get("quantization", "adaptive_8th_triplet_16th")),
            refined_regions=[tuple(map(int, item)) for item in quality_value.get("refined_regions", [])],
            timing_diagnostics=dict(quality_value.get("timing_diagnostics", {})),
            lead_segments=[LeadSegment(**item) for item in quality_value.get("lead_segments", [])],
            vocal_evidence=[VocalEvidence(**item) for item in quality_value.get("vocal_evidence", [])],
        )
    return TranscriptionResult(
        events=[NoteEvent(**item) for item in value.get("events", [])], song_notes=list(value.get("song_notes", [])),
        detected_key=str(value["detected_key"]), bpm=float(value["bpm"]), stats=dict(value.get("stats", {})),
        warnings=list(value.get("warnings", [])), engine=str(value.get("engine", "arrangement_v2")),
        source_file=str(value.get("source_file", "")), semitone_shift=int(value.get("semitone_shift", 0)),
        octave_shift=int(value.get("octave_shift", 0)), beat_times_ms=[int(item) for item in value.get("beat_times_ms", [])],
        options=TranscriptionOptions(**dict(value.get("options", {}))), source=source, analysis=analysis,
        quality_analysis=quality, separation_model=str(value.get("separation_model", "")),
        note_roles={str(key): str(role) for key, role in dict(value.get("note_roles", {})).items()},
    )


class DraftStore:
    """A draft is one self-contained manifest plus an optional managed source file."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()

    @staticmethod
    def new_record(name: str, source_path: str, result: TranscriptionResult, record_id: Optional[str] = None) -> DraftRecord:
        now = time.time()
        return DraftRecord(record_id or uuid.uuid4().hex, name, os.path.abspath(source_path), result, created_at=now, updated_at=now)

    def _entry_dir(self, record_id: str) -> Path:
        try:
            uuid.UUID(hex=record_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("invalid draft id") from exc
        path = (self.root / record_id).resolve()
        if path.parent != self.root:
            raise ValueError("invalid draft location")
        return path

    @staticmethod
    def _clean_name(name: str) -> str:
        value = " ".join(str(name or "").split())
        if not value:
            raise ValueError("草稿名称不能为空")
        return value[:180]

    def _manifest_payload(self, record: DraftRecord) -> Dict[str, Any]:
        source: Dict[str, Any] = {"kind": record.source_kind}
        if record.source_kind == "managed":
            source["filename"] = record.source_filename
        else:
            source["path"] = record.source_path
        return {
            "schemaVersion": DRAFT_SCHEMA_VERSION, "id": record.id, "name": record.name,
            "createdAt": record.created_at, "updatedAt": record.updated_at,
            "source": source, "result": _result_to_dict(record.result),
        }

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".draft-", suffix=".tmp", dir=path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _copy_source_atomic(source: str, target: Path) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".source-", suffix=".tmp", dir=target.parent)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def upsert(self, record: DraftRecord, managed_source: Optional[str] = None) -> DraftRecord:
        record.name = self._clean_name(record.name)
        entry = self._entry_dir(record.id)
        entry.mkdir(parents=True, exist_ok=True)
        if managed_source is not None:
            source = os.path.abspath(managed_source)
            if not os.path.isfile(source):
                raise OSError(f"暂存音频不存在：{source}")
            suffix = Path(source).suffix.lower() or ".audio"
            filename = f"source{suffix}"
            target = entry / filename
            self._copy_source_atomic(source, target)
            record.source_kind = "managed"
            record.source_filename = filename
            record.source_path = str(target.resolve())
        record.updated_at = time.time()
        self._write_json_atomic(entry / MANIFEST_NAME, self._manifest_payload(record))
        return record

    def rename(self, record: DraftRecord, name: str) -> DraftRecord:
        record.name = self._clean_name(name)
        return self.upsert(record)

    def delete(self, record: DraftRecord) -> None:
        entry = self._entry_dir(record.id)
        if entry.exists():
            shutil.rmtree(entry)

    def load_all(self) -> tuple[List[DraftRecord], List[str]]:
        if not self.root.exists():
            return [], []
        records: List[DraftRecord] = []
        warnings: List[str] = []
        try:
            entries: Iterable[Path] = list(self.root.iterdir())
        except OSError as exc:
            return [], [f"无法读取暂存草稿：{exc}"]
        for entry in entries:
            if not entry.is_dir():
                continue
            manifest = entry / MANIFEST_NAME
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if payload.get("schemaVersion") != DRAFT_SCHEMA_VERSION:
                    raise ValueError("草稿版本不受支持")
                record_id = str(payload["id"])
                if entry.resolve() != self._entry_dir(record_id):
                    raise ValueError("草稿目录不匹配")
                source = dict(payload.get("source", {}))
                kind = str(source.get("kind", "external"))
                if kind == "managed":
                    filename = os.path.basename(str(source["filename"]))
                    source_path = str((entry / filename).resolve())
                elif kind == "external":
                    filename, source_path = "", os.path.abspath(str(source.get("path", "")))
                else:
                    raise ValueError("未知音频来源")
                records.append(DraftRecord(
                    id=record_id, name=self._clean_name(payload["name"]), source_path=source_path,
                    result=_result_from_dict(dict(payload["result"])), source_kind=kind,
                    source_filename=filename, created_at=float(payload.get("createdAt", 0)),
                    updated_at=float(payload.get("updatedAt", 0)),
                ))
            except Exception as exc:
                warnings.append(f"已跳过损坏草稿 {entry.name}：{exc}")
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records, warnings
