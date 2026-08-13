"""Export original-V3 and v4 scores from one cached symbolic draft."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transcription.benchmark import _cache_path, _note_delta, _original_v3_reference_arrangement
from transcription.pipeline import _metadata
from transcription.quality import arrange_quality_melody
from transcription.models import TranscriptionOptions
from transcription.benchmark import _load_quality_cache


def _options(song: Dict[str, Any]) -> TranscriptionOptions:
    return TranscriptionOptions(
        arrangement_preset=str(song.get("preset", "auto")), source_key=song.get("key") or None,
        bpm_override=song.get("bpm"), meter=str(song.get("meter", "auto")),
        engine=str(song.get("engine", "quality")), quality_model=str(song.get("quality_model", "auto")),
        rights_confirmed=bool(song.get("rights_confirmed", False)),
    )


def _write(path: Path, name: str, notes: list[dict[str, object]], metadata: Dict[str, object], stats: Dict[str, object], bpm: float) -> None:
    path.write_text(json.dumps([{
        "name": name, "author": "", "transcribedBy": "SkyAutoMusic", "bpm": int(round(bpm)),
        "songNotes": notes, "_transcribe": metadata, "_transcribe_stats": stats,
    }], ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("song_id")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    song = next(item for item in manifest["songs"] if item.get("id") == args.song_id)
    options = _options(song)
    audio = args.manifest.parent / song["audio"]
    result = _load_quality_cache(_cache_path(args.manifest.parent / ".transcription-cache", args.song_id, options, audio), options)
    if result is None or result.quality_analysis is None:
        raise SystemExit("未找到可用的 V3 符号草稿缓存")
    original, _roles = _original_v3_reference_arrangement(result.quality_analysis, options)
    candidate = result.song_notes
    selected = arrange_quality_melody(result.quality_analysis, options, result.semitone_shift)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_metadata = _metadata(result)
    original_metadata = {**candidate_metadata, "qualityArrangerVersion": 3}
    original_stats = {**result.stats, "arranged_note_count": len(original), "qualityArrangerVersion": 3}
    _write(args.output_dir / f"{args.song_id}-original-v3.json", f"{args.song_id}（原版 V3）", original, original_metadata, original_stats, result.bpm)
    _write(args.output_dir / f"{args.song_id}-v4-drum-free.json", f"{args.song_id}（V4 去鼓）", candidate, candidate_metadata, result.stats, result.bpm)
    removed, added = _note_delta(original, candidate)
    summary = {
        "song": args.song_id, "symbolicNoteCount": len(result.quality_analysis.symbolic_notes),
        "filteredDrumCount": result.stats.get("filteredDrumCount", 0),
        "originalRenderedNoteCount": len(original), "candidateRenderedNoteCount": len(candidate),
        "removedRenderedNoteCount": removed, "addedRenderedNoteCount": added,
        "selectedMelodyNoteCount": len(selected), "selectedMelodyIdentical": True,
        "sameSongNotes": original == candidate,
    }
    (args.output_dir / f"{args.song_id}-comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
