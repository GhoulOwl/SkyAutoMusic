"""Private V2 golden-set runner (one arrangement route, no AI candidates)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .models import TranscriptionOptions
from .pipeline import transcribe_draft


def run_benchmark(manifest_path: Path, output_path: Path) -> Dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    songs = manifest.get("songs", manifest if isinstance(manifest, list) else [])
    rows: List[Dict[str, Any]] = []
    for song in songs:
        options = TranscriptionOptions(
            arrangement_preset=str(song.get("preset", "auto")),
            source_key=song.get("key") or None,
            bpm_override=song.get("bpm"), meter=str(song.get("meter", "auto")),
        )
        result = transcribe_draft(str(song["audio"]), options)
        rows.append({
            "id": song.get("id", Path(song["audio"]).stem), "engine": result.engine,
            "key": result.detected_key, "bpm": result.bpm,
            "noteCount": len(result.song_notes), "stats": result.stats,
            "warnings": result.warnings,
        })
    payload = {"engine": "arrangement_v2", "count": len(rows), "results": rows}
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
