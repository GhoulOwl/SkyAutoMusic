"""Validate a V3 melody-regression report against the v4 drum-only contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List


def _f1(metrics: Dict[str, Any], field: str) -> float:
    return float(metrics[field]["f1"])


def _recall(metrics: Dict[str, Any]) -> float:
    return float(metrics["key_onset"]["recall"])


def verify(payload: Dict[str, Any]) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    for row in payload.get("results", []):
        candidate = row.get("window_metrics")
        original = row.get("original_v3_window_metrics")
        regression = row.get("v3_regression")
        if not (candidate and original and regression):
            failures.append({"id": row.get("id"), "reason": "missing_v3_comparison"})
            continue
        item = {"id": row["id"], "topVoiceDelta": {}, "keyOnsetRecallDelta": {}}
        for window in ("all", "front", "back"):
            top_delta = _f1(candidate[window], "top_voice") - _f1(original[window], "top_voice")
            recall_delta = _recall(candidate[window]) - _recall(original[window])
            item["topVoiceDelta"][window] = round(top_delta, 4)
            item["keyOnsetRecallDelta"][window] = round(recall_delta, 4)
            if top_delta < -0.005:
                warnings.append({"id": row["id"], "window": window, "reason": "top_voice_drop_after_drum_removal", "delta": round(top_delta, 4)})
            if recall_delta < -0.01:
                warnings.append({"id": row["id"], "window": window, "reason": "key_onset_recall_drop_after_drum_removal", "delta": round(recall_delta, 4)})
        if not regression.get("selectedMelodyIdentical"):
            failures.append({"id": row["id"], "reason": "selected_melody_changed"})
        if int(regression.get("addedRenderedNoteCount", 0)):
            failures.append({"id": row["id"], "reason": "non_drum_output_added", "count": regression["addedRenderedNoteCount"]})
        if "nonDrumRemovedRenderedNoteCount" not in regression:
            failures.append({"id": row["id"], "reason": "missing_removal_provenance"})
        elif int(regression["nonDrumRemovedRenderedNoteCount"]):
            failures.append({"id": row["id"], "reason": "non_drum_output_removed", "count": regression["nonDrumRemovedRenderedNoteCount"]})
        comparisons.append(item)

    validation_ids = {str(row.get("id")) for row in payload.get("results", []) if row.get("split") == "validation"}
    median_checks = {}
    scopes = {"all": comparisons, "validation": [item for item in comparisons if str(item["id"]) in validation_ids]}
    for scope, rows in scopes.items():
        if rows:
            candidate_median = median(
                next(row for row in payload["results"] if row.get("id") == item["id"])["window_metrics"]["all"]["top_voice"]["f1"]
                for item in rows
            )
            original_median = median(
                next(row for row in payload["results"] if row.get("id") == item["id"])["original_v3_window_metrics"]["all"]["top_voice"]["f1"]
                for item in rows
            )
            value = candidate_median - original_median
            median_checks[scope] = {"candidate": round(candidate_median, 4), "original": round(original_median, 4), "delta": round(value, 4)}
            if value < 0:
                warnings.append({"scope": scope, "reason": "median_top_voice_drop_after_drum_removal", "delta": round(value, 4)})
    return {
        "passed": not failures, "songCount": len(comparisons), "failures": failures,
        "warnings": warnings, "medianTopVoiceDelta": median_checks, "songs": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verdict = verify(json.loads(args.report.read_text(encoding="utf-8")))
    text = json.dumps(verdict, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
