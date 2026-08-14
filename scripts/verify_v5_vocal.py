"""Validate V5 vocal-priority quality results against a V4 golden baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


_MAX_F1_DROP = 0.005


def _selected_melody_recall(row: Dict[str, Any]) -> float:
    return float(row["metrics"]["selected_melody"]["recall"])


def verify(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the V5 vocal-recall and golden-set non-regression contract."""
    failures: List[Dict[str, Any]] = []
    baseline_summary = baseline.get("summary", {})
    candidate_summary = candidate.get("summary", {})
    for field in (
        "median_onset_f1",
        "median_selected_melody_f1",
        "validation_median_onset_f1",
        "validation_median_selected_melody_f1",
    ):
        if field not in baseline_summary or field not in candidate_summary:
            failures.append({"reason": "missing_summary_metric", "metric": field})
        elif float(candidate_summary[field]) + 1e-9 < float(baseline_summary[field]) - _MAX_F1_DROP:
            failures.append({
                "reason": "metric_regression", "metric": field,
                "baseline": baseline_summary[field], "candidate": candidate_summary[field],
            })

    baseline_rows = {str(row.get("id")): row for row in baseline.get("results", []) if "metrics" in row}
    candidate_rows = {str(row.get("id")): row for row in candidate.get("results", []) if "metrics" in row}
    for song_id, original in baseline_rows.items():
        updated = candidate_rows.get(song_id)
        if updated is None:
            failures.append({"reason": "missing_song", "id": song_id})
            continue
        original_recall = _selected_melody_recall(original)
        updated_recall = _selected_melody_recall(updated)
        if updated_recall < original_recall:
            failures.append({
                "reason": "selected_melody_recall_drop", "id": song_id,
                "baseline": original_recall, "candidate": updated_recall,
            })
    return {"passed": not failures, "maxF1Drop": _MAX_F1_DROP, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path, help="V4 golden benchmark report")
    parser.add_argument("candidate", type=Path, help="V5 vocal-priority benchmark report")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
