from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from transcription.benchmark import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the private SkyAutoMusic V2/V3 15-key golden set.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-cached", action="store_true", help="Refuse a new MuScriptor run; reuse raw model notes")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    report = run_benchmark(args.manifest, args.output, require_cached=args.require_cached)
    print(json.dumps(report['summary'] if args.summary_only else report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
