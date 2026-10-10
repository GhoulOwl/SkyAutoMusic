"""Gate a candidate against a fresh V8 report without tuning on validation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

METRICS = ('median_onset_f1', 'median_selected_melody_f1',
           'validation_median_onset_f1', 'validation_median_selected_melody_f1')


def verify(baseline, candidate, max_drop=.005, targeted=None):
    failures = []
    for name in METRICS:
        a, b = baseline.get('summary', {}).get(name), candidate.get('summary', {}).get(name)
        if a is None or b is None:
            failures.append({'reason': 'missing_metric', 'metric': name})
        elif b + 1e-9 < a - max_drop:
            failures.append({'reason': 'metric_regression', 'metric': name, 'baseline': a, 'candidate': b})
    a = {row['id'] for row in baseline.get('results', [])}
    b = {row['id'] for row in candidate.get('results', [])}
    if a != b:
        failures.append({'reason': 'sample_set_changed', 'missing': sorted(a-b), 'extra': sorted(b-a)})
    if targeted is None or not targeted.get('passed'):
        failures.append({'reason': 'targeted_vocal_improvement_unverified'})
    return {'passed': not failures, 'max_f1_drop': max_drop, 'failures': failures,
            'default_strategy': 'candidate' if not failures else 'stable',
            'game_sound_acknowledgement': 'not_tested'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--targeted', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verdict = verify(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
                     targeted=json.loads(args.targeted.read_text()) if args.targeted else None)
    text = json.dumps(verdict, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)
    return 0 if verdict['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
