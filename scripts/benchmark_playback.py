"""Measure scheduler lateness without sending any real keyboard input."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from player import MusicPlayer


class Recorder:
    def press_keys(self, notes):
        pass

    def release_keys(self, notes):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval-ms', type=int, default=20)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.interval_ms <= 0 or args.count <= 0:
        parser.error('interval and count must be positive')
    score = {i * args.interval_ms: [f'1Key{i % 3}'] for i in range(args.count)}
    player = MusicPlayer(Recorder())
    player.START_DELAY = 0
    player.start(score, sorted(score))
    player.thread.join()
    report = player.diagnostics_summary()
    report['interval_ms'] = args.interval_ms
    report['planned_count'] = args.count
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
