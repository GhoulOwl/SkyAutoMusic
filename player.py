import random
import threading
import time
from enum import Enum


class PlaybackState(Enum):
    """Stopped / playing / paused playback state."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class MusicPlayer:
    """Playback engine using absolute timestamps instead of accumulated sleeps."""

    DEFAULT_MISS_PROB = 0.03
    DEFAULT_JITTER = (0.85, 1.15)
    NOTE_HOLD = 0.05
    START_DELAY = 0.5

    def __init__(
        self,
        key_controller,
        update_status=None,
        update_elapsed=None,
        update_total=None,
        update_progress=None,
        update_note=None,
        update_finished=None,
    ):
        self.key_controller = key_controller
        self.update_status = update_status or (lambda msg: None)
        self.update_elapsed = update_elapsed or (lambda sec: None)
        self.update_total = update_total or (lambda sec: None)
        self.update_progress = update_progress or (lambda frac: None)
        self.update_note = update_note or (lambda idx, t_ms, notes: None)
        # 演奏自然结束（非用户停止）时回调，供 UI 重置到"就绪"态
        self.update_finished = update_finished or (lambda: None)
        self.state = PlaybackState.STOPPED
        self._stop = threading.Event()
        self.notes_by_time = None
        self.sorted_times = None
        self.current_idx = 0
        self.thread = None
        self.speed = 1.0
        self.simulate = False
        self.miss_prob = self.DEFAULT_MISS_PROB
        self.jitter = self.DEFAULT_JITTER
        self._seek_requested = False
        self._seek_target_idx = 0
        self._held_keys = []
        self._origin_perf = 0.0
        self._score_start_ms = 0

    def is_playing(self):
        return self.state == PlaybackState.PLAYING

    def is_paused(self):
        return self.state == PlaybackState.PAUSED

    def is_stopped(self):
        return self.state == PlaybackState.STOPPED

    def start(self, notes_by_time, sorted_times):
        if self.state != PlaybackState.STOPPED or not sorted_times:
            return False
        self.notes_by_time = notes_by_time
        self.sorted_times = sorted_times
        self.current_idx = 0
        self._stop.clear()
        self._seek_requested = False
        self._held_keys = []
        self._score_start_ms = int(sorted_times[0])
        self._origin_perf = time.perf_counter() + self.START_DELAY
        self.state = PlaybackState.PLAYING
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def pause(self):
        if self.state == PlaybackState.PLAYING:
            self.state = PlaybackState.PAUSED

    def resume(self):
        if self.state == PlaybackState.PAUSED:
            self.state = PlaybackState.PLAYING

    def stop(self):
        self._stop.set()
        self.state = PlaybackState.STOPPED
        self.current_idx = 0
        self._release_held()

    def seek(self, target_idx):
        if self.state == PlaybackState.STOPPED or not self.sorted_times:
            return
        self._seek_target_idx = max(0, min(int(target_idx), len(self.sorted_times) - 1))
        self._seek_requested = True

    def _release_held(self):
        if self._held_keys:
            try:
                self.key_controller.release_keys(self._held_keys)
            except Exception:
                pass
            self._held_keys = []

    def _run(self):
        try:
            self._playback_loop()
        finally:
            self._release_held()
            self.state = PlaybackState.STOPPED
            self.current_idx = 0
            self.update_note(-1, None, [])
            if not self._stop.is_set():
                self.update_progress(1.0)
                self.update_status("演奏结束！")
                self.update_finished()

    def _playback_loop(self):
        notes_by_time = self.notes_by_time
        sorted_times = self.sorted_times
        total = len(sorted_times)
        if total == 0:
            return

        t0 = int(sorted_times[0])
        last_t = int(sorted_times[-1])
        span = max(1, last_t - t0)
        self.update_total((last_t - t0) / 1000.0)

        idx = self.current_idx
        while idx < total:
            if self._stop.is_set():
                break
            if not self._wait_if_paused():
                break
            if self._seek_requested:
                idx = self._consume_seek(t0, span)
                continue

            t_ms = int(sorted_times[idx])
            target_perf = self._target_perf(t_ms)
            if self.simulate:
                target_perf += random.uniform(-0.015, 0.015)
            if not self._sleep_until(target_perf):
                break
            if self._seek_requested:
                idx = self._consume_seek(t0, span)
                continue

            notes = list(notes_by_time[t_ms])
            elapsed_sec = max(0, (t_ms - t0) / 1000.0)
            progress = (t_ms - t0) / span
            self.update_elapsed(elapsed_sec)
            self.update_progress(progress)
            self.update_note(idx, t_ms, notes)

            if self.simulate and random.random() < self.miss_prob:
                self.update_status(f"演奏进度: {idx + 1}/{total}（失误跳过）")
            else:
                self._play_notes(notes)
                self.update_status(f"演奏进度: {idx + 1}/{total}")

            idx += 1
            self.current_idx = idx

    def _play_notes(self, notes):
        held = list(notes)
        self._held_keys = held
        self.key_controller.press_keys(held)
        hold = self.NOTE_HOLD
        if self.simulate:
            hold *= random.uniform(0.7, 1.4)
        self._sleep_note_hold(time.perf_counter() + hold, held)
        if self._held_keys is held:
            self._held_keys = []

    def _sleep_note_hold(self, target_perf, held):
        while True:
            if self._stop.is_set() or self._seek_requested:
                self.key_controller.release_keys(held)
                return False
            if self.state == PlaybackState.PAUSED:
                self.key_controller.release_keys(held)
                if self._held_keys is held:
                    self._held_keys = []
                return self._wait_if_paused()
            remaining = target_perf - time.perf_counter()
            if remaining <= 0:
                self.key_controller.release_keys(held)
                return True
            time.sleep(min(0.01, remaining))

    def _target_perf(self, t_ms):
        speed = self.speed if self.speed and self.speed > 0 else 1.0
        return self._origin_perf + ((int(t_ms) - self._score_start_ms) / 1000.0) / speed

    def _wait_if_paused(self):
        if self.state != PlaybackState.PAUSED:
            return not self._stop.is_set()

        paused_at = time.perf_counter()
        while self.state == PlaybackState.PAUSED:
            if self._stop.is_set():
                return False
            if self._seek_requested:
                return True
            time.sleep(0.05)
        self._origin_perf += time.perf_counter() - paused_at
        return not self._stop.is_set()

    def _consume_seek(self, t0, span):
        self._release_held()
        self._seek_requested = False
        target = max(0, min(self._seek_target_idx, len(self.sorted_times) - 1))
        self.current_idx = target
        t_ms = int(self.sorted_times[target])
        elapsed_sec = max(0, (t_ms - t0) / 1000.0)
        self.update_elapsed(elapsed_sec)
        self.update_progress((t_ms - t0) / span)
        self.update_note(target, t_ms, list(self.notes_by_time[t_ms]))
        speed = self.speed if self.speed and self.speed > 0 else 1.0
        self._origin_perf = time.perf_counter() - ((t_ms - self._score_start_ms) / 1000.0) / speed
        return target

    def _sleep_until(self, target_perf):
        while True:
            if self._stop.is_set():
                return False
            if self._seek_requested:
                return True
            if self.state == PlaybackState.PAUSED:
                return self._wait_if_paused()
            remaining = target_perf - time.perf_counter()
            if remaining <= 0:
                return True
            time.sleep(min(0.02, remaining))
