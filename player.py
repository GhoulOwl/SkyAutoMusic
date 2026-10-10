import heapq
import json
import random
import threading
import time
from enum import Enum


class PlaybackState(Enum):
    STOPPED = 'stopped'
    PLAYING = 'playing'
    PAUSED = 'paused'


class MusicPlayer:
    """One timestamp scheduler for attacks and independently scheduled releases."""
    DEFAULT_MISS_PROB = .03
    DEFAULT_JITTER = (.85, 1.15)
    NOTE_HOLD = .05
    RETRIGGER_GAP = .005
    START_DELAY = .5

    def __init__(self, key_controller, update_status=None, update_elapsed=None,
                 update_total=None, update_progress=None, update_note=None,
                 update_played=None, update_finished=None, update_diagnostics=None,
                 clock=None, waiter=None):
        self.key_controller = key_controller
        callbacks = locals()
        for name in ('update_status', 'update_elapsed', 'update_total', 'update_progress',
                     'update_note', 'update_played', 'update_finished', 'update_diagnostics'):
            setattr(self, name, callbacks[name] or (lambda *args: None))
        self.state = PlaybackState.STOPPED
        self.speed = 1.0
        self.simulate = False
        self.miss_prob = self.DEFAULT_MISS_PROB
        self.jitter = self.DEFAULT_JITTER
        self.notes_by_time = None
        self.sorted_times = None
        self.current_idx = 0
        self.thread = None
        self._clock = clock or time.perf_counter
        self._wake = threading.Event()
        self._waiter = waiter or self._wake.wait
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._session = 0
        self._token = 0
        self._held_keys = {}
        self._releases = []
        self._seek_requested = False
        self._seek_target_idx = 0
        self._origin_perf = 0.0
        self._score_start_ms = 0
        self._paused_at = None
        self.diagnostics = []
        self.source_name = ''

    def is_playing(self):
        return self.state == PlaybackState.PLAYING

    def is_paused(self):
        return self.state == PlaybackState.PAUSED

    def is_stopped(self):
        return self.state == PlaybackState.STOPPED

    def _identity(self, note):
        resolver = getattr(self.key_controller, 'physical_key', None)
        return resolver(note) if resolver else note

    def start(self, notes_by_time, sorted_times):
        with self._lock:
            if not self.is_stopped() or not sorted_times:
                return False
            self._session += 1
            self._stop = threading.Event()
            self._wake.clear()
            self.notes_by_time = {t: list(dict.fromkeys(notes_by_time[t])) for t in sorted_times}
            self.sorted_times = list(sorted_times)
            self.current_idx = 0
            self._seek_requested = False
            self._held_keys = {}
            self._releases = []
            self._paused_at = None
            self.diagnostics = []
            self._score_start_ms = int(sorted_times[0])
            self._origin_perf = self._clock() + self.START_DELAY
            self.state = PlaybackState.PLAYING
            self.thread = threading.Thread(target=self._run, args=(self._session, self._stop), daemon=True)
            self.thread.start()
            return True

    def pause(self):
        with self._lock:
            if self.is_playing():
                self._paused_at = self._clock()
                self.state = PlaybackState.PAUSED
                if self._release_held():
                    self._stop.set()
                    self.state = PlaybackState.STOPPED
                    self.update_status('演奏失败：暂停时未能松开按键')
                self._wake.set()

    def resume(self):
        with self._lock:
            if self.is_paused():
                self._origin_perf += self._clock() - self._paused_at
                self._paused_at = None
                self.state = PlaybackState.PLAYING
                self._wake.set()

    def stop(self):
        with self._lock:
            self._stop.set()
            self.state = PlaybackState.STOPPED
            self.current_idx = 0
            self._release_held()
            self._wake.set()

    def seek(self, target_idx):
        with self._lock:
            if self.is_stopped() or not self.sorted_times:
                return
            self._seek_target_idx = max(0, min(int(target_idx), len(self.sorted_times) - 1))
            self._seek_requested = True
            if self._release_held():
                self._stop.set()
                self.state = PlaybackState.STOPPED
                self.update_status('演奏失败：跳转时未能松开按键')
            self._wake.set()

    def _record(self, **entry):
        entry.setdefault('session', self._session)
        self.diagnostics.append(entry)
        self.update_diagnostics(dict(entry))

    def diagnostics_summary(self):
        with self._lock:
            attacks = [row for row in self.diagnostics if row.get('action') == 'press']
            delays = sorted(row['lateness_ms'] for row in attacks)
            return {
                'sent_count': sum(len(row['sent_notes']) for row in attacks),
                'source_name': self.source_name,
                'planned_count': sum(len(notes) for notes in (self.notes_by_time or {}).values()),
                'failed_press_count': sum(len(row.get('failed_notes', [])) for row in attacks),
                'failed_count': sum(len(row.get('failed_notes', [])) for row in self.diagnostics),
                'fallback_count': sum(row.get('fallback_count', 0) for row in self.diagnostics),
                'p95_lateness_ms': delays[min(len(delays)-1, int(.95*len(delays)))] if delays else 0,
                'max_lateness_ms': max(delays, default=0),
                'backends': sorted({v for row in self.diagnostics for v in row.get('backends', {}).values()}),
                'acknowledgement': 'input_sent_only',
            }

    def export_diagnostics(self, path):
        with self._lock:
            payload = {'summary': self.diagnostics_summary(), 'events': list(self.diagnostics)}
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _release_notes(self, notes):
        try:
            result = self.key_controller.release_keys(notes)
            failed = list(getattr(result, 'failed_notes', ()))
            self._record(action='release', notes=list(notes), failed_notes=failed,
                         actual_perf=self._clock(), backends=dict(getattr(result, 'backend_by_note', {})),
                         fallback_count=getattr(result, 'fallback_count', 0))
            return failed
        except Exception as exc:
            self._record(action='release', notes=list(notes), failed_notes=list(notes), error=str(exc))
            return list(notes)

    def _release_held(self):
        notes = [held[0] for held in self._held_keys.values()]
        self._held_keys.clear()
        self._releases.clear()
        if notes:
            failed = self._release_notes(notes)
            if failed:
                return self._release_notes(failed)
        return []

    def _run(self, session, stop_event):
        try:
            self._playback_loop(session, stop_event)
        except Exception as exc:
            with self._lock:
                if session == self._session:
                    stop_event.set()
                    self._record(action='error', error=str(exc))
                    self.update_status(f'演奏失败：{exc}')
        finally:
            with self._lock:
                if session == self._session:
                    if self._release_held():
                        stop_event.set()
                        self.update_status('演奏失败：未能松开按键')
                    self.state = PlaybackState.STOPPED
                    self.current_idx = 0
                    self.update_note(-1, None, [])
                    if not stop_event.is_set():
                        self.update_progress(1.0)
                        self.update_status('演奏结束！')
                        self.update_finished()

    def _playback_loop(self, session, stop_event):
        times = self.sorted_times
        t0, last = int(times[0]), int(times[-1])
        span = max(1, last-t0)
        self.update_total((last-t0)/1000)
        upcoming, next_same = {}, {}
        for index in reversed(range(len(times))):
            for note in self.notes_by_time[times[index]]:
                identity = self._identity(note)
                next_same[(index, identity)] = upcoming.get(identity)
                upcoming[identity] = times[index]
        index, jitter = 0, {}
        while True:
            with self._lock:
                if session != self._session or stop_event.is_set():
                    return
                if self._seek_requested:
                    self._seek_requested = False
                    index = self._seek_target_idx
                    point = int(times[index])
                    self._origin_perf = self._clock() - (point-self._score_start_ms)/1000/max(.001, self.speed)
                    if self.is_paused():
                        self._paused_at = self._clock()
                    self.update_note(index, point, self.notes_by_time[times[index]])
                if self.is_paused():
                    delay = .05
                else:
                    now = self._clock()
                    due = []
                    due_identities = []
                    while self._releases and self._releases[0][0] <= now:
                        _deadline, token, identity = heapq.heappop(self._releases)
                        held = self._held_keys.get(identity)
                        if held and held[1] == token:
                            due.append(held[0])
                            due_identities.append(identity)
                    if due:
                        failed = self._release_notes(due)
                        if failed:
                            failed = self._release_notes(failed)
                        for identity in due_identities:
                            if self._held_keys[identity][0] not in failed:
                                del self._held_keys[identity]
                        if failed:
                            raise RuntimeError('未能松开按键：' + ', '.join(failed))
                    if index >= len(times):
                        if not self._releases:
                            return
                        target = self._releases[0][0]
                    else:
                        if index not in jitter:
                            jitter[index] = random.uniform(-.015, .015) if self.simulate else 0
                        target = self._target_perf(times[index]) + jitter[index]
                        if target <= now:
                            notes = self.notes_by_time[times[index]]
                            self.update_elapsed((int(times[index])-t0)/1000)
                            self.update_progress((int(times[index])-t0)/span)
                            self.update_note(index, times[index], notes)
                            if self.simulate and random.random() < self.miss_prob:
                                self._record(action='simulated_miss', notes=notes)
                            else:
                                self._play_notes(notes, target, index, next_same)
                            index += 1
                            self.current_idx = index
                            self.update_status(f'演奏进度: {index}/{len(times)}')
                            continue
                        if self._releases:
                            target = min(target, self._releases[0][0])
                    delay = max(0, min(.02, target-self._clock()))
                self._wake.clear()
            self._waiter(delay)

    def _play_notes(self, notes, target=None, index=0, next_same=None):
        target = self._clock() if target is None else target
        unique = {}
        for note in notes:
            identity = self._identity(note)
            if identity in self._held_keys:
                held = self._held_keys[identity]
                failed = self._release_notes([held[0]])
                if failed:
                    failed = self._release_notes(failed)
                if failed:
                    raise RuntimeError('重奏前未能松开按键：' + held[0])
                del self._held_keys[identity]
            unique.setdefault(identity, note)
        notes = list(unique.values())
        result = self.key_controller.press_keys(notes)
        sent = list(getattr(result, 'sent_notes', notes))
        now = self._clock()
        self._record(action='press', notes=notes, sent_notes=sent,
                     score_time_ms=int(self.sorted_times[index]) if self.sorted_times else None,
                     onset_index=index, speed=self.speed,
                     failed_notes=list(getattr(result, 'failed_notes', ())),
                     backends=dict(getattr(result, 'backend_by_note', {})),
                     fallback_count=getattr(result, 'fallback_count', 0),
                     scheduled_perf=target, actual_perf=now,
                     lateness_ms=round(max(0, now-target)*1000, 3))
        if sent:
            self.update_played(sent)
        for note in sent:
            identity = self._identity(note)
            self._token += 1
            token = self._token
            hold = self.NOTE_HOLD * (random.uniform(.7, 1.4) if self.simulate else 1)
            deadline = now + hold
            following = (next_same or {}).get((index, identity))
            if following is not None:
                next_target = self._target_perf(following)
                deadline = max(now, min(deadline, next_target-self.RETRIGGER_GAP))
                if next_target-now < self.RETRIGGER_GAP:
                    self._record(action='retrigger_conflict', notes=[note])
            self._held_keys[identity] = (note, token)
            heapq.heappush(self._releases, (deadline, token, identity))
        if getattr(result, 'fatal_error', ''):
            raise RuntimeError(result.fatal_error)

    def _target_perf(self, t_ms):
        speed = self.speed if self.speed and self.speed > 0 else 1.0
        return self._origin_perf + (int(t_ms)-self._score_start_ms)/1000/speed
