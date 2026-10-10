import threading
import unittest
import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import patch

from key_controller import KeyBatchResult, KeyController, _KeyboardBackend, _PyAutoGUIBackend
from player import MusicPlayer, PlaybackState


class Clock:
    def __init__(self):
        self.now = 0.0
        self.hook = None

    def __call__(self):
        return self.now

    def wait(self, duration):
        self.now += max(duration, 1e-9)
        if self.hook:
            self.hook()


class Recorder:
    def __init__(self, clock, failed=()):
        self.clock, self.failed = clock, failed
        self.events = []

    def press_keys(self, notes):
        self.events.append(('press', self.clock(), tuple(notes)))
        return KeyBatchResult([n for n in notes if n not in self.failed],
                              [n for n in notes if n in self.failed])

    def release_keys(self, notes):
        self.events.append(('release', self.clock(), tuple(notes)))


class Backend:
    available = True
    def __init__(self, fail=False, partial=None):
        self.fail, self.partial, self.calls = fail, partial, []

    def send_many(self, action, keys):
        self.calls.append((action, tuple(keys)))
        if self.fail:
            raise RuntimeError('failed')
        return len(keys) if self.partial is None else self.partial


class PlaybackFidelityTests(unittest.TestCase):
    def run_score(self, score, failed=(), hook=None):
        clock = Clock()
        controller = Recorder(clock, failed)
        player = MusicPlayer(controller, clock=clock, waiter=clock.wait)
        player.START_DELAY = 0
        if hook:
            clock.hook = lambda: hook(player, clock)
        self.assertTrue(player.start(score, sorted(score)))
        player.thread.join(1)
        self.assertFalse(player.thread.is_alive())
        self.assertEqual(player._held_keys, {})
        return player, controller

    def test_twenty_ms_attacks_do_not_accumulate_hold_delays(self):
        player, controller = self.run_score({t: [f'1Key{i % 3}'] for i, t in enumerate(range(0, 181, 20))})
        actual = [t for action, t, _notes in controller.events if action == 'press']
        self.assertEqual(len(actual), 10)
        for i, value in enumerate(actual):
            self.assertAlmostEqual(value, i * .02, places=7)
        self.assertEqual(player.diagnostics_summary()['sent_count'], 10)

    def test_same_key_is_released_five_ms_before_next_attack(self):
        _player, controller = self.run_score({0: ['1Key0'], 20: ['1Key0'], 40: ['1Key0']})
        releases = [t for action, t, _notes in controller.events if action == 'release']
        self.assertAlmostEqual(releases[0], .015)
        self.assertAlmostEqual(releases[1], .035)

    def test_overlapping_keys_keep_their_independent_releases(self):
        _player, controller = self.run_score({0: ['1Key0'], 20: ['1Key1']})
        self.assertEqual([a for a, _t, _n in controller.events], ['press', 'press', 'release', 'release'])
        self.assertAlmostEqual(controller.events[2][1], .05)

    def test_partial_failure_is_not_reported_as_played(self):
        player, _controller = self.run_score({0: ['good', 'bad']}, failed=('bad',))
        self.assertEqual(player.diagnostics_summary()['sent_count'], 1)
        self.assertEqual(player.diagnostics_summary()['failed_count'], 1)
        self.assertEqual(player.diagnostics[0]['sent_notes'], ['good'])

    def test_pause_freezes_origin_and_seek_clears_old_releases(self):
        seen = set()
        def hook(player, clock):
            if not seen:
                seen.add('pause')
                player.pause()
                clock.now += 1
                player.seek(1)
                seen.add('seek')
            elif player.is_paused():
                player.resume()
        player, controller = self.run_score({0: ['a'], 100: ['b'], 200: ['c']}, hook=hook)
        self.assertEqual([n for a, _t, n in controller.events if a == 'press'], [('a',), ('b',), ('c',)])
        self.assertEqual(player.state, PlaybackState.STOPPED)

    def test_stop_restart_old_thread_cannot_finish_new_session(self):
        controller = Recorder(lambda: 0)
        player = MusicPlayer(controller)
        player.START_DELAY = .2
        player.start({0: ['old']}, [0])
        old = player.thread
        player.stop()
        player.START_DELAY = 0
        player.NOTE_HOLD = 0
        player.start({0: ['new']}, [0])
        player.thread.join(1)
        old.join(1)
        self.assertEqual([n for a, _t, n in controller.events if a == 'press'], [('new',)])

    def test_backend_partial_delivery_falls_back_only_for_unsent_keys(self):
        controller = KeyController({'a': 'y', 'b': 'u'}, log_func=lambda _: None)
        first, fallback = Backend(partial=1), Backend()
        controller._backends = {'interception': first, 'keyboard': fallback, 'pyautogui': Backend(fail=True)}
        controller.set_backend('interception')
        result = controller.press_keys(['a', 'b'])
        self.assertEqual(result.sent_notes, ['a', 'b'])
        self.assertEqual(fallback.calls, [('press', ('u',))])
        controller.set_backend('pyautogui')
        released = controller.release_keys(['a', 'b'])
        self.assertEqual(released.failed_notes, [])
        self.assertEqual(first.calls[-1], ('release', ('y',)))
        self.assertEqual(fallback.calls[-1], ('release', ('u',)))

    def test_aliases_are_one_physical_attack(self):
        controller = KeyController({'a': 'Y', 'alias': 'y'}, log_func=lambda _: None)
        backend = Backend()
        controller._backends['keyboard'] = backend
        controller.set_backend('keyboard')
        controller.press_keys(['a', 'alias'])
        self.assertEqual(backend.calls, [('press', ('y',))])

    def test_sendinput_batch_has_scan_codes_and_release_flags(self):
        calls = []
        def send(count, inputs, size):
            calls.append((count, [(entry.payload.ki.wScan, entry.payload.ki.dwFlags) for entry in inputs]))
            if sys.platform == 'win32':
                self.assertEqual(size, 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
            return count
        module = SimpleNamespace(key_to_scan_codes=lambda key: (21 if key == 'y' else 22,))
        with patch('key_controller.keyboard', module), patch.object(ctypes, 'WinDLL', return_value=SimpleNamespace(SendInput=send), create=True):
            backend = _KeyboardBackend()
            self.assertEqual(backend.send_many('press', ['y', 'u']), 2)
            self.assertEqual(backend.send_many('release', ['y', 'u']), 2)
        self.assertEqual(calls, [(2, [(21, 8), (22, 8)]), (2, [(21, 10), (22, 10)])])

    def test_pyautogui_playback_omits_implicit_pause(self):
        calls = []
        module = SimpleNamespace(keyDown=lambda key, **kwargs: calls.append(('down', key, kwargs)),
                                 keyUp=lambda key, **kwargs: calls.append(('up', key, kwargs)))
        with patch('key_controller.pyautogui', module):
            backend = _PyAutoGUIBackend(); backend.press('y'); backend.release('y')
        self.assertEqual(calls, [('down', 'y', {'_pause': False}), ('up', 'y', {'_pause': False})])
