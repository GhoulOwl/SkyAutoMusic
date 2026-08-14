import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.model_runtime import ModelRuntime
from transcription.backends import BasicPitchBackend
from transcription.quality import _MODELS, release_quality_models
from transcription.separation import TwoStemSeparator


class _Timer:
    instances = []

    def __init__(self, delay, callback, args=()):
        self.delay, self.callback, self.args = delay, callback, args
        self.daemon = False
        self.cancelled = False
        self.started = False
        self.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            return self.callback(*self.args)
        return False


class TestModelRuntime(unittest.TestCase):
    def setUp(self):
        _Timer.instances = []
        self.released = []
        self.runtime = ModelRuntime(180, _Timer)
        self.runtime.register("one", lambda: self.released.append("one") or True)
        self.runtime.register("two", lambda: self.released.append("two") or True)

    def test_releases_all_models_only_after_idle_timer_fires(self):
        with self.runtime.activity():
            self.assertFalse(self.released)
        self.assertEqual(len(_Timer.instances), 1)
        self.assertEqual(_Timer.instances[0].delay, 180)
        self.assertFalse(self.released)
        self.assertTrue(_Timer.instances[0].fire())
        self.assertEqual(self.released, ["one", "two"])

    def test_new_activity_cancels_pending_release_and_nested_activity_waits_for_outer(self):
        with self.runtime.activity():
            with self.runtime.activity():
                pass
            self.assertFalse(_Timer.instances)
        first = _Timer.instances[-1]
        with self.runtime.activity():
            self.assertTrue(first.cancelled)
            self.assertFalse(self.released)
        _Timer.instances[-1].fire()
        self.assertEqual(self.released, ["one", "two"])

    def test_release_now_refuses_while_activity_is_running(self):
        with self.runtime.activity():
            self.assertFalse(self.runtime.release_now())
            self.assertFalse(self.released)
        self.assertTrue(self.runtime.release_now())
        self.assertEqual(self.released, ["one", "two"])

    def test_shutdown_cancels_timer_and_releases_when_idle(self):
        with self.runtime.activity():
            pass
        timer = _Timer.instances[-1]
        self.runtime.shutdown()
        self.assertTrue(timer.cancelled)
        self.assertEqual(self.released, ["one", "two"])

    def test_backend_release_hooks_clear_their_cached_instances(self):
        previous = dict(_MODELS)
        try:
            _MODELS.clear()
            _MODELS[("small", "cpu")] = object()
            self.assertTrue(release_quality_models())
            self.assertFalse(_MODELS)
        finally:
            _MODELS.clear()
            _MODELS.update(previous)
        separator = TwoStemSeparator()
        separator._separator = object()
        self.assertTrue(separator.release())
        self.assertFalse(separator.release())
        backend = BasicPitchBackend()
        backend._model = object()
        self.assertTrue(backend.release())
        self.assertFalse(backend.release())


if __name__ == "__main__":
    unittest.main()
