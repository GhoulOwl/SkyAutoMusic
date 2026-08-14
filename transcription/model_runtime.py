"""Shared lifecycle management for heavyweight transcription models."""
from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Iterator, Optional


ReleaseCallback = Callable[[], bool | None]


class ModelRuntime:
    """Release registered model caches after the last inference has been idle.

    The activity counter is deliberately shared across backends: a Demucs job
    must keep a just-loaded MuScriptor model alive, and vice versa.  Releasers
    run while the lifecycle lock is held so a new activity cannot race a cache
    clear half way through its own model load.
    """

    def __init__(self, idle_seconds: float = 180.0, timer_factory=threading.Timer) -> None:
        self.idle_seconds = float(idle_seconds)
        self._timer_factory = timer_factory
        self._lock = threading.RLock()
        self._active = 0
        self._generation = 0
        self._timer: Optional[threading.Timer] = None
        self._closed = False
        self._releasers: "OrderedDict[str, ReleaseCallback]" = OrderedDict()

    def register(self, name: str, releaser: ReleaseCallback) -> None:
        with self._lock:
            self._releasers[str(name)] = releaser

    @contextmanager
    def activity(self) -> Iterator[None]:
        self._begin_activity()
        try:
            yield
        finally:
            self._finish_activity()

    def _begin_activity(self) -> None:
        with self._lock:
            if self._closed:
                self._closed = False
            self._cancel_timer_locked()
            self._active += 1

    def _finish_activity(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            if self._active == 0 and not self._closed:
                self._schedule_release_locked()

    def _cancel_timer_locked(self) -> None:
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_release_locked(self) -> None:
        self._cancel_timer_locked()
        generation = self._generation
        timer = self._timer_factory(self.idle_seconds, self._release_if_idle, args=(generation,))
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _release_if_idle(self, generation: int) -> bool:
        with self._lock:
            if self._closed or generation != self._generation or self._active:
                return False
            self._timer = None
            self._generation += 1
            return self._release_locked()

    def release_now(self) -> bool:
        """Release caches immediately when no inference activity is running."""
        with self._lock:
            if self._active:
                return False
            self._cancel_timer_locked()
            return self._release_locked()

    def _release_locked(self) -> bool:
        released = False
        for releaser in tuple(self._releasers.values()):
            try:
                released = bool(releaser()) or released
            except Exception:
                # Cache cleanup must never turn an otherwise healthy app idle
                # transition into a user-visible failure.
                continue
        if released:
            gc.collect()
            self._clear_accelerator_cache()
        return released

    @staticmethod
    def _clear_accelerator_cache() -> None:
        try:
            import torch
            for runtime_name in ("cuda", "mps"):
                clear = getattr(getattr(torch, runtime_name, None), "empty_cache", None)
                if callable(clear):
                    try:
                        clear()
                    except Exception:
                        pass
        except Exception:
            pass

    def shutdown(self) -> None:
        """Stop future idle callbacks; active workers retain their models safely."""
        with self._lock:
            self._closed = True
            self._cancel_timer_locked()
            if not self._active:
                self._release_locked()


model_runtime = ModelRuntime()
