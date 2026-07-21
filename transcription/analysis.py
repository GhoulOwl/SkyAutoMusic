from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .models import PitchEvent, TranscribeConfig, TranscribeStats
from .preprocess import AudioPreprocessor
from .fusion import fuse_events


class AnalyzerUnavailableError(RuntimeError):
    pass


def _muscriptor_unavailable_message(exc: Optional[BaseException] = None) -> str:
    suffix = f" ({exc})" if exc else ""
    return (
        "MuScriptor is not available. Please run: uv pip install "
        "--torch-backend=cu128 -r requirements.txt. MuScriptor model weights "
        "are hosted on gated Hugging Face repositories; accept the CC BY-NC 4.0 "
        "license and run `uvx hf auth login` or set HF_TOKEN before first use."
        f"{suffix}"
    )


@contextlib.contextmanager
def _quiet_muscriptor_logs():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    yield
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            "MuScriptor wrote non-UTF8 console output that this Windows console "
            "could not encode."
        ) from exc


def _load_muscriptor_model_class():
    try:
        from muscriptor import TranscriptionModel
    except Exception as exc:
        raise AnalyzerUnavailableError(_muscriptor_unavailable_message(exc)) from exc
    return TranscriptionModel


class MuScriptorAnalyzer:
    """MuScriptor-backed analyzer that emits raw pitch events."""

    def __init__(self, config: TranscribeConfig):
        self.config = config
        self.last_artifacts: Dict[str, bytes] = {}
        model_class = _load_muscriptor_model_class()
        device = None if config.muscriptor_device == "auto" else config.muscriptor_device
        try:
            with _quiet_muscriptor_logs():
                self.model = model_class.load_model(
                    config.muscriptor_model,
                    device=device,
                )
        except Exception as exc:
            raise AnalyzerUnavailableError(_muscriptor_unavailable_message(exc)) from exc

    @classmethod
    def is_available(cls) -> Tuple[bool, Optional[str]]:
        try:
            _load_muscriptor_model_class()
        except Exception as exc:
            return False, str(exc)
        return True, None

    def analyze(self, audio_path: str) -> Tuple[List[PitchEvent], TranscribeStats]:
        self.last_artifacts = {}
        starts: Dict[int, Any] = {}
        events: List[PitchEvent] = []
        debug_rows: List[Dict[str, Any]] = []
        dropped = 0

        # 维度1：音频预处理（降噪/分离/归一化），产出清洗后的音频供引擎与 F0 使用。
        clean_audio = AudioPreprocessor(self.config).run(audio_path)

        kwargs = self._transcribe_kwargs(clean_audio)
        with _quiet_muscriptor_logs():
            event_stream = self.model.transcribe(**kwargs)
            for event in event_stream:
                event_type = self._event_type(event)
                if event_type == "progress":
                    debug_rows.append(self._debug_row(event, "progress"))
                    continue
                if event_type == "start":
                    starts[int(event.index)] = event
                    debug_rows.append(self._debug_row(event, "start"))
                    continue
                if event_type == "end":
                    debug_rows.append(self._debug_row(event, "end"))
                    pitch_event = self._pitch_event_from_end(event, starts)
                    if pitch_event is None:
                        dropped += 1
                    else:
                        events.append(pitch_event)
                    continue
                dropped += 1

        # 维度2/3/5：并行 F0 重建置信度、八度校正、缺口补检与起始点校验。
        if os.path.exists(clean_audio):
            try:
                events = fuse_events(events, clean_audio, self.config)
            except Exception as exc:  # 融合失败不应破坏扒谱主流程
                warnings.warn(f"event fusion skipped: {exc}")

        events.sort(key=lambda event: (event.time_ms, event.midi, event.instrument))
        if debug_rows:
            payload = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in debug_rows
            )
            self.last_artifacts["events_jsonl"] = payload.encode("utf-8")

        instrument_counts = Counter(event.instrument or "unknown" for event in events)
        stats = TranscribeStats(
            mapping=self.config.mapping,
            engine_requested=self.config.engine,
            engine_used="muscriptor",
            raw_event_count=len(events),
            onset_count=len({event.time_ms for event in events}),
            dropped_count=dropped,
            muscriptor_model=self.config.muscriptor_model,
            muscriptor_device=self.config.muscriptor_device,
            instrument_counts=dict(sorted(instrument_counts.items())),
        )
        stats.duration_sec = self._duration_sec(audio_path)
        return events, stats

    def _transcribe_kwargs(self, audio_path: str) -> Dict[str, Any]:
        kwargs = {
            "audio": audio_path,
            "instruments": self.config.muscriptor_instruments,
            "batch_size": self.config.muscriptor_batch_size,
            "no_eos_is_ok": True,
            "beam_size": self.config.muscriptor_beam_size,
            "prelude_forcing": self.config.muscriptor_prelude_forcing,
            "cfg_coef": self.config.muscriptor_cfg_coef,
        }
        # MuScriptor 0.2.1 does not expose every documented/experimental option.
        # Keep config compatibility, but only send parameters supported by the installed package.
        try:
            signature = inspect.signature(self.model.transcribe)
        except (TypeError, ValueError):
            return kwargs
        allowed = set(signature.parameters)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in allowed}

    @staticmethod
    def _event_type(event: Any) -> str:
        if hasattr(event, "completed") and hasattr(event, "total"):
            return "progress"
        if hasattr(event, "pitch") and hasattr(event, "start_time") and hasattr(event, "index"):
            return "start"
        if hasattr(event, "end_time") and (
            hasattr(event, "start_event") or hasattr(event, "start_event_index")
        ):
            return "end"
        return "unknown"

    @classmethod
    def _pitch_event_from_end(
        cls,
        event: Any,
        starts: Dict[int, Any],
    ) -> Optional[PitchEvent]:
        start_event = getattr(event, "start_event", None)
        if start_event is None:
            start_index = getattr(event, "start_event_index", None)
            if start_index is not None:
                start_event = starts.get(int(start_index))
        if start_event is None:
            return None

        start_ms = max(0, int(round(float(start_event.start_time) * 1000)))
        end_ms = max(start_ms, int(round(float(event.end_time) * 1000)))
        instrument = str(getattr(start_event, "instrument", "") or "unknown")
        return PitchEvent(
            time_ms=start_ms,
            midi=float(start_event.pitch),
            confidence=1.0,
            source="muscriptor",
            duration_ms=end_ms - start_ms,
            instrument=instrument,
            confidence_source="muscriptor",
        )

    @staticmethod
    def _debug_row(event: Any, event_type: str) -> Dict[str, Any]:
        if event_type == "start":
            return {
                "type": "start",
                "index": int(event.index),
                "pitch": int(event.pitch),
                "start_time": float(event.start_time),
                "instrument": str(getattr(event, "instrument", "") or "unknown"),
            }
        if event_type == "end":
            start_event = getattr(event, "start_event", None)
            start_index = (
                getattr(event, "start_event_index", None)
                if start_event is None
                else getattr(start_event, "index", None)
            )
            return {
                "type": "end",
                "end_time": float(event.end_time),
                "start_event_index": int(start_index) if start_index is not None else None,
            }
        return {
            "type": "progress",
            "completed": int(getattr(event, "completed", 0)),
            "total": int(getattr(event, "total", 0)),
        }

    @staticmethod
    def _duration_sec(audio_path: str) -> float:
        try:
            import soundfile

            info = soundfile.info(audio_path)
            return float(info.duration)
        except Exception:
            return 0.0


class AnalyzerFactory:
    """Create the MuScriptor analyzer used by the transcription pipeline."""

    @classmethod
    def create(cls, config: TranscribeConfig):
        return MuScriptorAnalyzer(config), None


AudioAnalyzer = MuScriptorAnalyzer
