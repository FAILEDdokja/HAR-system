"""Wrist extraction from the pretrained pose network.

Wraps ``models/yolo11n-pose.pt`` (ultralytics 8.2.100, offline, no fine-tune)
and returns :class:`har.contracts.Wrist` objects. The pose model also detects
the ``person`` class, so it doubles as the "is an operator in frame?" gate -
``yolo11n.pt`` never needs to be loaded at runtime (§2 of the plan).

Two behaviours matter downstream and are deliberately strict here:

* **Skipped frames repeat the previous result.** An empty wrist list reads to
  the interaction FSM as "hands vanished" and corrupts pickup detection, so a
  frame on which pose is not re-run returns the cached wrists.
* The model is duck-typed (anything with ``.predict(frame, ...)``), so the
  unit tests run with stand-ins and no torch installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from har.contracts import Wrist
from har.perception.adapters import wrists_from_pose_result

__all__ = ["WristExtractor"]


class WristExtractor:
    """Run pose every ``every_n_frames`` frames; cache wrists in between."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.35,
        every_n_frames: int = 1,
        model: Any = None,
        keypoint_confidence: float = 0.2,
    ) -> None:
        self.weights = str(weights)
        self.conf = float(conf)
        self.every_n_frames = max(1, int(every_n_frames))
        self.keypoint_confidence = float(keypoint_confidence)
        self._model = model if model is not None else self._load_model(self.weights)
        self._last_wrists: list[Wrist] = []
        self._last_run_index: int | None = None
        #: Person count from the most recent pose pass. PerceptionStack reads
        #: this as the free frame-rate gate (§2): zero people -> skip work.
        self.person_count = 0

    @staticmethod
    def _load_model(weights: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on machine
            raise RuntimeError(
                "ultralytics is not installed; pass model= to WristExtractor "
                "or pip install ultralytics==8.2.100"
            ) from exc
        return YOLO(weights)

    @property
    def person_present(self) -> bool:
        return self.person_count > 0

    def wrists(self, frame: Any, frame_index: int) -> list[Wrist]:
        """Wrists for ``frame_index``; cached result on skipped frames."""
        if (
            self._last_run_index is not None
            and frame_index - self._last_run_index < self.every_n_frames
        ):
            return list(self._last_wrists)

        result = self._model.predict(
            frame,
            conf=self.conf,
            verbose=False,
        )[0]
        extracted = wrists_from_pose_result(result, min_confidence=self.keypoint_confidence)

        boxes = getattr(result, "boxes", None)
        xyxy = None if boxes is None else getattr(boxes, "xyxy", None)
        self.person_count = 0 if xyxy is None else len(xyxy)

        self._last_wrists = extracted
        self._last_run_index = frame_index
        return list(extracted)

    def reset(self) -> None:
        """Drop the cache so the next frame re-runs pose."""
        self._last_wrists = []
        self._last_run_index = None
        self.person_count = 0
