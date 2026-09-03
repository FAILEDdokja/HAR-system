"""Wrist extraction from the pretrained pose network.

Wraps ``models/yolo11n-pose.pt`` (ultralytics 8.2.100, offline, no fine-tune)
and returns :class:`har.contracts.Wrist` objects. The pose model also detects
the ``person`` class, so it doubles as the "is an operator in frame?" gate -
``yolo11n.pt`` never needs to be loaded at runtime (§2 of the plan).

Two behaviours matter downstream and are deliberately strict here:

* **Skipped frames repeat the previous result.** An empty wrist list reads to
  the interaction FSM as "hands vanished" and corrupts pickup detection, so a
  frame on which pose is not re-run returns the cached wrists.
* **Per-frame wrists are temporally confirmed.** A bare single-frame pose pass
  can hallucinate a wrist (a hand tucked out of view, an occluded arm, or a
  misread background) with a plausible keypoint confidence even when no hand
  is really in frame.  Raw wrists are pushed through a :class:`WristDebouncer`
  so a wrist is only exposed downstream once it has been observed on several
  consecutive pose frames, and is held a few frames after it disappears so a
  short detector dropout of a real hand does not read as "hands vanished".
* The model is duck-typed (anything with ``.predict(frame, ...)``), so the
  unit tests run with stand-ins and no torch installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from har.contracts import Wrist
from har.perception.adapters import wrists_from_pose_result

__all__ = ["WristExtractor", "WristDebouncer"]


class WristDebouncer:
    """Temporal confirmation of raw per-frame wrist detections (anti false-pos).

    The COCO-style pose model predicts *person* wrists as part of whole-body
    pose.  When a person is in frame but their hands are out of view, tucked
    away or occluded, the model still emits wrist keypoints for a fraction of
    frames with a plausible per-keypoint confidence — these are the "hands
    reported even though no hand is visible" false positives.  They are
    typically intermittent rather than persistent.

    This filter only exposes a wrist once it has been present for
    ``confirm_frames`` *consecutive* pose frames, and then holds it for up to
    ``forget_frames`` consecutive missing frames so a one-frame dropout of a
    real hand does not read as "hands vanished" (the B3 rule).  The single
    operator demo has at most one hand per side, so a wrist is keyed by its
    ``side`` (``"left"`` / ``"right"``); when several raw candidates for one
    side appear (e.g. multiple people), the highest-confidence one is used.
    """

    __slots__ = ("confirm_frames", "forget_frames", "_tracks")

    def __init__(self, confirm_frames: int = 1, forget_frames: int = 3) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.forget_frames = max(0, int(forget_frames))
        #: side -> {"seen", "missing", "point", "conf", "person_id"}
        self._tracks: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, raw: Iterable[Wrist]) -> list[Wrist]:
        """Advance the debounce state with one pose pass and return confirmed wrists.

        ``raw`` is the per-keypoint-conf-filtered wrist list for one pose pass
        (or the empty list when nobody / nothing was detected).
        """
        # Pick the highest-confidence raw wrist per side for this pass.
        best: dict[str, Wrist] = {}
        for wrist in raw:
            current = best.get(wrist.side)
            if current is None or wrist.confidence > current.confidence:
                best[wrist.side] = wrist

        for side in set(best) | set(self._tracks):
            seen_this_frame = best.get(side)
            track = self._tracks.get(side)
            if seen_this_frame is None:
                if track is not None:
                    track["missing"] += 1  # present this frame? no -> count it
                continue
            if track is None:
                track = {"seen": 0, "missing": 0}
                self._tracks[side] = track
            track["seen"] += 1
            track["missing"] = 0
            track["point"] = seen_this_frame.point
            track["conf"] = seen_this_frame.confidence
            track["person_id"] = seen_this_frame.person_id

        confirmed: list[Wrist] = []
        for side in list(self._tracks):
            track = self._tracks[side]
            if track["missing"] > self.forget_frames:
                del self._tracks[side]
                continue
            if track["seen"] >= self.confirm_frames:
                confirmed.append(
                    Wrist(
                        point=track["point"],
                        confidence=track["conf"],
                        side=side,
                        person_id=int(track["person_id"]),
                    )
                )
        confirmed.sort(key=lambda w: 0 if w.side == "left" else 1)
        return confirmed


class WristExtractor:
    """Run pose every ``every_n_frames`` frames; cache wrists in between.

    ``keypoint_confidence`` is the minimum per-wrist *visibility* score from the
    pose network (a low value lets occluded/hallucinated wrists through).
    ``confirm_frames`` / ``forget_frames`` tune :class:`WristDebouncer`.
    """

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.35,
        every_n_frames: int = 1,
        model: Any = None,
        keypoint_confidence: float = 0.2,
        confirm_frames: int = 1,
        forget_frames: int = 3,
    ) -> None:
        self.weights = str(weights)
        self.conf = float(conf)
        self.every_n_frames = max(1, int(every_n_frames))
        self.keypoint_confidence = float(keypoint_confidence)
        self._model = model if model is not None else self._load_model(self.weights)
        self._debounce = WristDebouncer(
            confirm_frames=max(1, int(confirm_frames)),
            forget_frames=max(0, int(forget_frames)),
        )
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

        # Temporal confirmation: only sustained detections become "hands".
        self._last_wrists = self._debounce.update(extracted)
        self._last_run_index = frame_index
        return list(self._last_wrists)

    def reset(self) -> None:
        """Drop the cache and debounce state so the next frame re-runs pose."""
        self._last_wrists = []
        self._last_run_index = None
        self.person_count = 0
        self._debounce.reset()
