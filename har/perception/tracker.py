"""Single-target association that tolerates detector/ByteTrack ID churn.

Moved from ``bottle_monitor.SingleBottleTracker`` (commit ``19c5436``) and
generalised from *one* hard-coded bottle to *one labelled* target, so a
``TrackerRegistry`` can hold one instance per protocol object.

The association arithmetic is unchanged and is still covered by
``tests/test_tracker.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from har.contracts import BBox, Detection
from har.perception.geometry import add_box, box_area, box_iou, box_size, center, distance

__all__ = ["TrackerConfig", "TrackResult", "SingleTargetTracker", "TrackerRegistry"]


@dataclass
class TrackerConfig:
    min_confidence: float = 0.45
    min_area_fraction: float = 0.0008
    max_area_fraction: float = 0.18
    min_aspect_ratio: float = 0.18
    max_aspect_ratio: float = 2.2
    acquire_frames: int = 4
    acquire_distance_fraction: float = 0.08
    acquire_iou: float = 0.08
    association_distance_fraction: float = 0.16
    carrying_distance_fraction: float = 0.34
    reacquire_growth_per_miss: float = 0.035
    max_reacquire_distance_fraction: float = 0.45
    association_iou: float = 0.02
    max_prediction_misses: int = 14
    roi: Optional[BBox] = None
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrackResult:
    label: str
    box: Optional[BBox]
    measured: bool
    lost_frames: int
    acquiring_hits: int = 0
    candidate_count: int = 0


class SingleTargetTracker:
    """Associates detections to one logical, labelled target.

    A target is never re-numbered: whatever the detector's own ID does, the
    application-level identity stays ``label``.
    """

    def __init__(self, label: str, config: TrackerConfig | None = None) -> None:
        self.label = label
        self.config = config or TrackerConfig()
        self.box: Optional[BBox] = None
        self.velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self.last_detector_track_id: Optional[int] = None
        self.lost_frames = 0
        self.locked = False
        self._tentative: Optional[Detection] = None
        self._tentative_hits = 0

    def reset_for_new_scene(self) -> None:
        """Forget the current target; the identity stays ``self.label``."""
        self.box = None
        self.velocity = (0.0, 0.0, 0.0, 0.0)
        self.last_detector_track_id = None
        self.lost_frames = 0
        self.locked = False
        self._tentative = None
        self._tentative_hits = 0

    def set_locked(self, locked: bool) -> None:
        self.locked = locked

    def valid_detections(
        self, detections: Iterable[Detection], frame_size: tuple[int, int]
    ) -> list[Detection]:
        """Reject implausible boxes before they can influence application state."""
        frame_width, frame_height = frame_size
        frame_area = frame_width * frame_height
        valid: list[Detection] = []
        for detection in detections:
            if self.config.labels and detection.label not in self.config.labels:
                continue
            width, height = box_size(detection.box)
            area_fraction = box_area(detection.box) / frame_area if frame_area else 0.0
            aspect = width / height if height else 0.0
            if detection.confidence < self.config.min_confidence:
                continue
            if area_fraction < self.config.min_area_fraction:
                continue
            if area_fraction > self.config.max_area_fraction:
                continue
            if not (self.config.min_aspect_ratio <= aspect <= self.config.max_aspect_ratio):
                continue
            if self.config.roi and box_iou(detection.box, self.config.roi) <= 0.0:
                continue
            valid.append(detection)
        return valid

    def update(
        self, detections: Iterable[Detection], frame_size: tuple[int, int]
    ) -> TrackResult:
        valid = self.valid_detections(detections, frame_size)
        if self.box is None:
            return self._acquire(valid, frame_size)

        match = self._best_match(valid, frame_size)
        if match is None:
            self.lost_frames += 1
            if self.lost_frames <= self.config.max_prediction_misses:
                self.box = add_box(self.box, self.velocity, 1)
            return TrackResult(self.label, self.box, False, self.lost_frames,
                               candidate_count=len(valid))

        old_box = self.box
        self.box = match.box
        self.velocity = tuple(new - old for new, old in zip(match.box, old_box))  # type: ignore[assignment]
        self.last_detector_track_id = match.track_id
        self.lost_frames = 0
        self._tentative = None
        self._tentative_hits = 0
        return TrackResult(self.label, self.box, True, 0, candidate_count=len(valid))

    def _acquire(self, valid: list[Detection], frame_size: tuple[int, int]) -> TrackResult:
        if not valid:
            self._tentative = None
            self._tentative_hits = 0
            return TrackResult(self.label, None, False, 0, candidate_count=0)

        candidate = max(valid, key=lambda item: (item.confidence, box_area(item.box)))
        if self._tentative is None:
            self._tentative = candidate
            self._tentative_hits = 1
        elif self._same_candidate(candidate, self._tentative, frame_size):
            self._tentative = candidate
            self._tentative_hits += 1
        else:
            self._tentative = candidate
            self._tentative_hits = 1

        if self._tentative_hits < self.config.acquire_frames:
            return TrackResult(self.label, candidate.box, False, 0,
                               acquiring_hits=self._tentative_hits,
                               candidate_count=len(valid))

        self.box = candidate.box
        self.velocity = (0.0, 0.0, 0.0, 0.0)
        self.last_detector_track_id = candidate.track_id
        return TrackResult(self.label, self.box, True, 0, self._tentative_hits, len(valid))

    def _same_candidate(
        self, current: Detection, previous: Detection, frame_size: tuple[int, int]
    ) -> bool:
        diagonal = math.hypot(*frame_size)
        return (
            distance(center(current.box), center(previous.box))
            <= diagonal * self.config.acquire_distance_fraction
            or box_iou(current.box, previous.box) >= self.config.acquire_iou
        )

    def _best_match(
        self, valid: list[Detection], frame_size: tuple[int, int]
    ) -> Optional[Detection]:
        if self.box is None:
            return None

        diagonal = math.hypot(*frame_size)
        predicted_box = add_box(self.box, self.velocity, max(1, self.lost_frames + 1))
        base_fraction = (
            self.config.carrying_distance_fraction
            if self.locked
            else self.config.association_distance_fraction
        )
        distance_fraction = min(
            self.config.max_reacquire_distance_fraction,
            base_fraction + self.lost_frames * self.config.reacquire_growth_per_miss,
        )
        max_distance = diagonal * distance_fraction

        best_detection: Optional[Detection] = None
        best_score = float("inf")
        for detection in valid:
            predicted_distance = distance(center(detection.box), center(predicted_box))
            current_distance = distance(center(detection.box), center(self.box))
            iou = max(box_iou(detection.box, predicted_box), box_iou(detection.box, self.box))
            same_detector_id = (
                self.last_detector_track_id is not None
                and detection.track_id == self.last_detector_track_id
            )
            if predicted_distance > max_distance and current_distance > max_distance:
                if iou < self.config.association_iou:
                    continue

            score = min(predicted_distance, current_distance) - iou * 80.0
            if same_detector_id:
                score -= 40.0
            score -= detection.confidence * 10.0
            if score < best_score:
                best_score = score
                best_detection = detection
        return best_detection


class TrackerRegistry:
    """One :class:`SingleTargetTracker` per protocol object label."""

    def __init__(self, labels: Iterable[str], config: TrackerConfig | None = None) -> None:
        self._trackers = {label: SingleTargetTracker(label, config) for label in labels}

    def __getitem__(self, label: str) -> SingleTargetTracker:
        return self._trackers[label]

    def labels(self) -> list[str]:
        return list(self._trackers)

    def update_all(
        self, detections: Iterable[Detection], frame_size: tuple[int, int]
    ) -> dict[str, TrackResult]:
        """Fan the same detection list out to every tracker.

        The list is materialised once: each tracker filters it by its own
        ``config.labels``.
        """
        pool = list(detections)
        return {label: tracker.update(pool, frame_size) for label, tracker in self._trackers.items()}

    def reset_for_new_scene(self) -> None:
        for tracker in self._trackers.values():
            tracker.reset_for_new_scene()
