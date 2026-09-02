"""PerceptionStack: detector + trackers + interaction FSMs -> FrameEvidence.

``FrameEvidence`` is the *only* thing Track A is allowed to see, so this is
the seam where all perception work is packed per frame (task B4). The flow per
frame is:

1. run the wrist extractor (pose every N frames; cached in between, with the
   person count from the last pose pass),
2. **person gate** - when nobody is in frame, skip the detector entirely and
   hand the trackers an empty detection list. That is a free frame-rate win:
   the trackers coast, the FSMs hold, and nothing protocol-relevant can
   happen without an operator anyway,
3. fan the detections through one :class:`SingleTargetTracker` per protocol
   label,
4. advance one :class:`InteractionMachine` per label with the tracked box and
   the wrist points, and lock tracker identity while an object is carried,
5. pack :class:`~har.contracts.FrameEvidence` - field-for-field what A1's
   fixtures use. The fixture is the spec.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Sequence

from har.contracts import FrameEvidence, ObjectDetector, ObjectTrack
from har.perception.interaction import InteractionConfig, InteractionMachine
from har.perception.tracker import TrackerConfig, TrackerRegistry

__all__ = ["PerceptionStack"]


class PerceptionStack:
    """Compose detector, tracker registry and interaction machines."""

    def __init__(
        self,
        detector: ObjectDetector,
        wrists: Any,
        labels: Sequence[str],
        frame_size: tuple[int, int],
        tracker_config: TrackerConfig | None = None,
        interaction_config: InteractionConfig | None = None,
    ) -> None:
        self.detector = detector
        self.wrists = wrists
        self.labels = tuple(labels)
        self.frame_size = tuple(int(v) for v in frame_size)
        if tracker_config is None:
            tracker_config = TrackerConfig(labels=self.labels)
        elif not tracker_config.labels:
            # A registry without a label filter would let every tracker chase
            # every detection; scope it unless the caller already did.
            tracker_config.labels = self.labels
        self.trackers = TrackerRegistry(self.labels, tracker_config)
        # Scope each tracker to exactly its own label: the registry fans every
        # detection out to every tracker, and without this the blue tracker
        # would happily acquire the red box's detections.
        for label in self.labels:
            self.trackers[label].config = replace(tracker_config, labels=(label,))
        self.interactions = {
            label: InteractionMachine(label, interaction_config) for label in self.labels
        }
        #: Optional ``har.perception.rack.RackFrame``; when set and ready, the
        #: produced evidence carries ``rack_ready=True``.
        self.rack: Any = None
        #: When True (default), no person in frame short-circuits detection.
        self.person_gate = True
        self._last_time: float | None = None
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def process(self, frame: Any, frame_index: int, t_rel: float) -> FrameEvidence:
        self._measure_fps(t_rel)

        wrist_list = self.wrists.wrists(frame, frame_index)
        wrists_points = [w.point for w in wrist_list]

        person_present = True
        if self.person_gate:
            person_present = bool(getattr(self.wrists, "person_present", True))
        detections = self.detector.detect(frame) if person_present else []

        results = self.trackers.update_all(detections, self.frame_size)

        objects: dict[str, ObjectTrack] = {}
        hoi: dict[str, str] = {}
        for label in self.labels:
            result = results[label]
            tracker = self.trackers[label]
            machine = self.interactions[label]
            interaction = machine.update(
                result.measured, result.box, wrists_points, self.frame_size
            )
            tracker.set_locked(machine.identity_locked())
            objects[label] = ObjectTrack(
                label=label,
                box=result.box,
                measured=result.measured,
                lost_frames=result.lost_frames,
            )
            hoi[label] = interaction.state.value

        rack_ready = bool(self.rack is not None and self.rack.ready())
        return FrameEvidence(
            frame_index=int(frame_index),
            t_rel=float(t_rel),
            frame_size=self.frame_size,  # type: ignore[arg-type]
            objects=objects,
            hands=tuple(wrist_list),
            hoi=hoi,
            rack_ready=rack_ready,
            fps=self._fps,
        )

    def reset(self) -> None:
        """New scene: forget every tracker, FSM and the fps estimate."""
        self.trackers.reset_for_new_scene()
        for machine in self.interactions.values():
            machine.state = machine.state.IDLE
            machine.pickup_counter = machine.release_counter = 0
            machine.near_counter = machine.stable_counter = 0
            machine.picked_up_counter = 0
        self._last_time = None
        self._fps = 0.0
        reset = getattr(self.detector, "reset", None)
        if callable(reset):
            reset()
        reset_wrists = getattr(self.wrists, "reset", None)
        if callable(reset_wrists):
            reset_wrists()

    def _measure_fps(self, t_rel: float) -> None:
        """EMA of end-to-end processing rate, driven by the wall clock."""
        now = time.monotonic()
        if self._last_time is not None:
            elapsed = now - self._last_time
            if elapsed > 0:
                instant = 1.0 / elapsed
                self._fps = instant if self._fps == 0.0 else 0.8 * self._fps + 0.2 * instant
        self._last_time = now
