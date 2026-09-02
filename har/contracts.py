"""Frozen data contracts shared by every track.

THIS FILE IS THE COORDINATION MECHANISM. Three people build three subsystems
against these types and never need to agree on anything else.

Rules
-----
1. Nothing in this module may import cv2, numpy, torch, ultralytics or anything
   outside the standard library. It must import and test in a bare interpreter.
2. Signatures here are frozen for the duration of a phase. A change requires a
   note in ``docs/DEVELOPMENT_PLAN.md`` under *Contract changes* and a bump of
   ``CONTRACT_VERSION``.
3. Every type crossing a track boundary is JSON-serialisable through
   ``to_dict()``. That is what lets Track C be built against canned JSON while
   Track A and Track B are still unfinished.
4. All pixel geometry is ``(x1, y1, x2, y2)`` in source-frame pixels, origin
   top-left, y down. Rack-normalised coordinates are carried separately and are
   never mixed into a ``BBox``.

Producer / consumer map
-----------------------
======================  ==================  ============================
Type                    Produced by         Consumed by
======================  ==================  ============================
Detection               B (perception)      B
ObjectTrack             B                   A, C
Wrist                   B                   A, C
HandObjectState         B                   A, C
FrameEvidence           B                   A, C
StepEvent               A (validator)       C
ProtocolSpec/StepSpec   config (yaml)       A, C
UiStatus                A + C               C
======================  ==================  ============================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

CONTRACT_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Geometry primitives
# --------------------------------------------------------------------------

BBox = tuple[float, float, float, float]
"""``(x1, y1, x2, y2)`` in source-frame pixels. Always x1<x2, y1<y2."""

Point = tuple[float, float]
"""``(x, y)`` in source-frame pixels."""


# --------------------------------------------------------------------------
# Track B -> everyone
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One raw detector hit for one frame, before any temporal filtering."""

    box: BBox
    confidence: float
    label: str
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": [round(float(v), 2) for v in self.box],
            "confidence": round(float(self.confidence), 4),
            "label": self.label,
            "track_id": self.track_id,
        }


@dataclass(frozen=True)
class ObjectTrack:
    """One protocol object, temporally stabilised.

    ``measured`` is False while the tracker is coasting on prediction, which
    means the box is an estimate and must not be used to *complete* a step.
    """

    label: str
    box: BBox | None
    measured: bool
    lost_frames: int = 0
    velocity: BBox = (0.0, 0.0, 0.0, 0.0)

    @property
    def present(self) -> bool:
        return self.box is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "box": None if self.box is None else [round(float(v), 2) for v in self.box],
            "measured": bool(self.measured),
            "lost_frames": int(self.lost_frames),
        }


@dataclass(frozen=True)
class Wrist:
    """One detected wrist. ``side`` is ``"left"`` or ``"right"``."""

    point: Point
    confidence: float
    side: str
    person_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": [round(float(v), 2) for v in self.point],
            "confidence": round(float(self.confidence), 4),
            "side": self.side,
            "person_id": int(self.person_id),
        }


class HandObjectState(str, Enum):
    """Interaction FSM state of a hand with one protocol object.

    Values match ``har.perception.interaction.InteractionState``; the enum is
    duplicated here so Track A never imports the perception package.
    """

    IDLE = "IDLE"
    NEAR_OBJECT = "NEAR_OBJECT"
    PICKED_UP = "PICKED UP"
    CARRYING = "CARRYING"
    RELEASED = "RELEASED"


@dataclass(frozen=True)
class FrameEvidence:
    """Everything Track A is allowed to know about one frame.

    This is the single input to ``SequenceValidator.update``. Track A must not
    import cv2 or the perception package; if a signal is not in this class,
    Track A cannot use it.
    """

    frame_index: int
    t_rel: float
    frame_size: tuple[int, int]
    objects: Mapping[str, ObjectTrack] = field(default_factory=dict)
    hands: Sequence[Wrist] = field(default_factory=tuple)
    hoi: Mapping[str, str] = field(default_factory=dict)
    rack_ready: bool = False
    fps: float = 0.0

    def object(self, label: str) -> ObjectTrack | None:
        return self.objects.get(label)

    def hoi_state(self, label: str) -> str:
        return self.hoi.get(label, HandObjectState.IDLE.value)

    def wrists(self) -> list[Point]:
        return [w.point for w in self.hands]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "t_rel": round(float(self.t_rel), 3),
            "frame_size": [int(v) for v in self.frame_size],
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "hands": [w.to_dict() for w in self.hands],
            "hoi": dict(self.hoi),
            "rack_ready": bool(self.rack_ready),
            "fps": round(float(self.fps), 2),
        }


# --------------------------------------------------------------------------
# Protocol definition (config -> A, C)
# --------------------------------------------------------------------------


class StepEventType(str, Enum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    TIMEOUT = "TIMEOUT"
    PROTOCOL_COMPLETE = "PROTOCOL_COMPLETE"
    ALERT = "ALERT"
    INFO = "INFO"


@dataclass(frozen=True)
class Zone:
    """A named region of the rack. Bounding box in source-frame pixels."""

    id: str
    box: BBox
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "box": [round(float(v), 2) for v in self.box], "label": self.label}


@dataclass(frozen=True)
class StepSpec:
    """One step of the experiment protocol."""

    step_id: str
    index: int
    title: str
    instruction: str
    predicate: str
    target: str = ""
    zone: str = ""
    requires: tuple[str, ...] = ()
    hold_frames: int = 1
    timeout_s: float = 60.0
    voice_prompt: str = ""
    voice_alert: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "index": int(self.index),
            "title": self.title,
            "instruction": self.instruction,
            "predicate": self.predicate,
            "target": self.target,
            "zone": self.zone,
            "requires": list(self.requires),
            "hold_frames": int(self.hold_frames),
            "timeout_s": float(self.timeout_s),
            "voice_prompt": self.voice_prompt,
            "voice_alert": self.voice_alert,
        }


@dataclass(frozen=True)
class ProtocolSpec:
    """A whole experiment procedure, loaded from ``protocols/*.yaml``."""

    protocol_id: str
    title: str
    version: str
    steps: tuple[StepSpec, ...]
    zones: tuple[Zone, ...] = ()
    objects: tuple[str, ...] = ()

    def step(self, step_id: str) -> StepSpec | None:
        for candidate in self.steps:
            if candidate.step_id == step_id:
                return candidate
        return None

    def zone(self, zone_id: str) -> Zone | None:
        for candidate in self.zones:
            if candidate.id == zone_id:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "title": self.title,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "zones": [z.to_dict() for z in self.zones],
            "objects": list(self.objects),
        }


# --------------------------------------------------------------------------
# Track A -> Track C
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StepEvent:
    """One thing that happened. Append-only; never mutated after emission."""

    t_iso: str
    t_rel: float
    frame_index: int
    step_id: str
    step_index: int
    event: str
    status: str
    message: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_iso": self.t_iso,
            "t_rel": round(float(self.t_rel), 3),
            "frame_index": int(self.frame_index),
            "step_id": self.step_id,
            "step_index": int(self.step_index),
            "event": self.event,
            "status": self.status,
            "message": self.message,
            "confidence": round(float(self.confidence), 4),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


@dataclass(frozen=True)
class UiStatus:
    """Snapshot polled by the browser GUI. Track C renders, never computes."""

    protocol_id: str
    protocol_title: str
    current_step_id: str
    current_step_index: int
    next_step_id: str
    next_instruction: str
    completed: tuple[str, ...]
    skipped: tuple[str, ...]
    violations: tuple[str, ...]
    state: str
    t_rel: float
    fps: float
    last_alert: str
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["t_rel"] = round(float(self.t_rel), 3)
        out["fps"] = round(float(self.fps), 2)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


# --------------------------------------------------------------------------
# Seams. Each track implements these; nobody reaches into another package.
# --------------------------------------------------------------------------


class ObjectDetector(Protocol):
    """Track B implements. Swappable: YOLO weights or HSV colour detector."""

    def detect(self, frame: Any) -> list[Detection]: ...

    @property
    def backend(self) -> str: ...


class EventSink(Protocol):
    """Track C implements (log file, voice, GUI). Track A only calls emit()."""

    def emit(self, event: StepEvent) -> None: ...

    def close(self) -> None: ...


class Speaker(Protocol):
    """Track C implements. Must never block the caller."""

    def say(self, text: str, priority: int = 0) -> None: ...

    def stop(self) -> None: ...


class FrameSource(Protocol):
    """Track C implements. Yields ``(frame_index, frame)``; frame is opaque to
    Track A and is ``None``-safe only in headless synthetic mode."""

    def __iter__(self) -> Any: ...

    def release(self) -> None: ...


__all__ = [
    "CONTRACT_VERSION",
    "BBox",
    "Point",
    "Detection",
    "ObjectTrack",
    "Wrist",
    "HandObjectState",
    "FrameEvidence",
    "StepEventType",
    "Zone",
    "StepSpec",
    "ProtocolSpec",
    "StepEvent",
    "UiStatus",
    "ObjectDetector",
    "EventSink",
    "Speaker",
    "FrameSource",
]
