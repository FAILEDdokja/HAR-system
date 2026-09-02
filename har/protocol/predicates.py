"""Protocol predicate implementations.

The functions in this module are the only frame-level facts the sequence
validator evaluates.  They deliberately depend only on :mod:`har.contracts`:
Track A must remain usable without cv2, numpy, torch, ultralytics, or the
perception package installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import hypot
from typing import Callable

from har.contracts import BBox, FrameEvidence, HandObjectState, ProtocolSpec, StepSpec


@dataclass
class PredicateState:
    """Mutable per-step predicate state owned by the validator.

    ``satisfied_frames`` is intentionally present here for the validator's hold
    counter, even though these predicate functions do not increment it.  The
    remaining fields remember short temporal facts needed to distinguish a real
    manipulation from a hand merely passing near a stationary prop.
    """

    satisfied_frames: int = 0
    hoi_seen: set[str] = field(default_factory=set)
    last_box: BBox | None = None
    initial_box: BBox | None = None


Predicate = Callable[[FrameEvidence, ProtocolSpec, StepSpec, PredicateState], bool]

_PREDICATE_RE = re.compile(r"^\s*([a-z_]+)\((.*)\)\s*$")
_MOVEMENT_EPS_PX = 8.0
_STABLE_EPS_PX = 5.0


def _predicate_args(step: StepSpec) -> tuple[str, ...]:
    """Return comma-separated arguments from ``step.predicate``.

    The YAML already validates predicate names, but parsing here keeps the
    implementation robust for unit tests that construct ``StepSpec`` directly.
    Empty arguments are ignored so ``name()`` yields ``()``.
    """

    match = _PREDICATE_RE.match(step.predicate)
    if not match:
        return ()
    raw_args = match.group(2).strip()
    if not raw_args:
        return ()
    return tuple(arg.strip() for arg in raw_args.split(",") if arg.strip())


def _arg(step: StepSpec, index: int, fallback: str = "") -> str:
    args = _predicate_args(step)
    return args[index] if index < len(args) else fallback


def _box_center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_box(point: tuple[float, float], box: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _center_in_box(obj_box: BBox, zone_box: BBox) -> bool:
    return _point_in_box(_box_center(obj_box), zone_box)


def _center_distance(a: BBox, b: BBox) -> float:
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return hypot(ax - bx, ay - by)


def _measured_box(evidence: FrameEvidence, label: str) -> BBox | None:
    track = evidence.object(label)
    if track is None or track.box is None or not track.measured:
        return None
    return track.box


def _zone_box(spec: ProtocolSpec, zone_id: str) -> BBox | None:
    zone = spec.zone(zone_id)
    return None if zone is None else zone.box


def _remember_initial(st: PredicateState, box: BBox) -> None:
    if st.initial_box is None:
        st.initial_box = box


def _moved_from_initial(st: PredicateState, box: BBox, eps: float = _MOVEMENT_EPS_PX) -> bool:
    return st.initial_box is not None and _center_distance(st.initial_box, box) >= eps


def _update_hoi_state(evidence: FrameEvidence, label: str, st: PredicateState) -> str:
    hoi_state = evidence.hoi_state(label)
    st.hoi_seen.add(hoi_state)
    return hoi_state


def object_stable(ev: FrameEvidence, spec: ProtocolSpec, step: StepSpec, st: PredicateState) -> bool:
    """Object is present, measured, inside its required zone, and not moving.

    With only one observation there is no evidence of motion, so the first
    measured frame is considered stable.  The validator's ``hold_frames`` then
    supplies the required persistence.
    """

    label = _arg(step, 0, step.target)
    box = _measured_box(ev, label)
    zone = _zone_box(spec, step.zone)
    if box is None or (zone is not None and not _center_in_box(box, zone)):
        st.last_box = box
        return False

    stable = st.last_box is None or _center_distance(st.last_box, box) <= _STABLE_EPS_PX
    st.last_box = box
    _remember_initial(st, box)
    return stable


def object_left_zone(ev: FrameEvidence, spec: ProtocolSpec, step: StepSpec, st: PredicateState) -> bool:
    """Object is present and measured, with its centre outside the named zone."""

    label = _arg(step, 0, step.target)
    zone_id = _arg(step, 1, step.zone)
    box = _measured_box(ev, label)
    zone = _zone_box(spec, zone_id)
    if box is None or zone is None:
        st.last_box = box
        return False
    st.last_box = box
    _remember_initial(st, box)
    return not _center_in_box(box, zone)


def hoi_cycle(ev: FrameEvidence, spec: ProtocolSpec, step: StepSpec, st: PredicateState) -> bool:
    """Object was picked/carried and then released inside the target zone.

    A real cycle must include both manipulation evidence and object motion.  A
    hand sweeping near a prop that remains stationary therefore cannot satisfy
    this predicate.
    """

    label = _arg(step, 0, step.target)
    zone_id = _arg(step, 1, step.zone)
    box = _measured_box(ev, label)
    zone = _zone_box(spec, zone_id)
    if box is None or zone is None:
        return False

    _remember_initial(st, box)
    hoi_state = _update_hoi_state(ev, label, st)
    st.last_box = box

    picked_or_carried = {
        HandObjectState.PICKED_UP.value,
        HandObjectState.CARRYING.value,
    }
    saw_manipulation = bool(st.hoi_seen & picked_or_carried)
    return (
        hoi_state == HandObjectState.RELEASED.value
        and saw_manipulation
        and _center_in_box(box, zone)
        and _moved_from_initial(st, box)
    )


def settled(ev: FrameEvidence, spec: ProtocolSpec, step: StepSpec, st: PredicateState) -> bool:
    """Target object's centre is in the zone and stationary."""

    label = _arg(step, 0, step.target)
    zone_id = _arg(step, 1, step.zone)
    box = _measured_box(ev, label)
    zone = _zone_box(spec, zone_id)
    if box is None or zone is None or not _center_in_box(box, zone):
        st.last_box = box
        return False

    stable = st.last_box is None or _center_distance(st.last_box, box) <= _STABLE_EPS_PX
    st.last_box = box
    _remember_initial(st, box)
    return stable


def transfer(ev: FrameEvidence, spec: ProtocolSpec, step: StepSpec, st: PredicateState) -> bool:
    """Destination object was removed from a source object and released in a zone.

    For PTS-01 this captures the vial being taken from the red box and inserted
    into the rack slot.  The source/destination/zone are parsed from the
    predicate string, with ``step.target`` and ``step.zone`` as fallbacks.
    """

    src_label = _arg(step, 0, "")
    dst_label = _arg(step, 1, step.target)
    zone_id = _arg(step, 2, step.zone)

    src_box = _measured_box(ev, src_label) if src_label else None
    dst_box = _measured_box(ev, dst_label)
    zone = _zone_box(spec, zone_id)
    if dst_box is None or zone is None:
        return False

    _remember_initial(st, dst_box)
    if src_box is not None and _center_in_box(dst_box, src_box):
        st.hoi_seen.add("FROM_SOURCE")

    hoi_state = _update_hoi_state(ev, dst_label, st)
    st.last_box = dst_box

    saw_manipulation = bool(st.hoi_seen & {HandObjectState.PICKED_UP.value, HandObjectState.CARRYING.value})
    return (
        "FROM_SOURCE" in st.hoi_seen
        and hoi_state == HandObjectState.RELEASED.value
        and saw_manipulation
        and _center_in_box(dst_box, zone)
        and _moved_from_initial(st, dst_box)
    )


def hands_clear(ev: FrameEvidence, spec: ProtocolSpec, step: StepSpec, st: PredicateState) -> bool:
    """No detected wrist is inside the named zone."""

    zone_id = _arg(step, 0, step.zone)
    zone = _zone_box(spec, zone_id)
    if zone is None:
        return False
    return all(not _point_in_box(wrist.point, zone) for wrist in ev.hands)


PREDICATES: dict[str, Predicate] = {
    "object_stable": object_stable,
    "object_left_zone": object_left_zone,
    "hoi_cycle": hoi_cycle,
    "settled": settled,
    "transfer": transfer,
    "hands_clear": hands_clear,
}


__all__ = [
    "PredicateState",
    "PREDICATES",
    "object_stable",
    "object_left_zone",
    "hoi_cycle",
    "settled",
    "transfer",
    "hands_clear",
]
