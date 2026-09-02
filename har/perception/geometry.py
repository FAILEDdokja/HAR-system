"""Dependency-free geometry helpers.

Moved verbatim from ``bottle_monitor.py`` (commit ``19c5436``) so the original
unit tests keep covering the same arithmetic. No cv2, no numpy.
"""

from __future__ import annotations

import math

from har.contracts import BBox, Point

__all__ = [
    "center",
    "box_size",
    "box_area",
    "box_iou",
    "point_to_box_distance",
    "distance",
    "add_box",
    "box_center_inside",
]


def center(box: BBox) -> Point:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_size(box: BBox) -> tuple[float, float]:
    return (max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1]))


def box_area(box: BBox) -> float:
    width, height = box_size(box)
    return width * height


def box_iou(first: BBox, second: BBox) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union else 0.0


def point_to_box_distance(point: Point, box: BBox) -> float:
    """Euclidean distance from a wrist to the nearest point on a box."""
    x, y = point
    nearest_x = min(max(x, box[0]), box[2])
    nearest_y = min(max(y, box[1]), box[3])
    return math.hypot(x - nearest_x, y - nearest_y)


def distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def add_box(box: BBox, velocity: tuple[float, float, float, float], steps: int) -> BBox:
    """Extrapolate a box forward by ``steps`` frames of constant velocity."""
    return tuple(value + delta * steps for value, delta in zip(box, velocity))  # type: ignore[return-value]


def box_center_inside(box: BBox, region: BBox) -> bool:
    """True when the centre of ``box`` falls inside ``region``.

    Used by the protocol predicates for zone containment checks.
    """
    cx, cy = center(box)
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]
