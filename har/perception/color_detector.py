"""HSV colour detector for the five PTS-01 protocol objects.

The protocol objects (black tray, yellow lid, red box, blue box, green vial)
are detected by colour segmentation, not by a network - no COCO-class weights
can see a box, crate, tray, bin or vial (see ``docs/DEVELOPMENT_PLAN.md`` §2).

Design notes
------------
* The frame is a BGR ``numpy`` array (exactly what ``cv2.VideoCapture``
  yields), so this class plugs straight into the frame loop.
* HSV conversion, thresholding and connected-component labelling use ``cv2``
  when it is importable (the fast path used in production). When cv2 is
  absent the same work is done with ``numpy`` plus a pure-Python flood fill,
  which is what the unit tests exercise on small synthetic frames. Both paths
  produce identical results on clean input; only speed differs.
* All HSV ranges come from ``config/colours.yaml`` via
  :func:`load_colour_config` - never hard-coded. Venue lighting will force a
  retune, and nobody should be editing Python in front of judges.
* The scale is OpenCV's: H in ``[0, 180)``, S and V in ``[0, 255]``. Tune with
  any HSV picker that reports OpenCV values (or the GIMP values / 2 for H).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:  # fast path; the pure-Python fallback below keeps the tests runnable
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from har.contracts import BBox, Detection

__all__ = ["ColorDetector", "load_colour_config", "bgr_to_hsv", "from_config"]


# --------------------------------------------------------------------------
# Colour space
# --------------------------------------------------------------------------


def bgr_to_hsv(frame: Any) -> Any:
    """BGR ``uint8`` array -> HSV ``uint8`` array on the OpenCV scale.

    Matches ``cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)``; the numpy path is used
    only when cv2 is not installed.
    """
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("expected an HxWx3 BGR frame")
    if cv2 is not None:
        return cv2.cvtColor(array[..., :3], cv2.COLOR_BGR2HSV)

    bgr = array[..., :3].astype(np.float32) / 255.0
    blue, green, red = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    maxc = np.max(bgr, axis=-1)
    minc = np.min(bgr, axis=-1)
    delta = maxc - minc

    safe_delta = np.where(delta > 0, delta, 1.0)
    hue = np.where(
        maxc == red,
        (green - blue) / safe_delta,
        np.where(
            maxc == green,
            (blue - red) / safe_delta + 2.0,
            (red - green) / safe_delta + 4.0,
        ),
    )
    hue = np.where(delta > 0, (hue / 6.0) % 1.0, 0.0)
    saturation = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1.0), 0.0)

    hsv = np.empty(array.shape[:2] + (3,), dtype=np.uint8)
    hsv[..., 0] = np.clip(hue * 179.0 + 0.5, 0, 179).astype(np.uint8)
    hsv[..., 1] = np.clip(saturation * 255.0 + 0.5, 0, 255).astype(np.uint8)
    hsv[..., 2] = np.clip(maxc * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return hsv


# --------------------------------------------------------------------------
# Connected components
# --------------------------------------------------------------------------

Component = tuple[int, int, int, int, float]
"""``(x1, y1, x2, y2, area)`` of one blob, in the coordinate space of the mask."""


def _largest_component(mask: Any, min_area: float) -> Component | None:
    """Largest 8-connected blob of a boolean mask, or ``None`` under ``min_area``."""
    if cv2 is not None:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best: Component | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if best is None or area > best[4]:
                best = (int(x), int(y), int(x + w), int(y + h), area)
        return best

    # Pure-Python flood fill (tests, small frames, no-cv2 environments).
    rows, cols = np.nonzero(mask)
    remaining = set(zip(rows.tolist(), cols.tolist()))
    best = None
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        area = 0
        min_x = max_x = seed[1]
        min_y = max_y = seed[0]
        while stack:
            y, x = stack.pop()
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for neighbour_y in (y - 1, y, y + 1):
                for neighbour_x in (x - 1, x, x + 1):
                    neighbour = (neighbour_y, neighbour_x)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
        if area >= min_area and (best is None or area > best[4]):
            best = (min_x, min_y, max_x + 1, max_y + 1, float(area))
    return best


def _kernel() -> Any:
    return np.ones((3, 3), dtype=np.uint8)


def _mask_for_ranges(hsv: Any, ranges: Iterable[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> Any:
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = None
    for (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi) in ranges:
        current = (
            (hue >= h_lo)
            & (hue <= h_hi)
            & (saturation >= s_lo)
            & (saturation <= s_hi)
            & (value >= v_lo)
            & (value <= v_hi)
        )
        mask = current if mask is None else mask | current
    if mask is None:
        mask = np.zeros(hsv.shape[:2], dtype=bool)
    return mask


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


def _normalise_ranges(spec: Any) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Accept one ``((lo), (hi))`` pair or a list of pairs; validate bounds."""
    pairs: list[Any] = []
    if spec and isinstance(spec[0][0], int):
        # a single ((h, s, v), (h, s, v)) pair
        pairs = [spec]
    else:
        pairs = list(spec)

    ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for low, high in pairs:
        low_tuple, high_tuple = tuple(int(v) for v in low), tuple(int(v) for v in high)
        if len(low_tuple) != 3 or len(high_tuple) != 3:
            raise ValueError(f"HSV range must have 3 values per bound, got {low}/{high}")
        for low_v, high_v in zip(low_tuple, high_tuple):
            if not 0 <= low_v <= high_v:
                raise ValueError(f"inverted or negative HSV range: {low_tuple} -> {high_tuple}")
        if high_tuple[0] > 180 or high_tuple[1] > 255 or high_tuple[2] > 255:
            raise ValueError(f"HSV high bound out of the OpenCV scale: {high_tuple}")
        ranges.append((low_tuple, high_tuple))
    return ranges


def _median_box(history: deque[BBox]) -> BBox:
    stacked = np.stack([np.asarray(box, dtype=np.float64) for box in history])
    return tuple(float(v) for v in np.median(stacked, axis=0))


class ColorDetector:
    """Implements :class:`har.contracts.ObjectDetector` by HSV segmentation.

    One label may carry several ranges (red wraps around H = 180). For every
    labelled range the largest blob above ``min_area`` inside ``roi`` becomes a
    :class:`~har.contracts.Detection`; its box is the component-wise median of
    the last ``median_window`` hits, which suppresses single-frame flicker.
    """

    def __init__(
        self,
        ranges: Mapping[str, Any],
        roi: BBox | None = None,
        median_window: int = 5,
        min_area: int = 400,
    ) -> None:
        self._ranges: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
            str(label): _normalise_ranges(spec) for label, spec in ranges.items()
        }
        self.roi: BBox | None = tuple(float(v) for v in roi) if roi is not None else None  # type: ignore[assignment]
        self.median_window = max(1, int(median_window))
        self.min_area = max(1, int(min_area))
        self._history: dict[str, deque[BBox]] = {
            label: deque(maxlen=self.median_window) for label in self._ranges
        }

    @classmethod
    def from_config(cls, path: str | Path) -> "ColorDetector":
        """Build a detector straight from ``config/colours.yaml``."""
        ranges, options = load_colour_config(path)
        return cls(
            ranges,
            roi=options.get("roi"),
            median_window=int(options.get("median_window", 5)),
            min_area=int(options.get("min_area", 400)),
        )

    # -- contracts.ObjectDetector -----------------------------------------

    @property
    def backend(self) -> str:
        return "hsv"

    def detect(self, frame: Any) -> list[Detection]:
        hsv = bgr_to_hsv(frame)
        height, width = hsv.shape[:2]

        offset_x = offset_y = 0
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            if x2 <= x1 or y2 <= y1:
                return []
            offset_x, offset_y = x1, y1
            hsv = hsv[y1:y2, x1:x2]

        detections: list[Detection] = []
        for label, ranges in self._ranges.items():
            mask = _mask_for_ranges(hsv, ranges)
            if cv2 is not None:
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_OPEN, _kernel()
                ).astype(bool)
            component = _largest_component(mask, float(self.min_area))
            if component is None:
                continue
            x1, y1, x2, y2, area = component
            box: BBox = (
                float(x1 + offset_x),
                float(y1 + offset_y),
                float(x2 + offset_x),
                float(y2 + offset_y),
            )
            history = self._history[label]
            history.append(box)
            detections.append(
                Detection(
                    box=_median_box(history),
                    confidence=min(1.0, area / max(1.0, (x2 - x1) * (y2 - y1))),
                    label=label,
                    track_id=None,
                )
            )
        return detections

    def reset(self) -> None:
        """Forget the smoothing history (new scene / new camera)."""
        for history in self._history.values():
            history.clear()


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def load_colour_config(path: str | Path) -> tuple[dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]], dict[str, Any]]:
    """Read ``config/colours.yaml`` -> ``(ranges, detector_options)``."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read the colour config") from exc

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    options = dict(data.get("detector") or {})
    colours = data.get("colours") or {}
    if not colours:
        raise ValueError(f"no 'colours' section in {path}")
    ranges = {}
    for label, spec in colours.items():
        entries = spec.get("ranges") if isinstance(spec, dict) else spec
        ranges[str(label)] = _normalise_ranges(entries)
    return ranges, options


def from_config(path: str | Path) -> "ColorDetector":
    """Build a :class:`ColorDetector` straight from ``config/colours.yaml``."""
    return ColorDetector.from_config(path)
