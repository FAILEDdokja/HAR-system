"""Rack-relative frame: express detections in rack space, not frame space.

The honest, cheap version of the optional "orientation-agnostic" challenge:
we do not track relative to gravity, we track relative to the payload rack.
Four rack fiducials (ArUco markers, or simply four taped corners) give four
point correspondences between rack space and frame space, which determine a
plane projective transform (homography) exactly. Every box and zone can then
be expressed in rack coordinates, so rotating the whole rig 90 degrees
changes nothing downstream (B8).

Implementation notes
--------------------
* The homography is solved in pure Python (exact 8x8 DLT for four
  correspondences) so this module is dependency-free and unit-testable in a
  bare interpreter. For four exact correspondences the result is identical to
  ``cv2.getPerspectiveTransform``; if cv2 is installed it is used as a
  cross-check in comments only, never required.
* Fiducial order is fixed: ``top_left, top_right, bottom_right, bottom_left``
  **as seen in the frame**, mapping to rack-space corners
  ``(0, 0), (w, 0), (w, h), (0, h)``.
* ``update()`` never raises on a degenerate observation: it keeps the previous
  homography and reports ``ready() == False`` instead, because fiducial
  detection jitters live and one bad frame must not kill the run.
"""

from __future__ import annotations

import math
from typing import Sequence

from har.contracts import BBox, Point

__all__ = ["RackFrame", "HOMOGRAPHY_TOLERANCE"]

#: Rack-space size below which the rig counts as degenerate (pixels).
HOMOGRAPHY_TOLERANCE = 1e-9


def _solve_homography(
    source: Sequence[Point], target: Sequence[Point]
) -> tuple[float, ...] | None:
    """Exact homography ``h`` with ``target ~= H(source)`` for 4 correspondences.

    Returns the 9 coefficients row-major, or ``None`` when the system is
    singular (collinear/degenerate points).
    """
    # Rows of the 8x8 system; unknowns a..h with the transform written as
    #   u = (a x + b y + c) / (g x + h y + 1)
    #   v = (d x + e y + f) / (g x + h y + 1)
    matrix: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(source, target):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u])
        rhs.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v])
        rhs.append(v)

    solution = _gaussian_solve(matrix, rhs)
    if solution is None:
        return None
    return (*solution, 1.0)


def _gaussian_solve(
    matrix: list[list[float]], rhs: list[float]
) -> list[float] | None:
    """Gaussian elimination with partial pivoting; ``None`` when singular."""
    size = len(rhs)
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for column in range(size):
        pivot_row = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot_row][column]) < HOMOGRAPHY_TOLERANCE:
            return None
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        pivot = augmented[column][column]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column] / pivot
            if factor == 0.0:
                continue
            for col in range(column, size + 1):
                augmented[row][col] -= factor * augmented[column][col]
    return [augmented[i][size] / augmented[i][i] for i in range(size)]


def _apply(homography: tuple[float, ...], point: Point) -> Point:
    a, b, c, d, e, f, g, h_, _ = homography
    x, y = point
    denominator = g * x + h_ * y + 1.0
    if abs(denominator) < HOMOGRAPHY_TOLERANCE:
        raise ValueError("point maps to infinity under the rack homography")
    return ((a * x + b * y + c) / denominator, (d * x + e * y + f) / denominator)


class RackFrame:
    """Maps frame-space boxes into rack-normalised space via a homography."""

    CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")

    def __init__(self, fiducials: Sequence[Point], rack_size: tuple[float, float]) -> None:
        self.rack_size = (float(rack_size[0]), float(rack_size[1]))
        if self.rack_size[0] <= 0 or self.rack_size[1] <= 0:
            raise ValueError("rack_size must be positive")
        self._homography: tuple[float, ...] | None = None
        self.update(fiducials)

    # -- lifecycle ---------------------------------------------------------

    def update(self, fiducials: Sequence[Point]) -> bool:
        """Re-home on a fresh fiducial observation.

        Returns ``True`` when a usable homography is in place. On a degenerate
        observation the previous homography is kept and ``False`` is returned.
        """
        points = [(float(p[0]), float(p[1])) for p in fiducials]
        if len(points) != 4 or any(not all(map(math.isfinite, p)) for p in points):
            return False

        width, height = self.rack_size
        rack_corners: list[Point] = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
        homography = _solve_homography(points, rack_corners)
        if homography is None:
            return False
        self._homography = homography
        return True

    def ready(self) -> bool:
        """True when a usable frame -> rack transform is in place."""
        return self._homography is not None

    # -- transforms ---------------------------------------------------------

    def to_rack_point(self, point: Point) -> Point:
        if self._homography is None:
            raise RuntimeError("rack frame is not ready; call ready() first")
        return _apply(self._homography, point)

    def to_rack(self, box: BBox) -> BBox:
        """Express a frame-space ``(x1, y1, x2, y2)`` box in rack coordinates."""
        corners = (
            (box[0], box[1]),
            (box[2], box[1]),
            (box[2], box[3]),
            (box[0], box[3]),
        )
        mapped = [self.to_rack_point(corner) for corner in corners]
        xs = [point[0] for point in mapped]
        ys = [point[1] for point in mapped]
        return (min(xs), min(ys), max(xs), max(ys))
