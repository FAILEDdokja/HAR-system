"""RackFrame (B7): homography from four fiducials, dependency-free.

The headline test is the B7 "done when": a 90-degree-rotated frame yields the
same rack-space box. The homography is solved in pure Python, so no cv2 or
numpy is needed here.
"""

import math
import unittest

from har.perception.rack import RackFrame

RACK_SIZE = (100.0, 80.0)
# Fiducials of an axis-aligned rack: scaled 1:1 and offset by (10, 10).
FIDUCIALS = [(10.0, 10.0), (110.0, 10.0), (110.0, 90.0), (10.0, 90.0)]
BOX = (30.0, 30.0, 50.0, 50.0)
RACK_BOX = (20.0, 20.0, 40.0, 40.0)


def rotate_90_ccw(point):
    """90-degree counter-clockwise rotation about the origin, shifted to +x."""
    x, y = point
    return (300.0 - y, x)


class RackFrameTests(unittest.TestCase):
    def test_axis_aligned_rack_maps_boxes_1_to_1_with_offset(self):
        rack = RackFrame(FIDUCIALS, RACK_SIZE)
        self.assertTrue(rack.ready())
        mapped = rack.to_rack(BOX)
        for got, want in zip(mapped, RACK_BOX):
            self.assertAlmostEqual(want, got, places=6)

    def test_90_degree_rotated_frame_yields_the_same_rack_box(self):
        """The B8 property, proven in unit-test form (B7's 'done when')."""
        rotated_fiducials = [rotate_90_ccw(p) for p in FIDUCIALS]
        rotated_box_corners = [rotate_90_ccw((x, y)) for x, y in
                               ((BOX[0], BOX[1]), (BOX[2], BOX[3]),
                                (BOX[0], BOX[3]), (BOX[2], BOX[1]))]
        xs = [p[0] for p in rotated_box_corners]
        ys = [p[1] for p in rotated_box_corners]
        rotated_box = (min(xs), min(ys), max(xs), max(ys))

        rack = RackFrame(rotated_fiducials, RACK_SIZE)
        mapped = rack.to_rack(rotated_box)
        for got, want in zip(mapped, RACK_BOX):
            self.assertAlmostEqual(want, got, places=6)

    def test_to_rack_point_maps_fiducials_to_rack_corners(self):
        rack = RackFrame(FIDUCIALS, RACK_SIZE)
        for fiducial, corner in zip(
            FIDUCIALS, [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)]
        ):
            mapped = rack.to_rack_point(fiducial)
            self.assertAlmostEqual(corner[0], mapped[0], places=6)
            self.assertAlmostEqual(corner[1], mapped[1], places=6)

    def test_to_rack_maps_all_four_box_corners(self):
        source = [(0.0, 0.0), (200.0, 20.0), (180.0, 140.0), (20.0, 120.0)]
        rack = RackFrame(source, RACK_SIZE)
        box = (20.0, 20.0, 180.0, 120.0)
        mapped_corners = [
            rack.to_rack_point((box[0], box[1])),
            rack.to_rack_point((box[2], box[1])),
            rack.to_rack_point((box[2], box[3])),
            rack.to_rack_point((box[0], box[3])),
        ]
        mapped = rack.to_rack(box)
        self.assertAlmostEqual(min(point[0] for point in mapped_corners), mapped[0], places=6)
        self.assertAlmostEqual(min(point[1] for point in mapped_corners), mapped[1], places=6)
        self.assertAlmostEqual(max(point[0] for point in mapped_corners), mapped[2], places=6)
        self.assertAlmostEqual(max(point[1] for point in mapped_corners), mapped[3], places=6)

    def test_degenerate_fiducials_leave_the_rack_not_ready(self):
        collinear = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (5.0, 0.0)]
        rack = RackFrame(collinear, RACK_SIZE)
        self.assertFalse(rack.ready())
        with self.assertRaises(RuntimeError):
            rack.to_rack(BOX)

    def test_update_keeps_the_last_good_homography_on_a_bad_observation(self):
        rack = RackFrame(FIDUCIALS, RACK_SIZE)
        self.assertTrue(rack.ready())
        self.assertFalse(rack.update([(0.0, 0.0), (1.0, 1.0)]))  # wrong count
        self.assertTrue(rack.ready(), "one bad observation must not kill the run")
        self.assertFalse(rack.update([(0.0, float("nan")), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]))
        self.assertTrue(rack.ready())

    def test_a_good_update_after_a_bad_one_recovers(self):
        rack = RackFrame([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (5.0, 0.0)], RACK_SIZE)
        self.assertFalse(rack.ready())
        self.assertTrue(rack.update(FIDUCIALS))
        self.assertTrue(rack.ready())
        mapped = rack.to_rack(BOX)
        for got, want in zip(mapped, RACK_BOX):
            self.assertAlmostEqual(want, got, places=6)

    def test_bad_rack_size_is_rejected(self):
        with self.assertRaises(ValueError):
            RackFrame(FIDUCIALS, (0.0, 80.0))

    def test_corner_order_contract_is_documented(self):
        self.assertEqual(
            ("top_left", "top_right", "bottom_right", "bottom_left"),
            RackFrame.CORNER_ORDER,
        )
        self.assertEqual(4, len(RackFrame.CORNER_ORDER))
        self.assertTrue(all(math.isfinite(p[0]) for p in FIDUCIALS))


if __name__ == "__main__":
    unittest.main()
