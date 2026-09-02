"""Geometry helpers - moved from test_bottle_monitor.py, unchanged assertions."""

import unittest

from har.perception.geometry import (
    box_center_inside,
    box_iou,
    center,
    point_to_box_distance,
)


class GeometryTests(unittest.TestCase):
    def test_wrist_distance_is_zero_on_the_box_edge(self):
        self.assertEqual(0.0, point_to_box_distance((100, 120), (100, 100, 180, 220)))
        self.assertEqual(20.0, point_to_box_distance((80, 120), (100, 100, 180, 220)))

    def test_distance_is_zero_inside_the_box(self):
        self.assertEqual(0.0, point_to_box_distance((140, 150), (100, 100, 180, 220)))

    def test_center_and_iou(self):
        self.assertEqual((140.0, 160.0), center((100, 100, 180, 220)))
        self.assertEqual(1.0, box_iou((0, 0, 10, 10), (0, 0, 10, 10)))
        self.assertEqual(0.0, box_iou((0, 0, 10, 10), (20, 20, 30, 30)))

    def test_zone_containment_uses_the_center(self):
        region = (0.0, 0.0, 100.0, 100.0)
        self.assertTrue(box_center_inside((40, 40, 60, 60), region))
        # Overlaps the region but its centre is outside it.
        self.assertFalse(box_center_inside((80, 80, 200, 200), region))


if __name__ == "__main__":
    unittest.main()
