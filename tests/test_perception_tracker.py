"""Single-target association - moved from test_bottle_monitor.py."""

import unittest

from har.contracts import Detection
from har.perception.tracker import (
    SingleTargetTracker,
    TrackerConfig,
    TrackerRegistry,
)

FRAME = (640, 480)


class SingleTargetTrackerTests(unittest.TestCase):
    def test_detector_id_changes_do_not_create_a_second_logical_object(self):
        tracker = SingleTargetTracker("red_box", TrackerConfig(acquire_frames=2, min_area_fraction=0.0))
        tracker.update([Detection((100, 100, 150, 240), 0.9, "red_box", 10)], FRAME)
        acquired = tracker.update([Detection((104, 100, 154, 240), 0.9, "red_box", 15)], FRAME)
        self.assertEqual("red_box", acquired.label)
        self.assertTrue(acquired.measured)

        # A high-confidence decoy far away must not steal the identity.
        missing = tracker.update([Detection((500, 20, 580, 180), 0.99, "red_box", 20)], FRAME)
        self.assertEqual("red_box", missing.label)
        self.assertFalse(missing.measured)

        recovered = tracker.update([Detection((110, 100, 160, 240), 0.85, "red_box", 23)], FRAME)
        self.assertEqual("red_box", recovered.label)
        self.assertTrue(recovered.measured)

    def test_low_confidence_detections_never_acquire(self):
        tracker = SingleTargetTracker("tray", TrackerConfig(acquire_frames=1))
        result = tracker.update([Detection((100, 100, 150, 240), 0.10, "tray", 1)], FRAME)
        self.assertIsNone(result.box)
        self.assertFalse(result.measured)

    def test_roi_gating_rejects_detections_outside_the_envelope(self):
        config = TrackerConfig(acquire_frames=1, min_area_fraction=0.0, roi=(300, 300, 400, 400))
        tracker = SingleTargetTracker("vial", config)
        result = tracker.update([Detection((0, 0, 40, 40), 0.9, "vial", 1)], FRAME)
        self.assertIsNone(result.box)

    def test_label_filter_ignores_other_classes(self):
        config = TrackerConfig(acquire_frames=1, min_area_fraction=0.0, labels=("tray",))
        tracker = SingleTargetTracker("tray", config)
        result = tracker.update([Detection((100, 100, 150, 240), 0.9, "bottle", 1)], FRAME)
        self.assertIsNone(result.box)


class TrackerRegistryTests(unittest.TestCase):
    def test_registry_holds_one_identity_per_label(self):
        registry = TrackerRegistry(
            ("tray", "red_box", "blue_box"),
            TrackerConfig(acquire_frames=1, min_area_fraction=0.0, max_area_fraction=1.0),
        )
        self.assertEqual(["tray", "red_box", "blue_box"], registry.labels())

        results = registry.update_all(
            [
                Detection((50, 50, 200, 200), 0.9, "tray", 1),
                Detection((250, 250, 320, 320), 0.8, "red_box", 2),
                Detection((400, 250, 470, 320), 0.8, "blue_box", 3),
            ],
            FRAME,
        )
        # No label filter is set, so each tracker may claim any detection; the
        # guarantee that matters is that identities stay distinct and stable.
        self.assertEqual({"tray", "red_box", "blue_box"}, set(results))
        self.assertEqual(
            ["tray", "red_box", "blue_box"],
            [registry[label].label for label in ("tray", "red_box", "blue_box")],
        )

    def test_reset_keeps_labels_but_forgets_boxes(self):
        registry = TrackerRegistry(("tray",), TrackerConfig(acquire_frames=1, min_area_fraction=0.0))
        registry.update_all([Detection((50, 50, 200, 200), 0.9, "tray", 1)], FRAME)
        self.assertIsNotNone(registry["tray"].box)
        registry.reset_for_new_scene()
        self.assertIsNone(registry["tray"].box)
        self.assertEqual(["tray"], registry.labels())


if __name__ == "__main__":
    unittest.main()
