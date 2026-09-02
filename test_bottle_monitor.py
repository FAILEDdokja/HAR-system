import unittest

from bottle_monitor import (
    Detection,
    InteractionConfig,
    InteractionMachine,
    InteractionState,
    SingleBottleTracker,
    TrackerConfig,
    point_to_box_distance,
)


class SingleBottleMonitorTests(unittest.TestCase):
    def test_wrist_distance_is_zero_on_the_box_edge(self):
        self.assertEqual(0.0, point_to_box_distance((100, 120), (100, 100, 180, 220)))
        self.assertEqual(20.0, point_to_box_distance((80, 120), (100, 100, 180, 220)))

    def test_yolo_id_changes_do_not_create_a_second_logical_bottle(self):
        tracker = SingleBottleTracker(TrackerConfig(acquire_frames=2, min_area_fraction=0.0))
        initial = Detection((100, 100, 150, 240), 0.9, 10)
        tracker.update([initial], (640, 480))
        acquired = tracker.update([Detection((104, 100, 154, 240), 0.9, 15)], (640, 480))
        self.assertEqual(1, acquired.logical_id)
        missing = tracker.update([Detection((500, 20, 580, 180), 0.99, 20)], (640, 480))
        self.assertEqual(1, missing.logical_id)
        self.assertFalse(missing.measured)
        recovered = tracker.update([Detection((110, 100, 160, 240), 0.85, 23)], (640, 480))
        self.assertEqual(1, recovered.logical_id)
        self.assertTrue(recovered.measured)

    def test_pickup_carry_release_sequence(self):
        machine = InteractionMachine(
            InteractionConfig(
                near_frames=1,
                pickup_frames=1,
                picked_up_frames=1,
                release_frames=1,
                stable_frames=1,
                movement_fraction=0.001,
            )
        )
        machine.update(True, (100, 100, 150, 220), [(95, 130)], (640, 480))
        self.assertEqual(InteractionState.NEAR_OBJECT, machine.state)
        machine.update(True, (120, 100, 170, 220), [(115, 130)], (640, 480))
        self.assertEqual(InteractionState.PICKED_UP, machine.state)
        machine.update(True, (140, 100, 190, 220), [(135, 130)], (640, 480))
        self.assertEqual(InteractionState.CARRYING, machine.state)
        machine.update(True, (140, 100, 190, 220), [(400, 400)], (640, 480))
        self.assertEqual(InteractionState.RELEASED, machine.state)
        machine.update(True, (140, 100, 190, 220), [(400, 400)], (640, 480))
        self.assertEqual(InteractionState.IDLE, machine.state)


if __name__ == "__main__":
    unittest.main()