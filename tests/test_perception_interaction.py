"""Hand-object interaction FSM - moved from test_bottle_monitor.py."""

import unittest

from har.contracts import HandObjectState
from har.perception.interaction import (
    InteractionConfig,
    InteractionMachine,
    InteractionState,
)

FRAME = (640, 480)


def fast_machine(label: str = "red_box") -> InteractionMachine:
    return InteractionMachine(
        label,
        InteractionConfig(
            near_frames=1,
            pickup_frames=1,
            picked_up_frames=1,
            release_frames=1,
            stable_frames=1,
            movement_fraction=0.001,
        ),
    )


class InteractionMachineTests(unittest.TestCase):
    def test_pickup_carry_release_sequence(self):
        machine = fast_machine()
        machine.update(True, (100, 100, 150, 220), [(95, 130)], FRAME)
        self.assertEqual(InteractionState.NEAR_OBJECT, machine.state)
        machine.update(True, (120, 100, 170, 220), [(115, 130)], FRAME)
        self.assertEqual(InteractionState.PICKED_UP, machine.state)
        machine.update(True, (140, 100, 190, 220), [(135, 130)], FRAME)
        self.assertEqual(InteractionState.CARRYING, machine.state)
        machine.update(True, (140, 100, 190, 220), [(400, 400)], FRAME)
        self.assertEqual(InteractionState.RELEASED, machine.state)
        machine.update(True, (140, 100, 190, 220), [(400, 400)], FRAME)
        self.assertEqual(InteractionState.IDLE, machine.state)

    def test_identity_is_locked_while_the_object_is_held(self):
        machine = fast_machine()
        self.assertFalse(machine.identity_locked())
        machine.update(True, (100, 100, 150, 220), [(95, 130)], FRAME)
        machine.update(True, (120, 100, 170, 220), [(115, 130)], FRAME)
        self.assertEqual(InteractionState.PICKED_UP, machine.state)
        self.assertTrue(machine.identity_locked())

    def test_a_hand_waving_past_is_not_a_pickup(self):
        machine = fast_machine()
        # Hand sweeps past while the object stays perfectly still: the relative
        # movement is large, so the machine must not reach PICKED_UP.
        machine.update(True, (100, 100, 150, 220), [(95, 130)], FRAME)
        machine.update(True, (100, 100, 150, 220), [(200, 130)], FRAME)
        machine.update(True, (100, 100, 150, 220), [(300, 130)], FRAME)
        self.assertNotEqual(InteractionState.PICKED_UP, machine.state)

    def test_missing_object_yields_idle_result(self):
        machine = fast_machine()
        result = machine.update(False, None, [], FRAME)
        self.assertEqual("red_box", result.label)
        self.assertIsNone(result.closest_hand)

    def test_state_values_match_the_frozen_contract_enum(self):
        """Track A imports only contracts.HandObjectState, never this module."""
        for state in InteractionState:
            self.assertIn(state.value, {s.value for s in HandObjectState})


if __name__ == "__main__":
    unittest.main()
