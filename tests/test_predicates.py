import unittest

from har.contracts import FrameEvidence, HandObjectState, ObjectTrack, ProtocolSpec, StepSpec, Wrist, Zone
from har.protocol.predicates import (
    PREDICATES,
    PredicateState,
    hands_clear,
    hoi_cycle,
    object_left_zone,
    object_stable,
    settled,
    transfer,
)


FRAME_SIZE = (640, 480)


def make_spec() -> ProtocolSpec:
    return ProtocolSpec(
        protocol_id="TEST",
        title="Predicate test protocol",
        version="1.0.0",
        steps=(),
        objects=("tray", "lid", "red", "blue", "vial"),
        zones=(
            Zone("rack", (0.0, 0.0, 640.0, 480.0)),
            Zone("tray_slot", (200.0, 200.0, 440.0, 420.0)),
            Zone("zone_a", (50.0, 240.0, 190.0, 390.0)),
            Zone("zone_b", (450.0, 240.0, 590.0, 390.0)),
            Zone("rack_slot", (280.0, 60.0, 360.0, 170.0)),
        ),
    )


def step(predicate: str, target: str, zone: str) -> StepSpec:
    return StepSpec(
        step_id="S",
        index=1,
        title="",
        instruction="",
        predicate=predicate,
        target=target,
        zone=zone,
    )


def ev(*, objects: dict[str, ObjectTrack], hoi: dict[str, str] | None = None, hands=()) -> FrameEvidence:
    return FrameEvidence(
        frame_index=1,
        t_rel=0.0,
        frame_size=FRAME_SIZE,
        objects=objects,
        hands=hands,
        hoi=hoi or {},
        rack_ready=True,
        fps=15.0,
    )


def track(label: str, box, measured: bool = True) -> ObjectTrack:
    return ObjectTrack(label=label, box=box, measured=measured)


class PredicateTests(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec()

    def test_predicate_registry_exports_yaml_vocabulary(self):
        self.assertEqual(
            {"object_stable", "object_left_zone", "hoi_cycle", "settled", "transfer", "hands_clear"},
            set(PREDICATES),
        )

    def test_object_stable_positive_and_negative(self):
        s = step("object_stable(tray)", "tray", "rack")
        st = PredicateState()
        self.assertTrue(object_stable(ev(objects={"tray": track("tray", (230, 240, 410, 410))}), self.spec, s, st))
        self.assertFalse(object_stable(ev(objects={"tray": track("tray", (230, 240, 410, 410), measured=False)}), self.spec, s, PredicateState()))

    def test_object_left_zone_positive_and_negative(self):
        s = step("object_left_zone(lid, tray_slot)", "lid", "tray_slot")
        self.assertTrue(object_left_zone(ev(objects={"lid": track("lid", (230, 60, 410, 160))}), self.spec, s, PredicateState()))
        self.assertFalse(object_left_zone(ev(objects={"lid": track("lid", (230, 240, 410, 340))}), self.spec, s, PredicateState()))

    def test_hoi_cycle_positive_and_stationary_hand_sweep_negative(self):
        s = step("hoi_cycle(red, zone_a)", "red", "zone_a")
        st = PredicateState()
        self.assertFalse(hoi_cycle(ev(objects={"red": track("red", (250, 260, 310, 320))}, hoi={"red": HandObjectState.NEAR_OBJECT.value}), self.spec, s, st))
        self.assertFalse(hoi_cycle(ev(objects={"red": track("red", (170, 270, 230, 330))}, hoi={"red": HandObjectState.PICKED_UP.value}), self.spec, s, st))
        self.assertTrue(hoi_cycle(ev(objects={"red": track("red", (80, 280, 150, 350))}, hoi={"red": HandObjectState.RELEASED.value}), self.spec, s, st))

        sweep_state = PredicateState()
        self.assertFalse(hoi_cycle(ev(objects={"red": track("red", (250, 260, 310, 320))}, hoi={"red": HandObjectState.NEAR_OBJECT.value}), self.spec, s, sweep_state))
        self.assertFalse(hoi_cycle(ev(objects={"red": track("red", (250, 260, 310, 320))}, hoi={"red": HandObjectState.RELEASED.value}), self.spec, s, sweep_state))

    def test_settled_positive_and_negative(self):
        s = step("settled(blue, zone_b)", "blue", "zone_b")
        st = PredicateState(last_box=(484, 280, 540, 344))
        self.assertTrue(settled(ev(objects={"blue": track("blue", (486, 281, 542, 345))}), self.spec, s, st))
        moving = PredicateState(last_box=(330, 260, 390, 320))
        self.assertFalse(settled(ev(objects={"blue": track("blue", (486, 281, 542, 345))}), self.spec, s, moving))

    def test_settled_refuses_an_object_still_in_hand(self):
        # B5 cross-check regression: right after a release the interaction FSM
        # can still report PICKED_UP/CARRYING for a few frames while the object
        # is already stationary in its zone.  An in-hand object is not settled;
        # without this guard VERIFY_* fires a false OUT_OF_ORDER on a correct
        # live run.  RELEASED and IDLE must still satisfy the predicate.
        s = step("settled(blue, zone_b)", "blue", "zone_b")
        box = (486, 281, 542, 345)
        for hoi_state in (HandObjectState.PICKED_UP, HandObjectState.CARRYING):
            st = PredicateState(last_box=box)
            self.assertFalse(
                settled(ev(objects={"blue": track("blue", box)}, hoi={"blue": hoi_state.value}), self.spec, s, st),
                hoi_state,
            )
        for hoi_state in (HandObjectState.RELEASED, HandObjectState.IDLE):
            st = PredicateState(last_box=box)
            self.assertTrue(
                settled(ev(objects={"blue": track("blue", box)}, hoi={"blue": hoi_state.value}), self.spec, s, st),
                hoi_state,
            )

    def test_transfer_positive_and_negative(self):
        s = step("transfer(red, vial, rack_slot)", "vial", "rack_slot")
        st = PredicateState()
        self.assertFalse(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (120, 300, 140, 320))}, hoi={"vial": HandObjectState.IDLE.value}), self.spec, s, st))
        self.assertFalse(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (190, 190, 220, 220))}, hoi={"vial": HandObjectState.PICKED_UP.value}), self.spec, s, st))
        self.assertTrue(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (305, 105, 335, 145))}, hoi={"vial": HandObjectState.RELEASED.value}), self.spec, s, st))

        no_source = PredicateState()
        self.assertFalse(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (200, 200, 220, 220))}, hoi={"vial": HandObjectState.PICKED_UP.value}), self.spec, s, no_source))
        self.assertFalse(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (305, 105, 335, 145))}, hoi={"vial": HandObjectState.RELEASED.value}), self.spec, s, no_source))

    # -- hands_clear -------------------------------------------------------
    #
    # ``hands_clear`` must never be *vacuously* true.  ``all()`` over an empty
    # ``ev.hands`` is True in Python, and that is exactly how the live build
    # spoke step 7's alert ("hands ... inside the work envelope") while the
    # run was still on step 1 / 2 with nobody in frame: the validator's
    # later-step scan saw STOW_AND_CLOSE as already satisfied.

    HAND_IN = Wrist((320.0, 240.0), 0.9, "right")     # inside `rack`
    HAND_OUT = Wrist((700.0, 10.0), 0.9, "left")      # outside `rack`

    def test_hands_clear_is_false_when_no_hand_was_ever_seen_in_zone(self):
        s = step("hands_clear(rack)", "", "rack")
        st = PredicateState()
        # Empty scene, repeatedly: absence of evidence is not "cleared".
        for _ in range(25):
            self.assertFalse(hands_clear(ev(objects={}, hands=[]), self.spec, s, st))
        self.assertFalse(st.hands_seen_in_zone)
        # A hand that is visible but has only ever been *outside* the zone is
        # not a cleared envelope either.
        self.assertFalse(hands_clear(ev(objects={}, hands=[self.HAND_OUT]), self.spec, s, st))
        self.assertFalse(st.hands_seen_in_zone)

    def test_hands_clear_is_false_while_a_wrist_is_inside_zone(self):
        s = step("hands_clear(rack)", "", "rack")
        st = PredicateState()
        self.assertFalse(hands_clear(ev(objects={}, hands=[self.HAND_IN]), self.spec, s, st))
        self.assertTrue(st.hands_seen_in_zone)
        # One hand out, one still in: still not clear.
        self.assertFalse(hands_clear(ev(objects={}, hands=[self.HAND_OUT, self.HAND_IN]), self.spec, s, st))

    def test_hands_clear_becomes_true_only_after_hands_enter_then_leave(self):
        s = step("hands_clear(rack)", "", "rack")
        st = PredicateState()
        self.assertFalse(hands_clear(ev(objects={}, hands=[self.HAND_IN]), self.spec, s, st))
        # Hands leave the zone: both "visible outside" and "not visible at
        # all" count as clear once the envelope was previously occupied.  The
        # validator's hold_frames then supplies the required persistence.
        for _ in range(20):
            self.assertTrue(hands_clear(ev(objects={}, hands=[self.HAND_OUT]), self.spec, s, st))
        for _ in range(20):
            self.assertTrue(hands_clear(ev(objects={}, hands=[]), self.spec, s, st))
        # Re-entering makes it false again; leaving again makes it true.
        self.assertFalse(hands_clear(ev(objects={}, hands=[self.HAND_IN]), self.spec, s, st))
        self.assertTrue(hands_clear(ev(objects={}, hands=[]), self.spec, s, st))

    def test_hands_clear_state_does_not_leak_across_fresh_state(self):
        # The validator hands every step (and every reset) a fresh
        # PredicateState, so the latch must live there and start False.
        s = step("hands_clear(rack)", "", "rack")
        st = PredicateState()
        hands_clear(ev(objects={}, hands=[self.HAND_IN]), self.spec, s, st)
        self.assertTrue(hands_clear(ev(objects={}, hands=[]), self.spec, s, st))
        self.assertFalse(hands_clear(ev(objects={}, hands=[]), self.spec, s, PredicateState()))

    def test_hands_clear_unknown_zone_is_false(self):
        s = step("hands_clear(nope)", "", "nope")
        st = PredicateState()
        self.assertFalse(hands_clear(ev(objects={}, hands=[]), self.spec, s, st))
        self.assertFalse(hands_clear(ev(objects={}, hands=[self.HAND_IN]), self.spec, s, st))


if __name__ == "__main__":
    unittest.main()
