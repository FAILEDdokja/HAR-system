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

    def test_transfer_positive_and_negative(self):
        s = step("transfer(red, vial, rack_slot)", "vial", "rack_slot")
        st = PredicateState()
        self.assertFalse(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (120, 300, 140, 320))}, hoi={"vial": HandObjectState.IDLE.value}), self.spec, s, st))
        self.assertFalse(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (190, 190, 220, 220))}, hoi={"vial": HandObjectState.PICKED_UP.value}), self.spec, s, st))
        self.assertTrue(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (305, 105, 335, 145))}, hoi={"vial": HandObjectState.RELEASED.value}), self.spec, s, st))

        no_source = PredicateState()
        self.assertFalse(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (200, 200, 220, 220))}, hoi={"vial": HandObjectState.PICKED_UP.value}), self.spec, s, no_source))
        self.assertFalse(transfer(ev(objects={"red": track("red", (100, 280, 156, 344)), "vial": track("vial", (305, 105, 335, 145))}, hoi={"vial": HandObjectState.RELEASED.value}), self.spec, s, no_source))

    def test_hands_clear_positive_and_negative(self):
        s = step("hands_clear(rack)", "", "rack")
        self.assertTrue(hands_clear(ev(objects={}, hands=[Wrist((700.0, 10.0), 0.9, "left")]), self.spec, s, PredicateState()))
        self.assertFalse(hands_clear(ev(objects={}, hands=[Wrist((320.0, 240.0), 0.9, "right")]), self.spec, s, PredicateState()))


if __name__ == "__main__":
    unittest.main()
