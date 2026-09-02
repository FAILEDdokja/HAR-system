"""Unit tests for the sequence validator (Track A, step A4).

A4 acceptance (plan §5): replaying ``tests/fixtures/evidence_correct.json``
through ``SequenceValidator`` against ``protocols/pts01.yaml`` must produce
exactly eight ``COMPLETED`` events in index order plus one
``PROTOCOL_COMPLETE``, with zero violations.

The violation-semantics tests below (rules 3 and 4) replay the other two A1
fixtures.  Step A5 extends this file with the timeout and ``measured=False``
tests.
"""

import json
import unittest
from datetime import datetime
from pathlib import Path

from har.contracts import FrameEvidence, ObjectTrack, StepEvent, Wrist
from har.protocol.spec import load_protocol
from har.protocol.validator import SequenceValidator

REPO = Path(__file__).resolve().parents[1]
PTS01 = REPO / "protocols" / "pts01.yaml"
FIXTURES = REPO / "tests" / "fixtures"
FRAME_SIZE = (640, 480)

VIOLATION_EVENTS = {"SKIPPED", "OUT_OF_ORDER", "TIMEOUT"}


def evidence_from_dict(d: dict) -> FrameEvidence:
    """Rebuild one ``FrameEvidence`` from a ``FrameEvidence.to_dict()`` row."""
    return FrameEvidence(
        frame_index=d["frame_index"],
        t_rel=d["t_rel"],
        frame_size=tuple(d["frame_size"]),
        objects={
            label: ObjectTrack(
                label=od["label"],
                box=tuple(od["box"]) if od["box"] is not None else None,
                measured=od["measured"],
                lost_frames=od.get("lost_frames", 0),
            )
            for label, od in d["objects"].items()
        },
        hands=tuple(
            Wrist(tuple(h["point"]), h["confidence"], h["side"], h.get("person_id", 0))
            for h in d["hands"]
        ),
        hoi=dict(d["hoi"]),
        rack_ready=d["rack_ready"],
        fps=d["fps"],
    )


def load_frames(name: str) -> list[FrameEvidence]:
    raw = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return [evidence_from_dict(d) for d in raw]


def make_validator() -> SequenceValidator:
    return SequenceValidator(load_protocol(PTS01, FRAME_SIZE))


def replay(frames: list[FrameEvidence]):
    """Run the frames; return (validator, all events)."""
    validator = make_validator()
    events: list[StepEvent] = []
    for frame in frames:
        events.extend(validator.update(frame))
    return validator, events


def events_of(events: list[StepEvent], kind: str) -> list[StepEvent]:
    return [e for e in events if e.event == kind]


class ValidatorCorrectRunTests(unittest.TestCase):
    """A4 done-when: the correct run completes 8/8 with zero violations."""

    @classmethod
    def setUpClass(cls):
        cls.frames = load_frames("evidence_correct.json")
        cls.validator, cls.events = replay(cls.frames)

    def test_eight_completions_in_index_order(self):
        completed = events_of(self.events, "COMPLETED")
        self.assertEqual(8, len(completed))
        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8], [e.step_index for e in completed])
        self.assertEqual(
            [
                "PRESENT_TRAY",
                "OPEN_TRAY",
                "EXTRACT_RED",
                "VERIFY_RED_PLACED",
                "EXTRACT_BLUE",
                "VERIFY_BLUE_PLACED",
                "SAMPLE_TRANSFER",
                "STOW_AND_CLOSE",
            ],
            [e.step_id for e in completed],
        )
        self.assertTrue(all(e.status == "OK" for e in completed))

    def test_single_protocol_complete_at_the_end(self):
        complete = events_of(self.events, "PROTOCOL_COMPLETE")
        self.assertEqual(1, len(complete))
        self.assertIs(self.events[-1], complete[0])
        self.assertEqual("STOW_AND_CLOSE", complete[0].step_id)
        self.assertEqual(8, complete[0].step_index)
        self.assertEqual(
            "Protocol PTS-01 completed with no violations",
            complete[0].message,
        )

    def test_zero_violations(self):
        self.assertEqual(0, len(events_of(self.events, "SKIPPED")))
        self.assertEqual(0, len(events_of(self.events, "OUT_OF_ORDER")))
        self.assertEqual(0, len(events_of(self.events, "TIMEOUT")))
        self.assertEqual((), self.validator.violations)

    def test_every_step_started_exactly_once(self):
        started = events_of(self.events, "STARTED")
        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8], [e.step_index for e in started])
        self.assertIs(self.events[0], started[0])
        self.assertEqual("PRESENT_TRAY", started[0].step_id)
        self.assertEqual("IN_PROGRESS", started[0].status)

    def test_finished_state_and_introspection(self):
        self.assertTrue(self.validator.finished)
        self.assertIsNone(self.validator.current)
        self.assertEqual(8, len(self.validator.completed_steps))
        status = self.validator.status()
        self.assertEqual("COMPLETE", status.state)
        self.assertEqual("PTS-01", status.protocol_id)
        self.assertEqual(8, status.current_step_index)
        self.assertEqual("", status.next_step_id)
        self.assertEqual((), status.skipped)
        self.assertEqual((), status.violations)
        self.assertEqual(15.0, status.fps)

    def test_update_returns_empty_after_completion(self):
        # Rule 6: once complete, update() is a no-op for the rest of the run.
        self.assertEqual([], self.validator.update(self.frames[-1]))
        self.assertEqual([], self.validator.update(self.frames[0]))

    def test_events_carry_the_frame_that_caused_them(self):
        by_index = {f.frame_index: f for f in self.frames}
        seen = set()
        for event in self.events:
            frame = by_index[event.frame_index]
            self.assertEqual(frame.t_rel, event.t_rel)
            self.assertLessEqual(0.0, event.t_rel)
            datetime.fromisoformat(event.t_iso)  # must parse
            seen.add(event.frame_index)
        self.assertTrue(seen)  # events were spread across the run


class ValidatorViolationSemanticsTests(unittest.TestCase):
    """Rules 3 and 4: skip and out-of-order detection on the A1 fixtures."""

    def test_skip_fixture_yields_exactly_one_skipped(self):
        _, events = replay(load_frames("evidence_skip.json"))
        skipped = events_of(events, "SKIPPED")
        self.assertEqual(1, len(skipped))
        self.assertEqual("EXTRACT_RED", skipped[0].step_id)
        self.assertEqual(3, skipped[0].step_index)
        self.assertEqual("VIOLATION", skipped[0].status)
        # Message comes from the skipped step's voice_alert (pts01.yaml).
        self.assertEqual(
            "Step 3 skipped. The red box must go to zone A before the blue box.",
            skipped[0].message,
        )
        out_of_order = events_of(events, "OUT_OF_ORDER")
        self.assertEqual(1, len(out_of_order))
        self.assertEqual("EXTRACT_BLUE", out_of_order[0].step_id)
        self.assertEqual(
            "Out of sequence. The red box must be placed before the blue box.",
            out_of_order[0].message,
        )
        # The skip re-baselines the cursor onto the satisfied later step.
        self.assertEqual(out_of_order[0].frame_index, skipped[0].frame_index)
        started = events_of(events, "STARTED")
        self.assertIn("EXTRACT_BLUE", [e.step_id for e in started])
        self.assertNotIn("VERIFY_RED_PLACED", [e.step_id for e in started])

    def test_wrong_order_fixture_yields_exactly_one_out_of_order(self):
        validator, events = replay(load_frames("evidence_wrong_order.json"))
        self.assertEqual(1, len(events_of(events, "OUT_OF_ORDER")))
        self.assertEqual(1, len(events_of(events, "SKIPPED")))
        self.assertEqual(0, len(events_of(events, "TIMEOUT")))
        # No protocol completion: the vial transfer never saw pickup evidence.
        self.assertEqual(0, len(events_of(events, "PROTOCOL_COMPLETE")))
        self.assertFalse(validator.finished)
        self.assertEqual(("EXTRACT_BLUE", "EXTRACT_RED"), validator.violations)

    def test_one_shot_out_of_order_detection(self):
        # After the first violation episode the validator must not re-alert:
        # replaying the rest of the wrong-order tail (hands leaving the
        # envelope while a step is still pending) adds no further events.
        frames = load_frames("evidence_wrong_order.json")
        validator, events = replay(frames)
        violations_before = validator.violations
        # The tail frame has both hands out of the envelope while the vial
        # transfer is still pending: an extra OUT_OF_ORDER/SKIPPED here would
        # be a false alarm.
        more = validator.update(frames[-1])
        self.assertEqual(0, len(more))
        self.assertEqual(violations_before, validator.violations)


class ValidatorInterfaceTests(unittest.TestCase):
    def test_reset_restarts_the_run(self):
        frames = load_frames("evidence_correct.json")
        validator, _ = replay(frames)
        validator.reset()
        self.assertFalse(validator.finished)
        self.assertIsNone(validator.current)
        self.assertEqual((), validator.completed_steps)
        events = validator.update(frames[0])
        self.assertEqual(1, len(events))
        self.assertEqual("STARTED", events[0].event)
        self.assertEqual("PRESENT_TRAY", events[0].step_id)

    def test_status_mid_run(self):
        frames = load_frames("evidence_correct.json")
        validator = make_validator()
        validator.update(frames[0])
        status = validator.status()
        self.assertEqual("IN_PROGRESS", status.state)
        self.assertEqual("PRESENT_TRAY", status.current_step_id)
        self.assertEqual(1, status.current_step_index)
        self.assertEqual("OPEN_TRAY", status.next_step_id)
        self.assertIn("tray lid", status.next_instruction)
        # A violation-free mid-run has no alert and no violations.
        self.assertEqual("", status.last_alert)
        self.assertEqual((), status.violations)

    def test_constructor_rejects_empty_protocol(self):
        from har.contracts import ProtocolSpec

        with self.assertRaises(ValueError):
            SequenceValidator(ProtocolSpec("EMPTY", "t", "0", steps=()))


if __name__ == "__main__":
    unittest.main()
