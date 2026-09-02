"""Unit tests for the sequence validator and its UiStatus producer (Track A).

A4 acceptance (plan §5): replaying ``tests/fixtures/evidence_correct.json``
through ``SequenceValidator`` against ``protocols/pts01.yaml`` must produce
exactly eight ``COMPLETED`` events in index order plus one
``PROTOCOL_COMPLETE``, with zero violations.

The violation-semantics tests below (rules 3 and 4) replay the other two A1
fixtures.  Step A5 (plan §5) extends this file with the remaining done-when
items: a stalled step emits ``TIMEOUT`` once and not twice (rule 5), and a
``measured=False`` track never completes a step (rule 7).  Those cases need
degenerate evidence (a step that never happens; a tracker that only coasts)
that the A1 fixtures deliberately do not model, so they synthesise
``FrameEvidence`` directly instead of replaying a fixture file.

Step A7 (plan §5) is the ``UiStatus`` producer: ``ValidatorUiStatusTests``
below asserts the *full* snapshot mid-run and again at completion — every
field of the frozen dataclass, by exact equality — plus the JSON shape the
``/status`` poller renders and the purity that makes 2 Hz polling safe.
"""

import json
import unittest
from datetime import datetime
from pathlib import Path

from har.contracts import (
    CONTRACT_VERSION,
    FrameEvidence,
    ObjectTrack,
    StepEvent,
    UiStatus,
    Wrist,
)
try:
    from har.protocol.spec import load_protocol
except ImportError:  # pragma: no cover - bare interpreter without PyYAML
    load_protocol = None
from har.protocol.validator import SequenceValidator

REPO = Path(__file__).resolve().parents[1]
PTS01 = REPO / "protocols" / "pts01.yaml"
FIXTURES = REPO / "tests" / "fixtures"
FRAME_SIZE = (640, 480)

VIOLATION_EVENTS = {"SKIPPED", "OUT_OF_ORDER", "TIMEOUT"}

PTS01_TITLE = "Payload Tray Sorting & Sample Transfer"


def not_started_ui_status() -> UiStatus:
    """The exact full snapshot a fresh (or freshly reset) validator exposes.

    Track C renders before the first frame arrives, so even the empty
    snapshot is a complete UiStatus: the checklist points at step 1 with
    state ``NOT_STARTED`` and every list field is an empty tuple.
    """
    return UiStatus(
        protocol_id="PTS-01",
        protocol_title=PTS01_TITLE,
        current_step_id="PRESENT_TRAY",
        current_step_index=1,
        next_step_id="OPEN_TRAY",
        next_instruction="Lift the tray lid clear of the tray slot.",
        completed=(),
        skipped=(),
        violations=(),
        state="NOT_STARTED",
        t_rel=0.0,
        fps=0.0,
        last_alert="",
        contract_version=CONTRACT_VERSION,
    )


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


# ---------------------------------------------------------------------------
# Synthetic evidence for the A5 tests.
#
# The A1 fixtures model runs where every step eventually happens, so they can
# never exercise rule 5 (a step that stalls past ``timeout_s``) or rule 7 (a
# tracker that only ever coasts).  These helpers build the degenerate frames
# directly.  A wrist is kept inside the rack envelope on every frame so that
# step 8 (``hands_clear``) cannot read an empty scene as satisfied and trip
# the out-of-order scan while we are stalling step 1.
# ---------------------------------------------------------------------------

HAND_IN_ENVELOPE = (Wrist((320.0, 430.0), 0.95, "right"),)
TRAY_BOX = (230.0, 240.0, 410.0, 410.0)  # centre well inside rack_roi
FPS = 15.0


def tray_frame(frame_index: int, *, present: bool = True, measured: bool = True) -> FrameEvidence:
    """One frame of evidence about the tray only (all other props absent)."""
    objects = {}
    if present:
        objects["tray"] = ObjectTrack(label="tray", box=TRAY_BOX, measured=measured)
    return FrameEvidence(
        frame_index=frame_index,
        t_rel=frame_index / FPS,
        frame_size=FRAME_SIZE,
        objects=objects,
        hands=HAND_IN_ENVELOPE,
        hoi={"tray": "IDLE"} if present else {},
        rack_ready=True,
        fps=FPS,
    )


@unittest.skipIf(load_protocol is None, "PyYAML is not installed")
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


@unittest.skipIf(load_protocol is None, "PyYAML is not installed")
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


@unittest.skipIf(load_protocol is None, "PyYAML is not installed")
class ValidatorTimeoutTests(unittest.TestCase):
    """Rule 5 (A5 done-when): a stalled step emits TIMEOUT once and not twice."""

    def test_stalled_step_times_out_exactly_once(self):
        validator = make_validator()
        events: list[StepEvent] = []
        # PRESENT_TRAY has timeout_s=60.  Stall it: the tray never appears.
        # Frame 0 enters the step (entered_at=0.0); every following second we
        # deliver one empty frame, running 30 s *past* the deadline so the
        # validator has plenty of chances to double-fire.
        for second in range(0, 91):
            frame = tray_frame(int(second * FPS), present=False)
            events.extend(validator.update(frame))

        timeouts = events_of(events, "TIMEOUT")
        self.assertEqual(1, len(timeouts))
        self.assertEqual("PRESENT_TRAY", timeouts[0].step_id)
        self.assertEqual(1, timeouts[0].step_index)
        self.assertEqual("VIOLATION", timeouts[0].status)
        # It fires on the first frame past the deadline, not at the end.
        self.assertEqual(61.0, timeouts[0].t_rel)
        self.assertEqual(("PRESENT_TRAY",), validator.violations)
        # Stalling is not skipping: no other violation kinds appear.
        self.assertEqual(0, len(events_of(events, "SKIPPED")))
        self.assertEqual(0, len(events_of(events, "OUT_OF_ORDER")))
        # The step stays current, and nothing fired after the one TIMEOUT:
        # the whole stalled run produced exactly STARTED + TIMEOUT.
        self.assertEqual("PRESENT_TRAY", validator.current.step_id)
        self.assertEqual(["STARTED", "TIMEOUT"], [e.event for e in events])

        # ...and can still complete when the work is finally done
        # (hold_frames=15 consecutive video frames of a stable tray).
        base = int(91 * FPS)
        late: list[StepEvent] = []
        for i in range(16):
            late.extend(validator.update(tray_frame(base + i)))
        completed = events_of(late, "COMPLETED")
        self.assertEqual(1, len(completed))
        self.assertEqual("PRESENT_TRAY", completed[0].step_id)
        # Completing late does not clear the recorded violation.
        self.assertEqual(("PRESENT_TRAY",), validator.violations)
        # And no further TIMEOUT was emitted for the step on the way out.
        self.assertEqual(0, len(events_of(late, "TIMEOUT")))

    def test_timeout_reflected_in_status(self):
        validator = make_validator()
        validator.update(tray_frame(0, present=False))
        events = validator.update(tray_frame(int(61 * FPS), present=False))
        timeouts = events_of(events, "TIMEOUT")
        self.assertEqual(1, len(timeouts))
        status = validator.status()
        self.assertEqual("IN_PROGRESS", status.state)
        self.assertEqual("PRESENT_TRAY", status.current_step_id)
        self.assertEqual(("PRESENT_TRAY",), status.violations)
        self.assertEqual(timeouts[0].message, status.last_alert)


@unittest.skipIf(load_protocol is None, "PyYAML is not installed")
class ValidatorMeasuredFalseTests(unittest.TestCase):
    """Rule 7 (A5 done-when): a ``measured=False`` track never completes a step."""

    def test_coasting_track_never_completes_a_step(self):
        validator = make_validator()
        events: list[StepEvent] = []
        # The tray sits perfectly inside the rack envelope for 45 consecutive
        # video frames — three times PRESENT_TRAY's hold_frames=15 — but the
        # tracker is coasting the whole time.  A predicted box is an estimate;
        # it must not confirm the step.
        for i in range(45):
            events.extend(validator.update(tray_frame(i, measured=False)))

        self.assertEqual(0, len(events_of(events, "COMPLETED")))
        self.assertEqual("PRESENT_TRAY", validator.current.step_id)
        self.assertEqual((), validator.completed_steps)
        # Only the initial STARTED was emitted; coasting is not a violation.
        self.assertEqual(1, len(events))
        self.assertEqual("STARTED", events[0].event)

    def test_completion_requires_a_fresh_measured_hold(self):
        validator = make_validator()
        for i in range(45):
            validator.update(tray_frame(i, measured=False))
        # Once real measurements resume, the hold must be re-earned from the
        # measured frames alone: 14 measured frames after 45 coasted ones is
        # still short of hold_frames=15...
        events: list[StepEvent] = []
        for i in range(45, 59):
            events.extend(validator.update(tray_frame(i, measured=True)))
        self.assertEqual(0, len(events_of(events, "COMPLETED")))
        # ...and the 15th measured frame completes the step.
        events = validator.update(tray_frame(59, measured=True))
        completed = events_of(events, "COMPLETED")
        self.assertEqual(1, len(completed))
        self.assertEqual("PRESENT_TRAY", completed[0].step_id)
        self.assertEqual((), validator.violations)

    def test_coasting_track_never_triggers_a_skip_jump(self):
        # Second half of rule 7: a later step must not be *jumped to* on a
        # coasted box either.  Present a coasting red box already sitting in
        # zone A (step 3's work, apparently done) while step 1 is unsatisfied.
        validator = make_validator()
        red_in_zone_a = ObjectTrack(
            label="red_box", box=(100.0, 250.0, 140.0, 290.0), measured=False
        )
        events: list[StepEvent] = []
        for i in range(30):
            events.extend(
                validator.update(
                    FrameEvidence(
                        frame_index=i,
                        t_rel=i / FPS,
                        frame_size=FRAME_SIZE,
                        objects={"red_box": red_in_zone_a},
                        hands=HAND_IN_ENVELOPE,
                        hoi={"red_box": "RELEASED"},
                        rack_ready=True,
                        fps=FPS,
                    )
                )
            )
        self.assertEqual(0, len(events_of(events, "OUT_OF_ORDER")))
        self.assertEqual(0, len(events_of(events, "SKIPPED")))
        self.assertEqual("PRESENT_TRAY", validator.current.step_id)


@unittest.skipIf(load_protocol is None, "PyYAML is not installed")
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


@unittest.skipIf(load_protocol is None, "PyYAML is not installed")
class ValidatorUiStatusTests(unittest.TestCase):
    """A7 done-when: the full ``UiStatus``, mid-run and again at completion.

    Each snapshot test asserts exact equality against a hand-written
    ``UiStatus``, so *every* field of the frozen contract is pinned —
    protocol identity, the current step, the announced next step and its
    instruction, the ``completed``/``skipped``/``violations`` tuples,
    ``state``, ``last_alert``, ``t_rel``, ``fps`` and the contract version.
    A field the validator forgot to populate, or populated from the wrong
    place, changes one of these snapshots and fails here, not in the demo.
    """

    # -- the two snapshots the done-when names --------------------------

    def test_full_status_mid_run(self):
        frames = load_frames("evidence_correct.json")
        validator = make_validator()
        for frame in frames[:5]:  # through f60/t4.0: three steps confirmed
            validator.update(frame)
        self.assertEqual(
            UiStatus(
                protocol_id="PTS-01",
                protocol_title=PTS01_TITLE,
                current_step_id="VERIFY_RED_PLACED",
                current_step_index=4,
                next_step_id="EXTRACT_BLUE",
                # next_instruction is the *next* step's instruction, verbatim
                # from pts01.yaml — not its title and not its voice_prompt.
                next_instruction="Pick the blue box out of the tray and place it in zone B.",
                completed=("PRESENT_TRAY", "OPEN_TRAY", "EXTRACT_RED"),
                skipped=(),
                violations=(),
                state="IN_PROGRESS",
                t_rel=4.0,  # carried from the last evidence frame consumed
                fps=15.0,
                last_alert="",  # a clean run has never alerted
                contract_version=CONTRACT_VERSION,
            ),
            validator.status(),
        )

    def test_full_status_at_completion(self):
        validator, _ = replay(load_frames("evidence_correct.json"))
        self.assertEqual(
            UiStatus(
                protocol_id="PTS-01",
                protocol_title=PTS01_TITLE,
                # At completion the checklist stays on the final step.
                current_step_id="STOW_AND_CLOSE",
                current_step_index=8,
                next_step_id="",
                next_instruction="",
                completed=(
                    "PRESENT_TRAY",
                    "OPEN_TRAY",
                    "EXTRACT_RED",
                    "VERIFY_RED_PLACED",
                    "EXTRACT_BLUE",
                    "VERIFY_BLUE_PLACED",
                    "SAMPLE_TRANSFER",
                    "STOW_AND_CLOSE",
                ),
                skipped=(),
                violations=(),
                state="COMPLETE",
                t_rel=14.5,
                fps=15.0,
                last_alert="",
                contract_version=CONTRACT_VERSION,
            ),
            validator.status(),
        )

    # -- the run states around those two snapshots ----------------------

    def test_full_status_before_the_first_frame(self):
        self.assertEqual(not_started_ui_status(), make_validator().status())

    def test_full_status_on_a_flagged_run(self):
        # The skip fixture is the run where a GUI must switch from progress
        # to its violation rendering: skipped + violations non-empty and
        # last_alert carrying the alert the speaker already said.
        frames = load_frames("evidence_skip.json")
        validator, events = replay(frames)
        last_violation = [e for e in events if e.status == "VIOLATION"][-1]
        self.assertEqual(
            UiStatus(
                protocol_id="PTS-01",
                protocol_title=PTS01_TITLE,
                # The cursor re-baselined onto EXTRACT_BLUE and ran on; by
                # the fixture's end step 7 is current.
                current_step_id="SAMPLE_TRANSFER",
                current_step_index=7,
                next_step_id="STOW_AND_CLOSE",
                next_instruction=(
                    "Return the lid to the tray and withdraw both hands from the rack envelope."
                ),
                completed=("PRESENT_TRAY", "OPEN_TRAY", "EXTRACT_BLUE", "VERIFY_BLUE_PLACED"),
                skipped=("EXTRACT_RED",),
                # First-occurrence order: the OUT_OF_ORDER alert was noted
                # on EXTRACT_BLUE before EXTRACT_RED was marked skipped.
                violations=("EXTRACT_BLUE", "EXTRACT_RED"),
                state="IN_PROGRESS",
                t_rel=9.5,
                fps=15.0,
                last_alert=last_violation.message,
                contract_version=CONTRACT_VERSION,
            ),
            validator.status(),
        )
        self.assertEqual(
            "Step 3 skipped. The red box must go to zone A before the blue box.",
            validator.status().last_alert,
        )

    def test_reset_returns_to_the_not_started_snapshot(self):
        validator, _ = replay(load_frames("evidence_skip.json"))
        self.assertNotEqual(not_started_ui_status(), validator.status())
        validator.reset()
        self.assertEqual(not_started_ui_status(), validator.status())

    # -- the properties Track C's renderers depend on ---------------------

    def test_status_is_a_pure_snapshot(self):
        # The GUI polls at 2 Hz — several polls can land between frames, so
        # reading the status must not disturb the run.  Drive two identical
        # validators and poll only one of them twice per frame.
        frames = load_frames("evidence_correct.json")
        polled, quiet = make_validator(), make_validator()
        events_polled, events_quiet = [], []
        for frame in frames:
            events_polled.extend(polled.update(frame))
            events_quiet.extend(quiet.update(frame))
            self.assertEqual(polled.status(), polled.status())
        self.assertEqual(events_quiet, events_polled)
        self.assertEqual(quiet.status(), polled.status())

    def test_status_serialises_for_the_status_endpoint(self):
        # C8/C9 serve ``status().to_dict()`` as JSON; the browser renders
        # field-for-field, so the dict shape is part of the contract.
        validator = make_validator()
        for frame in load_frames("evidence_correct.json"):
            validator.update(frame)
        payload = json.loads(validator.status().to_json())
        self.assertEqual(
            {
                "protocol_id",
                "protocol_title",
                "current_step_id",
                "current_step_index",
                "next_step_id",
                "next_instruction",
                "completed",
                "skipped",
                "violations",
                "state",
                "t_rel",
                "fps",
                "last_alert",
                "contract_version",
            },
            set(payload),
        )
        # Tuples become JSON-native lists; nothing else is transformed.
        self.assertIsInstance(payload["completed"], list)
        self.assertEqual(8, len(payload["completed"]))
        self.assertEqual("COMPLETE", payload["state"])
        self.assertEqual(CONTRACT_VERSION, payload["contract_version"])
        # to_dict rounds the floats exactly as the contract declares.
        self.assertEqual(round(validator.status().t_rel, 3), payload["t_rel"])
        self.assertEqual(round(validator.status().fps, 2), payload["fps"])


if __name__ == "__main__":
    unittest.main()
