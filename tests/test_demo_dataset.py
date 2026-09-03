"""A8 acceptance: the PTS-01 evaluation dataset under ``demo/``.

Plan §5 done-when: three mp4 files plus one ground-truth json, and a
replay of ``demo/correct`` reproduces its log.  The videos are scripted
stand-ins for webcam captures (no camera on the build host); the
companion ``*_evidence.json`` files are the frame-accurate recording
the validator consumes.  Tests here stay free of cv2 / numpy / torch.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from har.contracts import FrameEvidence, ObjectTrack, Wrist

try:
    import yaml  # noqa: F401
except ImportError:  # pragma: no cover
    yaml = None

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demo"
GT_PATH = DEMO / "ground_truth.json"
RUN_IDS = ("correct", "skip", "wrong_order")
STEP_IDS = (
    "PRESENT_TRAY",
    "OPEN_TRAY",
    "EXTRACT_RED",
    "VERIFY_RED_PLACED",
    "EXTRACT_BLUE",
    "VERIFY_BLUE_PLACED",
    "SAMPLE_TRANSFER",
    "STOW_AND_CLOSE",
)
MP4_FTYP = b"ftyp"


def evidence_from_dict(d: dict) -> FrameEvidence:
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


def load_gt() -> dict:
    return json.loads(GT_PATH.read_text(encoding="utf-8"))


class DemoDatasetFilesTests(unittest.TestCase):
    """The four A8 artefacts (and the evidence companions) are on disk."""

    def test_ground_truth_and_three_videos_exist(self):
        self.assertTrue(GT_PATH.is_file(), "demo/ground_truth.json missing")
        for run_id in RUN_IDS:
            mp4 = DEMO / f"{run_id}.mp4"
            self.assertTrue(mp4.is_file(), f"missing {mp4.name}")
            self.assertGreater(mp4.stat().st_size, 50_000, f"{mp4.name} is tiny")
            header = mp4.read_bytes()[:32]
            self.assertIn(MP4_FTYP, header, f"{mp4.name} is not an ISO BMFF mp4")

    def test_each_run_has_frame_evidence(self):
        for run_id in RUN_IDS:
            path = DEMO / f"{run_id}_evidence.json"
            self.assertTrue(path.is_file(), f"missing {path.name}")
            frames = json.loads(path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(frames), 10, run_id)
            # Rebuilds with no error — A1's original done-when, applied here.
            rebuilt = [evidence_from_dict(row) for row in frames]
            self.assertEqual(frames[0]["frame_index"], rebuilt[0].frame_index)
            self.assertEqual((640, 480), rebuilt[0].frame_size)
            self.assertEqual(15.0, rebuilt[0].fps)


class DemoGroundTruthSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gt = load_gt()

    def test_protocol_identity(self):
        self.assertEqual("PTS-01", self.gt["protocol_id"])
        self.assertEqual("1.0.0", self.gt["protocol_version"])
        self.assertEqual(15.0, self.gt["fps"])
        self.assertEqual([640, 480], self.gt["frame_size"])

    def test_three_named_runs_with_eight_steps_each(self):
        self.assertEqual(set(RUN_IDS), set(self.gt["runs"]))
        for run_id in RUN_IDS:
            run = self.gt["runs"][run_id]
            self.assertEqual(f"demo/{run_id}.mp4", run["video"])
            self.assertEqual(f"demo/{run_id}_evidence.json", run["evidence"])
            self.assertEqual(8, len(run["steps"]))
            self.assertEqual(list(STEP_IDS), [s["step_id"] for s in run["steps"]])
            self.assertEqual(list(range(1, 9)), [s["index"] for s in run["steps"]])

    def test_operator_windows_are_well_formed(self):
        for run_id, run in self.gt["runs"].items():
            for step in run["steps"]:
                start, end = step["t_start"], step["t_end"]
                if step["operator_outcome"] in {"SKIPPED", "NOT_PERFORMED"}:
                    self.assertIsNone(start, step)
                    self.assertIsNone(end, step)
                    continue
                self.assertIsInstance(start, (int, float), step)
                self.assertIsInstance(end, (int, float), step)
                self.assertLessEqual(start, end, f"{run_id} {step['step_id']}")
                self.assertGreaterEqual(start, 0.0)

    def test_correct_run_is_eight_completed_in_order(self):
        run = self.gt["runs"]["correct"]
        self.assertEqual(["COMPLETED"] * 8, [s["operator_outcome"] for s in run["steps"]])
        # Operator windows are monotonic along the protocol.
        ends = [s["t_end"] for s in run["steps"]]
        self.assertEqual(ends, sorted(ends))
        self.assertTrue(run["expected"]["protocol_complete"])
        self.assertEqual(list(STEP_IDS), run["expected"]["completed"])
        self.assertEqual([], run["expected"]["skipped"])
        self.assertEqual([], run["expected"]["violations"])

    def test_skip_run_omits_the_red_box(self):
        by_id = {s["step_id"]: s for s in self.gt["runs"]["skip"]["steps"]}
        self.assertEqual("SKIPPED", by_id["EXTRACT_RED"]["operator_outcome"])
        self.assertIsNone(by_id["EXTRACT_RED"]["t_start"])
        # Blue is what the operator actually did, and it happens in the
        # window that should have been red's.
        blue = by_id["EXTRACT_BLUE"]
        self.assertEqual("COMPLETED", blue["operator_outcome"])
        self.assertLess(blue["t_end"], 6.0)
        self.assertEqual("EXTRACT_BLUE", self.gt["runs"]["skip"]["expected"]["violations"][0])

    def test_wrong_order_places_blue_before_red(self):
        by_id = {s["step_id"]: s for s in self.gt["runs"]["wrong_order"]["steps"]}
        self.assertLess(by_id["EXTRACT_BLUE"]["t_start"], by_id["EXTRACT_RED"]["t_start"])
        self.assertLessEqual(by_id["EXTRACT_BLUE"]["t_end"], by_id["EXTRACT_RED"]["t_start"])
        self.assertEqual("EXTRACT_BLUE", self.gt["runs"]["wrong_order"]["expected"]["violations"][0])
        ooo = [
            e for e in self.gt["runs"]["wrong_order"]["expected"]["events"]
            if e["event"] == "OUT_OF_ORDER"
        ]
        self.assertEqual(1, len(ooo))
        self.assertEqual("EXTRACT_BLUE", ooo[0]["step_id"])


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class DemoReplayMatchesGroundTruthTests(unittest.TestCase):
    """`--source demo/correct.mp4` done-when, via the evidence recording.

    Track C's CLI is not on this branch, so we replay the committed
    ``FrameEvidence`` through ``SequenceValidator`` — the same log the
    CLI must write once it lands.
    """

    @classmethod
    def setUpClass(cls):
        from har.protocol.spec import load_protocol
        from har.protocol.validator import SequenceValidator

        cls.load_protocol = staticmethod(load_protocol)
        cls.SequenceValidator = SequenceValidator
        cls.gt = load_gt()

    def _replay(self, run_id: str):
        raw = json.loads((DEMO / f"{run_id}_evidence.json").read_text(encoding="utf-8"))
        frames = [evidence_from_dict(d) for d in raw]
        spec = self.load_protocol(REPO / "protocols" / "pts01.yaml", (640, 480))
        validator = self.SequenceValidator(spec)
        events = []
        for frame in frames:
            events.extend(validator.update(frame))
        return validator, events

    def test_correct_run_reproduces_its_log(self):
        # Live build: protocols/pts01.yaml is the 7-step demo build (the
        # SAMPLE_TRANSFER/vial step is dropped), so replaying the committed
        # footage scores the seven live steps; the ground truth still annotates
        # the full 8-step recorded run (the videos are not regenerated).
        validator, events = self._replay("correct")
        expected = self.gt["runs"]["correct"]["expected"]
        live_completed = [s for s in expected["completed"] if s != "SAMPLE_TRANSFER"]
        self.assertEqual(live_completed, list(validator.completed_steps))
        self.assertEqual([], list(validator.status().skipped))
        self.assertEqual([], list(validator.violations))
        self.assertTrue(validator.finished)
        completed = [e for e in events if e.event == "COMPLETED"]
        self.assertEqual(7, len(completed))
        self.assertEqual("PROTOCOL_COMPLETE", events[-1].event)

    def test_skip_and_wrong_order_each_flag_out_of_order_once(self):
        # Gate G1 semantics are unchanged by dropping the vial step: both runs
        # flag exactly one OUT_OF_ORDER on EXTRACT_BLUE (the blue box is placed
        # while the red step is still pending).
        for run_id in ("skip", "wrong_order"):
            _, events = self._replay(run_id)
            ooo = [e for e in events if e.event == "OUT_OF_ORDER"]
            self.assertEqual(1, len(ooo), run_id)
            self.assertEqual("EXTRACT_BLUE", ooo[0].step_id)

    def test_annotated_end_is_close_to_validator_completion_on_the_correct_run(self):
        # Latency = validator COMPLETED - operator t_end.  Hold_frames on
        # a 15 fps stream is at most a couple of seconds; anything larger
        # means the annotation drifted off the recording.
        for step in self.gt["runs"]["correct"]["steps"]:
            latency = step["validator_completed_s"] - step["t_end"]
            self.assertGreaterEqual(latency, -0.5, step["step_id"])
            self.assertLess(latency, 3.0, step["step_id"])


if __name__ == "__main__":
    unittest.main()
