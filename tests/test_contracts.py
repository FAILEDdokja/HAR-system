"""The contracts are the coordination mechanism - these tests guard the seams.

They assert two things that three people working in parallel depend on:
  1. every cross-track type survives a JSON round trip (Track C is built
     against canned JSON, so this is not optional);
  2. ``har.contracts`` imports nothing but the standard library, so Track A can
     run in a bare interpreter with no cv2 / torch installed.
"""

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

from har import contracts
from har.contracts import (
    Detection,
    FrameEvidence,
    HandObjectState,
    ObjectTrack,
    StepEvent,
    StepEventType,
    UiStatus,
    Wrist,
)


class JsonRoundTripTests(unittest.TestCase):
    def test_step_event_survives_json(self):
        event = StepEvent(
            t_iso="2026-09-03T09:00:00+05:30",
            t_rel=12.5,
            frame_index=375,
            step_id="EXTRACT_RED",
            step_index=3,
            event=StepEventType.SKIPPED.value,
            status="VIOLATION",
            message="Step 3 skipped",
            confidence=0.87,
        )
        restored = json.loads(event.to_json())
        self.assertEqual("EXTRACT_RED", restored["step_id"])
        self.assertEqual("SKIPPED", restored["event"])
        self.assertEqual(12.5, restored["t_rel"])

    def test_frame_evidence_survives_json(self):
        evidence = FrameEvidence(
            frame_index=1,
            t_rel=0.033,
            frame_size=(640, 480),
            objects={"red_box": ObjectTrack("red_box", (10.0, 10.0, 50.0, 50.0), True)},
            hands=[Wrist((30.0, 30.0), 0.9, "left")],
            hoi={"red_box": HandObjectState.CARRYING.value},
            rack_ready=True,
            fps=12.4,
        )
        restored = json.loads(json.dumps(evidence.to_dict()))
        self.assertEqual([10.0, 10.0, 50.0, 50.0], restored["objects"]["red_box"]["box"])
        self.assertEqual("CARRYING", restored["hoi"]["red_box"])

    def test_ui_status_is_fully_serialisable(self):
        status = UiStatus(
            protocol_id="PTS-01",
            protocol_title="Payload Tray Sorting & Sample Transfer",
            current_step_id="EXTRACT_RED",
            current_step_index=3,
            next_step_id="VERIFY_RED_PLACED",
            next_instruction="Confirm the red box is stationary inside zone A.",
            completed=("PRESENT_TRAY", "OPEN_TRAY"),
            skipped=(),
            violations=(),
            state="RUNNING",
            t_rel=12.5,
            fps=11.0,
            last_alert="",
        )
        restored = json.loads(status.to_json())
        self.assertEqual(("PRESENT_TRAY", "OPEN_TRAY"), tuple(restored["completed"]))
        self.assertEqual(contracts.CONTRACT_VERSION, restored["contract_version"])

    def test_detection_and_track_helpers(self):
        detection = Detection((1.5, 2.5, 3.5, 4.5), 0.9123, "tray", 7)
        self.assertEqual("tray", detection.to_dict()["label"])
        track = ObjectTrack("tray", None, False, 3)
        self.assertFalse(track.present)
        self.assertIsNone(ObjectTrack("tray", None, False, 0).to_dict()["box"])


class ContractIsolationTests(unittest.TestCase):
    def test_contracts_module_imports_only_the_standard_library(self):
        """Fails the build if anyone adds cv2 / numpy / torch to contracts.

        Parsed with ``ast`` so prose in docstrings cannot trip the check.
        """
        tree = ast.parse(Path(contracts.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.append(node.module.split(".")[0])

        allowed = {"__future__", "json", "dataclasses", "enum", "typing"}
        unexpected = sorted(set(imported) - allowed)
        self.assertEqual([], unexpected, f"contracts.py must stay dependency-free, found {unexpected}")

        forbidden = {"cv2", "numpy", "torch", "ultralytics", "flask", "pyttsx3", "har"}
        self.assertEqual([], sorted(set(imported) & forbidden))

    def test_contracts_import_in_a_clean_interpreter(self):
        """Proves Track A needs no heavy dependency to run its tests."""
        code = "import har.contracts as c; print(c.CONTRACT_VERSION)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(Path(contracts.__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(contracts.CONTRACT_VERSION, result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
