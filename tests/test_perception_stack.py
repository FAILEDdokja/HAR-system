"""PerceptionStack (B4) with duck-typed detector and wrist extractor fakes.

Proves the seam Track A consumes: ``process()`` returns a ``FrameEvidence``
whose ``to_dict()`` matches the A1 fixture shape field-for-field, and the
person gate actually skips the detector when nobody is in frame.
"""

import json
import unittest

from har.contracts import Detection, FrameEvidence, Wrist
from har.perception.perception import PerceptionStack

LABELS = ("red_box", "blue_box")
FRAME_SIZE = (640, 480)


class FakeDetector:
    backend = "hsv"

    def __init__(self):
        self.calls = 0
        self.detections: list[Detection] = []

    def detect(self, frame):
        self.calls += 1
        return list(self.detections)


class FakeWrists:
    def __init__(self, person_present=True):
        self.person_present = person_present
        self.value = []
        self.calls = 0

    def wrists(self, frame, frame_index):
        self.calls += 1
        return list(self.value)


class PerceptionStackTests(unittest.TestCase):
    def _stack(self, detector, wrists):
        return PerceptionStack(detector, wrists, LABELS, FRAME_SIZE)

    def test_tracked_object_reaches_measured_after_acquire_frames(self):
        detector = FakeDetector()
        detector.detections = [Detection((100.0, 100.0, 140.0, 150.0), 0.9, "red_box")]
        stack = self._stack(detector, FakeWrists())

        for frame_index in range(5):
            evidence = stack.process("frame", frame_index, frame_index / 15.0)
        red = evidence.objects["red_box"]
        self.assertTrue(red.measured)
        self.assertEqual((100.0, 100.0, 140.0, 150.0), red.box)
        self.assertEqual(0, red.lost_frames)
        self.assertEqual("IDLE", evidence.hoi["red_box"])
        blue = evidence.objects["blue_box"]
        self.assertFalse(blue.measured)
        self.assertIsNone(blue.box)

    def test_person_gate_skips_the_detector_when_nobody_is_in_frame(self):
        detector = FakeDetector()
        detector.detections = [Detection((100.0, 100.0, 140.0, 150.0), 0.9, "red_box")]
        stack = self._stack(detector, FakeWrists(person_present=False))

        for frame_index in range(6):
            evidence = stack.process("frame", frame_index, frame_index / 15.0)
        self.assertEqual(0, detector.calls)
        self.assertFalse(evidence.objects["red_box"].measured)
        self.assertIsNone(evidence.objects["red_box"].box)

    def test_gate_passes_when_the_extractor_reports_a_person(self):
        detector = FakeDetector()
        stack = self._stack(detector, FakeWrists(person_present=True))
        stack.process("frame", 0, 0.0)
        self.assertEqual(1, detector.calls)

    def test_gate_is_neutral_for_extractors_without_person_information(self):
        class WristsWithoutGate:
            def wrists(self, frame, frame_index):
                return []

        detector = FakeDetector()
        stack = self._stack(detector, WristsWithoutGate())
        stack.process("frame", 0, 0.0)
        self.assertEqual(1, detector.calls)

    def test_evidence_to_dict_matches_the_a1_fixture_shape(self):
        detector = FakeDetector()
        detector.detections = [Detection((100.0, 100.0, 140.0, 150.0), 0.9, "red_box")]
        wrists = FakeWrists()
        wrists.value = [Wrist((120.0, 130.0), 0.88, "left")]
        stack = self._stack(detector, wrists)

        evidence = stack.process("frame", 7, 0.5)
        self.assertIsInstance(evidence, FrameEvidence)
        restored = json.loads(json.dumps(evidence.to_dict()))
        self.assertEqual(
            [
                "frame_index",
                "t_rel",
                "frame_size",
                "objects",
                "hands",
                "hoi",
                "rack_ready",
                "fps",
            ],
            list(restored),
        )
        self.assertEqual(7, restored["frame_index"])
        self.assertEqual([640, 480], restored["frame_size"])
        self.assertEqual([100.0, 100.0, 140.0, 150.0], restored["objects"]["red_box"]["box"])
        self.assertEqual("IDLE", restored["hoi"]["red_box"])
        self.assertEqual(1, len(restored["hands"]))

    def test_rack_ready_reflects_the_attached_rack_frame(self):
        class FakeRack:
            def __init__(self, ready):
                self._ready = ready

            def ready(self):
                return self._ready

        stack = self._stack(FakeDetector(), FakeWrists())
        self.assertFalse(stack.process("frame", 0, 0.0).rack_ready)
        stack.rack = FakeRack(True)
        self.assertTrue(stack.process("frame", 1, 0.1).rack_ready)

    def test_reset_forgets_trackers_and_counters(self):
        detector = FakeDetector()
        detector.detections = [Detection((100.0, 100.0, 140.0, 150.0), 0.9, "red_box")]
        stack = self._stack(detector, FakeWrists())
        for frame_index in range(5):
            stack.process("frame", frame_index, frame_index / 15.0)
        stack.reset()
        evidence = stack.process("frame", 6, 0.5)
        self.assertFalse(evidence.objects["red_box"].measured)

    def test_fps_is_measured_end_to_end(self):
        stack = self._stack(FakeDetector(), FakeWrists())
        self.assertEqual(0.0, stack.fps)
        stack.process("frame", 0, 0.0)
        stack.process("frame", 1, 0.1)
        self.assertGreater(stack.fps, 0.0)


if __name__ == "__main__":
    unittest.main()
