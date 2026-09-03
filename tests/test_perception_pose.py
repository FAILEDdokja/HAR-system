"""WristExtractor (B3) driven by a duck-typed stand-in pose model.

No torch, no ultralytics: the fake model answers ``predict(frame, **kw)`` with
the same result surface the ultralytics adapters consume (see
``test_perception_adapters.py`` for that contract).
"""

import unittest
from pathlib import Path

from har.contracts import Wrist
from har.perception.pose import WristExtractor


class FakeArray:
    def __init__(self, data):
        self._data = data

    def astype(self, _dtype):
        return self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


class FakeTensor:
    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def numpy(self):
        return FakeArray(self._data)

    def __len__(self):
        return len(self._data)


class TruthHostileTensor(FakeTensor):
    def __bool__(self):
        raise AssertionError("tensor truthiness should never be inspected")


class FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = FakeTensor(xyxy) if xyxy is not None else None
        self.conf = None
        self.cls = None
        self.id = None


class TruthHostileBoxes(FakeBoxes):
    def __init__(self, xyxy):
        self.xyxy = TruthHostileTensor(xyxy)
        self.conf = None
        self.cls = None
        self.id = None


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = FakeTensor(xy)
        self.conf = FakeTensor(conf) if conf is not None else None


class FakeResult:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


def person_result(left, right, persons=1):
    row = [[0.0, 0.0]] * 17
    row[9] = list(left)
    row[10] = list(right)
    conf_row = [0.9] * 17
    return FakeResult(
        boxes=FakeBoxes([(0.0, 0.0, 100.0, 300.0)] * persons),
        keypoints=FakeKeypoints([row], [conf_row]),
    )


class FakeModel:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        return [self._results[min(self.calls - 1, len(self._results) - 1)]]


def empty_result():
    """A frame with nobody detected: empty boxes and empty keypoints."""
    return FakeResult(boxes=FakeBoxes([]), keypoints=FakeKeypoints([], []))


class WristDebounceTests(unittest.TestCase):
    """Issue 1: a 1-frame (or short-lived) hallucinated hand must not report."""

    def test_single_frame_flash_is_not_reported(self):
        # One pose frame with wrists, then nothing.  confirm_frames=3 so the
        # brief appearance never reaches the threshold -> no "hand" exposed.
        model = FakeModel([person_result((1.0, 1.0), (2.0, 2.0)),
                           empty_result(), empty_result(), empty_result()])
        extractor = WristExtractor("unused.pt", model=model, confirm_frames=3)
        for idx in range(4):
            self.assertEqual([], extractor.wrists("frame", idx),
                             f"frame {idx}: a single-frame flash must not report a hand")

    def test_sustained_hand_is_reported_only_after_confirm_frames(self):
        # Five consecutive pose frames with a real hand -> exposed from frame 2
        # (0-based) onward, then dropped once absent for more than forget_frames.
        person = person_result((100.0, 110.0), (200.0, 210.0))
        model = FakeModel([person, person, person, person, person,
                           empty_result(), empty_result(), empty_result(),
                           empty_result(), empty_result()])
        extractor = WristExtractor("unused.pt", model=model,
                                   confirm_frames=3, forget_frames=3)
        # confirmed from frame 2 (0-based), held through forget_frames=3 of
        # missing pose frames (frames 5-7), dropped once absent >3 (frame 8).
        expected_sides = [None, None, {"left", "right"}, {"left", "right"},
                          {"left", "right"}, {"left", "right"}, {"left", "right"},
                          {"left", "right"}, None, None]
        for idx, sides in enumerate(expected_sides):
            got = {w.side for w in extractor.wrists("frame", idx)}
            if sides is None:
                self.assertEqual(set(), got, f"frame {idx}: expected no hand")
            else:
                self.assertEqual(sides, got, f"frame {idx}: confirmed hand missing")

    def test_debounce_resets_with_the_extractor(self):
        # After reset the confirm counter restarts from scratch: the hand needs
        # confirm_frames=2 fresh pose frames again before it is exposed.
        model = FakeModel([person_result((1.0, 1.0), (2.0, 2.0))] * 6)
        extractor = WristExtractor("unused.pt", model=model, confirm_frames=2)
        self.assertEqual([], extractor.wrists("frame", 0))    # seen=1
        self.assertEqual({"left", "right"},
                         {w.side for w in extractor.wrists("frame", 1)})  # seen=2
        extractor.reset()
        self.assertEqual([], extractor.wrists("frame", 2))    # counter cleared
        self.assertEqual({"left", "right"},
                         {w.side for w in extractor.wrists("frame", 3)})  # seen=2 again


class WristExtractionTests(unittest.TestCase):
    def test_both_wrists_are_extracted_with_sides_and_confidence(self):
        model = FakeModel([person_result((100.0, 110.0), (200.0, 210.0))])
        extractor = WristExtractor("unused.pt", model=model)
        wrists = extractor.wrists("frame", 0)
        self.assertEqual(
            {("left", (100.0, 110.0)), ("right", (200.0, 210.0))},
            {(w.side, w.point) for w in wrists},
        )
        self.assertTrue(all(isinstance(w, Wrist) for w in wrists))
        self.assertTrue(extractor.person_present)
        self.assertEqual(1, extractor.person_count)

    def test_skipped_frames_return_the_previous_result_not_an_empty_list(self):
        model = FakeModel([person_result((100.0, 110.0), (200.0, 210.0))])
        extractor = WristExtractor("unused.pt", every_n_frames=2, model=model)
        first = extractor.wrists("frame", 0)
        cached = extractor.wrists("frame", 1)
        self.assertEqual(1, model.calls)  # pose did not re-run
        self.assertEqual([(w.side, w.point) for w in first],
                         [(w.side, w.point) for w in cached])
        self.assertNotEqual(0, len(cached), "empty list would read as 'hands vanished'")

    def test_every_nth_frame_re_runs_pose(self):
        model = FakeModel([person_result((1.0, 1.0), (2.0, 2.0))])
        extractor = WristExtractor("unused.pt", every_n_frames=3, model=model)
        extractor.wrists("frame", 0)
        extractor.wrists("frame", 1)
        extractor.wrists("frame", 2)
        self.assertEqual(1, model.calls)
        extractor.wrists("frame", 3)
        self.assertEqual(2, model.calls)

    def test_no_person_detected_clears_the_person_gate(self):
        model = FakeModel([FakeResult(boxes=FakeBoxes([]), keypoints=FakeKeypoints([], []))])
        extractor = WristExtractor("unused.pt", model=model)
        extractor.wrists("frame", 0)
        self.assertEqual(0, extractor.person_count)
        self.assertFalse(extractor.person_present)

    def test_person_count_does_not_use_tensor_truthiness(self):
        model = FakeModel([
            FakeResult(
                boxes=TruthHostileBoxes([(0.0, 0.0, 100.0, 300.0)]),
                keypoints=FakeKeypoints([], []),
            )
        ])
        extractor = WristExtractor("unused.pt", model=model)
        extractor.wrists("frame", 0)
        self.assertEqual(1, extractor.person_count)

    def test_reset_forces_a_fresh_pose_pass(self):
        model = FakeModel([person_result((1.0, 1.0), (2.0, 2.0))])
        extractor = WristExtractor("unused.pt", every_n_frames=5, model=model)
        extractor.wrists("frame", 0)
        extractor.reset()
        extractor.wrists("frame", 1)
        self.assertEqual(2, model.calls)

    def test_returned_list_is_a_copy_of_the_cache(self):
        model = FakeModel([person_result((1.0, 1.0), (2.0, 2.0))])
        extractor = WristExtractor("unused.pt", every_n_frames=2, model=model)
        first = extractor.wrists("frame", 0)
        first.append("junk")
        self.assertEqual(2, len(extractor.wrists("frame", 1)))

    def test_missing_ultralytics_gives_an_actionable_error(self):
        try:
            import ultralytics  # noqa: F401

            have_ultralytics = True
        except ImportError:
            have_ultralytics = False
        if have_ultralytics:
            self.skipTest("ultralytics is installed; nothing to prove")
        with self.assertRaises(RuntimeError) as caught:
            WristExtractor("models/yolo11n-pose.pt")
        self.assertIn("model=", str(caught.exception))


class ShippedWeightsTests(unittest.TestCase):
    """The committed weights must load on the pinned ultralytics.

    ``models/yolo11n-pose.pt`` is a YOLO11 checkpoint; its ``C3k2`` block only
    exists from ultralytics 8.3.0 on.  On 8.2.100 the load raised, ``har.app``
    caught it and quietly fell back to ``--wrists none`` — the live-camera run
    kept going but observed no manipulation steps and emitted spurious
    violations.  This test is the tripwire for the next pin bump.
    """

    def test_committed_pose_weights_load(self):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # ImportError, or a torch without its runtime
            self.skipTest(f"ultralytics unavailable: {exc}")
        weights = Path(__file__).resolve().parents[1] / "models" / "yolo11n-pose.pt"
        if not weights.exists():
            self.skipTest(f"{weights.name} is not present")
        model = YOLO(str(weights))
        self.assertEqual({0: "person"}, dict(model.names))


if __name__ == "__main__":
    unittest.main()
