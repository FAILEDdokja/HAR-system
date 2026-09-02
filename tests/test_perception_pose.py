"""WristExtractor (B3) driven by a duck-typed stand-in pose model.

No torch, no ultralytics: the fake model answers ``predict(frame, **kw)`` with
the same result surface the ultralytics adapters consume (see
``test_perception_adapters.py`` for that contract).
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
