"""Ultralytics adapters, exercised with stand-in objects so no torch is needed.

The adapters reach for ``.cpu().numpy()`` / ``.astype(int)``, which is exactly
the surface faked below. This keeps the whole suite runnable in a bare
interpreter - the property Track A depends on.
"""

import unittest

from har.perception.adapters import (
    COCO_WRIST_KEYPOINTS,
    detections_from_yolo_result,
    wrists_from_pose_result,
)


class FakeArray:
    """Stands in for the ndarray that ``.numpy()`` hands back."""

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


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf) if conf is not None else None
        self.cls = FakeTensor(cls) if cls is not None else None
        self.id = FakeTensor(ids) if ids is not None else None


class FakeResult:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = FakeTensor(xy)
        self.conf = FakeTensor(conf) if conf is not None else None


def person(left_wrist, right_wrist):
    """A 17-keypoint row where only the two wrists are non-zero."""
    row = [[0.0, 0.0]] * 17
    row[9] = list(left_wrist)
    row[10] = list(right_wrist)
    return row


def conf_row(left_conf, right_conf):
    row = [1.0] * 17
    row[9] = left_conf
    row[10] = right_conf
    return row


class DetectionAdapterTests(unittest.TestCase):
    def test_boxes_become_labelled_detections(self):
        result = FakeResult(
            boxes=FakeBoxes(
                xyxy=[(10.0, 20.0, 30.0, 40.0), (50.0, 60.0, 70.0, 80.0)],
                conf=[0.91, 0.42],
                cls=[39, 39],
                ids=[3, 4],
            )
        )
        detections = detections_from_yolo_result(result, {39: "bottle"})
        self.assertEqual(2, len(detections))
        self.assertEqual("bottle", detections[0].label)
        self.assertEqual((10.0, 20.0, 30.0, 40.0), detections[0].box)
        self.assertEqual(3, detections[0].track_id)

    def test_untracked_result_yields_no_track_ids(self):
        result = FakeResult(boxes=FakeBoxes([(0.0, 0.0, 5.0, 5.0)], [0.8], [0], None))
        detections = detections_from_yolo_result(result)
        self.assertIsNone(detections[0].track_id)
        self.assertEqual("0", detections[0].label)

    def test_empty_result_is_safe(self):
        self.assertEqual([], detections_from_yolo_result(FakeResult()))
        self.assertEqual([], detections_from_yolo_result(object()))


class WristAdapterTests(unittest.TestCase):
    def test_both_wrists_are_extracted_with_their_side(self):
        result = FakeResult(
            keypoints=FakeKeypoints(
                xy=[person((100.0, 110.0), (200.0, 210.0))],
                conf=[conf_row(0.9, 0.7)],
            )
        )
        wrists = wrists_from_pose_result(result)
        self.assertEqual(2, len(wrists))
        self.assertEqual({("left", (100.0, 110.0)), ("right", (200.0, 210.0))},
                         {(w.side, w.point) for w in wrists})

    def test_low_confidence_wrists_are_dropped(self):
        result = FakeResult(
            keypoints=FakeKeypoints(
                xy=[person((100.0, 110.0), (200.0, 210.0))],
                conf=[conf_row(0.9, 0.05)],
            )
        )
        wrists = wrists_from_pose_result(result, min_confidence=0.2)
        self.assertEqual(["left"], [w.side for w in wrists])

    def test_zero_coordinate_keypoints_are_treated_as_missing(self):
        result = FakeResult(keypoints=FakeKeypoints(xy=[person((0.0, 0.0), (200.0, 210.0))], conf=None))
        wrists = wrists_from_pose_result(result)
        self.assertEqual(["right"], [w.side for w in wrists])

    def test_coco_wrist_indices_are_nine_and_ten(self):
        self.assertEqual(((9, "left"), (10, "right")), COCO_WRIST_KEYPOINTS)


if __name__ == "__main__":
    unittest.main()
