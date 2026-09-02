"""HSV colour detector (B2) on small synthetic frames.

The synthetic BGR frames are plain ``numpy`` arrays, so these tests need numpy
(and PyYAML for the config round trip) but not cv2: the detector's pure-Python
labelling path runs whenever cv2 is absent. Like the PyYAML-skipped protocol
tests, everything here skips cleanly in a bare interpreter.
"""

import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from har.contracts import Detection

REPO_ROOT = Path(__file__).resolve().parents[1]

WIDTH, HEIGHT = 160, 120


def make_frame():
    """White 160x120 BGR frame with the five protocol props painted on it."""
    frame = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    frame[10:30, 10:60] = (30, 30, 30)  # black tray
    frame[10:30, 100:130] = (0, 230, 230)  # yellow lid
    frame[60:90, 20:50] = (40, 40, 230)  # red box
    frame[60:85, 100:125] = (240, 80, 0)  # blue box
    frame[93:117, 68:92] = (60, 200, 60)  # green vial (24x24: survives a real
    # cv2 morphological open + contourArea at min_area=400; a 20x20 blob does not)
    return frame


@unittest.skipIf(np is None, "numpy is not installed")
class ColourConversionTests(unittest.TestCase):
    def test_synthetic_colours_land_on_the_expected_hue(self):
        from har.perception.color_detector import bgr_to_hsv

        hsv = bgr_to_hsv(make_frame())
        yellow = hsv[20, 110]
        red = hsv[70, 30]
        blue = hsv[70, 110]
        green = hsv[105, 80]
        self.assertEqual(0, int(red[0]))  # red box hue
        self.assertTrue(20 <= int(yellow[0]) <= 35)  # lid hue
        self.assertTrue(100 <= int(blue[0]) <= 125)  # blue box hue
        self.assertTrue(40 <= int(green[0]) <= 85)  # vial hue

    def test_white_and_black_saturate_to_zero_saturation(self):
        from har.perception.color_detector import bgr_to_hsv

        hsv = bgr_to_hsv(make_frame())
        self.assertEqual((0, 0, 255), tuple(int(v) for v in hsv[0, 0]))  # white bg
        self.assertEqual((0, 0, 30), tuple(int(v) for v in hsv[20, 30]))  # black tray

    def test_rejects_non_colour_frames(self):
        from har.perception.color_detector import bgr_to_hsv

        with self.assertRaises(ValueError):
            bgr_to_hsv(np.zeros((10, 10), dtype=np.uint8))


@unittest.skipIf(np is None, "numpy is not installed")
class DetectionTests(unittest.TestCase):
    def setUp(self):
        from har.perception.color_detector import ColorDetector

        self.frame = make_frame()
        self.detector = ColorDetector(
            {
                "tray": ((0, 0, 0), (180, 255, 75)),
                "tray_lid": ((20, 80, 80), (35, 255, 255)),
                "red_box": [((0, 100, 80), (8, 255, 255)), ((172, 100, 80), (179, 255, 255))],
                "blue_box": ((100, 90, 60), (125, 255, 255)),
                "vial": ((40, 60, 60), (85, 255, 255)),
            },
            min_area=100,
        )

    def _by_label(self, detections):
        return {d.label: d for d in detections}

    def test_all_five_labels_are_detected_with_their_own_box(self):
        found = self._by_label(self.detector.detect(self.frame))
        self.assertEqual(
            {"tray", "tray_lid", "red_box", "blue_box", "vial"}, set(found)
        )
        tray = found["tray"].box
        self.assertTrue(tray[0] < 60 and tray[2] <= 61 and tray[3] <= 31, tray)
        red = found["red_box"].box
        self.assertTrue(19 <= red[0] and red[2] <= 51, red)

    def test_detections_are_contract_types_round_tripping_to_json(self):
        import json

        detections = self.detector.detect(self.frame)
        for detection in detections:
            self.assertIsInstance(detection, Detection)
            self.assertIsNone(detection.track_id)
            restored = json.loads(json.dumps(detection.to_dict()))
            self.assertEqual(detection.label, restored["label"])

    def test_small_blobs_below_min_area_are_ignored(self):
        from har.perception.color_detector import ColorDetector

        blank = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        blank[50:53, 50:53] = (40, 40, 230)  # 3x3 red speck
        detector = ColorDetector(
            {"red_box": ((0, 100, 80), (8, 255, 255))}, min_area=100
        )
        self.assertEqual([], detector.detect(blank))

    def test_roi_restricts_where_we_look_and_keeps_frame_coordinates(self):
        from har.perception.color_detector import ColorDetector

        detector = ColorDetector(
            {
                "tray": ((0, 0, 0), (180, 255, 75)),
                "blue_box": ((100, 90, 60), (125, 255, 255)),
            },
            roi=(80, 0, 160, 120),
            min_area=100,
        )
        found = self._by_label(detector.detect(self.frame))
        self.assertNotIn("tray", found)  # tray lives at x < 60, outside the ROI
        blue = found["blue_box"].box
        self.assertTrue(99 <= blue[0] and blue[2] <= 126, blue)

    def test_median_smoothing_stabilises_the_box(self):
        from har.perception.color_detector import ColorDetector

        detector = ColorDetector(
            {"red_box": ((0, 100, 80), (8, 255, 255))}, median_window=3, min_area=100
        )
        frame_a = self.frame.copy()
        frame_a[60:90, 10:55] = (255, 255, 255)  # clear the base frame's red box
        frame_a[60:90, 10:40] = (40, 40, 230)  # red at x 10..40
        first = detector.detect(frame_a)[0]
        self.assertEqual((10.0, 60.0, 40.0, 90.0), first.box)

        frame_b = self.frame.copy()
        frame_b[60:90, 10:55] = (255, 255, 255)
        frame_b[60:90, 20:50] = (40, 40, 230)  # red shifted +10
        second = detector.detect(frame_b)[0]
        self.assertEqual((15.0, 60.0, 45.0, 90.0), second.box)  # median of two hits

    def test_reset_clears_the_smoothing_history(self):
        from har.perception.color_detector import ColorDetector

        detector = ColorDetector(
            {"red_box": ((0, 100, 80), (8, 255, 255))}, median_window=3, min_area=100
        )
        frame_a = self.frame.copy()
        frame_a[60:90, 10:55] = (255, 255, 255)
        frame_a[60:90, 10:40] = (40, 40, 230)
        detector.detect(frame_a)
        detector.reset()
        frame_b = self.frame.copy()
        frame_b[60:90, 10:55] = (255, 255, 255)
        frame_b[60:90, 20:50] = (40, 40, 230)
        self.assertEqual((20.0, 60.0, 50.0, 90.0), detector.detect(frame_b)[0].box)

    def test_invalid_ranges_are_rejected_at_construction(self):
        from har.perception.color_detector import ColorDetector

        with self.assertRaises(ValueError):
            ColorDetector({"red_box": ((9, 0, 0), (8, 255, 255))})
        with self.assertRaises(ValueError):
            ColorDetector({"red_box": ((0, 0, 0), (200, 255, 255))})


@unittest.skipIf(yaml is None or np is None, "PyYAML or numpy is not installed")
class ColourConfigTests(unittest.TestCase):
    def test_config_file_drives_a_working_detector(self):
        from har.perception.color_detector import ColorDetector, load_colour_config

        ranges, options = load_colour_config(REPO_ROOT / "config" / "colours.yaml")
        self.assertEqual(
            {"tray", "tray_lid", "red_box", "blue_box", "vial"}, set(ranges)
        )
        detector = ColorDetector(ranges, min_area=int(options.get("min_area", 400)))
        labels = {d.label for d in detector.detect(make_frame())}
        self.assertEqual({"tray", "tray_lid", "red_box", "blue_box", "vial"}, labels)

    def test_from_config_classmethod_builds_and_detects(self):
        from har.perception.color_detector import from_config

        detector = from_config(REPO_ROOT / "config" / "colours.yaml")
        self.assertEqual("hsv", detector.backend)
        labels = {d.label for d in detector.detect(make_frame())}
        self.assertIn("red_box", labels)


if __name__ == "__main__":
    unittest.main()
