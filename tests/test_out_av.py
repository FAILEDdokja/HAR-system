"""C6 — recorder, streamer, web GUI and overlay tests.

Module-level imports stay stdlib-only (the repo rule: the suite must collect
in a bare interpreter).  cv2 / numpy / flask are imported inside the test
bodies and the classes skip cleanly when they are absent.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from har.contracts import FrameEvidence, ObjectTrack, UiStatus, Wrist  # noqa: E402
from har.out.streamer import MjpegStreamer  # noqa: E402
from tools.replay_events import load_events  # noqa: E402

HAVE_CV2 = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None
HAVE_FLASK = importlib.util.find_spec("flask") is not None
FIXTURES = REPO / "tests" / "fixtures"


def _np():
    import numpy as np

    return np


def make_status(**overrides) -> UiStatus:
    base = dict(
        protocol_id="PTS-01",
        protocol_title="Payload Tray Sorting & Sample Transfer",
        current_step_id="EXTRACT_RED",
        current_step_index=3,
        next_step_id="VERIFY_RED_PLACED",
        next_instruction="Confirm the red box is stationary inside zone A.",
        completed=("PRESENT_TRAY", "OPEN_TRAY"),
        skipped=(),
        violations=(),
        state="IN_PROGRESS",
        t_rel=6.4,
        fps=14.7,
        last_alert="",
    )
    base.update(overrides)
    return UiStatus(**base)


def make_evidence() -> FrameEvidence:
    return FrameEvidence(
        frame_index=42,
        t_rel=2.8,
        frame_size=(640, 480),
        objects={
            "tray": ObjectTrack("tray", (230.0, 240.0, 410.0, 410.0), True),
            "red_box": ObjectTrack("red_box", (250.0, 260.0, 310.0, 330.0), True),
            "blue_box": ObjectTrack("blue_box", None, False, 3),
        },
        hands=[Wrist((240.0, 250.0), 0.9, "left")],
        hoi={"tray": "IDLE", "red_box": "NEAR_OBJECT", "blue_box": "IDLE"},
        rack_ready=False,
        fps=14.7,
    )


@unittest.skipUnless(HAVE_CV2, "opencv/numpy not installed")
class RecorderTests(unittest.TestCase):
    """C7: the stored video plays in a stock player with correct fps/duration."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_written_file_reopens_with_matching_geometry(self):
        import cv2

        from har.out.recorder import VideoRecorder

        np = _np()
        path = self.dir / "recordings" / "run_test.mp4"  # parent created by the recorder
        with VideoRecorder(path, (320, 240), fps=15.0) as recorder:
            for i in range(30):
                frame = np.full((240, 320, 3), i * 8 % 255, dtype=np.uint8)
                recorder.write(frame)
        self.assertEqual(30, recorder.frames_written)

        capture = cv2.VideoCapture(str(path))
        self.assertTrue(capture.isOpened(), "stock cv2 cannot reopen the recording")
        self.assertAlmostEqual(15.0, capture.get(cv2.CAP_PROP_FPS), delta=0.5)
        self.assertEqual(320, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.assertEqual(240, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        read = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            read += 1
        capture.release()
        self.assertGreaterEqual(read, 25)  # mp4v may drop a tail frame; nothing more

    def test_wrong_sized_frame_is_rejected_and_close_is_idempotent(self):
        from har.out.recorder import VideoRecorder

        np = _np()
        recorder = VideoRecorder(self.dir / "r.mp4", (160, 120), fps=10.0)
        recorder.write(np.zeros((120, 160, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            recorder.write(np.zeros((60, 80, 3), dtype=np.uint8))  # silent corruption, refused
        recorder.close()
        recorder.close()  # idempotent
        with self.assertRaises(RuntimeError):
            recorder.write(np.zeros((120, 160, 3), dtype=np.uint8))


@unittest.skipUnless(HAVE_CV2, "opencv/numpy not installed")
class StreamerTests(unittest.TestCase):
    """C8: latest-frame-only, non-blocking, honest MJPEG bytes."""

    def test_latest_jpeg_wins_and_is_real_jpeg(self):
        np = _np()
        streamer = MjpegStreamer()
        self.addCleanup(streamer.shutdown)
        self.assertIsNone(streamer.latest_jpeg())
        streamer.publish(np.full((60, 80, 3), 10, dtype=np.uint8))
        first = streamer.latest_jpeg()
        streamer.publish(np.full((60, 80, 3), 240, dtype=np.uint8))
        second = streamer.latest_jpeg()
        self.assertTrue(first.startswith(b"\xff\xd8") and first.endswith(b"\xff\xd9"))
        self.assertTrue(second.startswith(b"\xff\xd8"))
        self.assertNotEqual(first, second)
        self.assertEqual(2, streamer.frames_published)  # one slot, overwritten

    def test_publish_never_blocks_and_consumers_see_new_frames(self):
        np = _np()
        streamer = MjpegStreamer()
        self.addCleanup(streamer.shutdown)
        start = time.monotonic()
        for i in range(200):
            streamer.publish(np.full((48, 64, 3), i % 255, dtype=np.uint8))
        self.assertLess(time.monotonic() - start, 3.0)  # encode cost only, no waiting

        seen = []
        stop = threading.Event()

        def consume():
            for _chunk in streamer.mjpeg_stream():
                seen.append(streamer.frames_published)
                if len(seen) >= 5:
                    stop.set()
                    break

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        time.sleep(0.05)
        for i in range(30):
            streamer.publish(np.full((48, 64, 3), 128, dtype=np.uint8))
            time.sleep(0.005)
        thread.join(timeout=5.0)
        self.assertGreaterEqual(len(seen), 1)

    def test_wait_jpeg_wakes_on_shutdown(self):
        streamer = MjpegStreamer()
        result = []
        thread = threading.Thread(
            target=lambda: result.append(streamer.wait_jpeg(timeout=30.0)), daemon=True
        )
        thread.start()
        time.sleep(0.1)
        streamer.shutdown()
        thread.join(timeout=5.0)
        self.assertEqual(1, len(result))  # released, not left hanging
        streamer.shutdown()  # idempotent
        streamer.publish(_np().zeros((10, 10, 3), dtype=_np().uint8))  # no-op after shutdown
        self.assertEqual(0, streamer.frames_published)


@unittest.skipUnless(HAVE_FLASK, "flask not installed")
class WebAppTests(unittest.TestCase):
    """C9: the four frozen routes serve what the page polls."""

    def _app(self):
        from har.ui import web

        streamer = MjpegStreamer()
        self.addCleanup(streamer.shutdown)
        status = make_status()
        events = load_events(FIXTURES / "events_wrong_order.jsonl")
        web.bind_protocol(None)
        app = web.create_app(streamer, lambda: status, lambda n: events[-n:])
        app.config["TESTING"] = True
        return app, streamer, status, events

    def test_index_serves_the_console_page(self):
        app, _s, _st, _ev = self._app()
        with app.test_client() as client:
            resp = client.get("/")
        self.assertEqual(200, resp.status_code)
        body = resp.get_data(as_text=True)
        for marker in ('src="/stream"', "/status", "/events", "checklist"):
            self.assertIn(marker.lower(), body.lower())

    def test_status_round_trips_the_ui_status(self):
        import json

        app, _s, status, _ev = self._app()
        with app.test_client() as client:
            data = client.get("/status").get_json()
        # JSON domain comparison (tuples serialise to arrays).
        self.assertEqual(json.loads(status.to_json()), data)
        self.assertEqual("EXTRACT_RED", data["current_step_id"])

    def test_events_tail_respects_n(self):
        app, _s, _st, events = self._app()
        with app.test_client() as client:
            data = client.get("/events?n=2").get_json()
        self.assertEqual([e.to_dict() for e in events[-2:]], data)
        with app.test_client() as client:
            self.assertEqual(6, len(client.get("/events?n=99").get_json()))
            self.assertEqual(6, len(client.get("/events").get_json()))  # default n covers it
            self.assertFalse(client.get("/events?n=abc").status_code >= 500)

    def test_stream_is_multipart_and_carries_frames(self):
        np = _np()
        app, streamer, _st, _ev = self._app()
        streamer.publish(np.full((40, 60, 3), 80, dtype=np.uint8))  # before subscribe
        with app.test_client() as client:
            resp = client.get("/stream")  # unbuffered: pull a single chunk, then close
            self.assertEqual(200, resp.status_code)
            self.assertIn("multipart/x-mixed-replace", resp.content_type)
            chunk = next(resp.response)
            self.assertIn(b"Content-Type: image/jpeg", chunk)
            self.assertIn(b"\xff\xd8", chunk)  # JPEG magic bytes inside the part
            resp.close()

    def test_protocol_route_reporting(self):
        from har.ui import web

        app, _s, _st, _ev = self._app()
        with app.test_client() as client:
            self.assertEqual(404, client.get("/protocol").status_code)  # unbound

        class _Spec:
            def to_dict(self):
                return {"protocol_id": "PTS-01", "title": "t", "version": "1.0.0",
                        "steps": [], "zones": [], "objects": []}

        web.bind_protocol(_Spec())
        self.addCleanup(web.bind_protocol, None)
        with app.test_client() as client:
            self.assertEqual("PTS-01", client.get("/protocol").get_json()["protocol_id"])


@unittest.skipUnless(HAVE_CV2, "opencv/numpy not installed")
class OverlayTests(unittest.TestCase):
    """C10: the HUD draws in place, at demo and at thumbnail sizes."""

    def test_draw_hud_mutates_the_frame_and_returns_none(self):
        np = _np()
        from har.ui.overlay import draw_hud

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_hud(frame, make_status(
            violations=("EXTRACT_BLUE",),
            last_alert="Out of sequence. The red box must be placed before the blue box.",
        ), make_evidence())
        self.assertIsNone(result)  # in place, per Appendix A
        self.assertGreater(int(np.count_nonzero(frame)), 0)  # pixels were drawn
        # The red alert banner must actually be red somewhere near the bottom.
        bottom = frame[-160:, :, :]
        red_pixels = int(((bottom[:, :, 2] > 150) & (bottom[:, :, 0] < 100)).sum())
        self.assertGreater(red_pixels, 1000)

    def test_draw_hud_survives_tiny_frames(self):
        np = _np()
        from har.ui.overlay import draw_hud

        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        draw_hud(frame, make_status(), make_evidence())
        self.assertGreater(int(np.count_nonzero(frame)), 0)


if __name__ == "__main__":
    unittest.main()
