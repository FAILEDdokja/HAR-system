"""C3/C6 — OfflineSpeaker tests.  Standard-library only: the engine is
mocked through the ``engine_factory`` seam, so no pyttsx3, no espeak and no
audio device are needed."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from har.out.speaker import OfflineSpeaker  # noqa: E402


class FakeEngine:
    """pyttsx3-shaped stand-in with a deliberately slow speak cycle."""

    def __init__(self, per_say_s: float = 0.0, gate: threading.Event | None = None) -> None:
        self.per_say_s = per_say_s
        self.gate = gate
        self.utterances: list[str] = []
        self.rate = None
        self.stopped = False

    def setProperty(self, name: str, value) -> None:
        if name == "rate":
            self.rate = value

    def say(self, text: str) -> None:
        if self.gate is not None:
            self.gate.wait(timeout=10.0)
        if self.per_say_s:
            time.sleep(self.per_say_s)
        self.utterances.append(text)

    def runAndWait(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class NonBlockingTests(unittest.TestCase):
    """The C3 proof: the frame loop never waits on TTS."""

    def test_100_says_in_a_tight_loop_cost_nearly_nothing(self):
        # 100 utterances x 50 ms of speech = 5 s of engine work; the caller
        # must return in a tiny fraction of that.
        engine = FakeEngine(per_say_s=0.05)
        speaker = OfflineSpeaker(engine_factory=lambda: engine)
        self.addCleanup(speaker.stop)
        self.assertTrue(wait_until(lambda: speaker.available))
        start = time.monotonic()
        for i in range(100):
            speaker.say(f"utterance {i}")
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 1.0, f"say() blocked the caller: {elapsed:.3f}s")
        # The engine really is speaking (serially, behind the caller), and
        # everything that could not queue was dropped and counted, never waited on.
        self.assertTrue(wait_until(lambda: len(engine.utterances) >= 1))
        self.assertGreaterEqual(speaker.dropped, 90)

    def test_speak_cycle_runs_on_the_worker_thread(self):
        engine = FakeEngine()
        speaker = OfflineSpeaker(engine_factory=lambda: engine)
        self.addCleanup(speaker.stop)
        self.assertTrue(wait_until(lambda: speaker.available))
        speaker.say("step 3 skipped")
        self.assertTrue(wait_until(lambda: "step 3 skipped" in engine.utterances))
        self.assertEqual(165, engine.rate)  # rate property made it to the engine


class DropPolicyTests(unittest.TestCase):
    """One-deep queue: drop rather than block; priority evicts a queued low one."""

    def test_queue_never_exceeds_one_and_drops_are_counted(self):
        gate = threading.Event()  # engine busy until we let it finish
        engine = FakeEngine(gate=gate)
        speaker = OfflineSpeaker(engine_factory=lambda: engine)
        self.addCleanup(speaker.stop)
        self.assertTrue(wait_until(lambda: speaker.available))
        speaker.say("first")  # grabbed by the worker, which blocks on the gate
        self.assertTrue(wait_until(lambda: speaker._queue.empty()))
        for i in range(5):
            speaker.say(f"burst {i}")  # at most one can sit in the queue
        self.assertLessEqual(speaker._queue.qsize(), 1)
        self.assertGreaterEqual(speaker.dropped, 4)
        gate.set()
        self.assertTrue(wait_until(lambda: "first" in engine.utterances))

    def test_priority_evicts_a_queued_low_priority_utterance(self):
        gate = threading.Event()
        engine = FakeEngine(gate=gate)
        speaker = OfflineSpeaker(engine_factory=lambda: engine)
        self.addCleanup(speaker.stop)
        self.assertTrue(wait_until(lambda: speaker.available))
        speaker.say("speaking now")
        # Wait until the worker holds it (blocked on the gate) before queueing
        # the rest, or the assertions below race the queue.
        self.assertTrue(wait_until(lambda: speaker._queue.empty()))
        speaker.say("routine prompt", priority=0)
        speaker.say("SKIPPED: red box before blue box", priority=1)  # evicts the prompt
        gate.set()
        finished = wait_until(lambda: len(engine.utterances) >= 2)
        self.assertTrue(finished)
        self.assertIn("SKIPPED: red box before blue box", engine.utterances)
        self.assertNotIn("routine prompt", engine.utterances)


class DegradationTests(unittest.TestCase):
    """--no-voice and missing-driver machines must both be silent non-events."""

    def test_disabled_speaker_is_a_pure_noop(self):
        speaker = OfflineSpeaker(enabled=False, engine_factory=lambda: FakeEngine())
        self.assertIsNone(speaker._thread)  # no engine, no thread
        start = time.monotonic()
        for i in range(100):
            speaker.say(f"muted {i}", priority=1)
        self.assertLess(time.monotonic() - start, 0.5)
        speaker.stop()  # safe without start
        speaker.stop()  # and idempotent
        self.assertFalse(speaker.available)

    def test_driver_failure_disables_quietly(self):
        def boom():
            raise RuntimeError("no espeak on this box")

        speaker = OfflineSpeaker(engine_factory=boom)
        self.addCleanup(speaker.stop)
        self.assertTrue(wait_until(lambda: speaker.init_error is not None))
        self.assertFalse(speaker.available)
        self.assertIn("no espeak", speaker.init_error)
        for _ in range(10):
            speaker.say("still safe with a dead driver", priority=1)

    def test_engine_dying_mid_run_keeps_accepting_calls(self):
        class SickEngine(FakeEngine):
            def say(self, text):
                raise OSError("audio device vanished")

        speaker = OfflineSpeaker(engine_factory=SickEngine)
        self.addCleanup(speaker.stop)
        self.assertTrue(wait_until(lambda: speaker._available is True))
        speaker.say("will crash the engine")
        self.assertTrue(wait_until(lambda: speaker.init_error is not None))
        speaker.say("after the crash")  # must not raise

    def test_stop_shuts_the_engine_down(self):
        engine = FakeEngine()
        speaker = OfflineSpeaker(engine_factory=lambda: engine)
        self.assertTrue(wait_until(lambda: speaker.available))
        speaker.stop()
        self.assertTrue(engine.stopped)
        speaker.say("after stop")  # ignored, not queued


if __name__ == "__main__":
    unittest.main()
