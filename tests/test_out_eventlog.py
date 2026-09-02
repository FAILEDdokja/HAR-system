"""C2/C6 — JsonlEventLog tests.  Standard-library only: runs in a bare
interpreter alongside the 28 dependency-free tests in the baseline suite."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from har.contracts import StepEvent  # noqa: E402
from har.out.eventlog import CSV_FIELDS, JsonlEventLog  # noqa: E402
from tools.replay_events import load_events  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"


def make_event(**overrides) -> StepEvent:
    base = dict(
        t_iso="2026-09-03T09:00:07+05:30",
        t_rel=7.25,
        frame_index=190,
        step_id="EXTRACT_RED",
        step_index=3,
        event="SKIPPED",
        status="VIOLATION",
        message="Step 3 skipped. The red box must go to zone A, before the blue box.",
        confidence=0.91,
    )
    base.update(overrides)
    return StepEvent(**base)


class JsonlRoundTripTests(unittest.TestCase):
    """Every StepEvent field must survive the file (D4 is the audit trail)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.jsonl = self.dir / "events.jsonl"
        self.csv = self.dir / "events.csv"

    def _write_fixtures(self) -> list[StepEvent]:
        events = load_events(FIXTURES / "events_correct.jsonl") + load_events(
            FIXTURES / "events_wrong_order.jsonl"
        )
        with JsonlEventLog(self.jsonl, self.csv) as log:
            for event in events:
                log.emit(event)
        return events

    def test_every_field_survives_jsonl(self):
        events = self._write_fixtures()
        restored = load_events(self.jsonl)
        self.assertEqual(len(events), len(restored))
        for original, back in zip(events, restored):
            self.assertEqual(original, back)  # frozen dataclass equality over all fields
            self.assertEqual(original.to_dict(), json.loads(json.dumps(back.to_dict())))

    def test_csv_mirrors_jsonl_row_for_row(self):
        events = self._write_fixtures()
        with self.csv.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(list(CSV_FIELDS), rows[0])  # header
        self.assertEqual(len(events), len(rows) - 1)
        # The message fixture contains a comma — csv quoting must protect it.
        messages = [row[CSV_FIELDS.index("message")] for row in rows[1:]]
        self.assertEqual([e.message for e in events], messages)
        ids = [row[CSV_FIELDS.index("step_id")] for row in rows[1:]]
        self.assertEqual([e.step_id for e in events], ids)

    def test_append_mode_preserves_an_earlier_run(self):
        with JsonlEventLog(self.jsonl) as log:
            log.emit(make_event(message="first run"))
        with JsonlEventLog(self.jsonl) as log:  # reopened, not truncated
            log.emit(make_event(message="second run"))
        restored = load_events(self.jsonl)
        self.assertEqual(["first run", "second run"], [e.message for e in restored])

    def test_emit_after_close_raises(self):
        log = JsonlEventLog(self.jsonl)
        log.close()
        with self.assertRaises(RuntimeError):
            log.emit(make_event())
        log.close()  # close is idempotent

    def test_count_and_repr(self):
        with JsonlEventLog(self.jsonl) as log:
            self.assertEqual(0, log.count)
            log.emit(make_event())
            self.assertEqual(1, log.count)
            self.assertIn("1 events", repr(log))
        self.assertTrue(log.closed)


class FlushDurabilityTests(unittest.TestCase):
    """'Flush after every event — a crash must not lose the log' (plan C2)."""

    def test_rows_are_on_disk_without_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "events.jsonl"
            log = JsonlEventLog(jsonl)
            log.emit(make_event())
            # Read through a *separate* handle while the writer is still open.
            rows = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln]
            self.assertEqual(1, len(rows))
            self.assertEqual("SKIPPED", json.loads(rows[0])["event"])
            log.close()

    def test_log_survives_an_abrupt_exit(self):
        """A killed process (os._exit — no close, no atexit) loses nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "events.jsonl"
            csv_path = Path(tmp) / "events.csv"
            code = (
                "import os, sys\n"
                "sys.path.insert(0, {repo!r})\n"
                "from har.out.eventlog import JsonlEventLog\n"
                "from tests.test_out_eventlog import make_event\n"
                "log = JsonlEventLog({jsonl!r}, {csv!r})\n"
                "log.emit(make_event(message='alive'))\n"
                "os._exit(0)\n"
            ).format(repo=str(REPO), jsonl=str(jsonl), csv=str(csv_path))
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
            )
            self.assertEqual(0, result.returncode, result.stderr)
            restored = load_events(jsonl)
            self.assertEqual(["alive"], [e.message for e in restored])
            with csv_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(2, len(rows))  # header + the row that outlived its process


if __name__ == "__main__":
    unittest.main()
