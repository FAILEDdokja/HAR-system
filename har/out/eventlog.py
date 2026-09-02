"""Append-only event log (C2) — deliverable D4, the timestamped text file.

``JsonlEventLog`` implements the frozen ``contracts.EventSink`` seam.  Every
``StepEvent`` is written twice:

* one compact JSON object per line in ``events.jsonl`` (round-trips exactly
  through ``StepEvent.to_dict()`` — ``tools/replay_events.py`` and A9's
  evaluator parse it back losslessly), and
* one row in the mirrored ``events.csv`` for judges who open spreadsheets.

**Flush per event is the point of this class.**  The log is the audit trail
of the whole run; a crash, a power cut or a killed process must not lose the
last warning.  ``emit`` therefore flushes the Python buffer and issues
``os.fsync`` after every event, and the C6 tests prove the file is readable
after an abrupt ``os._exit`` with no ``close``.

The module is standard-library only, so it imports and runs in a bare
interpreter — same rule the contracts follow.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from pathlib import Path
from typing import Any

from har.contracts import StepEvent

__all__ = ["JsonlEventLog", "CSV_FIELDS"]

#: Column order of the mirrored CSV — exactly the ``StepEvent`` schema.
CSV_FIELDS: tuple[str, ...] = (
    "t_iso",
    "t_rel",
    "frame_index",
    "step_id",
    "step_index",
    "event",
    "status",
    "message",
    "confidence",
)


class JsonlEventLog:
    """Append-only ``StepEvent`` sink writing JSONL plus a mirrored CSV.

    Files are opened in append mode so a restarted run never truncates an
    earlier log; the CSV header is (re)written only when the CSV is empty.
    """

    def __init__(self, jsonl_path: str | Path, csv_path: str | Path | None = None) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = 0
        self._closed = False

        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._csv_file = None
        self._csv_writer: csv.writer | None = None
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = (
                not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            )
            # newline="" per csv module docs; utf-8 so voice-alert text is safe.
            self._csv_file = self.csv_path.open("a", encoding="utf-8", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            if write_header:
                self._csv_writer.writerow(CSV_FIELDS)
                self._flush()

    # ------------------------------------------------------------------
    # contracts.EventSink
    # ------------------------------------------------------------------

    def emit(self, event: StepEvent) -> None:
        """Append one event and flush it to stable storage before returning."""
        with self._lock:
            if self._closed:
                raise RuntimeError("JsonlEventLog.emit() after close()")
            row = event.to_dict()
            self._jsonl.write(json.dumps(row, separators=(",", ":")) + "\n")
            if self._csv_writer is not None:
                self._csv_writer.writerow([row[field] for field in CSV_FIELDS])
            self._flush()
            self._count += 1

    def close(self) -> None:
        """Flush and close both files.  Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._flush()
            self._jsonl.close()
            if self._csv_file is not None:
                self._csv_file.close()

    # ------------------------------------------------------------------
    # extras
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of events emitted so far."""
        return self._count

    @property
    def closed(self) -> bool:
        return self._closed

    def _flush(self) -> None:
        """Push both streams all the way to the OS (survives SIGKILL)."""
        self._jsonl.flush()
        os.fsync(self._jsonl.fileno())
        if self._csv_file is not None:
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())

    def __enter__(self) -> "JsonlEventLog":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{self._count} events"
        return f"JsonlEventLog({self.jsonl_path!s}, {state})"
