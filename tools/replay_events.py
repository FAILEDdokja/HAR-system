#!/usr/bin/env python3
"""C1 — Replay canned ``StepEvent`` fixtures: Track C's stub for Person A.

The output/interface layer (log, voice, GUI) consumes ``StepEvent`` objects,
but at the start of Phase 1 the ``SequenceValidator`` that produces them did
not exist yet.  This tool reads ``tests/fixtures/events_*.jsonl`` (written by
hand to match ``StepEvent.to_dict()`` line-for-line) back into real
``contracts.StepEvent`` objects, so every C component could be built and
tested before Person A landed.

It stays useful afterwards:

* ``--pace`` re-emits the events at their recorded cadence, which drives the
  speaker (C3) and any sink exactly as a live run would — handy for checking
  the voice wiring with no camera and no validator in the loop.
* It is the reference ``StepEvent`` JSONL parser; ``tools/evaluate.py`` (A9)
  and the GUI's log tail accept the same format.

Usage (from the repo root)::

    .venv/bin/python tools/replay_events.py                          # both fixtures, table view
    .venv/bin/python tools/replay_events.py --fixture wrong_order
    .venv/bin/python tools/replay_events.py --pace --voice           # spoken replay
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from har.contracts import StepEvent  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
KNOWN_FIXTURES = {
    "correct": FIXTURES / "events_correct.jsonl",
    "wrong_order": FIXTURES / "events_wrong_order.jsonl",
}


def step_event_from_dict(d: dict) -> StepEvent:
    """Build a ``StepEvent`` from its ``to_dict()`` form.  Strict on fields:
    a drifted schema must fail loudly here rather than silently drop data."""
    return StepEvent(
        t_iso=str(d["t_iso"]),
        t_rel=float(d["t_rel"]),
        frame_index=int(d["frame_index"]),
        step_id=str(d["step_id"]),
        step_index=int(d["step_index"]),
        event=str(d["event"]),
        status=str(d["status"]),
        message=str(d.get("message", "")),
        confidence=float(d.get("confidence", 0.0)),
    )


def iter_events(path: str | Path) -> Iterator[StepEvent]:
    """Yield ``StepEvent`` objects from a JSONL file, skipping blank lines."""
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield step_event_from_dict(json.loads(line))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_no}: not a StepEvent row: {exc}") from exc


def load_events(path: str | Path) -> list[StepEvent]:
    """Read a whole ``events_*.jsonl`` file into a list of ``StepEvent``."""
    return list(iter_events(path))


def resolve_fixture(name_or_path: str) -> Path:
    """Accept a known fixture name or an explicit path."""
    if name_or_path in KNOWN_FIXTURES:
        return KNOWN_FIXTURES[name_or_path]
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    known = ", ".join(sorted(KNOWN_FIXTURES))
    raise SystemExit(f"unknown fixture {name_or_path!r} (known: {known}, or pass a file path)")


def _print_table(events: Sequence[StepEvent]) -> None:
    for e in events:
        message = f"  {e.message}" if e.message else ""
        print(f"  t={e.t_rel:7.3f}  f={e.frame_index:4d}  {e.event:17s} {e.status:11s} {e.step_id}{message}")


def replay_paced(events: Sequence[StepEvent], voice: bool = False, speed: float = 1.0) -> float:
    """Re-emit events at their recorded cadence; returns wall seconds taken.

    ``speed`` scales the cadence (2.0 = twice as fast).  When ``voice`` is
    set, violation/completion messages are spoken through the C3 speaker —
    the no-camera proof that the voice path is wired correctly.
    """
    speaker = None
    if voice:
        from har.out.speaker import OfflineSpeaker

        speaker = OfflineSpeaker()
    start = time.monotonic()
    t0 = events[0].t_rel if events else 0.0
    try:
        for event in events:
            wait = (event.t_rel - t0) / max(speed, 1e-6) - (time.monotonic() - start)
            if wait > 0:
                time.sleep(wait)
            print(f"  t={event.t_rel:7.3f}  {event.event:17s} {event.status:11s} {event.step_id}")
            if speaker is not None and (event.message or event.status == "VIOLATION"):
                speaker.say(event.message or f"{event.event} {event.step_id}",
                            priority=1 if event.status == "VIOLATION" else 0)
    finally:
        if speaker is not None:
            speaker.stop()
    return time.monotonic() - start


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fixture",
        action="append",
        default=None,
        help="fixture name (correct, wrong_order) or a path; repeatable. Default: both.",
    )
    parser.add_argument("--pace", action="store_true", help="re-emit at the recorded cadence")
    parser.add_argument("--speed", type=float, default=1.0, help="cadence multiplier for --pace")
    parser.add_argument("--voice", action="store_true", help="speak messages through har.out.speaker")
    args = parser.parse_args(argv)

    names = args.fixture or sorted(KNOWN_FIXTURES)
    for name in names:
        path = resolve_fixture(name)
        events = load_events(path)
        print(f"{name}: {len(events)} events from {path.relative_to(REPO)}")
        if args.pace:
            replay_paced(events, voice=args.voice, speed=args.speed)
        else:
            _print_table(events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
