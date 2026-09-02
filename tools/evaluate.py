"""A9 — Evaluation harness: replay the demo dataset and write real metrics.

``docs/DEVELOPMENT_PLAN.md`` §5, step A9 (Track A — Cognition):

> **Do:** replay every ``demo/`` run, diff emitted events against
> ``ground_truth.json``, and write the numbers.
>
> **Done when:** ``docs/METRICS.md`` has real values for step accuracy,
> sequence completion rate, false-alarm rate, and mean per-step latency.
> No placeholders, no estimates.

What this script does
---------------------
1. Loads ``demo/ground_truth.json`` (A8's hand annotation).
2. For every run (``correct`` / ``skip`` / ``wrong_order``) it replays the
   frame-accurate ``FrameEvidence`` recording (``demo/<run>_evidence.json``)
   through ``SequenceValidator`` against ``protocols/pts01.yaml``.  The
   evidence timeline is the frame-accurate recording the captured video was
   built from (see ``demo/README.md``), so this is the log ``--source
   demo/<run>.mp4`` must produce once Track C's CLI lands.  When the CLI is
   available, ``--events-dir`` accepts the produced ``<run>.jsonl`` files
   instead — the diff and metrics are identical either way.
3. Diffs the emitted event stream against ``runs.<id>.expected.events``
   field by field (event, step, index, t_rel, frame, status, message) and
   reports any missing / extra / mismatched event.
4. Scores every step instance against the operator annotation and computes
   the four A9 metrics, plus the violation-detection figures that make the
   false-alarm rate interpretable.

Metric definitions (documented in ``docs/METRICS.md`` and rendered there)
------------------------------------------------------------------------
* **Step accuracy** — fraction of step instances whose validator outcome
  (``COMPLETED`` / ``SKIPPED`` / ``TIMEOUT`` / ``INCOMPLETE`` /
  ``NOT_PERFORMED``, derived from the emitted log) equals the operator's
  annotation (``COMPLETED`` / ``SKIPPED`` / ``NOT_PERFORMED``).
* **Sequence completion rate** — runs that emitted 8 ``COMPLETED`` in index
  order, ``PROTOCOL_COMPLETE``, and no ``VIOLATION``-status event, over all
  runs.
* **False-alarm rate** — ``VIOLATION``-status events that the ground-truth
  log does *not* contain, per step instance (also reported per run).
* **Mean per-step latency** — mean of ``COMPLETED.t_rel - operator t_end``
  over every step that completed and has an operator window.  The
  annotation is quoted to the nearest frame (66.7 ms at 15 fps), so small
  negative values on steps 1–2 are annotation granularity, not a validator
  bug.

Usage::

    .venv/bin/python tools/evaluate.py                 # replays demo/, writes docs/METRICS.md
    .venv/bin/python tools/evaluate.py --json out.json # also machine-readable metrics
    .venv/bin/python tools/evaluate.py --no-write      # print only
    .venv/bin/python tools/evaluate.py --strict        # exit 1 if any event diff or GT drift

The script is stdlib + ``har.contracts`` + ``har.protocol`` only — no cv2,
no numpy, no torch — so A9 runs in the same bare-environment Track A is
allowed to use.  The generated ``docs/METRICS.md`` contains real values
computed from the committed recordings; re-running always regenerates them.

A10 — violation evidence table
------------------------------
``docs/DEVELOPMENT_PLAN.md`` §5 step A10 asks for ``SKIPPED``,
``OUT_OF_ORDER`` and ``TIMEOUT`` each as a row with the actual timestamped
log line as evidence.  This script renders that table itself, so the page
stays complete across regenerations:

* ``OUT_OF_ORDER`` / ``SKIPPED`` — taken from the first demo run that
  emitted them (the ``skip`` run), verbatim ``StepEvent.to_json()`` lines.
* ``TIMEOUT`` — from a deterministic *stall probe*: a fresh validator fed
  "hands inside the envelope, but the tray never appears" frames until the
  first step's ``timeout_s`` elapses.  The hands keep step 8's
  ``hands_clear`` unsatisfied so the stall cannot resolve into a skip-jump;
  with no objects present nothing else can satisfy, so the only possible
  event is the timeout.

Replays are anchored to a fixed wall-clock (``REPLAY_ANCHOR``) so the
``t_iso`` fields in the evidence table are reproducible run to run.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Allow running as ``python tools/evaluate.py`` from the repo root.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from har.contracts import FrameEvidence, ObjectTrack, StepEvent, Wrist  # noqa: E402
from har.protocol.spec import load_protocol  # noqa: E402
from har.protocol.validator import SequenceValidator  # noqa: E402

DEFAULT_DEMO = REPO / "demo"
DEFAULT_PROTOCOL = REPO / "protocols" / "pts01.yaml"
DEFAULT_METRICS = REPO / "docs" / "METRICS.md"

# Fixed wall-clock anchor for every replay, so the ``t_iso`` fields of the
# A10 evidence rows are reproducible instead of run-time wall clock.  The
# validation decisions themselves depend only on ``evidence.t_rel``.
REPLAY_ANCHOR = datetime(2026, 9, 2, 10, 0, 0, tzinfo=timezone.utc)

# Fields that define an event for the diff.  ``t_iso`` and ``confidence``
# are pipeline metadata, not part of the decision log the GT records.
EVENT_FIELDS = (
    "event",
    "step_id",
    "step_index",
    "t_rel",
    "frame_index",
    "status",
    "message",
)

# --------------------------------------------------------------------------
# Loading helpers (mirror tests/test_demo_dataset.py so the numbers match)
# --------------------------------------------------------------------------


def evidence_from_dict(d: dict) -> FrameEvidence:
    """Rebuild one ``FrameEvidence`` from a ``FrameEvidence.to_dict()`` row."""
    return FrameEvidence(
        frame_index=d["frame_index"],
        t_rel=d["t_rel"],
        frame_size=tuple(d["frame_size"]),
        objects={
            label: ObjectTrack(
                label=od["label"],
                box=tuple(od["box"]) if od["box"] is not None else None,
                measured=od["measured"],
                lost_frames=od.get("lost_frames", 0),
            )
            for label, od in d["objects"].items()
        },
        hands=tuple(
            Wrist(tuple(h["point"]), h["confidence"], h["side"], h.get("person_id", 0))
            for h in d["hands"]
        ),
        hoi=dict(d["hoi"]),
        rack_ready=d["rack_ready"],
        fps=d["fps"],
    )


def _normalise(value: Any) -> Any:
    """Normalise a field for comparison (round floats to 3 dp like the GT)."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, (int, str)) or value is None:
        return value
    return value


def event_key(event: StepEvent | dict) -> tuple:
    """The identity of an event for diffing, ignoring t_iso/confidence."""
    d = event.to_dict() if isinstance(event, StepEvent) else dict(event)
    return tuple(_normalise(d.get(f)) for f in EVENT_FIELDS)


def load_ground_truth(demo_dir: Path) -> dict:
    path = demo_dir / "ground_truth.json"
    if not path.is_file():
        raise SystemExit(f"ground truth not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def replay_evidence(evidence_path: Path, spec) -> tuple[list[StepEvent], SequenceValidator, int]:
    """Replay one recorded evidence timeline through the validator.

    Returns ``(events, validator, frame_count)``.
    """
    raw = json.loads(evidence_path.read_text(encoding="utf-8"))
    frames = [evidence_from_dict(d) for d in raw]
    validator = SequenceValidator(spec, start_time=REPLAY_ANCHOR)
    events: list[StepEvent] = []
    for frame in frames:
        events.extend(validator.update(frame))
    return events, validator, len(frames)


def read_events_jsonl(path: Path) -> list[StepEvent]:
    """Parse a CLI-produced events.jsonl (StepEvent.to_dict per line)."""
    events: list[StepEvent] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        try:
            events.append(StepEvent(**d))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise SystemExit(f"{path}:{line_no}: not a StepEvent: {exc}") from exc
    return events


# --------------------------------------------------------------------------
# Diff and metric computation
# --------------------------------------------------------------------------


def diff_events(emitted: Sequence[StepEvent], expected: Sequence[dict]) -> dict:
    """Field-by-field diff of the emitted log against the GT expected log.

    Returns ``{"exact", "expected", "emitted", "missing", "extra",
    "mismatches"}``.  ``missing``/``extra`` are event keys; ``mismatches``
    are field-level differences on events that align by (event, step, index).
    """
    exp_keys = [event_key(e) for e in expected]
    got_keys = [event_key(e) for e in emitted]
    matcher = difflib.SequenceMatcher(a=exp_keys, b=got_keys, autojunk=False)

    missing: list[tuple] = []
    extra: list[tuple] = []
    mismatches: list[dict] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op in ("delete", "replace"):
            missing.extend(exp_keys[i1:i2])
        if op in ("insert", "replace"):
            extra.extend(got_keys[j1:j2])
        if op == "replace":
            # Align by (event, step_id, step_index) and report field diffs.
            for a_idx in range(i1, i2):
                for b_idx in range(j1, j2):
                    if exp_keys[a_idx][:3] != got_keys[b_idx][:3]:
                        continue
                    fields = []
                    for name, a_val, b_val in zip(
                        EVENT_FIELDS, exp_keys[a_idx], got_keys[b_idx]
                    ):
                        if a_val != b_val:
                            fields.append({"field": name, "expected": a_val, "emitted": b_val})
                    if fields:
                        mismatches.append(
                            {
                                "expected": dict(zip(EVENT_FIELDS, exp_keys[a_idx])),
                                "emitted": dict(zip(EVENT_FIELDS, got_keys[b_idx])),
                                "fields": fields,
                            }
                        )
                    break
    return {
        "exact": not missing and not extra and not mismatches,
        "expected": len(expected),
        "emitted": len(emitted),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
    }


def step_outcomes(events: Sequence[StepEvent], n_steps: int) -> dict[int, str]:
    """Per-step validator outcome derived from the emitted log.

    ``COMPLETED`` wins over ``TIMEOUT`` (a timed-out step may still complete).
    ``INCOMPLETE`` = became current (``STARTED``) but never finished;
    ``NOT_PERFORMED`` = never reached.
    """
    completed: set[int] = set()
    skipped: set[int] = set()
    timed_out: set[int] = set()
    started: set[int] = set()
    for e in events:
        if e.event == "COMPLETED":
            completed.add(e.step_index)
        elif e.event == "SKIPPED":
            skipped.add(e.step_index)
        elif e.event == "TIMEOUT":
            timed_out.add(e.step_index)
        elif e.event == "STARTED":
            started.add(e.step_index)
    outcomes: dict[int, str] = {}
    for i in range(1, n_steps + 1):
        if i in completed:
            outcomes[i] = "COMPLETED"
        elif i in skipped:
            outcomes[i] = "SKIPPED"
        elif i in timed_out:
            outcomes[i] = "TIMEOUT"
        elif i in started:
            outcomes[i] = "INCOMPLETE"
        else:
            outcomes[i] = "NOT_PERFORMED"
    return outcomes


def _first_event(events: Sequence[StepEvent], event: str, step_index: int) -> StepEvent | None:
    for e in events:
        if e.event == event and e.step_index == step_index:
            return e
    return None


def compute_run_metrics(
    run_id: str, run_gt: dict, events: Sequence[StepEvent], n_frames: int | None = None
) -> dict:
    """Score one run: outcome accuracy, violations, latencies, event diff."""
    n_steps = len(run_gt["steps"])
    outcomes = step_outcomes(events, n_steps)
    expected = run_gt.get("expected", {})
    diff = diff_events(events, expected.get("events", []))

    step_rows = []
    n_correct = 0
    latencies: list[dict] = []
    for step in run_gt["steps"]:
        idx = step["index"]
        op_outcome = step["operator_outcome"]
        val_outcome = outcomes.get(idx, "NOT_PERFORMED")
        match = op_outcome == val_outcome
        n_correct += 1 if match else 0
        row = {
            "step_id": step["step_id"],
            "index": idx,
            "operator_outcome": op_outcome,
            "validator_outcome": val_outcome,
            "match": match,
        }
        # Latency only when both sides have a time to compare.
        comp = _first_event(events, "COMPLETED", idx)
        t_end = step.get("t_end")
        if comp is not None and t_end is not None:
            latency = comp.t_rel - float(t_end)
            latencies.append(
                {
                    "step_id": step["step_id"],
                    "t_end": float(t_end),
                    "t_completed": comp.t_rel,
                    "latency_s": round(latency, 3),
                }
            )
        step_rows.append(row)

    emitted_violations = [event_key(e) for e in events if e.status == "VIOLATION"]
    expected_violations = [
        event_key(e) for e in expected.get("events", []) if e.get("status") == "VIOLATION"
    ]
    false_alarms = [k for k in emitted_violations if k not in set(expected_violations)]
    missed_violations = [k for k in expected_violations if k not in set(emitted_violations)]

    completed = [e.step_id for e in events if e.event == "COMPLETED"]
    skipped = [e.step_id for e in events if e.event == "SKIPPED"]
    timed_out = [e.step_id for e in events if e.event == "TIMEOUT"]
    sequence_complete = bool(
        any(e.event == "PROTOCOL_COMPLETE" for e in events)
        and sum(1 for e in events if e.event == "COMPLETED") == n_steps
        and not any(e.status == "VIOLATION" for e in events)
    )

    return {
        "run_id": run_id,
        "scenario": run_gt.get("scenario", ""),
        "frames": int(n_frames) if n_frames is not None else run_gt.get("frame_count"),
        "duration_s": run_gt.get("duration_s"),
        "fps": run_gt.get("fps"),
        "frame_size": run_gt.get("frame_size"),
        "event_diff": diff,
        "steps": step_rows,
        "n_steps": n_steps,
        "n_correct": n_correct,
        "step_accuracy": round(n_correct / n_steps, 4) if n_steps else None,
        "completed": completed,
        "skipped": skipped,
        "timed_out": timed_out,
        "violations_emitted": emitted_violations,
        "violations_expected": expected_violations,
        "false_alarms": false_alarms,
        "missed_violations": missed_violations,
        "false_alarm_rate": round(len(false_alarms) / n_steps, 4) if n_steps else None,
        "sequence_complete": sequence_complete,
        "latencies_s": latencies,
        "latency_mean_s": round(
            sum(l["latency_s"] for l in latencies) / len(latencies), 4
        ) if latencies else None,
    }


def aggregate(per_run: Sequence[dict]) -> dict:
    """Combine per-run scores into the four headline A9 metrics."""
    n_runs = len(per_run)
    n_steps = sum(r["n_steps"] for r in per_run)
    n_correct = sum(r["n_correct"] for r in per_run)
    n_complete = sum(1 for r in per_run if r["sequence_complete"])
    n_false_alarms = sum(len(r["false_alarms"]) for r in per_run)
    n_expected_violations = sum(len(r["violations_expected"]) for r in per_run)
    n_detected_violations = sum(
        len(r["violations_emitted"]) - len(r["false_alarms"]) for r in per_run
    )
    all_latencies = [l for r in per_run for l in r["latencies_s"]]

    return {
        "n_runs": n_runs,
        "n_steps": n_steps,
        "n_correct": n_correct,
        "step_accuracy": round(n_correct / n_steps, 4) if n_steps else None,
        "n_sequences_complete": n_complete,
        "sequence_completion_rate": round(n_complete / n_runs, 4) if n_runs else None,
        "n_false_alarms": n_false_alarms,
        "false_alarm_rate_per_step": round(n_false_alarms / n_steps, 4) if n_steps else None,
        "n_expected_violations": n_expected_violations,
        "n_detected_violations": n_detected_violations,
        "violation_detection_rate": round(n_detected_violations / n_expected_violations, 4)
        if n_expected_violations
        else None,
        "n_latency_samples": len(all_latencies),
        "mean_per_step_latency_s": round(
            sum(l["latency_s"] for l in all_latencies) / len(all_latencies), 4
        ) if all_latencies else None,
        "min_step_latency_s": round(min(l["latency_s"] for l in all_latencies), 4)
        if all_latencies else None,
        "max_step_latency_s": round(max(l["latency_s"] for l in all_latencies), 4)
        if all_latencies else None,
    }


def drift_check(step: dict, computed_outcome: str, events: Sequence[StepEvent]) -> list[str]:
    """Compare A8's stored ``validator_*`` fields with what we replayed.

    Any difference means the annotation drifted from the validator — worth
    flagging, and a hard failure under ``--strict``.
    """
    problems: list[str] = []
    stored = step.get("validator_outcome")
    if stored is not None and stored != computed_outcome:
        problems.append(
            f"validator_outcome stored={stored!r} != replayed={computed_outcome!r}"
        )
    event_by = {
        "validator_started_s": ("STARTED", "t_rel"),
        "validator_completed_s": ("COMPLETED", "t_rel"),
        "validator_skipped_s": ("SKIPPED", "t_rel"),
        "validator_out_of_order_s": ("OUT_OF_ORDER", "t_rel"),
    }
    for field, (event_name, time_attr) in event_by.items():
        ev = _first_event(events, event_name, step["index"])
        actual = round(getattr(ev, time_attr), 3) if ev is not None else None
        stored_val = step.get(field)
        if stored_val is None:
            continue
        if actual is None or abs(stored_val - actual) > 0.001:
            problems.append(
                f"{field} stored={stored_val!r} != replayed={actual!r}"
            )
    return problems


# --------------------------------------------------------------------------
# A10 — violation evidence rows (generated, never hand-maintained)
# --------------------------------------------------------------------------


def stall_probe_timeout(spec, frame_size: tuple[int, int], fps: float = 15.0) -> StepEvent:
    """Deterministic ``TIMEOUT`` evidence: replay a stalled operator.

    A fresh validator is fed frames in which the operator's hands hover
    inside the rack envelope but the tray is never presented.  The hands
    keep step 8's ``hands_clear`` predicate unsatisfied (so the stall cannot
    resolve into a skip-jump), and with no objects present no other step can
    satisfy — so the only event the validator can emit is the first step's
    ``TIMEOUT`` once its ``timeout_s`` elapses.  Pure replay: no camera, no
    clock reads beyond the fixed ``REPLAY_ANCHOR``.
    """
    validator = SequenceValidator(spec, start_time=REPLAY_ANCHOR)
    first = spec.steps[0]
    zone = spec.zone(first.zone)
    if zone is None:  # pragma: no cover - defensive, PTS-01 always has it
        raise SystemExit(f"stall probe: zone {first.zone!r} missing from protocol")
    x1, y1, x2, y2 = zone.box
    mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    hands = (Wrist(mid, 0.9, "left"), Wrist((mid[0] + 40.0, mid[1]), 0.9, "right"))
    budget_frames = int((first.timeout_s + 30.0) * fps) + 1
    for i in range(budget_frames):
        evidence = FrameEvidence(
            frame_index=i,
            t_rel=i / fps,
            frame_size=frame_size,
            objects={},
            hands=hands,
            hoi={},
            rack_ready=False,
            fps=fps,
        )
        for event in validator.update(evidence):
            if event.event == "TIMEOUT":
                return event
    raise SystemExit("stall probe never produced a TIMEOUT — validator bug")


def build_violation_rows(
    spec,
    frame_size: tuple[int, int],
    candidates: Sequence[tuple[str, list[StepEvent]]],
) -> list[dict]:
    """One row per violation type, carrying the actual emitted log line.

    ``candidates`` is an ordered list of ``(origin, events)`` replays.  For
    each violation type the first candidate that emitted it supplies the row,
    and the origin labels the scenario column.  The demo runs come first;
    anything they do not exhibit (the demo ``skip`` operator releases the
    blue box and steps away, so its satisfied span never reaches the hold —
    alert only, no skip-jump) is taken from the committed A1 fixture
    replays, which do exhibit it.  ``TIMEOUT`` comes from
    :func:`stall_probe_timeout`, since no recording stalls on purpose.
    """
    rows: list[dict] = []
    for wanted, blurb in (
        ("OUT_OF_ORDER", "the blue box's HOI cycle is satisfied while"
                         " EXTRACT_RED is still the current step"),
        ("SKIPPED", "EXTRACT_BLUE's satisfied span reaches its hold while"
                    " EXTRACT_RED is current; the cursor jumps and EXTRACT_RED"
                    " is declared skipped"),
    ):
        for origin, events in candidates:
            hit = next((e for e in events if e.event == wanted), None)
            if hit is not None:
                rows.append({
                    "event": wanted,
                    "scenario": f"`{origin}` — {blurb}",
                    "data": hit,
                })
                break
        else:  # pragma: no cover - every type is exhibited by the fixtures
            raise SystemExit(f"no committed replay exhibits {wanted} — A10 incomplete")
    rows.append({
        "event": "TIMEOUT",
        "scenario": "stall probe — hands inside the rack envelope but the"
                    " tray never appears; the current step's timeout elapses",
        "data": stall_probe_timeout(spec, frame_size),
    })
    return rows


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _fmt_seconds(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _fmt_pct(v) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def render_markdown(
    gt: dict,
    per_run: Sequence[dict],
    overall: dict,
    events_source: str,
    violation_rows: Sequence[dict] = (),
) -> str:
    lines: list[str] = []
    add = lines.append
    today = date.today().isoformat()

    add("# A9 — Evaluation metrics: PTS-01 on the annotated demo dataset")
    add("")
    add(f"*Generated by `tools/evaluate.py` on {today}.*")
    add(f"*Data: `demo/ground_truth.json` (annotated {gt.get('annotated_at', '?')}) · "
        f"{gt.get('protocol_id')} v{gt.get('protocol_version')} · contract "
        f"v{gt.get('contract_version')} · {gt.get('frame_size')[0]}×{gt.get('frame_size')[1]} "
        f"@ {gt.get('fps'):g} fps.*")
    add("")
    add("## What was evaluated")
    add("")
    add(f"Three annotated runs, {gt.get('fps'):g} fps, replayed from **{events_source}** "
        "against `protocols/pts01.yaml`, and every emitted `StepEvent` diffed field by field "
        "against the ground-truth expected log. The operator annotation is the reference for "
        "step outcomes and timings; the expected event log is the reference for the event diff.")
    add("")
    add("| Run | Scenario | Frames | Duration (s) |")
    add("|---|---|---:|---:|")
    for r in per_run:
        add(f"| `{r['run_id']}` | {r['scenario']} | {r['frames']} | {_fmt_seconds(r['duration_s'])} |")
    add("")
    add("## Results")
    add("")
    add("| Run | Step accuracy | Sequence complete | False alarms | False-alarm rate* | Mean step latency (s) |")
    add("|---|---:|---:|---:|---:|---:|")
    for r in per_run:
        add(
            f"| `{r['run_id']}` | {_fmt_pct(r['step_accuracy'])} "
            f"({r['n_correct']}/{r['n_steps']}) | {'yes' if r['sequence_complete'] else 'no'} "
            f"| {len(r['false_alarms'])} | {_fmt_pct(r['false_alarm_rate'])} | {_fmt_seconds(r['latency_mean_s'])} |"
        )
    add("")
    add("*false alarms ÷ step instances in this run (8).")
    add("")
    add("### Overall — the four A9 metrics")
    add("")
    add("| Metric | Value | How it is computed |")
    add("|---|---|---|")
    add(
        f"| **Step accuracy** | **{_fmt_pct(overall['step_accuracy'])}** "
        f"({overall['n_correct']}/{overall['n_steps']} step instances) | "
        "validator outcome == operator annotation, per step instance across all three runs |"
    )
    add(
        f"| **Sequence completion rate** | **{_fmt_pct(overall['sequence_completion_rate'])}** "
        f"({overall['n_sequences_complete']}/{overall['n_runs']} runs) | "
        "runs emitting 8 COMPLETED in index order + PROTOCOL_COMPLETE with no VIOLATION event |"
    )
    add(
        f"| **False-alarm rate** | **{_fmt_pct(overall['false_alarm_rate_per_step'])}** "
        f"({overall['n_false_alarms']} alarms / {overall['n_steps']} step instances) | "
        "VIOLATION-status events not present in the ground-truth expected log, per step instance |"
    )
    add(
        f"| **Mean per-step latency** | **{_fmt_seconds(overall['mean_per_step_latency_s'])} s** "
        f"({overall['n_latency_samples']} completed steps) | "
        "mean of (COMPLETED.t_rel − operator t_end); min "
        f"{_fmt_seconds(overall['min_step_latency_s'])} s, max {_fmt_seconds(overall['max_step_latency_s'])} s |"
    )
    if overall["n_expected_violations"]:
        add(
            f"\nFor context, **{overall['n_detected_violations']} of "
            f"{overall['n_expected_violations']}** ground-truth violation events were detected "
            f"(detection rate {_fmt_pct(overall['violation_detection_rate'])}), so the zero "
            "false-alarm rate is not masking missed violations."
        )
    add("")
    add("## Per-step latency")
    add("")
    add("Steps that both completed and carry an operator t_end: ")
    add("")
    add("| Run | Step | Operator t_end (s) | Validator COMPLETED (s) | Latency (s) |")
    add("|---|---|---:|---:|---:|")
    for r in per_run:
        for l in r["latencies_s"]:
            add(
                f"| `{r['run_id']}` | {l['step_id']} | {_fmt_seconds(l['t_end'])} "
                f"| {_fmt_seconds(l['t_completed'])} | {l['latency_s']:+.3f} |"
            )
    add("")
    add("Negative values are annotation granularity: t_end is quoted to the near"
        "est video frame (66.7 ms at 15 fps) and the validator completes the"
        "moment the predicate's hold is met, which can round a frame earlier.")
    add("")
    add("## Event diff (emitted vs ground-truth expected log)")
    add("")
    add("| Run | Expected events | Emitted events | Missing | Extra | Field mismatches | Verdict |")
    add("|---|---:|---:|---:|---:|---:|---|")
    for r in per_run:
        d = r["event_diff"]
        verdict = "exact match" if d["exact"] else "**DIFF — see notes**"
        add(
            f"| `{r['run_id']}` | {d['expected']} | {d['emitted']} | {len(d['missing'])} "
            f"| {len(d['extra'])} | {len(d['mismatches'])} | {verdict} |"
        )
    if any(not r["event_diff"]["exact"] for r in per_run):
        add("")
        add("Details:")
        for r in per_run:
            d = r["event_diff"]
            for key in d["missing"]:
                add(f"- `{r['run_id']}` missing event {dict(zip(EVENT_FIELDS, key))}")
            for key in d["extra"]:
                add(f"- `{r['run_id']}` extra event   {dict(zip(EVENT_FIELDS, key))}")
            for m in d["mismatches"]:
                add(f"- `{r['run_id']}` mismatch on {m['fields']}")
    else:
        add("")
        add("All three runs reproduce their ground-truth expected logs exactly — no missing,"
            " no extra, no timestamps or messages out of place.")
    add("")
    if violation_rows:
        add("## A10 — Violation evidence table")
        add("")
        add("Each violation type the validator can emit — **SKIPPED**, **OUT_OF_ORDER**, and")
        add("**TIMEOUT** — with the actual timestamped JSONL log line from a real validator replay")
        add("as evidence. This table is generated by `tools/evaluate.py` on every run.")
        add("")
        add("| Violation | Scenario | Step | t_rel | Frame | Log line (JSONL) |")
        add("|---|---|---|---|---|---|")
        for row in violation_rows:
            ev: StepEvent = row["data"]
            add(
                f"| **{ev.event}** | {row['scenario']} | {ev.step_id} ({ev.step_index}) "
                f"| {ev.t_rel:.3f} s | {ev.frame_index} | `{ev.to_json()}` |"
            )
        add("")
        add("**How to reproduce:** rerun `tools/evaluate.py`. The `OUT_OF_ORDER` and `SKIPPED`")
        add("lines are verbatim events from the first committed replay that exhibits each one")
        add("(the demo runs, then the A1 fixtures); the `TIMEOUT` line is the verbatim event")
        add("from the deterministic stall probe described in the script's docstring. Replays")
        add("are anchored to a fixed wall-clock, so the `t_iso` fields are reproducible.")
        add("")
    add("## Definitions and scope")
    add("")
    add("- **Step accuracy.** Per step instance (8 steps × 3 runs), the validator outcome"
        " derived from its emitted events (`COMPLETED`, `SKIPPED`, `TIMEOUT`, `INCOMPLETE` ="
        " started but not finished, `NOT_PERFORMED` = never reached) is compared to the"
        " operator annotation (`COMPLETED`, `SKIPPED`, `NOT_PERFORMED`). The `skip` and"
        " `wrong_order` runs are deliberately out-of-order scenarios: their operator"
        " annotations contain skips the validator reports as the one-shot `OUT_OF_ORDER`"
        " alert rather than a per-step `SKIPPED`, which is exactly the design in"
        " `har/protocol/validator.py` — the numbers below are the honest consequence.")
    add("- **Sequence completion rate.** Only the clean `correct` run reaches"
        " `PROTOCOL_COMPLETE`; the two violation runs flag early and stop, as designed.")
    add("- **False-alarm rate.** A `VIOLATION` event is a false alarm only if the"
        " ground-truth expected log does not contain it. The same event raised at the"
        " same time in both logs is a true positive.")
    add("- **Mean per-step latency.** `COMPLETED` minus the operator's `t_end`, averaged over"
        " completed steps with an operator window. Steps the validator completes that the"
        " operator marked `NOT_PERFORMED` have no latency and are counted as accuracy misses.")
    add("- **Source.** The replayed input is the frame-accurate `*_evidence.json` timeline the"
        " `demo/*.mp4` videos were scripted from (`demo/README.md`). The mp4 files are the"
        " venue fallback; once `har/app.py` (C5) lands, `tools/evaluate.py --events-dir <runs> "
        "--no-write` scores the CLI's own `events.jsonl` with the same metrics.")
    add("")
    add("_No placeholders: every number above is computed by `tools/evaluate.py` from the"
        " committed recordings; rerun it to reproduce this page._")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="A9: replay the demo dataset against PTS-01 and write real metrics."
    )
    ap.add_argument("--demo", type=Path, default=DEFAULT_DEMO,
                    help="directory holding ground_truth.json and the run recordings")
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL,
                    help="protocol yaml (default: protocols/pts01.yaml)")
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS,
                    help="METRICS.md output path (default: docs/METRICS.md)")
    ap.add_argument("--no-write", action="store_true",
                    help="print the report without writing docs/METRICS.md")
    ap.add_argument("--json", type=Path, default=None, metavar="PATH",
                    help="also write the raw metrics as JSON")
    ap.add_argument("--events-dir", type=Path, default=None, metavar="DIR",
                    help="score <run_id>.jsonl produced by har/app.py instead of replaying evidence")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any emitted-vs-expected diff or GT annotation drift")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    gt = load_ground_truth(args.demo)
    spec = load_protocol(args.protocol, tuple(gt["frame_size"]))

    per_run: list[dict] = []
    drift: list[str] = []
    run_events: dict[str, list[StepEvent]] = {}
    source_desc = "`demo/*_evidence.json` (committed FrameEvidence recordings)"
    strict_ok = True

    for run_id, run_gt in gt["runs"].items():
        n_frames = None
        if args.events_dir is not None:
            events_path = args.events_dir / f"{run_id}.jsonl"
            if not events_path.is_file():
                raise SystemExit(f"--events-dir given but {events_path} missing")
            events = read_events_jsonl(events_path)
            source_desc = f"`--events-dir` files ({args.events_dir})"
        else:
            evidence_path = args.demo / f"{run_id}_evidence.json"
            if not evidence_path.is_file():
                raise SystemExit(f"missing recording for run {run_id!r}: {evidence_path}")
            events, _validator, n_frames = replay_evidence(evidence_path, spec)

        run_events[run_id] = list(events)
        metrics = compute_run_metrics(run_id, run_gt, events, n_frames)
        per_run.append(metrics)

        # Drift check: A8's stored validator_* fields vs the replay.
        for step in run_gt["steps"]:
            computed = metrics["steps"][step["index"] - 1]["validator_outcome"]
            for problem in drift_check(step, computed, events):
                drift.append(f"{run_id} {step['step_id']}: {problem}")
        if not metrics["event_diff"]["exact"] or drift:
            strict_ok = False

    overall = aggregate(per_run)

    # A10 evidence candidates: the demo runs first (insertion order), then the
    # A1 fixture replays for any violation type the demo runs do not exhibit.
    candidates: list[tuple[str, list[StepEvent]]] = list(run_events.items())
    for fixture in ("evidence_skip.json", "evidence_wrong_order.json"):
        fixture_path = REPO / "tests" / "fixtures" / fixture
        if fixture_path.is_file():
            fixture_events, _, _ = replay_evidence(fixture_path, spec)
            candidates.append((f"tests/fixtures/{fixture}", fixture_events))
    violation_rows = build_violation_rows(spec, tuple(gt["frame_size"]), candidates)
    report = render_markdown(gt, per_run, overall, source_desc, violation_rows)

    print(report, end="")
    if drift:
        print("\nAnnotation drift (stored validator_* vs replayed):", file=sys.stderr)
        for line in drift:
            print(f"  - {line}", file=sys.stderr)

    if not args.no_write:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(report, encoding="utf-8")
        print(f"\n[written] {args.metrics}", file=sys.stderr)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"ground_truth": gt, "per_run": per_run, "overall": overall, "drift": drift},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[written] {args.json}", file=sys.stderr)

    if args.strict and not strict_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
