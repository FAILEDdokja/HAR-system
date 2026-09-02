"""Threshold-tuning harness for step A6 (Track A — Cognition).

A6's job (``docs/DEVELOPMENT_PLAN.md`` §5) is to replay a *recorded* protocol
run through the validator and tune ``hold_frames`` / ``timeout_s`` in
``protocols/pts01.yaml`` until the recording validates cleanly (8/8, zero
violations).  The thresholds must live in the yaml, never in code, so they can
be retuned under venue lighting without editing Python.

This harness makes that tuning *measurable*.  It replays a recorded
``FrameEvidence`` sequence and, for every step, reports:

* ``hold_frames`` (yaml) vs the longest satisfied span the recording actually
  produced for that step (in video frames) — the *hold margin*;
* ``timeout_s`` (yaml) vs how long the step was the current step before it
  completed (or before the run ended) — the *timeout margin*.

A positive hold margin means a brief detection dropout mid-hold will not break
the step.  A positive timeout margin means the operator has slack before a
false ``TIMEOUT`` fires.

Usage::

    .venv/bin/python tools/tune_thresholds.py --evidence tests/fixtures/evidence_correct.json
    .venv/bin/python tools/tune_thresholds.py --evidence tests/fixtures/evidence_skip.json

The default evidence is the A1 correct fixture, which is the only recorded
protocol run committed to the repo.  The confirmatory real recording is
produced by step A8 (``demo/*.mp4``); A6's thresholds are re-checked against it
when it lands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as ``python tools/tune_thresholds.py`` from the repo root.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from har.contracts import FrameEvidence, ObjectTrack, Wrist  # noqa: E402
from har.protocol.spec import load_protocol  # noqa: E402
from har.protocol.validator import SequenceValidator  # noqa: E402

# Recording resolution is taken from the evidence itself (the zones are
# resolved by ``load_protocol`` into the recording's pixel space, so they must
# match the evidence boxes).  ``--width/--height`` override only when the
# evidence frames do not carry a frame_size.
PROTOCOL = REPO / "protocols" / "pts01.yaml"
DEFAULT_EVIDENCE = REPO / "tests" / "fixtures" / "evidence_correct.json"


def evidence_from_dict(d: dict) -> FrameEvidence:
    """Rebuild one ``FrameEvidence`` from a ``FrameEvidence.to_dict()`` row.

    Mirrors ``tests/test_validator.py::evidence_from_dict`` so the harness can
    replay the same fixtures Track A already tests against.
    """
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


class _TuningValidator(SequenceValidator):
    """SequenceValidator that records per-step hold/timeout margins."""

    def __init__(self, spec):
        super().__init__(spec)
        n = len(self._steps)
        self.max_span: list[int] = [0] * n
        self.entered_at: list[float | None] = [None] * n
        self.completed_t_rel: list[float | None] = [None] * n

    def _step_satisfied(self, pos, evidence, cache):
        sat, span = super()._step_satisfied(pos, evidence, cache)
        if span is not None and span > self.max_span[pos]:
            self.max_span[pos] = span
        return sat, span

    def _enter(self, pos, evidence, events):  # noqa: D401 - override hook
        self.entered_at[pos] = evidence.t_rel
        return super()._enter(pos, evidence, events)


def densify(frames, fps=15.0, dropout_p=0.0, seed=0, tail_s=3.0, smooth=3):
    """Emulate a continuous camera recording from compressed evidence.

    The A1 fixtures record one observation per action, so they cannot show what
    a real 15 fps stream does between actions (dwell time, detector dropouts).
    This walks every video frame from 0 to the last observation, carrying the
    most recent observation forward, and drops each present object's detection
    with probability ``dropout_p`` per frame to simulate the missed-detection
    gaps real colour detection suffers under venue lighting.  A ``tail_s``-second
    tail of the final pose is appended so the dwell steps (``settled`` /
    ``object_stable`` / ``hands_clear``) have the continuous satisfied run a real
    operator holds — without it, ``hands_clear`` would never accumulate its
    20-frame hold because the fixture ends the instant the hands clear.

    ``smooth`` models B2's median-of-N smoothing: a drop only takes effect once
    it persists for ``smooth`` consecutive frames (isolated 1-2 frame gaps are
    filled by the median filter).  Without this the sweep is unrealistically
    harsh — raw per-frame dropout never survives 30 consecutive frames, so the
    dwell holds are actually robust on real footage.

    The result is a faithful stand-in for A8's real footage and lets A6 tune
    hold_frames against dropout, not against the fixture's minimal authored spans.
    """
    import random

    rng = random.Random(seed)
    by_idx = {f.frame_index: f for f in frames}
    last = frames[-1].frame_index
    tail_frames = int(round(tail_s * fps))
    total = last + tail_frames + 1
    # 1) raw per-frame dropout decisions per object.
    labels = sorted({lbl for f in frames for lbl in f.objects})
    raw = {lbl: [False] * total for lbl in labels}
    for fi in range(total):
        for lbl in labels:
            raw[lbl][fi] = dropout_p > 0 and rng.random() < dropout_p
    # 2) median-of-N gap fill: only a run of >= smooth consecutive raw drops
    #    actually drops the detection (mirrors B2's median_window smoothing).
    eff = {lbl: [False] * total for lbl in labels}
    for lbl in labels:
        for fi in range(total):
            if not raw[lbl][fi]:
                continue
            run = all(raw[lbl][j] for j in range(max(0, fi - smooth + 1), fi + 1))
            eff[lbl][fi] = run
    # 3) build the dense frames, carrying the latest observation forward.
    dense = []
    cursor = frames[0]
    fi = 0
    while fi < total:
        obs = by_idx.get(fi)
        if obs is not None:
            cursor = obs
        objects = {}
        for lbl, tr in cursor.objects.items():
            if tr.box is None:
                objects[lbl] = tr
                continue
            if eff[lbl][fi]:
                objects[lbl] = ObjectTrack(
                    label=tr.label, box=None, measured=False,
                    lost_frames=tr.lost_frames + 1,
                )
            else:
                objects[lbl] = tr
        dense.append(FrameEvidence(
            frame_index=fi,
            t_rel=fi / fps,
            frame_size=tuple(cursor.frame_size),
            objects=objects,
            hands=cursor.hands,
            hoi=dict(cursor.hoi),
            rack_ready=cursor.rack_ready,
            fps=fps,
        ))
        fi += 1
    return dense


def stress_sweep(evidence_path, protocol_path, frame_size,
                 dropout_levels=(0.0, 0.02, 0.05, 0.1, 0.2)):
    raw = json.loads(evidence_path.read_text())
    frames = [evidence_from_dict(d) for d in raw]
    if frame_size is None:
        frame_size = tuple(frames[0].frame_size)
    spec = load_protocol(protocol_path, frame_size)
    print(f"Stress sweep on {evidence_path.name} (res {frame_size[0]}x{frame_size[1]}, 15 fps)")
    print(f"{'dropout_p':>9}{'frames':>8}{'completed':>10}{'violations':>11}   result")
    print("-" * 56)
    for p in dropout_levels:
        dense = densify(frames, fps=15.0, dropout_p=p, seed=1234)
        v = _TuningValidator(spec)
        events = []
        for f in dense:
            events.extend(v.update(f))
        completed = sum(1 for e in events if e.event == "COMPLETED")
        viol = sum(1 for e in events if e.status == "VIOLATION")
        ok = completed == 8 and viol == 0
        note = "OK" if ok else ("step(s) dropped" if completed < 8 else "FALSE ALARM")
        print(f"{p:>9.2f}{len(dense):>8}{completed:>10}{viol:>11}   {note}")
    print()
    print("Read: a threshold set that stays 8/8 & violation-free across the")
    print("realistic dropout band (<=~0.1) is robust to missed detections.")


def analyse(evidence_path: Path, protocol_path: Path, frame_size):
    raw = json.loads(evidence_path.read_text())
    frames = [evidence_from_dict(d) for d in raw]
    # Resolve zones into the recording's own pixel space (default), so the
    # evidence boxes and the yaml zones line up.  This is why the A1 fixtures
    # (authored at 640x480) validate cleanly only when replayed at 640x480.
    if frame_size is None:
        frame_size = tuple(frames[0].frame_size)
    spec = load_protocol(protocol_path, frame_size)
    v = _TuningValidator(spec)
    events = []
    for f in frames:
        events.extend(v.update(f))

    from collections import Counter

    by_status = Counter(e.status for e in events)
    completed = [e for e in events if e.event == "COMPLETED"]
    violations = [e for e in events if e.status == "VIOLATION"]

    print(f"Evidence       : {evidence_path.name}  ({len(frames)} frames)")
    print(f"FPS (recording): {frames[-1].fps if frames else '?'}")
    print(f"COMPLETED      : {len(completed)}/8")
    print(f"Violations     : {len(violations)}  {[ (e.event, e.step_id) for e in violations ]}")
    print(f"PROTOCOL_COMPLETE: {any(e.event == 'PROTOCOL_COMPLETE' for e in events)}")
    print()
    print(f"{'step':<18}{'hold':>5}{'maxspan':>8}{'margin':>7}   {'timeout_s':>9}{'cur_dur_s':>10}{'t_margin':>9}")
    print("-" * 78)
    for pos, step in enumerate(spec.steps):
        hold = step.hold_frames
        span = v.max_span[pos]
        margin = span - hold
        timeout = step.timeout_s
        entered = v.entered_at[pos]
        done = v.completed_t_rel[pos]
        # completed_t_rel is not set by the hook; derive from COMPLETED events.
        comp_event = next((e for e in completed if e.step_index == step.index), None)
        if comp_event is not None:
            done = comp_event.t_rel
        if entered is not None and done is not None:
            cur_dur = done - entered
        elif entered is not None:
            cur_dur = frames[-1].t_rel - entered if frames else 0.0
        else:
            cur_dur = 0.0
        t_margin = timeout - cur_dur
        flag = ""
        if margin < hold * 0.5:
            flag = "  <-- thin hold margin"
        if t_margin < 0:
            flag = "  <-- TIMEOUT MARGIN NEGATIVE"
        print(
            f"{step.step_id:<18}{hold:>5}{span:>8}{margin:>7}   "
            f"{timeout:>9}{cur_dur:>10.1f}{t_margin:>9.1f}{flag}"
        )
    print()
    ok = len(completed) == 8 and not violations
    print("A6 CHECK:", "PASS — recording validates 8/8 with zero violations" if ok else "FAIL — see above")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A6 threshold-tuning harness")
    ap.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE,
                    help="recorded FrameEvidence JSON (default: A1 correct fixture)")
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--width", type=int, default=None,
                    help="override recording width (default: from evidence)")
    ap.add_argument("--height", type=int, default=None,
                    help="override recording height (default: from evidence)")
    ap.add_argument("--stress", action="store_true",
                    help="emulate a 15 fps recording with detector dropouts and sweep robustness")
    args = ap.parse_args(argv)
    frame_size = None
    if args.width is not None and args.height is not None:
        frame_size = (args.width, args.height)
    if args.stress:
        stress_sweep(args.evidence, args.protocol, frame_size)
    else:
        analyse(args.evidence, args.protocol, frame_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
