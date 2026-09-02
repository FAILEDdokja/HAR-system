"""Sequence validation: the core of the protocol-compliance logic (step A4).

This module implements the ``SequenceValidator`` interface frozen in
``docs/DEVELOPMENT_PLAN.md`` Appendix A.  It consumes ``FrameEvidence``
produced by Track B and emits ``StepEvent`` objects consumed by Track C
(log, voice, GUI).  It depends only on ``har.contracts`` and
``har.protocol.predicates`` — no cv2, no numpy, no torch, no YAML.

Semantics (plan §5, A4 — implemented exactly)
---------------------------------------------
1.  Exactly one step is *current*.  The run starts at index 1 and emits
    ``STARTED`` once on entry.
2.  The current step's predicate is evaluated every frame.  After
    ``hold_frames`` consecutive satisfied frames the step emits
    ``COMPLETED``, the cursor advances, and ``STARTED`` is emitted for
    the next step (which is then evaluated on the same frame).
3.  Every frame, all *later* steps are also evaluated.  If a later step
    ``k > current`` is satisfied while the current step is not, an
    ``OUT_OF_ORDER`` event is emitted for ``k`` (message from that
    step's ``voice_alert``).
4.  If that later step stays satisfied for its ``hold_frames``, a
    ``SKIPPED`` event (status ``VIOLATION``, message from the *current*
    step's ``voice_alert``) is emitted for the current step and the
    cursor jumps to ``k``.  This is the skip detection: the operator
    performed step k's work while step current was still pending.
5.  A step whose ``timeout_s`` elapses (measured from when it became
    current) emits ``TIMEOUT`` once and stays current.  A timed-out step
    can still complete when its work is eventually done.
6.  After the last step completes, ``PROTOCOL_COMPLETE`` is emitted and
    ``update()`` returns ``[]`` forever after.
7.  A step is never completed — nor jumped to — on a track reported with
    ``measured=False``.  The predicates already refuse to be satisfied by
    coasting (predicted) boxes; this module adds the same guard around
    completion and skip-jumps as a second line of defence.
8.  Pure: no file IO, no cv2, and no clock reads for validation
    decisions — time arrives via ``evidence.t_rel``.  Replaying the same
    evidence sequence always yields the same events (the only wall-clock
    use is a one-time anchor for the ``t_iso`` log *metadata*, see
    ``_timestamp``).

Out-of-order detection is one-shot per run
------------------------------------------
The first time a later step is satisfied while the current one is not,
the validator alerts (``OUT_OF_ORDER``) and, once the later step's hold
is met, re-baselines the cursor (``SKIPPED`` + jump).  After that first
violation episode the validator no longer emits further ``OUT_OF_ORDER``
or ``SKIPPED`` events for the remainder of the run: the operator has
already been alerted, the run is flagged, and any *remaining* work must
now be finished in order.  This keeps the event stream at one violation
episode per run, which is exactly what the A1 fixtures and gate G1
expect (e.g. ``evidence_wrong_order.json`` yields exactly one
``OUT_OF_ORDER`` even though the operator also withdraws both hands from
the envelope before the vial transfer is done — an absence, not an
action, and not worth a second alarm).

Hold counting with decimated evidence
-------------------------------------
Evidence rows are sampled observations of a continuous video, but
``frame_index`` carries the true video frame number.  A hold of N frames
therefore means "the predicate has held for N consecutive *video*
frames".  Between two observations the truth value is unknown, so a
satisfied run is credited from just after the last unsatisfied
observation:  ``span = frame_index - last_false_frame``.  If the
predicate has never been observed unsatisfied, the run is credited from
the first satisfied observation:  ``span = frame_index - first_true_frame + 1``.
This is what lets a compressed fixture (one row per action) satisfy the
full ``hold_frames`` of a 15 fps camera without changing the yaml.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from har.contracts import (
    CONTRACT_VERSION,
    FrameEvidence,
    ProtocolSpec,
    StepEvent,
    StepSpec,
    UiStatus,
)
from har.protocol.predicates import PREDICATES, PredicateState

__all__ = ["SequenceValidator"]

_PREDICATE_NAME_RE = re.compile(r"^\s*([a-z_]+)\s*\(")


class _StepRuntime:
    """Per-step mutable bookkeeping owned by the validator (not exposed)."""

    __slots__ = (
        "state",
        "first_true_frame",
        "last_false_frame",
        "entered_at",
        "timed_out",
        "ooo_active",
    )

    def __init__(self) -> None:
        self.state = PredicateState()
        self.first_true_frame: int | None = None
        self.last_false_frame: int | None = None
        self.entered_at: float | None = None
        self.timed_out = False
        self.ooo_active = False

    def reset(self) -> None:
        self.state = PredicateState()
        self.first_true_frame = None
        self.last_false_frame = None
        self.entered_at = None
        self.timed_out = False
        self.ooo_active = False

    def record(self, satisfied: bool, frame_index: int) -> int | None:
        """Record one observation; return the satisfied span if satisfied."""
        if satisfied:
            if self.first_true_frame is None:
                self.first_true_frame = frame_index
            if self.last_false_frame is not None:
                return frame_index - self.last_false_frame
            return frame_index - self.first_true_frame + 1
        self.last_false_frame = frame_index
        self.first_true_frame = None
        return None


class SequenceValidator:
    """Validate a live or replayed stream of ``FrameEvidence`` against a
    ``ProtocolSpec`` and emit ``StepEvent``s.

    Usage::

        validator = SequenceValidator(spec)          # optional: *, start_time=...
        for evidence in frames:
            for event in validator.update(evidence):
                sinks.emit(event)
        status = validator.status()                  # polled by the GUI

    ``start_time`` (optional) anchors the ``t_iso`` log timestamps.  When
    omitted the wall clock is captured once, on the first frame, and used
    for log metadata only — every validation decision depends solely on
    ``evidence.t_rel``, so replays are deterministic.
    """

    def __init__(self, spec: ProtocolSpec, *, start_time: datetime | None = None) -> None:
        if not spec.steps:
            raise ValueError("SequenceValidator requires a protocol with at least one step")
        self._spec = spec
        self._steps = spec.steps
        self._predicate_fns = [self._resolve(step) for step in self._steps]
        self._runtime = [_StepRuntime() for _ in self._steps]
        self._cursor: int | None = None
        self._completed: list[str] = []
        self._skipped: list[str] = []
        self._violations: list[str] = []
        self._last_alert = ""
        self._last_t_rel = 0.0
        self._last_fps = 0.0
        self._finished = False
        self._ooo_reported = False
        self._start_time = start_time

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(step: StepSpec):
        match = _PREDICATE_NAME_RE.match(step.predicate)
        if not match:
            raise ValueError(f"Step '{step.step_id}' has a malformed predicate: {step.predicate!r}")
        name = match.group(1)
        try:
            return PREDICATES[name]
        except KeyError as exc:
            raise ValueError(f"Step '{step.step_id}' uses unknown predicate '{name}'") from exc

    def _timestamp(self, t_rel: float) -> str:
        if self._start_time is None:
            self._start_time = datetime.now(timezone.utc)
        return (self._start_time + timedelta(seconds=t_rel)).isoformat(timespec="seconds")

    def _event(
        self,
        evidence: FrameEvidence,
        step: StepSpec,
        event: str,
        status: str,
        message: str,
    ) -> StepEvent:
        return StepEvent(
            t_iso=self._timestamp(evidence.t_rel),
            t_rel=evidence.t_rel,
            frame_index=evidence.frame_index,
            step_id=step.step_id,
            step_index=step.index,
            event=event,
            status=status,
            message=message,
            confidence=1.0,
        )

    def _enter(self, pos: int, evidence: FrameEvidence, events: list[StepEvent]) -> None:
        self._cursor = pos
        step = self._steps[pos]
        self._runtime[pos].entered_at = evidence.t_rel
        events.append(self._event(evidence, step, "STARTED", "IN_PROGRESS", f"Step {step.index} started"))

    @staticmethod
    def _measured_ok(evidence: FrameEvidence, step: StepSpec) -> bool:
        """Rule 7: never act on the step's target while it is coasting."""
        if not step.target:
            return True
        track = evidence.object(step.target)
        return track is None or track.measured

    def _step_satisfied(
        self, pos: int, evidence: FrameEvidence, cache: dict[int, tuple[bool, int | None]]
    ) -> tuple[bool, int | None]:
        if pos in cache:
            return cache[pos]
        fn = self._predicate_fns[pos]
        satisfied = bool(fn(evidence, self._spec, self._steps[pos], self._runtime[pos].state))
        span = self._runtime[pos].record(satisfied, evidence.frame_index)
        cache[pos] = (satisfied, span)
        return cache[pos]

    def _note_violation(self, step_id: str) -> None:
        if step_id not in self._violations:
            self._violations.append(step_id)

    # ------------------------------------------------------------------
    # frame loop
    # ------------------------------------------------------------------

    def update(self, evidence: FrameEvidence) -> list[StepEvent]:
        """Advance the validation by one frame of evidence.

        Returns the list of events this frame caused (may be empty).
        Returns ``[]`` forever once the protocol is complete.
        """
        if self._finished:
            return []

        events: list[StepEvent] = []
        if self._cursor is None:
            self._enter(0, evidence, events)

        # Per-frame cache so each step's predicate is evaluated at most once
        # per frame (a step can become current via a skip-jump and would
        # otherwise be evaluated twice on the jump frame).
        cache: dict[int, tuple[bool, int | None]] = {}

        # The cursor only ever moves forward: at most one pass per step,
        # plus one more frame of slack for the final bookkeeping.
        for _ in range(2 * len(self._steps) + 2):
            if self._finished:
                break
            pos = self._cursor
            step = self._steps[pos]
            rt = self._runtime[pos]

            # (5) timeout — at most once per step; the step stays current.
            if (
                not rt.timed_out
                and rt.entered_at is not None
                and evidence.t_rel > rt.entered_at + step.timeout_s
            ):
                rt.timed_out = True
                self._note_violation(step.step_id)
                message = f"Step {step.index} timed out after {step.timeout_s:g}s"
                events.append(self._event(evidence, step, "TIMEOUT", "VIOLATION", message))
                self._last_alert = message

            # (2) evaluate the current step.
            satisfied, span = self._step_satisfied(pos, evidence, cache)
            if (
                satisfied
                and span is not None
                and span >= step.hold_frames
                and self._measured_ok(evidence, step)
            ):
                events.append(self._event(evidence, step, "COMPLETED", "OK", f"Step {step.index} confirmed"))
                self._completed.append(step.step_id)
                if pos == len(self._steps) - 1:
                    # (6) protocol complete.
                    n = len(self._violations)
                    message = (
                        f"Protocol {self._spec.protocol_id} completed with no violations"
                        if n == 0
                        else f"Protocol {self._spec.protocol_id} completed with {n} violation(s)"
                    )
                    events.append(self._event(evidence, step, "PROTOCOL_COMPLETE", "OK", message))
                    self._finished = True
                    self._last_t_rel = evidence.t_rel
                    self._last_fps = evidence.fps
                    return events
                self._enter(pos + 1, evidence, events)
                continue  # cascade: evaluate the new current on this frame

            # (3)+(4) scan every later step.
            if self._scan_later(evidence, events, satisfied, cache):
                continue  # cursor was jumped; re-evaluate the new current
            break

        self._last_t_rel = evidence.t_rel
        self._last_fps = evidence.fps
        return events

    def _scan_later(
        self,
        evidence: FrameEvidence,
        events: list[StepEvent],
        current_satisfied: bool,
        cache: dict[int, tuple[bool, int | None]],
    ) -> bool:
        """Evaluate all steps after the cursor; flag out-of-order work.

        Returns ``True`` when the cursor was jumped (the caller must
        continue so the new current step is evaluated on this frame).
        """
        assert self._cursor is not None
        for pos in range(self._cursor + 1, len(self._steps)):
            step = self._steps[pos]
            rt = self._runtime[pos]
            satisfied, span = self._step_satisfied(pos, evidence, cache)
            if not satisfied:
                rt.ooo_active = False
                continue
            if current_satisfied or self._ooo_reported:
                # (3) only an *unsatisfied* current step makes this out of
                # order, and (module docstring) detection is one-shot.
                continue
            if not rt.ooo_active:
                rt.ooo_active = True
                self._ooo_reported = True
                self._note_violation(step.step_id)
                message = step.voice_alert or f"Step {step.index} out of sequence"
                events.append(self._event(evidence, step, "OUT_OF_ORDER", "VIOLATION", message))
                self._last_alert = message
            if (
                span is not None
                and span >= step.hold_frames
                and self._measured_ok(evidence, step)
            ):
                # (4) the operator did step k's work; skip the current one.
                current = self._steps[self._cursor]
                skip_message = current.voice_alert or f"Step {current.index} skipped"
                events.append(self._event(evidence, current, "SKIPPED", "VIOLATION", skip_message))
                self._note_violation(current.step_id)
                self._skipped.append(current.step_id)
                self._last_alert = skip_message
                self._enter(pos, evidence, events)
                return True
        return False

    # ------------------------------------------------------------------
    # introspection (Appendix A)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return the validator to its fresh state (a new run)."""
        for rt in self._runtime:
            rt.reset()
        self._cursor = None
        self._completed = []
        self._skipped = []
        self._violations = []
        self._last_alert = ""
        self._last_t_rel = 0.0
        self._last_fps = 0.0
        self._finished = False
        self._ooo_reported = False
        self._start_time = None

    @property
    def current(self) -> StepSpec | None:
        """The step the cursor is on, or ``None`` before start / once complete."""
        if self._cursor is None or self._finished:
            return None
        return self._steps[self._cursor]

    @property
    def completed_steps(self) -> tuple[str, ...]:
        """Step ids in completion order."""
        return tuple(self._completed)

    @property
    def violations(self) -> tuple[str, ...]:
        """Step ids involved in a violation (OUT_OF_ORDER / SKIPPED / TIMEOUT),
        in first-occurrence order."""
        return tuple(self._violations)

    @property
    def finished(self) -> bool:
        return self._finished

    def status(self) -> UiStatus:
        """Snapshot for the GUI.  Track C renders; it never computes."""
        if self._cursor is None and not self._finished:
            state = "NOT_STARTED"
            current = self._steps[0]
        elif self._finished:
            state = "COMPLETE"
            current = self._steps[-1]
        else:
            state = "IN_PROGRESS"
            assert self._cursor is not None
            current = self._steps[self._cursor]

        # ``current.index`` is 1-based, so it doubles as the 0-based
        # position of the following step.
        next_index = current.index
        next_step = self._steps[next_index] if next_index < len(self._steps) else None

        return UiStatus(
            protocol_id=self._spec.protocol_id,
            protocol_title=self._spec.title,
            current_step_id=current.step_id,
            current_step_index=current.index,
            next_step_id=next_step.step_id if next_step else "",
            next_instruction=next_step.instruction if next_step else "",
            completed=tuple(self._completed),
            skipped=tuple(self._skipped),
            violations=tuple(self._violations),
            state=state,
            t_rel=self._last_t_rel,
            fps=self._last_fps,
            last_alert=self._last_alert,
            contract_version=CONTRACT_VERSION,
        )
