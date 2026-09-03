# SIH26174 — Fix report: step-7 alert spoken while stuck on step 1 / 2

## Symptom

The run is on step 1 (`PRESENT_TRAY`) or step 2 (`OPEN_TRAY`) but the GUI banner and
the voice say **"Hands are still inside the work envelope."** — the `voice_alert` of
step 7 (`STOW_AND_CLOSE`), not a step 1/2 message.  With no hands in frame at all it
fired on the very first frame of the run.

## Root cause (one paragraph)

`SequenceValidator` (`har/protocol/validator.py`) evaluates every *later* step on every
frame for out-of-order / skip detection.  Step 7's predicate `hands_clear(rack_roi)`
(`har/protocol/predicates.py`) was `all(not in_zone(w) for w in ev.hands)`, and Python's
`all()` over an empty iterable is `True`.  So whenever no wrist was detected — start of
the run, hands out of frame, a YOLO-pose miss — `ev.hands == []`, `hands_clear` was
vacuously true, the validator treated step 7 as satisfied while still on step 1/2,
emitted `OUT_OF_ORDER` carrying step 7's `voice_alert`, and after step 7's
`hold_frames=20` could `SKIP` the current early step and jump the cursor forward.  The
yellow tray lid is unrelated: live hands come only from YOLO-pose wrists
(`har/perception/pose.py`), never from HSV colour.

## Fix

### A. `har/protocol/predicates.py` — `hands_clear` is never vacuously true

* New `PredicateState.hands_seen_in_zone: bool = False`.  The validator already owns
  one `PredicateState` per step and replaces it on `reset()`, so the latch is per-step,
  per-run and is cleared by the manual Restart / Reset path for free.
* `hands_clear` now:
  1. if any wrist is inside the zone → set `hands_seen_in_zone = True`, return `False`;
  2. else return `hands_seen_in_zone` — `False` until hands have actually been seen in
     the envelope this run, `True` once they were in and are now out (whether detected
     outside the zone or not detected at all).  `hold_frames` still supplies persistence.

### B. `protocols/pts01.yaml` — step 7 `voice_alert`

`voice_alert` is only ever spoken by the validator when the step is satisfied *early*
(`OUT_OF_ORDER`), so the text now describes that:

```yaml
voice_alert: "Out of sequence: work envelope was cleared before earlier steps finished."
```

While step 7 is *current* and hands are still inside, the predicate is simply
unsatisfied; the step's `voice_prompt` keeps instructing and the unchanged 60 s
`timeout_s` produces its own `Step 7 timed out after 60s` alert.

### C. Unchanged

Hand detection, HOI / `hoi_cycle` for `EXTRACT_RED` / `EXTRACT_BLUE`, the 60 s step
timeout, the one-shot out-of-order episode, and the GUI Restart / Reset path.

## Tests

* `tests/test_predicates.py` — `hands_clear` unit tests: empty hands → `False`;
  hand visible only outside the zone → `False`; wrist inside → `False` and latch set;
  in-then-out → `True` (for both "detected outside" and "not detected"); fresh
  `PredicateState` does not inherit the latch; unknown zone → `False`.
* `tests/test_validator.py::ValidatorEmptyHandsTests` — against the real
  `protocols/pts01.yaml`: 90 empty frames on step 1 produce only `STARTED`; sitting on
  step 2 with no hands emits no step-7 `OUT_OF_ORDER` / `SKIPPED`; a *genuine* early
  clearance (hands in, then out, tray never presented) still raises exactly one
  `OUT_OF_ORDER` with the new message; `reset()` clears the latch; the normal step-7
  completion path (hands in, then out for `hold_frames`) completes the protocol with
  zero violations; and if hands were never seen in the envelope step 7 waits and then
  `TIMEOUT`s at 60 s instead of auto-completing.

```bash
python -m pytest tests/test_predicates.py tests/test_validator.py -q
```

## Manual verification

1. Start the app with nobody at the rack (no hands in frame).  The checklist must stay
   on **step 1**, the banner must stay empty, and nothing about the work envelope is
   spoken.
2. Present the tray, open it, place red then blue.  When step 7 becomes current, keep a
   hand inside the envelope: it must **not** complete.  Withdraw both hands: step 7
   completes after ~1.3 s (20 frames at 15 fps) and `PROTOCOL_COMPLETE` fires.
3. Press **Restart / Reset** and repeat step 1 of this list: the latch is cleared, so
   the fresh run again stays on step 1 with no alert.

## Note on `tests/test_demo_dataset.py`

Two tests in `DemoReplayMatchesGroundTruthTests` fail on `main` **before** this change
and still fail after it, for an unrelated reason: the committed `demo/*_evidence.json`
recordings place the red box at x≈128 (camera-left) and the blue box at x≈512
(camera-right), while the current `pts01.yaml` defines `zone_red` on the camera-right
and `zone_blue` on the camera-left.  The demo evidence / ground truth need regenerating
for the swapped pad layout; that is outside the scope of this fix.
