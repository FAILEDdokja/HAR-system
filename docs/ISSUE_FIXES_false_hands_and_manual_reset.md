# SIH26174 — Fix report: false hand detections + manual restart

Scope covered here:

1. **Issue 1** — hands reported when no hand is visible (false positives).
2. **Issue 2** — no way to restart the experiment sequence without restarting
   the whole application.

The files changed and the exact behaviour are summarised per issue. All diffs are
on top of commit `2f1cfc6` on the working branch. The existing 60 s automatic
step/inactivity timeout is **unchanged**.

---

## 0. How the system actually detects "hands" (facts from the code)

* Hands = **wrists**. They come from whole-body pose, not a dedicated hand model.
* Live camera (`--wrists auto` → `pose`): `har/perception/pose.py` → `WristExtractor`
  runs `models/yolo11n-pose.pt` (ultralytics). Per frame it calls
  `har/perception/adapters.py::wrists_from_pose_result`, which keeps a wrist if its
  COCO keypoint **9 (left) / 10 (right)** passes `min_confidence`.
* The same pose pass also yields `person_count`, which feeds the **person gate** in
  `har/perception/perception.py` (nobody in frame ⇒ object detector skipped).
* A single pose model emits a *person* box and all 17 keypoints together. When the
  operator is in frame but a hand is tucked away, behind the body, or occluded, the
  network **still predicts wrist keypoints** for some frames with a plausible
  confidence. Those wrists then flow straight to the interaction FSMs and the
  `hands_clear` predicate as if a real hand were present.
* Rendered/synthetic footage (`--wrists auto` → `hsv`) uses `HsvHandTracker`
  (orange-ring blob in `har/app.py`) — a colour stand-in, only used for demo videos.

---

## 1. Issue 1 — False hand detections

### 1.1 Root cause

In the pose path, three things make a phantom hand "real":

1. **Per-keypoint confidence floor far too low.** `wrists_from_pose_result` is called
   with `min_confidence=keypoint_confidence`, and `WristExtractor` defaulted to
   `keypoint_confidence = 0.2`. A hallucinated wrist keypoint with confidence
   `0.2–0.5` was accepted.
2. **No temporal confirmation.** Every pose pass that produced a keypoint reported
   it immediately. A **single-frame** hallucination (intermittent occlusion, motion
   blur, background) instantly became a reported hand. There was no "must persist for
   N frames" requirement.
3. **Person-index instability / no gating on wrist reuse.** Because wrists were read
   per person with no cross-frame association, a flickering per-frame read was never
   smoothed out.

Net effect: with the operator in frame but hands out of view, short-lived wrist
keypoints passed a 0.2 confidence check and were reported as hands.

The HSV path (`HsvHandTracker`) is a secondary risk when an *orange/skin-coloured*
object appears (its HSV gate is a single blob, min area 60). It is only exercised on
rendered footage, so we did not alter it here, but its `min_area` / ROI tuning is the
right lever if that backend ever runs on a live feed with an orange prop.

### 1.2 Fixes (code)

**A. Higher, exposed per-keypoint (visibility) confidence — `har/app.py` + `pose.py`.**

New CLI knob `--wrist-kp-conf` (default **0.5**, was effectively fixed at **0.2**),
threaded through `WristExtractor(keypoint_confidence=...)`.

**B. Require N consecutive frames before a hand is accepted — new `WristDebouncer`.**
`har/perception/pose.py` now has a `WristDebouncer` that wraps every pose pass. A wrist
is only exposed once it is seen on `confirm_frames` consecutive pose frames, and it is
held for `forget_frames` consecutive missing frames so a one-frame dropout of a real
hand does not read as "hands vanished" (which the B3 note says corrupts pickup
detection). It keys on wrist `side`, taking the highest-confidence candidate per side
per frame, and resets with the extractor.

CLI knobs: `--wrist-confirm` (default **3**) and `--wrist-forget` (default **5**).

### 1.3 New configuration parameters (exposed)

| Parameter | Default | Meaning |
|---|---|---|
| `--wrist-kp-conf F` | `0.5` | min wrist keypoint/visibility score to consider a hand. Raise to kill occluded/hallucinated wrists. |
| `--wrist-confirm N` | `3` | report a hand only after N **consecutive pose frames**. Raises = fewer false hands. |
| `--wrist-forget N` | `5` | hold a confirmed hand N missing pose frames before dropping (guards real hands against 1-frame dropouts). |

### 1.4 Testing for Issue 1
```bash
# pose path with a camera:
python -m har.app --source 0 --wrists pose --wrist-kp-conf 0.6 --wrist-confirm 5
# stand with hands behind your back / out of frame: no "hand" HUD rings, no
# hoi / hands_clear activity. Then reach a hand in: it appears ~0.2-0.3 s after
# it actually arrives (confirm delay) and stays put if you wiggle in/out quickly.
python -m unittest tests.test_perception_pose.WristDebounceTests -v
```

---

## 2. Issue 2 — Missing manual restart / reset

### 2.1 What exists today (facts from the code)

* The sequence state machine is `har/protocol/validator.py::SequenceValidator`
  (`update/status/reset`), with per-step `timeout_s` (60 s on most steps). On timeout it
  emits `TIMEOUT` once and **keeps the step current**; it does **not** auto-advance or
  auto-reset the whole run.
* `reset()` exists on `SequenceValidator` and on `PerceptionStack`, but today it is only
  called from the frame loop for `--loop` file replay (`har/app.py`). There was **no**
  GUI/API path to invoke it, so after a timeout the only way back to step 1 was restarting
  the app.

### 2.2 The requirement + design

A **manual "Restart / Reset" button** in the GUI must:
* reset the sequence to step 1/idle,
* clear completed / skipped / violations / last-alert / per-step runtimes,
* restart timers (including the fresh step-1 `entered_at`, i.e. the 60 s window),
* **not** restart the app, camera, models, or the GUI,
* log the manual reset event with a wall-clock timestamp.

Because the GUI lives in a Flask **daemon thread** while the validator/perception live in
the **main frame loop**, resetting them from the GUI thread would be a data race. So the
button only sets a thread-safe flag; the frame loop performs the real reset **between
frames** and logs it.

### 2.3 Backend changes

* `har/ui/web.py` — additive `bind_reset(handler)` hook (mirrors the existing
  `bind_protocol` pattern, so the frozen `create_app` signature is untouched) plus a
  new `POST /reset` route that invokes the bound handler.
* `har/app.py`:
  * `request_manual_reset()` (GUI thread) — sets a `threading.Event`.
  * `_manual_reset(reason)` (main loop thread) — `validator.reset()`, `perception.reset()`
    (this clears trackers, interaction FSMs, wrist cache/debounce), re-anchors the elapsed
    clock `t0` (so the new step-1 60 s timeout restarts), and logs a
    `MANUAL_RESET` / `INFO` `StepEvent` with `_now_iso()`.
  * The frame loop drains the flag each iteration and calls `_manual_reset`.
  * `_start_web_server(..., reset_handler=request_manual_reset)` wires the hook.
* The auto 60 s timeout path is **not touched** — `validator.update`'s timeout logic is
  unchanged; a reset simply re-arms it from step 1.

### 2.4 GUI changes (`har/ui/index.html`)

* A clearly visible **"⟳ Restart / Reset"** button in the header (red, always enabled,
  with a tooltip explaining it does not restart the app).
* JS `doReset()` does `POST /reset`, briefly shows "Restarting…", disables itself during
  the call, and refreshes the event-log tail. The next `/status` poll shows the run back
  at step 1.

### 2.5 Testing for Issue 2
```bash
# run the GUI + stream (camera, or replay a file with --loop if you want repeatable steps):
python -m har.app --source 0
# open the GUI at http://localhost:8080, let a step time out (or mid-run),
# click "⟳ Restart / Reset":
#   - checklist returns to step 1,
#   - red violation banner clears,
#   - a "MANUAL_RESET ... sequence reset to step 1" line appears in the event log,
#   - events.jsonl (append-only) gains a MANUAL_RESET row with an ISO timestamp,
#   - camera/stream/GUI keep running (no app restart).

# unit coverage added:
python -m unittest tests.test_ui_web -v
```

### 2.6 Full test command
```bash
python -m unittest discover -s tests
```
(208 tests; heavy-dependent ones skip when torch/ultralytics/opencv/PyYAML/flask are not
installed — the two new suites added above run in a bare interpreter.)

---

## 3. Files changed

| File | Change |
|---|---|
| `har/perception/pose.py` | new `WristDebouncer`; temporal confirmation wired into `WristExtractor` (`confirm_frames`/`forget_frames`), reset clears it |
| `har/app.py` | new CLI knobs `--wrist-kp-conf`, `--wrist-confirm`, `--wrist-forget`; threaded into pose path; reset request event + `_manual_reset` in frame loop; `reset_handler` passed to web server |
| `har/ui/web.py` | `bind_reset()` hook + `POST /reset` route |
| `har/ui/index.html` | Restart/Reset button + `doReset()` JS |
| `tests/test_perception_pose.py` | `WristDebounceTests` |
| `tests/test_ui_web.py` | reset-hook tests |
