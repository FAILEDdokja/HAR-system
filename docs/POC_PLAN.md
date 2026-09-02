# SIH26174 — Alignment Assessment & Proof-of-Life Plan

Status: **plan only — no implementation started.**
Repo state assessed at commit `19c5436` on branch `arena/01a062b8-har-system`.

---

## 1. What the repo actually is (verified)

16 files, one commit ("Initial commit"), 5,799 lines of Python + 2 committed
model weights (`yolo11n.pt` 5.6 MB, `yolo11n-pose.pt` 6.3 MB).

| Layer | Contents | Verdict |
|---|---|---|
| Tested library | `bottle_monitor.py` (606 L) + `test_bottle_monitor.py` (55 L) | The only real asset |
| Legacy camera scripts | `detect.py`, `track.py`, `pose.py`, `interaction.py`, `persistent_tracking.py`, `pickup_detection.py`, `pickup_detection_v2.py`, `pickup_detectionv_4.py`, `pickup_tracking_v3.py` | 5,127 L = **88.6 %** of all Python, dead ends |
| Config | `custom_bytetrack.yaml`, `pickup_detection_v3.py` (11-line shim) | keep |

The legacy files are version control by copy-paste — their own window titles say
`"SIH26174 - Step 6B"`, `"Step 6D"`, `"Step 6E"` (those are *development* steps, not
experiment steps). 8 of the 9 scripts execute `cv2.VideoCapture(0)` at **module import
time**, so they cannot be imported, tested, or run against a file.

`bottle_monitor.py` is genuinely good and is the only import-safe module (cv2/ultralytics
are imported lazily inside `run_webcam`/`_draw_debug`). It contains:

- geometry helpers: `center`, `box_area`, `box_iou`, `point_to_box_distance`
- `SingleBottleTracker` — single-target association that tolerates ByteTrack ID churn,
  with tentative acquisition, velocity extrapolation on misses, growing reacquire radius,
  optional ROI gating
- `InteractionMachine` — hand-object-interaction FSM
  `IDLE → NEAR_OBJECT → PICKED_UP → CARRYING → RELEASED → IDLE`, driven by wrist-to-box
  distance plus object/hand/relative movement
- `detections_from_yolo_result` / `wrists_from_pose_result` — duck-typed adapters
  (`getattr`), which is exactly why the unit tests run with **no cv2/torch installed**

**Checks I ran:**
- `python3 -m unittest -v test_bottle_monitor` → `Ran 3 tests ... OK`
- `python3 -m py_compile *.py` → all 12 files compile
- Unpickle of `yolo11n.pt` → saved by **ultralytics 8.2.100**; `names` dict is exactly
  **COCO-80**, `class 39 = bottle`, `class 0 = person`
- Sandbox: 2 CPU cores, 3 GB RAM, **no GPU**, 20 GB free disk, PyPI reachable

---

## 2. Alignment scorecard

Against the 7 bullets of the PS "Expected Solution / Deliverables":

| # | Requirement | Repo | Evidence |
|---|---|---|---|
| a | Continuously process local video | **Yes** | `cv2.VideoCapture(0)` ×8; `run_webcam` @ `bottle_monitor.py:479` |
| b | Suggest next step at start / after each step | **No** | no step or protocol model anywhere |
| c | Voice alert on skipped / out-of-sequence step | **No** | grep `pyttsx3\|piper\|coqui\|espeak` → 0 files |
| d | Timestamped structured lightweight text log | **No** | grep `json\|csv\|logging\|open(` → 0 files |
| e | Stream video to a specific IP **and** store locally | **No** | grep `VideoWriter\|ffmpeg\|rtsp\|mp4` → 0 files |
| f | GUI for monitoring | **No** | grep `tkinter\|PyQt\|flask\|gradio\|streamlit` → 0 files (only `cv2.imshow` debug overlay) |
| g | Trained AI model, offline standalone | **No** | only stock COCO weights; no `data.yaml`, no train script |

Against the required capability set:

| Capability | Repo | Note |
|---|---|---|
| Object detection | **Partial** | `classes=[39]` (bottle) in 7 files |
| Pose estimation | **Yes** | `yolo11n-pose.pt`, wrist kp indices 9/10 |
| Hand-object interaction | **Yes — strong** | `SingleBottleTracker` + `InteractionMachine`, 3 passing tests |
| Sequence / protocol validation | **No** | the core of the PS is entirely absent |
| Own synthetic/custom dataset | **No** | no images, no labels, no `data.yaml` |
| Orientation-agnostic 3D HMR (optional) | **No** | only a `TrackerConfig.roi` field (`bottle_monitor.py:80`) |

**Bottom line: roughly 1 of 7 deliverables (14 %) exists.** The repo has a solid
*perception primitive* layer and **zero** *system* layer. It is a strong starting point
for the hand-object-interaction detector and a weak starting point for everything ISRO
actually listed as expected output.

### The one hard blocker nobody has hit yet

The PS experiment is about **boxes** — an outer box containing a red box and a blue box.
I unpickled `yolo11n.pt` and dumped its `names` dict: it is COCO-80, and there is **no**
`box`, `crate`, `container`, `tray`, `bin`, `package` or `cube` class. Every script in the
repo works around this by detecting a **bottle** (`classes=[39]`). So as written the system
cannot see a single object in the PS protocol. This must be resolved on day one — it is the
first real design decision, not a detail.

---

## 3. Target architecture for the PoL

Keep `bottle_monitor.py`'s math; everything new is a thin layer on top.

```
har/
  perception/
    detector.py     ObjectDetector protocol; YoloDetector + ColorBoxDetector (HSV)
    pose.py         wrist extractor  (reuse wrists_from_pose_result)
    tracker.py      SingleTargetTracker(label)  <- generalised SingleBottleTracker
                    TrackerRegistry: one tracker per protocol object
  protocol/
    protocol.yaml   the 8 steps, machine-readable
    steps.py        StepSpec + predicates over (tracker state, HOI state, zone)
    validator.py    SequenceValidator: advance | skip | out-of-order | timeout
  out/
    eventlog.py     JSONL + CSV, ISO-8601 timestamps
    recorder.py     cv2.VideoWriter -> recordings/run_<ts>.mp4
    streamer.py     MJPEG over HTTP on a configurable host:port
    speaker.py      pyttsx3 on a background thread (never blocks frames)
  ui/
    overlay.py      in-frame HUD: current step, next step, alerts, FPS
    web.py          one page: /stream (video) + /status (JSON poll) + /events
  app.py            single entrypoint wiring all of it
tools/
  make_synthetic_video.py   renders scripted correct / error runs -> headless CI
  capture_dataset.py        webcam grab + YOLO auto-label for the fine-tune
legacy/             the 9 old scripts, git-mv'd (not deleted, so history reads clean)
```

Key decisions:

1. **One `ObjectDetector` interface, two implementations.** `YoloDetector` (stock or
   fine-tuned weights) and `ColorBoxDetector` (HSV segmentation + contours inside the
   rack ROI). This makes the "did you train a model?" question a config flag, so the
   fine-tune is a bonus rather than a blocker.
2. **The sequence validator is pure logic**, unit-testable with no cv2/torch — same
   pattern that already makes `test_bottle_monitor.py` pass in this sandbox.
3. **Browser GUI and "stream to a specific IP" are the same component** — one MJPEG
   endpoint serves the video, one page polls `/status`. Two PS bullets for ~120 lines.
4. **`--source 0|path.mp4`** everywhere. Demoing from a recorded file is the single
   cheapest de-risking move available.

---

## 4. Minimum change set

### Tier 1 — required for a proof of life (≈ 800 new LOC, ~30 edited)

| ID | Change | Reuses | Est. |
|---|---|---|---|
| M1 | Generalise `SingleBottleTracker` → `SingleTargetTracker(label=...)` + `TrackerRegistry` (one per protocol object). Keep `bottle_monitor` importable as a shim so the 3 tests keep passing. | all existing association math | 100 L / 1 h |
| M2 | `protocol.yaml` + `StepSpec` + **`SequenceValidator`**. Advance on predicate; emit `SKIPPED` when a later step's predicate holds while the current one does not; emit `OUT_OF_ORDER` when a non-current predicate fires. | nothing (all new) | 220 L / 2 h |
| M3 | `eventlog.py` — append-only JSONL + mirrored CSV; one row per step event with `t_iso`, `t_rel`, `step_id`, `event`, `status`, `confidence`. | — | 60 L / 0.5 h |
| M4 | `speaker.py` — pyttsx3 in a daemon thread with a 1-deep queue (drop, never block the frame loop). | — | 50 L / 0.5 h |
| M5 | `recorder.py` (mp4v VideoWriter) + `streamer.py` (MJPEG HTTP). | — | 120 L / 1 h |
| M6 | `web.py` + `index.html` — `<img src="/stream">`, step checklist, alert banner, event tail. | reuses M5's stream | 120 L / 1 h |
| M7 | `app.py` — the only entrypoint. `--source`, `--protocol`, `--headless`, `--record`, `--stream-host`, `--stream-port`, `--no-voice`, `--pose-every-n`. Kills the import-time `VideoCapture(0)`. | `run_webcam` loop shape | 150 L / 1 h |
| M8 | `ColorBoxDetector` (HSV red/blue + contours inside rack ROI, median-of-N smoothing) behind the M1 interface. | `point_to_box_distance`, ROI | 100 L / 1 h |
| M9 | Housekeeping: `README.md`, pinned `requirements.txt` (ultralytics **8.2.100** to match the committed weights), `.gitignore` += `runs/ logs/ recordings/ *.onnx`, `git mv` the 9 legacy scripts to `legacy/`. | — | 0.5 h |

Tier 1 total: ~8 h of focused work. That is a demoable, end-to-end PoL touching every
one of the 7 PS bullets.

### Tier 2 — differentiators, only if Tier 1 lands early

| ID | Change | Why it wins points | Est. |
|---|---|---|---|
| D1 | **Rack-relative frame**: 4 ArUco/fiducial corners on the payload rack → `cv2.getPerspectiveTransform` → all box/hand/zone coords expressed in rack space. | The honest, cheap version of "orientation-agnostic". Demo trick: rotate the whole setup 90° mid-demo and the sequence still validates. No 3D HMR needed. | 80 L / 1.5 h |
| D2 | **Fine-tune** `yolo11n` on ~150 webcam images of 3 classes (tray / red box / blue box) + `data.yaml` + `train.py` + a metrics table. | Turns bullet (g) "a trained AI model" and "own dataset" from *missing* to *done*. Needs a GPU — not available in this sandbox. | 3 h (mostly waiting) |
| D3 | **Synthetic video harness**: OpenCV renders scripted runs → `app.py --source synth.mp4 --headless` → assert on the JSONL. | Gives real numbers (step accuracy, false alarms, FPS) instead of "it looked right". Runs headless on CPU here. | 120 L / 1 h |
| D4 | `model.export(format="onnx")` + an FPS/latency table CPU vs ONNX. | The edge/offline story, evidenced. | 1 h |

### Tier 3 — cut. Mention in slides as roadmap only.

3D HMR (CLIFF/PyMAF) — will not run on 2 CPU cores, let alone overnight.
Temporal Transformer/LSTM step classifier — a rule-based FSM is *more* defensible for
flight software (explainable, auditable); frame it as the roadmap item.
RTSP — MJPEG already satisfies "stream to a specific IP".
Multi-astronaut, multi-camera fusion.

---

## 5. Protocol to lock (unblocks everything else)

**PTS-01 — Payload Tray Sorting & Sample Transfer.** 8 steps. Props: outer tray box with
lid, red inner box, blue inner box, sample vial, two marked placement zones (A, B) on the
rack panel.

| # | Step ID | Completion predicate (all over existing signals) |
|---|---|---|
| 1 | `PRESENT_TRAY` | tray detected in rack ROI, object movement < θ for N frames |
| 2 | `OPEN_TRAY` | HOI on lid reaches `PICKED_UP`, lid box exits tray ROI |
| 3 | `EXTRACT_RED` | red box HOI `NEAR → PICKED_UP → CARRYING → RELEASED` with release inside zone A |
| 4 | `VERIFY_RED_PLACED` | red box centre inside zone A and stationary for T s |
| 5 | `EXTRACT_BLUE` | as 3, blue → zone B |
| 6 | `VERIFY_BLUE_PLACED` | as 4, blue in zone B |
| 7 | `SAMPLE_TRANSFER` | vial HOI pickup from red box → release inside rack slot |
| 8 | `STOW_AND_CLOSE` | lid returns to tray ROI and both wrists clear the work zone |

Every predicate is a function of `{tracker.box, tracker.measured, InteractionState, zone}`
— all of which `bottle_monitor.py` already computes. The new code is genuinely just the
sequencing layer.

**Three error cases to demo:** (1) jump 2 → 5 ⇒ "Step 3 skipped"; (2) do blue before red
⇒ "Out of sequence — expected EXTRACT_RED"; (3) stall on step 4 ⇒ timeout alert.

---

## 6. Sequenced plan for tomorrow

| Slot | Work | Gate |
|---|---|---|
| 0:00–0:30 | Freeze PTS-01, write `protocol.yaml` | file reviewed by whole team |
| 0:30–1:30 | M9 housekeeping + M1 tracker generalisation | 3 old tests still pass |
| 1:30–3:30 | **M2 SequenceValidator + unit tests** | green run / skip / out-of-order tests |
| 3:30–4:30 | M3 eventlog + M4 voice | JSONL rows appear, audio fires off-thread |
| 4:30–6:00 | M5 recorder+streamer, M6 web GUI | page shows live video + step list |
| 6:00–7:00 | M7 `app.py`, run end-to-end on a recorded `.mp4` | one full clean run logged |
| 7:00–8:00 | Record the 3 demo videos, grab log excerpts + screenshots | assets in `demo/` |
| 8:00–9:00 | README, slides, metrics table, rehearsal | — |
| parallel | D2 fine-tune (needs a 2nd person **and** a GPU) | weights or fall back to M8 |

Hard stop rule: if M2 is not green by 3:30, drop M6 to the in-frame HUD only
(`overlay.py`) and ship without the browser GUI.

---

## 7. Risk register

| Risk | Likelihood | Fallback |
|---|---|---|
| No GPU → 2 models/frame too slow on the demo laptop | High | `imgsz=480`, `conf=0.45`, pose every 2nd frame, or drop YOLO-detect entirely and use `ColorBoxDetector` (halves cost) |
| Webcam fails on demo machine | Medium | `--source demo.mp4` (M7) |
| HSV red/blue flaky under venue lighting | Medium | rack ROI + median-of-N + manual white-balance key at start; last resort = COCO proxy props |
| pyttsx3/espeak missing on demo OS | Medium | `--no-voice` + on-screen banner + a generated beep |
| Fine-tune doesn't converge overnight | Medium | M8 colour detector is already behind the same interface |
| Scope creep into 3D HMR | High | Tier 3 is cut; D1 gets the same talking point in 80 lines |

---

## 8. Verification plan (what "it works" must mean)

1. `python -m unittest discover` — existing 3 tests **plus** new `SequenceValidator`,
   `eventlog` and detector-filter tests, all dependency-free.
2. `tools/make_synthetic_video.py` renders scripted frames →
   `python -m har.app --source synth_correct.mp4 --headless` →
   assert the JSONL has 8 `COMPLETED` events in order and 0 alerts.
3. Same for `synth_skipstep.mp4` → assert exactly 1 `SKIPPED` + 1 `ALERT`.
4. Same for `synth_wrongorder.mp4` → assert 1 `OUT_OF_ORDER`.
5. Report measured FPS and per-frame latency on the actual demo hardware, CPU and ONNX.

That yields the numbers the PS asks for — step accuracy, sequence completion rate, false
alarm rate — instead of an unmeasured claim.

---

## 9. Unchecked / open items

- **Not executed:** no live camera run, no YOLO inference — this sandbox has no cv2,
  torch or ultralytics installed and no GPU. Every claim above comes from reading the
  code, running the dependency-free unit tests, and unpickling the checkpoint.
- The PS text for the sample experiment is truncated in public sources ("...two smaller
  boxes of color red and"), so PTS-01 in §5 is our reconstruction and needs a team
  sign-off before code is written against it.
- Whether a GPU is available to anyone on the team tonight decides D2 in or out.
- Demo-laptop OS decides the TTS backend (SAPI on Windows, `espeak-ng` on Linux).
