# SIH26174 — Development Plan

**Branch:** `arena/01a062b8-har-system` · **Contract version:** 1.0.0 · **Protocol:** PTS-01 v1.0.0

Read §1 once, then work only from your own track section. The file-ownership table in
§2 is the entire coordination protocol — if you stay inside your column you cannot
collide with the other two people.

---

## 1. Review — what we are actually being asked to do, and what "qualify" means

### 1.1 Restating the problem in one sentence

ISRO wants an **offline** system that watches a fixed-payload camera, knows which step of
a pre-defined experiment the astronaut is on, **tells them the next step**, **speaks up
when they skip or reorder one**, and leaves behind a **timestamped text log** and a
**stored / streamed video** — with no ground in the loop.

The hard part is not the computer vision. It is **protocol compliance**: the system must
hold a model of a procedure and compare the observed world against it.

### 1.2 The seven expected deliverables, with the evidence each one needs

| # | Deliverable (verbatim intent) | What "done" looks like in our demo | Evidence artefact |
|---|---|---|---|
| D1 | Continuously process local video | Feed runs at a stable frame rate from camera **or** file, no stalls | FPS readout in the GUI |
| D2 | Suggest the next step at start / after each step | The panel always shows the current step and the next instruction in plain words | GUI screenshot |
| D3 | Voice alert on skipped / out-of-sequence step | We deliberately do step 5 before step 3; the box speaks the warning | Screen recording with audio |
| D4 | Timestamped, structured, lightweight text file | One JSONL + mirrored CSV row per step event | The actual file, opened on screen |
| D5 | Stream video to a specific IP **and** store locally | `http://<host>:8080/stream` opens on a second machine while `recordings/*.mp4` grows | Two-screen shot + `ls -la recordings/` |
| D6 | GUI for monitoring | One browser page: video + step checklist + violations + log tail | GUI screenshot |
| D7 | **A trained AI model, offline standalone** | A detector **we** trained on **our** dataset, running with no network | `weights/best.pt` + `datasets/` + metrics table |

### 1.3 Where we stand right now — verified, not assumed

This is the honest position after the restructure. Claims below were checked, not recalled.

**Present and tested (37 tests passing, `.venv/bin/python -m unittest discover` → `OK`):**

| Capability | Location | Status |
|---|---|---|
| Geometry primitives | `har/perception/geometry.py` | moved from `bottle_monitor.py`, 4 tests |
| Single-target association (survives detector ID churn) | `har/perception/tracker.py` | generalised to a labelled target + `TrackerRegistry`, 6 tests |
| Hand-object interaction FSM | `har/perception/interaction.py` | `IDLE→NEAR→PICKED_UP→CARRYING→RELEASED`, 5 tests |
| Duck-typed ultralytics adapters | `har/perception/adapters.py` | runs with no torch installed, 7 tests |
| Frozen cross-track contracts | `har/contracts.py` | stdlib-only, enforced by an `ast` test, 6 tests |
| Locked 8-step protocol | `protocols/pts01.yaml` | 11 validation tests |
| Event fixtures for Track C | `tests/fixtures/*.jsonl` | 23 events, correct + wrong-order |

**Absent — every one of these is a task in §4–§6:**

| Missing | Deliverable blocked | Track |
|---|---|---|
| Any detector that sees a *box* | D7 | B |
| `FrameEvidence` assembly (perception stack) | D1 | B |
| Protocol loader, predicates, `SequenceValidator` | D2, D3 | A |
| Event log writer | D4 | C |
| Voice output | D3 | C |
| Video recorder + MJPEG stream | D5 | C |
| Browser GUI | D6 | C |
| CLI entrypoint (`har/app.py`) | D1 | C |
| Own dataset + trained weights | D7 | B |
| Rack-relative (orientation-agnostic) frame | differentiator | B |

### 1.4 The three things that actually decide whether we qualify

1. **A trained detector of our own.** This is the single highest-leverage item and the one
   most teams will fudge. I unpickled `models/yolo11n.pt`: its `names` dict is exactly
   **COCO-80** (`class 39 = bottle`, `class 0 = person`), and there is **no** box, crate,
   container, tray, bin, package or cube class. So stock weights *cannot* see a single
   object in PTS-01. Either we fine-tune on our own 3-class dataset (satisfies D7 and the
   "generate your own dataset" requirement) or we fall back to HSV colour segmentation
   (works, but D7 stays unmet). **Decision needed before Phase 2: is a GPU available to
   anyone tonight?** ~150 webcam images and 100 epochs of `yolo11n` at 640 is a
   20–40 minute train on a modest GPU.

2. **Sequence validation that is demonstrably right.** Skipping and reordering must be
   caught *and* logged *and* spoken. This is the "Core Challenge" sentence of the problem
   statement. It is 100 % absent today and it is Track A's whole job. Because it is pure
   logic on `FrameEvidence`, it can be finished and unit-tested without a camera, without
   torch, and without waiting for anybody.

3. **A demo that cannot fail.** Every prior script in this repo opened `cv2.VideoCapture(0)`
   at import time. `har/app.py` must take `--source 0|path.mp4` so the entire demo can be
   replayed from a recording if the venue laptop's camera misbehaves.

**What we should *not* spend time on:** 3D Human Mesh Recovery (CLIFF/PyMAF). It is an
*optional advanced* challenge, it will not run on 2 CPU cores, and it cannot be trained in
one night. The rack-relative homography in Phase 3 buys the same talking point
("we track relative to the payload rack, not the floor") in ~80 lines. Say the HMR roadmap
in the pitch; do not build it.

---

## 2. Architecture and ownership

```
HAR-system/
├── har/
│   ├── contracts.py            SHARED — frozen, see §3
│   ├── app.py                  C
│   ├── perception/             B
│   │   ├── geometry.py         B   (done)
│   │   ├── tracker.py          B   (done)
│   │   ├── interaction.py      B   (done)
│   │   ├── adapters.py         B   (done)
│   │   ├── detector.py         B   YoloDetector
│   │   ├── color_detector.py   B   HSV fallback, same interface
│   │   ├── pose.py             B   wrist extraction wrapper
│   │   ├── rack.py             B   Phase 3 homography
│   │   └── perception.py       B   assembles FrameEvidence
│   ├── protocol/               A
│   │   ├── spec.py             A   yaml -> ProtocolSpec
│   │   ├── predicates.py       A   the 6 frozen predicates
│   │   └── validator.py        A   SequenceValidator
│   ├── out/                    C
│   │   ├── eventlog.py         C   JSONL + CSV
│   │   ├── recorder.py         C   mp4
│   │   ├── streamer.py         C   MJPEG
│   │   └── speaker.py          C   offline TTS
│   └── ui/                     C
│       ├── overlay.py          C   in-frame HUD
│       └── web.py              C   Flask page + /status + /stream
├── protocols/pts01.yaml        SHARED — the locked protocol
├── config/bytetrack.yaml       SHARED — tracker tuning
├── models/                     B — weights in, trained weights out
├── tests/                      each track owns tests for its own package
├── tools/                      B: capture_dataset.py · C: make_synthetic_video.py, replay_events.py
└── docs/
```

### Ownership rules (this is the whole coordination protocol)

| Person | Owns | May **not** touch | Dependencies |
|---|---|---|---|
| **A — Cognition** | `har/protocol/**`, `protocols/*.yaml`, `tests/test_protocol_*.py`, `tests/test_validator.py` | `har/perception/**`, `har/out/**`, `har/ui/**`, `har/app.py` | PyYAML **only** |
| **B — Perception** | `har/perception/**`, `models/`, `datasets/`, `tools/capture_dataset.py`, `tests/test_perception*.py` | `har/protocol/**`, `har/out/**`, `har/ui/**`, `har/app.py` | ultralytics, opencv, numpy |
| **C — Output & interface** | `har/out/**`, `har/ui/**`, `har/app.py`, `tools/make_synthetic_video.py`, `tools/replay_events.py`, `demo/`, `tests/test_out*.py` | `har/perception/**`, `har/protocol/**` | flask, pyttsx3, opencv |

**Shared files** — `har/contracts.py`, `protocols/pts01.yaml`, `config/bytetrack.yaml`,
`requirements.txt`. To change one: post it in the group chat, bump `CONTRACT_VERSION` if it
is `contracts.py`, and log it in §11. Nobody edits a shared file silently.

**Import direction (one-way, enforced by the tests):**

```
C  ──►  contracts  ◄──  A
B  ──►  contracts
A  ──X──►  B        A never imports perception
C  ──X──►  B, A     C never imports perception or protocol internals
```

Track A's package must stay importable with **no cv2, torch or ultralytics installed**.
`tests/test_contracts.py` fails the build if that breaks. This is not pedantry: it is what
lets A finish and prove their half of the system in an environment that has no camera.

---

## 3. The frozen contract

`har/contracts.py` v1.0.0 defines the only things that cross a boundary:

| Type | Producer | Consumer | Note |
|---|---|---|---|
| `Detection` | B | B | raw hit, one frame |
| `ObjectTrack` | B | A, C | `measured=False` means predicted — **never complete a step on it** |
| `Wrist` | B | A, C | COCO kp 9=left, 10=right |
| `FrameEvidence` | B | A, C | the *only* input to `SequenceValidator.update` |
| `StepEvent` | A | C | append-only, JSON round-trips |
| `ProtocolSpec` / `StepSpec` / `Zone` | yaml | A, C | |
| `UiStatus` | A+C | C | GUI polls this; C renders, never computes |

Seams (each side implements, nobody reaches across): `ObjectDetector` (B),
`EventSink` + `Speaker` + `FrameSource` (C).

**Why this matters for a 3-person split:** every one of these has `to_dict()` and a tested
JSON round trip. That means **Track C can be built and finished against
`tests/fixtures/events_correct.jsonl` before Track A has written a line of the validator**,
and **Track A can be built and finished against a hand-written `FrameEvidence` sequence
before Track B has a working detector.** Neither is blocked on the other at any point.

---

## 4. Phase 1 — Headless spine

**Goal:** run `har/app.py --source tests/fixtures/synthetic_correct.mp4 --headless` and get
a correct `events.jsonl`, with all three tracks working from disjoint files and no camera.

**Duration:** ~3.5 h · **Integration gate G1** at the end.

### Track A — protocol loader, predicates, validator

| # | File | Task | Acceptance |
|---|---|---|---|
| A1 | `tests/fixtures/evidence_*.json` | **First, in the first 20 minutes.** Hand-write three `FrameEvidence` sequences: correct, skip, wrong-order. 10–15 frames each, JSON via `FrameEvidence.to_dict()`. | Files exist; Track C and the validator both consume them |
| A2 | `har/protocol/spec.py` | `load_protocol(path, frame_size) -> ProtocolSpec`. Resolve normalised zone boxes to pixels. Raise on unknown predicate, dangling `requires`, duplicate `step_id`. | `tests/test_protocol_loader.py` |
| A3 | `har/protocol/predicates.py` | Implement the 6 frozen predicates over `FrameEvidence`. Signature `def name(evidence, spec, step, state) -> bool`. Unknown name → `KeyError` at load, not at runtime. | one test per predicate, plus a "wave past is not a pickup" negative case |
| A4 | `har/protocol/validator.py` | `SequenceValidator(spec, clock) -> .update(FrameEvidence) -> list[StepEvent]` | see rules below |
| A5 | `tests/test_validator.py` | Drive the validator with A1's fixtures. | ≥12 assertions green |

**Validator rules — these are the semantics of the whole project, implement them exactly:**

1. Exactly one step is *current*. Start at index 1, emit `STARTED` once on entry.
2. Evaluate the current step's predicate every frame. On `hold_frames` consecutive
   satisfied frames → emit `COMPLETED`, advance, emit `STARTED` for the next.
3. **Every frame, also evaluate all later steps.** If step *k > current* is satisfied while
   the current one is not, emit `OUT_OF_ORDER` for step *k*.
4. If a later step stays satisfied for `hold_frames`, emit `SKIPPED` for every intervening
   step (status `VIOLATION`, message from its `voice_alert`) and jump the cursor there.
5. A step whose `timeout_s` elapses without completing emits `TIMEOUT` and stays current.
   Emit `TIMEOUT` at most once per step.
6. After step 8 completes, emit `PROTOCOL_COMPLETE`. `update()` returns `[]` forever after.
7. **Never** complete a step on a track with `measured=False`.
8. The validator is pure: no clock reads, no file IO, no cv2. Time comes in via
   `evidence.t_rel`. That is what makes it testable without a camera.

### Track B — detection and evidence assembly

| # | File | Task | Acceptance |
|---|---|---|---|
| B1 | `models/` inventory + `tools/probe_fps.py` | Measure single-model and dual-model FPS on the target hardware at `imgsz` 480/640. Record it. | A number in `docs/METRICS.md` |
| B2 | `har/perception/color_detector.py` | HSV red/blue + tray contour detector inside the rack ROI, median-of-N smoothing. Implements `contracts.ObjectDetector`. | detects the 5 PTS-01 objects in a still frame |
| B3 | `har/perception/detector.py` | `YoloDetector` wrapping `ultralytics`, `classes` from config, `backend="yolo11n"`. Same interface. | swap with B2 via one flag |
| B4 | `har/perception/pose.py` | Thin wrapper returning `list[Wrist]`; supports `--pose-every-n` frame skipping. | wrists present in a test frame |
| B5 | `har/perception/perception.py` | `PerceptionStack.process(frame, frame_index, t_rel) -> FrameEvidence`. Runs `TrackerRegistry` + one `InteractionMachine` per object and packs the result. | `evidence.to_dict()` matches A1's fixture shape exactly |
| B6 | `tools/capture_dataset.py` | Webcam grabber: `space` to save, auto-increments, writes `datasets/raw/`. | 150 frames captured in one sitting |

**B5 is the critical interface.** Compare your output against
`tests/fixtures/evidence_correct.json` byte-for-byte on the contract fields before
declaring it done. If they differ, the fixture is the spec — change your code, or propose a
contract change in §11. Do not silently diverge.

### Track C — log, voice, entrypoint

| # | File | Task | Acceptance |
|---|---|---|---|
| C1 | `tools/replay_events.py` | **First.** Read `tests/fixtures/events_*.jsonl` → `list[StepEvent]`. This is C's stub for Track A. | both fixtures parse |
| C2 | `har/out/eventlog.py` | `JsonlEventLog(path)` implements `EventSink`. Append-only JSONL + mirrored CSV, ISO-8601, flush per event (a crash must not lose the log). | round-trip test |
| C3 | `har/out/speaker.py` | pyttsx3 on a daemon thread, 1-deep queue, **drop** rather than block. `--no-voice` support. | frame loop never waits on TTS |
| C4 | `har/app.py` | CLI: `--source 0\|path.mp4`, `--protocol`, `--headless`, `--out-dir`, `--no-voice`, `--pose-every-n`, `--detector yolo\|color`. | `--help` works; runs `--headless` on a file |
| C5 | `tests/test_out.py` | Log + replay tests against the fixtures. | green |

**C's Phase-1 definition of done does not need A or B.** `tools/replay_events.py` feeds
`eventlog.py` and `speaker.py` directly. Wire the real validator in at G1.

### Gate G1 — everyone stops and integrates (~30 min, all three present)

```bash
.venv/bin/python -m unittest discover          # every track's tests, together
.venv/bin/python tools/make_synthetic_video.py # C's scripted mp4
.venv/bin/python -m har.app --source tests/fixtures/synthetic_correct.mp4 \
    --protocol protocols/pts01.yaml --headless --out-dir runs/g1
cat runs/g1/events.jsonl
```

**G1 passes when:** the run exits 0; `events.jsonl` contains 8 `COMPLETED` in index order
plus one `PROTOCOL_COMPLETE`; and zero `SKIPPED`/`OUT_OF_ORDER`. Then re-run on
`synthetic_wrong_order.mp4` and confirm exactly one `OUT_OF_ORDER` and one `SKIPPED`.

---

## 5. Phase 2 — Live system

**Goal:** the full demo works from a real webcam with the browser GUI, voice, recording and
streaming all live.

**Duration:** ~3 h · **Integration gate G2.**

| Track | Tasks |
|---|---|
| **A** | A6: threshold pass — replay the G1 runs and tune `hold_frames` / `timeout_s` in `protocols/pts01.yaml` until a real recording validates cleanly. A7: `UiStatus` producer (`current`, `next_instruction`, `completed`, `skipped`, `violations`, `last_alert`). A8: three recorded demo runs (correct, skip, wrong order) with their event logs committed under `demo/`. |
| **B** | B7: live camera at a usable frame rate — `imgsz=480`, `conf=0.45`, pose every 2nd frame. B8: `datasets/` capture (~150 frames, 3 classes: tray, red box, blue box; vary angle, lighting, distance). B9: `tools/train.py` + `datasets/data.yaml`; **fine-tune `yolo11n`, 100 epochs** → `models/pts01_best.pt`. B10: swap `--detector yolo` onto the trained weights and re-measure. |
| **C** | C6: `har/out/recorder.py` — `cv2.VideoWriter` mp4v into `recordings/run_<ts>.mp4`. C7: `har/out/streamer.py` — MJPEG at `--stream-host 0.0.0.0 --stream-port 8080`. C8: `har/ui/web.py` + `index.html` — `<img src="/stream">`, step checklist, violation banner, live log tail, FPS. C9: `har/ui/overlay.py` — in-frame HUD as the fallback when a browser is impractical. C10: bind `0.0.0.0`, not `127.0.0.1`. |

### Gate G2 — the live rehearsal

```bash
.venv/bin/python -m har.app --source 0 --protocol protocols/pts01.yaml \
    --detector yolo --record --stream-host 0.0.0.0 --stream-port 8080
```

**G2 passes when, on a live camera, all of these are simultaneously true:** the GUI shows
the current and next step; a deliberately skipped step produces a spoken warning *and* a
`SKIPPED` row in the JSONL *and* a red banner; `recordings/*.mp4` is playable afterwards;
and `http://<other-machine>:8080` shows the video. Anything not true at G2 gets cut from
the demo and moved to the roadmap.

**Cut rule:** if B9's fine-tune has not converged by the start of G2, ship
`--detector color` and state plainly in the pitch that the trained detector is in progress.
Do not let a training run hold the demo hostage.

---

## 6. Phase 3 — Evidence and differentiation

**Goal:** numbers, robustness, and the two things that separate us from a MediaPipe demo.

**Duration:** ~2.5 h · **Gate G3** = rehearsed.

| Track | Tasks |
|---|---|
| **A** | A9: `tools/evaluate.py` — replay every `demo/` run and emit `docs/METRICS.md`: step accuracy, sequence completion rate, false-alarm rate, mean per-step latency. A10: a deliberate-failure table showing each violation class caught. |
| **B** | B11: **`har/perception/rack.py`** — 4 rack fiducials (ArUco or drawn corners) → `cv2.getPerspectiveTransform` → all boxes and zones expressed in rack space. B12: the demo move — rotate the whole rig 90° mid-run and show the sequence still validates. B13: `model.export(format="onnx")` and a CPU-vs-ONNX latency row. |
| **C** | C11: `requirements.lock` + a local wheelhouse, so the "offline standalone" claim is literal. C12: `README.md` quickstart that a judge can follow. C13: demo script rehearsal and the slide deck. |

### Gate G3

`docs/METRICS.md` has real numbers; the 90° rotation trick works on camera; the machine has
been disconnected from the network and the demo still runs end to end.

---

## 7. Coordination protocol — how three people avoid each other

1. **Disjoint files.** The §2 table means git cannot produce a conflict between two people.
   If you find yourself editing outside your column, stop and raise it.
2. **Fixtures before features.** A1 (evidence fixtures) and C1 (event replay) are each
   track's first task, precisely so nobody waits on anybody.
3. **Contract changes are loud.** `har/contracts.py` and `protocols/pts01.yaml` changes go
   to the group chat first and get logged in §11 with a `CONTRACT_VERSION` bump.
4. **Push green, pull first.** Run `python -m unittest discover` before every push. Pull
   before starting any work session.
5. **Gates are synchronous.** G1/G2/G3 are the only moments all three people are in the same
   room. Everything between them is independent.
6. **No new top-level directories** without saying so.

---

## 8. Risk register

| Risk | Likelihood | Fallback |
|---|---|---|
| No GPU → fine-tune impossible | Medium | `--detector color` (B2) ships instead; D7 becomes roadmap |
| 2 models/frame too slow on demo hardware | **High** | `imgsz=480`, `conf=0.45`, pose every 2nd frame, or drop YOLO detect entirely and run colour-only (halves the cost) |
| Venue webcam fails | Medium | `--source demo/*.mp4` — the entire demo is replayable (A8 records it) |
| HSV red/blue unstable under venue lights | Medium | rack ROI + median-of-N + a white-balance key at startup; last resort = COCO proxy props |
| pyttsx3/espeak missing on the demo OS | Medium | `--no-voice` + on-screen banner + a generated beep |
| Predicate thresholds wrong on real footage | **High** | A6 exists for this; keep `hold_frames` in yaml, never hard-coded |
| Scope creep into 3D HMR | **High** | Cut in §1.4. Rack homography gets the same talking point in 80 lines |
| Two people edit `contracts.py` | Low | §7.3 |

---

## 9. Demo script (target ~4 minutes)

1. **0:00** Open the GUI. Show the rack, the tray, the two boxes. Point at "Step 1 of 8".
2. **0:20** Run the protocol correctly. Voice announces each step; the checklist fills in.
3. **1:40** **The money shot.** Skip step 3, go straight for the blue box. The system
   speaks "Out of sequence", the banner goes red, and a `SKIPPED` row appears in the log.
4. **2:20** Open `events.jsonl` and the CSV. Show timestamps, step ids, statuses.
5. **2:40** Open `http://<second-machine>:8080` — the stream — then `ls -la recordings/`.
6. **3:00** Airplane-mode the laptop. Re-run. Still works: offline standalone.
7. **3:20** *If Phase 3 landed:* rotate the rig 90° and show the sequence still validates —
   "we track relative to the payload rack, not to gravity."
8. **3:40** `docs/METRICS.md`: step accuracy, false-alarm rate, FPS on CPU.

---

## 10. What we are explicitly not doing

3D Human Mesh Recovery · temporal Transformer/LSTM step classifier (a rule-based FSM is
more defensible for flight software: explainable and auditable) · RTSP (MJPEG satisfies
"stream to a specific IP") · multi-astronaut · multi-camera fusion · cloud anything.

---

## 11. Contract change log

| Date | Version | Change | By |
|---|---|---|---|
| 2026-09-02 | 1.0.0 | Initial freeze: `Detection`, `ObjectTrack`, `Wrist`, `FrameEvidence`, `StepEvent`, `ProtocolSpec`, `StepSpec`, `Zone`, `UiStatus`, seams `ObjectDetector`/`EventSink`/`Speaker`/`FrameSource`. | — |

---

## Appendix A — repo history

Everything removed in the restructure is recoverable from the initial commit:

```bash
git show 19c5436 --stat                       # what used to be here
git show 19c5436:bottle_monitor.py            # reference webcam loop + debug overlay
git show 19c5436:pickup_detectionv_4.py       # reference tuning constants
```

The 10 legacy camera scripts (5,127 of the original 5,788 Python lines) were
version-control-by-copy-paste — their own window titles read `"SIH26174 - Step 6B/6D/6E"`,
which were *development* steps, not experiment steps. The reusable arithmetic from
`bottle_monitor.py` was moved into `har/perception/` and is still covered by tests; the
frame loop and debug overlay are recoverable from the commit above when Track C writes
`har/app.py` and `har/ui/overlay.py`.

## Appendix B — quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover
```

The suite collects **37 tests**: 28 run in a bare interpreter (no cv2, no torch, no numpy)
and 9 skip when PyYAML is absent. Verified on 2026-09-02 with system Python 3.11.2:
`Ran 37 tests ... OK (skipped=9)`; with PyYAML installed: `Ran 37 tests ... OK`.
