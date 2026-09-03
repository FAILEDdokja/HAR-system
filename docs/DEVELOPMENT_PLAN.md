# SIH26174 — Development Plan

**Branch:** `arena/01a062b8-har-system` · **Contract:** v1.0.0 · **Protocol:** PTS-01 v1.0.0
**Target:** a working, well-documented prototype by tomorrow.

## How to use this document

You are one of three people. **Read §1 and §2 once, then work only from your own section:**

| You are | Read | Your steps |
|---|---|---|
| **Person A — Cognition** | §5 | **A1 → A10**, in order |
| **Person B — Perception** | §6 | **B1 → B9**, in order |
| **Person C — Output & interface** | §7 | **C1 → C12**, in order |

Every step is written as *Do* / *Done when*, so you never have to guess what finished means.
§8 is the shared timeline; §3 tells you which files are yours. Nothing else is required
reading.

---

## 1. Review — what we are being asked to do, and what qualifies

### 1.1 The problem in one sentence

ISRO wants an **offline** system that watches a fixed-payload camera, knows which step of a
pre-defined experiment the astronaut is on, **tells them the next step**, **speaks up when
they skip or reorder one**, and leaves a **timestamped log** plus a **stored and streamed
video** — with no ground station in the loop.

The hard part is not computer vision. It is **protocol compliance**: holding a model of a
procedure and comparing the observed world against it.

### 1.2 The seven expected deliverables and the evidence each needs

| # | Deliverable | What "done" looks like in our demo | Evidence |
|---|---|---|---|
| D1 | Continuously process local video | Runs at a stable frame rate from camera **or** file, no stalls | FPS readout in the GUI |
| D2 | Suggest the next step | Panel always shows the current step and next instruction in plain words | GUI screenshot |
| D3 | Voice alert on skip / out-of-sequence | We deliberately do step 5 before step 3; the box speaks the warning | Screen recording with audio |
| D4 | Timestamped structured lightweight text file | One JSONL + mirrored CSV row per step event | The file, opened on screen |
| D5 | Stream to a specific IP **and** store locally | `http://<host>:8080/stream` opens on a second machine while `recordings/*.mp4` grows | Two-screen shot + `ls -la recordings/` |
| D6 | GUI for monitoring | One browser page: video, step checklist, violations, log tail | GUI screenshot |
| D7 | A trained AI model, offline standalone | **See §2 — we meet this partially and say so openly** | pretrained model + metrics |

### 1.3 Where we stand right now — verified

Checked on 2026-09-02, not recalled. `.venv/bin/python -m unittest discover` → **37 tests, OK.**

**Present and tested:**

| Capability | Location | Tests |
|---|---|---|
| Geometry primitives | `har/perception/geometry.py` | 4 |
| Single-target association (survives detector ID churn) | `har/perception/tracker.py` | 6 |
| Hand-object interaction FSM `IDLE→NEAR→PICKED_UP→CARRYING→RELEASED` | `har/perception/interaction.py` | 5 |
| Duck-typed ultralytics adapters (run with no torch installed) | `har/perception/adapters.py` | 7 |
| Frozen cross-track contracts, stdlib only | `har/contracts.py` | 6 |
| Locked 8-step protocol | `protocols/pts01.yaml` | 9 |

**Missing — every line is somebody's step in §5–§7:**

| Missing | Blocks | Whose step |
|---|---|---|
| A detector that sees the protocol's objects | D1 | B2 |
| `FrameEvidence` assembly | D1 | B4 |
| Protocol loader, predicates, `SequenceValidator` | D2, D3 | A2, A3, A4 |
| Event log writer | D4 | C2 |
| Voice output | D3 | C3 |
| Video recorder + MJPEG stream | D5 | C7, C8 |
| Browser GUI | D6 | C9 |
| CLI entrypoint | D1 | C5 |
| Metrics on our own recorded footage | evaluation rigour | A8, A9 |

### 1.4 The three things that decide whether we qualify

1. **Sequence validation that is demonstrably right.** Skipping and reordering must be
   caught, logged *and* spoken. This is the literal "Core Challenge" of the problem
   statement, it is 100 % absent today, and it is Person A's whole job. Because it is pure
   logic over `FrameEvidence`, it can be finished and proven with no camera and no torch.

2. **A detector that reliably sees the five props.** Not by training (§2) — by colour.
   This is Person B's whole job, and the practical move is to **tape a distinct colour on
   every prop** so detection becomes trivial and repeatable. See B2.

3. **A demo that cannot fail.** Every legacy script opened `cv2.VideoCapture(0)` at import
   time. `har/app.py` must take `--source 0|path.mp4` so the entire demo replays from a
   recording if the venue laptop's camera misbehaves.

---

## 2. The no-training decision — read this before touching detection

**We are not training anything. There is no GPU and no time. This is final.**

That has one honest consequence, and the plan does not paper over it:

> **D7 as literally written — "a trained AI model" — is only partly met.** We ship a
> **pretrained** YOLO11n-pose (a real trained deep network, used offline for human pose and
> for the `person` class) plus a **purpose-built classical colour detector** for the five
> protocol objects. We do **not** ship a model we trained ourselves.

What we do instead, and why it is still a strong submission:

- **The protocol objects are detected by colour, not by a network.** I unpickled
  `models/yolo11n.pt`: its `names` dict is exactly COCO-80 (`class 39 = bottle`,
  `class 0 = person`) with **no** box, crate, container, tray, bin, package or cube class.
  Stock weights could never see a PTS-01 object anyway. Colour segmentation is the correct
  engineering choice here, not a compromise — the props are ours to design, so we make them
  trivially detectable.
- **We still generate our own dataset** — the problem statement asks for one. Person A
  records three webcam runs and hand-annotates the true step timings (A8). That dataset is
  used for **evaluation**, producing real numbers (A9), which is the part judges can
  actually check.
- **We say the fine-tune out loud as the costed next step**, with the recipe in §11, rather
  than implying we did it. A team that states its own gap precisely reads as more credible
  than one that overclaims.

**Consequences for the plan:** no `tools/train.py`, no `datasets/`, no `data.yaml`, no
`models/pts01_best.pt`, no background training terminal. If anyone finds themselves
labelling images, they have drifted — stop and pick up their next step.

### Are any model changes required? No.

Both weights are already committed and already correct. I unpickled them and read the
pickles directly:

| File | ultralytics version | Head | Classes | Used for |
|---|---|---|---|---|
| `models/yolo11n.pt` | 8.2.100 | detection only | COCO-80 (`0 = person`, `39 = bottle`, …) | **nothing — see below** |
| `models/yolo11n-pose.pt` | 8.2.100 | **pose head present** (`kpt_shape`) | 1 (`person`) | wrists, and the person gate |

**Nothing to train, nothing to download, nothing to convert.** Both were saved by
ultralytics **8.2.100**, which is exactly what `requirements.txt` pins — so there is no
version mismatch waiting to bite on the demo machine.

**One simplification worth making (B4).** `yolo11n-pose` already detects the `person` class
*and* returns keypoints, so it can serve both the wrist extraction and the "is an operator
in frame?" gate. That means **`yolo11n.pt` does not need to be loaded at all** — one
inference pass per frame instead of two, and one fewer model in memory. Drop it from the
runtime path; leave the file in the repo as a spare. The protocol objects come from colour,
never from either network.

---

## 3. Architecture and file ownership

```
HAR-system/
├── har/
│   ├── contracts.py            SHARED — frozen, do not edit
│   ├── app.py                  C5
│   ├── perception/             PERSON B
│   │   ├── geometry.py         done   ├── color_detector.py  B2
│   │   ├── tracker.py          done   ├── pose.py            B3
│   │   ├── interaction.py      done   ├── rack.py            B7
│   │   ├── adapters.py         done   └── perception.py      B4
│   ├── protocol/               PERSON A
│   │   ├── spec.py             A2  ├── predicates.py  A3  └── validator.py  A4, A7
│   ├── out/                    PERSON C — eventlog C2 · speaker C3 · recorder C7 · streamer C8
│   └── ui/                     PERSON C — web C9 · overlay C10
├── protocols/pts01.yaml        SHARED
├── config/                     bytetrack.yaml · colours.yaml (B2)
├── models/                     pretrained weights only
├── tests/                      each person owns tests for their own package
├── tools/                      B: probe_fps · C: replay_events, make_synthetic_video · A: evaluate
└── docs/
```

### Ownership rules — this is the entire coordination protocol

| Person | Owns (only you write here) | May **not** touch | Dependencies |
|---|---|---|---|
| **A — Cognition** | `har/protocol/**` · `protocols/*.yaml` · `tools/evaluate.py` · `demo/` · `docs/METRICS.md` · `tests/test_protocol_config.py`, `tests/test_predicates.py`, `tests/test_validator.py` | `har/perception/**`, `har/out/**`, `har/ui/**`, `har/app.py`, `docs/PERF.md` | **PyYAML only** |
| **B — Perception** | `har/perception/**` · `config/colours.yaml` · `tools/probe_fps.py` · `docs/PERF.md` · `tests/test_perception_geometry.py`, `_tracker.py`, `_interaction.py`, `_adapters.py`, `_color_detector.py` | `har/protocol/**`, `har/out/**`, `har/ui/**`, `har/app.py`, `docs/METRICS.md` | ultralytics, opencv, numpy |
| **C — Output & interface** | `har/out/**` · `har/ui/**` · `har/app.py` · `tools/replay_events.py`, `tools/make_synthetic_video.py` · `README.md` · `requirements.lock` · `wheelhouse/` · `tests/test_out_eventlog.py`, `tests/test_out_speaker.py` | `har/perception/**`, `har/protocol/**` | flask, pyttsx3, opencv |

The globs are not theoretical — the test files on disk are already named to match them, so
`ls tests/` tells you who owns what. Nobody renames a test file that is not theirs.

**Shared files — post in the group chat before editing, and log it in §12:**
`har/contracts.py` · `tests/test_contracts.py` · `protocols/pts01.yaml` ·
`config/bytetrack.yaml` · `requirements.txt` · `.gitignore`.

`docs/METRICS.md` (A) and `docs/PERF.md` (B) are deliberately two files. Three people
writing one metrics file is a guaranteed merge conflict, and a merge conflict at 17:00 is
how demos get lost.

**Import direction:**

```
A ──► contracts            A imports PyYAML + contracts. Nothing else.
B ──► contracts
C ──► contracts            in har/out/** and har/ui/**
```

**`har/app.py` is the single exception and the only composition root.** It — and nothing
else in C's tree — imports `PerceptionStack` and `SequenceValidator`. If you find yourself
importing perception or protocol inside `har/out/` or `har/ui/`, stop: pass the data in
instead.

Person A's package must stay importable with **no cv2, torch or ultralytics installed**.
`tests/test_contracts.py` fails the build if that breaks. This is what lets A finish and
prove their half of the system on a laptop with no camera.

---

## 4. The frozen contract

`har/contracts.py` v1.0.0 is already written. The types that cross a boundary:

| Type | Producer | Consumer | Note |
|---|---|---|---|
| `Detection` | B | B | one raw hit, one frame |
| `ObjectTrack` | B | A, C | `measured=False` means predicted — **never complete a step on it** |
| `Wrist` | B | A, C | COCO keypoint 9 = left, 10 = right |
| `FrameEvidence` | B | A, C | the *only* input to `SequenceValidator.update` |
| `StepEvent` | A | C | append-only, JSON round-trips |
| `ProtocolSpec`/`StepSpec`/`Zone` | yaml | A, C | |
| `UiStatus` | A | C | GUI polls it; C renders, never computes |

Every type has `to_dict()` with a tested JSON round trip. **That is why nobody is blocked
on anybody:** C builds against `tests/fixtures/events_*.jsonl` before A has written the
validator, and A builds against hand-written `FrameEvidence` fixtures before B has a
working detector.

Exact signatures for every file you must create are in **Appendix A**. If what you need is
there, do not ask. If it is not, that is a contract change — post it and log it in §12.

---

## 5. Person A — Cognition

*Protocol model, predicates, sequence validation, evaluation.*
**Environment: PyYAML only. No cv2, no torch, no camera needed for A1–A5.**

### A1 — Write the evidence fixtures · 20 min · Phase 1

**Files:** `tests/fixtures/evidence_correct.json`, `evidence_skip.json`, `evidence_wrong_order.json`

**Do:** hand-author three sequences of 10–15 frames each, as a JSON list of
`FrameEvidence.to_dict()` objects. Model the PTS-01 objects appearing, being picked up and
released in the right zones. `skip` omits the red box entirely; `wrong_order` places blue
before red.

**Done when:** a five-line script loads each file into `FrameEvidence` with no error.

*Do this first. It unblocks A4 and lets B5 check their output against a real target.*

### A2 — Protocol loader · 40 min · Phase 1

**Files:** `har/protocol/spec.py`, `tests/test_protocol_loader.py`

**Do:** `load_protocol(path, frame_size) -> ProtocolSpec`. Resolve normalised zone boxes to
pixels. Raise `ProtocolError` on: unknown predicate name, dangling `requires`, duplicate
`step_id`, target or zone not declared, non-linear chain.

**Done when:** `tests/test_protocol_config.py` (9 tests) still green **and** your loader
tests green. Every failure mode above is asserted.

### A3 — Predicates · 60 min · Phase 1

**Files:** `har/protocol/predicates.py`, `tests/test_predicates.py`

**Do:** implement the six names in the `protocols/pts01.yaml` header — `object_stable`,
`object_left_zone`, `hoi_cycle`, `settled`, `transfer`, `hands_clear`. Uniform signature
`(evidence, spec, step, state) -> bool`, exported in a `PREDICATES` dict. Use
`har/perception/geometry.py` helpers only through values already in `FrameEvidence` — **do
not import the perception package.**

**Done when:** one positive and one negative test per predicate. The negative case that
matters most: a hand sweeping past an object that does not move must return `False`.

### A4 — SequenceValidator · 75 min · Phase 1

**File:** `har/protocol/validator.py`

**Do:** `SequenceValidator(spec)` with `.update(evidence) -> list[StepEvent]`. These
semantics are the whole project — implement them exactly:

1. Exactly one step is *current*. Start at index 1, emit `STARTED` once on entry.
2. Evaluate the current step's predicate every frame. On `hold_frames` consecutive
   satisfied frames → emit `COMPLETED`, advance, emit `STARTED` for the next.
3. **Every frame, also evaluate all later steps.** If step *k > current* is satisfied while
   the current one is not, emit `OUT_OF_ORDER` for *k*.
4. If a later step stays satisfied for `hold_frames`, emit `SKIPPED` for every intervening
   step (status `VIOLATION`, message from that step's `voice_alert`) and jump the cursor.
5. A step whose `timeout_s` elapses emits `TIMEOUT` and stays current. At most once per step.
6. After step 8 completes, emit `PROTOCOL_COMPLETE`; `update()` returns `[]` forever after.
7. **Never** complete a step on a track with `measured=False`.
8. Pure: no clock reads, no file IO, no cv2. Time arrives via `evidence.t_rel`.

**Done when:** A1's `evidence_correct.json` produces 8 `COMPLETED` in index order plus one
`PROTOCOL_COMPLETE`, and zero violations.

### A5 — Validator violation tests · 30 min · Phase 1

**File:** `tests/test_validator.py`

**Done when:** `evidence_skip.json` → exactly one `SKIPPED`; `evidence_wrong_order.json` →
exactly one `OUT_OF_ORDER`; a stalled step emits `TIMEOUT` once and not twice; a
`measured=False` track never completes a step.

### A6 — Threshold tuning on real footage · 45 min · Phase 2

**File:** `protocols/pts01.yaml`

**Do:** replay the G1 runs and tune `hold_frames` / `timeout_s` until a real recording
validates cleanly. Thresholds live in the yaml, never in code.

**Done when:** one real recording validates 8/8 with zero violations.

### A7 — UiStatus producer · 45 min · Phase 2

**File:** `har/protocol/validator.py` — `status() -> UiStatus`

**Do:** populate `current_step_id`, `next_step_id`, `next_instruction`, `completed`,
`skipped`, `violations`, `state`, `last_alert`.

**Done when:** a test asserts a full `UiStatus` mid-run and again at completion.

### A8 — Record and annotate our own dataset · 45 min · Phase 2

**Files:** `demo/correct.mp4`, `demo/skip.mp4`, `demo/wrong_order.mp4`, `demo/ground_truth.json`

**Do:** use C7's recorder to capture three real runs, then hand-annotate the true start and
end time of each step. **This is the custom dataset the problem statement asks for**, used
for evaluation rather than training.

**Done when:** three mp4 files plus one ground-truth json committed, and
`--source demo/correct.mp4` reproduces its log.

### A9 — Evaluation script and metrics · 60 min · Phase 3

**Files:** `tools/evaluate.py`, `docs/METRICS.md`

**Do:** replay every `demo/` run, diff emitted events against `ground_truth.json`, and write
the numbers.

**Done when:** `docs/METRICS.md` has real values for step accuracy, sequence completion
rate, false-alarm rate, and mean per-step latency. No placeholders, no estimates.

### A10 — Violation evidence table · 20 min · Phase 3

**File:** `docs/METRICS.md`

**Done when:** `SKIPPED`, `OUT_OF_ORDER` and `TIMEOUT` each appear as a row with the actual
timestamped log line as evidence.

---

## 6. Person B — Perception

*Colour detection, pose, tracking, and assembling `FrameEvidence`.*
**Environment: ultralytics, opencv, numpy.**

> **The single most important practical instruction in this plan: put a distinct colour on
> every prop.** We are detecting by colour, so the props are ours to design. Tape or paint:
> **black tray, yellow lid, red box, blue box, green vial.** Also print a small colour-key
> card and lay it in frame — it is the white-balance reference at startup and it costs
> nothing. Do this before B2, not during it.

### B1 — Measure what the hardware can do · 20 min · Phase 1

**Files:** `tools/probe_fps.py`, `docs/PERF.md`

**Do:** time `yolo11n-pose` alone, with and without the `person`-gate optimisation from §2,
at `imgsz` 480 and 640.

**Done when:** the numbers are written into `docs/PERF.md` — **your** file, not A's
`docs/METRICS.md`. Everything downstream is sized from this, so do not skip it.

### B2 — Colour detector · 75 min · Phase 1

**Files:** `har/perception/color_detector.py`, `config/colours.yaml`, `tests/test_color_detector.py`

**Do:** `ColorDetector` implementing `contracts.ObjectDetector`. HSV range per label,
restricted to the rack ROI, median-of-N smoothing, minimum contour area. Ranges come from
`config/colours.yaml`, never hard-coded — venue lighting will force you to retune them and
you do not want to be editing code in front of judges.

**Done when:** all five labels are detected in a still frame, and `--detector color` returns
them with stable identities for 60 seconds.

*If five colours prove unreliable under time pressure, the cheapest trim is to delete step 7
(`SAMPLE_TRANSFER`, the vial) from `protocols/pts01.yaml` and re-point step 8's `requires`
at `VERIFY_BLUE_PLACED`. That is a two-minute yaml edit, not a code change — and
`tests/test_protocol_config.py` will catch it if you get the chain wrong. Post it in the
group chat first (§3).*

### B3 — Wrist extractor · 30 min · Phase 1

**File:** `har/perception/pose.py`

**Do:** `WristExtractor` wrapping `yolo11n-pose`, returning `list[Wrist]`, with
`every_n_frames` support.

**Done when:** wrists appear on a test frame, **and** on a skipped frame the extractor
returns the previous result rather than an empty list — an empty list reads to the
interaction FSM as "hands vanished" and will corrupt the pickup detection.

### B4 — PerceptionStack · 60 min · Phase 1

**File:** `har/perception/perception.py`

**Do:** `PerceptionStack.process(frame, frame_index, t_rel) -> FrameEvidence`. Runs the
detector, the `TrackerRegistry` (one tracker per protocol label) and one
`InteractionMachine` per label, then packs the result. Gate on the COCO `person` class:
when nobody is in frame, skip the expensive work — that is a free frame-rate win.

**Done when:** `evidence.to_dict()` matches A1's fixture field-for-field. **The fixture is
the spec.** If they differ, change your code or propose a contract change in §12 — do not
silently diverge.

### B5 — Cross-check against the validator · 15 min · Phase 1

**Done when:** B4's output on C4's synthetic video is accepted by A4's validator with no
contract errors and produces 8 `COMPLETED`. *This is gate G1 — do it together with A and C.*

### B6 — Live camera tuning · 45 min · Phase 2

**Done when:** ≥ 12 FPS sustained on the demo hardware with all five labels stable for 60
seconds. Levers, in order: `imgsz=480`, `conf=0.45`, pose every 2nd frame.

### B7 — Rack-relative frame · 60 min · Phase 3

**File:** `har/perception/rack.py`

**Do:** four rack fiducials (ArUco markers, or four taped corners) →
`cv2.getPerspectiveTransform` → express every box and zone in rack space instead of frame
space. This is the honest, cheap version of the optional "orientation-agnostic" challenge:
we track relative to the payload rack, not to gravity.

**Done when:** a unit test shows a 90°-rotated frame yields the same rack-space box.

### B8 — The rotation demo · 20 min · Phase 3

**Done when:** rotate the whole rig 90° mid-run and the sequence still validates. Record it
— this is the most memorable 20 seconds of the pitch.

### B9 — ONNX export and latency · 30 min · Phase 3 · *optional*

**Done when:** `model.export(format="onnx")` succeeds and CPU-vs-ONNX latency is a row in
`docs/PERF.md`. First thing to cut if time runs short.

---

## 7. Person C — Output & interface

*Logging, voice, recording, streaming, GUI, entrypoint.*
**Environment: flask, pyttsx3, opencv.**

### C1 — Event replay tool · 20 min · Phase 1

**File:** `tools/replay_events.py`

**Do:** read `tests/fixtures/events_*.jsonl` into `list[StepEvent]`. **This is your stub for
Person A** — it lets you build C2 and C3 without waiting for the validator.

**Done when:** both fixtures parse, including the wrong-order one.

### C2 — Event log · 40 min · Phase 1

**Files:** `har/out/eventlog.py`, `tests/test_out.py`

**Do:** `JsonlEventLog` implementing `contracts.EventSink`. Append-only JSONL plus a
mirrored CSV. **Flush after every event** — a crash must not lose the log, and the log is
deliverable D4.

**Done when:** a round-trip test proves every `StepEvent` field survives, and the file is
readable after an abrupt exit.

### C3 — Voice · 40 min · Phase 1

**File:** `har/out/speaker.py`

**Do:** `OfflineSpeaker` implementing `contracts.Speaker`. pyttsx3 on a daemon thread with a
one-deep queue that **drops** rather than blocks. `--no-voice` must work.

**Done when:** the frame loop never waits on TTS (prove it: call `say` 100 times in a tight
loop and show elapsed time is near zero), and audio is audible on the demo machine.

*pyttsx3 uses SAPI5 on Windows and needs `espeak-ng` installed on Linux. Check the demo
laptop now, not at the gate.*

### C4 — Synthetic video generator · 30 min · Phase 1

**File:** `tools/make_synthetic_video.py`

**Do:** render scripted frames — coloured rectangles moving through the PTS-01 zones — into
`tests/fixtures/synthetic_correct.mp4` and `synthetic_wrong_order.mp4`.

**Done when:** both files exist and B4 detects the props in them. *Gate G1 runs on these, so
they must exist before the 12:15 gate.*

### C5 — CLI entrypoint · 60 min · Phase 1

**File:** `har/app.py`

**Do:** `main()` wiring frame source → perception → validator → sinks. Every flag in
Appendix A. `--source` accepts a camera index **or** a file path.

**A4 and B4 land at the same moment you do, so do not wait for them.** Build `main()` with
two tiny stubs in `har/app.py` itself — a `StubPerception` that replays A1's evidence
fixtures and a `StubValidator` that replays `tests/fixtures/events_correct.jsonl` — behind
`--stub`. Write the real imports last, in the final 10 minutes, and swap them in at G1.

**Done when:** `--help` works; `--headless --stub` runs to completion; and
`--headless --source <file>` writes `events.jsonl`, `events.csv` and `meta.json` into
`--out-dir` once the real components are wired.

### C6 — Output tests · 20 min · Phase 1

**Done when:** `tests/test_out.py` green against both fixtures.

### C7 — Recorder · 30 min · Phase 2

**File:** `har/out/recorder.py`

**Done when:** `recordings/run_<ts>.mp4` plays in a stock player with correct duration and
fps. Person A needs this for A8, so land it early in Phase 2.

### C8 — MJPEG streamer · 45 min · Phase 2

**File:** `har/out/streamer.py`

**Do:** latest-frame-only. **Never queue frames** — a slow consumer must not stall the
pipeline. Bind `0.0.0.0`, never `127.0.0.1`.

**Done when:** `curl localhost:8080/stream` returns `multipart/x-mixed-replace`, and the
page loads from a second machine on the LAN.

### C9 — Browser GUI · 60 min · Phase 2

**Files:** `har/ui/web.py`, `index.html`

**Do:** `<img src="/stream">`, the eight-step checklist with the current one highlighted, a
violation banner, the live log tail, and FPS. Poll `/status` at 2 Hz.

**Done when:** the page renders live and the banner turns red on a violation. **This is
deliverable D6 and it is also how we show D5 — do not cut it.**

### C10 — In-frame overlay · 20 min · Phase 2

**File:** `har/ui/overlay.py`

**Do:** `draw_hud` in place on the frame — current step, next instruction, alerts, FPS. The
fallback for when a browser is impractical on a projector.

**Done when:** readable at 1280×720 from three metres away.

### C11 — Offline proof and README · 40 min · Phase 3

**Files:** `requirements.lock`, `wheelhouse/`, `README.md`

**Do:** freeze the resolved dependency set and download the wheels locally.

**Done when:** `pip install --no-index --find-links wheelhouse/ -r requirements.lock`
succeeds **with the network disconnected**, and a teammate who has not read this plan can get
a running demo from the README alone. This is what makes "offline standalone system" literal
rather than rhetorical.

### C12 — Rehearsal and deck · 45 min · Phase 3

**Done when:** two timed runs of the §10 demo script, each under 4:30, and the deck built.

---

## 8. Shared timeline and gates

Nine hours and forty-five minutes, three phases. **Gates are the only moments all three
people are in the same place.** Everything between them is independent.

### Time budget — the estimates summed, so the day is honest

| Person | Phase 1 | Phase 2 | Phase 3 | Total |
|---|---|---|---|---|
| **A — Cognition** | 225 min | 135 min | 80 min | **440 min (7.3 h)** |
| **B — Perception** | 200 min | 45 min | 110 min | **355 min (5.9 h)** |
| **C — Output & interface** | 210 min | 155 min | 85 min | **450 min (7.5 h)** |
| Window allotted | 210 min | 150 min | 120 min | 480 min + 75 min of gates |

Three consequences, and they are deliberate:

- **Person A is 15 min over in Phase 1.** If A4 is still moving at 12:00, **let A5 slip past
  G1** — the wrong-order behaviour can be eyeballed from `events.jsonl` at the gate and the
  tests written during lunch. A4 is not negotiable; A5 is.
- **Person C is 5 min over in Phase 2.** Absorbed; no action.
- **Person B has 105 min of slack in Phase 2. B is the designated floater for that phase.**
  Spend it on C9 (the browser GUI, the largest single task in the day) or pull B7 forward.
  Do not invent new perception work to fill it.

### Timeline

| Time | Person A | Person B | Person C | All |
|---|---|---|---|---|
| 08:30 | | | | pull, `venv`, `unittest discover` → **37 green baseline** |
| 08:45 | **A1** fixtures | **B1** FPS probe | **C1** replay | confirm fixtures exist, then separate |
| 09:05 | A2 loader | B2 colour detector | C2 event log | *props taped with colours before this* |
| 09:45 | A3 predicates | B2 continued | C3 voice | |
| 10:45 | A4 validator | B3 pose | C4 synthetic video | |
| 12:00 | A5 violation tests | B4 stack | C5 CLI | |
| **12:15** | | B5 cross-check | C6 tests | **GATE G1** |
| 12:45 | A6 thresholds | B6 live tuning → **floater** | C7 recorder | |
| 13:30 | A7 `UiStatus` | floater: help C9 | C8 streamer | |
| 14:30 | A8 record + annotate | floater: help C9 | C9 GUI, C10 overlay | |
| **15:15** | | | | **GATE G2** |
| 15:45 | A9 evaluate | B7 rack frame | C11 offline + README | |
| 16:45 | A10 failure table | B8 rotation demo | C12 rehearsal | |
| 17:15 | — | B9 ONNX *(optional)* | — | |
| **17:45** | | | | **GATE G3** |
| 18:00 | | | | Two timed rehearsals. Buffer to 18:45. |

### Gate G1 — the headless spine (12:15, ~30 min, all three present)

```bash
.venv/bin/python -m unittest discover
.venv/bin/python tools/make_synthetic_video.py
.venv/bin/python -m har.app --source tests/fixtures/synthetic_correct.mp4 \
    --protocol protocols/pts01.yaml --headless --out-dir runs/g1
cat runs/g1/events.jsonl
```

**Passes when** the run exits 0; `events.jsonl` holds 8 `COMPLETED` in index order plus one
`PROTOCOL_COMPLETE` and zero violations; and a re-run on `synthetic_wrong_order.mp4` yields
exactly one `OUT_OF_ORDER` and one `SKIPPED`.

### Gate G2 — the live system (15:15)

```bash
.venv/bin/python -m har.app --source 0 --protocol protocols/pts01.yaml \
    --detector color --record --stream-host 0.0.0.0 --stream-port 8080
```

**Passes when, simultaneously on a live camera:** the GUI shows current and next step; a
deliberately skipped step produces a spoken warning *and* a `SKIPPED` row *and* a red
banner; `recordings/*.mp4` plays afterwards; and a second machine sees the stream.

**Anything not true at G2 is cut from the demo and moved to §11. Nobody debugs through a
gate.**

### Gate G3 — rehearsed (17:45)

`docs/METRICS.md` (A) and `docs/PERF.md` (B) both hold real numbers; the 90° rotation works
on camera; the machine is disconnected from the network and the demo still runs end to end.

### Cut order when time runs out

Cut in this sequence: **B9** ONNX → **B7/B8** rack frame → **C10** overlay (if the browser
GUI works) → **A10** failure table → step 7 of the protocol (drop the vial).

**Never cut A4 (validator), C2 (log), C3 (voice) or C9 (GUI).** Those four *are* the
submission.

---

## 9. Risks and fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| HSV colours unstable under venue lighting | **High** | `config/colours.yaml` retune (no code change) + colour-key card + rack ROI + median-of-N. Last resort: fewer props, drop step 7 |
| Frame rate too low on demo hardware | **High** | `imgsz=480`, `conf=0.45`, pose every 2nd frame, person-gated processing (B4) |
| Venue webcam fails | Medium | `--source demo/*.mp4` — the entire demo is replayable (A8 records it) |
| pyttsx3 / espeak missing on the demo OS | Medium | `--no-voice` + on-screen banner + a generated beep. **Check the laptop today** |
| Predicate thresholds wrong on real footage | **High** | A6 exists for this. Thresholds stay in yaml, never in code |
| Two people edit `contracts.py` | Low | §3 shared-file rule |
| Someone starts labelling images to "train" | Low | §2. There is no training. Redirect them to their next step |

---

## 10. Demo script (~4 minutes)

1. **0:00** Open the GUI. Show the rack, the tray, the two coloured boxes. Point at "Step 1 of 8".
2. **0:20** Run the protocol correctly. Voice announces each step; the checklist fills in.
3. **1:40** **The money shot.** Skip step 3 and go straight for the blue box. The system
   speaks "Out of sequence", the banner goes red, a `SKIPPED` row appears in the log.
4. **2:20** Open `events.jsonl` and the CSV. Timestamps, step ids, statuses.
5. **2:40** Open `http://<second-machine>:8080` — the stream — then `ls -la recordings/`.
6. **3:00** Put the laptop in airplane mode. Re-run. Still works: offline standalone.
7. **3:20** *If B7/B8 landed:* rotate the rig 90° and show the sequence still validates —
   "we track relative to the payload rack, not to gravity."
8. **3:40** `docs/METRICS.md` and `docs/PERF.md`: step accuracy, false-alarm rate, FPS on CPU.

---

## 11. What we are explicitly not doing

**Training or fine-tuning any model** (§2) · 3D Human Mesh Recovery · a temporal
Transformer/LSTM step classifier (a rule-based FSM is more defensible for flight software:
explainable and auditable) · RTSP (MJPEG satisfies "stream to a specific IP") ·
multi-astronaut · multi-camera fusion · cloud anything.

**The documented next step after the hackathon,** if a judge asks what is missing: collect
~150 webcam frames of the five props, label them, and fine-tune `yolo11n` for 100 epochs at
`imgsz 640` behind the same `contracts.ObjectDetector` interface — a drop-in swap, no other
code changes. That is the honest gap, stated precisely, with the recipe attached.

---

## 12. Contract change log

| Date | Version | Change | By |
|---|---|---|---|
| 2026-09-02 | 1.0.0 | Initial freeze: `Detection`, `ObjectTrack`, `Wrist`, `FrameEvidence`, `StepEvent`, `ProtocolSpec`, `StepSpec`, `Zone`, `UiStatus`; seams `ObjectDetector`/`EventSink`/`Speaker`/`FrameSource`. | — |

---

## Appendix A — Interface specification

**If it is here, do not ask anybody.** If you need something absent, that is a contract
change: post it in the group chat and log it in §12.

### Person A — `har/protocol/`

```python
# spec.py  (A2)
class ProtocolError(ValueError): ...
def load_protocol(path: str | Path, frame_size: tuple[int, int]) -> ProtocolSpec

# predicates.py  (A3) — identical signature for all six
def object_stable(ev: FrameEvidence, spec: ProtocolSpec, step: StepSpec, st: PredicateState) -> bool
def object_left_zone(ev, spec, step, st) -> bool
def hoi_cycle(ev, spec, step, st) -> bool
def settled(ev, spec, step, st) -> bool
def transfer(ev, spec, step, st) -> bool
def hands_clear(ev, spec, step, st) -> bool
PREDICATES: dict[str, Callable[[FrameEvidence, ProtocolSpec, StepSpec, PredicateState], bool]]

@dataclass
class PredicateState:                     # per-step mutable counters, owned by the validator
    satisfied_frames: int = 0
    hoi_seen: set[str] = field(default_factory=set)
    last_box: BBox | None = None
    initial_box: BBox | None = None
    hands_seen_in_zone: bool = False      # hands_clear latch: never vacuously true on an empty scene

# validator.py  (A4, A7)
class SequenceValidator:
    def __init__(self, spec: ProtocolSpec) -> None
    def update(self, evidence: FrameEvidence) -> list[StepEvent]   # may be empty
    def status(self) -> UiStatus
    def reset(self) -> None
    @property
    def current(self) -> StepSpec | None
    @property
    def completed_steps(self) -> tuple[str, ...]
    @property
    def violations(self) -> tuple[str, ...]
    @property
    def finished(self) -> bool
```

### Person B — `har/perception/`

```python
# color_detector.py  (B2) — implements contracts.ObjectDetector
class ColorDetector:
    def __init__(self, ranges: Mapping[str, tuple[tuple[int,int,int], tuple[int,int,int]]],
                 roi: BBox | None = None, median_window: int = 5, min_area: int = 400) -> None
    def detect(self, frame: Any) -> list[Detection]
    @property
    def backend(self) -> str                       # "hsv"

# pose.py  (B3)
class WristExtractor:
    def __init__(self, weights: str | Path, conf: float = 0.35, every_n_frames: int = 1) -> None
    def wrists(self, frame: Any, frame_index: int) -> list[Wrist]
        # On a skipped frame return the previous result — never an empty list.

# perception.py  (B4)
class PerceptionStack:
    def __init__(self, detector: ObjectDetector, wrists: WristExtractor,
                 labels: Sequence[str], frame_size: tuple[int, int],
                 tracker_config: TrackerConfig | None = None,
                 interaction_config: InteractionConfig | None = None) -> None
    def process(self, frame: Any, frame_index: int, t_rel: float) -> FrameEvidence

# rack.py  (B7)
class RackFrame:
    def __init__(self, fiducials: Sequence[Point], rack_size: tuple[float, float]) -> None
    def to_rack(self, box: BBox) -> BBox
    def ready(self) -> bool
```

### Person C — `har/out/`, `har/ui/`, `har/app.py`

```python
# out/eventlog.py  (C2) — implements contracts.EventSink
class JsonlEventLog:
    def __init__(self, jsonl_path: str | Path, csv_path: str | Path | None = None) -> None
    def emit(self, event: StepEvent) -> None       # flush per event
    def close(self) -> None

# out/speaker.py  (C3) — implements contracts.Speaker
class OfflineSpeaker:
    def __init__(self, rate: int = 165, enabled: bool = True) -> None
    def say(self, text: str, priority: int = 0) -> None    # drop, never block
    def stop(self) -> None

# out/recorder.py  (C7)
class VideoRecorder:
    def __init__(self, path: str | Path, frame_size: tuple[int, int],
                 fps: float = 15.0, fourcc: str = "mp4v") -> None
    def write(self, frame: Any) -> None
    def close(self) -> None

# out/streamer.py  (C8)
class MjpegStreamer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None
    def publish(self, frame: Any) -> None          # latest-frame-only, non-blocking
    def latest_jpeg(self) -> bytes | None
    def shutdown(self) -> None

# ui/web.py  (C9)
def create_app(streamer: MjpegStreamer,
               status_provider: Callable[[], UiStatus],
               log_tail: Callable[[int], list[StepEvent]]) -> "flask.Flask"
    # Routes: GET /   GET /stream   GET /status   GET /events?n=20

# ui/overlay.py  (C10)
def draw_hud(frame: Any, status: UiStatus, evidence: FrameEvidence) -> None   # in place

# app.py  (C5)
def build_arg_parser() -> argparse.ArgumentParser
def main(argv: Sequence[str] | None = None) -> int
```

### The CLI surface (C owns it, A and B consume it)

```
--source 0|PATH        camera index or video file        (default 0)
--protocol PATH        protocols/pts01.yaml
--detector color|yolo  detector backend                  (default color)
--colours PATH         config/colours.yaml
--out-dir DIR          runs/<timestamp>/                 (default runs/latest)
--headless             no GUI, no window, exit at end of source
--no-voice             disable TTS
--record / --no-record write recordings/run_<ts>.mp4
--stream-host HOST     0.0.0.0
--stream-port PORT     8080
--pose-every-n N       run pose on every Nth frame       (default 1)
--imgsz N              480
--conf F               0.45
--max-frames N         stop after N frames (0 = no limit)
--contract             print CONTRACT_VERSION and exit 0
```

`--out-dir` always contains exactly `events.jsonl`, `events.csv`, `meta.json` (source,
protocol id and version, contract version, detector backend, fps, frame count, start/end
ISO-8601) and, with `--record`, the mp4. A9's evaluator and C9's GUI both read these paths —
do not invent alternatives.

---

## Appendix B — repo history

Everything removed in the restructure is recoverable from the initial commit:

```bash
git show 19c5436 --stat                    # what used to be here
git show 19c5436:bottle_monitor.py         # reference webcam loop + debug overlay
git show 19c5436:pickup_detectionv_4.py    # reference tuning constants
```

The 10 legacy camera scripts (5,127 of the original 5,788 Python lines) were
version-control-by-copy-paste; their own window titles read `"SIH26174 - Step 6B/6D/6E"`,
which were *development* steps, not experiment steps. The reusable arithmetic from
`bottle_monitor.py` now lives in `har/perception/` and is still under test. The frame loop
and debug overlay are recoverable from the commit above when C writes `har/app.py` (C5) and
`har/ui/overlay.py` (C10).

## Appendix C — quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover
```

Collects **37 tests**: 28 run in a bare interpreter (no cv2, no torch, no numpy) and 9 skip
without PyYAML. Verified 2026-09-02 on Python 3.11.2: `Ran 37 tests ... OK (skipped=9)`
without PyYAML, `Ran 37 tests ... OK` with it.
