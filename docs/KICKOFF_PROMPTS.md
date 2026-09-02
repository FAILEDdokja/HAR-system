# Kickoff prompts — paste one per person

Copy the **shared preamble** plus **one person block** and send it. Nothing else needs
saying out loud; the rules in the preamble are the whole coordination protocol.

---

## SHARED PREAMBLE — send this to all three

> **SIH26174 — AI Human Activity Recognition for On-board BAS Experiments.**
> We are building a working, documented prototype by tomorrow.
>
> Repo: `KrSikchi/HAR-system`, branch `arena/01a062b8-har-system`. Pull it first.
> Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
> Baseline before you touch anything: `.venv/bin/python -m unittest discover` must print
> **37 tests, OK**. If it does not, stop and say so.
>
> Read `docs/DEVELOPMENT_PLAN.md` **§1, §2 and §3 only**, then work exclusively from your
> own numbered section. Every step has *Do* and *Done when* — you do not need to ask what
> finished means.
>
> **Hard rules:**
> 1. **We are not training anything.** No fine-tuning, no datasets, no `data.yaml`, no
>    labelling images. Protocol objects are detected by HSV colour. Details in §2.
> 2. **Only write inside the files you own** (§3). The ownership map is disjoint, so if you
>    stay in your lane git cannot produce a conflict. If you think you need someone else's
>    file, post in the group chat instead of editing it.
> 3. **`har/contracts.py` is frozen.** Never edit it. It is stdlib-only by design and an
>    `ast` test fails the build if that breaks.
> 4. **Exact signatures are in Appendix A.** If what you need is there, do not ask anybody.
>    If it is not there, that is a contract change — post it, do not improvise.
> 5. **Push green.** Run the test suite before every push; pull before you start.
> 6. **Gates at 12:15, 15:15 and 17:45** are the only times we sync. If something is not
>    done at a gate, it is cut from the demo — nobody debugs through a gate.
>
> Your steps are below. Do them in order. Start with step 1 immediately, because the other
> two people are waiting on it.

---

## PERSON A — Cognition

> You are **Person A**. You build the protocol model and the sequence validator — the part
> of this project that actually answers the problem statement.
>
> **Your section: `docs/DEVELOPMENT_PLAN.md` §5. Your steps: A1 → A10.**
>
> Your environment is **PyYAML only** — no cv2, no torch, no camera. You can finish and
> prove A1–A5 on a laptop with no hardware at all, and that is deliberate: you are never
> blocked on the other two.
>
> | Step | What | Phase |
> |---|---|---|
> | **A1** | Evidence fixtures — **do this first, the other two are waiting on it** | 1 |
> | A2 | Protocol loader `har/protocol/spec.py` | 1 |
> | A3 | The six predicates `har/protocol/predicates.py` | 1 |
> | A4 | `SequenceValidator` — implement the 8 semantics in §5 exactly | 1 |
> | A5 | Violation tests | 1 |
> | A6 | Threshold tuning on real footage (yaml only) | 2 |
> | A7 | `UiStatus` producer | 2 |
> | A8 | Record + hand-annotate three demo runs — this is our own dataset | 2 |
> | A9 | `tools/evaluate.py` → `docs/METRICS.md` | 3 |
> | A10 | Violation evidence table | 3 |
>
> **You own** `har/protocol/**`, `protocols/*.yaml`, `tools/evaluate.py`, `demo/`,
> `docs/METRICS.md`, and `tests/test_protocol_config.py` / `test_predicates.py` /
> `test_validator.py`. **Never touch** `har/perception/`, `har/out/`, `har/ui/`,
> `har/app.py` or `docs/PERF.md`.
>
> The single most important detail: your validator is **pure** — no clock reads, no file IO,
> no cv2. Time arrives via `evidence.t_rel`. That is what makes it testable without a
> camera, and it is why your half can be finished before the other two exist.
>
> Start A1 now.

---

## PERSON B — Perception

> You are **Person B**. You build the detector, the pose extraction and `FrameEvidence` —
> everything the validator is allowed to see.
>
> **Your section: `docs/DEVELOPMENT_PLAN.md` §6. Your steps: B1 → B9.**
>
> **Before you write any code, do this physical task:** tape or paint a **distinct colour on
> every prop** — black tray, yellow lid, red box, blue box, green vial — and print a small
> colour-key card to leave in frame as the white-balance reference. We detect by colour, so
> the props are ours to design. This costs ten minutes and it is the difference between B2
> taking 75 minutes and taking all day.
>
> | Step | What | Phase |
> |---|---|---|
> | B1 | Measure real FPS first — everything downstream is sized from it | 1 |
> | **B2** | HSV `ColorDetector` + `config/colours.yaml` | 1 |
> | B3 | `WristExtractor` (return the previous result on skipped frames, never empty) | 1 |
> | B4 | `PerceptionStack` → `FrameEvidence` | 1 |
> | B5 | Cross-check against A's validator — this is gate G1 | 1 |
> | B6 | Live camera tuning | 2 |
> | B7 | Rack-relative homography (the orientation-agnostic differentiator) | 3 |
> | B8 | The 90° rotation demo | 3 |
> | B9 | ONNX export — *optional, first thing cut* | 3 |
>
> **You own** `har/perception/**`, `config/colours.yaml`, `tools/probe_fps.py`,
> `docs/PERF.md`, and `tests/test_perception_*.py`. **Never touch** `har/protocol/`,
> `har/out/`, `har/ui/`, `har/app.py` or `docs/METRICS.md`.
>
> **Two things worth knowing:** no model changes are needed — both weights are already
> committed and correct, and `yolo11n-pose.pt` alone gives you person boxes *and* keypoints,
> so you do not need to load `yolo11n.pt` at all (§2). And **B4's output must match A1's
> fixture field for field — the fixture is the spec.** If they differ, change your code.
>
> You have ~105 min of slack in Phase 2 and you are the designated floater. Spend it on
> Person C's browser GUI. Do not invent new perception work to fill it.

---

## PERSON C — Output & interface

> You are **Person C**. You build everything a judge can see and hear: the log, the voice,
> the recording, the stream, the GUI, and the entrypoint that ties it together.
>
> **Your section: `docs/DEVELOPMENT_PLAN.md` §7. Your steps: C1 → C12.**
>
> | Step | What | Phase |
> |---|---|---|
> | **C1** | `tools/replay_events.py` — **do this first, it is your stub for Person A** | 1 |
> | C2 | `JsonlEventLog` (JSONL + CSV, flush per event) | 1 |
> | C3 | `OfflineSpeaker` — daemon thread, drop rather than block | 1 |
> | C4 | Synthetic video generator — gate G1 runs on these | 1 |
> | C5 | `har/app.py` CLI | 1 |
> | C6 | Output tests | 1 |
> | C7 | Video recorder | 2 |
> | C8 | MJPEG streamer — latest-frame-only, bind `0.0.0.0` | 2 |
> | C9 | Browser GUI — **the largest single task in the day** | 2 |
> | C10 | In-frame overlay HUD (the projector fallback) | 2 |
> | C11 | `requirements.lock` + wheelhouse, so "offline" is literal | 3 |
> | C12 | Rehearsal and deck | 3 |
>
> **You own** `har/out/**`, `har/ui/**`, `har/app.py`, `tools/replay_events.py`,
> `tools/make_synthetic_video.py`, `README.md`, `requirements.lock`, `wheelhouse/`, and
> `tests/test_out_*.py`. **Never touch** `har/perception/` or `har/protocol/`.
>
> **You are never blocked on the other two.** C1 gives you canned `StepEvent` objects, so
> C2 and C3 are fully buildable and testable before Person A has written the validator. For
> C5, build `main()` behind a `--stub` flag with a `StubPerception` and `StubValidator` that
> replay the fixtures, and wire the real components in the last 10 minutes before the gate.
>
> **`har/app.py` is the only file in your tree that may import `PerceptionStack` and
> `SequenceValidator`.** Nothing in `har/out/` or `har/ui/` may — pass the data in instead.
>
> Check today that the demo laptop's TTS actually speaks (pyttsx3 needs SAPI5 on Windows,
> `espeak-ng` on Linux). Do not discover this at 15:15.
