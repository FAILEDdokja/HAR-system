# SIH26174 — AI Human Activity Recognition for On-board BAS Experiments

**Problem statement:** SIH26174, Indian Space Research Organisation (ISRO), Smart India
Hackathon 2026, Software / Smart Automation.

An **offline** system that watches a fixed-payload camera, tracks which step of a
pre-defined experiment the astronaut is on, announces the next step, speaks up when a step
is skipped or performed out of sequence, and leaves a timestamped log plus a stored and
streamed video — with no ground station in the loop.

**Protocol implemented:** `PTS-01 — Payload Tray Sorting & Sample Transfer`, 8 steps.

---

## Start here

| If you want to… | Read |
|---|---|
| Understand the task and what we must deliver | [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) §1 |
| Know what to build today | [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) §4–§6, your track only |
| Know which files are yours | [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) §2 |
| See the data contracts between tracks | [`har/contracts.py`](har/contracts.py) |
| See the experiment procedure | [`protocols/pts01.yaml`](protocols/pts01.yaml) |
| See how the repo got here | [`docs/POC_PLAN.md`](docs/POC_PLAN.md) |

## Layout

```
har/
├── contracts.py       frozen cross-track data contracts (stdlib only)
├── app.py             CLI entrypoint                      [Track C]
├── perception/        detection, pose, tracking, HOI      [Track B]
├── protocol/          protocol model, sequence validator  [Track A]
├── out/               event log, recorder, streamer, TTS  [Track C]
└── ui/                in-frame overlay, browser GUI       [Track C]
protocols/             PTS-01 procedure definition
config/                tracker tuning
models/                YOLO weights
tests/                 unit tests + JSON fixtures
tools/                 dataset capture, synthetic video, replay
```

Tracks own disjoint file sets, so three people work in parallel without merge conflicts.
`har/contracts.py` and `protocols/pts01.yaml` are the only shared files.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover
```

The suite collects 37 tests: **28 run in a bare interpreter** (no cv2, no torch) and 9 skip
without PyYAML. `har/contracts.py`, `har/protocol/**` and `tests/**` are deliberately
forbidden from importing cv2, numpy, torch or ultralytics — an `ast` test fails the build
if that rule is broken. It is what lets the sequence validator be developed and proven
without a camera.

## Runtime (once Phase 1 lands)

```bash
# Replay a recording, headless, and inspect the log
.venv/bin/python -m har.app --source demo/correct.mp4 --headless --out-dir runs/demo
cat runs/demo/events.jsonl

# Live camera with GUI, voice, recording and streaming
.venv/bin/python -m har.app --source 0 --detector yolo --record \
    --stream-host 0.0.0.0 --stream-port 8080
```

## Status

See [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) §1.3 for a verified inventory of
what exists, what is tested, and what is still missing. In short: the perception primitives
and the protocol definition are in place and tested; the sequence validator, the detector
for the protocol's objects, and the entire output/interface layer are the next build.
