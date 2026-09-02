"""Derive the container install set from ``requirements.lock`` (D-deps).

``requirements.lock`` is the frozen demo-machine set (C11).  On Linux x86_64
``torch`` resolves there to the CUDA build, so the lock also pins the NVIDIA
runtime wheels and ``triton`` — roughly 2.5 GB of binaries a CPU container has
no use for.  Hand-maintaining a second requirements file for the image would
drift from the lock the first time someone bumps a pin, so the Dockerfile
filters the lock at build time instead::

    python tools/docker_requirements.py --channel cpu requirements.lock

Channels
--------
``cpu`` (default)
    Drop ``torch``/``torchvision`` — the Dockerfile installs those from the
    CPU wheel index — plus every CUDA-only distribution the lock pins
    (``nvidia-*``, ``cuda-*``, ``triton``).
``cuda``
    Drop only ``torch``/``torchvision`` (installed from the matching CUDA
    wheel index) and keep the NVIDIA runtime wheels, because the lock's
    ``nvidia-*-cu13`` pins are exactly what that build links against.

Everything else is passed through unchanged, so the container installs the
versions the lock froze and nothing else.  One substitution is applied: the
GUI ``opencv-python`` build becomes ``opencv-python-headless`` at the same
pinned version, because a container has no display and the codebase opens no
window — that is what lets the image skip ``libgl1``.  ``--gui-opencv`` turns
the substitution off.
  ``--torch-only`` prints just the
torch pins (for the separate wheel-index install step) and ``--check``
validates the result without printing it, which is what the unit test and CI
use to catch a lock bump that silently reintroduces a CUDA wheel.

Standard library only — Track A's rule applies to anything the tests import.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "CUDA_ONLY_PREFIXES",
    "TORCH_DISTRIBUTIONS",
    "REQUIRED_TOP_LEVEL",
    "parse_pins",
    "filter_pins",
    "torch_pins",
    "channel_problems",
    "main",
]

#: Distributions that exist only to serve a CUDA build of torch.
CUDA_ONLY_PREFIXES: tuple[str, ...] = ("nvidia-", "cuda-", "triton")

#: Installed from the wheel index (``--index-url``) rather than from the lock's
#: PyPI resolution, so they are always removed from the generated set.
TORCH_DISTRIBUTIONS: tuple[str, ...] = ("torch", "torchvision")

#: ``opencv-python`` links ``libGL`` and the X11 client libraries; a container
#: has no display server and nothing in ``har/`` or ``tools/`` opens a window
#: (no ``imshow`` / ``waitKey`` anywhere), so the image installs the headless
#: build *of the same pinned version* and drops the ``libgl1`` apt dependency.
GUI_TO_HEADLESS: dict[str, str] = {"opencv-python": "opencv-python-headless"}

#: The top-level packages ``requirements.txt`` declares; a generated set that
#: lost one of these is a bug, not a size optimisation.
REQUIRED_TOP_LEVEL: tuple[str, ...] = (
    "ultralytics",
    "opencv-python",
    "numpy",
    "pyyaml",
    "pyttsx3",
    "flask",
)

_PIN_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<extras>\[[^\]]*\])?\s*(?P<spec>[<>=!~].*)$")


def _normalise(name: str) -> str:
    """PEP 503 normalisation, so ``PyYAML`` and ``pyyaml`` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(lines: Iterable[str]) -> list[tuple[str, str]]:
    """Return ``(normalised name, raw pin line)`` for every pinned requirement.

    Comments and blank lines are skipped; anything that is not an exact pin
    (``name==version``) is reported by :func:`channel_problems` rather than
    silently dropped.
    """
    pins: list[tuple[str, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_RE.match(line)
        if match is None:
            continue
        pins.append((_normalise(match.group("name")), line))
    return pins


def _is_cuda_only(name: str) -> bool:
    return name.startswith(CUDA_ONLY_PREFIXES)


def filter_pins(pins: Sequence[tuple[str, str]], channel: str, headless: bool = True) -> list[str]:
    """The lock's pins to install for ``channel`` (torch handled separately).

    ``headless`` swaps the GUI OpenCV build for its headless twin, keeping the
    pinned version — see :data:`GUI_TO_HEADLESS`.
    """
    out: list[str] = []
    seen: set[str] = set()
    for name, line in pins:
        if name in TORCH_DISTRIBUTIONS:
            continue
        if channel == "cpu" and _is_cuda_only(name):
            continue
        if name in seen:  # the lock lists opencv-python twice (top-level + transitive)
            continue
        seen.add(name)
        if headless and name in GUI_TO_HEADLESS:
            replacement = GUI_TO_HEADLESS[name]
            line = line.replace(name, replacement, 1)
            name = _normalise(replacement)
        out.append(line)
    return out


def torch_pins(pins: Sequence[tuple[str, str]]) -> list[str]:
    """The ``torch``/``torchvision`` pins, installed from the wheel index."""
    return [line for name, line in pins if name in TORCH_DISTRIBUTIONS]


def unparsed_lines(lines: Iterable[str]) -> list[str]:
    """Requirement lines that are neither comments, blanks nor exact pins."""
    bad: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _PIN_RE.match(line) is None:
            bad.append(line)
    return bad


def channel_problems(lines: Sequence[str], channel: str, headless: bool = True) -> list[str]:
    """Consistency problems with ``lines`` for ``channel``; empty means clean."""
    problems: list[str] = []
    if channel not in ("cpu", "cuda"):
        problems.append(f"unknown channel {channel!r} (expected 'cpu' or 'cuda')")
        return problems

    pins = parse_pins(lines)
    names = {name for name, _ in pins}

    problems.extend(f"unpinned requirement: {line}" for line in unparsed_lines(lines))

    for dist in TORCH_DISTRIBUTIONS:
        if dist not in names:
            problems.append(f"the lock file no longer pins {dist}")

    kept = {_normalise(_PIN_RE.match(line).group("name")) for line in filter_pins(pins, channel, headless)}
    for required in REQUIRED_TOP_LEVEL:
        expected = _normalise(GUI_TO_HEADLESS.get(required, required) if headless else required)
        if expected not in kept:
            problems.append(f"top-level package {required} missing from the generated set")

    if channel == "cpu":
        cuda = sorted(name for name in kept if _is_cuda_only(name))
        if cuda:
            problems.append(f"CUDA-only wheels left in the cpu set: {', '.join(cuda)}")
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docker_requirements",
        description="Print the container install set derived from requirements.lock.",
    )
    parser.add_argument("lock", nargs="?", default="requirements.lock",
                        help="path to the frozen lock file (default: requirements.lock)")
    parser.add_argument("--channel", choices=("cpu", "cuda"), default="cpu",
                        help="cpu: drop the CUDA wheels; cuda: keep them (default: cpu)")
    parser.add_argument("--torch-only", action="store_true",
                        help="print only the torch/torchvision pins (wheel-index step)")
    parser.add_argument("--gui-opencv", action="store_true",
                        help="keep the GUI opencv-python build instead of the headless twin")
    parser.add_argument("--check", action="store_true",
                        help="validate the lock for this channel and print nothing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.lock)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2

    headless = not args.gui_opencv
    problems = channel_problems(lines, args.channel, headless)
    if problems:
        for problem in problems:
            print(f"{path}: {problem}", file=sys.stderr)
        return 1
    if args.check:
        return 0

    pins = parse_pins(lines)
    selected = torch_pins(pins) if args.torch_only else filter_pins(pins, args.channel, headless)
    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
