"""Fail the image build if the resolved torch build does not match the channel.

The Dockerfile installs ``torch``/``torchvision`` from
``--index-url $TORCH_INDEX_URL``.  If anyone adds PyPI back as an
``--extra-index-url`` (torch's own dependencies live there), pip is free to
prefer the PyPI build — which on Linux x86_64 is the CUDA one, and would
silently ship a multi-gigabyte image that then cannot find a GPU, or the
opposite mistake on a GPU box.  So the builder stage runs this immediately
after the install and fails the build instead of finding out at the demo.

Usage::

    python docker/check_torch.py cpu      # assert no CUDA in the build
    python docker/check_torch.py cuda     # assert CUDA is in the build
"""

from __future__ import annotations

import sys
from typing import Sequence

__all__ = ["check", "main"]


def check(channel: str, cuda_version: str | None, torch_version: str) -> str | None:
    """Return an error message when ``cuda_version`` contradicts ``channel``."""
    if channel not in ("cpu", "cuda"):
        return f"unknown channel {channel!r} (expected 'cpu' or 'cuda')"
    if channel == "cpu" and cuda_version:
        return (f"channel cpu but torch {torch_version} was built for CUDA {cuda_version}; "
                f"pass --index-url https://download.pytorch.org/whl/cpu (no PyPI extra index)")
    if channel == "cuda" and not cuda_version:
        return (f"channel cuda but torch {torch_version} is a CPU-only build; "
                f"point TORCH_INDEX_URL at the matching CUDA wheel index")
    return ""


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    channel = argv[0] if argv else "cpu"
    try:
        import torch
    except ImportError as exc:
        print(f"cannot import torch in the builder stage: {exc}", file=sys.stderr)
        return 2

    problem = check(channel, torch.version.cuda, torch.__version__)
    if problem:
        print(f"docker/check_torch.py: {problem}", file=sys.stderr)
        return 1
    print(f"torch {torch.__version__} — cuda={torch.version.cuda or 'none'} — channel {channel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
