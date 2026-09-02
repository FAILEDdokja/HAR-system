#!/usr/bin/env bash
# Fill the wheelhouse for offline installation (C11).
#
# Run this ONCE, on a machine with network, before travelling to the venue:
#
#     bash wheelhouse/download.sh
#
# Then, on the disconnected demo machine:
#
#     python3 -m venv .venv
#     .venv/bin/pip install --no-index --find-links wheelhouse/ -r requirements.lock
#
# The wheels pinned here target CPython 3.11 / Linux x86_64 (the lock's
# platform).  For a Windows demo laptop, re-run this script on that OS —
# pip resolves platform-matching wheels automatically.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec pip download \
    --requirement "$HERE/../requirements.lock" \
    --dest "$HERE" \
    --quiet --progress-bar off
