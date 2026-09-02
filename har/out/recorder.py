"""Video recorder (C7) — deliverable D5's "store locally" half.

``VideoRecorder`` wraps ``cv2.VideoWriter`` with the niceties a demo run
needs: parent directories are created, frame sizes are validated (a size
mismatch produces a corrupted file instead of an error on most backends, so
it is rejected loudly here), and ``close`` is idempotent and reports what was
actually written.

Frames are expected to arrive with the HUD already drawn (``har.ui.overlay``)
so the stored video is self-explaining when a judge opens it in a stock
player.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["VideoRecorder"]


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:  # pragma: no cover - depends on machine
        raise RuntimeError("VideoRecorder needs opencv (pip install opencv-python)") from exc


class VideoRecorder:
    """Record frames to an mp4 with a stock-player-friendly codec."""

    def __init__(
        self,
        path: str | Path,
        frame_size: tuple[int, int],
        fps: float = 15.0,
        fourcc: str = "mp4v",
    ) -> None:
        cv2 = _require_cv2()
        self.path = Path(path)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.fps = float(fps)
        self.fourcc = fourcc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*fourcc), self.fps, self.frame_size
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"VideoWriter failed to open {self.path}")
        self._frames = 0
        self._closed = False

    def write(self, frame: Any) -> None:
        """Append one BGR frame of exactly ``frame_size``."""
        if self._closed:
            raise RuntimeError("VideoRecorder.write() after close()")
        shape = getattr(frame, "shape", None)
        if shape is not None and (int(shape[1]), int(shape[0])) != self.frame_size:
            raise ValueError(
                f"frame is {int(shape[1])}x{int(shape[0])}, recorder expects {self.frame_size}"
            )
        self._writer.write(frame)
        self._frames += 1

    def close(self) -> None:
        """Finalise the file.  Idempotent; safe on the shutdown path."""
        if self._closed:
            return
        self._closed = True
        self._writer.release()
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError(f"recording {self.path} was not written")

    @property
    def frames_written(self) -> int:
        return self._frames

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"VideoRecorder({self.path!s}, {self._frames} frames, {self.fps:g} fps)"
