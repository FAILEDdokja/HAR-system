"""MJPEG frame streamer (C8) — deliverable D5's "stream to a specific IP" half.

``MjpegStreamer`` is the frame buffer between the frame loop and the HTTP
layer (``har.ui.web`` serves the bytes).  Three rules from the plan are
enforced here:

* **Latest-frame-only.**  There is exactly one slot.  ``publish`` overwrites
  it; a slow browser watches the feed fall behind and then jump to "now" —
  it never drags the pipeline back.
* **Never queue, never block.**  ``publish`` does its JPEG encode and one
  short lock hold; there is no queue to grow and no consumer to wait on, so
  a stalled client cannot stall video processing.
* **Bind ``0.0.0.0``.**  ``host``/``port`` are recorded for the web layer to
  bind; the default is the LAN-visible address, never ``127.0.0.1``.

Consumers use :meth:`mjpeg_stream` (a ``multipart/x-mixed-replace`` byte
generator driven by a condition variable — a 15 fps feed costs no polling),
or :meth:`latest_jpeg` for one-shot grabs.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Iterator

__all__ = ["MjpegStreamer"]


def _jpeg_encode(frame: Any, quality: int) -> bytes:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on machine
        raise RuntimeError("MjpegStreamer needs opencv (pip install opencv-python)") from exc
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buffer.tobytes()


class MjpegStreamer:
    """One-slot latest-frame buffer plus MJPEG packaging."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, *, jpeg_quality: int = 80) -> None:
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._version = 0
        self._published = 0
        self._shutdown = False
        # A unique boundary per instance keeps two test servers in one
        # process from producing indistinguishable streams.
        self._boundary = f"frame{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # producer side
    # ------------------------------------------------------------------

    def publish(self, frame: Any) -> None:
        """Offer a new frame.  Bounded-time; after shutdown it is a no-op."""
        if self._shutdown:
            return
        jpeg = _jpeg_encode(frame, self.jpeg_quality)
        with self._condition:
            if self._shutdown:
                return
            self._jpeg = jpeg
            self._version += 1
            self._published += 1
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # consumer side
    # ------------------------------------------------------------------

    @property
    def boundary(self) -> str:
        return self._boundary

    @property
    def frames_published(self) -> int:
        return self._published

    def latest_jpeg(self) -> bytes | None:
        """The most recent frame as JPEG bytes, or None before the first."""
        with self._condition:
            return self._jpeg

    def wait_jpeg(self, after_version: int = 0, timeout: float = 5.0) -> tuple[int, bytes | None]:
        """Block until a frame newer than ``after_version`` exists.

        Returns ``(version, jpeg)``; ``jpeg`` None means the timeout elapsed
        or the streamer shut down.  One waiter costs exactly one wake per
        published frame — no busy loop.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._shutdown and self._version <= after_version:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return self._version, self._jpeg

    def mjpeg_stream(self, *, idle_timeout: float = 30.0) -> Iterator[bytes]:
        """Yield one multipart chunk per published frame until shutdown."""
        version = 0
        head = (
            f"--{self._boundary}\r\nContent-Type: image/jpeg\r\n\r\n".encode("ascii")
        )
        while not self._shutdown:
            version, jpeg = self.wait_jpeg(after_version=version, timeout=idle_timeout)
            if jpeg is None:
                if self._shutdown:
                    break
                continue  # idle camera: keep the connection alive
            yield head + jpeg + b"\r\n"

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Wake every consumer and stop accepting frames.  Idempotent."""
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        return self._shutdown

    def __repr__(self) -> str:
        state = "shutdown" if self._shutdown else f"{self._published} frames"
        return f"MjpegStreamer({self.host}:{self.port}, {state})"
