"""Offline text-to-speech (C3) — deliverable D3, the voice alert.

``OfflineSpeaker`` implements the frozen ``contracts.Speaker`` seam on top of
``pyttsx3`` — fully offline TTS (SAPI5 on Windows, ``espeak-ng`` on Linux;
no network, no ground station, per the problem statement).

The two rules that keep the frame loop honest:

* **Never block.**  Speech runs on a single daemon thread behind a one-deep
  queue.  ``say`` is a bounded-time enqueue: when the queue is occupied the
  new utterance is *dropped* (counted in ``dropped``), never waited on.  A
  violation burst at 30 fps therefore cannot stall video, and the C6 proof
  is 100 ``say`` calls in a tight loop finishing in milliseconds.
* **Degrade silently.**  ``enabled=False`` (the CLI's ``--no-voice``) makes
  every call a no-op, and if the pyttsx3 driver cannot initialise on this
  machine (no ``espeak-ng``, no audio device) the speaker disables itself
  once and keeps accepting — and dropping — calls instead of raising.  The
  §9 fallback (on-screen banner instead of voice) stays available because
  the caller never has to care.

Priority is a drop policy, not a queue reorder: a queued low-priority
utterance is *evicted* by an incoming higher-priority one, so a
``SKIPPED`` alert is not left waiting behind a routine step prompt.

The module is standard-library only at import time; pyttsx3 is imported
inside the worker thread.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable

from har.contracts import StepEvent  # noqa: F401  (docstring reference only)

__all__ = ["OfflineSpeaker"]

_STOP = object()  # sentinel that drains the worker


class OfflineSpeaker:
    """One-deep, drop-don't-block, offline TTS speaker.

    ``engine_factory`` is a test seam (keyword-only, so the frozen Appendix A
    signature is unchanged): pass a zero-arg callable returning a pyttsx3-like
    engine (``setProperty`` / ``say`` / ``runAndWait`` / ``stop``) to drive
    the speaker without pyttsx3 or audio hardware.
    """

    def __init__(
        self,
        rate: int = 165,
        enabled: bool = True,
        *,
        engine_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.rate = int(rate)
        self.enabled = bool(enabled)
        self._engine_factory = engine_factory
        #: one-deep per the plan; the queue holds (priority, sequence, text)
        self._queue: queue.Queue[tuple[int, int, Any]] = queue.Queue(maxsize=1)
        self._sequence = 0
        self._dropped = 0
        self._spoken = 0
        self._stopped = False
        self._available: bool | None = None  # None until the worker decides
        self.init_error: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._run, name="har-speaker", daemon=True
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # contracts.Speaker
    # ------------------------------------------------------------------

    def say(self, text: str, priority: int = 0) -> None:
        """Speak ``text`` when the engine gets to it.  Bounded-time, never blocks.

        A higher ``priority`` evicts a *queued* (not yet speaking) lower one;
        anything currently being spoken runs to the end of its utterance.
        """
        if not self.enabled or self._stopped or not text:
            return
        with self._lock:
            self._sequence += 1
            item = (int(priority), self._sequence, str(text))
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                pass
            try:
                queued = self._queue.get_nowait()
            except queue.Empty:  # worker grabbed it between the two calls
                return
            if item[0] > queued[0]:
                # evict the lower-priority utterance in favour of this one
                self._dropped += 1
                try:
                    self._queue.put_nowait(item)
                    return
                except queue.Full:  # pragma: no cover - single-threaded here
                    pass
            else:
                # keep the queued one; drop the newcomer (requeue if it won)
                try:
                    self._queue.put_nowait(queued)
                except queue.Full:  # pragma: no cover
                    pass
            self._dropped += 1

    def stop(self) -> None:
        """Drain the queue and stop the engine.  Idempotent; safe without start."""
        if self._stopped:
            return
        self._stopped = True
        if self._thread is None:
            return
        while True:
            try:
                self._queue.get_nowait()  # make room for the sentinel
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((2**31 - 1, -1, _STOP))
                break
            except queue.Full:  # pragma: no cover - we just drained it
                continue
        self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True once the engine has initialised.  False after a driver failure.

        Starts False for a disabled speaker; may briefly read False on a live
        one until the worker thread has finished importing pyttsx3 — check
        ``init_error`` to tell "starting up" from "failed".
        """
        return bool(self.enabled and self._available)

    @property
    def dropped(self) -> int:
        """Utterances dropped because the queue was occupied (by design)."""
        return self._dropped

    def wait_ready(self, timeout: float = 2.0) -> bool:
        """Block (briefly, at startup) until the engine has spoken its fate.

        Returns True when the engine is usable, False on driver failure —
        check ``init_error``.  Never wait on this in the frame loop; it is
        for the CLI to print the actual TTS situation once, before the run.
        """
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.enabled or self._available is not None:
                break
            time.sleep(0.02)
        return self.available

    @property
    def spoken(self) -> int:
        """Utterances actually handed to the engine."""
        return self._spoken

    # ------------------------------------------------------------------
    # worker
    # ------------------------------------------------------------------

    def _make_engine(self) -> Any:
        if self._engine_factory is not None:
            return self._engine_factory()
        import pyttsx3  # late import: pyttsx3 is optional (plan §9 fallback)

        return pyttsx3.init()

    def _run(self) -> None:
        engine = None
        try:
            engine = self._make_engine()
            engine.setProperty("rate", self.rate)
            self._available = True
        except Exception as exc:  # driver missing, no audio device, ...
            self._available = False
            self.init_error = f"{type(exc).__name__}: {exc}"
        while True:
            try:
                _priority, _seq, text = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stopped:
                    break
                continue
            if text is _STOP:
                break
            if engine is None:
                self._dropped += 1
                continue
            try:
                engine.say(text)
                engine.runAndWait()
                self._spoken += 1
            except Exception as exc:  # engine died mid-run: disable, keep draining
                self._available = False
                self.init_error = f"{type(exc).__name__}: {exc}"
                engine = None
        if engine is not None:
            try:
                engine.stop()
            except Exception:  # pragma: no cover - best effort on shutdown
                pass

    def __enter__(self) -> "OfflineSpeaker":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        if not self.enabled:
            return "OfflineSpeaker(disabled)"
        state = "available" if self.available else "unavailable"
        return f"OfflineSpeaker(rate={self.rate}, {state}, spoken={self._spoken}, dropped={self._dropped})"
