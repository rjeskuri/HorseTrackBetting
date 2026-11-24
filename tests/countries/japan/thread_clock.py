"""A fake clock with one virtual time per thread (ticket #38): for tests of logging two sources at
once, each on a thread of its own.

Each thread's `now()` starts at `start`, and only that thread's `sleep(s)` moves its time. So a
source run on its own thread lives through its own day exactly as it would alone, whatever the
other thread does, and one thread running every source in turn would see one day go by once.
Nothing really waits. A busy loop, a negative wait or a run still sleeping 24 h after `start` ends
the thread with `Runaway` (a BaseException) and is noted in `errors`.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta

EPS = 1e-6


class Runaway(BaseException):
    """A run kept going where a day cannot (BaseException: no `except Exception` hides it)."""


class ThreadClock:
    def __init__(self, start: datetime, *, hours: float = 24.0):
        self.start = start
        self.limit = start + timedelta(hours=hours)
        self.errors: list[str] = []
        self._local = threading.local()

    @property
    def t(self) -> datetime:
        """The calling thread's time."""
        return getattr(self._local, "t", self.start)

    def _fail(self, message: str) -> None:
        self.errors.append(message)
        raise Runaway(message)

    def now(self) -> datetime:
        reads = getattr(self._local, "reads", 0) + 1
        self._local.reads = reads
        if reads > 200_000:
            self._fail("now() read 200000 times without a sleep: a busy loop")
        return self.t

    def sleep(self, seconds: float) -> None:
        seconds = float(seconds)
        if seconds < -EPS:
            self._fail(f"sleep({seconds}): a negative wait")
        self._local.reads = 0
        zeros = getattr(self._local, "zeros", 0)
        self._local.zeros = zeros + 1 if seconds <= 0 else 0
        if self._local.zeros > 10_000:
            self._fail("10000 zero-length sleeps in a row")
        self._local.t = self.t + timedelta(seconds=max(seconds, 0.0))
        if self._local.t > self.limit:
            self._fail(f"still sleeping at {self._local.t.isoformat()}: the run never returned")
