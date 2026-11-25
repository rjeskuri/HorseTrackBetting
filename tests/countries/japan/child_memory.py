"""Peak RSS of a child process, counting only the memory the child itself touched.

Why not `ru_maxrss` everywhere: on Linux a process's `ru_maxrss` keeps the high-water mark of the
address space it replaced at execve (the kernel folds the old mm's peak into `signal->maxrss`),
so a child spawned by a pytest process that has run a few thousand tests reports the parent's
peak, not its own. Per platform:

- Linux: `VmHWM` from `/proc/self/status`. It belongs to the mm, which execve builds new (zeroed),
  so it starts at the child's own interpreter; the kernel keeps it as for `ru_maxrss`
  (max of the recorded mark and the current RSS, the mark refreshed before every unmap).
  If `/proc/self/status` is missing, unreadable or has no `VmHWM`, the child exits non-zero
  with the reason and the test fails: CI runs on Linux, so it must never pass unmeasured.
- macOS: `ru_maxrss` (bytes). It does not carry the parent's peak across fork and exec
  (checked: a child of a parent that touched 700 MB reports ~9 MB).
- Anywhere else: no method, the test is skipped with the platform named.

Each child also reports `start`, its peak before importing anything heavy; `run_child` fails when
that is above STARTUP_LIMIT_BYTES, which is what an inherited mark would look like.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

# a bare CPython 3 interpreter peaks at ~10 MB; an inherited pytest mark is hundreds of MB
STARTUP_LIMIT_BYTES = 60 * 1000 * 1000

PROBE = r'''
import sys

def peak_bytes():
    """This process's own peak RSS in bytes (see tests/countries/japan/child_memory.py)."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="ascii") as status:
                lines = status.read().splitlines()
        except OSError as exc:
            sys.exit(f"UNMEASURABLE: cannot read /proc/self/status: {exc!r}")
        for line in lines:
            if line.startswith("VmHWM:"):
                value, unit = line.split()[1:3]
                if unit != "kB":
                    sys.exit(f"UNMEASURABLE: unexpected VmHWM unit: {line!r}")
                return int(value) * 1024  # the kernel's kB is KiB
        sys.exit("UNMEASURABLE: /proc/self/status has no VmHWM line")
    if sys.platform == "darwin":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    sys.exit(f"UNMEASURABLE: no peak-RSS method on {sys.platform}")

START_PEAK = peak_bytes()
'''


def method() -> str | None:
    if sys.platform.startswith("linux"):
        return "VmHWM"
    if sys.platform == "darwin":
        return "ru_maxrss"
    return None


def run_child(code: str, *args: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run `code` after PROBE in a fresh interpreter. `code` calls `peak_bytes()` and prints one
    JSON object as its last line; the result gains `start` (the child's peak at startup)."""
    if method() is None:
        pytest.skip(f"no way to measure a child's own peak RSS on {sys.platform}")
    done = subprocess.run(
        [sys.executable, "-c", PROBE + code, *args],
        capture_output=True,
        text=True,
        timeout=170,
        check=False,
        env=env,
    )
    if "UNMEASURABLE:" in done.stderr:
        pytest.fail(done.stderr.strip().splitlines()[-1])
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["start"] < STARTUP_LIMIT_BYTES, (
        f"the child's peak RSS is {out['start'] / 1e6:.0f} MB before it does anything: "
        f"{method()} is counting memory the child did not touch"
    )
    return out
