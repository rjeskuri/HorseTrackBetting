#!/usr/bin/env python3
"""Guard the public tree: no data, licensed files, secrets or personal data in git.

    python3 scripts/check_public_tree.py

Checks every file `git ls-files` lists in the repository, prints one line per violation
(`path: rule: detail`) and exits 1, or prints nothing and exits 0; 2 when git cannot list the
files. `tests/test_public_tree.py` runs the same checks on every pull request. Stdlib only.

Rules, by id:
  data      a tracked path under the root data/ directory
  fixture   a file under tests/fixtures/ outside tests/fixtures/synthetic/, or an .html/.htm
            there that does not start with the synthetic marker (see that folder's README)
  size      a file over 256 KB, except uv.lock (whose content is still scanned)
  jrdb      a JRDB-style data file name: three capital letters, six digits, .txt
  secret    a private-key header, or a GitHub, AWS or Slack token
  personal  a home path, an email address other than the two noreply ones, a claude.ai
            artifact link

A matched secret or personal value is shown only as its first 4 characters then ***. Every
pattern is assembled from parts, so this file never matches itself.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass

DATA, FIXTURE, SIZE, JRDB, SECRET, PERSONAL = (
    "data",
    "fixture",
    "size",
    "jrdb",
    "secret",
    "personal",
)

LIMIT = 256 * 1024
SIZE_EXEMPT = "uv.lock"
FIXTURES = "tests/fixtures/"
SYNTHETIC = FIXTURES + "synthetic/"
MARKER = b"<!-- synthetic:"
JRDB_NAME = re.compile(r"[A-Z]{3}[0-9]{6}\.txt")

# Every repetition is bounded or anchored on a literal prefix, so each scan stays linear.
SECRETS = (
    ("private key", re.compile(r"-{4,5} ?BEGIN (?:[A-Z0-9]{1,20} ){0,4}" + "PRIV" + "ATE KEY")),
    ("GitHub token", re.compile("gh" + r"[pousr]_[A-Za-z0-9]{30,}")),
    ("GitHub token", re.compile("github" + r"_pat_[A-Za-z0-9_]{22,}")),
    ("AWS access key", re.compile("AK" + r"IA[A-Z0-9]{16}")),
    ("Slack token", re.compile("xo" + r"x[abeprs]-[A-Za-z0-9-]{10,}")),
)
HOME_PATH = re.compile("/(?:Us" + "ers|ho" + r"me)/[A-Za-z0-9._-]{1,64}")
ARTIFACT_URL = re.compile(
    "claude" + r"\.ai/(?:[\w-]{1,32}/){0,3}artifacts?/[\w-]{8,}", re.IGNORECASE
)

# Emails are found from each "@": the local part is read backwards (at most 64 characters), so
# a long alphanumeric run costs nothing unless it touches an "@".
LOCAL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._%+-")
DOMAIN = re.compile(r"(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,63}")
GH_NOREPLY_DOMAIN = "users.noreply" + ".github.com"
ANTHROPIC_NOREPLY = ("noreply", "anthropic" + ".com")


@dataclass(frozen=True)
class Violation:
    path: str
    rule: str
    detail: str


def check(paths: list[str], read: Callable[[str], bytes]) -> list[Violation]:
    """Every violation in `paths`, each file's bytes coming from `read(path)`."""
    out: list[Violation] = []
    for path in paths:
        data = read(path)
        out.extend(_path_violations(path, data))
        out.extend(_content_violations(path, data))
    return out


def _path_violations(path: str, data: bytes) -> Iterator[Violation]:
    if path.startswith("data/"):
        yield Violation(path, DATA, "tracked under data/; data lives under $HORSETRACK_DATA_DIR")
    if path.startswith(FIXTURES):
        if not path.startswith(SYNTHETIC):
            yield Violation(path, FIXTURE, "fixture outside " + SYNTHETIC)
        elif path.lower().endswith((".html", ".htm")) and not data.startswith(MARKER):
            yield Violation(path, FIXTURE, "html fixture does not start with the synthetic marker")
    if path != SIZE_EXEMPT and len(data) > LIMIT:
        yield Violation(path, SIZE, f"{len(data)} bytes, over the {LIMIT}-byte limit")
    if JRDB_NAME.fullmatch(path.rsplit("/", 1)[-1]):
        yield Violation(path, JRDB, "JRDB-style data file name")


def _content_violations(path: str, data: bytes) -> Iterator[Violation]:
    text = data.decode("utf-8", errors="replace")
    line, last = 1, 0
    for pos, rule, label, value in sorted(_matches(text)):
        line += text.count("\n", last, pos)
        last = pos
        yield Violation(path, rule, f"line {line}: {label} {_mask(value)}")


def _matches(text: str) -> Iterator[tuple[int, str, str, str]]:
    """(offset, rule, label, matched text) for each secret or personal value in `text`."""
    for label, pattern in SECRETS:
        for m in pattern.finditer(text):
            yield m.start(), SECRET, label, m.group()
    for m in HOME_PATH.finditer(text):
        yield m.start(), PERSONAL, "home path", m.group()
    for m in ARTIFACT_URL.finditer(text):
        yield m.start(), PERSONAL, "claude.ai artifact link", m.group()
    for start, local, domain in _emails(text):
        if not _allowed_email(local, domain):
            yield start, PERSONAL, "email address", local + "@" + domain


def _emails(text: str) -> Iterator[tuple[int, str, str]]:
    at = text.find("@")
    while at != -1:
        start = at
        while start > 0 and at - start < 64 and text[start - 1] in LOCAL_CHARS:
            start -= 1
        if start < at:
            m = DOMAIN.match(text, at + 1)
            if m:
                yield start, text[start:at], m.group()
        at = text.find("@", at + 1)


def _allowed_email(local: str, domain: str) -> bool:
    domain = domain.lower()
    return domain == GH_NOREPLY_DOMAIN or (local.lower(), domain) == ANTHROPIC_NOREPLY


def _mask(value: str) -> str:
    return value[:4] + "***"


def _read_tracked(path: str) -> bytes:
    """A file's bytes; a symlink's target (what git stores); nothing for a missing entry."""
    if os.path.islink(path):
        return os.fsencode(os.readlink(path))
    if not os.path.isfile(path):
        return b""
    with open(path, "rb") as f:
        return f.read()


def main() -> int:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True
        ).stdout
        os.chdir(os.fsdecode(top.rstrip(b"\n")))
        listed = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"check_public_tree: cannot list the tracked files: {exc}", file=sys.stderr)
        return 2
    paths = [os.fsdecode(p) for p in listed.split(b"\0") if p]
    violations = check(paths, _read_tracked)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    for v in violations:
        print(f"{v.path}: {v.rule}: {v.detail}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
