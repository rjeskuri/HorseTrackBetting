"""Settings from the process environment, else from the repo-root `.env` file.

No dependency: `.env` is `NAME=value` lines; blank lines and `#` comments are skipped and the value
is everything after the first `=`. One pair of matching surrounding quotes (`"..."` or `'...'`) is
stripped from a `.env` value, as python-dotenv does; an inner quote is kept. A process-environment
value is returned verbatim, and `.env` is never copied into the process environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """The directory holding pyproject.toml, found by walking up from this package."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("repo root not found: no pyproject.toml above the horsetrack package")


def _unquote(value: str) -> str:
    """Drop one pair of matching surrounding quotes, if the value has one."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def get_setting(name: str) -> str | None:
    """The process environment wins; else the `.env` file at the repo root; else None."""
    value = os.environ.get(name)
    if value is not None:
        return value
    try:
        text = (repo_root() / ".env").read_text()
    except FileNotFoundError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition("=")
        if sep and key.strip() == name:
            return _unquote(val)
    return None
