"""Settings from the process environment, else from the repo-root `.env` file.

No dependency: `.env` is `NAME=value` lines; blank lines and `#` comments are skipped and the value
is everything after the first `=`, verbatim.
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
            return val
    return None
