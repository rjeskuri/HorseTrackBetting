"""Shared pytest setup: the `local` marker's `real_fixtures` fixture.

This file is copied verbatim into the inner pytester project of tests/test_local_marker.py, so it
imports only the standard library, pytest and `horsetrack`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from horsetrack.core import paths

pytest_plugins = ["pytester"]


@pytest.fixture
def real_fixtures() -> Path:
    try:
        path = paths.fixtures_dir()
    except paths.DataDirNotSet:
        pytest.skip(f"{paths.DATA_DIR_VAR} is not set: real-page fixtures unavailable")
    if not path.is_dir():
        pytest.skip(f"real-page fixtures directory missing: {path}")
    return path
