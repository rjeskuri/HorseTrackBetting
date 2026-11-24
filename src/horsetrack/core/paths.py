"""The data directory and the on-disk layout under it (kept from HKTrack).

Everything licensed or scraped lives under HORSETRACK_DATA_DIR, outside the repository. The
directory is read on every call, never cached, so a changed environment takes effect at once.
"""

from __future__ import annotations

from pathlib import Path

from horsetrack.core import env

DATA_DIR_VAR = "HORSETRACK_DATA_DIR"


class DataDirNotSet(RuntimeError):
    pass


class DataDirInsideRepo(RuntimeError):
    pass


def data_dir() -> Path:
    """The resolved data directory, from the environment or `.env`; never inside the repo."""
    raw = env.get_setting(DATA_DIR_VAR)
    if not raw or not raw.strip():
        raise DataDirNotSet(f"{DATA_DIR_VAR} is not set: export it or add it to .env")
    path = Path(raw.strip()).expanduser().resolve()
    root = env.repo_root().resolve()
    if path.is_relative_to(root):
        raise DataDirInsideRepo(f"{DATA_DIR_VAR}={path} is inside the repository at {root}")
    return path


def raw_dir(source: str) -> Path:
    return data_dir() / "raw" / source


def raw_page_path(source: str, page_type: str, year: int, key: str) -> Path:
    return raw_dir(source) / page_type / f"{year:04d}" / f"{key}.html.gz"


def manifest_path(source: str) -> Path:
    return raw_dir(source) / "manifest.sqlite"


def interim_dir(source: str) -> Path:
    return data_dir() / "interim" / source


def processed_dir(source: str) -> Path:
    return data_dir() / "processed" / source


def odds_snapshot_path(source: str, race_date: str, venue: str) -> Path:
    return data_dir() / "odds_snapshots" / source / race_date / f"{venue}.jsonl.gz"


def logs_dir(country: str) -> Path:
    return data_dir() / "logs" / country


def reports_dir() -> Path:
    return data_dir() / "reports"


def fixtures_dir() -> Path:
    return data_dir() / "fixtures"
