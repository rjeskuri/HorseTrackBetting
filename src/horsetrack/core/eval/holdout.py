"""The locked final holdout per country, and the log of every use (#71).

``split_off`` divides a frame into development rows and a ``LockedHoldout``. The holdout is
defined in ``<countries_dir>/<country>/config/holdout.yaml`` (with the default root, read through
``horsetrack.core.config.load_country_yaml(country, "holdout")``)::

    holdout:
      start: 2026-11-28
      seed: "<hex string>"

A race day (its local date) on or after ``start`` is holdout when
``int(sha256(f"{seed}:{YYYY-MM-DD}").hexdigest(), 16) % 2 == 0``; every other day is
development. A missing or invalid definition raises ``HoldoutConfigError``; there is no default.

The holdout rows are reached only through ``LockedHoldout.open(ticket=..., purpose=...)``, which
appends one JSON line to ``<countries_dir>/<country>/holdout_log.jsonl`` before it returns them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from horsetrack.core.config import load_country_yaml

DEFAULT_COUNTRIES_DIR = Path(__file__).resolve().parents[2] / "countries"
CONFIG_STEM = "holdout"
CONFIG_NAME = Path("config") / f"{CONFIG_STEM}.yaml"
LOG_NAME = "holdout_log.jsonl"

_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


class HoldoutConfigError(ValueError):
    """The country's holdout definition is missing or invalid."""


def _is_holdout_day(seed: str, day: str) -> bool:
    return int(hashlib.sha256(f"{seed}:{day}".encode()).hexdigest(), 16) % 2 == 0


def _read_config(countries_dir: Path, country: str) -> tuple[Any, Path]:
    """The parsed holdout.yaml of ``country`` and its path; the default root goes through the
    shared ``load_country_yaml`` (cached: the result is only read, never mutated)."""
    config_path = countries_dir / country / CONFIG_NAME
    try:
        if countries_dir.resolve() == DEFAULT_COUNTRIES_DIR.resolve():
            return load_country_yaml(country, CONFIG_STEM), config_path
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        return yaml.safe_load(config_path.read_text(encoding="utf-8")), config_path
    except OSError as exc:
        raise HoldoutConfigError(
            f"no holdout definition: {config_path} cannot be read ({exc})"
        ) from exc
    except yaml.YAMLError as exc:
        raise HoldoutConfigError(f"{config_path} is not valid YAML: {exc}") from exc


def _load_definition(countries_dir: Path, country: str) -> tuple[date, str]:
    config, config_path = _read_config(countries_dir, country)
    section = config.get("holdout") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        raise HoldoutConfigError(f"{config_path} has no holdout section")

    start = section.get("start")
    if isinstance(start, str) and _ISO_DAY.fullmatch(start.strip()):
        try:
            start = date.fromisoformat(start.strip())
        except ValueError:
            pass
    if not isinstance(start, date) or isinstance(start, datetime):
        raise HoldoutConfigError(
            f"{config_path}: holdout.start must be a YYYY-MM-DD date, got {start!r}"
        )

    seed = section.get("seed")
    if not isinstance(seed, str) or not seed.strip():
        raise HoldoutConfigError(f"{config_path}: holdout.seed must be a non-empty string")
    return start, seed


def _race_days(frame: pd.DataFrame, date_col: str) -> pd.Series:
    """Each row's race day as YYYY-MM-DD, from naive local dates, timestamps or strings."""
    stamps = pd.to_datetime(frame[date_col])
    if stamps.dt.tz is not None:
        raise ValueError(f"{date_col!r} is timezone-aware; pass the local race date (naive)")
    if stamps.isna().any():
        raise ValueError(f"{date_col!r} has missing dates; every row needs a race day")
    days = stamps.dt.normalize()
    labels = {day: day.strftime("%Y-%m-%d") for day in days.unique()}
    return days.map(labels)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot read the git commit for the holdout log") from exc
    return result.stdout.strip()


class LockedHoldout:
    """The holdout rows of one country, reachable only through ``open``."""

    __slots__ = ("__rows", "__log_path", "__country", "__start", "__seed", "__n_days")

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        log_path: Path,
        country: str,
        start: date,
        seed: str,
        n_days: int,
    ) -> None:
        self.__rows = rows
        self.__log_path = log_path
        self.__country = country
        self.__start = start
        self.__seed = seed
        self.__n_days = n_days

    def open(self, *, ticket: int, purpose: str) -> pd.DataFrame:
        """Log this use of the holdout, then return its rows.

        One JSON line (utc, ticket, purpose, git_sha, n_rows, period) is appended and synced to
        the country's holdout log before any row is returned; if it cannot be written, no rows
        are returned.
        """
        if not isinstance(ticket, int) or isinstance(ticket, bool):
            raise TypeError(f"ticket must be an issue number (int), got {ticket!r}")
        if ticket <= 0:
            raise ValueError(f"ticket must be a positive issue number, got {ticket!r}")
        if not isinstance(purpose, str):
            raise TypeError(f"purpose must be a string, got {purpose!r}")
        if not purpose.strip():
            raise ValueError("purpose must say why the holdout is opened")

        record = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "ticket": ticket,
            "purpose": purpose,
            "git_sha": _git_sha(),
            "n_rows": len(self.__rows),
            "period": {
                "start": self.__start.isoformat(),
                "seed_sha256": hashlib.sha256(self.__seed.encode()).hexdigest(),
            },
        }
        line = json.dumps(record) + "\n"
        with open(self.__log_path, "a", encoding="utf-8") as log:
            log.write(line)
            log.flush()
            os.fsync(log.fileno())
        return self.__rows.copy()

    def __reduce__(self):
        raise TypeError("a LockedHoldout cannot be pickled or copied; use open()")

    def __repr__(self) -> str:
        return (
            f"LockedHoldout(country={self.__country!r}, start={self.__start.isoformat()}, "
            f"n_rows={len(self.__rows)}, n_days={self.__n_days})"
        )


def split_off(
    frame: pd.DataFrame,
    date_col: str,
    country: str,
    *,
    countries_dir: Path | str = DEFAULT_COUNTRIES_DIR,
) -> tuple[pd.DataFrame, LockedHoldout]:
    """Return the development rows and the country's locked holdout.

    Raises ``HoldoutConfigError`` when the country has no valid holdout definition.
    """
    country_dir = Path(countries_dir) / country
    start, seed = _load_definition(Path(countries_dir), country)

    days = _race_days(frame, date_col)
    start_label = start.isoformat()
    held_days = {d for d in days.unique() if d >= start_label and _is_holdout_day(seed, d)}
    mask = days.isin(held_days).to_numpy()

    development = frame.loc[~mask].copy()
    holdout = LockedHoldout(
        frame.loc[mask].copy(),
        log_path=country_dir / LOG_NAME,
        country=country,
        start=start,
        seed=seed,
        n_days=len(held_days),
    )
    return development, holdout
