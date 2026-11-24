"""Ticket #71: the locked final holdout per country, and the log of every use.

Contract pinned here (the coder implements what these tests call):

- ``split_off(frame, date_col, country, *, countries_dir=...)`` returns
  ``(development, LockedHoldout)``. ``countries_dir`` is the one seam for the
  countries root; by default it is the package's ``src/horsetrack/countries``
  (the root ``horsetrack.core.config.load_country_yaml`` reads). The config is
  ``<countries_dir>/<country>/config/holdout.yaml`` (one file per eval ticket;
  ``evaluation.yaml`` is not read). Hong Kong's country id is ``hongkong``::

      holdout:
        start: 2026-11-28        # quoted or unquoted YAML date
        seed: <hex string>

- A race day (YYYY-MM-DD) on or after ``start`` is holdout when
  ``int(sha256(f"{seed}:{day}").hexdigest(), 16) % 2 == 0``; every other day is
  development.
- A missing file, holdout section, seed or start (or a null/invalid one) raises
  ``HoldoutConfigError``; there is never a default.
- ``LockedHoldout.open(*, ticket, purpose)`` appends one JSON line to
  ``<countries_dir>/<country>/holdout_log.jsonl`` with ``utc``, ``ticket``,
  ``purpose``, ``git_sha``, ``n_rows`` and ``period`` (a mapping with at least
  ``start`` as YYYY-MM-DD, and something that identifies the seed) before it
  returns the rows. Nothing else on the object reaches the rows.

Every synthetic test points ``countries_dir`` at ``tmp_path``; one test loads the
committed definitions through the default root without ever calling ``open``.
An autouse guard fails any test that touches a real
``src/horsetrack/countries/*/holdout_log.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from horsetrack.core.eval.holdout import HoldoutConfigError, LockedHoldout, split_off

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"

SEED_A = "3f9c2a7be1d04c55a8e6f0b19d7c4e21"
SEED_B = "b7e1d0c9a8f746352e1f0a9b8c7d6e5f"
SEED_C = "0123456789abcdef0123456789abcdef"
JAPAN_START = "2026-11-28"


# --------------------------------------------------------------------------- helpers


def is_holdout_day(day: date, start: date, seed: str) -> bool:
    """The owner's rule, computed independently of the code under test."""
    if day < start:
        return False
    digest = hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest()
    return int(digest, 16) % 2 == 0


def write_config(
    root: Path,
    country: str,
    *,
    start: str | None = JAPAN_START,
    seed: str | None = SEED_A,
    raw: str | None = None,
) -> Path:
    config_dir = root / country / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    if raw is None:
        raw = f'holdout:\n  start: "{start}"\n  seed: "{seed}"\n'
    path = config_dir / "holdout.yaml"
    path.write_text(raw)
    return path


def make_frame(
    first: str = "2026-10-01",
    n_days: int = 180,
    date_kind: str = "datetime",
) -> pd.DataFrame:
    """Synthetic races: 1 to 3 rows per day, a unique row_id, a marker column."""
    rows = []
    day0 = date.fromisoformat(first)
    row_id = 0
    for i in range(n_days):
        day = day0 + timedelta(days=i)
        for race in range(1 + i % 3):
            rows.append(
                {
                    "row_id": row_id,
                    "race_date": datetime(day.year, day.month, day.day, 10 + race, 35),
                    "horse": f"SECRETHORSE{row_id:05d}",
                    "speed": 50.0 + (row_id * 7919) % 1000 / 10.0,
                }
            )
            row_id += 1
    frame = pd.DataFrame(rows)
    if date_kind == "datetime":
        frame["race_date"] = pd.to_datetime(frame["race_date"]).dt.normalize()
    elif date_kind == "datetime_with_time":
        frame["race_date"] = pd.to_datetime(frame["race_date"])
    elif date_kind == "string":
        frame["race_date"] = pd.to_datetime(frame["race_date"]).dt.strftime("%Y-%m-%d")
    else:  # pragma: no cover - test bug
        raise AssertionError(date_kind)
    return frame


def race_days(series: pd.Series) -> list[date]:
    return list(pd.to_datetime(series).dt.date)


def expected_ids(frame: pd.DataFrame, start: str, seed: str) -> tuple[set[int], set[int]]:
    start_day = date.fromisoformat(start)
    holdout, development = set(), set()
    for row_id, day in zip(frame["row_id"], race_days(frame["race_date"]), strict=True):
        (holdout if is_holdout_day(day, start_day, seed) else development).add(int(row_id))
    return development, holdout


def log_path(root: Path, country: str) -> Path:
    return root / country / "holdout_log.jsonl"


def log_lines(root: Path, country: str) -> list[str]:
    path = log_path(root, country)
    if not path.exists():
        return []
    return path.read_text().splitlines()


def repo_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture(autouse=True)
def real_logs_untouched():
    """No test may create or change a real country holdout log."""

    def snapshot() -> dict[Path, bytes]:
        if not REAL_COUNTRIES.exists():
            return {}
        return {p: p.read_bytes() for p in REAL_COUNTRIES.glob("*/holdout_log.jsonl")}

    before = snapshot()
    yield
    assert snapshot() == before, "a test wrote to a real holdout_log.jsonl"


# --------------------------------------------------------------------------- partition


PARTITION_CASES = [
    pytest.param(SEED_A, JAPAN_START, "datetime", id="seedA-japan-start-datetime"),
    pytest.param(SEED_B, JAPAN_START, "string", id="seedB-japan-start-strings"),
    pytest.param(SEED_C, JAPAN_START, "datetime_with_time", id="seedC-japan-start-timestamps"),
    pytest.param(SEED_A, "2027-01-15", "datetime", id="seedA-later-start"),
    pytest.param(SEED_B, "2026-10-20", "string", id="seedB-early-start"),
]


@pytest.mark.parametrize(("seed", "start", "date_kind"), PARTITION_CASES)
def test_development_never_contains_holdout_dates(tmp_path, seed, start, date_kind):
    write_config(tmp_path, "japan", start=start, seed=seed)
    frame = make_frame(date_kind=date_kind)
    original = frame.copy(deep=True)
    exp_dev, exp_hold = expected_ids(frame, start, seed)
    # The fixture must exercise both sides of the rule, or the test proves nothing.
    assert exp_hold and any(
        d >= date.fromisoformat(start)
        for d, i in zip(race_days(frame["race_date"]), frame["row_id"], strict=True)
        if int(i) in exp_dev
    )

    development, holdout = split_off(frame, "race_date", "japan", countries_dir=tmp_path)

    assert isinstance(development, pd.DataFrame)
    assert isinstance(holdout, LockedHoldout)
    start_day = date.fromisoformat(start)
    leaked = [d for d in race_days(development["race_date"]) if is_holdout_day(d, start_day, seed)]
    assert leaked == []
    assert set(development["row_id"].astype(int)) == exp_dev
    assert len(development) == len(exp_dev)
    assert list(development.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(frame, original)  # the input frame is not mutated


@pytest.mark.parametrize(("seed", "start", "date_kind"), PARTITION_CASES)
def test_open_returns_exactly_the_holdout_days(tmp_path, seed, start, date_kind):
    write_config(tmp_path, "japan", start=start, seed=seed)
    frame = make_frame(date_kind=date_kind)
    exp_dev, exp_hold = expected_ids(frame, start, seed)

    development, holdout = split_off(frame, "race_date", "japan", countries_dir=tmp_path)
    rows = holdout.open(ticket=71, purpose="partition check")

    assert isinstance(rows, pd.DataFrame)
    assert set(rows["row_id"].astype(int)) == exp_hold
    assert len(rows) == len(exp_hold)
    assert list(rows.columns) == list(frame.columns)
    assert set(rows["row_id"]).isdisjoint(set(development["row_id"]))
    assert len(rows) + len(development) == len(frame)


def test_days_before_start_are_always_development(tmp_path):
    write_config(tmp_path, "japan", start="2030-01-01", seed=SEED_A)
    frame = make_frame()

    development, holdout = split_off(frame, "race_date", "japan", countries_dir=tmp_path)

    assert set(development["row_id"]) == set(frame["row_id"])
    rows = holdout.open(ticket=71, purpose="nothing assigned yet")
    assert len(rows) == 0
    (line,) = log_lines(tmp_path, "japan")
    assert json.loads(line)["n_rows"] == 0


def test_each_country_uses_its_own_definition_and_log(tmp_path):
    write_config(tmp_path, "japan", start=JAPAN_START, seed=SEED_A)
    write_config(tmp_path, "hongkong", start="2026-12-10", seed=SEED_B)
    frame = make_frame()

    jp_dev, jp_hold = split_off(frame, "race_date", "japan", countries_dir=tmp_path)
    hk_dev, hk_hold = split_off(frame, "race_date", "hongkong", countries_dir=tmp_path)

    exp_jp_dev, exp_jp_hold = expected_ids(frame, JAPAN_START, SEED_A)
    exp_hk_dev, exp_hk_hold = expected_ids(frame, "2026-12-10", SEED_B)
    assert exp_jp_hold != exp_hk_hold
    assert set(jp_dev["row_id"]) == exp_jp_dev
    assert set(hk_dev["row_id"]) == exp_hk_dev

    hk_rows = hk_hold.open(ticket=72, purpose="hk check")
    assert set(hk_rows["row_id"]) == exp_hk_hold
    assert log_lines(tmp_path, "japan") == []
    (hk_line,) = log_lines(tmp_path, "hongkong")
    assert json.loads(hk_line)["n_rows"] == len(exp_hk_hold)


# --------------------------------------------------------------------------- open and the log


MISSING_TICKET_OR_PURPOSE = [
    pytest.param({}, id="neither"),
    pytest.param({"purpose": "final evaluation"}, id="no-ticket"),
    pytest.param({"ticket": 71}, id="no-purpose"),
    pytest.param({"ticket": None, "purpose": "final evaluation"}, id="ticket-None"),
    pytest.param({"ticket": 0, "purpose": "final evaluation"}, id="ticket-zero"),
    pytest.param({"ticket": -5, "purpose": "final evaluation"}, id="ticket-negative"),
    pytest.param({"ticket": "", "purpose": "final evaluation"}, id="ticket-empty"),
    pytest.param({"ticket": 71, "purpose": None}, id="purpose-None"),
    pytest.param({"ticket": 71, "purpose": ""}, id="purpose-empty"),
    pytest.param({"ticket": 71, "purpose": "   "}, id="purpose-blank"),
]


@pytest.mark.parametrize("kwargs", MISSING_TICKET_OR_PURPOSE)
def test_open_requires_a_ticket_and_a_purpose(tmp_path, kwargs):
    write_config(tmp_path, "japan")
    _, holdout = split_off(make_frame(), "race_date", "japan", countries_dir=tmp_path)

    with pytest.raises((TypeError, ValueError)):
        holdout.open(**kwargs)
    assert log_lines(tmp_path, "japan") == []

    # A refused open leaves the holdout usable with a proper ticket and purpose.
    rows = holdout.open(ticket=71, purpose="final evaluation")
    assert len(rows) > 0
    assert len(log_lines(tmp_path, "japan")) == 1


def test_open_is_keyword_only(tmp_path):
    write_config(tmp_path, "japan")
    _, holdout = split_off(make_frame(), "race_date", "japan", countries_dir=tmp_path)

    with pytest.raises(TypeError):
        holdout.open(71, "final evaluation")
    assert log_lines(tmp_path, "japan") == []


@pytest.mark.parametrize(
    ("seed", "start", "ticket", "purpose"),
    [
        (SEED_A, JAPAN_START, 71, "final evaluation"),
        (SEED_B, "2027-01-15", 133, "calibration re-check"),
    ],
)
def test_open_appends_one_log_line(tmp_path, seed, start, ticket, purpose):
    write_config(tmp_path, "japan", start=start, seed=seed)
    frame = make_frame()
    _, exp_hold = expected_ids(frame, start, seed)
    _, holdout = split_off(frame, "race_date", "japan", countries_dir=tmp_path)
    assert log_lines(tmp_path, "japan") == []  # split_off alone logs nothing

    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    rows = holdout.open(ticket=ticket, purpose=purpose)
    after = datetime.now(timezone.utc) + timedelta(seconds=1)

    raw = log_path(tmp_path, "japan").read_text()
    assert raw.endswith("\n")
    lines = raw.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert {"utc", "ticket", "purpose", "git_sha", "n_rows", "period"} <= set(record)
    assert record["ticket"] == ticket
    assert record["purpose"] == purpose
    assert record["n_rows"] == len(rows) == len(exp_hold)
    assert record["git_sha"] == repo_head()
    stamp = datetime.fromisoformat(record["utc"].replace("Z", "+00:00"))
    assert stamp.utcoffset() == timedelta(0)
    assert before <= stamp <= after
    assert isinstance(record["period"], dict)
    assert record["period"]["start"] == start


def test_the_logged_period_identifies_the_rule(tmp_path):
    """Same start, different seed: the logged period must tell them apart."""
    periods = []
    for country, seed in (("japan", SEED_A), ("hongkong", SEED_B)):
        write_config(tmp_path, country, start=JAPAN_START, seed=seed)
        _, holdout = split_off(make_frame(), "race_date", country, countries_dir=tmp_path)
        holdout.open(ticket=71, purpose="period identity")
        (line,) = log_lines(tmp_path, country)
        periods.append(json.loads(line)["period"])
    assert periods[0]["start"] == periods[1]["start"] == JAPAN_START
    assert periods[0] != periods[1]


def test_the_log_is_append_only(tmp_path):
    write_config(tmp_path, "japan")
    earlier = '{"utc": "2026-11-29T00:00:00+00:00", "ticket": 1, "purpose": "earlier"}\n'
    log = log_path(tmp_path, "japan")
    log.write_text(earlier)
    frame = make_frame()

    _, first = split_off(frame, "race_date", "japan", countries_dir=tmp_path)
    _, second = split_off(frame, "race_date", "japan", countries_dir=tmp_path)
    opens = [
        (first, 71, "first look"),
        (first, 72, "second look, same object"),
        (second, 73, "third look, new split"),
    ]

    previous = log.read_bytes()
    for n, (holdout, ticket, purpose) in enumerate(opens, start=1):
        holdout.open(ticket=ticket, purpose=purpose)
        current = log.read_bytes()
        assert current.startswith(previous), "an earlier line was rewritten"
        added = current[len(previous) :].decode()
        assert added.endswith("\n") and added.count("\n") == 1
        assert json.loads(added)["ticket"] == ticket
        assert len(current.decode().splitlines()) == 1 + n
        previous = current
    assert log.read_text().splitlines()[0] == earlier.rstrip("\n")


def test_open_returns_no_rows_when_the_log_cannot_be_written(tmp_path):
    write_config(tmp_path, "japan")
    _, holdout = split_off(make_frame(), "race_date", "japan", countries_dir=tmp_path)
    log_path(tmp_path, "japan").mkdir()  # a directory where the log file should be

    with pytest.raises(OSError):
        holdout.open(ticket=71, purpose="log is unwritable")


# --------------------------------------------------------------------------- the lock


ROW_LIKE = (pd.DataFrame, pd.Series, pd.Index, np.ndarray)


def exposes_rows(value) -> bool:
    if isinstance(value, ROW_LIKE):
        return True
    if isinstance(value, dict):
        return any(exposes_rows(v) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(exposes_rows(v) for v in value)
    return False


def test_rows_are_not_reachable_without_open(tmp_path):
    write_config(tmp_path, "japan")
    _, holdout = split_off(make_frame(), "race_date", "japan", countries_dir=tmp_path)

    public = {name: getattr(holdout, name) for name in dir(holdout) if not name.startswith("_")}
    callables = {name for name, value in public.items() if callable(value)}
    assert callables == {"open"}
    for name, value in public.items():
        if name != "open":
            assert not exposes_rows(value), f"public attribute {name!r} exposes data"

    with pytest.raises(TypeError):
        iter(holdout)
    for key in (0, "race_date", slice(None)):
        with pytest.raises((TypeError, KeyError, IndexError, NotImplementedError)):
            holdout[key]

    text = repr(holdout) + str(holdout)
    assert "SECRETHORSE" not in text
    assert log_lines(tmp_path, "japan") == []  # inspecting the object is not a use


@pytest.mark.parametrize("n_days", [60, 150])
def test_repr_shows_counts_only(tmp_path, n_days):
    write_config(tmp_path, "japan", start="2026-10-01", seed=SEED_C)
    frame = make_frame(n_days=n_days)
    _, exp_hold = expected_ids(frame, "2026-10-01", SEED_C)
    _, holdout = split_off(frame, "race_date", "japan", countries_dir=tmp_path)

    text = repr(holdout)
    assert str(len(exp_hold)) in text
    assert "SECRETHORSE" not in text
    for speed in frame["speed"].head(20):
        assert f"{speed:.1f}" not in text
    assert log_lines(tmp_path, "japan") == []


# --------------------------------------------------------------------------- the definition


MISSING_DEFINITIONS = [
    pytest.param(None, id="no-config-file"),
    pytest.param("", id="empty-file"),
    pytest.param("other: 1\n", id="no-holdout-section"),
    pytest.param("holdout:\n", id="holdout-null"),
    pytest.param(f'holdout:\n  seed: "{SEED_A}"\n', id="no-start"),
    pytest.param(f'holdout:\n  start: null\n  seed: "{SEED_A}"\n', id="start-null"),
    pytest.param(f'holdout:\n  start: "not-a-date"\n  seed: "{SEED_A}"\n', id="start-invalid"),
    pytest.param(f'holdout:\n  start: "{JAPAN_START}"\n', id="no-seed"),
    pytest.param(f'holdout:\n  start: "{JAPAN_START}"\n  seed: null\n', id="seed-null"),
    pytest.param(f'holdout:\n  start: "{JAPAN_START}"\n  seed: ""\n', id="seed-empty"),
]


@pytest.mark.parametrize("raw", MISSING_DEFINITIONS)
def test_a_missing_holdout_period_is_an_error(tmp_path, raw):
    if raw is None:
        (tmp_path / "hongkong").mkdir()
    else:
        write_config(tmp_path, "hongkong", raw=raw)

    with pytest.raises(HoldoutConfigError):
        split_off(make_frame(), "race_date", "hongkong", countries_dir=tmp_path)
    assert log_lines(tmp_path, "hongkong") == []


def test_an_unquoted_yaml_start_date_is_read(tmp_path):
    write_config(tmp_path, "japan", raw=f"holdout:\n  start: {JAPAN_START}\n  seed: {SEED_A}\n")
    frame = make_frame()
    exp_dev, _ = expected_ids(frame, JAPAN_START, SEED_A)

    development, _ = split_off(frame, "race_date", "japan", countries_dir=tmp_path)

    assert set(development["row_id"]) == exp_dev


def test_a_config_named_evaluation_yaml_is_not_read(tmp_path):
    """The definition lives in holdout.yaml only; the old evaluation.yaml name is ignored."""
    config_dir = tmp_path / "japan" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "evaluation.yaml").write_text(
        f'holdout:\n  start: "{JAPAN_START}"\n  seed: "{SEED_A}"\n'
    )

    with pytest.raises(HoldoutConfigError):
        split_off(make_frame(), "race_date", "japan", countries_dir=tmp_path)
    assert log_lines(tmp_path, "japan") == []


# --------------------------------------------------------------------------- committed definitions


HEX32 = re.compile(r"[0-9a-f]{32}")


def committed_holdout(country: str) -> dict:
    """The committed definition, read straight from the YAML, not through the code under test."""
    path = REAL_COUNTRIES / country / "config" / "holdout.yaml"
    return yaml.safe_load(path.read_text())["holdout"]


def as_day(value) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def test_the_committed_japan_definition_loads_by_default():
    """Pins the pre-registered A2 definition: start 2026-11-28 and a 32-hex seed."""
    definition = committed_holdout("japan")
    seed = definition["seed"]
    assert isinstance(seed, str) and HEX32.fullmatch(seed)
    assert as_day(definition["start"]) == date(2026, 11, 28)

    frame = make_frame(first="2026-11-14", n_days=40)
    exp_dev, exp_hold = expected_ids(frame, JAPAN_START, seed)
    assert exp_hold and len(exp_dev) > len(frame[frame["race_date"] < JAPAN_START])

    development, holdout = split_off(frame, "race_date", "japan")  # default countries root

    assert set(development["row_id"].astype(int)) == exp_dev
    assert len(development) == len(exp_dev)
    assert str(len(exp_hold)) in repr(holdout)  # counts only; the holdout is never opened


def test_the_committed_hongkong_definition_has_no_start_yet():
    definition = committed_holdout("hongkong")
    assert definition["start"] is None
    assert isinstance(definition["seed"], str) and HEX32.fullmatch(definition["seed"])

    with pytest.raises(HoldoutConfigError):
        split_off(make_frame(), "race_date", "hongkong")  # default countries root
