"""Ticket #126: gate a holdout opening on its family's power target, and roll a fresh block forward.

Problem statement section 5: D12 (opened only at a power target; after an opening a fresh block
rolls forward from that date), D15 (once per pre-registered family, every opening logged, a failed
claim counts as a use). The target values are owner decisions: every test here registers its own
families under a tmp ``countries_dir``; the committed configs register none.

Contract pinned here (the coder implements what these tests call):

- A family is pre-registered in ``<countries_dir>/<country>/config/families.yaml``::

      families:
        <name>:
          power_target_races: <int >= 1>   # the races needed to detect the stated gain at D7
          # other keys (the stated gain, the claims, ...) are allowed and not read here

  A target that is not a positive int (0, negative, null, a string, a float, a bool, or a missing
  ``power_target_races``) raises ``HoldoutConfigError``.
- ``split_off(frame, date_col, country, *, countries_dir=..., race_col=None)``. A race is one
  distinct (local race day, ``race_col`` value) pair: rows may be runner-level, and a race id may
  repeat across days (a race number). ``race_col`` may be omitted for a development-only split.
- ``LockedHoldout.open(*, ticket, purpose, family)`` raises ``HoldoutOpenError`` and logs nothing
  when: the family has no registered target; the split had no ``race_col``; the block has fewer
  races than the target (equal opens); the family already has a line in the log; or the block was
  opened after this holdout was split. A refused opening is not a use: nothing rolls forward.
- An opening's log line also carries ``family``, ``power_target`` and ``n_races``, and
  ``period.start`` is the start of the block it opened.
- Every line in the log is an opening. The current block starts on the later of the configured
  ``start`` and the local race date (the country's timezone: Asia/Tokyo, Asia/Hong_Kong) of the
  latest logged opening's ``utc``; the opening date itself belongs to the new block. Days of an
  opened block stay holdout days (development output is unchanged); ``split_off`` returns only the
  current block's rows in the ``LockedHoldout``. A log line that cannot be read refuses.

Every fixture writes ``timezone`` into holdout.yaml as well, so the timezone may come from there or
from a fixed per-country map.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

from horsetrack.core.eval.holdout import (
    HoldoutConfigError,
    HoldoutOpenError,
    split_off,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"

SEED_A = "3f9c2a7be1d04c55a8e6f0b19d7c4e21"
SEED_B = "b7e1d0c9a8f746352e1f0a9b8c7d6e5f"
SEED_C = "0123456789abcdef0123456789abcdef"
ZONES = {"japan": "Asia/Tokyo", "hongkong": "Asia/Hong_Kong"}
UTC = timezone.utc


# --------------------------------------------------------------------------- helpers


def seeded_holdout(day: date, seed: str) -> bool:
    """The owner's per-day rule, computed independently of the code under test."""
    return int(hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest(), 16) % 2 == 0


def local_day(utc_stamp: datetime, country: str) -> date:
    return utc_stamp.astimezone(ZoneInfo(ZONES[country])).date()


def write_definition(root: Path, country: str, *, start: date, seed: str) -> None:
    config_dir = root / country / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "holdout.yaml").write_text(
        f'holdout:\n  start: "{start.isoformat()}"\n  seed: "{seed}"\n'
        f'  timezone: "{ZONES[country]}"\n'
    )


def write_families(root: Path, country: str, targets: dict[str, object] | None = None, *, raw=None):
    config_dir = root / country / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    if raw is None:
        raw = yaml.safe_dump(
            {
                "families": {
                    name: {"power_target_races": target, "stated_gain_nats": 0.002}
                    for name, target in (targets or {}).items()
                }
            }
        )
    (config_dir / "families.yaml").write_text(raw)


def make_runner_frame(first: date, n_days: int) -> pd.DataFrame:
    """Runner-level rows: 1 to 4 races a race day, 2 to 5 runners a race, a race number that
    repeats across days, some days without racing, and a post time on each row."""
    rows = []
    for i in range(n_days):
        if i % 7 == 3:
            continue  # no racing that day
        day = first + timedelta(days=i)
        for race_no in range(1, 2 + (i * 5) % 4):
            for runner in range(2 + (i + race_no) % 4):
                rows.append(
                    {
                        "row_id": len(rows),
                        "race_date": datetime.combine(day, time(10 + race_no, 5)),
                        "race_no": race_no,
                        "horse": f"SECRETHORSE{len(rows):05d}",
                        "runner": runner,
                    }
                )
    return pd.DataFrame(rows)


def days_of(frame: pd.DataFrame) -> list[date]:
    return list(pd.to_datetime(frame["race_date"]).dt.date)


def block(frame: pd.DataFrame, *, start: date, seed: str, block_start: date | None = None):
    """(row ids, races) on the seeded holdout days from max(start, block_start) on."""
    first = max(start, block_start or start)
    ids, races = set(), set()
    for row_id, day, race_no in zip(frame["row_id"], days_of(frame), frame["race_no"], strict=True):
        if day >= first and seeded_holdout(day, seed):
            ids.add(int(row_id))
            races.add((day, int(race_no)))
    return ids, races


def development_ids(frame: pd.DataFrame, *, start: date, seed: str) -> set[int]:
    every_holdout, _ = block(frame, start=start, seed=seed)
    return set(frame["row_id"].astype(int)) - every_holdout


def log_path(root: Path, country: str) -> Path:
    return root / country / "holdout_log.jsonl"


def log_records(root: Path, country: str) -> list[dict]:
    path = log_path(root, country)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def log_bytes(root: Path, country: str) -> bytes:
    path = log_path(root, country)
    return path.read_bytes() if path.exists() else b""


def write_past_opening(root: Path, country: str, utc_stamp: datetime, family: str) -> None:
    """A line as an earlier opening would have left it in the log."""
    record = {
        "utc": utc_stamp.isoformat(),
        "ticket": 90,
        "purpose": f"earlier opening of {family}",
        "git_sha": "0" * 40,
        "n_rows": 10,
        "period": {"start": "2026-11-28", "seed_sha256": "f" * 64},
        "family": family,
        "power_target": 1,
        "n_races": 4,
    }
    path = log_path(root, country)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")


def split(frame, root, country="japan", **kwargs):
    kwargs.setdefault("race_col", "race_no")
    return split_off(frame, "race_date", country, countries_dir=root, **kwargs)


@pytest.fixture(autouse=True)
def real_logs_untouched():
    """No test may create or change a real country holdout log; restore one if a test did."""
    paths = [REAL_COUNTRIES / c / "holdout_log.jsonl" for c in ZONES]
    before = {p: p.read_bytes() if p.exists() else None for p in paths}
    yield
    changed = []
    for path, content in before.items():
        now = path.read_bytes() if path.exists() else None
        if now != content:
            changed.append(path)
            if content is None:
                path.unlink()
            else:
                path.write_bytes(content)
    assert not changed, f"a test wrote to a real holdout log: {changed}"


# --------------------------------------------------------------------------- the power target


GATE_SETUPS = [
    pytest.param(SEED_A, date(2026, 11, 28), date(2026, 10, 1), 150, id="seedA-japan-start"),
    pytest.param(SEED_B, date(2027, 1, 4), date(2026, 10, 1), 180, id="seedB-later-start"),
    pytest.param(SEED_C, date(2026, 10, 15), date(2026, 10, 1), 60, id="seedC-short"),
]
TARGETS = [
    pytest.param(lambda n: 1, True, id="target-1"),
    pytest.param(lambda n: n - 1, True, id="target-n-minus-1"),
    pytest.param(lambda n: n, True, id="target-n"),
    pytest.param(lambda n: n + 1, False, id="target-n-plus-1"),
    pytest.param(lambda n: 2 * n, False, id="target-2n"),
]


@pytest.mark.parametrize(("seed", "start", "first", "n_days"), GATE_SETUPS)
@pytest.mark.parametrize(("target_of", "opens"), TARGETS)
def test_open_is_gated_on_the_family_power_target(
    tmp_path, seed, start, first, n_days, target_of, opens
):
    frame = make_runner_frame(first, n_days)
    ids, races = block(frame, start=start, seed=seed)
    n = len(races)
    # Rows are runners and race numbers repeat: neither a row count nor a race-number count is n.
    assert len(ids) > n > frame["race_no"].nunique()
    target = target_of(n)
    write_definition(tmp_path, "japan", start=start, seed=seed)
    write_families(tmp_path, "japan", {"F": target})

    _, holdout = split(frame, tmp_path)
    if not opens:
        with pytest.raises(HoldoutOpenError):
            holdout.open(ticket=126, purpose="below the power target", family="F")
        assert log_bytes(tmp_path, "japan") == b""
        return

    rows = holdout.open(ticket=126, purpose="at or above the power target", family="F")
    assert set(rows["row_id"].astype(int)) == ids
    (record,) = log_records(tmp_path, "japan")
    assert record["family"] == "F"
    assert record["power_target"] == target
    assert record["n_races"] == n
    assert record["n_rows"] == len(ids)
    assert record["period"]["start"] == start.isoformat()


def test_a_refused_opening_spends_nothing(tmp_path):
    start, seed = date(2026, 11, 28), SEED_A
    frame = make_runner_frame(date(2026, 10, 1), 150)
    ids, races = block(frame, start=start, seed=seed)
    write_definition(tmp_path, "japan", start=start, seed=seed)
    write_families(tmp_path, "japan", {"BIG": len(races) + 1, "FITS": len(races)})

    _, first = split(frame, tmp_path)
    with pytest.raises(HoldoutOpenError):
        first.open(ticket=126, purpose="too few races", family="BIG")
    assert log_bytes(tmp_path, "japan") == b""

    _, second = split(frame, tmp_path)
    rows = second.open(ticket=126, purpose="the same block, unspent", family="FITS")
    assert set(rows["row_id"].astype(int)) == ids
    (record,) = log_records(tmp_path, "japan")
    assert record["period"]["start"] == start.isoformat()


NO_REGISTERED_TARGET = [
    pytest.param(None, None, id="no-families-file"),
    pytest.param("families: {}\n", None, id="no-family-registered"),
    pytest.param(None, {"OTHER": 1}, id="another-family-registered"),
    pytest.param(None, "hongkong", id="registered-for-another-country-only"),
]


@pytest.mark.parametrize(("raw", "targets"), NO_REGISTERED_TARGET)
def test_open_refuses_without_a_registered_target(tmp_path, raw, targets):
    write_definition(tmp_path, "japan", start=date(2026, 11, 28), seed=SEED_A)
    if targets == "hongkong":
        write_definition(tmp_path, "hongkong", start=date(2026, 11, 28), seed=SEED_B)
        write_families(tmp_path, "hongkong", {"F": 1})
    elif raw is not None or targets is not None:
        write_families(tmp_path, "japan", targets, raw=raw)

    _, holdout = split(make_runner_frame(date(2026, 10, 1), 150), tmp_path)
    with pytest.raises(HoldoutOpenError):
        holdout.open(ticket=126, purpose="no target recorded", family="F")
    assert log_bytes(tmp_path, "japan") == b""
    assert log_bytes(tmp_path, "hongkong") == b""


INVALID_TARGETS = [
    pytest.param("families:\n  F:\n    power_target_races: 0\n", id="zero"),
    pytest.param("families:\n  F:\n    power_target_races: -40\n", id="negative"),
    pytest.param("families:\n  F:\n    power_target_races: null\n", id="null"),
    pytest.param('families:\n  F:\n    power_target_races: "40"\n', id="string"),
    pytest.param("families:\n  F:\n    power_target_races: 12.5\n", id="float"),
    pytest.param("families:\n  F:\n    power_target_races: true\n", id="bool"),
    pytest.param("families:\n  F:\n    stated_gain_nats: 0.002\n", id="no-target-key"),
]


@pytest.mark.parametrize("raw", INVALID_TARGETS)
def test_an_invalid_power_target_is_a_config_error(tmp_path, raw):
    write_definition(tmp_path, "japan", start=date(2026, 11, 28), seed=SEED_A)
    write_families(tmp_path, "japan", raw=raw)

    with pytest.raises(HoldoutConfigError):
        _, holdout = split(make_runner_frame(date(2026, 10, 1), 150), tmp_path)
        holdout.open(ticket=126, purpose="invalid target", family="F")
    assert log_bytes(tmp_path, "japan") == b""


def test_counting_races_needs_a_race_identity(tmp_path):
    start, seed = date(2026, 11, 28), SEED_A
    frame = make_runner_frame(date(2026, 10, 1), 150)
    write_definition(tmp_path, "japan", start=start, seed=seed)
    write_families(tmp_path, "japan", {"F": 1})

    development, holdout = split_off(frame, "race_date", "japan", countries_dir=tmp_path)
    assert set(development["row_id"].astype(int)) == development_ids(frame, start=start, seed=seed)
    with pytest.raises(HoldoutOpenError):
        holdout.open(ticket=126, purpose="races cannot be counted", family="F")
    assert log_bytes(tmp_path, "japan") == b""


# --------------------------------------------------------------------------- rolling forward


def test_after_an_opening_the_next_block_starts_on_its_date_and_the_first_stays_logged(tmp_path):
    """Two real openings in a row; the frame is built around today, since an opening is now."""
    today = datetime.now(ZoneInfo(ZONES["japan"])).date()
    start, seed = today - timedelta(days=80), SEED_B
    frame = make_runner_frame(today - timedelta(days=90), 150)
    write_definition(tmp_path, "japan", start=start, seed=seed)
    write_families(tmp_path, "japan", {"F1": 1, "F2": 1})
    exp_dev = development_ids(frame, start=start, seed=seed)

    dev_before, first = split(frame, tmp_path)
    assert set(dev_before["row_id"].astype(int)) == exp_dev
    rows1 = first.open(ticket=126, purpose="first family", family="F1")
    (rec1,) = log_records(tmp_path, "japan")
    opened_on = local_day(datetime.fromisoformat(rec1["utc"].replace("Z", "+00:00")), "japan")
    ids1, races1 = block(frame, start=start, seed=seed)
    ids2, races2 = block(frame, start=start, seed=seed, block_start=opened_on)
    # The fixture has holdout days on both sides of the opening.
    assert ids2 and ids1 - ids2 and ids2 < ids1
    assert set(rows1["row_id"].astype(int)) == ids1
    assert rec1["period"]["start"] == start.isoformat()
    assert rec1["n_races"] == len(races1)
    after_first = log_bytes(tmp_path, "japan")

    dev_after, second = split(frame, tmp_path)
    pd.testing.assert_frame_equal(dev_after, dev_before)  # the opened block's days stay locked
    rows2 = second.open(ticket=127, purpose="second family", family="F2")

    assert set(rows2["row_id"].astype(int)) == ids2
    raw = log_bytes(tmp_path, "japan")
    assert raw.startswith(after_first)  # the opened block's line is kept as it was
    rec1_again, rec2 = log_records(tmp_path, "japan")
    assert rec1_again == rec1
    assert rec2["family"] == "F2"
    assert rec2["period"]["start"] == opened_on.isoformat()
    assert rec2["n_races"] == len(races2)
    assert rec2["n_rows"] == len(ids2)


def first_holdout_race_day(frame: pd.DataFrame, seed: str, after: date) -> date:
    racing = set(days_of(frame))
    return min(d for d in racing if d >= after and seeded_holdout(d, seed))


# (country, earlier openings as (day offset from D, UTC hour, minute), expected block start - D)
LOCAL_DATE_CASES = [
    pytest.param("japan", [(0, 20, 0)], 1, id="japan-20h-utc-is-next-jst-day"),
    pytest.param("japan", [(0, 14, 59)], 0, id="japan-14h59-utc-is-same-jst-day"),
    pytest.param("japan", [(0, 15, 0)], 1, id="japan-15h00-utc-is-next-jst-day"),
    pytest.param("japan", [(0, 2, 0)], 0, id="japan-morning-opening-day-is-in-new-block"),
    pytest.param("japan", [(-20, 3, 0), (0, 20, 0)], 1, id="japan-latest-opening-wins"),
    pytest.param("hongkong", [(0, 15, 30)], 0, id="hongkong-15h30-utc-is-same-hkt-day"),
    pytest.param("hongkong", [(0, 16, 0)], 1, id="hongkong-16h00-utc-is-next-hkt-day"),
]


@pytest.mark.parametrize(("country", "openings", "shift"), LOCAL_DATE_CASES)
def test_the_new_block_starts_on_the_local_race_date_of_the_latest_opening(
    tmp_path, country, openings, shift
):
    start, seed = date(2026, 11, 28), SEED_C
    frame = make_runner_frame(date(2026, 10, 1), 180)
    d = first_holdout_race_day(frame, seed, date(2026, 12, 20))
    write_definition(tmp_path, country, start=start, seed=seed)
    write_families(tmp_path, country, {"NEW": 1})
    for offset, hour, minute in openings:
        when = datetime.combine(d + timedelta(days=offset), time(hour, minute), tzinfo=UTC)
        write_past_opening(tmp_path, country, when, family=f"PAST{offset}")
    expected_start = d + timedelta(days=shift)
    ids, races = block(frame, start=start, seed=seed, block_start=expected_start)
    # D is a holdout day with races, so whether it is in the new block shows in the rows.
    assert any(day == d for day, _ in races) == (shift == 0)
    assert ids

    development, holdout = split(frame, tmp_path, country=country)
    assert set(development["row_id"].astype(int)) == development_ids(frame, start=start, seed=seed)
    rows = holdout.open(ticket=126, purpose="after earlier openings", family="NEW")

    assert set(rows["row_id"].astype(int)) == ids
    record = log_records(tmp_path, country)[-1]
    assert record["period"]["start"] == expected_start.isoformat()
    assert record["n_races"] == len(races)


# --------------------------------------------------------------------------- once per family


def test_a_family_opens_once(tmp_path):
    today = datetime.now(ZoneInfo(ZONES["japan"])).date()
    start, seed = today - timedelta(days=80), SEED_A
    frame = make_runner_frame(today - timedelta(days=90), 150)
    write_definition(tmp_path, "japan", start=start, seed=seed)
    write_families(tmp_path, "japan", {"USED": 1, "F2": 1})
    write_past_opening(
        tmp_path, "japan", datetime.combine(start + timedelta(days=5), time(6), UTC), "USED"
    )

    _, holdout = split(frame, tmp_path)
    before = log_bytes(tmp_path, "japan")
    with pytest.raises(HoldoutOpenError):
        holdout.open(ticket=126, purpose="a second look for USED", family="USED")
    assert log_bytes(tmp_path, "japan") == before

    holdout.open(ticket=126, purpose="F2 once", family="F2")
    after_f2 = log_bytes(tmp_path, "japan")
    assert len(after_f2.splitlines()) == 2

    _, fresh = split(frame, tmp_path)
    with pytest.raises(HoldoutOpenError):
        fresh.open(ticket=127, purpose="F2 again on the fresh block", family="F2")
    assert log_bytes(tmp_path, "japan") == after_f2


def test_a_block_opened_since_the_split_refuses(tmp_path):
    today = datetime.now(ZoneInfo(ZONES["japan"])).date()
    start, seed = today - timedelta(days=80), SEED_C
    frame = make_runner_frame(today - timedelta(days=90), 150)
    write_definition(tmp_path, "japan", start=start, seed=seed)
    write_families(tmp_path, "japan", {"F1": 1, "F2": 1, "F3": 1})

    _, h1 = split(frame, tmp_path)
    _, h2 = split(frame, tmp_path)
    h1.open(ticket=126, purpose="F1 opens the block", family="F1")
    after = log_bytes(tmp_path, "japan")

    for holdout, family in ((h1, "F2"), (h2, "F2"), (h2, "F3")):
        with pytest.raises(HoldoutOpenError):
            holdout.open(ticket=126, purpose="the block is already spent", family=family)
    assert log_bytes(tmp_path, "japan") == after

    (rec1,) = log_records(tmp_path, "japan")
    opened_on = local_day(datetime.fromisoformat(rec1["utc"].replace("Z", "+00:00")), "japan")
    ids, _ = block(frame, start=start, seed=seed, block_start=opened_on)
    assert ids
    _, fresh = split(frame, tmp_path)
    rows = fresh.open(ticket=126, purpose="F2 on the fresh block", family="F2")
    assert set(rows["row_id"].astype(int)) == ids


UNREADABLE_LOG_LINES = [
    pytest.param("not json at all\n", id="not-json"),
    pytest.param('{"ticket": 90, "purpose": "no utc", "family": "PAST"}\n', id="no-utc"),
    pytest.param('{"utc": "yesterday", "ticket": 90, "family": "PAST"}\n', id="utc-unparseable"),
]


@pytest.mark.parametrize("line", UNREADABLE_LOG_LINES)
def test_an_unreadable_log_line_refuses(tmp_path, line):
    """A line that cannot be read could hide an opening; the block cannot be placed safely."""
    write_definition(tmp_path, "japan", start=date(2026, 11, 28), seed=SEED_A)
    write_families(tmp_path, "japan", {"F": 1})
    log_path(tmp_path, "japan").write_text(line)

    with pytest.raises((ValueError, HoldoutOpenError)):
        _, holdout = split(make_runner_frame(date(2026, 10, 1), 150), tmp_path)
        holdout.open(ticket=126, purpose="log unreadable", family="F")
    assert log_path(tmp_path, "japan").read_text() == line


# --------------------------------------------------------------------------- committed configs


def test_the_committed_japan_holdout_registers_no_family_and_cannot_be_opened():
    """Registering a family and its target is the owner's act; until then the real holdout stays
    shut. Read through the default root; the autouse guard restores the real log if a write slips
    through."""
    for country in ZONES:
        path = REAL_COUNTRIES / country / "config" / "families.yaml"
        if path.exists():
            assert not (yaml.safe_load(path.read_text()) or {}).get("families")

    frame = make_runner_frame(date(2026, 11, 14), 60)
    _, holdout = split_off(frame, "race_date", "japan", race_col="race_no")
    for family in ("F", "win", "default"):
        with pytest.raises(HoldoutOpenError):
            holdout.open(ticket=126, purpose="no family is registered", family=family)
