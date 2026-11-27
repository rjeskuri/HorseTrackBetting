"""Ticket #454: the `horsetrack japan bets` commands, `check`, `record` and `settle`.

Contract pinned here, always through the root CLI (`horsetrack.cli.app`), so the registration
`japan.add_typer(bets_app, name='bets')` is tested too:

- `bets check --date D --stake N` writes nothing, prints the cap left before this stake (the
  day cap of countries/japan/config/betting.yaml minus the day's staked total, as
  `BetLog.check` gives it) and exits 0 when the day's total plus N is at most the cap, 1 past it.
- `bets record --date D --race-id ... --pool ... --combination ... --stake N [--option-id ...]
  [--placed-utc ...] [--note ...]` appends the bet to the day's `BetLog` (placed_utc defaults to
  now, UTC). A bet past the cap is kept: the over-cap flag is printed and it exits 1. A refused
  stake or bet exits non-zero, cleanly, and writes nothing.
- `bets settle --date D` settles the day's bets against `settle.day_dividends(D)`, writes
  `$HORSETRACK_DATA_DIR/bets/japan/<D>.settled.json` (one JSON object per bet, its outcome and
  paid_yen among its fields), leaves the bet log as it was, and prints the day's staked, paid and
  return per 100 yen.
- Nothing lands in the repository: the bet log is the owner's financial record.

Every bet is synthetic; HORSETRACK_DATA_DIR points at tmp_path in every test. The `local` test
reads the real jra store through a temp data dir that links to it, so no bet reaches the real one.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.bets.log import BetLog, bet_log_path, load_limits

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[4]
BETTING = REPO / "src" / "horsetrack" / "countries" / "japan" / "config" / "betting.yaml"
CAP = yaml.safe_load(BETTING.read_text(encoding="utf-8"))["day_cap_yen"]
DAY = date(2026, 12, 5)  # a development day by the committed rule (asserted below)
HELD = date(2026, 11, 29)  # the first holdout day by the committed rule (asserted below)
DIVIDEND_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
OVER_CAP = re.compile(
    r"(?:over|past|exceeds?)[\s_-]*(?:the\s+)?(?:day(?:'s)?\s+|race[\s-]day\s+)?cap", re.I
)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
runner = CliRunner()


def _real_data_dir() -> Path | None:
    """The owner's data dir, read once at import, before any test points the setting elsewhere."""
    try:
        return paths.data_dir()
    except (paths.DataDirNotSet, paths.DataDirInsideRepo):
        return None


REAL_DATA = _real_data_dir()


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture(autouse=True)
def D(tmp_path, monkeypatch) -> Path:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    monkeypatch.chdir(REPO)  # a write relative to the working directory lands in the repo: caught
    return data


def bets(*args: str) -> tuple[int, str]:
    """`horsetrack japan bets ...`: the exit code and the plain output. A crash is not an exit."""
    result = runner.invoke(app, ["japan", "bets", *args])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, _ANSI.sub("", result.output)


def shows(out: str, yen: int) -> bool:
    """Whether ``out`` prints the whole figure ``yen``, with or without thousands separators."""
    flat = re.sub(r"(?<=\d)[,_](?=\d{3}(?!\d))", "", out)
    return re.search(rf"(?<![\d.\-\u2212]){yen}(?:\.0+)?(?!\.?\d)", flat) is not None


def log_of(day: date) -> BetLog:
    return BetLog(bet_log_path(day), day=day, limits=load_limits())


def seed(day: date, total: int) -> int:
    """Stake ``total`` yen on ``day`` through the bet log itself; returns the number of bets."""
    log, n = log_of(day), 0
    while total > 0:
        stake = min(total, 10_000)
        n += 1
        log.record(
            {
                "race_id": f"jra:{day.isoformat()}:NAK:{1 + n % 12:02d}",
                "pool": "WIN",
                "combination": str(1 + n % 16),
                "stake_yen": stake,
                "placed_utc": f"{day.isoformat()}T01:00:00Z",
            }
        )
        total -= stake
    return n


def record_args(day: date = DAY, **given: str) -> list[str]:
    """`record` options for a valid bet of ``day``, overridden by ``given`` (names with _)."""
    options = {
        "date": day.isoformat(),
        "race_id": f"jra:{day.isoformat()}:NAK:11",
        "pool": "WIN",
        "combination": "4",
        "stake": "1000",
        **given,
    }
    out: list[str] = []
    for name, value in options.items():
        out += [f"--{name.replace('_', '-')}", value]
    return out


def repo_files() -> dict[Path, tuple[int, int]]:
    found = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        if Path(root) == REPO / ".claude":
            dirs[:] = [d for d in dirs if d != "worktrees"]
        for name in files:
            path = Path(root) / name
            stat = path.lstat()
            found[path] = (stat.st_mtime_ns, stat.st_size)
    return found


def written_in_repo(before: dict[Path, tuple[int, int]]) -> list[Path]:
    """Files the test created in the repository, or JSON it changed there."""
    after = repo_files()
    return sorted(
        path
        for path, stamp in after.items()
        if path not in before or (stamp != before[path] and path.suffix in (".json", ".jsonl"))
    )


# ---------------------------------------------------------------- check


def test_check_exits_1_past_the_cap(D):
    assert CAP == 30_000  # the owner's answer on #453: 29,900 staked leaves 100
    before = repo_files()
    seed(DAY, 29_900)
    raw = bet_log_path(DAY).read_bytes()
    code, out = bets("check", "--date", DAY.isoformat(), "--stake", "200")
    assert code == 1, out
    assert shows(out, CAP - 29_900), f"the remaining 100 yen is not printed: {out!r}"
    assert bet_log_path(DAY).read_bytes() == raw, "check writes nothing"
    assert written_in_repo(before) == []


@pytest.mark.parametrize(
    ("day", "prior", "stake"),
    [
        (DAY, 0, 1_000),
        (DAY, 12_400, 1_000),
        (DAY, 27_300, 3_000),
        (DAY, 29_000, 1_000),  # exactly at the cap fits
        (DAY, 29_900, 100),  # exactly at the cap fits
        (DAY, 30_500, 100),  # a day already past the cap
        (HELD, 29_900, 200),  # a holdout day's stakes count like any other's
        (HELD, 18_300, 1_000),
    ],
    ids=lambda value: value.isoformat() if isinstance(value, date) else None,
)
def test_check_answers_from_the_days_staked_total(D, day, prior, stake):
    assert is_holdout_day("japan", day) is (day == HELD)
    seed(day, prior)
    path = bet_log_path(day)
    raw = path.read_bytes() if path.exists() else None
    code, out = bets("check", "--date", day.isoformat(), "--stake", str(stake))
    fits = prior + stake <= CAP
    assert code == (0 if fits else 1), out
    remaining = CAP - prior
    if remaining >= 0:
        assert shows(out, remaining), f"the remaining {remaining} yen is not printed: {out!r}"
    assert (path.read_bytes() if path.exists() else None) == raw, "check writes nothing"


# ---------------------------------------------------------------- record


def test_record_keeps_an_over_cap_bet_and_exits_1(D):
    before = repo_files()
    n = seed(DAY, 29_900)
    code, out = bets(
        "record",
        *record_args(
            pool="QUINELLA",
            combination="7,4",
            stake="1000",
            option_id="opt-quinella-1",
            placed_utc="2026-12-05T05:40:00Z",
            note="own pick",
        ),
    )
    assert code == 1, out
    assert OVER_CAP.search(out), f"the over-cap flag is not printed: {out!r}"
    records = log_of(DAY).read()
    assert len(records) == n + 1, "the over-cap bet is kept"
    last = records[-1]
    assert {k: last[k] for k in ("race_id", "pool", "combination", "stake_yen")} == {
        "race_id": f"jra:{DAY.isoformat()}:NAK:11",
        "pool": "QUINELLA",
        "combination": "4,7",
        "stake_yen": 1_000,
    }
    assert (last["option_id"], last["placed_utc"], last["note"]) == (
        "opt-quinella-1",
        "2026-12-05T05:40:00Z",
        "own pick",
    )
    assert last["over_cap"] is True
    assert log_of(DAY).staked_yen() == 30_900

    # the next bet on a day already past the cap is kept and flagged too
    code, out = bets("record", *record_args(stake="100", combination="9"))
    assert code == 1, out
    assert OVER_CAP.search(out), out
    assert log_of(DAY).read()[-1]["over_cap"] is True
    assert log_of(DAY).staked_yen() == 31_000
    assert written_in_repo(before) == []


@pytest.mark.parametrize(
    ("day", "prior"),
    [(DAY, 0), (DAY, 12_400), (DAY, 29_000), (HELD, 0)],
    ids=lambda value: value.isoformat() if isinstance(value, date) else None,
)
def test_record_within_the_cap_exits_0(D, day, prior):
    n = seed(day, prior)
    start = datetime.now(UTC)
    code, out = bets("record", *record_args(day, pool="TRIO", combination="9,2,5", stake="1000"))
    end = datetime.now(UTC)
    assert code == 0, out
    records = log_of(day).read()
    assert len(records) == n + 1
    last = records[-1]
    assert (last["race_id"], last["pool"], last["combination"], last["stake_yen"]) == (
        f"jra:{day.isoformat()}:NAK:11",
        "TRIO",
        "2,5,9",
        1_000,
    )
    assert last["over_cap"] is False
    assert last.get("option_id") is None and last.get("note") is None
    stamp = last["placed_utc"]
    placed = datetime.fromisoformat(stamp[:-1] + "+00:00" if stamp.endswith("Z") else stamp)
    assert placed.utcoffset() == timedelta(0), f"placed_utc defaults to UTC: {stamp}"
    assert start - timedelta(seconds=1) <= placed <= end + timedelta(seconds=1), stamp


def test_a_refused_stake_or_bet_exits_non_zero_and_writes_nothing(D):
    code, out = bets("record", *record_args())
    assert code == 0, out
    raw = bet_log_path(DAY).read_bytes()
    refused = [
        ["record", *record_args(pool="QUINELLA", combination="4,4")],  # a horse twice
        ["record", *record_args(race_id="jra:2026-12-06:NAK:11")],  # another day's race
        ["record", *record_args(stake="150")],  # not whole hundreds
        ["check", "--date", DAY.isoformat(), "--stake", "150"],
    ]
    for args in refused:
        code, out = bets(*args)
        assert code != 0, f"{args}: {out!r}"
        assert bet_log_path(DAY).read_bytes() == raw, args


# ---------------------------------------------------------------- settle


def patch_dividends(monkeypatch, frames: dict[date, pd.DataFrame]) -> None:
    """``day_dividends`` replaced by synthetic frames, wherever the command looks it up."""

    def fake(day):
        assert isinstance(day, date) and not isinstance(day, datetime), repr(day)
        return frames.get(day, pd.DataFrame(columns=DIVIDEND_COLUMNS)).copy()

    settle = importlib.import_module("horsetrack.countries.japan.bets.settle")
    monkeypatch.setattr(settle, "day_dividends", fake)
    cli = importlib.import_module("horsetrack.countries.japan.bets.cli")
    monkeypatch.setattr(cli, "day_dividends", fake, raising=False)


def outcome_rows(obj) -> list[dict]:
    """Every JSON object in the settled file that carries an outcome, wherever it sits."""
    if isinstance(obj, dict):
        found = [obj] if "outcome" in obj else []
        return found + [row for value in obj.values() for row in outcome_rows(value)]
    if isinstance(obj, list):
        return [row for value in obj for row in outcome_rows(value)]
    return []


def test_settle_writes_under_the_data_dir(D, monkeypatch):
    before = repo_files()
    race = f"jra:{DAY.isoformat()}:NAK:11"
    log = log_of(DAY)
    for pool, combination, stake in (
        ("WIN", "4", 1_000),
        ("PLACE", "4", 500),
        ("QUINELLA", "4,7", 500),
    ):
        log.record(
            {
                "race_id": race,
                "pool": pool,
                "combination": combination,
                "stake_yen": stake,
                "placed_utc": f"{DAY.isoformat()}T05:40:00Z",
            }
        )
    raw = bet_log_path(DAY).read_bytes()
    frame = pd.DataFrame(
        [
            {
                "race_id": race,
                "pool": "WIN",
                "combination": "4",
                "dividend": 350.0,
                "is_refund": False,
            },
            {
                "race_id": race,
                "pool": "PLACE",
                "combination": "4",
                "dividend": 300.0,
                "is_refund": False,
            },
            {
                "race_id": race,
                "pool": "PLACE",
                "combination": "7",
                "dividend": 120.0,
                "is_refund": False,
            },
            {
                "race_id": race,
                "pool": "QUINELLA",
                "combination": "4,9",
                "dividend": 900.0,
                "is_refund": False,
            },
        ],
        columns=DIVIDEND_COLUMNS,
    )
    patch_dividends(monkeypatch, {DAY: frame})

    code, out = bets("settle", "--date", DAY.isoformat())
    assert code == 0, out
    settled = D / "bets" / "japan" / f"{DAY.isoformat()}.settled.json"
    assert settled.is_file(), sorted(p.relative_to(D) for p in D.rglob("*"))
    rows = outcome_rows(json.loads(settled.read_text(encoding="utf-8")))
    got = sorted((row["outcome"], float(row["paid_yen"])) for row in rows)
    assert got == [("lost", 0.0), ("won", 1_500.0), ("won", 3_500.0)]  # 10 x 350, 5 x 300, 0

    # the day: staked 1000 + 500 + 500, paid 3500 + 1500 + 0, return 5000 / 2000 x 100
    for figure in (2_000, 5_000, 250):
        assert shows(out, figure), f"the day's {figure} is not printed: {out!r}"
    assert bet_log_path(DAY).read_bytes() == raw, "settling leaves the bet log as it was"
    assert written_in_repo(before) == []


# ---------------------------------------------------------------- the real store (local)


@pytest.mark.local
def test_real_store_settles_a_development_day(D, monkeypatch):
    if REAL_DATA is None:
        pytest.skip(f"{paths.DATA_DIR_VAR} is not set: the jra store is unavailable")
    store = REAL_DATA / "processed" / "jra"
    if not ((store / "races.parquet").is_file() and (store / "dividends.parquet").is_file()):
        pytest.skip(f"the jra store is missing: {store}")

    races = pq.read_table(
        store / "races.parquet", columns=["race_id", "race_date", "venue", "race_no"]
    ).to_pandas()
    races["day"] = [pd.Timestamp(d).date() for d in races["race_date"]]
    days = sorted(d for d in set(races["day"]) if not is_holdout_day("japan", d))
    if not days:
        pytest.skip("the jra store holds no development day")
    day = days[-1]
    keyed = {
        r.race_id: f"jra:{day.isoformat()}:{r.venue}:{int(r.race_no):02d}"
        for r in races[races["day"] == day].itertuples()
    }
    paid = (
        ds.dataset(store / "dividends.parquet")
        .to_table(columns=DIVIDEND_COLUMNS, filter=pc.field("race_id").isin(list(keyed)))
        .to_pandas()
    )
    paid["race_id"] = paid["race_id"].map(keyed)

    def key(frame: pd.DataFrame) -> list[tuple]:
        rows = frame[DIVIDEND_COLUMNS].to_dict("records")
        return sorted(
            (
                r["race_id"],
                r["pool"],
                r["combination"],
                None if pd.isna(r["dividend"]) else float(r["dividend"]),
                bool(r["is_refund"]),
            )
            for r in rows
        )

    (D / "processed").mkdir(parents=True)
    (D / "processed" / "jra").symlink_to(store, target_is_directory=True)
    from horsetrack.countries.japan.bets.settle import day_dividends

    assert key(day_dividends(day)) == key(paid)

    # a race with one winner and a placed horse that did not win: three bets, settled by hand
    bets_placed: list[tuple] = []
    chosen = None
    for candidate, rows in paid.groupby("race_id"):
        rows = rows[~rows["is_refund"].astype(bool)]
        win = rows[rows["pool"] == "WIN"]
        place = rows[(rows["pool"] == "PLACE") & ~rows["combination"].isin(win["combination"])]
        if len(win) == 1 and len(place):
            chosen, winner, placer = candidate, win.iloc[0], place.iloc[0]
            bets_placed = [
                ("WIN", winner["combination"], 200, "won", 2 * winner["dividend"]),
                ("WIN", placer["combination"], 100, "lost", 0.0),
                ("PLACE", placer["combination"], 100, "won", placer["dividend"]),
            ]
            break
    if not bets_placed:
        pytest.skip(f"no race of {day} has a single winner and another placed horse")

    real_log = REAL_DATA / "bets" / "japan"
    real_before = sorted(real_log.glob(f"{day.isoformat()}*")) if real_log.is_dir() else []
    real_stamps = [(p, p.stat().st_mtime_ns) for p in real_before]
    log, expected = log_of(day), []
    for pool, combination, stake, outcome, amount in bets_placed:
        log.record(
            {
                "race_id": chosen,
                "pool": pool,
                "combination": combination,
                "stake_yen": stake,
                "placed_utc": f"{day.isoformat()}T05:40:00Z",
                "note": f"{outcome} {pool}",
            }
        )
        expected.append((f"{outcome} {pool}", outcome, float(amount)))

    code, out = bets("settle", "--date", day.isoformat())
    assert code == 0, out
    settled = D / "bets" / "japan" / f"{day.isoformat()}.settled.json"
    rows = outcome_rows(json.loads(settled.read_text(encoding="utf-8")))
    got = sorted((row["note"], row["outcome"], float(row["paid_yen"])) for row in rows)
    assert got == sorted(expected)
    after = sorted(real_log.glob(f"{day.isoformat()}*")) if real_log.is_dir() else []
    assert [(p, p.stat().st_mtime_ns) for p in after] == real_stamps, (
        "the real bet log is untouched"
    )
