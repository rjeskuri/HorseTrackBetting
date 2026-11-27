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
- Fix round 1 (review of PR #597):
  - A runner scratched before the start (取消, 除外: `runners.is_scratched`, placing codes WV and
    WX) has its tickets refunded, though the store holds no refund row for it (the results parser
    and jravan write `is_refund` false). A bet whose combination holds such a runner of its race
    settles 'refunded', paid its stake; BRACKET_QUINELLA only when one of its brackets has no
    runner left running. A runner pulled up (競走中止, PU) or disqualified (失格, DISQ) started:
    a loss. A holdout day's bets stay 'holdout'.
  - `bets settle` on a day the holdout rule holds needs no results store: every bet 'holdout'.

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
import pyarrow as pa
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


# ---------------------------------------------------------------- scratched runners

# A placing as the results page prints it -> (finish_pos, placing_code, is_scratched), as
# jra/netkeiba/parse_results.py and jravan/runners.py store it: 取 (出走取消) and 除 (発走除外,
# 競走除外) never started; 中 (競走中止) and 失 (失格) started and lost.
PLACINGS = {
    "取": (None, "WV", True),
    "除": (None, "WX", True),
    "中": (None, "PU", False),
    "失": (None, "DISQ", False),
}


def write_results_store(data: Path, races: list[dict]) -> None:
    """A synthetic jra store in the shape the results parser writes, under data/processed/jra.
    Each race is {id, day, venue, no, field: [(horse_no, bracket, placing)], paid: [(pool,
    combination, dividend)]}; ``id`` is the store's own key. Dividends hold the paid combinations
    only, ``is_refund`` false: no refund row for a scratched runner, as in the real store."""
    races_rows, runner_rows, paid_rows = [], [], []
    for r in races:
        meeting = f"jra:{r['day'].isoformat()}:{r['venue']}"
        where = {"race_date": r["day"], "venue": r["venue"], "race_no": r["no"]}
        starters = sum(not PLACINGS.get(p, (0, None, False))[2] for _, _, p in r["field"])
        races_rows.append(
            {"race_id": r["id"], "meeting_id": meeting, **where, "field_size": starters}
        )
        for horse_no, bracket, raw in r["field"]:
            finish, code, scratched = PLACINGS[raw] if raw in PLACINGS else (int(raw), None, False)
            runner_rows.append(
                {
                    "race_id": r["id"],
                    "meeting_id": meeting,
                    **where,
                    "horse_id": f"H{r['id']}{horse_no:02d}",
                    "horse_no": horse_no,
                    "bracket": bracket,
                    "placing_raw": raw,
                    "finish_pos": finish,
                    "placing_code": code,
                    "dead_heat": False,
                    "is_scratched": scratched,
                }
            )
        for pool, combination, dividend in r["paid"]:
            paid_rows.append(
                {
                    "race_id": r["id"],
                    "pool": pool,
                    "combination": combination,
                    "dividend": float(dividend),
                    "is_refund": False,
                }
            )

    def table(rows: list[dict], types: dict[str, pa.DataType]) -> pa.Table:
        columns = {"jurisdiction": pa.array(["jra"] * len(rows), pa.string())}
        for name, kind in types.items():
            columns[name] = pa.array([row[name] for row in rows], kind)
        return pa.table(columns)

    where_types = {"race_date": pa.date32(), "venue": pa.string(), "race_no": pa.int64()}
    folder = data / "processed" / "jra"
    folder.mkdir(parents=True)
    race_types = {"race_id": pa.string(), "meeting_id": pa.string(), **where_types}
    pq.write_table(
        table(
            [{**row, "status": "run"} for row in races_rows],
            {**race_types, "field_size": pa.int64(), "status": pa.string()},
        ),
        folder / "races.parquet",
    )
    runner_types = {
        **race_types,
        "horse_id": pa.string(),
        "horse_no": pa.int64(),
        "bracket": pa.int64(),
        "placing_raw": pa.string(),
        "finish_pos": pa.int64(),
        "placing_code": pa.string(),
        "dead_heat": pa.bool_(),
        "is_scratched": pa.bool_(),
    }
    pq.write_table(table(runner_rows, runner_types), folder / "runners.parquet")
    paid_types = {
        "race_id": pa.string(),
        "pool": pa.string(),
        "combination": pa.string(),
        "dividend": pa.float64(),
        "is_refund": pa.bool_(),
    }
    pq.write_table(table(paid_rows, paid_types), folder / "dividends.parquet")


def log_bets(day: date, cases: list[tuple]) -> None:
    """cases: (tag, race_id, pool, combination, stake, ...); each bet recorded with ``tag`` as its
    note."""
    log = log_of(day)
    for tag, race_id, pool, combination, stake, *_ in cases:
        log.record(
            {
                "race_id": race_id,
                "pool": pool,
                "combination": combination,
                "stake_yen": stake,
                "placed_utc": f"{day.isoformat()}T05:40:00Z",
                "note": tag,
            }
        )


def settled_by_note(day: date, data: Path) -> dict[str, dict]:
    path = data / "bets" / "japan" / f"{day.isoformat()}.settled.json"
    assert path.is_file(), sorted(p.relative_to(data) for p in data.rglob("*"))
    rows = outcome_rows(json.loads(path.read_text(encoding="utf-8")))
    by_note = {row["note"]: row for row in rows}
    assert len(by_note) == len(rows), "one settled row per bet"
    return by_note


# TOK 11 on DAY, 12 runners in JRA's brackets for 12 (1-4 one horse, 5-8 two): 3 取消 (bracket 3
# left empty), 5 取消 and 6 除外 (bracket 5 left empty), 9 除外 (bracket 7 keeps 10), 11 競走中止,
# 8 失格 (bracket 6 keeps 7). Finish: 4, 10, 1 (brackets 4, 7, 1).
TOK11 = {
    "id": "S-TOK11",
    "day": DAY,
    "venue": "TOK",
    "no": 11,
    "field": [
        (1, 1, "3"), (2, 2, "5"), (3, 3, "取"), (4, 4, "1"), (5, 5, "取"), (6, 5, "除"),
        (7, 6, "4"), (8, 6, "失"), (9, 7, "除"), (10, 7, "2"), (11, 8, "中"), (12, 8, "6"),
    ],
    "paid": [
        ("WIN", "4", 520), ("PLACE", "4", 180), ("PLACE", "10", 240), ("PLACE", "1", 150),
        ("BRACKET_QUINELLA", "4,7", 1_350), ("QUINELLA", "4,10", 1_980),
        ("QUINELLA_PLACE", "4,10", 640), ("QUINELLA_PLACE", "1,4", 330),
        ("QUINELLA_PLACE", "1,10", 720), ("FORECAST", "4,10", 3_860), ("TRIO", "1,4,10", 2_750),
        ("TIERCE", "4,10,1", 15_420),
    ],
}  # fmt: skip
# TOK 12 on DAY, 10 runners (brackets 1-6 one horse, 7-8 two), none scratched: the numbers
# scratched in race 11 run here. Finish: 3, 9, 5 (brackets 3, 8, 5).
TOK12 = {
    "id": "S-TOK12",
    "day": DAY,
    "venue": "TOK",
    "no": 12,
    "field": [
        (1, 1, "6"), (2, 2, "4"), (3, 3, "1"), (4, 4, "7"), (5, 5, "3"), (6, 6, "8"),
        (7, 7, "9"), (8, 7, "10"), (9, 8, "2"), (10, 8, "5"),
    ],
    "paid": [
        ("WIN", "3", 410), ("PLACE", "3", 150), ("PLACE", "9", 330), ("PLACE", "5", 200),
        ("BRACKET_QUINELLA", "3,8", 1_870), ("QUINELLA", "3,9", 2_240),
        ("QUINELLA_PLACE", "3,9", 690), ("QUINELLA_PLACE", "3,5", 450),
        ("QUINELLA_PLACE", "5,9", 1_020), ("FORECAST", "3,9", 4_100), ("TRIO", "3,5,9", 5_010),
        ("TIERCE", "3,9,5", 28_650),
    ],
}  # fmt: skip
# NAK 5 on DAY, 8 runners, one per bracket: 6 除外 (bracket 6 left empty). Finish: 3, 1, 5.
NAK05 = {
    "id": "S-NAK05",
    "day": DAY,
    "venue": "NAK",
    "no": 5,
    "field": [
        (1, 1, "2"), (2, 2, "4"), (3, 3, "1"), (4, 4, "5"), (5, 5, "3"), (6, 6, "除"),
        (7, 7, "6"), (8, 8, "7"),
    ],
    "paid": [
        ("WIN", "3", 290), ("PLACE", "3", 130), ("PLACE", "1", 210), ("PLACE", "5", 160),
        ("BRACKET_QUINELLA", "1,3", 880), ("QUINELLA", "1,3", 910),
        ("QUINELLA_PLACE", "1,3", 350), ("QUINELLA_PLACE", "3,5", 280),
        ("QUINELLA_PLACE", "1,5", 560), ("FORECAST", "3,1", 1_620), ("TRIO", "1,3,5", 1_340),
        ("TIERCE", "3,1,5", 6_080),
    ],
}  # fmt: skip
# NAK 11 on HELD, a holdout day: 2 取消. Finish: 4, 7, 1.
HELD11 = {
    "id": "S-HELD11",
    "day": HELD,
    "venue": "NAK",
    "no": 11,
    "field": [
        (1, 1, "3"), (2, 2, "取"), (3, 3, "5"), (4, 4, "1"), (5, 5, "4"), (6, 6, "6"),
        (7, 7, "2"), (8, 8, "7"),
    ],
    "paid": [("WIN", "4", 350), ("PLACE", "4", 140), ("QUINELLA", "4,7", 1_230)],
}  # fmt: skip


def test_settle_refunds_bets_on_runners_scratched_before_the_start(D):
    assert not is_holdout_day("japan", DAY) and is_holdout_day("japan", HELD)
    write_results_store(D, [TOK11, TOK12, NAK05, HELD11])
    r1, r2, r3 = (f"jra:{DAY.isoformat()}:{race}" for race in ("TOK:11", "TOK:12", "NAK:05"))
    R, W, L = "refunded", "won", "lost"
    # (tag, race, pool, combination, stake, outcome, paid): paid worked out by hand, a refund its
    # stake, a win stake / 100 x the dividend of the row it matches
    cases = [
        ("win on 3, 取消", r1, "WIN", "3", 1_000, R, 1_000),
        ("win on 9, 除外", r1, "WIN", "9", 500, R, 500),
        ("place on 6, 除外", r1, "PLACE", "6", 300, R, 300),
        ("quinella 3 with the winner", r1, "QUINELLA", "3,4", 200, R, 200),
        ("quinella of two scratched", r1, "QUINELLA", "9,5", 100, R, 100),
        ("wide with 9", r1, "QUINELLA_PLACE", "1,9", 400, R, 400),
        ("trio with 5 and the first two", r1, "TRIO", "10,5,4", 100, R, 100),
        ("exacta with 3 second", r1, "FORECAST", "4,3", 200, R, 200),
        ("exacta with 9 first", r1, "FORECAST", "9,4", 100, R, 100),
        ("trifecta with 6 third", r1, "TIERCE", "4,10,6", 100, R, 100),
        ("win on the winner", r1, "WIN", "4", 1_000, W, 5_200),
        ("quinella paid", r1, "QUINELLA", "10,4", 300, W, 5_940),
        ("place on 10", r1, "PLACE", "10", 500, W, 1_200),
        ("trio paid", r1, "TRIO", "4,1,10", 100, W, 2_750),
        ("exacta in the wrong order", r1, "FORECAST", "10,4", 100, L, 0),
        ("trio of runners, unpaid", r1, "TRIO", "1,2,4", 100, L, 0),
        ("win on 11, 競走中止", r1, "WIN", "11", 1_000, L, 0),
        ("place on 11, 競走中止", r1, "PLACE", "11", 200, L, 0),
        ("quinella 11 with the winner", r1, "QUINELLA", "4,11", 100, L, 0),
        ("win on 8, 失格", r1, "WIN", "8", 100, L, 0),
        ("bracket 3 left empty", r1, "BRACKET_QUINELLA", "3,4", 100, R, 100),
        ("bracket 5 left empty", r1, "BRACKET_QUINELLA", "5,8", 200, R, 200),
        ("bracket 7 keeps 10", r1, "BRACKET_QUINELLA", "6,7", 100, L, 0),
        ("bracket 7 keeps 10 and paid", r1, "BRACKET_QUINELLA", "7,4", 100, W, 1_350),
        ("bracket 6 keeps 7, 8 失格", r1, "BRACKET_QUINELLA", "1,6", 100, L, 0),
        ("race 12: 3 runs and wins", r2, "WIN", "3", 300, W, 1_230),
        ("race 12: 9 runs", r2, "WIN", "9", 100, L, 0),
        ("race 12: 9 places", r2, "PLACE", "9", 200, W, 660),
        ("race 12: quinella 9,3", r2, "QUINELLA", "9,3", 100, W, 2_240),
        ("race 12: trio 6,5,4", r2, "TRIO", "6,5,4", 100, L, 0),
        ("race 12: brackets 3,5", r2, "BRACKET_QUINELLA", "3,5", 100, L, 0),
        ("race 12: brackets 8,3", r2, "BRACKET_QUINELLA", "8,3", 100, W, 1_870),
        ("race 12: exacta 3,9", r2, "FORECAST", "3,9", 100, W, 4_100),
        ("NAK 5: win on 6, 除外", r3, "WIN", "6", 700, R, 700),
        ("NAK 5: bracket 6 left empty", r3, "BRACKET_QUINELLA", "3,6", 100, R, 100),
        ("NAK 5: trifecta with 6", r3, "TIERCE", "3,1,6", 100, R, 100),
        ("NAK 5: win on 3", r3, "WIN", "3", 200, W, 580),
        ("NAK 5: quinella 3,1", r3, "QUINELLA", "3,1", 100, W, 910),
        ("NAK 5: win on 1", r3, "WIN", "1", 100, L, 0),
    ]
    log_bets(DAY, cases)
    code, out = bets("settle", "--date", DAY.isoformat())
    assert code == 0, out
    got = settled_by_note(DAY, D)
    assert set(got) == {case[0] for case in cases}
    wrong = []
    for tag, race, pool, combination, _, outcome, paid in cases:
        row = got[tag]
        if (
            row["outcome"] != outcome
            or row["paid_yen"] is None
            or not (float(row["paid_yen"]) == pytest.approx(paid))
        ):
            wrong.append(
                f"{tag} ({pool} {combination} on {race}): want {outcome} paid {paid}, "
                f"got {row['outcome']} paid {row['paid_yen']}"
            )
    assert wrong == [], "\n".join(wrong)

    # a holdout day's bets stay 'holdout', a scratched runner's too, with the store in place
    held = f"jra:{HELD.isoformat()}:NAK:11"
    held_cases = [
        ("held: win on 2, 取消", held, "WIN", "2", 1_000),
        ("held: quinella 2 with the winner", held, "QUINELLA", "2,4", 200),
        ("held: win on the winner", held, "WIN", "4", 100),
    ]
    log_bets(HELD, held_cases)
    code, out = bets("settle", "--date", HELD.isoformat())
    assert code == 0, out
    got = settled_by_note(HELD, D)
    assert {tag: (row["outcome"], row["paid_yen"]) for tag, row in got.items()} == {
        case[0]: ("holdout", None) for case in held_cases
    }


@pytest.mark.parametrize(
    "held",
    [HELD, date(2026, 12, 6)],  # holdout days by the committed rule (asserted below)
    ids=lambda value: value.isoformat(),
)
def test_settle_a_holdout_day_needs_no_results_store(D, held):
    # A fresh data dir: `bets record` can log a holdout day's bets before any results store exists;
    # `settle` still writes them, every one 'holdout', and does not read the store for that day.
    assert is_holdout_day("japan", held)
    assert not (D / "processed").exists()
    day = held.isoformat()
    cases = [
        ("win", f"jra:{day}:NAK:11", "WIN", "4", 1_000),
        ("quinella", f"jra:{day}:NAK:11", "QUINELLA", "7,4", 500),
        ("trifecta at another venue", f"jra:{day}:HSN:02", "TIERCE", "3,1,2", 100),
        ("win5", f"jra:{day}:WIN5", "WIN5", "1,2,3,4,5", 200),
    ]
    log_bets(held, cases)
    raw = bet_log_path(held).read_bytes()
    code, out = bets("settle", "--date", day)
    assert code == 0, out
    got = settled_by_note(held, D)
    assert {tag: (row["outcome"], row["paid_yen"]) for tag, row in got.items()} == {
        case[0]: ("holdout", None) for case in cases
    }
    assert bet_log_path(held).read_bytes() == raw, "settling leaves the bet log as it was"
    assert not (D / "processed").exists(), "no store is made for a holdout day"


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


@pytest.mark.local
def test_real_store_refunds_a_scratched_runner_and_not_a_pulled_up_one(D):
    if REAL_DATA is None:
        pytest.skip(f"{paths.DATA_DIR_VAR} is not set: the jra store is unavailable")
    store = REAL_DATA / "processed" / "jra"
    if not all((store / f"{t}.parquet").is_file() for t in ("races", "runners", "dividends")):
        pytest.skip(f"the jra store is missing: {store}")
    (D / "processed").mkdir(parents=True)
    (D / "processed" / "jra").symlink_to(store, target_is_directory=True)
    from horsetrack.countries.japan.models.v0.store import read_store_table

    # development days only: read_store_table drops the holdout days
    runners = read_store_table(
        "runners",
        columns=["race_id", "race_date", "venue", "race_no", "horse_no", "placing_code"],
        start=date(2000, 1, 1),
        end=date.today(),
        touches_price=False,
        where=pc.field("is_scratched") | pc.field("placing_code").isin(["PU"]),
    ).to_pandas()
    runners["day"] = [pd.Timestamp(d).date() for d in runners["race_date"]]
    runners["log_id"] = [
        f"jra:{r.day.isoformat()}:{r.venue}:{int(r.race_no):02d}" for r in runners.itertuples()
    ]
    gone = runners[runners["placing_code"].isin(["WV", "WX"])]
    pulled = runners[runners["placing_code"].eq("PU")]

    def paid_of(race_ids: list[str]) -> pd.DataFrame:
        frame = (
            ds.dataset(store / "dividends.parquet")
            .to_table(columns=DIVIDEND_COLUMNS, filter=pc.field("race_id").isin(race_ids))
            .to_pandas()
        )
        return frame[~frame["is_refund"].fillna(False).astype(bool)]

    def one_winner(paid: pd.DataFrame, race_id: str) -> str | None:
        mine = paid[paid["race_id"] == race_id]
        win = mine[mine["pool"] == "WIN"]
        has_quinella = mine["pool"].eq("QUINELLA").any()
        return win.iloc[0]["combination"] if len(win) == 1 and has_quinella else None

    chosen = None
    for day in sorted(set(gone["day"]) & set(pulled["day"]), reverse=True):
        assert not is_holdout_day("japan", day)
        today_gone, today_pulled = gone[gone["day"] == day], pulled[pulled["day"] == day]
        paid = paid_of(sorted(set(today_gone["race_id"]) | set(today_pulled["race_id"])))
        g = [(r, one_winner(paid, r.race_id)) for r in today_gone.itertuples()]
        p = [(r, one_winner(paid, r.race_id)) for r in today_pulled.itertuples()]
        g, p = [x for x in g if x[1]], [x for x in p if x[1]]
        if g and p:
            chosen = day, g[0], p[0], paid
            break
    if chosen is None:
        pytest.skip("no development day has a scratched and a pulled-up runner in paid races")
    day, (gone_runner, gone_winner), (pulled_runner, pulled_winner), paid = chosen
    win_dividend = float(
        paid[
            (paid["race_id"] == gone_runner.race_id)
            & (paid["pool"] == "WIN")
            & (paid["combination"] == gone_winner)
        ].iloc[0]["dividend"]
    )
    g_no, p_no = str(int(gone_runner.horse_no)), str(int(pulled_runner.horse_no))
    g_race, p_race = gone_runner.log_id, pulled_runner.log_id
    cases = [  # (tag, race, pool, combination, stake, outcome, paid)
        ("win on the scratched runner", g_race, "WIN", g_no, 200, "refunded", 200.0),
        (
            "quinella of it and the winner",
            g_race,
            "QUINELLA",
            f"{g_no},{gone_winner}",
            100,
            "refunded",
            100.0,
        ),
        ("win on its race's winner", g_race, "WIN", gone_winner, 100, "won", win_dividend),
        ("win on the pulled-up runner", p_race, "WIN", p_no, 100, "lost", 0.0),
        (
            "quinella of it and its winner",
            p_race,
            "QUINELLA",
            f"{p_no},{pulled_winner}",
            100,
            "lost",
            0.0,
        ),
    ]
    log_bets(day, cases)
    code, out = bets("settle", "--date", day.isoformat())
    assert code == 0, out
    got = settled_by_note(day, D)
    assert {tag: (row["outcome"], row["paid_yen"]) for tag, row in got.items()} == {
        tag: (outcome, pytest.approx(amount)) for tag, _, _, _, _, outcome, amount in cases
    }
