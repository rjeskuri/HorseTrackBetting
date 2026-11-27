"""Ticket #453: a hash-chained manual bet log per Japan race day, with the race-day cap.

Contract pinned here, module ``horsetrack.countries.japan.bets.log``:

- ``POOLS``, the jra config's pool codes. ``Limits(day_cap_yen, flat_stake_yen)``;
  ``load_limits()`` reads the committed ``countries/japan/config/betting.yaml`` (the owner's
  answer, 2026-11-27: a ¥30,000 cap per race day, ¥1,000 flat stake).
- ``bet_log_path(day)`` is ``$HORSETRACK_DATA_DIR/bets/japan/<YYYY-MM-DD>.jsonl``, the data dir
  read on every call.
- ``BetLog(path, *, day, limits)`` accepts only ``bet_log_path(day)``, compared after resolving: a
  path inside the repository raises ``BetLogInRepo`` (a RuntimeError), any other path a
  ValueError, and nothing is created (review fix, ruling 1: one log per race day, at one place).
  An open ``BetLog`` re-resolves ``bet_log_path(day)`` before every call (``record``, ``read``,
  ``check``, ``staked_yen``); once that is no longer its path the call raises ValueError and writes
  nothing anywhere (second review fix). Paths are compared by filesystem identity, so a data dir
  spelled in another letter case on a case-insensitive filesystem is the same log, before the
  day's first bet too; a blank setting is the same ValueError, not ``DataDirNotSet`` (third review).
- A bet is a dict of ``race_id`` (``jra:<day>:<VENUE>:<NN>``, the log's own day, a jra venue code;
  a WIN5 bet, and only a WIN5 bet, takes the day-level ``jra:<day>:WIN5``), ``pool`` (one of
  POOLS), ``combination`` (horse numbers joined by ``,``, as many as the pool's legs, no leading
  zeros, no repeats except in BRACKET_QUINELLA and WIN5; stored in the store's order: FORECAST,
  TIERCE and WIN5 as given, the others ascending), ``stake_yen`` (an int, at least 100, a multiple
  of 100), ``placed_utc`` (ISO text with a UTC offset) and the optional ``option_id`` and ``note``.
  Anything else raises ``BetRefused`` (a ValueError) and writes nothing.
- ``record(bet)`` appends one line and returns the stored record: the bet plus ``seq``,
  ``written_utc``, ``day_status`` (holdout or development, from the committed holdout rule),
  ``over_cap`` (the day's total including this bet exceeds the cap) and ``off_flat`` (a flat stake
  is set and this stake differs from it). A bet past the cap or off the flat stake is recorded.
- ``check(stake) -> CapCheck(fits, staked_yen, remaining_yen, off_flat)``: the day's total so far,
  the cap left before this stake, whether the stake fits (total + stake <= cap). It writes nothing.
- Lines are ``{"seq", "prev_sha256", "record", "sha256"}``, compact ASCII JSON, chained as
  docs/specs/shadow-log.md describes; ``sha256`` is the sha256 of the line's own encoding without
  ``sha256``, so an edit to the last line is caught too. A broken chain or checksum, or a record
  dated another day than the log's, raises ``BetLogTampered`` on every call.
- The previous line format (format 1, a565ac0: ``{seq, prev_sha256, record}``) is read as an
  append-only prefix (third review fix): a file may begin with format-1 lines, chain-verified and
  their records checked as a565ac0 checked a bet; they count like any other; ``record`` appends a
  format-2 line chained onto the last of them. A format-1 line after a format-2 line is tampering.
- Holdout days: bets are written, read back and counted toward the cap like any other day.

Every bet is synthetic. HORSETRACK_DATA_DIR points at tmp_path in every test.
"""

from __future__ import annotations  # noqa: I001 (isort counts bets.log third-party until it exists)

import ast
import hashlib
import json
import os
import re
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.bets.log import (
    POOLS,
    BetLog,
    BetLogInRepo,
    BetLogTampered,
    BetRefused,
    CapCheck,
    Limits,
    bet_log_path,
    load_limits,
)

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[4]
JAPAN_CONFIG = REPO / "src" / "horsetrack" / "countries" / "japan" / "config"
BETS_PACKAGE = REPO / "src" / "horsetrack" / "countries" / "japan" / "bets"
ZERO = "0" * 64
DEV_DAY = date(2026, 11, 22)  # before the holdout start: development by any reading
CAP = 30_000
NO_FLAT = Limits(day_cap_yen=CAP, flat_stake_yen=None)
ADDED = {"seq", "written_utc", "day_status", "over_cap", "off_flat"}
OPTIONAL = {"option_id", "note"}
LINE_KEYS = {"seq", "prev_sha256", "record", "sha256"}
FORBIDDEN_IMPORTS = (
    "requests",
    "httpx",
    "urllib",
    "aiohttp",
    "socket",
    "playwright",
    "selenium",
    "horsetrack.core.ingest",
    "urllib3",  # HTTP clients the ticket's list does not name
    "http.client",
)


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture(autouse=True)
def D(tmp_path, monkeypatch) -> Path:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    return data


def jra_config() -> dict:
    return yaml.safe_load((JAPAN_CONFIG / "jra.yaml").read_text(encoding="utf-8"))


def holdout_rule() -> Callable[[date], bool]:
    """The committed holdout rule, from holdout.yaml itself."""
    section = yaml.safe_load((JAPAN_CONFIG / "holdout.yaml").read_text(encoding="utf-8"))
    start, seed = section["holdout"]["start"], section["holdout"]["seed"]
    start = start if isinstance(start, date) else date.fromisoformat(start)

    def held(day: date) -> bool:
        digest = hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest()
        return day >= start and int(digest, 16) % 2 == 0

    return held


def bet(
    stake: int = 100,
    *,
    day: date = DEV_DAY,
    venue: str = "TOK",
    race: int = 11,
    pool: str = "WIN",
    combination: str = "4",
    placed: str | None = None,
    **extra,
) -> dict:
    """A synthetic hand-placed bet."""
    out = {
        "race_id": f"jra:{day.isoformat()}:{venue}:{race:02d}",
        "pool": pool,
        "combination": combination,
        "stake_yen": stake,
        "placed_utc": placed or f"{day.isoformat()}T05:40:00+00:00",
    }
    out.update(extra)
    return out


def win5(stake: int = 100, *, day: date = DEV_DAY, combination: str = "1,2,3,4,5") -> dict:
    """A synthetic WIN5 bet, keyed by its day (``jra:<day>:WIN5``) as the odds log keys the pool."""
    race_id = f"jra:{day.isoformat()}:WIN5"
    return bet(stake, day=day, pool="WIN5", combination=combination, race_id=race_id)


def open_log(day: date = DEV_DAY, limits: Limits = NO_FLAT) -> tuple[BetLog, Path]:
    path = bet_log_path(day)
    return BetLog(path, day=day, limits=limits), path


def sha(line: bytes) -> str:
    return hashlib.sha256(line).hexdigest()


def encode(obj) -> bytes:
    """The log's own encoding: compact, ASCII-escaped JSON."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode()


def line_sha(obj: dict) -> str:
    """The checksum a line carries: the sha256 of the line's own encoding without ``sha256``."""
    return sha(encode({key: value for key, value in obj.items() if key != "sha256"}))


def split_lines(raw: bytes) -> list[bytes]:
    assert raw.endswith(b"\n"), "every line ends in a newline"
    return raw[:-1].split(b"\n")


def join_lines(lines: list[bytes]) -> bytes:
    return b"".join(line + b"\n" for line in lines)


def lines_of(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in split_lines(raw)]


def chain(raw: bytes) -> list[dict]:
    """The parsed lines of a bet log, checked against the ticket's format from first principles."""
    objs, prev = [], ZERO
    for n, line in enumerate(split_lines(raw), 1):
        assert line.isascii(), f"line {n} is not ASCII"
        obj = json.loads(line)
        assert line == encode(obj), f"line {n} is not compact JSON"
        assert set(obj) == LINE_KEYS, f"line {n} has keys {sorted(obj)}, not {sorted(LINE_KEYS)}"
        assert obj["prev_sha256"] == prev, f"line {n} does not chain onto line {n - 1}"
        assert obj["sha256"] == line_sha(obj), f"line {n}'s sha256 is not of the line without it"
        assert obj["seq"] == n and type(obj["seq"]) is int
        assert obj["record"]["seq"] == n
        prev = sha(line)
        objs.append(obj)
    return objs


def rechain(objs: list[dict], *, keep: frozenset[int] = frozenset()) -> bytes:
    """The lines re-encoded as an editor who knows the format would write them: every
    ``prev_sha256`` and ``sha256`` recomputed, except the ``sha256`` of the lines (from 0) in
    ``keep``. A line without ``sha256`` stays without one."""
    out, prev = [], ZERO
    for i, obj in enumerate(objs):
        obj = {**obj, "prev_sha256": prev}
        if "sha256" in obj and i not in keep:
            obj["sha256"] = line_sha(obj)
        line = encode(obj)
        out.append(line)
        prev = sha(line)
    return join_lines(out)


def assert_tampered(path: Path, day: date = DEV_DAY) -> None:
    """Every call, each through a fresh BetLog of ``day``, raises BetLogTampered and leaves the
    file byte-identical."""
    raw = path.read_bytes()
    calls = {
        "read": lambda log: log.read(),
        "record": lambda log: log.record(bet(100, day=day, race=9)),
        "staked_yen": lambda log: log.staked_yen(),
        "check": lambda log: log.check(100),
    }
    for name, call in calls.items():
        try:
            call(BetLog(path, day=day, limits=NO_FLAT))
        except BetLogTampered:
            pass
        else:
            pytest.fail(f"{name} accepted the tampered log of {day}")
        assert path.read_bytes() == raw, f"{name} wrote onto a tampered log"


def instant(text: str) -> datetime:
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert stamp.utcoffset() is not None, f"{text} has no UTC offset"
    return stamp


def assert_stored(stored: dict, given: dict, *, seq: int, status: str, over: bool, off: bool):
    """The stored record is the bet as given plus the five fields the log adds."""
    assert {k: stored[k] for k in given if k != "placed_utc"} == {
        k: v for k, v in given.items() if k != "placed_utc"
    }
    assert instant(stored["placed_utc"]) == instant(given["placed_utc"])
    assert set(stored) - set(given) - ADDED <= OPTIONAL, f"unexpected keys {sorted(stored)}"
    assert ADDED <= set(stored)
    assert stored["seq"] == seq
    assert stored["day_status"] == status
    assert stored["over_cap"] is over
    assert stored["off_flat"] is off
    instant(stored["written_utc"])


def snapshot(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def assert_refused(log: BetLog, path: Path, bad) -> None:
    """``record(bad)`` raises BetRefused and leaves the file as it was (or absent, or empty)."""
    before, staked, records = snapshot(path), log.staked_yen(), log.read()
    with pytest.raises(BetRefused):
        log.record(bad)
    after = snapshot(path)
    if before is None:
        assert after in (None, b""), "a refused first bet wrote to the log"
    else:
        assert after == before, "a refused bet changed the log"
    assert log.staked_yen() == staked
    assert log.read() == records


# ---------------------------------------------------------------- writing and reading back


def test_a_bet_is_recorded_with_its_flags_and_day_status():
    day = date(2026, 12, 5)
    status = "holdout" if is_holdout_day("japan", day) else "development"
    assert is_holdout_day("japan", day) is holdout_rule()(day)

    log, path = open_log(day, Limits(day_cap_yen=CAP, flat_stake_yen=100))
    given = bet(100, day=day, venue="NAK", race=11, pool="WIN", combination="4")
    before = datetime.now(UTC)
    stored = log.record(dict(given))
    after = datetime.now(UTC)
    assert_stored(stored, given, seq=1, status=status, over=False, off=False)
    written = instant(stored["written_utc"])
    assert before - timedelta(seconds=2) <= written <= after + timedelta(seconds=2)

    for reader in (log, BetLog(path, day=day, limits=NO_FLAT)):  # from the file, not memory
        assert reader.read() == [stored]
        assert reader.staked_yen() == 100
    assert [obj["record"] for obj in chain(path.read_bytes())] == [stored]

    # a spread of bets on another day: every venue, varied pools, options and notes, in order
    venues = list(jra_config()["venues"])
    assert len(venues) == 10
    log, path = open_log(DEV_DAY)
    given_all, stored_all, total = [], [], 0
    for i, venue in enumerate(venues):
        stake = 100 * (i + 1)
        extra = {"option_id": f"lb:win:{i}", "note": f"race {i}"} if i % 3 else {}
        if i == 4:
            extra = {"option_id": None, "note": "中山 11R, my own pick"}  # null: the owner's pick
        given = bet(stake, venue=venue, race=1 + i, combination=str(i + 1), **extra)
        caller_copy = dict(given)
        stored = log.record(given)
        assert given == caller_copy, "record changed the caller's bet"
        total += stake
        assert_stored(stored, given, seq=i + 1, status="development", over=False, off=False)
        given_all.append(given)
        stored_all.append(stored)
        assert log.staked_yen() == total
    assert BetLog(path, day=DEV_DAY, limits=NO_FLAT).read() == stored_all
    assert [obj["record"] for obj in chain(path.read_bytes())] == stored_all
    assert total == sum(b["stake_yen"] for b in given_all) == 5_500


def test_every_pool_code_is_accepted():
    pools = jra_config()["pools"]
    assert tuple(pools) == POOLS
    assert POOLS == (
        "WIN",
        "PLACE",
        "BRACKET_QUINELLA",
        "QUINELLA",
        "QUINELLA_PLACE",
        "FORECAST",
        "TRIO",
        "TIERCE",
        "WIN5",
    )
    log, path = open_log()
    for i, pool in enumerate(POOLS):
        legs = pools[pool]["legs"]
        combination = ",".join(str(n + 1) for n in range(legs))
        # WIN5 spans five races: it is keyed by its day, as the odds log keys its pool
        key = {"race_id": f"jra:{DEV_DAY.isoformat()}:WIN5"} if pool == "WIN5" else {}
        stored = log.record(bet(200, race=1 + i, pool=pool, combination=combination, **key))
        assert stored["pool"] == pool and stored["combination"] == combination
        assert stored["race_id"] == key.get("race_id", f"jra:{DEV_DAY.isoformat()}:TOK:{1 + i:02d}")
        assert stored["seq"] == i + 1
    assert [r["pool"] for r in BetLog(path, day=DEV_DAY, limits=NO_FLAT).read()] == list(POOLS)
    assert log.staked_yen() == 200 * len(POOLS)


# (pool, combination as given, combination as stored): the store's order (jravan/dividends.py),
# FORECAST and TIERCE in finishing order, the others ascending by number
STORE_ORDER = [
    ("QUINELLA", "7,4", "4,7"),
    ("QUINELLA", "4,7", "4,7"),
    ("QUINELLA", "12,3", "3,12"),  # by number, not as text
    ("QUINELLA_PLACE", "18,9", "9,18"),
    ("QUINELLA_PLACE", "2,1", "1,2"),
    ("BRACKET_QUINELLA", "8,1", "1,8"),
    ("BRACKET_QUINELLA", "3,3", "3,3"),  # two horses of one bracket
    ("TRIO", "9,2,5", "2,5,9"),
    ("TRIO", "15,2,10", "2,10,15"),
    ("TRIO", "3,1,2", "1,2,3"),
    ("FORECAST", "7,4", "7,4"),
    ("FORECAST", "4,7", "4,7"),
    ("FORECAST", "12,3", "12,3"),
    ("TIERCE", "9,2,5", "9,2,5"),
    ("TIERCE", "15,2,10", "15,2,10"),
    ("WIN", "7", "7"),
    ("PLACE", "12", "12"),
]


def test_a_combination_is_stored_in_the_stores_order():
    ordered = {p for p, c in jra_config()["pools"].items() if c["kind"].startswith("ordered_")}
    assert ordered == {"FORECAST", "TIERCE"}  # the pools whose legs are a finishing order
    log, path = open_log()
    stored_all = []
    for i, (pool, given, expected) in enumerate(STORE_ORDER):
        placed = bet(100, race=1 + i % 12, pool=pool, combination=given)
        caller_copy = dict(placed)
        stored = log.record(placed)
        assert placed == caller_copy, "record changed the caller's bet"
        assert stored["combination"] == expected, (
            f"{pool} {given!r} stored as {stored['combination']!r}"
        )
        as_stored = {**placed, "combination": expected}
        assert_stored(stored, as_stored, seq=i + 1, status="development", over=False, off=False)
        stored_all.append(stored)
    assert BetLog(path, day=DEV_DAY, limits=NO_FLAT).read() == stored_all
    in_file = [json.loads(line)["record"]["combination"] for line in split_lines(path.read_bytes())]
    assert in_file == [expected for _, _, expected in STORE_ORDER]


def test_a_win5_bet_is_keyed_by_its_day():
    for day in (DEV_DAY, DEV_DAY - timedelta(days=1)):
        log, path = open_log(day)
        given = [
            win5(1_000, day=day, combination="1,1,1,1,1"),  # horse 1 in each of five races
            win5(200, day=day, combination="12,3,1,7,2"),  # in race order, as given
            bet(300, day=day, race=11, combination="4"),
        ]
        stored = [log.record(dict(placed)) for placed in given]
        assert [s["race_id"] for s in stored] == [
            f"jra:{day.isoformat()}:WIN5",
            f"jra:{day.isoformat()}:WIN5",
            f"jra:{day.isoformat()}:TOK:11",
        ]
        assert [s["combination"] for s in stored] == ["1,1,1,1,1", "12,3,1,7,2", "4"]
        assert [s["pool"] for s in stored] == ["WIN5", "WIN5", "WIN"]
        assert [s["seq"] for s in stored] == [1, 2, 3]
        assert BetLog(path, day=day, limits=NO_FLAT).read() == stored
        assert log.staked_yen() == 1_500

    # the day-level key is dated too: a WIN5-only log copied to another day is not that day's log
    day, other = DEV_DAY - timedelta(days=2), DEV_DAY + timedelta(days=3)
    log, path = open_log(day)
    log.record(win5(100, day=day, combination="3,3,8,1,14"))
    target = bet_log_path(other)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    assert_tampered(target, other)


# ---------------------------------------------------------------- the cap


CHECKS = {  # name -> (cap, stakes already recorded, stake checked)
    "the ticket's case, fits exactly": (30_000, [10_000, 15_000], 5_000),
    "the ticket's case, 100 over": (30_000, [10_000, 15_000], 5_100),
    "an empty day, the whole cap": (30_000, [], 30_000),
    "an empty day, 100 over": (30_000, [], 30_100),
    "nine flat bets of 1000 under a 10000 cap": (10_000, [1_000] * 9, 1_000),
    "nine flat bets, then 1100": (10_000, [1_000] * 9, 1_100),
    "a small cap, 100 left": (5_000, [4_900], 100),
    "a small cap, 200 asked": (5_000, [4_900], 200),
    "a small stake far under": (30_000, [100, 200, 300], 100),
}


@pytest.mark.parametrize(("cap", "prior", "stake"), list(CHECKS.values()), ids=list(CHECKS))
def test_check_counts_every_bet_of_the_day(cap, prior, stake):
    limits = Limits(day_cap_yen=cap, flat_stake_yen=None)
    other_day = DEV_DAY - timedelta(days=1)
    elsewhere, _ = open_log(other_day, limits)
    elsewhere.record(bet(cap - 100 if cap > 100 else 100, day=other_day))  # another day's stakes

    path = bet_log_path(DEV_DAY)
    for i, amount in enumerate(prior):  # a fresh BetLog per bet: totals come from the file
        BetLog(path, day=DEV_DAY, limits=limits).record(bet(amount, race=1 + i % 12))
    before = snapshot(path)

    staked = sum(prior)
    expected = CapCheck(
        fits=staked + stake <= cap,
        staked_yen=staked,
        remaining_yen=cap - staked,
        off_flat=False,
    )
    log = BetLog(path, day=DEV_DAY, limits=limits)
    assert log.check(stake) == expected
    assert log.staked_yen() == staked
    assert snapshot(path) == before, "check wrote to the log"

    # the same file under another cap: the stakes are the file's, the cap is the limits'
    wider = Limits(day_cap_yen=cap + 1_000, flat_stake_yen=None)
    assert BetLog(path, day=DEV_DAY, limits=wider).check(stake) == CapCheck(
        fits=staked + stake <= cap + 1_000,
        staked_yen=staked,
        remaining_yen=cap + 1_000 - staked,
        off_flat=False,
    )


def test_a_bet_past_the_cap_is_kept_and_flagged():
    log, path = open_log()
    first = log.record(bet(29_900, race=1))
    assert first["over_cap"] is False
    assert log.check(200).fits is False

    stored = log.record(bet(200, race=2))
    assert stored["over_cap"] is True
    assert stored["seq"] == 2
    assert log.staked_yen() == 30_100
    assert [r["stake_yen"] for r in BetLog(path, day=DEV_DAY, limits=NO_FLAT).read()] == [
        29_900,
        200,
    ]

    later = log.check(100)
    assert later.fits is False and later.staked_yen == 30_100
    assert later.remaining_yen in (0, -100)  # clamped or signed: the ticket does not say which

    # flags follow the running total, computed here from the definition, under two caps
    sequences = {
        30_000: [10_000, 15_000, 5_000, 100, 1_000],
        5_000: [2_000, 3_000, 100, 4_000],
    }
    for n, (cap, stakes) in enumerate(sequences.items()):
        day = DEV_DAY + timedelta(days=1 + n)
        log, path = open_log(day, Limits(day_cap_yen=cap, flat_stake_yen=None))
        total, flags = 0, []
        for i, amount in enumerate(stakes):
            total += amount
            flags.append(total > cap)
            assert log.record(bet(amount, day=day, race=1 + i))["over_cap"] is (total > cap)
        assert [r["over_cap"] for r in log.read()] == flags
        assert log.staked_yen() == sum(stakes)
        assert True in flags and False in flags


def test_off_flat_stakes_are_flagged_only_when_a_flat_stake_is_set():
    cases = {  # flat stake -> stakes recorded in order
        200: [300, 200, 100, 200, 1_000],
        1_000: [1_000, 100, 2_000, 1_000],
        None: [100, 200, 300, 1_000, 5_000],
    }
    for n, (flat, stakes) in enumerate(cases.items()):
        day = DEV_DAY + timedelta(days=n)
        log, _ = open_log(day, Limits(day_cap_yen=CAP, flat_stake_yen=flat))
        expected = [flat is not None and s != flat for s in stakes]
        for i, stake in enumerate(stakes):
            assert log.check(stake).off_flat is expected[i]
            assert log.record(bet(stake, day=day, race=1 + i))["off_flat"] is expected[i]
        assert [r["off_flat"] for r in log.read()] == expected
    # the ticket's case
    log, _ = open_log(DEV_DAY + timedelta(days=5), Limits(day_cap_yen=CAP, flat_stake_yen=200))
    assert log.record(bet(300, day=DEV_DAY + timedelta(days=5)))["off_flat"] is True


# ---------------------------------------------------------------- refused bets


@pytest.mark.parametrize("stake", [0, -100, 150, 100.0, 50, 99, 1, 101, 1_050, "100", None, 2e3])
def test_stakes_are_whole_hundreds_of_yen(stake):
    assert issubclass(BetRefused, ValueError)
    log, path = open_log()
    assert_refused(log, path, bet(stake))  # as the first bet of the day
    log.record(bet(100, race=1))
    assert_refused(log, path, bet(stake, race=2))
    assert log.record(bet(1_000, race=3))["seq"] == 2  # a refusal uses no seq
    assert log.staked_yen() == 1_100


MALFORMED = {
    "a race id dated another day": bet(race_id="jra:2026-11-23:TOK:11"),
    "a race id dated the day before": bet(race_id="jra:2026-11-21:TOK:11"),
    "an unknown venue": bet(venue="XXX"),
    "a race id of another source": bet(race_id="nar:2026-11-22:TOK:11"),
    "a race id without a race number": bet(race_id="jra:2026-11-22:TOK"),
    "a race id that is not text": bet(race_id=11),
    "an unknown pool": bet(pool="SHOW"),
    "a blank pool": bet(pool=""),
    "a blank combination": bet(combination=""),
    "a whitespace combination": bet(combination="   "),
    "a combination that is not text": bet(combination=4),
    "an unknown key": bet(odds=3.4),
    "a stake under another name": {**bet(), "stake": 100},
    "a caller-set seq": bet(seq=1),
    "a caller-set over_cap": bet(over_cap=False),
    "a caller-set off_flat": bet(off_flat=False),
    "a caller-set day_status": bet(day_status="development"),
    "a caller-set written_utc": bet(written_utc="2026-11-22T05:40:00+00:00"),
    "a placed_utc without an offset": bet(placed="2026-11-22T05:40:00"),
    "a placed_utc that is a date only": bet(placed="2026-11-22"),
    "a placed_utc that is not ISO": bet(placed="Sunday afternoon"),
    "a placed_utc that is not text": bet(placed=datetime(2026, 11, 22, 5, 40, tzinfo=UTC)),
    "not a dict": [bet()],
    # combinations: the wrong number of legs for the pool
    "two horses in WIN": bet(pool="WIN", combination="4,7"),
    "one horse in QUINELLA": bet(pool="QUINELLA", combination="4"),
    "two horses in TRIO": bet(pool="TRIO", combination="1,2"),
    "four races in WIN5": win5(combination="1,2,3,4"),
    # combinations: not horse numbers
    "a dash between horses": bet(pool="QUINELLA", combination="4-7"),
    "a letter": bet(combination="a"),
    "horse 0": bet(combination="0"),
    "a leg of 0": bet(pool="QUINELLA", combination="4,0"),
    "a leading zero": bet(combination="04"),
    "a leading zero in a leg": bet(pool="QUINELLA", combination="4,07"),
    # combinations: one horse twice (BRACKET_QUINELLA and WIN5 take repeats)
    "a repeat in WIN": bet(pool="WIN", combination="4,4"),
    "a repeat in PLACE": bet(pool="PLACE", combination="4,4"),
    "a repeat in QUINELLA": bet(pool="QUINELLA", combination="4,4"),
    "a repeat in QUINELLA_PLACE": bet(pool="QUINELLA_PLACE", combination="4,4"),
    "a repeat in FORECAST": bet(pool="FORECAST", combination="4,4"),
    "a repeat in TRIO": bet(pool="TRIO", combination="4,7,4"),
    "one horse three times in TRIO": bet(pool="TRIO", combination="7,7,7"),
    "a repeat in TIERCE": bet(pool="TIERCE", combination="4,7,4"),
    "a repeat first in TIERCE": bet(pool="TIERCE", combination="4,4,7"),
    # race ids: race 00, and the WIN5 key for WIN5 alone
    "race number 00": bet(race_id="jra:2026-11-22:TOK:00"),
    "a WIN5 bet keyed by one race": bet(pool="WIN5", combination="1,2,3,4,5"),
    "a WIN5 key of another day": {**win5(), "race_id": "jra:2026-11-23:WIN5"},
    "a WIN bet on the WIN5 key": bet(race_id="jra:2026-11-22:WIN5"),
    "a TRIO bet on the WIN5 key": bet(
        pool="TRIO", combination="1,2,3", race_id="jra:2026-11-22:WIN5"
    ),
    # option_id and note
    "a blank option_id": bet(option_id=""),
    "a whitespace option_id": bet(option_id="   "),
    "an option_id that is not text": bet(option_id=7),
    "a note that is a number": bet(note=5),
    "a note that is a list": bet(note=["own pick"]),
}
REQUIRED = ("race_id", "pool", "combination", "stake_yen", "placed_utc")


@pytest.mark.parametrize("bad", list(MALFORMED.values()), ids=list(MALFORMED))
def test_a_malformed_bet_is_refused_and_nothing_written(bad):
    log, path = open_log()
    assert_refused(log, path, bad)
    log.record(bet(1_000, race=1))
    assert_refused(log, path, bad)
    assert log.record(bet(1_000, race=2))["seq"] == 2
    assert log.staked_yen() == 2_000


@pytest.mark.parametrize("missing", REQUIRED)
def test_a_bet_missing_a_field_is_refused(missing):
    log, path = open_log()
    log.record(bet(100, race=1))
    bad = bet(100, race=2)
    del bad[missing]
    assert_refused(log, path, bad)
    assert log.record(bet(100, race=2))["seq"] == 2


ACCEPTED_INSTANTS = {
    "Z": "2026-11-22T05:40:00Z",
    "+00:00": "2026-11-22T05:40:00+00:00",
    "+09:00, JST": "2026-11-22T14:40:00+09:00",
    "fractional seconds": "2026-11-22T05:40:00.250000+00:00",
}


@pytest.mark.parametrize("placed", list(ACCEPTED_INSTANTS.values()), ids=list(ACCEPTED_INSTANTS))
def test_a_placed_utc_with_any_offset_is_accepted(placed):
    log, _ = open_log()
    stored = log.record(bet(100, placed=placed, option_id=None))
    assert instant(stored["placed_utc"]) == instant(placed)
    assert log.read() == [stored]


# ---------------------------------------------------------------- the chain


def three_bets() -> tuple[Path, bytes]:
    """Three bets through two BetLogs on the same file (lines 1-3)."""
    log, path = open_log()
    log.record(bet(10_000, race=1))
    again = BetLog(path, day=DEV_DAY, limits=NO_FLAT)  # a restart carries on the chain
    second = again.record(bet(2_000, race=2, pool="PLACE", combination="7"))
    assert second["seq"] == 2
    again.record(bet(300, race=3, pool="QUINELLA", combination="4,7", note="中山"))
    raw = path.read_bytes()
    lines = split_lines(raw)  # the line format itself is pinned by chain(), where a test needs it
    assert len(lines) == 3 and json.loads(lines[1])["prev_sha256"] == sha(lines[0])
    return path, raw


def only_bet() -> tuple[Path, bytes]:
    """A log of one bet of 1000 yen."""
    log, path = open_log()
    log.record(bet(1_000, race=1))
    return path, path.read_bytes()


def flip_byte(pattern: bytes) -> Callable[[bytes], bytes]:
    """Change the last byte ``pattern`` matches (a digit or a hex character) to another one."""

    def edit(line: bytes) -> bytes:
        match = re.search(pattern, line)
        assert match, f"{pattern!r} not found in {line!r}"
        i = match.end() - 1
        new = b"8" if line[i : i + 1] != b"8" else b"9"
        return line[:i] + new + line[i + 1 :]

    return edit


def respace(line: bytes) -> bytes:
    new = json.dumps(json.loads(line), separators=(" , ", " : ")).encode()
    assert new != line and json.loads(new) == json.loads(line)
    return new


STAKE = rb'"stake_yen"\s*:\s*\d+'
PREV = rb'"prev_sha256"\s*:\s*"[0-9a-f]'
EDITS: dict[str, Callable[[list[bytes]], list[bytes]]] = {
    "one byte of the stake in line 1": lambda ls: [flip_byte(STAKE)(ls[0]), *ls[1:]],
    "one byte of the race in line 1": lambda ls: [flip_byte(rb"TOK:\d\d")(ls[0]), *ls[1:]],
    "prev_sha256 of line 1": lambda ls: [flip_byte(PREV)(ls[0]), *ls[1:]],
    "prev_sha256 of line 2": lambda ls: [ls[0], flip_byte(PREV)(ls[1]), ls[2]],
    "the stake in line 2": lambda ls: [ls[0], flip_byte(STAKE)(ls[1]), ls[2]],
    "line 1 re-spaced, same JSON": lambda ls: [respace(ls[0]), *ls[1:]],
    "line 2 no longer JSON": lambda ls: [ls[0], b"#" + ls[1][1:], ls[2]],
    "line 1 cut short": lambda ls: [ls[0][:-7], *ls[1:]],
    "line 1 deleted": lambda ls: ls[1:],
    "line 2 deleted": lambda ls: [ls[0], ls[2]],
    "lines 1 and 2 swapped": lambda ls: [ls[1], ls[0], ls[2]],
    "line 1 replayed after it": lambda ls: [ls[0], ls[0], *ls[1:]],
}


@pytest.mark.parametrize("edit", list(EDITS.values()), ids=list(EDITS))
def test_the_log_is_append_only_and_chained(edit):
    path, raw = three_bets()
    tampered = join_lines(edit(split_lines(raw)))
    assert tampered != raw
    path.write_bytes(tampered)
    calls = {  # each through a fresh BetLog, so a log that verifies on open passes too
        "read": lambda log: log.read(),
        "record": lambda log: log.record(bet(100, race=9)),
        "staked_yen": lambda log: log.staked_yen(),
        "check": lambda log: log.check(100),
    }
    for name, call in calls.items():
        with pytest.raises(BetLogTampered):
            call(BetLog(path, day=DEV_DAY, limits=NO_FLAT))
        assert path.read_bytes() == tampered, f"{name} wrote onto a broken chain"


def swap(old: bytes, new: bytes) -> Callable[[bytes], bytes]:
    """Replace ``old`` by ``new`` of the same length, once: a one-byte edit that keeps the line
    well formed."""

    def edit(line: bytes) -> bytes:
        assert line.count(old) == 1 and len(old) == len(new), f"{old!r} in {line!r}"
        return line.replace(old, new)

    return edit


LAST_LINE_EDITS = {
    "the only bet's stake, 1000 to 2000": (
        only_bet,
        swap(b'"stake_yen":1000', b'"stake_yen":2000'),
    ),
    "the last bet's stake, 300 to 800": (three_bets, swap(b'"stake_yen":300', b'"stake_yen":800')),
    "the last bet's race, 03 to 04": (three_bets, swap(b":TOK:03", b":TOK:04")),
    "the last bet's combination, 4,7 to 4,8": (three_bets, swap(b'"4,7"', b'"4,8"')),
}


@pytest.mark.parametrize(
    ("build", "edit"), list(LAST_LINE_EDITS.values()), ids=list(LAST_LINE_EDITS)
)
def test_an_edit_to_the_last_line_is_detected(build, edit):
    path, raw = build()
    lines = split_lines(raw)
    tampered = join_lines([*lines[:-1], edit(lines[-1])])
    assert tampered != raw and len(tampered) == len(raw)
    path.write_bytes(tampered)
    assert_tampered(path)


SHA = rb'"sha256"\s*:\s*"[0-9a-f]'  # the first hex digit of a line's own sha256


def other_hex(digest: str) -> str:
    return ("8" if digest[0] != "8" else "9") + digest[1:]


def on_last(edit: Callable[[bytes], bytes]) -> Callable[[bytes], bytes]:
    def apply(raw: bytes) -> bytes:
        lines = split_lines(raw)
        return join_lines([*lines[:-1], edit(lines[-1])])

    return apply


def with_sha(value: str | None) -> Callable[[bytes], bytes]:
    """The line with its sha256 set to ``value``, or without one (None), re-encoded."""

    def edit(line: bytes) -> bytes:
        obj = json.loads(line)
        assert "sha256" in obj
        if value is None:
            del obj["sha256"]
        else:
            obj["sha256"] = value
        return encode(obj)

    return edit


def rechained(n: int, change: Callable[[dict], object]) -> Callable[[bytes], bytes]:
    """Change line ``n`` (from 0), then recompute every hash after it, as an editor who knows the
    format would, but leave line ``n``'s own sha256 as the change left it."""

    def apply(raw: bytes) -> bytes:
        objs = lines_of(raw)
        assert rechain(objs) == raw, "the test's model of the format is not the log's"
        change(objs[n])
        return rechain(objs, keep=frozenset({n}))

    return apply


CHECKSUM_EDITS = {
    "the only line's sha256, one hex digit": (only_bet, on_last(flip_byte(SHA))),
    "the only line's sha256 missing": (only_bet, on_last(with_sha(None))),
    "the last line's sha256, one hex digit": (three_bets, on_last(flip_byte(SHA))),
    "the last line's sha256 missing": (three_bets, on_last(with_sha(None))),
    "the last line's sha256 all zeros": (three_bets, on_last(with_sha(ZERO))),
    "the last line's sha256 copied from line 2": (
        three_bets,
        lambda raw: on_last(with_sha(lines_of(raw)[1]["sha256"]))(raw),
    ),
    "line 1's stake edited, the hashes after it recomputed": (
        three_bets,
        rechained(0, lambda obj: obj["record"].update(stake_yen=20_000)),
    ),
    "line 1's sha256 missing, the hashes after it recomputed": (
        three_bets,
        rechained(0, lambda obj: obj.pop("sha256")),
    ),
    "line 2's sha256 wrong, the hashes after it recomputed": (
        three_bets,
        rechained(1, lambda obj: obj.update(sha256=other_hex(obj["sha256"]))),
    ),
}


@pytest.mark.parametrize(("build", "edit"), list(CHECKSUM_EDITS.values()), ids=list(CHECKSUM_EDITS))
def test_each_line_carries_its_own_sha256(build, edit):
    path, raw = build()
    chain(raw)  # every line carries a sha256 of itself
    tampered = edit(raw)
    assert tampered != raw
    path.write_bytes(tampered)
    assert_tampered(path)


COPIES = {  # name -> (bets in the log, days from the log's day to the day it is copied to)
    "one bet, to the next day": (1, 1),
    "three bets, to a week before": (3, -7),
    "three bets, to a year later": (3, 365),
}


@pytest.mark.parametrize(("n", "offset"), list(COPIES.values()), ids=list(COPIES))
def test_a_log_copied_to_another_day_is_tampered(n, offset):
    log, path = open_log()
    bets = [
        bet(1_000, race=1),
        bet(2_000, race=2, pool="PLACE", combination="7"),
        bet(300, race=3, pool="QUINELLA", combination="4,7"),
    ][:n]
    for placed in bets:
        log.record(placed)
    stakes = [placed["stake_yen"] for placed in bets]
    other = DEV_DAY + timedelta(days=offset)
    target = bet_log_path(other)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())  # the right file name, the chain intact, another day
    assert_tampered(target, other)
    assert BetLog(path, day=DEV_DAY, limits=NO_FLAT).staked_yen() == sum(stakes)  # its own day


def test_one_record_dated_another_day_is_tampered():
    path, raw = three_bets()
    objs = lines_of(raw)
    assert rechain(objs) == raw, "the test's model of the format is not the log's"
    objs[1]["record"]["race_id"] = "jra:2026-11-23:TOK:02"  # the day after, hashes recomputed
    path.write_bytes(rechain(objs))
    assert_tampered(path)


def test_a_cut_off_last_line_is_detected():
    path, raw = three_bets()
    path.write_bytes(raw[:-1])  # no newline after the last line
    with pytest.raises(BetLogTampered):
        BetLog(path, day=DEV_DAY, limits=NO_FLAT).read()
    path.write_bytes(raw[: len(raw) - 20])  # the last line cut part way
    with pytest.raises(BetLogTampered):
        BetLog(path, day=DEV_DAY, limits=NO_FLAT).record(bet(100, race=9))
    assert path.read_bytes() == raw[: len(raw) - 20]


def test_earlier_bytes_never_change_and_each_bet_is_fsynced(monkeypatch):
    fsyncs: list[int] = []
    real_fsync = os.fsync

    def counting_fsync(fd):
        fsyncs.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    log, path = open_log(limits=Limits(day_cap_yen=5_000, flat_stake_yen=1_000))
    stakes = [1_000, 1_000, 200, 3_000, 1_000]
    for i, stake in enumerate(stakes):
        if i in (2, 4):
            log = BetLog(path, day=DEV_DAY, limits=Limits(day_cap_yen=5_000, flat_stake_yen=1_000))
        before = snapshot(path) or b""
        calls = len(fsyncs)
        log.record(bet(stake, race=1 + i))
        assert len(fsyncs) > calls, "each bet is fsynced before record returns"
        after = path.read_bytes()
        assert after[: len(before)] == before, f"bet {i + 1} changed earlier bytes"
        assert after.count(b"\n") == before.count(b"\n") + 1, "one line per bet"
    records = [obj["record"] for obj in chain(path.read_bytes())]
    assert [r["stake_yen"] for r in records] == stakes
    assert [r["over_cap"] for r in records] == [False, False, False, True, True]
    assert [r["off_flat"] for r in records] == [False, False, True, True, False]


# ---------------------------------------------------------------- the previous line format

# Format 1 is the line a565ac0 wrote before review round 1 gave each line its own sha256: exactly
# {seq, prev_sha256, record}, chained as format 2 is. Review round 3: a day begun in format 1
# carries on in format 2, append-only. The lines are built here from first principles.


def stored_as(seq: int, placed: dict, *, written: str, status: str, over: bool, off: bool) -> dict:
    """A stored record: the bet as given, then the five fields the log adds, in the log's order."""
    added = {"seq": seq, "written_utc": written, "day_status": status}
    return {**placed, **added, "over_cap": over, "off_flat": off}


def legacy(seq: int, placed: dict, *, status="development", over=False, off=False) -> dict:
    """A record as a565ac0 stored it (the combination as given, never reordered), written while
    format 1 was the branch's format (a565ac0 to f366f20: 2026-11-27, 09:02 to 09:31 UTC)."""
    written = f"2026-11-27T09:{10 + seq:02d}:00.125000+00:00"
    return stored_as(seq, placed, written=written, status=status, over=over, off=off)


def current(seq: int, placed: dict) -> dict:
    """A record as the log stores it today, development day, no flags."""
    written = f"2026-11-28T03:{seq:02d}:00.500000+00:00"
    return stored_as(seq, placed, written=written, status="development", over=False, off=False)


def lines_from(
    entries: list[tuple[int, dict]], *, seqs: list[int] | None = None, first_prev: str = ZERO
) -> list[bytes]:
    """The lines of ``entries``, (format, record) pairs, chained from ``first_prev`` and numbered
    from 1 (or by ``seqs``): format 1 is ``{seq, prev_sha256, record}``; format 2 adds the sha256
    of that encoding."""
    out, prev = [], first_prev
    for i, (fmt, record) in enumerate(entries):
        body = {"seq": seqs[i] if seqs else i + 1, "prev_sha256": prev, "record": record}
        line = encode(body if fmt == 1 else {**body, "sha256": sha(encode(body))})
        out.append(line)
        prev = sha(line)
    return out


# name -> (day, the day's format-1 records): bets a565ac0 took, some of which format 2 refuses
LEGACY_DAYS = {
    "five bets, 29000 yen": (
        DEV_DAY,
        [
            legacy(1, bet(20_000, race=1, option_id="lb:win:1")),
            legacy(2, bet(5_000, venue="NAK", race=11, pool="QUINELLA", combination="4,2")),
            legacy(3, bet(2_000, venue="NAK", race=11, pool="WIN5", combination="3,3,8,1,14")),
            legacy(4, bet(1_500, venue="KYO", race=5, pool="TRIO", combination="7,7,3")),
            legacy(
                5,
                bet(500, venue="KYO", race=6, pool="FORECAST", combination="7,4", note="中山"),
                off=True,
            ),
        ],
    ),
    "one bet, the day before": (
        DEV_DAY - timedelta(days=1),
        [legacy(1, bet(1_000, day=DEV_DAY - timedelta(days=1), venue="HSN", combination="12"))],
    ),
    "two bets labelled holdout, 29900 yen": (
        DEV_DAY - timedelta(days=2),
        [
            legacy(1, bet(29_000, day=DEV_DAY - timedelta(days=2), race=1), status="holdout"),
            legacy(
                2,
                bet(900, day=DEV_DAY - timedelta(days=2), race=2, pool="PLACE", option_id=None),
                status="holdout",
            ),
        ],
    ),
}


@pytest.mark.parametrize(("day", "records"), list(LEGACY_DAYS.values()), ids=list(LEGACY_DAYS))
def test_a_day_begun_in_the_previous_format_carries_on(day, records):
    path = bet_log_path(day)
    path.parent.mkdir(parents=True)
    old = lines_from([(1, record) for record in records])
    prefix = join_lines(old)
    path.write_bytes(prefix)
    total = sum(record["stake_yen"] for record in records)
    held = any(record["day_status"] == "holdout" for record in records)

    # the format-1 bets are the day's bets, read back exactly as stored, and count toward the cap
    log = BetLog(path, day=day, limits=NO_FLAT)
    assert log.read() == records
    assert log.staked_yen() == total
    left = CAP - total
    assert log.check(left) == CapCheck(
        fits=True, staked_yen=total, remaining_yen=left, off_flat=False
    )
    assert log.check(left + 100).fits is False
    assert path.read_bytes() == prefix

    # the day carries on in format 2, chained onto the last format-1 line, nothing rewritten
    added, prev, k = [], sha(old[-1]), len(records)
    for i, (stake, extra) in enumerate([(1_000, {}), (100, {"pool": "PLACE", "combination": "2"})]):
        placed = bet(stake, day=day, race=12, **extra)
        before = path.read_bytes()
        writer = log if i == 0 else BetLog(path, day=day, limits=NO_FLAT)
        record = writer.record(dict(placed))
        total += stake
        status = "holdout" if held or is_holdout_day("japan", day) else "development"
        assert_stored(record, placed, seq=k + 1 + i, status=status, over=total > CAP, off=False)
        after = path.read_bytes()
        assert after[: len(before)] == before and after[: len(prefix)] == prefix
        lines = split_lines(after)
        assert len(lines) == k + 1 + i
        obj = json.loads(lines[-1])
        assert list(obj) == ["seq", "prev_sha256", "record", "sha256"], "the new line is format 2"
        assert lines[-1] == encode(obj)
        assert obj["seq"] == k + 1 + i
        assert obj["prev_sha256"] == prev, "not chained onto the line before it"
        assert obj["sha256"] == line_sha(obj)
        assert obj["record"] == record
        prev = sha(lines[-1])
        added.append(record)

    fresh = BetLog(path, day=day, limits=NO_FLAT)
    assert fresh.read() == records + added
    assert fresh.staked_yen() == total
    assert fresh.check(100).staked_yen == total


def legacy_three() -> list[tuple[int, dict]]:
    """Three format-1 lines whose records format 2 would take too, so each edit below is what
    breaks the file."""
    return [
        (1, legacy(1, bet(10_000, race=1))),
        (1, legacy(2, bet(2_000, race=2, pool="QUINELLA", combination="4,7"))),
        (1, legacy(3, bet(300, race=3, pool="PLACE", combination="9"))),
    ]


def upgraded() -> list[tuple[int, dict]]:
    """Two format-1 lines, then two format-2 lines."""
    first, second, _ = legacy_three()
    return [
        first,
        second,
        (2, current(3, bet(1_000, race=4))),
        (2, current(4, bet(500, race=5, pool="QUINELLA", combination="2,6"))),
    ]


def recomputed(change: Callable[[list[tuple[int, dict]]], object], **kw) -> Callable:
    """``change`` applied to a copy of the entries, then every line and hash rebuilt."""

    def apply(entries: list[tuple[int, dict]]) -> bytes:
        copy = [(fmt, json.loads(json.dumps(record))) for fmt, record in entries]
        change(copy)
        return join_lines(lines_from(copy, **kw))

    return apply


def record_set(n: int, **changes) -> Callable:
    """Record ``n`` (from 0) with ``changes``, every hash recomputed."""
    return recomputed(lambda entries: entries[n][1].update(changes))


def on_built(edit: Callable[[list[bytes]], list[bytes]]) -> Callable:
    """``edit`` applied to the built lines, no hash recomputed."""
    return lambda entries: join_lines(edit(lines_from(entries)))


def last_stripped(**changes) -> Callable:
    """The last line without its sha256, its record changed by ``changes``, its prev kept."""

    def edit(lines: list[bytes]) -> list[bytes]:
        obj = json.loads(lines[-1])
        del obj["sha256"]
        obj["record"].update(changes)
        return [*lines[:-1], encode(obj)]

    return on_built(edit)


def renumbered(seqs: list[int]) -> Callable:
    def change(entries):
        for (_, record), seq in zip(entries, seqs, strict=True):
            record["seq"] = seq

    return recomputed(change, seqs=seqs)


LEGACY_TAMPERED = {  # name -> (the file as written, the edit)
    # the format order: format 1 only before the first format-2 line
    "a format-1 line after the format-2 lines": (
        upgraded,
        recomputed(lambda e: e.append((1, legacy(5, bet(100, race=6))))),
    ),
    "a format-2 line between format-1 lines": (
        legacy_three,
        recomputed(lambda e: e.__setitem__(1, (2, e[1][1]))),
    ),
    "the last format-2 line stripped of its sha256": (upgraded, last_stripped()),
    "the last format-2 line stripped of its sha256, its stake edited": (
        upgraded,
        last_stripped(stake_yen=800),
    ),
    # the chain through the format-1 lines
    "a format-1 stake edited, a format-1 line after it": (
        legacy_three,
        on_built(lambda ls: [swap(b'"stake_yen":10000', b'"stake_yen":80000')(ls[0]), *ls[1:]]),
    ),
    "the last format-1 stake edited, a format-2 line after it": (
        upgraded,
        on_built(
            lambda ls: [ls[0], swap(b'"stake_yen":2000', b'"stake_yen":8000')(ls[1]), *ls[2:]]
        ),
    ),
    "a format-1 line deleted": (legacy_three, on_built(lambda ls: [ls[0], ls[2]])),
    "the first format-1 line not chained onto zeros": (
        legacy_three,
        recomputed(lambda e: None, first_prev=sha(b"another file")),
    ),
    "format-1 seqs skipping one, hashes recomputed": (legacy_three, renumbered([1, 3, 4])),
    "format-1 seqs from 2, hashes recomputed": (legacy_three, renumbered([2, 3, 4])),
    "a format-1 record's seq not its line's": (legacy_three, record_set(1, seq=3)),
    "a format-1 line re-spaced, same JSON": (
        legacy_three,
        on_built(lambda ls: [respace(ls[0]), *ls[1:]]),
    ),
    # records a565ac0 could not have written, hashes recomputed
    "a format-1 record dated another day": (
        legacy_three,
        record_set(1, race_id="jra:2026-11-23:TOK:02"),
    ),
    "a format-1 record of an unknown pool": (legacy_three, record_set(1, pool="SHOW")),
    "a format-1 record at an unknown venue": (
        legacy_three,
        record_set(1, race_id="jra:2026-11-22:XXX:02"),
    ),
    "a format-1 WIN5 bet on the day key, which a565ac0 did not take": (
        legacy_three,
        record_set(1, race_id="jra:2026-11-22:WIN5", pool="WIN5", combination="1,2,3,4,5"),
    ),
    "a format-1 combination of the wrong number of legs": (
        legacy_three,
        record_set(1, combination="4"),
    ),
    "a format-1 stake JRA does not sell": (legacy_three, record_set(1, stake_yen=150)),
    "a format-1 placed_utc without an offset": (
        legacy_three,
        record_set(1, placed_utc="2026-11-22T05:40:00"),
    ),
    "a format-1 blank option_id": (legacy_three, record_set(1, option_id="")),
    "a format-1 record with a key the log does not write": (legacy_three, record_set(1, odds=3.4)),
    "a format-1 record without written_utc": (
        legacy_three,
        recomputed(lambda e: e[1][1].pop("written_utc")),
    ),
    "a format-1 flag that is not a boolean": (legacy_three, record_set(1, over_cap=0)),
}


@pytest.mark.parametrize(
    ("base", "edit"), list(LEGACY_TAMPERED.values()), ids=list(LEGACY_TAMPERED)
)
def test_a_previous_format_prefix_is_verified(base, edit):
    entries = base()
    path = bet_log_path(DEV_DAY)
    path.parent.mkdir(parents=True)
    good = join_lines(lines_from(entries))
    path.write_bytes(good)
    records = [record for _, record in entries]
    log = BetLog(path, day=DEV_DAY, limits=NO_FLAT)  # unedited, the file is the day's log
    assert log.read() == records
    assert log.staked_yen() == sum(record["stake_yen"] for record in records)

    tampered = edit(entries)
    assert tampered != good
    assert [record for _, record in entries] == records, "the edit changed the base"
    path.write_bytes(tampered)
    assert_tampered(path)


# ---------------------------------------------------------------- where it lives


def test_the_log_path_is_under_the_data_dir(D, tmp_path, monkeypatch):
    for day in (DEV_DAY, date(2026, 12, 5), date(2027, 1, 3), date(2026, 2, 1)):
        assert bet_log_path(day) == D / "bets" / "japan" / f"{day.isoformat()}.jsonl"

    elsewhere = (tmp_path / "elsewhere").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(elsewhere))  # read on every call
    assert bet_log_path(DEV_DAY) == elsewhere / "bets" / "japan" / "2026-11-22.jsonl"
    log = BetLog(bet_log_path(DEV_DAY), day=DEV_DAY, limits=NO_FLAT)
    log.record(bet(100))
    assert (elsewhere / "bets" / "japan" / "2026-11-22.jsonl").is_file()
    assert not (D / "bets").exists()


def test_a_day_has_one_log_at_its_bet_log_path(D, tmp_path, monkeypatch):
    name = f"{DEV_DAY.isoformat()}.jsonl"
    own = bet_log_path(DEV_DAY)
    BetLog(own, day=DEV_DAY, limits=NO_FLAT).record(bet(1_000, race=1))

    # other spellings of the same file are the same log: compared after resolving
    alias = tmp_path / "alias"
    alias.symlink_to(own.parent, target_is_directory=True)
    same = {
        "through ..": D / "bets" / "japan" / ".." / "japan" / name,
        "through a symlink": alias / name,
        "relative to the working directory": Path(os.path.relpath(own)),
    }
    for i, path in enumerate(same.values()):
        BetLog(path, day=DEV_DAY, limits=NO_FLAT).record(bet(100 * (i + 1), race=2 + i))
    assert BetLog(own, day=DEV_DAY, limits=NO_FLAT).staked_yen() == 1_600
    assert len(split_lines(own.read_bytes())) == 4

    # the day's file name anywhere else is refused, and nothing is created
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "elsewhere-link").symlink_to(elsewhere, target_is_directory=True)
    refused = {
        "outside the data dir": tmp_path / "ok" / name,
        "the data dir's root": D / name,
        "the bets folder": D / "bets" / name,
        "a folder under the day's": D / "bets" / "japan" / "old" / name,
        "another country's folder": D / "bets" / "hongkong" / name,
        "the shadow log's folder": D / "shadow" / "japan" / name,
        "another data dir's layout": tmp_path / "data-2" / "bets" / "japan" / name,
        "a symlink to another folder": tmp_path / "elsewhere-link" / name,
    }
    before = own.read_bytes()
    for where, path in refused.items():
        with pytest.raises(ValueError) as err:
            BetLog(path, day=DEV_DAY, limits=NO_FLAT)
        assert not isinstance(err.value, BetLogInRepo), where
        assert not path.exists(), where
    for folder in (
        tmp_path / "ok",
        tmp_path / "data-2",
        D / "bets" / "japan" / "old",
        D / "bets" / "hongkong",
        D / "shadow",
    ):
        assert not folder.exists(), f"a refused path created {folder}"
    assert not (D / name).exists() and not (D / "bets" / name).exists()
    assert list(elsewhere.iterdir()) == []
    assert own.read_bytes() == before

    # the data dir is read on every call: once it moves, the old file is no longer the day's log
    moved = (tmp_path / "moved").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(moved))
    with pytest.raises(ValueError):
        BetLog(own, day=DEV_DAY, limits=NO_FLAT)
    assert BetLog(bet_log_path(DEV_DAY), day=DEV_DAY, limits=NO_FLAT).staked_yen() == 0


def test_an_open_log_refuses_every_call_once_the_data_dir_moves(D, tmp_path, monkeypatch):
    # review fix: an open BetLog re-resolves bet_log_path(day) on every call, so a process holding
    # one cannot stake into the old file while a fresh check() at the new place sees an empty day
    log, path = open_log()
    first = log.record(bet(1_000, race=1))
    before = path.read_bytes()

    moved = (tmp_path / "moved").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(moved))
    assert bet_log_path(DEV_DAY) == moved / "bets" / "japan" / path.name != path
    calls = {
        "record": lambda: log.record(bet(100, race=2)),
        "read": lambda: log.read(),
        "check": lambda: log.check(100),
        "staked_yen": lambda: log.staked_yen(),
    }
    for name, call in calls.items():
        try:
            call()
        except ValueError:  # the constructor's one-path refusal
            pass
        else:
            pytest.fail(f"{name} on an open log ran after {paths.DATA_DIR_VAR} moved")
        assert path.read_bytes() == before, f"{name} wrote to the old log"
        assert not moved.exists(), f"{name} created something under the new data dir"

    fresh = BetLog(bet_log_path(DEV_DAY), day=DEV_DAY, limits=NO_FLAT)
    assert fresh.staked_yen() == 0
    assert fresh.check(100) == CapCheck(fits=True, staked_yen=0, remaining_yen=CAP, off_flat=False)

    # moved back, the same object works again: the setting is read on each call, not cached
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(D))
    assert log.read() == [first]
    assert log.staked_yen() == 1_000
    assert log.check(100).staked_yen == 1_000
    assert log.record(bet(100, race=2))["seq"] == 2
    assert BetLog(path, day=DEV_DAY, limits=NO_FLAT).staked_yen() == 1_100
    assert not moved.exists()


def case_insensitive(folder: Path) -> bool:
    """Whether ``folder``'s filesystem ignores letter case (macOS's default APFS does, Linux's
    ext4 does not)."""
    probe = folder / "case-probe"
    probe.touch()
    try:
        return (folder / "CASE-PROBE").exists()
    finally:
        probe.unlink()


def test_an_open_log_takes_the_data_dir_in_another_letter_case(D, monkeypatch):
    # review round 3: the setting spelled in another case can name the same directory; the open log
    # and a fresh one then use the same file, compared by filesystem identity
    log, path = open_log()
    first = log.record(bet(1_000, race=1))
    spellings = {
        "the data dir's name": D.with_name(D.name.swapcase()),
        "the data dir's name, mixed": D.with_name("dAtA"),
        "a folder above it": D.parent.with_name(D.parent.name.swapcase()) / D.name,
    }
    assert all(str(spelled) != str(D) for spelled in spellings.values())

    if not case_insensitive(D.parent):
        # here another spelling is another directory, refused as a moved data dir is
        before = path.read_bytes()
        for how, spelled in spellings.items():
            monkeypatch.setenv(paths.DATA_DIR_VAR, str(spelled))
            for call in (lambda: log.record(bet(100, race=2)), log.read, log.staked_yen):
                with pytest.raises(ValueError):
                    call()
            assert not spelled.exists(), how
        assert path.read_bytes() == before
        return

    recorded, total = [first], 1_000
    for i, (how, spelled) in enumerate(spellings.items()):
        assert os.path.samefile(spelled, D), how
        monkeypatch.setenv(paths.DATA_DIR_VAR, str(spelled))
        assert bet_log_path(DEV_DAY) != path, how  # another spelling of the same file
        assert log.read() == recorded, how
        assert log.staked_yen() == total, how
        assert log.check(100) == CapCheck(
            fits=True, staked_yen=total, remaining_yen=CAP - total, off_flat=False
        ), how
        stake = 100 * (i + 2)
        recorded.append(log.record(bet(stake, race=2 + i)))
        total += stake
        assert recorded[-1]["seq"] == i + 2, how
        for given in (bet_log_path(DEV_DAY), path):  # either spelling is the day's log
            assert BetLog(given, day=DEV_DAY, limits=NO_FLAT).read() == recorded, how
    assert [obj["record"] for obj in chain(path.read_bytes())] == recorded
    assert [p.name for p in D.parent.iterdir()] == [D.name], "one data dir, under its own name"


@pytest.mark.parametrize("made", [True, False], ids=["the data dir made", "nothing made yet"])
def test_a_day_before_its_first_bet_takes_another_letter_case(D, monkeypatch, made):
    if not case_insensitive(D.parent):
        pytest.skip("a case-sensitive filesystem: another spelling is another directory")
    if made:  # the data dir exists, bets/japan does not
        D.mkdir()
        spelled = D.with_name(D.name.swapcase())
    else:  # only the folder above the data dir exists
        spelled = D.parent.with_name(D.parent.name.swapcase()) / D.name
    name = f"{DEV_DAY.isoformat()}.jsonl"
    own_path = D / "bets" / "japan" / name

    early = BetLog(spelled / "bets" / "japan" / name, day=DEV_DAY, limits=NO_FLAT)
    assert not (D / "bets").exists(), "the constructor created the log's folder"
    assert early.read() == [] and early.staked_yen() == 0
    assert early.check(CAP) == CapCheck(fits=True, staked_yen=0, remaining_yen=CAP, off_flat=False)
    own = BetLog(bet_log_path(DEV_DAY), day=DEV_DAY, limits=NO_FLAT)

    monkeypatch.setenv(paths.DATA_DIR_VAR, str(spelled))  # the setting in the other spelling
    assert own.staked_yen() == 0 and own.check(100).fits is True
    first = own.record(bet(1_000, race=1))
    assert first["seq"] == 1
    assert early.read() == [first]
    second = early.record(bet(200, race=2))
    assert second["seq"] == 2
    assert BetLog(own_path, day=DEV_DAY, limits=NO_FLAT).staked_yen() == 1_200

    monkeypatch.setenv(paths.DATA_DIR_VAR, str(D))
    assert own.read() == early.read() == [first, second]
    assert [obj["record"] for obj in chain(own_path.read_bytes())] == [first, second]
    assert [p.name for p in D.parent.iterdir()] == [D.name], "one data dir, under its own name"


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "spaces"])
def test_an_open_log_refuses_every_call_once_the_data_dir_is_blank(D, tmp_path, monkeypatch, blank):
    # review round 3: a blank setting is the documented refusal, a ValueError, not DataDirNotSet
    log, path = open_log()
    first = log.record(bet(1_000, race=1))
    before = path.read_bytes()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)  # where a blank read as "." would write
    monkeypatch.setenv(paths.DATA_DIR_VAR, blank)  # set, so the repo-root .env is not read
    assert env.get_setting(paths.DATA_DIR_VAR) == blank

    calls = {
        "record": lambda: log.record(bet(100, race=2)),
        "read": lambda: log.read(),
        "check": lambda: log.check(100),
        "staked_yen": lambda: log.staked_yen(),
    }
    for name, call in calls.items():
        try:
            call()
        except Exception as err:
            assert isinstance(err, ValueError), f"{name} raised {type(err).__name__}: {err}"
            assert not isinstance(err, (BetLogTampered, BetRefused)), f"{name}: {err!r}"
        else:
            pytest.fail(f"{name} on an open log ran with {paths.DATA_DIR_VAR} blank")
        assert path.read_bytes() == before, f"{name} wrote to the log"
        assert list(cwd.iterdir()) == [], f"{name} wrote under the working directory"

    monkeypatch.setenv(paths.DATA_DIR_VAR, str(D))  # set again, the same object works
    assert log.read() == [first]
    assert log.record(bet(100, race=2))["seq"] == 2
    assert log.staked_yen() == 1_100


@pytest.mark.parametrize(
    "name", ["2026-11-23.jsonl", "2026-11-22.json", "2026-11-22.jsonl.bak", "bets.jsonl"]
)
def test_a_file_named_for_another_day_is_refused(D, name):
    path = D / "bets" / "japan" / name
    with pytest.raises(ValueError):
        BetLog(path, day=DEV_DAY, limits=NO_FLAT)
    assert not path.exists()


def test_the_log_refuses_a_path_inside_the_repository(tmp_path, monkeypatch):
    name = f"{DEV_DAY.isoformat()}.jsonl"
    link = tmp_path / "link"
    link.symlink_to(REPO / "docs", target_is_directory=True)
    inside = {
        "the repo root": REPO / name,
        "a new folder in the repo": REPO / "docs" / "bets-453" / name,
        "through a symlink": link / name,
        "through ..": tmp_path / os.path.relpath(REPO / "docs", tmp_path) / name,
        "relative to the working directory": Path(os.path.relpath(REPO / "tests" / name)),
    }
    assert issubclass(BetLogInRepo, RuntimeError)
    for where, path in inside.items():
        with pytest.raises(BetLogInRepo):
            BetLog(path, day=DEV_DAY, limits=NO_FLAT)
        assert not (REPO / name).exists(), where
        assert not (REPO / "docs" / name).exists(), where
        assert not (REPO / "tests" / name).exists(), where
        assert not (REPO / "docs" / "bets-453").exists(), where

    # outside the repository the same name elsewhere is not the day's log either (ruling 1)
    outside = (tmp_path / "ok" / name).resolve()
    with pytest.raises(ValueError) as refused:
        BetLog(outside, day=DEV_DAY, limits=NO_FLAT)
    assert not isinstance(refused.value, BetLogInRepo)
    assert not (tmp_path / "ok").exists()
    BetLog(bet_log_path(DEV_DAY), day=DEV_DAY, limits=NO_FLAT).record(bet(100))
    assert bet_log_path(DEV_DAY).is_file()

    # and the default path cannot be pointed at the repository
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(REPO / "data"))
    with pytest.raises(paths.DataDirInsideRepo):
        bet_log_path(DEV_DAY)


# ---------------------------------------------------------------- holdout days


def test_holdout_day_bets_are_logged_and_counted():
    held = holdout_rule()
    start = date(2026, 11, 28)
    days = [start + timedelta(days=i) for i in range(30)]
    assert [is_holdout_day("japan", d) for d in days] == [held(d) for d in days]
    first_held = next(d for d in days if held(d))
    first_open = next(d for d in days if not held(d))

    limits = Limits(day_cap_yen=CAP, flat_stake_yen=None)
    log, path = open_log(first_held, limits)
    a = log.record(bet(20_000, day=first_held, race=1))
    b = BetLog(path, day=first_held, limits=limits).record(bet(10_000, day=first_held, race=2))
    assert a["day_status"] == b["day_status"] == "holdout"
    assert (a["over_cap"], b["over_cap"]) == (False, False)

    reader = BetLog(path, day=first_held, limits=limits)
    assert reader.read() == [a, b]  # stakes are not outcomes: read on any day
    assert reader.staked_yen() == 30_000
    check = reader.check(100)
    assert check.fits is False and check.staked_yen == 30_000
    assert reader.record(bet(100, day=first_held, race=3))["over_cap"] is True

    for day, status in ((first_open, "development"), (start - timedelta(days=6), "development")):
        log, _ = open_log(day, limits)
        assert log.record(bet(100, day=day))["day_status"] == status


# ---------------------------------------------------------------- the committed config


def test_the_committed_config_caps_a_race_day_at_30000_yen():
    limits = load_limits()
    assert limits.day_cap_yen == 30_000 and type(limits.day_cap_yen) is int
    assert limits.flat_stake_yen == 1_000 and type(limits.flat_stake_yen) is int  # owner, 11-27
    assert limits == Limits(day_cap_yen=30_000, flat_stake_yen=1_000)

    log, _ = open_log(limits=limits)
    assert log.check(1_000) == CapCheck(
        fits=True, staked_yen=0, remaining_yen=30_000, off_flat=False
    )
    assert log.check(100).off_flat is True
    for i in range(30):
        log.record(bet(1_000, race=1 + i % 12, combination=str(1 + i % 18)))
    assert log.staked_yen() == 30_000
    last = log.record(bet(1_000, race=12))
    assert last["over_cap"] is True and last["off_flat"] is False


# ---------------------------------------------------------------- no code places a bet


def imported_names(source: str, module: str) -> set[str]:
    """Every module ``source`` imports, relative imports resolved against ``module``'s package."""
    package = module.split(".")[:-1]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            parent = ".".join([*base, *([node.module] if node.module else [])])
            names.add(parent)
            names.update(f"{parent}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called in ("import_module", "__import__") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
    return names


def forbidden(names: set[str]) -> set[str]:
    return {
        name
        for name in names
        for bad in FORBIDDEN_IMPORTS
        if name == bad or name.startswith(f"{bad}.")
    }


def test_nothing_in_the_bets_package_can_place_a_bet():
    # the scan itself catches every form of import it is meant to
    probe = "\n".join(
        [
            "import requests",
            "import urllib.request",
            "from httpx import Client",
            "import aiohttp as h",
            "from socket import socket",
            "from playwright.sync_api import sync_playwright",
            "import selenium.webdriver",
            "from horsetrack.core.ingest import http",
            "from ....core import ingest",
            "import importlib; importlib.import_module('requests')",
        ]
    )
    caught = forbidden(imported_names(probe, "horsetrack.countries.japan.bets.log"))
    assert {name.split(".")[0] for name in caught} >= {
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
        "playwright",
        "selenium",
        "horsetrack",
    }
    assert "horsetrack.core.ingest" in caught
    allowed = "\n".join(  # what a bet log needs is not flagged
        [
            "import hashlib, json, os",
            "from horsetrack.core import paths",
            "from horsetrack.core.config import load_country_yaml",
            "from horsetrack.core.eval.holdout import is_holdout_day",
            "from horsetrack.core.eval.reports import is_inside",
            "from ....core.env import repo_root",
            "import urllib_free_helpers",
        ]
    )
    assert not forbidden(imported_names(allowed, "horsetrack.countries.japan.bets.log"))

    modules = sorted(BETS_PACKAGE.rglob("*.py"))
    assert {p.name for p in modules} >= {"__init__.py", "log.py"}
    for module_path in modules:
        rel = module_path.relative_to(REPO / "src").with_suffix("")
        module = ".".join(rel.parts)
        names = imported_names(module_path.read_text(encoding="utf-8"), module)
        assert not forbidden(names), f"{module} imports {sorted(forbidden(names))}"
