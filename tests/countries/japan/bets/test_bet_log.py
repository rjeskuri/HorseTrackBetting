"""Ticket #453: a hash-chained manual bet log per Japan race day, with the race-day cap.

Contract pinned here, module ``horsetrack.countries.japan.bets.log``:

- ``POOLS``, the jra config's pool codes. ``Limits(day_cap_yen, flat_stake_yen)``;
  ``load_limits()`` reads the committed ``countries/japan/config/betting.yaml`` (the owner's
  answer, 2026-11-27: a ¥30,000 cap per race day, ¥1,000 flat stake).
- ``bet_log_path(day)`` is ``$HORSETRACK_DATA_DIR/bets/japan/<YYYY-MM-DD>.jsonl``, the data dir
  read on every call.
- ``BetLog(path, *, day, limits)`` refuses a file name other than ``<day>.jsonl`` (ValueError) and
  a path inside the repository.
- A bet is a dict of ``race_id`` (``jra:<day>:<VENUE>:<NN>``, the log's own day, a jra venue code),
  ``pool`` (one of POOLS), ``combination`` (non-blank text), ``stake_yen`` (an int, at least 100, a
  multiple of 100), ``placed_utc`` (ISO text with a UTC offset) and the optional ``option_id`` and
  ``note``. Anything else raises ``BetRefused`` (a ValueError) and writes nothing.
- ``record(bet)`` appends one line and returns the stored record: the bet plus ``seq``,
  ``written_utc``, ``day_status`` (holdout or development, from the committed holdout rule),
  ``over_cap`` (the day's total including this bet exceeds the cap) and ``off_flat`` (a flat stake
  is set and this stake differs from it). A bet past the cap or off the flat stake is recorded.
- ``check(stake) -> CapCheck(fits, staked_yen, remaining_yen, off_flat)``: the day's total so far,
  the cap left before this stake, whether the stake fits (total + stake <= cap). It writes nothing.
- Lines are ``{"seq", "prev_sha256", "record"}``, compact ASCII JSON, chained as
  docs/specs/shadow-log.md describes; a broken chain raises ``BetLogTampered`` on every call.
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

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.bets.log import (
    POOLS,
    BetLog,
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


def open_log(day: date = DEV_DAY, limits: Limits = NO_FLAT) -> tuple[BetLog, Path]:
    path = bet_log_path(day)
    return BetLog(path, day=day, limits=limits), path


def sha(line: bytes) -> str:
    return hashlib.sha256(line).hexdigest()


def split_lines(raw: bytes) -> list[bytes]:
    assert raw.endswith(b"\n"), "every line ends in a newline"
    return raw[:-1].split(b"\n")


def join_lines(lines: list[bytes]) -> bytes:
    return b"".join(line + b"\n" for line in lines)


def chain(raw: bytes) -> list[dict]:
    """The parsed lines of a bet log, checked against the ticket's format from first principles."""
    objs, prev = [], ZERO
    for n, line in enumerate(split_lines(raw), 1):
        assert line.isascii(), f"line {n} is not ASCII"
        obj = json.loads(line)
        compact = json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode()
        assert line == compact, f"line {n} is not compact JSON"
        assert set(obj) == {"seq", "prev_sha256", "record"}, f"line {n} has keys {sorted(obj)}"
        assert obj["prev_sha256"] == prev, f"line {n} does not chain onto line {n - 1}"
        assert obj["seq"] == n and type(obj["seq"]) is int
        assert obj["record"]["seq"] == n
        prev = sha(line)
        objs.append(obj)
    return objs


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
        stored = log.record(bet(200, race=1 + i, pool=pool, combination=combination))
        assert stored["pool"] == pool and stored["combination"] == combination
        assert stored["seq"] == i + 1
    assert [r["pool"] for r in BetLog(path, day=DEV_DAY, limits=NO_FLAT).read()] == list(POOLS)
    assert log.staked_yen() == 200 * len(POOLS)


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
    objs = chain(raw)
    assert objs[1]["prev_sha256"] == sha(split_lines(raw)[0])
    return path, raw


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
    for where, path in inside.items():
        with pytest.raises((ValueError, PermissionError, RuntimeError)):
            BetLog(path, day=DEV_DAY, limits=NO_FLAT)
        assert not (REPO / name).exists(), where
        assert not (REPO / "docs" / name).exists(), where
        assert not (REPO / "tests" / name).exists(), where
        assert not (REPO / "docs" / "bets-453").exists(), where

    # outside the repository the same name is accepted
    outside = (tmp_path / "ok" / name).resolve()
    BetLog(outside, day=DEV_DAY, limits=NO_FLAT).record(bet(100))
    assert outside.is_file()

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
