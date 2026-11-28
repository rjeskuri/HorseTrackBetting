"""Ticket #119: a sealed, append-only shadow log per race day.

Contract pinned here (the coder implements what these tests call):

- ``shadow_log_path(day)`` is ``data_dir()/shadow/japan/<YYYY-MM-DD>.jsonl``, with the data dir read
  on every call (a data dir inside the repo raises ``paths.DataDirInsideRepo``).
- ``day_status(day)`` is "holdout" when ``is_holdout_day("japan", day)`` is true, else
  "development". The shadow module imports ``is_holdout_day`` by name from
  ``horsetrack.core.eval.holdout``; these tests patch ``horsetrack.countries.japan.shadow.log
  .is_holdout_day`` (one test runs the real rule against the committed holdout.yaml).
- ``ShadowLog(path, *, day)``. The file is UTF-8 JSON lines, each ending in a newline. A record line
  is ``{"seq", "prev_sha256", "record"}``; the seal line is ``{"seq", "prev_sha256", "seal":
  {"n_records", "sealed_utc"}}``. ``seq`` counts up by one per line (from 0 or 1). ``prev_sha256``
  is the hex sha256 of the previous line's bytes without the newline, 64 zeros on the first line.
  ``record`` is the caller's record, unchanged, plus the ``day_status`` the log sets.
- ``append(records) -> int`` writes the whole batch or nothing, creates the parent directory, and
  fsyncs before it returns; the caller's dicts are not modified. It raises ``RecordRefused``
  (ValueError) for: a missing required field (board_capture_utc may be null only when status is
  "no_board"); a board_capture_utc later than decision_utc, compared as instants; a key containing
  stake, bet, kelly, amount or wager, in any case, at any depth, lists included; a duplicate
  (race_id, pool, combination, tau_min) within the batch or against the file; a caller-set
  day_status.
- ``seal(*, sealed_utc) -> str`` (the seal line's sha256). After it, ``append`` and ``seal`` raise
  ``LogSealed`` (ValueError), also through a new ``ShadowLog`` on the same file.
- ``read()`` returns the records in order on a development day. It raises ``HoldoutSealed``
  (PermissionError) on a holdout day, judged by the rule at the time of reading; a holdout day's
  file opened under another day's date gives no records either (HoldoutSealed or ValueError).
- ``logged_keys()`` returns the 4-tuple keys from the file on any day (empty before the first
  append), so a restarted logger skips what it already wrote.
- ``verify() -> LogSummary(n_records, by_tau, no_board, sealed, day_status)``, counts only;
  ``by_tau`` counts every record line, so its values sum to n_records. It raises
  ``ShadowLogTampered`` (ValueError) for an edited, inserted or deleted line that has a later line
  after it (the seal line covers the last record).

Every record is synthetic. HORSETRACK_DATA_DIR points at tmp_path in every test.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from horsetrack.core import paths
from horsetrack.countries.japan.shadow import log as shadow_log
from horsetrack.countries.japan.shadow.log import (
    HoldoutSealed,
    LogSealed,
    LogSummary,
    RecordRefused,
    ShadowLog,
    ShadowLogTampered,
    day_status,
    shadow_log_path,
)

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[4]
JAPAN_HOLDOUT = REPO / "src" / "horsetrack" / "countries" / "japan" / "config" / "holdout.yaml"
ZERO = "0" * 64
DEV_DAY = date(2026, 11, 21)  # before the holdout start: development by any reading
HOLD_DAY = date(2026, 11, 28)
SEALED_AT = datetime(2026, 11, 21, 9, 30, tzinfo=UTC)
REQUIRED = (
    "race_id",
    "pool",
    "combination",
    "tau_min",
    "decision_utc",
    "board_capture_utc",
    "written_utc",
    "mode",
    "status",
    "models",
    "git_sha",
)
# Directories no shadow log may write into anyway, skipped when the repo is listed.
PRUNE = {".git", ".venv", ".claude", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}


# ---------------------------------------------------------------- fixtures


class Rule:
    """Stands in for the shadow module's ``is_holdout_day``; ``decide`` says which days are held."""

    def __init__(self, real):
        self.real = real
        self.decide = lambda country, day: False

    def __call__(self, country, day):
        return self.decide(country, day)


@pytest.fixture(autouse=True)
def D(tmp_path, monkeypatch) -> Path:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    return data


@pytest.fixture(autouse=True)
def rule(monkeypatch) -> Rule:
    stand_in = Rule(shadow_log.is_holdout_day)
    monkeypatch.setattr(shadow_log, "is_holdout_day", stand_in)
    return stand_in


# ---------------------------------------------------------------- helpers


def rec(race_no=11, *, combination="7", pool="WIN", tau=2, day=DEV_DAY, status="ok", **extra):
    """A synthetic shadow record shaped like the ticket's example."""
    d = day.isoformat()
    record = {
        "race_id": f"jra:{d}:TOK:{race_no:02d}",
        "pool": pool,
        "combination": combination,
        "tau_min": tau,
        "decision_utc": f"{d}T06:38:00Z",
        "board_capture_utc": f"{d}T06:37:44Z" if status == "ok" else None,
        "written_utc": f"{d}T06:38:16Z",
        "mode": "live",
        "status": status,
        "board_odds": 3.4 if status == "ok" else None,
        "value_mean": 0.81,
        "value_q05": 0.74,
        "value_q95": 0.88,
        "flags": ["tail"],
        "models": {"win": "a1", "ordering": "b2", "nowcast": "c3"},
        "git_sha": "0123abc",
    }
    record.update(extra)
    return record


def key(record: dict) -> tuple:
    return (record["race_id"], record["pool"], record["combination"], record["tau_min"])


def as_logged(record: dict, status: str = "development") -> dict:
    return {**record, "day_status": status}


def open_log(day: date = DEV_DAY) -> tuple[ShadowLog, Path]:
    path = shadow_log_path(day)
    return ShadowLog(path, day=day), path


def sha(line: bytes) -> str:
    return hashlib.sha256(line).hexdigest()


def split_lines(raw: bytes) -> list[bytes]:
    assert raw.endswith(b"\n"), "every line ends in a newline"
    return raw[:-1].split(b"\n")


def join_lines(lines: list[bytes]) -> bytes:
    return b"".join(line + b"\n" for line in lines)


def chain(raw: bytes) -> list[dict]:
    """The parsed lines of a log, checked against the outline's format from first principles."""
    lines = split_lines(raw)
    objs, prev = [], ZERO
    for i, line in enumerate(lines):
        obj = json.loads(line)
        assert obj["prev_sha256"] == prev, f"line {i + 1} does not chain onto line {i}"
        body = "seal" if i == len(lines) - 1 and "seal" in obj else "record"
        assert set(obj) == {"seq", "prev_sha256", body}, f"line {i + 1} has keys {sorted(obj)}"
        prev = sha(line)
        objs.append(obj)
    seqs = [obj["seq"] for obj in objs]
    assert all(type(s) is int for s in seqs)
    assert seqs[0] in (0, 1) and seqs == list(range(seqs[0], seqs[0] + len(seqs)))
    return objs


def instant(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def assert_refused(log: ShadowLog, path: Path, batch: list[dict]) -> None:
    """The batch raises RecordRefused and writes nothing: same bytes, same keys."""
    before, keys = path.read_bytes(), log.logged_keys()
    with pytest.raises(RecordRefused):
        log.append(batch)
    assert path.read_bytes() == before, "a refused batch wrote to the log"
    assert log.logged_keys() == keys


def repo_files() -> dict[str, int]:
    out = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in PRUNE]
        for name in files:
            p = Path(root, name)
            out[str(p.relative_to(REPO))] = p.stat().st_size
    return out


# ---------------------------------------------------------------- writing and reading back


def test_an_append_reads_back_on_a_development_day():
    log, path = open_log()
    first = [
        rec(11, combination="7", tau=2),
        rec(11, combination="3", pool="PLACE", tau=5, value_mean=1.07, flags=[]),
        rec(12, combination="1-4", pool="QUINELLA", tau=0, status="no_board", models=None),
    ]
    second = [
        rec(1, combination="2", tau=10, mode="replay", value_q05=0.5),
        rec(12, combination="9", tau=2, models={"win": "zz", "ordering": None, "nowcast": "q"}),
    ]
    given = [dict(r) for r in first + second]

    assert log.append(first) == 3
    assert log.read() == [as_logged(r) for r in first]
    assert log.append(second) == 2
    expected = [as_logged(r) for r in first + second]
    assert log.read() == expected
    assert ShadowLog(path, day=DEV_DAY).read() == expected  # from the file, not from memory
    assert [obj["record"] for obj in chain(path.read_bytes())] == expected
    assert first + second == given, "append changed the caller's records"


def test_earlier_lines_are_byte_identical_after_more_appends(monkeypatch):
    fsyncs: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        fsyncs.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    batches = [
        [rec(1, tau=2)],
        [rec(2, tau=2), rec(2, combination="8", tau=5)],
        [rec(3, tau=0, status="no_board")],
        [rec(4, tau=10), rec(5, tau=10), rec(6, pool="WIDE", combination="1-2", tau=2)],
    ]
    log, path = open_log()
    for i, batch in enumerate(batches):
        if i in (1, 3):  # a restarted logger carries on the same file
            log = ShadowLog(path, day=DEV_DAY)
        before = path.read_bytes() if path.exists() else b""
        calls = len(fsyncs)
        assert log.append(batch) == len(batch)
        assert len(fsyncs) > calls, "each append is fsynced before it returns"
        after = path.read_bytes()
        assert after[: len(before)] == before, f"append {i + 1} changed earlier bytes"
        assert after.count(b"\n") == before.count(b"\n") + len(batch)

    records = [r for batch in batches for r in batch]
    assert [obj["record"] for obj in chain(path.read_bytes())] == [as_logged(r) for r in records]

    before = path.read_bytes()
    calls = len(fsyncs)
    digest = log.seal(sealed_utc=SEALED_AT)
    assert len(fsyncs) > calls, "the seal is fsynced before it returns"
    after = path.read_bytes()
    assert after[: len(before)] == before, "the seal changed earlier bytes"
    assert after.count(b"\n") == before.count(b"\n") + 1
    objs = chain(after)
    assert objs[-1]["seal"]["n_records"] == len(records)
    assert digest == sha(split_lines(after)[-1])


# ---------------------------------------------------------------- tampering


def written_log(*, sealed: bool) -> tuple[ShadowLog, Path, bytes]:
    """Four records in two appends (lines 1-4), then the seal (line 5) when sealed."""
    log, path = open_log()
    log.append([rec(1, tau=2), rec(2, tau=5)])
    log.append([rec(3, tau=2, status="no_board"), rec(4, tau=10)])
    if sealed:
        log.seal(sealed_utc=SEALED_AT)
    summary = log.verify()  # the untouched log verifies
    assert summary.n_records == 4 and summary.sealed is sealed
    return log, path, path.read_bytes()


def assert_tampered(log: ShadowLog, path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    for reader in (log, ShadowLog(path, day=DEV_DAY)):
        with pytest.raises(ShadowLogTampered):
            reader.verify()


def flip_byte(pattern: bytes) -> Callable[[bytes], bytes]:
    """Change the last byte `pattern` matches (a digit or a hex character) to another one."""

    def edit(line: bytes) -> bytes:
        match = re.search(pattern, line)
        assert match, f"{pattern!r} not found in {line!r}"
        i = match.end() - 1
        new = b"8" if line[i : i + 1] != b"8" else b"9"
        return line[:i] + new + line[i + 1 :]

    return edit


def respace(line: bytes) -> bytes:
    """The same JSON, other whitespace: other bytes."""
    new = json.dumps(json.loads(line), separators=(" ,  ", " :  ")).encode()
    assert new != line and json.loads(new) == json.loads(line)
    return new


RACE = rb"TOK:\d\d"
PREV = rb'"prev_sha256"\s*:\s*"[0-9a-f]'
EDITS = {  # name -> (sealed, line index, edit)
    "a value in line 1": (False, 0, flip_byte(RACE)),
    "a value in line 1 of a sealed log": (True, 0, flip_byte(RACE)),
    "prev_sha256 of line 1": (False, 0, flip_byte(PREV)),
    "prev_sha256 of line 2": (False, 1, flip_byte(PREV)),
    "seq of line 2": (False, 1, flip_byte(rb'"seq"\s*:\s*\d+')),
    "tau_min in line 3": (False, 2, flip_byte(rb'"tau_min"\s*:\s*\d+')),
    "the last record before the seal": (True, 3, flip_byte(RACE)),
    "n_records in the seal": (True, 4, flip_byte(rb'"n_records"\s*:\s*\d+')),
    "line 1 re-spaced, same JSON": (False, 0, respace),
    "line 2 no longer JSON": (False, 1, lambda line: b"#" + line[1:]),
}


@pytest.mark.parametrize(("sealed", "index", "edit"), list(EDITS.values()), ids=list(EDITS))
def test_an_edited_line_is_detected(sealed, index, edit):
    assert issubclass(ShadowLogTampered, ValueError)
    log, path, raw = written_log(sealed=sealed)
    lines = split_lines(raw)
    lines[index] = edit(lines[index])
    assert_tampered(log, path, join_lines(lines))


DELETIONS = {  # name -> (sealed, line indexes removed)
    "line 1": (False, [0]),
    "line 2": (False, [1]),
    "lines 2 and 3": (False, [1, 2]),
    "line 1 of a sealed log": (True, [0]),
    "the last record before the seal": (True, [3]),
}


@pytest.mark.parametrize(("sealed", "gone"), list(DELETIONS.values()), ids=list(DELETIONS))
def test_a_deleted_line_is_detected(sealed, gone):
    log, path, raw = written_log(sealed=sealed)
    lines = [line for i, line in enumerate(split_lines(raw)) if i not in gone]
    assert_tampered(log, path, join_lines(lines))


def forged(seq: int, prev: str) -> bytes:
    """A well-formed record line for a key nobody logged."""
    record = as_logged(rec(9, combination="5", tau=1))
    return json.dumps({"seq": seq, "prev_sha256": prev, "record": record}).encode()


def chained_after(line: bytes) -> bytes:
    return forged(json.loads(line)["seq"] + 1, sha(line))


INSERTIONS = {  # name -> (sealed, insert before this index, line to insert)
    "a forged first line": (False, 0, lambda lines: forged(json.loads(lines[0])["seq"], ZERO)),
    "a forged line chained onto line 2": (False, 2, lambda lines: chained_after(lines[1])),
    "a forged record chained before the seal": (True, 4, lambda lines: chained_after(lines[3])),
    "line 2 replayed": (False, 2, lambda lines: lines[1]),
}


@pytest.mark.parametrize(("sealed", "at", "make"), list(INSERTIONS.values()), ids=list(INSERTIONS))
def test_an_inserted_line_is_detected(sealed, at, make):
    log, path, raw = written_log(sealed=sealed)
    lines = split_lines(raw)
    lines.insert(at, make(lines))
    assert_tampered(log, path, join_lines(lines))


# ---------------------------------------------------------------- the seal


def test_a_sealed_log_refuses_appends():
    assert issubclass(LogSealed, ValueError)
    cases = [  # (day, records before the seal, sealed_utc)
        (DEV_DAY, 1, SEALED_AT),
        (date(2026, 11, 22), 3, datetime(2026, 11, 22, 10, 5, 17, tzinfo=UTC)),
    ]
    for day, n, sealed_at in cases:
        log, path = open_log(day)
        records = [rec(i + 1, day=day, tau=2 * i) for i in range(n)]
        assert log.append(records) == n
        before = path.read_bytes()
        digest = log.seal(sealed_utc=sealed_at)
        raw = path.read_bytes()
        assert raw[: len(before)] == before
        objs = chain(raw)
        assert len(objs) == n + 1 and set(objs[-1]["seal"]) == {"n_records", "sealed_utc"}
        assert objs[-1]["seal"]["n_records"] == n
        assert instant(objs[-1]["seal"]["sealed_utc"]) == sealed_at
        assert digest == sha(split_lines(raw)[-1])

        late = rec(20, day=day, tau=2)
        for writer in (log, ShadowLog(path, day=day)):
            with pytest.raises(LogSealed):
                writer.append([late])
            with pytest.raises(LogSealed):
                writer.seal(sealed_utc=sealed_at + timedelta(minutes=5))
            assert path.read_bytes() == raw
            assert key(late) not in writer.logged_keys()
        summary = ShadowLog(path, day=day).verify()
        assert summary.sealed is True and summary.n_records == n
        assert ShadowLog(path, day=day).read() == [as_logged(r) for r in records]


# ---------------------------------------------------------------- holdout days


def test_holdout_day_values_cannot_be_read(rule):
    assert issubclass(HoldoutSealed, PermissionError)
    rule.decide = lambda country, day: True
    records = [
        rec(11, day=HOLD_DAY, tau=2, value_mean=0.8137, board_odds=3.41),
        rec(11, day=HOLD_DAY, combination="4", tau=5),
        rec(12, day=HOLD_DAY, tau=2, status="no_board"),
    ]
    log, path = open_log(HOLD_DAY)
    assert log.append(records[:2]) == 2  # written prospectively
    assert log.append(records[2:]) == 1

    keys = {key(r) for r in records}
    expected = LogSummary(
        n_records=3, by_tau={2: 2, 5: 1}, no_board=1, sealed=False, day_status="holdout"
    )
    for reader in (log, ShadowLog(path, day=HOLD_DAY)):
        with pytest.raises(HoldoutSealed):
            reader.read()
        assert reader.logged_keys() == keys
        assert reader.verify() == expected
    assert [obj["record"]["day_status"] for obj in chain(path.read_bytes())] == ["holdout"] * 3

    log.seal(sealed_utc=datetime(2026, 11, 28, 9, 0, tzinfo=UTC))
    with pytest.raises(HoldoutSealed):
        ShadowLog(path, day=HOLD_DAY).read()
    assert ShadowLog(path, day=HOLD_DAY).verify().sealed is True


def test_holdout_values_are_not_read_through_a_later_rule_or_another_day(rule):
    # Written while the day counted as development; a later reader goes by the rule now.
    records = [rec(11, day=HOLD_DAY, tau=2), rec(12, day=HOLD_DAY, tau=5)]
    log, path = open_log(HOLD_DAY)
    log.append(records)
    assert log.read() == [as_logged(r) for r in records]
    rule.decide = lambda country, day: True
    with pytest.raises(HoldoutSealed):
        ShadowLog(path, day=HOLD_DAY).read()

    # A holdout day's file opened under the date of a development day.
    held = date(2026, 11, 29)
    rule.decide = lambda country, day: day == held
    log, path = open_log(held)
    log.append([rec(11, day=held, tau=2)])
    with pytest.raises((HoldoutSealed, ValueError)):
        ShadowLog(path, day=date(2026, 12, 5)).read()


def test_day_status_is_set_by_the_log(rule):
    held = {date(2026, 11, 21), date(2026, 11, 28), date(2026, 12, 6)}
    rule.decide = lambda country, day: country == "japan" and day in held
    days = held | {date(2026, 11, 22), date(2026, 11, 29), date(2026, 12, 5), date(2027, 1, 10)}
    assert {d: day_status(d) for d in days} == {
        d: "holdout" if d in held else "development" for d in days
    }

    for day in (date(2026, 11, 28), date(2026, 11, 29)):
        status = "holdout" if day in held else "development"
        log, path = open_log(day)
        good = rec(11, day=day)
        assert log.append([good]) == 1
        for claimed in ("development", "holdout", None, ""):
            assert_refused(log, path, [rec(12, day=day), rec(13, day=day, day_status=claimed)])
        assert log.append([rec(12, day=day)]) == 1
        logged = [obj["record"] for obj in chain(path.read_bytes())]
        assert logged == [as_logged(good, status), as_logged(rec(12, day=day), status)]
        assert log.verify().day_status == status


def test_day_status_follows_the_committed_holdout_rule(rule):
    rule.decide = rule.real  # the real is_holdout_day, reading the committed holdout.yaml
    section = yaml.safe_load(JAPAN_HOLDOUT.read_text(encoding="utf-8"))["holdout"]
    start, seed = section["start"], section["seed"]
    start = start if isinstance(start, date) else date.fromisoformat(start)

    def held(day: date) -> bool:
        digest = hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest()
        return day >= start and int(digest, 16) % 2 == 0

    days = [start + timedelta(days=i) for i in range(-7, 35)]
    assert {d: day_status(d) for d in days} == {
        d: "holdout" if held(d) else "development" for d in days
    }

    first_held = next(d for d in days if held(d))
    first_open = next(d for d in days if d >= start and not held(d))
    for day in (first_held, first_open, start - timedelta(days=7)):
        log, _ = open_log(day)
        log.append([rec(11, day=day)])
        if held(day):
            with pytest.raises(HoldoutSealed):
                log.read()
        else:
            assert log.read() == [as_logged(rec(11, day=day))]


# ---------------------------------------------------------------- refused records


@pytest.mark.parametrize("missing", REQUIRED)
def test_a_missing_required_field_is_refused(missing):
    assert issubclass(RecordRefused, ValueError)
    log, path = open_log()
    log.append([rec(1)])
    bad = rec(2)
    del bad[missing]
    assert_refused(log, path, [rec(3), bad])
    assert_refused(log, path, [rec(2, board_capture_utc=None)])  # status "ok" needs a board
    assert log.append([rec(2), rec(3), rec(4, status="no_board")]) == 3
    assert log.read() == [as_logged(r) for r in (rec(1), rec(2), rec(3), rec(4, status="no_board"))]


LOOK_AHEAD = {  # name -> (board_capture_utc, decision_utc, refused)
    "one second after": ("2026-11-21T06:38:01Z", "2026-11-21T06:38:00Z", True),
    "ten minutes after": ("2026-11-21T06:48:00Z", "2026-11-21T06:38:00Z", True),
    "half a second after": ("2026-11-21T06:38:00.500000Z", "2026-11-21T06:38:00Z", True),
    "after, +00:00 against Z": ("2026-11-21T06:38:01+00:00", "2026-11-21T06:38:00Z", True),
    "sixteen seconds before": ("2026-11-21T06:37:44Z", "2026-11-21T06:38:00Z", False),
    "the same instant": ("2026-11-21T06:38:00Z", "2026-11-21T06:38:00Z", False),
    "the same instant, Z against +00:00": (
        "2026-11-21T06:38:00Z",
        "2026-11-21T06:38:00+00:00",
        False,
    ),
    "a millisecond before": ("2026-11-21T06:37:59.999000+00:00", "2026-11-21T06:38:00Z", False),
}


@pytest.mark.parametrize(
    ("board", "decision", "refused"), list(LOOK_AHEAD.values()), ids=list(LOOK_AHEAD)
)
def test_a_board_after_the_decision_time_is_refused(board, decision, refused):
    log, path = open_log()
    log.append([rec(1)])
    record = rec(2, board_capture_utc=board, decision_utc=decision)
    if refused:
        assert_refused(log, path, [rec(3), record])
        assert log.append([rec(3)]) == 1
        assert log.logged_keys() == {key(rec(1)), key(rec(3))}
    else:
        assert log.append([record]) == 1
        assert log.logged_keys() == {key(rec(1)), key(record)}


STAKE_LIKE = {
    "stake at the top": {"stake": 100},
    "kelly one level down": {"values": {"kelly": 0.1}},
    "stake inside a longer key": {"stake_yen": 100},
    "bet inside a key": {"bet_type": "WIN"},
    "amount": {"amount": 500},
    "wager in a dict inside a list": {"legs": [{"wager": 1}]},
    "kelly inside models": {"models": {"win": "a1", "kelly": "k1"}},
    "three levels down": {"values": {"q": {"max_bet": 1}}},
    "capitals": {"Kelly_Fraction": 0.02},
}


@pytest.mark.parametrize("extra", list(STAKE_LIKE.values()), ids=list(STAKE_LIKE))
def test_stake_like_keys_are_refused(extra):
    log, path = open_log()
    log.append([rec(1)])
    assert_refused(log, path, [rec(2), rec(3, **extra)])
    clean = rec(3, values={"mean": 0.8, "q": {"q05": 0.7}}, legs=[{"leg": 1}])
    assert log.append([rec(2), clean]) == 2  # nested values without such keys are fine
    assert log.read()[1:] == [as_logged(rec(2)), as_logged(clean)]


def test_a_duplicate_key_is_refused():
    log, path = open_log()
    base = rec(11, pool="WIN", combination="7", tau=2)
    assert log.append([base]) == 1

    # the same key with other values, alone and after a good record in the same batch
    assert_refused(log, path, [rec(11, value_mean=0.2, written_utc="2026-11-21T06:39:00Z")])
    assert_refused(log, path, [rec(11, combination="8"), rec(11, value_mean=0.2)])
    # the same new key twice in one batch
    fresh = rec(11, combination="8")
    assert_refused(log, path, [fresh, dict(fresh, value_mean=0.2)])

    # change any one part of the key and it is a new record
    variants = [
        rec(12, combination="7", tau=2),
        rec(11, pool="PLACE", combination="7", tau=2),
        rec(11, combination="8", tau=2),
        rec(11, combination="7", tau=5),
    ]
    assert log.append(variants) == 4

    again = ShadowLog(path, day=DEV_DAY)  # a restart sees the file's keys
    for record in [base, *variants]:
        assert_refused(again, path, [dict(record, value_mean=0.5)])
    assert again.logged_keys() == {key(r) for r in [base, *variants]}


def test_logged_keys_survive_a_restart():
    log, path = open_log()
    assert log.logged_keys() == set()  # the first start of the day: nothing logged yet
    first = [rec(11, tau=2), rec(11, combination="3", tau=2), rec(12, pool="PLACE", tau=5)]
    log.append(first[:2])
    log.append(first[2:])

    restarted = ShadowLog(path, day=DEV_DAY)
    keys = restarted.logged_keys()
    assert keys == {key(r) for r in first}
    assert all(isinstance(k, tuple) and len(k) == 4 and type(k[3]) is int for k in keys)

    raw = path.read_bytes()
    assert_refused(restarted, path, [rec(11, tau=2, value_mean=0.2)])
    late = rec(12, tau=10)
    assert restarted.append([late]) == 1
    after = path.read_bytes()
    assert after[: len(raw)] == raw
    assert [obj["record"] for obj in chain(after)] == [as_logged(r) for r in [*first, late]]
    assert ShadowLog(path, day=DEV_DAY).logged_keys() == {key(r) for r in [*first, late]}
    assert ShadowLog(path, day=DEV_DAY).verify().n_records == 4


# ---------------------------------------------------------------- where it lives


def test_the_log_path_is_under_the_data_dir(D, tmp_path, monkeypatch):
    assert shadow_log_path(date(2026, 11, 28)) == D / "shadow" / "japan" / "2026-11-28.jsonl"
    for day in (date(2026, 12, 5), date(2027, 1, 3), date(2026, 2, 1)):
        assert shadow_log_path(day) == D / "shadow" / "japan" / f"{day.isoformat()}.jsonl"

    elsewhere = (tmp_path / "elsewhere").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(elsewhere))  # read on every call
    assert shadow_log_path(date(2026, 12, 5)) == elsewhere / "shadow" / "japan" / "2026-12-05.jsonl"

    monkeypatch.setenv(paths.DATA_DIR_VAR, str(REPO / "data"))
    with pytest.raises(paths.DataDirInsideRepo):
        shadow_log_path(date(2026, 11, 28))


def test_nothing_is_written_in_the_repo(D, tmp_path):
    before = repo_files()
    path = shadow_log_path(DEV_DAY)
    assert not path.parent.exists()

    log = ShadowLog(path, day=DEV_DAY)
    assert log.logged_keys() == set()
    log.append([rec(1), rec(2, status="no_board")])
    with pytest.raises(RecordRefused):
        log.append([rec(3, stake=1)])
    assert len(log.read()) == 2 and log.verify().n_records == 2
    log.seal(sealed_utc=SEALED_AT)
    with pytest.raises(LogSealed):
        log.append([rec(4)])

    assert path.is_file() and path.parent == D / "shadow" / "japan"
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written and all(p.resolve().is_relative_to(D) for p in written)
    assert repo_files() == before, "the shadow log wrote inside the repository"


# ---------------------------------------------------------------- verify


SCENARIOS = {  # name -> (holdout day, [(tau_min, status), ...])
    "development": (
        False,
        [(0, "ok"), (2, "ok"), (2, "ok"), (2, "no_board"), (5, "ok"), (5, "no_board"), (10, "ok")],
    ),
    "holdout": (True, [(1, "no_board"), (1, "ok"), (15, "ok"), (30, "no_board"), (30, "no_board")]),
}


@pytest.mark.parametrize(("held", "spec"), list(SCENARIOS.values()), ids=list(SCENARIOS))
def test_verify_reports_counts_by_tau_and_status(rule, held, spec):
    rule.decide = lambda country, day: held
    day = HOLD_DAY if held else DEV_DAY
    records = [
        rec(11 + i % 2, day=day, combination=str(i + 1), tau=tau, status=status)
        for i, (tau, status) in enumerate(spec)
    ]
    log, path = open_log(day)
    half = len(records) // 2
    log.append(records[:half])
    ShadowLog(path, day=day).append(records[half:])

    counts = {
        "n_records": len(spec),
        "by_tau": dict(Counter(tau for tau, _ in spec)),
        "no_board": sum(status == "no_board" for _, status in spec),
        "day_status": "holdout" if held else "development",
    }
    summary = ShadowLog(path, day=day).verify()
    assert summary == LogSummary(**counts, sealed=False)
    assert all(type(tau) is int for tau in summary.by_tau)

    ShadowLog(path, day=day).seal(sealed_utc=datetime(2026, 11, 28, 9, 0, tzinfo=UTC))
    assert ShadowLog(path, day=day).verify() == LogSummary(**counts, sealed=True)


# ---------------------------------------------------------------- versioned values (#566)
#
# A record's values are versioned by two plain keys the log keeps as given: `values_version` (2
# from #566 on) and `starters` (the runners at the start of sales the values used). A record
# written before has neither and is version 1: it keeps verifying, and it stays in a file that
# later records are appended to.


def _line(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode()


def old_format_file(records: list[dict]) -> bytes:
    """A log file as the writer before #566 wrote it: no version, no starters, chained by sha256."""
    out, prev = [], ZERO
    for seq, record in enumerate(records, 1):
        line = _line({"seq": seq, "prev_sha256": prev, "record": as_logged(record)})
        out.append(line + b"\n")
        prev = sha(line)
    return b"".join(out)


def test_an_old_format_log_still_verifies_and_takes_new_records():
    old = [rec(1, tau=2), rec(1, tau=5, combination="3")]
    log, path = open_log()
    path.parent.mkdir(parents=True)
    path.write_bytes(old_format_file(old))
    assert all("values_version" not in r and "starters" not in r for r in old)
    assert log.verify() == LogSummary(
        n_records=2, by_tau={2: 1, 5: 1}, no_board=0, sealed=False, day_status="development"
    )

    new = [
        rec(2, tau=2, values_version=2, starters=8),
        rec(2, tau=2, combination="3", values_version=2, starters=8, pool="PLACE"),
    ]
    assert log.append(new) == 2
    summary = ShadowLog(path, day=DEV_DAY).verify()
    assert (summary.n_records, summary.by_tau) == (4, {2: 3, 5: 1})
    read = ShadowLog(path, day=DEV_DAY).read()
    assert read == [as_logged(r) for r in old + new]  # stored as given, old and new side by side
    assert [r.get("values_version") for r in read] == [None, None, 2, 2]
    assert [r.get("starters") for r in read] == [None, None, 8, 8]
    log.seal(sealed_utc=SEALED_AT)
    assert ShadowLog(path, day=DEV_DAY).verify().sealed


@pytest.mark.parametrize("starters", [5, 8, 16])
def test_a_new_record_keeps_its_version_and_starters_and_they_are_covered_by_the_chain(starters):
    log, path = open_log()
    log.append([rec(1, tau=2, values_version=2, starters=starters)])
    log.append([rec(2, tau=2)])
    raw = path.read_bytes()
    assert ShadowLog(path, day=DEV_DAY).read()[0]["starters"] == starters
    assert ShadowLog(path, day=DEV_DAY).read()[0]["values_version"] == 2
    mark = f'"starters":{starters}'.encode()
    assert raw.count(mark) == 1
    for forged in (starters + 1, starters - 1):
        assert_tampered(log, path, raw.replace(mark, f'"starters":{forged}'.encode()))
    assert_tampered(log, path, raw.replace(b'"values_version":2', b'"values_version":1'))
    path.write_bytes(raw)
    assert log.verify().n_records == 2  # the untouched file verifies
