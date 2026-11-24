"""The Japan odds snapshot log and its post-time sidecar (ticket #22, ported from HKTrack).

Layout kept from HKTrack: `odds_snapshots/jra/<YYYY-MM-DD>/<VENUE>.jsonl.gz`, one gzip member per
capture, one JSON object per line, pool totals as rows with `combination` "__pool__". Logs copied
from HKTrack must read exactly like logs written here.

Contract pinned here:
- `append_capture(path, rows) -> int` writes one gzip member per call (creating parent directories),
  timestamps as ISO UTC ("+00:00", as HKTrack wrote them), dates as ISO dates; no rows, no write.
- `read_rows(path) -> list[dict]` returns the logged rows as logged (no keys added or dropped), from
  whole CRC-checked members only; a damaged member is skipped and reading resumes at the next one.
- `latest_prices(rows)` -> {(race_id, pool, combination): row}, without "__pool__" rows;
  `latest_totals(rows)` -> {(race_id, pool): row}, "__pool__" rows only. Newest by `snapshot_utc`;
  between rows of the same instant the later one in the log wins (HKTrack's stable sort).
- `write_post_times(path, post_times, updated_utc)` / `read_post_times(path) -> {int: aware UTC}`;
  a failed write leaves the previous sidecar intact and no temporary file behind.

Tailing (#22 as amended, implemented by #120):
- `iter_members(path, *, start=0)` keeps yielding each whole member's rows (`list[dict]`), now from
  byte offset `start`; its existing callers are unchanged.
- `iter_members_with_offsets(path, *, start=0) -> Iterator[tuple[int, list[dict]]]` yields
  (end offset, rows) per whole member from byte offset `start`: the end offset is the byte just
  after that member. A torn trailing member (a write still in progress, no later member header) is
  not yielded, and no offset passes it.
- `LogTail(path).poll() -> list[dict]` returns the rows of the members appended since the previous
  poll, reading the file from where the last whole member ended; a missing file gives []. A torn
  trailing member is read on the poll after it is complete.

All fixtures are synthetic and built in tmp_path.
"""

from __future__ import annotations

import gzip
import json
import os
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from horsetrack.countries.japan import odds_log

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
TOKYO = ZoneInfo("Asia/Tokyo")
T0 = datetime(2026, 11, 28, 5, 0, tzinfo=UTC)  # 14:00 in Tokyo
HKTRACK_KEYS = {
    "snapshot_utc",
    "race_id",
    "pool",
    "combination",
    "odds",
    "status_raw",
    "source_update_utc",
}


def race(no: int) -> str:
    return f"jra:2026-11-28:TOK:{no:02d}"


def price(combination: str, odds, *, at: datetime, pool: str = "WIN", race_no: int = 11, **extra):
    row = {
        "snapshot_utc": at,
        "race_id": race(race_no),
        "pool": pool,
        "combination": combination,
        "odds": odds,
        "status_raw": "SCR" if odds is None else "14時00分現在",
        "source_update_utc": at - timedelta(seconds=12),
    }
    row.update(extra)
    return row


def total(pool: str, investment, *, at: datetime, race_no: int = 11, odds=None):
    return {
        "snapshot_utc": at,
        "race_id": race(race_no),
        "pool": pool,
        "combination": "__pool__",
        "odds": odds,
        "investment": investment,
        "status_raw": None,
        "source_update_utc": at - timedelta(seconds=12),
    }


def as_logged(row: dict) -> dict:
    """The row as the log must hold it, derived here: aware datetimes in UTC ISO form."""
    out = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            value = value.astimezone(UTC).isoformat()
        out[key] = value
    return out


def instant(value) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def log_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "odds_snapshots" / "jra" / "2026-11-28" / "TOK.jsonl.gz"


def raw_lines(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def append_members(path: Path, captures: list[list[dict]]) -> list[int]:
    """Append each capture with append_capture; returns the file size after each one."""
    sizes = []
    for rows in captures:
        assert odds_log.append_capture(path, rows) == len(rows)
        sizes.append(path.stat().st_size)
    return sizes


# ---------------------------------------------------------------- writing


def test_each_capture_is_one_gzip_member(tmp_path, monkeypatch):
    path = log_path(tmp_path)  # its parent directories do not exist yet
    fsyncs: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        fsyncs.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    captures = [
        [price("1", 3.4, at=T0), price("2", 8.0, at=T0)],
        [price("1", 3.1, at=T0 + timedelta(seconds=30))],
        [
            price("1", 2.9, at=T0 + timedelta(seconds=60)),
            price("1-2", 12.5, at=T0 + timedelta(seconds=60), pool="QUINELLA"),
            price("3", None, at=T0 + timedelta(seconds=60), race_no=12),
            total("WIN", 1234500.0, at=T0 + timedelta(seconds=60)),
        ],
    ]
    sizes = []
    for i, rows in enumerate(captures):
        assert odds_log.append_capture(path, rows) == len(rows)
        sizes.append(path.stat().st_size)
        assert len(fsyncs) >= i + 1, "each capture is fsynced before append_capture returns"

    raw = path.read_bytes()
    starts = [0, *sizes[:-1]]
    for rows, start, end in zip(captures, starts, sizes, strict=True):
        chunk = raw[start:end]
        assert chunk[:2] == b"\x1f\x8b"
        member = zlib.decompressobj(wbits=31)
        data = member.decompress(chunk)
        assert member.eof and member.unused_data == b"", "exactly one gzip member per capture"
        assert [json.loads(line) for line in data.decode("utf-8").splitlines()] == [
            as_logged(r) for r in rows
        ]
    assert odds_log.read_rows(path) == [as_logged(r) for rows in captures for r in rows]


def test_timestamps_are_written_in_utc(tmp_path):
    cases = [  # (value given, string the log must hold)
        (datetime(2026, 11, 28, 14, 0, tzinfo=JST), "2026-11-28T05:00:00+00:00"),
        (datetime(2026, 11, 28, 9, 5, 7, tzinfo=JST), "2026-11-28T00:05:07+00:00"),
        (datetime(2026, 11, 29, 1, 0, tzinfo=TOKYO), "2026-11-28T16:00:00+00:00"),
        (
            datetime(2026, 11, 27, 22, 0, tzinfo=timezone(timedelta(hours=-5))),
            "2026-11-28T03:00:00+00:00",
        ),
        (
            datetime(2026, 11, 28, 8, 0, tzinfo=timezone(timedelta(hours=8))),
            "2026-11-28T00:00:00+00:00",
        ),
        (datetime(2026, 11, 28, 5, 0, 0, 250000, tzinfo=UTC), "2026-11-28T05:00:00.250000+00:00"),
        # naive: read as UTC, as HKTrack did
        (datetime(2026, 11, 28, 6, 30), "2026-11-28T06:30:00+00:00"),
        (None, None),
    ]
    n = len(cases)
    rows = []
    for i in range(n):
        row = price(str(i + 1), 2.0 + i, at=T0)
        row["snapshot_utc"] = cases[i][0] or T0
        row["fetched_utc"] = cases[(i + 1) % n][0]
        row["source_update_utc"] = cases[(i + 2) % n][0]
        row["race_date"] = date(2026, 11, 28 - i % 2)
        rows.append(row)
    path = log_path(tmp_path)
    assert odds_log.append_capture(path, rows) == n

    lines = raw_lines(path)
    assert len(lines) == n
    for i, line in enumerate(lines):
        assert line["snapshot_utc"] == (cases[i][1] or "2026-11-28T05:00:00+00:00")
        assert line["fetched_utc"] == cases[(i + 1) % n][1]
        assert line["source_update_utc"] == cases[(i + 2) % n][1]
        assert line["race_date"] == f"2026-11-{28 - i % 2}"
        assert line["combination"] == str(i + 1) and line["odds"] == 2.0 + i
    assert odds_log.read_rows(path) == lines


def test_an_empty_capture_writes_nothing(tmp_path):
    path = log_path(tmp_path)
    for empty in ([], ()):
        assert odds_log.append_capture(path, empty) == 0
        assert not path.exists()
    assert odds_log.read_rows(path) == []  # no log yet: nothing to read, no error

    odds_log.append_capture(path, [price("1", 3.4, at=T0)])
    before = path.read_bytes()
    assert odds_log.append_capture(path, []) == 0
    assert path.read_bytes() == before


# ---------------------------------------------------------------- damage


def three_captures() -> list[list[dict]]:
    return [
        [price("1", 3.4, at=T0), price("2", 8.0, at=T0, race_no=12)],
        [price("1", 3.1, at=T0 + timedelta(seconds=30))],
        [
            price("1", 2.8, at=T0 + timedelta(seconds=60)),
            total("WIN", 5000.0, at=T0 + timedelta(seconds=60)),
        ],
    ]


TORN = {
    "kill -9 near the end": lambda m: m[:-9],
    "inside the gzip header": lambda m: m[:6],
    "half the member": lambda m: m[: len(m) // 2],
    "zero-filled tail": lambda m: m[: len(m) // 2] + b"\x00" * 64,
    "one byte": lambda m: m[:1],
}


@pytest.mark.parametrize("damage", list(TORN.values()), ids=list(TORN))
def test_a_torn_final_write_costs_that_capture_only(tmp_path, damage):
    path = log_path(tmp_path)
    captures = three_captures()
    sizes = append_members(path, captures)
    raw = path.read_bytes()
    path.write_bytes(raw[: sizes[1]] + damage(raw[sizes[1] :]))

    rows = odds_log.read_rows(path)
    assert rows == [as_logged(r) for r in captures[0] + captures[1]]
    latest = odds_log.latest_prices(rows)
    assert latest[(race(11), "WIN", "1")]["odds"] == 3.1
    assert latest[(race(12), "WIN", "2")]["odds"] == 8.0
    assert odds_log.latest_totals(rows) == {}


def _flip_crc(member: bytes) -> bytes:
    """Body intact, CRC32 wrong: only a reader that checks the CRC rejects it."""
    crc = bytes(b ^ 0xFF for b in member[-8:-4])
    return member[:-8] + crc + member[-4:]


DAMAGED = {
    **TORN,
    "not gzip at all": lambda m: b"not gzip at all",
    "bad crc": _flip_crc,
}


@pytest.mark.parametrize("damage", list(DAMAGED.values()), ids=list(DAMAGED))
def test_a_damaged_middle_member_does_not_hide_later_captures(tmp_path, damage):
    """The logger is restarted after a crash and keeps appending after the damaged member."""
    path = log_path(tmp_path)
    captures = three_captures()
    sizes = append_members(path, captures[:2])
    raw = path.read_bytes()
    path.write_bytes(raw[: sizes[0]] + damage(raw[sizes[0] :]))
    append_members(path, captures[2:])

    rows = odds_log.read_rows(path)
    assert rows == [as_logged(r) for r in captures[0] + captures[2]]
    latest = odds_log.latest_prices(rows)
    assert latest[(race(11), "WIN", "1")]["odds"] == 2.8
    assert instant(latest[(race(11), "WIN", "1")]["snapshot_utc"]) == T0 + timedelta(seconds=60)
    assert latest[(race(12), "WIN", "2")]["odds"] == 8.0
    totals = odds_log.latest_totals(rows)
    assert {k: v["investment"] for k, v in totals.items()} == {(race(11), "WIN"): 5000.0}


# ---------------------------------------------------------------- latest views


def test_latest_prices_keep_the_newest_row_and_skip_pool_totals(tmp_path):
    t30, t10 = T0 + timedelta(seconds=30), T0 + timedelta(seconds=10)
    captures = [
        [
            price("1", 3.4, at=T0),
            price("2", 8.0, at=T0),
            price("1-2", 12.0, at=T0, pool="QUINELLA"),
            price("1", 5.0, at=T0, race_no=12),
            total("WIN", 1000.0, at=T0, odds=2.5),  # a total carrying a price is still not a price
        ],
        [
            price("1", 3.1, at=t30.astimezone(JST)),  # same instant, given in Tokyo time
            price("2", None, at=t30),
            total("WIN", 1500.0, at=t30),
        ],
        [price("1", 99.0, at=t10)],  # appended later but captured earlier: never the latest
        [price("2", 7.7, at=t30)],  # same instant as the SCR row: the later row in the log wins
    ]
    path = log_path(tmp_path)
    append_members(path, captures)
    latest = odds_log.latest_prices(odds_log.read_rows(path))

    expected = {  # key -> (odds, status_raw, snapshot)
        (race(11), "WIN", "1"): (3.1, "14時00分現在", t30),
        (race(11), "WIN", "2"): (7.7, "14時00分現在", t30),
        (race(11), "QUINELLA", "1-2"): (12.0, "14時00分現在", T0),  # not re-sent: still the latest
        (race(12), "WIN", "1"): (5.0, "14時00分現在", T0),
    }
    assert set(latest) == set(expected)
    for key, (odds, status, at) in expected.items():
        row = latest[key]
        assert (row["race_id"], row["pool"], row["combination"]) == key
        assert row["odds"] == odds and row["status_raw"] == status
        assert instant(row["snapshot_utc"]) == at
    assert all(combination != "__pool__" for _, _, combination in latest)
    assert odds_log.latest_prices([]) == {}


def test_latest_totals_are_per_race_and_pool(tmp_path):
    t30, t15 = T0 + timedelta(seconds=30), T0 + timedelta(seconds=15)
    captures = [
        [
            price("1", 3.4, at=T0),
            total("WIN", 1000.0, at=T0),
            total("PLACE", 500.0, at=T0),
            total("WIN", 700.0, at=T0, race_no=12),
            total("WIN", 100.0, at=T0, race_no=1),
            total("QUINELLA_PLACE", 5.0, at=T0, race_no=1),
        ],
        [
            price("1", 3.1, at=t30),
            total("WIN", 1500.0, at=t30),
            total("WIN", 999.0, at=t30, race_no=1),
        ],
        [total("WIN", 1200.0, at=t15)],  # appended later but captured earlier: never the latest
    ]
    path = log_path(tmp_path)
    append_members(path, captures)
    rows = odds_log.read_rows(path)
    totals = odds_log.latest_totals(rows)

    expected = {
        (race(11), "WIN"): (1500.0, t30),
        (race(11), "PLACE"): (500.0, T0),
        (race(12), "WIN"): (700.0, T0),
        (race(1), "WIN"): (999.0, t30),
        (race(1), "QUINELLA_PLACE"): (5.0, T0),
    }
    assert set(totals) == set(expected)
    for key, (investment, at) in expected.items():
        row = totals[key]
        assert (row["race_id"], row["pool"]) == key and row["combination"] == "__pool__"
        assert row["investment"] == investment
        assert instant(row["snapshot_utc"]) == at
    assert odds_log.latest_totals([as_logged(price("1", 3.4, at=T0))]) == {}
    assert odds_log.latest_totals([]) == {}


def test_odds_hi_round_trips_and_rows_without_it_are_unchanged(tmp_path):
    t1 = T0 + timedelta(seconds=30)
    first = [
        price("1", 2.2, at=T0),
        price("2", 3.2, at=T0),
        price("1", 1.3, at=T0, pool="PLACE", odds_hi=1.6),
        price("2", 1.5, at=T0, pool="PLACE", odds_hi=1.9),
        price("3", None, at=T0, pool="PLACE", odds_hi=None),
        price("1-2", 4.1, at=T0, pool="QUINELLA_PLACE", odds_hi=6.35),
    ]
    second = [price("1", 1.4, at=t1, pool="PLACE", odds_hi=1.7)]
    path = log_path(tmp_path)
    append_members(path, [first, second])

    rows = odds_log.read_rows(path)
    assert rows == [as_logged(r) for r in first + second]
    assert [set(r) for r in rows[:2]] == [HKTRACK_KEYS] * 2  # nothing added to rows without it
    assert [r["odds_hi"] for r in rows[2:]] == [1.6, 1.9, None, 6.35, 1.7]

    latest = odds_log.latest_prices(rows)
    assert (
        latest[(race(11), "PLACE", "1")]["odds"],
        latest[(race(11), "PLACE", "1")]["odds_hi"],
    ) == (
        1.4,
        1.7,
    )
    assert latest[(race(11), "PLACE", "2")]["odds_hi"] == 1.9
    assert latest[(race(11), "PLACE", "3")]["odds_hi"] is None
    assert latest[(race(11), "QUINELLA_PLACE", "1-2")]["odds_hi"] == 6.35
    for combination in ("1", "2"):
        assert "odds_hi" not in latest[(race(11), "WIN", combination)]
    assert latest[(race(11), "WIN", "2")] == as_logged(first[1])


# ---------------------------------------------------------------- HKTrack logs


def hktrack_member(rows: list[dict]) -> bytes:
    """A member exactly as HKTrack's capture_odds_snapshot wrote it."""
    payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    return gzip.compress(payload.encode("utf-8"))


def test_reads_a_log_in_hktrack_format(tmp_path):
    """No fetched_utc, no page_state; status_raw holds the page's time text; ASCII-escaped JSON."""
    first = [
        {
            "snapshot_utc": "2026-11-28T05:00:12.345678+00:00",
            "race_id": race(11),
            "pool": "WIN",
            "combination": "1",
            "odds": 3.4,
            "status_raw": "14時00分現在",
            "source_update_utc": "2026-11-28T05:00:00+00:00",
        },
        {
            "snapshot_utc": "2026-11-28T05:00:12.345678+00:00",
            "race_id": race(11),
            "pool": "WIN",
            "combination": "5",
            "odds": None,
            "status_raw": "SCR",
            "source_update_utc": "2026-11-28T05:00:00+00:00",
        },
        {
            "snapshot_utc": "2026-11-28T05:00:12.345678+00:00",
            "race_id": race(11),
            "pool": "PLACE",
            "combination": "1",
            "odds": 1.3,
            "odds_hi": 1.8,
            "status_raw": "14時00分現在",
            "source_update_utc": "2026-11-28T05:00:00+00:00",
        },
        {
            "snapshot_utc": "2026-11-28T05:00:12.345678+00:00",
            "race_id": race(11),
            "pool": "WIN",
            "combination": "__pool__",
            "odds": None,
            "status_raw": None,
            "investment": 81234500.0,
            "source_update_utc": "2026-11-28T05:00:00+00:00",
        },
    ]
    second = [
        {
            "snapshot_utc": "2026-11-28T05:20:40.000001+00:00",
            "race_id": race(12),
            "pool": "WIN",
            "combination": "7",
            "odds": 21.6,
            "status_raw": "最終オッズ",
            "source_update_utc": None,
        }
    ]
    path = log_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(hktrack_member(first) + hktrack_member(second))
    assert b"\\u" in gzip.decompress(path.read_bytes())  # HKTrack escaped the Japanese text

    rows = odds_log.read_rows(path)
    assert rows == first + second
    assert all("fetched_utc" not in r and "page_state" not in r for r in rows)

    later = T0 + timedelta(minutes=1)
    new = [
        price("1", 3.0, at=later, fetched_utc=later - timedelta(seconds=2), page_state="発売中"),
        total("WIN", 82000000.0, at=later),
    ]
    assert odds_log.append_capture(path, new) == 2
    rows = odds_log.read_rows(path)
    assert rows == first + second + [as_logged(r) for r in new]

    latest = odds_log.latest_prices(rows)
    assert set(latest) == {
        (race(11), "WIN", "1"),
        (race(11), "WIN", "5"),
        (race(11), "PLACE", "1"),
        (race(12), "WIN", "7"),
    }
    assert latest[(race(11), "WIN", "1")]["odds"] == 3.0  # the new capture is newer
    assert latest[(race(11), "WIN", "5")]["status_raw"] == "SCR"
    assert latest[(race(11), "PLACE", "1")] == first[2]
    assert latest[(race(12), "WIN", "7")]["status_raw"] == "最終オッズ"
    totals = odds_log.latest_totals(rows)
    assert {k: v["investment"] for k, v in totals.items()} == {(race(11), "WIN"): 82000000.0}


# ---------------------------------------------------------------- post-time sidecar


def test_post_times_sidecar_round_trips_and_is_written_atomically(tmp_path, monkeypatch):
    folder = tmp_path / "data" / "odds_snapshots" / "jra" / "2026-11-28"
    folder.mkdir(parents=True)
    path = folder / "TOK.races.json"
    assert odds_log.read_post_times(path) == {}  # missing: empty, no error

    first = {
        1: datetime(2026, 11, 28, 10, 5, tzinfo=JST),
        2: datetime(2026, 11, 28, 10, 40, tzinfo=TOKYO),
        11: datetime(2026, 11, 28, 6, 40, tzinfo=UTC),
        12: datetime(2026, 11, 28, 7, 20, 30, tzinfo=UTC),
    }
    odds_log.write_post_times(path, first, datetime(2026, 11, 28, 9, 30, tzinfo=JST))
    json.loads(path.read_text(encoding="utf-8"))  # a whole JSON document

    def check(got: dict, want: dict) -> None:
        assert set(got) == set(want) and all(type(k) is int for k in got)
        for number, at in want.items():
            assert got[number] == at, number
            assert got[number].tzinfo is not None and got[number].utcoffset() == timedelta(0)

    check(odds_log.read_post_times(path), first)

    second = {
        3: datetime(2026, 11, 28, 11, 15, tzinfo=JST),
        1: datetime(2026, 11, 28, 1, 6, tzinfo=UTC),
    }
    odds_log.write_post_times(path, second, datetime(2026, 11, 28, 1, 0, tzinfo=UTC))
    check(odds_log.read_post_times(path), second)  # replaced as a whole, not merged
    assert sorted(os.listdir(folder)) == ["TOK.races.json"]

    def failing_replace(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            odds_log.write_post_times(path, first, datetime(2026, 11, 28, 2, 0, tzinfo=UTC))
    check(odds_log.read_post_times(path), second)
    assert sorted(os.listdir(folder)) == ["TOK.races.json"]  # no temporary file left behind

    with pytest.raises(Exception):  # noqa: B017 - any failure to serialise will do
        odds_log.write_post_times(path, {4: object()}, datetime(2026, 11, 28, 2, 0, tzinfo=UTC))
    check(odds_log.read_post_times(path), second)
    assert sorted(os.listdir(folder)) == ["TOK.races.json"]


# ---------------------------------------------------------------- tailing (#22 as amended, #120)


def stored_member(rows: list[dict]) -> bytes:
    """One gzip member of `rows` with stored (uncompressed) deflate blocks: its length depends only
    on the payload's length, so a same-length payload can replace it in place."""
    payload = "".join(json.dumps(as_logged(r), separators=(",", ":")) + "\n" for r in rows)
    return gzip.compress(payload.encode("utf-8"), compresslevel=0, mtime=0)


def four_captures() -> list[list[dict]]:
    return [
        *three_captures(),
        [price(str(n), 4.0 + n, at=T0 + timedelta(seconds=90), race_no=12) for n in (1, 2, 3)],
    ]


@pytest.mark.parametrize("first", [0, 1, 2, 3, 4])
def test_iter_members_resumes_from_an_offset(tmp_path, first):
    path = log_path(tmp_path)
    captures = four_captures()
    sizes = append_members(path, captures)
    start = 0 if first == 0 else sizes[first - 1]
    logged = [[as_logged(r) for r in rows] for rows in captures]

    got = list(odds_log.iter_members_with_offsets(path, start=start))
    assert got == list(zip(sizes[first:], logged[first:], strict=True))
    if first == 0:
        assert list(odds_log.iter_members_with_offsets(path)) == got  # start defaults to 0
    # iter_members keeps its yield type (rows only) and takes the same offset
    assert list(odds_log.iter_members(path, start=start)) == logged[first:]
    assert list(odds_log.iter_members(path)) == logged
    # resuming from each yielded end offset gives exactly what follows it
    for i, (end, _) in enumerate(got):
        rest = list(odds_log.iter_members_with_offsets(path, start=end))
        assert rest == got[i + 1 :]
    missing = tmp_path / "nowhere" / "TOK.jsonl.gz"
    assert list(odds_log.iter_members_with_offsets(missing, start=start)) == []


TEARS = {
    "kill -9 near the end": lambda m: len(m) - 9,
    "inside the gzip header": lambda m: 6,
    "half the member": lambda m: len(m) // 2,
    "one byte": lambda m: 1,
}


@pytest.mark.parametrize("tear", list(TEARS.values()), ids=list(TEARS))
def test_a_torn_trailing_member_is_retried_on_the_next_poll(tmp_path, tear):
    path = log_path(tmp_path)
    captures = four_captures()
    sizes = append_members(path, captures[:2])
    member = gzip.compress(
        "".join(
            json.dumps(as_logged(r), separators=(",", ":")) + "\n" for r in captures[2]
        ).encode()
    )
    cut = tear(member)
    with open(path, "ab") as fh:  # the write of the third capture is still in progress
        fh.write(member[:cut])

    logged = [[as_logged(r) for r in rows] for rows in captures]
    got = list(odds_log.iter_members_with_offsets(path))
    assert got == [(sizes[0], logged[0]), (sizes[1], logged[1])]
    assert list(odds_log.iter_members_with_offsets(path, start=sizes[1])) == []

    tail = odds_log.LogTail(path)
    assert tail.poll() == logged[0] + logged[1]
    assert tail.poll() == []  # still torn: nothing new, nothing repeated
    with open(path, "ab") as fh:  # the write completes
        fh.write(member[cut:])
    assert tail.poll() == logged[2]
    assert tail.poll() == []
    odds_log.append_capture(path, captures[3])
    assert tail.poll() == logged[3]
    assert odds_log.read_rows(path) == [row for rows in logged for row in rows]


def test_a_torn_member_left_behind_by_a_restarted_logger_is_skipped(tmp_path):
    """A capture whose write never completes is lost when the logger restarts and appends after it,
    as `read_rows` skips it; the tail then reads on."""
    path = log_path(tmp_path)
    captures = four_captures()
    append_members(path, captures[:1])
    member = gzip.compress(
        "".join(
            json.dumps(as_logged(r), separators=(",", ":")) + "\n" for r in captures[1]
        ).encode()
    )
    with open(path, "ab") as fh:
        fh.write(member[: len(member) // 2])
    tail = odds_log.LogTail(path)
    assert tail.poll() == [as_logged(r) for r in captures[0]]
    append_members(path, captures[2:])
    assert tail.poll() == [as_logged(r) for r in captures[2] + captures[3]]
    assert tail.poll() == []


def test_log_tail_returns_only_new_rows(tmp_path):
    path = log_path(tmp_path)
    captures = four_captures()
    first = stored_member(captures[0])
    path.parent.mkdir(parents=True)
    path.write_bytes(first)

    tail = odds_log.LogTail(path)
    assert tail.poll() == [as_logged(r) for r in captures[0]]
    assert tail.poll() == []

    # The bytes already read are changed in place (same length). A tail reads on from where the
    # last whole member ended, so it never sees them again; a reader that starts over would.
    changed = [dict(r, odds=None if r["odds"] is None else r["odds"] + 0.1) for r in captures[0]]
    replaced = stored_member(changed)
    assert len(replaced) == len(first) and replaced != first
    path.write_bytes(replaced)

    odds_log.append_capture(path, captures[1])
    assert tail.poll() == [as_logged(r) for r in captures[1]]
    assert tail.poll() == []
    append_members(path, captures[2:])
    assert tail.poll() == [as_logged(r) for r in captures[2] + captures[3]]
    assert tail.poll() == []

    # a second tail on the same file starts at its beginning and is independent of the first
    other = odds_log.LogTail(path)
    whole = [as_logged(r) for r in changed + captures[1] + captures[2] + captures[3]]
    assert other.poll() == whole
    assert tail.poll() == []


def test_log_tail_on_a_missing_file_is_empty(tmp_path):
    path = log_path(tmp_path)
    tail = odds_log.LogTail(path)
    assert tail.poll() == []
    assert tail.poll() == []
    assert not path.exists() and not path.parent.exists()  # polling creates nothing

    captures = four_captures()
    odds_log.append_capture(path, captures[0])
    assert tail.poll() == [as_logged(r) for r in captures[0]]
    assert tail.poll() == []
