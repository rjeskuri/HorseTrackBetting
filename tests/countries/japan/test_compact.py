"""Compacting a day's odds logs into a Parquet time series (ticket #32).

Contract pinned here:
- `compact_day(race_date, source="jra") -> Path` (in `horsetrack.countries.japan.logger.compact`)
  reads every `<VENUE>.jsonl.gz` and `<VENUE>.races.json` under `odds_snapshots/<source>/<date>/`
  and writes `processed/<source>/odds_timeseries/<YYYY-MM-DD>.parquet`, returning that path.
- One row per logged row, pool-total ("__pool__") rows included, with at least the columns
  race_id, venue, race_no, pool, combination, snapshot_utc, fetched_utc, post_utc, minutes_to_post,
  odds, odds_hi, investment, status_raw, is_final_capture, dividend, is_refund.
- snapshot_utc, fetched_utc and post_utc are tz-aware UTC timestamps in the Parquet schema, even
  when a column is all null; absent optional keys (fetched_utc, odds_hi, investment) are null.
- minutes_to_post = (post_utc - (fetched_utc or snapshot_utc)) in minutes, from the venue's own
  sidecar by race number; negative after the post; null when the venue has no sidecar or the
  sidecar has no entry for the race.
- (#411) Where the sidecar has no entry for a race (or there is no sidecar), post_utc comes from
  the stored race card, `processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet` (`race_no`,
  `post_time_local`, Tokyo time: the card `board.snapshot.post_times_for` reads). Per race the
  sidecar's entry wins; a race on neither, or with a null post time on the card, stays null.
- is_final_capture marks the row with the latest snapshot_utc instant per (race_id, pool,
  combination), pool totals included; exactly one per key, and at the same instant the later row
  in the log wins (as `odds_log.latest_prices` does).
- dividend / is_refund: a left join on exact (race_id, pool, combination) to
  `processed/<source>/dividends.parquet`; null where there is no match or no table. The day file's
  schema does not depend on whether the table exists.
- Identical inputs give identical bytes; a re-run rebuilds the file from the logs; the write leaves
  no temporary file beside it.
- A day with no logs raises FileNotFoundError and writes nothing.
- `horsetrack japan compact --date D` compacts one day (non-zero exit when it has no logs);
  `--from D1 --to D2` compacts every day in the closed range that has logs and skips the rest.
  Exactly one of `--date` or `--from/--to` is required.

Not pinned: row order (only that it is deterministic), column order, extra columns, timestamp unit.
All fixtures are synthetic and built in tmp_path.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log

UTC = timezone.utc
D = "2026-11-22"
POOL = odds_log.POOL_TOTAL
COLUMNS = {
    "race_id",
    "venue",
    "race_no",
    "pool",
    "combination",
    "snapshot_utc",
    "fetched_utc",
    "post_utc",
    "minutes_to_post",
    "odds",
    "odds_hi",
    "investment",
    "status_raw",
    "is_final_capture",
    "dividend",
    "is_refund",
}
TIMESTAMPS = ("snapshot_utc", "fetched_utc", "post_utc")
UTC_NAMES = {"UTC", "+00:00", "Etc/UTC"}
runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    """The data directory under tmp_path; no real `.env` can supply a setting."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    return tmp_path / "data"


def compact_day(*args, **kwargs) -> Path:
    """The unit under test, imported per call so each test fails on its own while it is missing."""
    from horsetrack.countries.japan.logger.compact import compact_day as impl

    return impl(*args, **kwargs)


# ---------------------------------------------------------------- synthetic logs


def at(day: str, hh: int, mm: int, ss: int = 0) -> datetime:
    y, m, d = (int(part) for part in day.split("-"))
    return datetime(y, m, d, hh, mm, ss, tzinfo=UTC)


def rid(venue: str, no: int, day: str = D) -> str:
    return f"jra:{day}:{venue}:{no:02d}"


def price(venue, no, pool, combination, odds, when, *, day=D, **extra) -> dict:
    row = {
        "snapshot_utc": when,
        "race_id": rid(venue, no, day),
        "pool": pool,
        "combination": combination,
        "odds": odds,
        "status_raw": "SCR" if odds is None else f"{when.hour + 9}時{when.minute:02d}分現在",
        "source_update_utc": when - timedelta(seconds=12),
    }
    row.update(extra)
    return row


def total(venue, no, pool, investment, when, *, day=D) -> dict:
    return {
        "snapshot_utc": when,
        "race_id": rid(venue, no, day),
        "pool": pool,
        "combination": POOL,
        "odds": None,
        "investment": investment,
        "status_raw": None,
        "source_update_utc": when - timedelta(seconds=12),
    }


def write_log(venue: str, captures: list[list[dict]], *, day: str = D) -> list[dict]:
    """Append each capture as the logger does; returns every row written, in log order."""
    path = paths.odds_snapshot_path("jra", day, venue)
    written = []
    for rows in captures:
        assert odds_log.append_capture(path, rows) == len(rows)
        written.extend(rows)
    return written


def write_sidecar(venue: str, post_times: dict[int, datetime], *, day: str = D) -> None:
    path = paths.odds_snapshot_path("jra", day, venue).with_name(f"{venue}.races.json")
    odds_log.write_post_times(path, post_times, updated_utc=at(day, 0, 30))


def write_dividends(rows: list[tuple]) -> Path:
    """A synthetic dividends.parquet in the HKTrack store shape."""
    path = paths.processed_dir("jra") / "dividends.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        rows, columns=["jurisdiction", "race_id", "pool", "combination", "dividend", "is_refund"]
    )
    frame["dividend"] = frame["dividend"].astype("float64")
    frame["is_refund"] = frame["is_refund"].astype("bool")
    frame.to_parquet(path, index=False)
    return path


def day_file(day: str = D) -> Path:
    return paths.data_dir() / "processed" / "jra" / "odds_timeseries" / f"{day}.parquet"


def read_day(path: Path) -> tuple[pa.Schema, list[dict]]:
    table = pq.read_table(path)
    return table.schema, table.to_pylist()


def key(row: dict) -> tuple:
    return (row["race_id"], row["pool"], row["combination"], row["snapshot_utc"])


def by_key(rows: list[dict]) -> dict[tuple, dict]:
    out = {}
    for row in rows:
        assert key(row) not in out, f"fixture keys must be unique here: {key(row)}"
        out[key(row)] = row
    return out


def is_null(value) -> bool:
    return value is None or (not isinstance(value, (str, bool)) and bool(pd.isna(value)))


def assert_utc_timestamp(schema: pa.Schema, name: str) -> None:
    kind = schema.field(name).type
    assert pa.types.is_timestamp(kind), f"{name} is {kind}, not a timestamp"
    assert kind.tz in UTC_NAMES, f"{name} carries tz {kind.tz!r}, not UTC"


def plain(text: str) -> str:
    return _ANSI.sub("", text)


def invoke(*args: str):
    from horsetrack.cli import app

    return runner.invoke(app, ["japan", "compact", *args])


# ---------------------------------------------------------------- the named tests


def test_a_day_compacts_to_one_row_per_logged_row():
    t = [at(D, 6, 20), at(D, 6, 30), at(D, 6, 35, 30), at(D, 6, 41)]
    tok = write_log(
        "TOK",
        [
            [
                price("TOK", 11, "WIN", "1", 3.4, t[0]),
                price("TOK", 11, "WIN", "2", 8.0, t[0], fetched_utc=t[0] - timedelta(seconds=7)),
                price("TOK", 11, "WIN", "3", None, t[0]),
                price("TOK", 11, "PLACE", "1", 1.2, t[0], odds_hi=1.6),
                total("TOK", 11, "WIN", 1234500.0, t[0]),
                price("TOK", 12, "WIN", "1", 15.5, t[0]),
            ],
            [
                price("TOK", 11, "WIN", "1", 3.1, t[1], fetched_utc=t[1] - timedelta(seconds=3)),
                price("TOK", 11, "PLACE", "1", 1.1, t[1], odds_hi=1.4),
                price("TOK", 11, "QUINELLA", "1-2", 12.5, t[1]),
                total("TOK", 11, "WIN", 2345600.0, t[1]),
            ],
            [price("TOK", 11, "WIN", "1", 2.9, t[3])],
        ],
    )
    kyo = write_log(
        "KYO",
        [
            [price("KYO", 11, "WIN", "4", 5.6, t[2]), total("KYO", 11, "WIN", 99000.0, t[2])],
            [price("KYO", 11, "WIN", "4", 5.2, t[3])],
        ],
    )
    other_day = "2026-11-23"
    write_log(
        "TOK",
        [[price("TOK", 1, "WIN", "1", 2.0, at(other_day, 1, 0), day=other_day)]],
        day=other_day,
    )
    write_sidecar("TOK", {11: at(D, 6, 40), 12: at(D, 7, 20)})
    write_sidecar("KYO", {11: at(D, 6, 25)})
    logged = tok + kyo

    path = compact_day(D)

    assert Path(path) == day_file(D)
    assert day_file(D).is_file()
    schema, rows = read_day(day_file(D))
    assert COLUMNS <= set(schema.names), f"missing columns: {COLUMNS - set(schema.names)}"
    for name in TIMESTAMPS:
        assert_utc_timestamp(schema, name)
    assert pa.types.is_integer(schema.field("race_no").type)
    assert pa.types.is_boolean(schema.field("is_final_capture").type)

    assert len(rows) == len(logged)
    assert Counter(key(r) for r in rows) == Counter(key(r) for r in logged)
    assert all(r["race_id"].startswith(f"jra:{D}:") for r in rows), "another day's log leaked in"

    out = by_key(rows)
    for want in logged:
        got = out[key(want)]
        venue, no = want["race_id"].split(":")[2], int(want["race_id"].split(":")[3])
        assert got["venue"] == venue
        assert got["race_no"] == no
        assert got["odds"] == want["odds"] or (want["odds"] is None and is_null(got["odds"]))
        for optional in ("odds_hi", "investment", "fetched_utc"):
            if optional in want:
                assert got[optional] == want[optional], (optional, want)
            else:
                assert is_null(got[optional]), (optional, want)
        assert got["status_raw"] == want["status_raw"] or (
            want["status_raw"] is None and is_null(got["status_raw"])
        )
        assert got["snapshot_utc"].utcoffset() == timedelta(0)


def test_minutes_to_post_from_the_sidecar():
    post = {11: at(D, 6, 40), 12: at(D, 7, 20)}
    write_sidecar("TOK", post)
    instants = [
        at(D, 6, 20),  # T-20 for race 11, T-60 for race 12
        at(D, 6, 34, 30),  # T-5.5
        at(D, 6, 39, 45),  # T-0.25
        at(D, 6, 40),  # at the post
        at(D, 6, 41, 30),  # after the post: negative
        at(D, 7, 23),  # well after race 11, after race 12 too
    ]
    logged = write_log(
        "TOK",
        [
            [
                price("TOK", 11, "WIN", "1", 3.0, when),
                price("TOK", 12, "WIN", "1", 9.0, when),
                total("TOK", 12, "WIN", 1000.0, when),
            ]
            for when in instants
        ],
    )

    _, rows = read_day(compact_day(D))

    assert len(rows) == len(logged)
    for row in rows:
        no = int(row["race_id"].split(":")[3])
        assert row["post_utc"] == post[no]
        expected = (post[no] - row["snapshot_utc"]).total_seconds() / 60
        assert row["minutes_to_post"] == pytest.approx(expected, abs=1e-9), row
    after = [r for r in rows if r["race_id"] == rid("TOK", 11) and r["snapshot_utc"] == instants[4]]
    assert [r["minutes_to_post"] for r in after] == [pytest.approx(-1.5)]
    at_post = [
        r for r in rows if r["race_id"] == rid("TOK", 11) and r["snapshot_utc"] == instants[3]
    ]
    assert [r["minutes_to_post"] for r in at_post] == [pytest.approx(0.0)]
    race12_late = [
        r for r in rows if r["race_id"] == rid("TOK", 12) and r["snapshot_utc"] == instants[5]
    ]
    assert [r["minutes_to_post"] for r in race12_late] == [pytest.approx(-3.0)] * 2
    assert len(race12_late) == 2  # the price row and the pool-total row


def test_fetched_utc_is_preferred_over_snapshot_utc():
    post = at(D, 6, 40)
    write_sidecar("TOK", {11: post})
    snap = at(D, 6, 30)
    cases = {  # combination -> (extra keys, the instant minutes_to_post counts from)
        "1": ({}, snap),
        "2": ({"fetched_utc": snap - timedelta(seconds=40)}, snap - timedelta(seconds=40)),
        "3": ({"fetched_utc": snap - timedelta(minutes=3)}, snap - timedelta(minutes=3)),
        "4": ({"fetched_utc": None}, snap),  # present but null: the snapshot instant counts
    }
    write_log(
        "TOK", [[price("TOK", 11, "WIN", c, 4.0, snap, **extra) for c, (extra, _) in cases.items()]]
    )

    _, rows = read_day(compact_day(D))

    assert len(rows) == len(cases)
    for row in rows:
        _, origin = cases[row["combination"]]
        assert row["minutes_to_post"] == pytest.approx((post - origin).total_seconds() / 60)
    got = {r["combination"]: r["minutes_to_post"] for r in rows}
    assert got["2"] == pytest.approx(10 + 40 / 60)
    assert got["3"] == pytest.approx(13.0)
    assert got["1"] == got["4"] == pytest.approx(10.0)


def test_without_post_times_minutes_to_post_is_null():
    when = at(D, 6, 30)
    write_sidecar("TOK", {11: at(D, 6, 40)})  # no entry for race 12
    write_log(
        "TOK",
        [[price("TOK", 11, "WIN", "1", 3.0, when), price("TOK", 12, "WIN", "1", 7.0, when)]],
    )
    write_log("KYO", [[price("KYO", 11, "WIN", "1", 5.0, when)]])  # KYO has no sidecar at all

    schema, rows = read_day(compact_day(D))

    assert_utc_timestamp(schema, "post_utc")
    assert pa.types.is_floating(schema.field("minutes_to_post").type)
    got = {r["race_id"]: r for r in rows}
    assert set(got) == {rid("TOK", 11), rid("TOK", 12), rid("KYO", 11)}
    assert got[rid("TOK", 11)]["minutes_to_post"] == pytest.approx(10.0)  # the control
    for race_id in (rid("TOK", 12), rid("KYO", 11)):
        assert is_null(got[race_id]["post_utc"]), race_id
        assert is_null(got[race_id]["minutes_to_post"]), race_id

    # A day with no sidecar anywhere and no fetched_utc keys: the columns are null, still typed.
    other = "2026-11-21"
    write_log("NAK", [[price("NAK", 5, "WIN", "2", 6.0, at(other, 3, 0), day=other)]], day=other)
    schema, rows = read_day(compact_day(other))
    assert len(rows) == 1
    for name in TIMESTAMPS:
        assert_utc_timestamp(schema, name)
    assert pa.types.is_floating(schema.field("minutes_to_post").type)
    assert is_null(rows[0]["post_utc"])
    assert is_null(rows[0]["fetched_utc"])
    assert is_null(rows[0]["minutes_to_post"])


def test_dividends_join_by_race_pool_and_combination():
    t1, t2 = at(D, 6, 30), at(D, 6, 39)
    write_dividends(
        [
            ("jra", rid("TOK", 11), "WIN", "3", 450.0, False),
            ("jra", rid("TOK", 11), "PLACE", "3", 160.0, False),
            ("jra", rid("TOK", 11), "QUINELLA", "1-3", 1230.0, False),
            ("jra", rid("TOK", 12), "WIN", "5", 100.0, True),
            ("jra", rid("KYO", 11), "WIN", "3", 999.0, False),
            ("jra", rid("TOK", 10), "WIN", "1", 310.0, False),  # a race not in the logs
        ]
    )
    captures = [
        [
            price("TOK", 11, "WIN", "3", 4.5, when),
            price("TOK", 11, "WIN", "1", 2.0, when),  # no dividend for runner 1
            price("TOK", 11, "PLACE", "3", 1.5, when, odds_hi=1.9),  # same runner, other pool
            price("TOK", 11, "PLACE", "1", 1.1, when, odds_hi=1.3),
            price("TOK", 11, "QUINELLA", "1-3", 12.0, when),
            price("TOK", 11, "QUINELLA", "1-2", 20.0, when),  # a near miss
            price("TOK", 11, "QUINELLA", "3-1", 12.0, when),  # not the store's spelling
            price("TOK", 11, "EXACTA", "3", 1.0, when),  # a pool the table lacks
            total("TOK", 11, "WIN", 5000.0, when),
            price("TOK", 12, "WIN", "5", 30.0, when),  # refunded
            price("TOK", 12, "WIN", "3", 6.0, when),  # runner 3 won race 11, not race 12
        ]
        for when in (t1, t2)
    ]
    logged = write_log("TOK", captures)
    logged += write_log("KYO", [[price("KYO", 11, "WIN", "3", 7.0, t2)]])
    expected = {  # (race_id, pool, combination) -> (dividend, is_refund), derived by hand
        (rid("TOK", 11), "WIN", "3"): (450.0, False),
        (rid("TOK", 11), "PLACE", "3"): (160.0, False),
        (rid("TOK", 11), "QUINELLA", "1-3"): (1230.0, False),
        (rid("TOK", 12), "WIN", "5"): (100.0, True),
        (rid("KYO", 11), "WIN", "3"): (999.0, False),
    }

    _, rows = read_day(compact_day(D))

    assert len(rows) == len(logged), "the join must not add or drop rows"
    matched = Counter()
    for row in rows:
        k = (row["race_id"], row["pool"], row["combination"])
        if k in expected:
            dividend, refund = expected[k]
            assert row["dividend"] == pytest.approx(dividend), k
            assert row["is_refund"] is refund, k
            matched[k] += 1
        else:
            assert is_null(row["dividend"]), k
            assert is_null(row["is_refund"]), k
    assert matched[(rid("TOK", 11), "WIN", "3")] == 2, "every capture of a combination joins"
    assert set(matched) == set(expected)


def test_without_a_dividends_table_the_columns_are_null():
    when = at(D, 6, 30)
    logged = write_log(
        "TOK",
        [
            [
                price("TOK", 11, "WIN", "3", 4.5, when),
                price("TOK", 11, "PLACE", "3", 1.5, when, odds_hi=1.9),
                total("TOK", 11, "WIN", 5000.0, when),
            ]
        ],
    )
    assert not (paths.processed_dir("jra") / "dividends.parquet").exists()

    schema_without, rows = read_day(compact_day(D))

    assert {"dividend", "is_refund"} <= set(schema_without.names)
    assert len(rows) == len(logged)
    assert all(is_null(r["dividend"]) and is_null(r["is_refund"]) for r in rows)

    # Once the store has dividends the same day file gains values, not a different schema.
    write_dividends([("jra", rid("TOK", 11), "WIN", "3", 450.0, False)])
    schema_with, rows = read_day(compact_day(D))
    assert {r["dividend"] for r in rows if r["pool"] == "WIN" and r["combination"] == "3"} == {
        450.0
    }
    assert {f.name: f.type for f in schema_with} == {f.name: f.type for f in schema_without}


def test_the_last_capture_of_a_combination_is_flagged_final():
    t1, t2, t3 = at(D, 6, 20), at(D, 6, 30), at(D, 6, 38)
    captures = [
        [  # t1
            price("TOK", 11, "WIN", "1", 3.4, t1),
            price("TOK", 11, "WIN", "2", 8.0, t1),
            price("TOK", 11, "PLACE", "1", 1.2, t1, odds_hi=1.5),
            price("TOK", 12, "WIN", "1", 11.0, t1),
            total("TOK", 11, "WIN", 1000.0, t1),
        ],
        [  # t3, logged before t2: the instant decides, not the log order
            price("TOK", 11, "WIN", "1", 2.9, t3),
            total("TOK", 11, "WIN", 3000.0, t3),
        ],
        [  # t2
            price("TOK", 11, "WIN", "1", 3.1, t2),
            price("TOK", 11, "WIN", "2", 7.5, t2),
            price("TOK", 11, "PLACE", "1", 1.1, t2, odds_hi=1.4),
            total("TOK", 11, "WIN", 2000.0, t2),
        ],
    ]
    logged = write_log("TOK", captures)
    logged += write_log("KYO", [[price("KYO", 11, "WIN", "1", 4.0, t1)]])

    _, rows = read_day(compact_day(D))

    assert len(rows) == len(logged)
    latest: dict[tuple, datetime] = {}
    for row in logged:
        k = (row["race_id"], row["pool"], row["combination"])
        latest[k] = max(latest.get(k, row["snapshot_utc"]), row["snapshot_utc"])
    flagged = Counter()
    for row in rows:
        k = (row["race_id"], row["pool"], row["combination"])
        assert row["is_final_capture"] is (row["snapshot_utc"] == latest[k]), (k, row)
        flagged[k] += row["is_final_capture"]
    assert flagged == Counter({k: 1 for k in latest}), "exactly one final row per combination"
    final = {(r["race_id"], r["pool"], r["combination"]): r for r in rows if r["is_final_capture"]}
    assert final[(rid("TOK", 11), "WIN", "1")]["odds"] == 2.9
    assert final[(rid("TOK", 11), "WIN", "2")]["odds"] == 7.5
    assert final[(rid("TOK", 11), "PLACE", "1")]["odds"] == 1.1
    assert final[(rid("TOK", 12), "WIN", "1")]["odds"] == 11.0
    assert final[(rid("KYO", 11), "WIN", "1")]["odds"] == 4.0
    assert final[(rid("TOK", 11), "WIN", POOL)]["investment"] == 3000.0


def test_a_tie_at_the_last_instant_flags_the_later_logged_row():
    t1, t2 = at(D, 6, 20), at(D, 6, 30)
    write_log(
        "TOK",
        [
            [price("TOK", 11, "WIN", "1", 3.4, t1)],
            [price("TOK", 11, "WIN", "1", 3.0, t2)],
            [price("TOK", 11, "WIN", "1", 2.8, t2)],  # the same instant, logged later
        ],
    )

    _, rows = read_day(compact_day(D))

    assert len(rows) == 3
    assert [r["odds"] for r in rows if r["is_final_capture"]] == [2.8]


def build_day(*, venues_order: tuple[str, ...]) -> None:
    """A day with two venues, sidecars and a dividends table, venues written in the given order."""
    t = [at(D, 6, 20), at(D, 6, 30, 15), at(D, 6, 41)]
    logs = {
        "TOK": [
            [
                price("TOK", 11, "WIN", "1", 3.4, w, fetched_utc=w - timedelta(seconds=5)),
                price("TOK", 11, "PLACE", "1", 1.2, w, odds_hi=1.5),
                total("TOK", 11, "WIN", 1000.0 * (i + 1), w),
                price("TOK", 12, "WIN", "2", None, w),
            ]
            for i, w in enumerate(t)
        ],
        "KYO": [[price("KYO", 11, "WIN", "3", 5.0 - i, w)] for i, w in enumerate(t)],
    }
    for venue in venues_order:
        write_log(venue, logs[venue])
    write_sidecar("TOK", {11: at(D, 6, 40)})
    write_sidecar("KYO", {11: at(D, 6, 25)})
    write_dividends([("jra", rid("KYO", 11), "WIN", "3", 640.0, False)])


def test_compaction_is_byte_stable(tmp_path, monkeypatch):
    build_day(venues_order=("TOK", "KYO"))
    first = compact_day(D).read_bytes()
    second = compact_day(D).read_bytes()
    assert first == second

    # The same logs written into a fresh data directory, venues in the other order.
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data-again"))
    build_day(venues_order=("KYO", "TOK"))
    assert compact_day(D).read_bytes() == first


def test_a_rerun_rebuilds_the_day_file_from_the_logs_and_leaves_no_temp_file():
    when = at(D, 6, 30)
    write_log("TOK", [[price("TOK", 11, "WIN", "1", 3.0, when)]])
    day_file(D).parent.mkdir(parents=True, exist_ok=True)
    day_file(D).write_bytes(b"stale, not parquet")  # whatever was there is replaced

    _, rows = read_day(compact_day(D))
    assert len(rows) == 1

    write_log(
        "TOK",
        [
            [
                price("TOK", 11, "WIN", "1", 2.5, when + timedelta(minutes=1)),
                price("TOK", 11, "WIN", "2", 9.0, when + timedelta(minutes=1)),
            ]
        ],
    )
    _, rows = read_day(compact_day(D))
    assert len(rows) == 3, "rebuilt from the logs, not appended to the previous file"
    assert sorted(r["odds"] for r in rows) == [2.5, 3.0, 9.0]
    assert sorted(p.name for p in day_file(D).parent.iterdir()) == [f"{D}.parquet"]


def test_a_day_without_logs_raises_and_writes_nothing():
    write_log("TOK", [[price("TOK", 11, "WIN", "1", 3.0, at(D, 6, 30))]])  # another day has logs

    with pytest.raises(FileNotFoundError):
        compact_day("2026-11-25")

    assert not day_file("2026-11-25").exists()


def test_several_venues_in_one_day_use_their_own_sidecars():
    when = at(D, 6, 10)
    write_sidecar("TOK", {11: at(D, 6, 40)})
    write_sidecar("KYO", {11: at(D, 6, 25)})
    write_sidecar("NAK", {11: at(D, 7, 0)})
    for venue in ("TOK", "KYO", "NAK"):
        write_log(venue, [[price(venue, 11, "WIN", "1", 3.0, when)]])

    _, rows = read_day(compact_day(D))

    got = {r["venue"]: (r["race_id"], r["race_no"], r["minutes_to_post"]) for r in rows}
    assert got == {
        "TOK": (rid("TOK", 11), 11, pytest.approx(30.0)),
        "KYO": (rid("KYO", 11), 11, pytest.approx(15.0)),
        "NAK": (rid("NAK", 11), 11, pytest.approx(50.0)),
    }


# ---------------------------------------------------------------- post times from the card (#411)


def write_card(venue: str, races: dict[int, str | None], *, day: str = D) -> Path:
    """The stored race card of one meeting, where `board.snapshot` reads it:
    `processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet`, with `post_time_local` the Tokyo
    clock as ISO text (with or without its +09:00 offset; without one it is Tokyo time)."""
    path = paths.processed_dir("jra") / "upcoming" / f"jra_{day}_{venue}" / "races.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    nos = sorted(races)
    table = pa.table(
        {
            "race_id": pa.array([rid(venue, no, day) for no in nos], pa.string()),
            "meeting_id": pa.array([f"jra:{day}:{venue}"] * len(nos), pa.string()),
            "race_no": pa.array(nos, pa.int64()),
            "race_class": pa.array(["C1"] * len(nos), pa.string()),
            "post_time_local": pa.array([races[no] for no in nos], pa.string()),
        }
    )
    pq.write_table(table, path)
    return path


def test_minutes_to_post_from_the_race_card_where_the_sidecar_has_none():
    """#411: a logged day whose venue has no `<VENUE>.races.json` entry for a race takes the race's
    post time from the stored race card, so `minutes_to_post` is set. Per race, the sidecar's
    entry wins over the card's; the card fills the races the sidecar does not list; a race on
    neither (or on the card with a null post time) stays null.

    TOK: card only. KYO: a sidecar for race 11 (which the card times differently) and the card for
    race 12, whose post time is written without an offset (Tokyo time). NAK: on the card with a
    null post time (race 5) and not on it (race 6). HAN: neither card nor sidecar."""
    write_card("TOK", {11: f"{D}T15:40:00+09:00", 12: f"{D}T16:15:00+09:00"})
    write_sidecar("KYO", {11: at(D, 6, 25)})
    write_card("KYO", {11: f"{D}T15:30:00+09:00", 12: f"{D}T16:05:00"})
    write_card("NAK", {5: None, 7: f"{D}T12:00:00+09:00"})
    expected_post = {
        rid("TOK", 11): at(D, 6, 40),
        rid("TOK", 12): at(D, 7, 15),
        rid("KYO", 11): at(D, 6, 25),  # the sidecar's, not the card's 06:30
        rid("KYO", 12): at(D, 7, 5),
        rid("NAK", 5): None,
        rid("NAK", 6): None,
        rid("HAN", 3): None,
    }
    instants = [at(D, 6, 0), at(D, 6, 21, 30), at(D, 6, 38), at(D, 7, 9)]
    for venue, nos in (("TOK", (11, 12)), ("KYO", (11, 12)), ("NAK", (5, 6)), ("HAN", (3,))):
        captures = []
        for i, when in enumerate(instants):
            capture = []
            for no in nos:
                extra = {"fetched_utc": when - timedelta(seconds=20)} if i == 1 else {}
                capture.append(price(venue, no, "WIN", "1", 3.0 + i, when, **extra))
                capture.append(total(venue, no, "WIN", 1e6 * (i + 1), when))
            captures.append(capture)
        write_log(venue, captures)

    schema, rows = read_day(compact_day(D))

    assert_utc_timestamp(schema, "post_utc")
    assert {r["race_id"] for r in rows} == set(expected_post)
    assert len(rows) == len(instants) * 2 * len(expected_post)
    for row in rows:
        post = expected_post[row["race_id"]]
        if post is None:
            assert is_null(row["post_utc"]), row
            assert is_null(row["minutes_to_post"]), row
            continue
        assert row["post_utc"] == post, row
        origin = row["fetched_utc"] if not is_null(row["fetched_utc"]) else row["snapshot_utc"]
        want = (post - origin).total_seconds() / 60
        assert row["minutes_to_post"] == pytest.approx(want, abs=1e-9), row


# ---------------------------------------------------------------- the CLI


def test_cli_compacts_one_date():
    when = at(D, 6, 30)
    logged = write_log(
        "TOK",
        [[price("TOK", 11, "WIN", "1", 3.0, when), total("TOK", 11, "WIN", 10.0, when)]],
    )

    result = invoke("--date", D)

    assert result.exit_code == 0, plain(result.output)
    _, rows = read_day(day_file(D))
    assert Counter(key(r) for r in rows) == Counter(key(r) for r in logged)


def test_cli_compacts_a_range_and_skips_days_without_logs():
    logged = {}
    for day, n in (("2026-11-20", 1), ("2026-11-21", 2), ("2026-11-23", 3), ("2026-11-24", 1)):
        logged[day] = write_log(
            "TOK",
            [
                [
                    price("TOK", no, "WIN", "1", 2.0 + no, at(day, 2, 0), day=day)
                    for no in range(1, n + 1)
                ]
            ],
            day=day,
        )

    result = invoke("--from", "2026-11-21", "--to", "2026-11-23")

    assert result.exit_code == 0, plain(result.output)
    for day in ("2026-11-21", "2026-11-23"):
        _, rows = read_day(day_file(day))
        assert Counter(key(r) for r in rows) == Counter(key(r) for r in logged[day]), day
    for day in ("2026-11-20", "2026-11-22", "2026-11-24"):
        assert not day_file(day).exists(), f"{day} is outside the range or has no logs"


def assert_compact_is_registered() -> None:
    """A refusal below counts only if `compact` exists: an unknown command also exits non-zero."""
    result = invoke("--help")
    assert result.exit_code == 0, plain(result.output)
    help_text = plain(result.output)
    assert "--date" in help_text and "--from" in help_text and "--to" in help_text


def test_cli_date_without_logs_fails():
    assert_compact_is_registered()

    result = invoke("--date", "2026-11-25")

    assert result.exit_code != 0
    assert not day_file("2026-11-25").exists()


@pytest.mark.parametrize(
    "args",
    [(), ("--date", D, "--from", D, "--to", D)],
    ids=["neither", "both"],
)
def test_cli_needs_either_a_date_or_a_range(args):
    write_log("TOK", [[price("TOK", 11, "WIN", "1", 3.0, at(D, 6, 30))]])
    assert_compact_is_registered()

    result = invoke(*args)

    assert result.exit_code != 0
    assert not day_file(D).exists()
