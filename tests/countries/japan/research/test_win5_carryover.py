"""The WIN5 carryover and its pool expected return (ticket #135).

Contract pinned here (the coder implements against it), module
`horsetrack.countries.japan.research.win5`:

Pool arithmetic (yen in, floats out; `TAKEOUT = 0.30`, WIN5's standard takeout):
- `pool_ev(pool_yen, carryover_yen, *, takeout=TAKEOUT) -> float`: the gross return per 1 yen of
  the whole pool, `(pool * (1 - takeout) + carryover) / pool`. No carryover gives 0.70; pool 1e9
  with carryover 5e8 gives 1.20. ValueError for a pool that is None, NaN, infinite, zero or
  negative, a carryover that is None, NaN or negative, and a takeout outside [0, 1).
- The chase model: a carryover draws extra money into the pool. With a base pool `P0` (what the
  day would have sold without the carryover) and a chase factor `k >= 0` (extra yen staked per yen
  of carryover), the pool is `P = P0 + k * C`, and the per-ticket expected return is the pool EV
  on that pool: `(1 - t) + C / (P0 + k * C)`. k = 0 is the naive figure; any k > 0 with C > 0 is
  strictly lower. Worked: P0 = 1e9, C = 5e8, t = 0.30, k = 1 -> P = 1.5e9 -> 0.7 + 1/3 = 1.0333.
  `chased_pool_ev(base_pool_yen, carryover_yen, *, chase, takeout=TAKEOUT) -> float`; ValueError
  for a negative chase (and the pool_ev refusals).
- `break_even_carryover(base_pool_yen, *, takeout=TAKEOUT, chase=0.0) -> float`: the carryover at
  which the chased pool EV is exactly 1: `t * P0 / (1 - t * k)`, `t * P0` without chase, and
  `math.inf` when `t * k >= 1` (no carryover ever lifts the EV to 1).

History (one row per WIN5 day; the real JRA source page is not pinned, see below):
- `WIN5_START = date(2011, 4, 24)`, the first WIN5 day.
- `parse_yen(text) -> int`: a printed yen amount, `312,450,000円` or with 億/万 units
  `2億6,716万3,960円`; ValueError for anything else (blank, no digits, a negative, junk).
- `parse_results_page(body: bytes) -> list[dict]`: a results page (SYNTHETIC shape, built by
  `results_page` below: one `<table class="win5_results">` whose header cells name the columns
  開催日, 発売金額, 前回からのキャリーオーバー, 的中票数, 払戻金, 次回へのキャリーオーバー, in any
  order, other columns ignored; dates `2011年4月24日`; amounts as `parse_yen`; 的中票数 as
  `1,234票`; 払戻金 `該当なし` when nobody hit). One dict per day with keys `race_date` (date),
  `pool_yen`, `carryover_in_yen`, `carryover_out_yen`, `winning_tickets` (int), `dividend_yen`
  (int per 100-yen ticket, None when nobody hit). ValueError for a page without the table, a
  missing column, or a cell that does not parse.
- `build_history(rows) -> pandas.DataFrame`: one row per race_date, ascending, from parsed rows of
  any number of pages; identical duplicates collapse, a date repeated with other figures raises
  ValueError. Columns: the six above plus `hit` (bool, winning_tickets > 0) and `pool_ev`
  (`pool_ev(pool_yen, carryover_in_yen)`).
- `history_counts(table) -> dict`, aggregates only: exactly `days`, `carryover_days` (carryover in
  > 0), `no_winner_days` (winning_tickets == 0), `first_day`, `last_day`.
- `missing_days(table, expected_days) -> list[date]`: the expected days with no row, sorted.
- `write_history(table) -> Path` writes `processed_dir('jra') / 'win5.parquet'`; `read_history()`
  reads it back.

Live:
- `live_pool_ev(rows, race_date, *, now: datetime) -> dict | None`: over odds-log rows
  (docs/specs/odds-log.md), the WIN5 `__pool__` rows of race_id `jra:<date>:WIN5` captured at or
  before `now`. None when there is none (not a WIN5 day, or not captured yet). Else
  `{race_date (ISO text), capture_utc (ISO UTC text), investment, carryover, pool_ev}`:
  `capture_utc` is the latest such row's snapshot_utc whatever figures it carries; `investment`
  is the figure of the latest such row carrying one, `carryover` likewise; `pool_ev` =
  `pool_ev(investment, carryover)`, None without an investment figure.
- The shadow feed (`board.shadow_feed.build_shadow`) carries it as the top-level key `win5`: that
  dict on a WIN5 day, None otherwise. The shadow page (`shadow.js` `renderShadow`) draws an element
  with a `data-win5` attribute whose text shows `pool_ev` with two decimals when the shadow feed's
  `win5` is not null, and no such element otherwise. The plain feed and board page (#40, #42) stay
  free of any expected value, so the figure lives on the shadow page and feed.

The study (D18 positive-edge family, pre-registered, development days only, D14): round 2 froze
the owner's spec (carryover bins as multiples of the break-even carryover, a day-block bootstrap)
and superseded the yen-edge `score_study` tests that lived here; its contract is pinned in
test_win5_study.py. The history table also carries `payout_yen` and `winning_combinations`
(test_win5_backfill.py, dead heats).

Every page, row and figure here is synthetic; HORSETRACK_DATA_DIR points at tmp_path.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import json
import math
import shutil
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.flags import FlagRules
from horsetrack.countries.japan.board.shadow_feed import build_shadow
from horsetrack.countries.japan.research import win5

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
POOL = odds_log.POOL_TOTAL
REPO = Path(__file__).resolve().parents[4]
WEB = REPO / "src" / "horsetrack" / "countries" / "japan" / "board" / "web"
NODE = shutil.which("node")
RULES = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def ev(pool: float, carry: float, takeout: float = 0.30) -> float:
    """The pool EV from first principles, independent of the code under test."""
    return (pool * (1 - takeout) + carry) / pool


# ---------------------------------------------------------------- pool arithmetic

POOL_CASES = [
    # (pool, carryover, takeout, expected)
    (1e9, 0.0, 0.30, 0.70),
    (1e9, 5e8, 0.30, 1.20),  # a carryover that lifts the pool above break-even
    (1e9, 3e8, 0.30, 1.00),  # exactly the break-even carryover
    (2e9, 3e8, 0.30, 0.85),
    (4.5e8, 1e8, 0.25, 0.75 + 1e8 / 4.5e8),
    (123_456_700, 987_654_300, 0.30, 0.70 + 987_654_300 / 123_456_700),
    (8e8, 2e8, 0.0, 1.25),
]


@pytest.mark.parametrize("pool, carry, takeout, want", POOL_CASES)
def test_pool_ev_is_the_pool_after_takeout_plus_the_carryover_per_yen(pool, carry, takeout, want):
    assert math.isclose(want, ev(pool, carry, takeout), rel_tol=1e-12)  # the table is right
    got = win5.pool_ev(pool, carry, takeout=takeout)
    assert math.isclose(got, want, rel_tol=1e-9), (pool, carry, takeout, got)


def test_pool_ev_defaults_to_the_win5_takeout_and_crosses_one_above_the_break_even():
    assert win5.TAKEOUT == pytest.approx(0.30)
    assert win5.pool_ev(1e9, 0) == pytest.approx(0.70)
    assert win5.pool_ev(1e9, 5e8) == pytest.approx(1.20)
    assert win5.pool_ev(1e9, 2.9e8) < 1.0 < win5.pool_ev(1e9, 3.1e8)


@pytest.mark.parametrize(
    "pool, carry, takeout",
    [
        (0, 1e8, 0.30),
        (-1e9, 1e8, 0.30),
        (None, 1e8, 0.30),
        (float("nan"), 1e8, 0.30),
        (float("inf"), 1e8, 0.30),
        (1e9, -1.0, 0.30),
        (1e9, None, 0.30),
        (1e9, float("nan"), 0.30),
        (1e9, 1e8, -0.1),
        (1e9, 1e8, 1.0),
        (1e9, 1e8, 1.5),
    ],
)
def test_pool_ev_refuses_a_missing_or_impossible_figure(pool, carry, takeout):
    with pytest.raises(ValueError):
        win5.pool_ev(pool, carry, takeout=takeout)


CHASE_CASES = [
    # (base pool, carryover, chase, takeout, expected per-ticket EV on P0 + k * C)
    (1e9, 5e8, 0.0, 0.30, 1.20),
    (1e9, 5e8, 1.0, 0.30, 0.70 + 5e8 / 1.5e9),  # the docstring's worked number, 1.0333
    (1e9, 5e8, 2.0, 0.30, 0.95),
    (2e9, 4e8, 0.5, 0.25, 0.75 + 4e8 / 2.2e9),
    (1e9, 0.0, 3.0, 0.30, 0.70),  # no carryover: nothing to chase
]


@pytest.mark.parametrize("base, carry, chase, takeout, want", CHASE_CASES)
def test_chased_pool_ev_is_the_pool_ev_on_the_pool_the_carryover_drew(
    base, carry, chase, takeout, want
):
    assert math.isclose(want, ev(base + chase * carry, carry, takeout), rel_tol=1e-12)
    got = win5.chased_pool_ev(base, carry, chase=chase, takeout=takeout)
    assert math.isclose(got, want, rel_tol=1e-9), (base, carry, chase, got)


def test_chasing_bettors_lower_the_per_ticket_ev_below_the_naive_figure():
    for base, carry in [(1e9, 5e8), (6e8, 1e8), (3e9, 2e9)]:
        naive = win5.pool_ev(base, carry)
        last = naive
        for chase in (0.25, 0.5, 1.0, 2.0, 4.0):
            got = win5.chased_pool_ev(base, carry, chase=chase)
            assert got < last, (base, carry, chase, got, last)  # more chase, lower EV
            last = got
        assert win5.chased_pool_ev(base, carry, chase=0.0) == pytest.approx(naive)
    with pytest.raises(ValueError):
        win5.chased_pool_ev(1e9, 5e8, chase=-0.5)


@pytest.mark.parametrize(
    "base, takeout, chase, want",
    [
        (1e9, 0.30, 0.0, 3e8),
        (2e9, 0.30, 0.0, 6e8),
        (1e9, 0.25, 0.0, 2.5e8),
        (1e9, 0.30, 1.0, 3e8 / 0.7),
        (1e9, 0.30, 2.0, 7.5e8),
        (1e9, 0.30, 10 / 3, math.inf),  # t * k == 1: the EV only tends to 1
        (1e9, 0.30, 5.0, math.inf),
    ],
)
def test_break_even_carryover_is_where_the_chased_ev_reaches_one(base, takeout, chase, want):
    got = win5.break_even_carryover(base, takeout=takeout, chase=chase)
    if math.isinf(want):
        assert math.isinf(got) and got > 0, got
        return
    assert math.isclose(got, want, rel_tol=1e-9), (base, takeout, chase, got)
    assert math.isclose(ev(base + chase * got, got, takeout), 1.0, rel_tol=1e-9)


def test_break_even_carryover_defaults_to_thirty_percent_of_the_pool_without_chase():
    for pool in (2.5e8, 1e9, 3.7e9):
        assert win5.break_even_carryover(pool) == pytest.approx(0.30 * pool)


# ---------------------------------------------------------------- parsing

YEN_CASES = [
    ("312,450,000円", 312_450_000),
    ("2億6,716万3,960円", 267_163_960),
    ("5億3,990万5,240円", 539_905_240),
    ("1億円", 100_000_000),
    ("5,000万円", 50_000_000),
    ("8億5,280万8,741円", 852_808_741),
    ("960円", 960),
    ("0円", 0),
    (" 1,234,560円 ", 1_234_560),
]


@pytest.mark.parametrize("text, want", YEN_CASES)
def test_parse_yen_reads_comma_and_oku_man_amounts(text, want):
    assert win5.parse_yen(text) == want


@pytest.mark.parametrize("text", ["", "円", "該当なし", "abc", "-5円", "1億x円", "12.5円"])
def test_parse_yen_refuses_what_is_not_an_amount(text):
    with pytest.raises(ValueError):
        win5.parse_yen(text)


COLUMNS = (
    "開催日",
    "発売金額",
    "前回からのキャリーオーバー",
    "的中票数",
    "払戻金",
    "次回へのキャリーオーバー",
)


def oku_man(yen: int) -> str:
    """`2億6,716万3,960円` style, as JRA prints large amounts."""
    oku, rest = divmod(yen, 100_000_000)
    man, one = divmod(rest, 10_000)
    out = (f"{oku}億" if oku else "") + (f"{man:,}万" if man else "") + (f"{one:,}" if one else "")
    return (out or "0") + "円"


def results_page(days: list[dict], *, columns=COLUMNS, extra=True, units=False) -> bytes:
    """A SYNTHETIC WIN5 results page (invented for these tests; not a JRA page): one table, a
    header row naming the columns in `columns` order, one row per day; `extra` adds a 対象レース
    column the parser must ignore; `units` prints amounts in 億/万 form."""
    money = oku_man if units else (lambda yen: f"{yen:,}円")

    def cell(day: dict, column: str) -> str:
        if column == "開催日":
            d = day["race_date"]
            return f"{d.year}年{d.month}月{d.day}日"
        if column == "発売金額":
            return money(day["pool_yen"])
        if column == "前回からのキャリーオーバー":
            return money(day["carryover_in_yen"])
        if column == "的中票数":
            return f"{day['winning_tickets']:,}票"
        if column == "払戻金":
            return "該当なし" if day["dividend_yen"] is None else money(day["dividend_yen"])
        if column == "次回へのキャリーオーバー":
            return money(day["carryover_out_yen"])
        raise AssertionError(column)

    heads = list(columns) + (["対象レース"] if extra else [])
    head = "".join(f"<th>{h}</th>" for h in heads)
    rows = ""
    for day in days:
        cells = [cell(day, c) for c in columns] + (["東京11R ほか"] if extra else [])
        rows += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    return (
        "<html><head><meta charset='utf-8'><title>SYNTHETIC WIN5 results</title></head><body>"
        "<!-- synthetic test page: every figure invented -->"
        f'<table class="win5_results"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'
        "</body></html>"
    ).encode()


def day_row(d, pool, carry_in, tickets, dividend, carry_out) -> dict:
    return {
        "race_date": d,
        "pool_yen": pool,
        "carryover_in_yen": carry_in,
        "winning_tickets": tickets,
        "dividend_yen": dividend,
        "carryover_out_yen": carry_out,
    }


# Invented WIN5 days, 2011 to 2013 (before any Premium takeout boost).
DAYS = [
    day_row(date(2011, 4, 24), 612_345_600, 0, 0, None, 428_641_920),
    day_row(date(2011, 5, 1), 1_050_000_000, 428_641_920, 3, 245_000_000, 0),
    day_row(date(2011, 5, 8), 734_120_000, 0, 1_234, 416_400, 0),
    day_row(date(2011, 5, 15), 801_500_300, 0, 0, None, 561_050_210),
    day_row(date(2012, 1, 8), 1_420_000_000, 561_050_210, 27, 55_588_400, 0),
    day_row(date(2013, 12, 22), 2_512_345_000, 0, 58, 30_321_000, 0),
]


@pytest.mark.parametrize("units", [False, True], ids=["commas", "oku-man"])
def test_results_page_gives_one_row_per_day_with_its_figures(units):
    got = win5.parse_results_page(results_page(DAYS, units=units))
    assert got == DAYS


def test_results_page_columns_are_read_by_their_header_not_their_place():
    shuffled = (
        "的中票数",
        "次回へのキャリーオーバー",
        "開催日",
        "払戻金",
        "発売金額",
        "前回からのキャリーオーバー",
    )
    days = DAYS[1:4]
    assert win5.parse_results_page(results_page(days, columns=shuffled, extra=False)) == days


@pytest.mark.parametrize(
    "body",
    [
        b"<html><body><p>Error</p></body></html>",
        results_page(DAYS[:2], columns=COLUMNS[:-1]),  # no carryover-out column
        results_page(DAYS[:2]).replace("2011年5月1日".encode(), b"2011/05/xx"),
        results_page(DAYS[:2]).replace("1,050,000,000円".encode(), "十億円".encode()),
    ],
    ids=["no-table", "missing-column", "bad-date", "bad-amount"],
)
def test_a_malformed_results_page_raises(body):
    with pytest.raises(ValueError):
        win5.parse_results_page(body)


# ---------------------------------------------------------------- the history table

HISTORY_COLUMNS = {
    "race_date",
    "pool_yen",
    "carryover_in_yen",
    "carryover_out_yen",
    "winning_tickets",
    "dividend_yen",
    "payout_yen",
    "winning_combinations",
    "hit",
    "pool_ev",
}


def as_date(value) -> date:
    """A table's race_date (date, datetime or pandas Timestamp) as a date."""
    return value.date() if isinstance(value, datetime) else value


def records(table) -> list[dict]:
    """The table's rows as dicts, race_date as a date and a missing dividend as None."""
    out = []
    for row in table.to_dict("records"):
        row["race_date"] = as_date(row["race_date"])
        div = row["dividend_yen"]
        row["dividend_yen"] = None if div is None or pd.isna(div) else int(div)
        out.append(row)
    return out


def test_history_has_one_row_per_win5_day_in_date_order_with_its_pool_ev():
    first, second = results_page(DAYS[3:] + DAYS[:2]), results_page(DAYS[1:4])  # overlapping
    rows = win5.parse_results_page(first) + win5.parse_results_page(second)

    table = win5.build_history(rows)

    assert HISTORY_COLUMNS <= set(table.columns)
    got = records(table)
    assert [r["race_date"] for r in got] == [d["race_date"] for d in DAYS]
    for row, want in zip(got, DAYS, strict=True):
        for key, value in want.items():
            assert row[key] == value, (want["race_date"], key, row[key])
        assert bool(row["hit"]) is (want["winning_tickets"] > 0), want["race_date"]
        expected = ev(want["pool_yen"], want["carryover_in_yen"])
        assert math.isclose(row["pool_ev"], expected, rel_tol=1e-9), want["race_date"]
    above = [r["race_date"] for r in got if r["pool_ev"] > 1]
    # 0.70 + 428.6M / 1,050M = 1.108 and 0.70 + 561.1M / 1,420M = 1.095; the rest at or below 0.70
    assert above == [date(2011, 5, 1), date(2012, 1, 8)]


def test_a_day_repeated_with_other_figures_is_refused():
    other = dict(DAYS[2], pool_yen=DAYS[2]["pool_yen"] + 100)
    with pytest.raises(ValueError, match="2011-05-08"):
        win5.build_history(DAYS + [other])


def test_history_counts_are_aggregates_of_the_table():
    table = win5.build_history(DAYS[::-1])
    counts = win5.history_counts(table)
    assert set(counts) == {"days", "carryover_days", "no_winner_days", "first_day", "last_day"}
    assert counts["days"] == len(DAYS)
    assert counts["carryover_days"] == sum(d["carryover_in_yen"] > 0 for d in DAYS) == 2
    assert counts["no_winner_days"] == sum(d["winning_tickets"] == 0 for d in DAYS) == 2
    assert counts["first_day"] == date(2011, 4, 24)
    assert counts["last_day"] == date(2013, 12, 22)

    fewer = win5.history_counts(win5.build_history(DAYS[2:5]))
    assert (fewer["days"], fewer["carryover_days"], fewer["no_winner_days"]) == (3, 1, 1)
    assert (fewer["first_day"], fewer["last_day"]) == (date(2011, 5, 8), date(2012, 1, 8))


def test_missing_days_names_the_expected_win5_days_without_a_row():
    table = win5.build_history([DAYS[0], DAYS[2], DAYS[4]])
    expected = [d["race_date"] for d in DAYS] + [date(2011, 6, 5)]
    assert win5.missing_days(table, reversed(expected)) == [
        date(2011, 5, 1),
        date(2011, 5, 15),
        date(2011, 6, 5),
        date(2013, 12, 22),
    ]
    assert win5.missing_days(win5.build_history(DAYS), [d["race_date"] for d in DAYS]) == []
    # a row the calendar does not expect is not a gap
    assert win5.missing_days(table, [date(2011, 5, 8)]) == []


def test_win5_started_on_2011_04_24():
    assert win5.WIN5_START == date(2011, 4, 24)


def test_history_is_written_under_the_data_dir_and_reads_back(data_dir):
    table = win5.build_history(DAYS)
    path = win5.write_history(table)
    assert Path(path) == paths.processed_dir("jra") / "win5.parquet"
    assert Path(path).is_file()
    back = win5.read_history()
    assert records(back) == records(table)


# ---------------------------------------------------------------- live

WIN5_DAY = date(2026, 11, 29)
WIN5_ID = f"jra:{WIN5_DAY.isoformat()}:WIN5"


def jst(hh: int, mm: int, day: date = WIN5_DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=JST)


def win5_row(at: datetime, investment, carryover, *, race_id: str = WIN5_ID) -> dict:
    return {
        "snapshot_utc": at.astimezone(UTC).isoformat(),
        "race_id": race_id,
        "pool": "WIN5",
        "combination": POOL,
        "odds": None,
        "status_raw": None,
        "investment": investment,
        "carryover": carryover,
        "legs": [f"jra:{WIN5_DAY.isoformat()}:TOK:{n:02d}" for n in (9, 10, 11)]
        + [f"jra:{WIN5_DAY.isoformat()}:KYO:{n:02d}" for n in (10, 11)],
    }


def win_row(at: datetime, odds: float, day: date = WIN5_DAY) -> dict:
    return {
        "snapshot_utc": at.astimezone(UTC).isoformat(),
        "race_id": f"jra:{day.isoformat()}:TOK:11",
        "pool": "WIN",
        "combination": "7",
        "odds": odds,
        "status_raw": "13時00分現在",
    }


LIVE_ROWS = [
    win5_row(jst(9, 30), 210_000_000.0, 312_450_000.0),
    win_row(jst(9, 30), 4.1),
    win5_row(jst(11, 0), 640_000_000.0, 312_450_000.0),
    win5_row(jst(11, 30), None, 312_450_000.0),  # a capture without the tickets-sold figure
    win5_row(jst(12, 0), 900_000_000.0, None, race_id="jra:2026-11-22:WIN5"),  # another day
    win5_row(jst(13, 0), 1_100_000_000.0, 312_450_000.0),
    win_row(jst(13, 0), 3.8),
]


@pytest.mark.parametrize(
    "now, investment, capture",
    [
        (jst(10, 0), 210_000_000.0, jst(9, 30)),
        (jst(11, 45), 640_000_000.0, jst(11, 30)),
        (jst(14, 0), 1_100_000_000.0, jst(13, 0)),
    ],
)
def test_live_pool_ev_is_the_latest_pool_and_carryover_as_of_now(now, investment, capture):
    got = win5.live_pool_ev(LIVE_ROWS[::-1], WIN5_DAY, now=now)
    assert got is not None
    assert got["race_date"] == WIN5_DAY.isoformat()
    assert got["investment"] == investment
    assert got["carryover"] == 312_450_000.0
    assert math.isclose(got["pool_ev"], ev(investment, 312_450_000.0), rel_tol=1e-9)
    assert datetime.fromisoformat(got["capture_utc"]) == capture


def test_live_pool_ev_is_none_off_a_win5_day_or_before_the_first_capture():
    assert win5.live_pool_ev(LIVE_ROWS, WIN5_DAY, now=jst(9, 0)) is None
    assert win5.live_pool_ev([win_row(jst(9, 30), 4.1)], WIN5_DAY, now=jst(14, 0)) is None
    assert win5.live_pool_ev(LIVE_ROWS, date(2026, 11, 28), now=jst(14, 0)) is None


def test_live_pool_ev_is_none_without_a_tickets_sold_figure():
    rows = [win5_row(jst(9, 30), None, 312_450_000.0)]
    got = win5.live_pool_ev(rows, WIN5_DAY, now=jst(10, 0))
    assert got is not None and got["carryover"] == 312_450_000.0
    assert got["investment"] is None and got["pool_ev"] is None


def write_log(venue: str, rows: list[dict], day: date) -> None:
    captures: dict[str, list[dict]] = {}
    for row in rows:
        captures.setdefault(row["snapshot_utc"], []).append(row)
    path = paths.odds_snapshot_path("jra", day.isoformat(), venue)
    for capture in captures.values():
        odds_log.append_capture(path, capture)


def test_the_shadow_feed_carries_the_win5_pool_ev_on_a_win5_day_only():
    write_log("TOK", LIVE_ROWS, WIN5_DAY)
    write_log("KYO", [win_row(jst(9, 30), 2.2)], WIN5_DAY)
    other = date(2026, 11, 28)
    write_log("TOK", [win_row(jst(9, 30, other), 5.0, other)], other)

    feed = build_shadow(WIN5_DAY, now=lambda: jst(11, 10), models=None, flag_rules=RULES)
    assert feed["win5"]["investment"] == 640_000_000.0
    assert feed["win5"]["carryover"] == 312_450_000.0
    assert math.isclose(feed["win5"]["pool_ev"], ev(640_000_000.0, 312_450_000.0), rel_tol=1e-9)
    assert all(not r["race_id"].endswith(":WIN5") for r in feed["races"])
    json.dumps(feed, allow_nan=False)  # still JSON-ready

    later = build_shadow(WIN5_DAY, now=lambda: jst(14, 0), models=None, flag_rules=RULES)
    assert later["win5"]["investment"] == 1_100_000_000.0

    plain_day = build_shadow(other, now=lambda: jst(14, 0, other), models=None, flag_rules=RULES)
    assert "win5" in plain_day and plain_day["win5"] is None


RENDER = r"""
const fs = require("fs");
const [, SHADOW_JS, IN, OUT] = process.argv;
const input = JSON.parse(fs.readFileSync(IN, "utf8"));
const shadow = require(SHADOW_JS);
const out = input.cases.map((c) => shadow.renderShadow(c.feed, c.shadow, { now: c.now }));
fs.writeFileSync(OUT, JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_shadow_page_shows_the_win5_pool_ev_only_when_the_feed_has_one(tmp_path):
    now = jst(11, 10)
    plain = {
        "race_date": WIN5_DAY.isoformat(),
        "generated_utc": now.astimezone(UTC).isoformat(),
        "venues": [
            {
                "venue": "TOK",
                "races": [
                    {
                        "race_no": 11,
                        "race_id": f"jra:{WIN5_DAY.isoformat()}:TOK:11",
                        "post_utc": jst(15, 40).astimezone(UTC).isoformat(),
                        "last_capture_utc": now.astimezone(UTC).isoformat(),
                        "pools": {
                            "WIN": {
                                "investment": 1_000_000,
                                "rows": [
                                    {
                                        "combination": "7",
                                        "odds": 4.1,
                                        "odds_hi": None,
                                        "status": "11時10分現在",
                                        "morning_odds": 4.1,
                                        "move_morning_pct": 0.0,
                                        "t20_odds": None,
                                        "move_t20_pct": None,
                                    }
                                ],
                            }
                        },
                    }
                ],
            }
        ],
    }

    def shadow_feed(win5_entry):
        return {
            "race_date": WIN5_DAY.isoformat(),
            "generated_utc": now.astimezone(UTC).isoformat(),
            "models": None,
            "races": [],
            "win5": win5_entry,
        }

    entry = {
        "race_date": WIN5_DAY.isoformat(),
        "capture_utc": now.astimezone(UTC).isoformat(),
        "investment": 1_000_000_000.0,
        "carryover": 500_000_000.0,
        "pool_ev": 1.2,
    }
    low = dict(entry, carryover=300_000_000.0, investment=2_000_000_000.0, pool_ev=0.85)
    ms = int(now.timestamp() * 1000)
    cases = [
        {"feed": plain, "shadow": shadow_feed(entry), "now": ms},
        {"feed": plain, "shadow": shadow_feed(low), "now": ms},
        {"feed": plain, "shadow": shadow_feed(None), "now": ms},
        {"feed": plain, "shadow": None, "now": ms},
    ]
    src, out = tmp_path / "in.json", tmp_path / "out.json"
    src.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    proc = subprocess.run(
        [NODE, "-e", RENDER, str(WEB / "shadow.js"), str(src), str(out)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    pages = [BeautifulSoup(html, "html.parser") for html in json.loads(out.read_text("utf-8"))]

    for page, want in ((pages[0], "1.20"), (pages[1], "0.85")):
        found = page.select("[data-win5]")
        assert len(found) == 1, page
        assert want in found[0].get_text(" "), found[0]
    assert pages[2].select("[data-win5]") == []
    assert pages[3].select("[data-win5]") == []
