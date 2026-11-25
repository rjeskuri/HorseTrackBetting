"""The WIN5 backfill's year-list cache and the carryover chain check (ticket #135, review fixes).

Contract pinned here, modules `horsetrack.countries.japan.research.win5` and `.win5_backfill`:

1. A year list cached while its year was still running is stale. `win5_backfill.backfill(*, years,
   today, fetcher, quiet, echo)` returns a tuple whose item 1 is the WIN5 days the year lists name
   before `today` (`expected`). A year's list (`/top/win5_results.html?year=YYYY`, page type
   `win5_year`, key `str(year)`, cached by `core.ingest.Fetcher` at
   `raw_page_path('jra', 'win5_year', year, str(year))`, whose file mtime is the fetch time the
   cache records) is:
   - fetched again when the cached copy was fetched before 1 January of the following year (JST),
     whatever `today` is, so the days added after that fetch are expected;
   - served from cache, with no request, when the cached copy was fetched after its year ended;
   - fetched again for the current year (`year >= today.year`), as before.
   The fetch times in these tests stay well clear of midnight in both UTC and JST.

2. `win5.carryover_chain_breaks(table) -> list[date]`: the days of a history table (as
   `build_history` gives it, rows in date order) that break the carryover chain, sorted, each day
   once:
   - a no-winner day (`winning_tickets == 0`) whose `carryover_out_yen` is not > 0;
   - a day whose `carryover_in_yen` differs from the previous row's `carryover_out_yen` (the
     previous WIN5 day in the table). The first row has no previous row and is not checked on this
     rule.
   A day with a winner and `carryover_out_yen > 0` is not a break by itself: JRA caps the WIN5
   payout (6億円 per 100 yen), and the rest carries over. An empty table has no break.
   `horsetrack japan win5-history` prints `carryover_breaks=<n>` in its summary, n the number of
   such days in the table it built (`carryover_breaks=0` and exit 0 for a consistent table).
   `history_counts` keeps its exact keys.

3. The netkeiba pages, in the SYNTHETIC shape the parsers read: the day page is
   `table.Win5_Table` cells, `th` label then `td` value (発売金額, 的中票数 `N票`, 払戻金 with `-円`
   for no hit, 先週までのキャリーオーバー, 次週へのキャリーオーバー); the year list links
   `win5.html?date=YYYYMMDD`.

4. Dead heats (round 2). When a leg dead-heats, netkeiba's day page lists several winning
   combinations, each its own `table.Win5_Table` of 的中馬番 / 払戻金 / 的中票数 rows (seen on a
   real 2019 page; the pages below copy that structure with invented figures). `parse_day_page`
   reads every combination:
   - `winning_tickets`: the total tickets over the combinations (0 on a no-hit page);
   - `payout_yen` (int): sum over the combinations of dividend x tickets, the yen the pool paid
     (0 on a no-hit page); one combination gives dividend x tickets;
   - `winning_combinations` (int): how many combinations have tickets (0 on a no-hit page);
   - `dividend_yen`: the LARGEST dividend per 100 yen over the combinations (one combination: its
     dividend; None when nobody hit).
   `build_history` keeps `payout_yen` and `winning_combinations` (int64 columns, through
   `write_history` / `read_history`); a row without them (e.g. from `parse_results_page`) gets
   `payout_yen = dividend_yen x winning_tickets` (0 without a hit) and `winning_combinations` 1
   with a hit, 0 without. A day given twice with another payout_yen is refused (ValueError).

Every page, row and figure here is synthetic; HORSETRACK_DATA_DIR points at tmp_path; the network
is an httpx.MockTransport behind the real ingest Fetcher.
"""

from __future__ import annotations

import gzip
import os
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.countries.japan.research import win5
from horsetrack.countries.japan.research import win5_backfill as wb

UTC = timezone.utc
HTTP: dict[str, Any] = {
    "user_agent": "horsetrack-tests/1.0",
    "timeout_s": 5,
    "max_retries": 0,
    "circuit_breaker": {"stop_on_status": [403, 429], "max_consecutive_failures": 50},
    "hosts": {"default": {"requests_per_second": 1000.0, "jitter": 0.0}},
}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


# ---------------------------------------------------------------- synthetic pages


def oku_man(yen: int) -> str:
    oku, rest = divmod(yen, 100_000_000)
    man, one = divmod(rest, 10_000)
    out = f"{oku}億" if oku else ""
    out += f"{man:,}万" if man else ""
    return out + (f"{one:,}円" if one or not out else "円")


def year_list(days: list[date], *, noise: bool = True) -> bytes:
    links = "".join(
        f'<li><a href="/top/win5.html?date={d:%Y%m%d}">{d:%m/%d}</a></li>' for d in days
    )
    extra = (
        '<a href="/top/win5_results.html?year=2019">2019</a>'
        '<a href="/race/result.html?race_id=202006010101">race</a>'
        if noise
        else ""
    )
    return f"<html><body><ul>{links}</ul>{extra}</body></html>".encode()


def day_page(cells: dict[str, str], *, split: bool = False) -> bytes:
    rows = [f"<tr><th> {label} </th><td>{value}</td></tr>" for label, value in cells.items()]
    if split:  # the labels spread over two tables of the class
        half = len(rows) // 2
        body = (
            f'<table class="Win5_Table">{"".join(rows[:half])}</table>'
            f'<div>other</div><table class="Win5_Table">{"".join(rows[half:])}</table>'
        )
    else:
        body = f'<table class="Win5_Table">{"".join(rows)}</table>'
    return f"<html><body><h1>WIN5</h1>{body}</body></html>".encode()


def day_cells(pool, carry_in, tickets, dividend, carry_out, *, units=False) -> dict[str, str]:
    yen = oku_man if units else (lambda v: f"{v:,}円")
    return {
        "発売金額": yen(pool),
        "先週までのキャリーオーバー": yen(carry_in),
        "的中票数": f"{tickets:,}票",
        "払戻金": "-円" if dividend is None else yen(dividend),
        "次週へのキャリーオーバー": yen(carry_out),
    }


def figures(d, pool, carry_in, tickets, dividend, carry_out) -> dict:
    """A one-combination day's parsed row (round 2 adds payout_yen and winning_combinations)."""
    return {
        "race_date": d,
        "pool_yen": pool,
        "carryover_in_yen": carry_in,
        "carryover_out_yen": carry_out,
        "winning_tickets": tickets,
        "dividend_yen": dividend,
        "payout_yen": 0 if dividend is None else dividend * tickets,
        "winning_combinations": 1 if tickets > 0 else 0,
    }


# ---------------------------------------------------------------- parse_day_page (synthetic)

D = date(2024, 3, 10)
DAY_CASES = [
    # pool, carry_in, tickets, dividend, carry_out
    (1_234_567_800, 0, 25, 3_456_780, 0),
    (987_654_300, 0, 0, None, 691_358_010),
    (2_512_340_000, 691_358_010, 3, 600_000_000, 120_000_000),  # capped winner
    (1_500_000_100, 120_000_000, 1_204, 87_650, 0),
]


@pytest.mark.parametrize("units", [False, True], ids=["commas", "oku-man"])
@pytest.mark.parametrize("case", DAY_CASES)
def test_synthetic_day_page_gives_the_days_figures(case, units):
    body = day_page(day_cells(*case, units=units))
    assert win5.parse_day_page(body, D) == figures(D, *case)


def test_synthetic_day_page_labels_are_found_across_tables_in_any_order():
    case = DAY_CASES[2]
    cells = dict(reversed(list(day_cells(*case).items())))
    assert win5.parse_day_page(day_page(cells, split=True), D) == figures(D, *case)


@pytest.mark.parametrize("drop", ["発売金額", "的中票数", "払戻金"])
def test_synthetic_day_page_without_the_result_raises(drop):
    cells = day_cells(*DAY_CASES[0])
    del cells[drop]
    with pytest.raises(ValueError):
        win5.parse_day_page(day_page(cells), D)


def test_synthetic_day_page_with_an_unreadable_amount_raises():
    cells = day_cells(*DAY_CASES[0])
    cells["発売金額"] = "集計中"
    with pytest.raises(ValueError):
        win5.parse_day_page(day_page(cells), D)


# ---------------------------------------------------------------- dead heats (round 2)


def combination_page(pool, carry_in, combos, carry_out, *, units=True) -> bytes:
    """A SYNTHETIC day page in the per-combination shape (invented figures): a carryover-in table
    when there is one, a sales table (発売票数 and 発売金額 in one row), one table per winning
    combination (的中馬番, 払戻金, 的中票数) and a carryover-out table when there is one.
    `combos`: (dividend per 100 yen or None, tickets) per combination, in page order."""
    yen = oku_man if units else (lambda v: f"{v:,}円")
    tables = []
    if carry_in:
        tables.append(f'<tr><th class="Win5TextSmall W136">先週までのキャリーオーバー</th>'
                      f"<td> {yen(carry_in)} </td></tr>")  # fmt: skip
    tables.append(f'<tr><th class="W136">発売票数</th><td>{pool // 100:,}票</td>'
                  f'<th class="W136">発売金額</th><td>{yen(pool)}</td></tr>')  # fmt: skip
    for n, (dividend, tickets) in enumerate(combos):
        horses = " ".join(str((n + k) % 16 + 1) for k in range(5))
        paid = "-円" if dividend is None else yen(dividend)
        tables.append(f"<tr><th>的中馬番</th><td> {horses} </td></tr>"
                      f"<tr><th>払戻金</th><td>{paid}</td></tr>"
                      f"<tr><th>的中票数</th><td>{tickets:,}票</td></tr>")  # fmt: skip
    if carry_out:
        tables.append(f"<tr><th> 次週への キャリーオーバー </th><td>{yen(carry_out)}</td></tr>")
    body = "".join(f'<table class="Win5_Table">{t}</table>' for t in tables)
    return f"<html><body><h1>WIN5 (synthetic)</h1>{body}</body></html>".encode()


# (pool, carry_in, [(dividend, tickets), ...], carry_out). The dead-heat days split the pool after
# takeout plus the carryover between the combinations, each dividend rounded down to 10 yen.
COMBO_CASES = {
    # 1.8e9 * 0.7 + 3e8 = 1.56e9, halves of 7.8e8: 7.8e8 / 120 = 6.5e6; 7.8e8 / 900 -> 866,660
    "two combinations, larger first": (1_800_000_000, 300_000_000,
                                       [(6_500_000, 120), (866_660, 900)], 0),
    "two combinations, smaller first": (1_800_000_000, 300_000_000,
                                        [(866_660, 900), (6_500_000, 120)], 0),
    # 2.4e9 * 0.7 = 1.68e9, thirds of 5.6e8: / 350, / 70, / 1,400
    "three combinations, largest in the middle": (2_400_000_000, 0,
                                                  [(1_600_000, 350), (8_000_000, 70),
                                                   (400_000, 1_400)], 0),
    "one combination": (1_234_567_800, 0, [(3_456_780, 25)], 0),
    "one combination after a carryover": (1_500_000_100, 120_000_000, [(87_650, 1_204)], 0),
    "no hit": (987_654_300, 0, [(None, 0)], 691_358_010),
    "no hit after a carryover": (900_000_000, 691_358_010, [(None, 0)], 1_321_358_010),
}  # fmt: skip


def combination_figures(d, pool, carry_in, combos, carry_out) -> dict:
    """The row the page must give, from the test's own sums."""
    hits = [(div, n) for div, n in combos if n > 0]
    return {
        "race_date": d,
        "pool_yen": pool,
        "carryover_in_yen": carry_in,
        "carryover_out_yen": carry_out,
        "winning_tickets": sum(n for _, n in hits),
        "dividend_yen": max(div for div, _ in hits) if hits else None,
        "payout_yen": sum(div * n for div, n in hits),
        "winning_combinations": len(hits),
    }


@pytest.mark.parametrize("units", [True, False], ids=["oku-man", "commas"])
@pytest.mark.parametrize("case", COMBO_CASES.values(), ids=COMBO_CASES.keys())
def test_a_day_page_reads_every_winning_combination(case, units):
    got = win5.parse_day_page(combination_page(*case, units=units), D)
    assert got == combination_figures(D, *case)


def test_a_dead_heat_day_pays_the_sum_of_its_combinations_not_the_first():
    pool, carry_in, combos, carry_out = COMBO_CASES["two combinations, larger first"]
    got = win5.parse_day_page(combination_page(pool, carry_in, combos, carry_out), D)
    assert got["winning_tickets"] == 1_020
    assert got["winning_combinations"] == 2
    assert got["payout_yen"] == 6_500_000 * 120 + 866_660 * 900 == 1_559_994_000
    assert got["dividend_yen"] == 6_500_000  # the largest, here first
    # the payouts add up to the pool after takeout plus the carryover, short only by the rounding
    # of each dividend down to 10 yen per 100-yen ticket
    net = pool * 7 // 10 + carry_in
    assert 0 <= net - got["payout_yen"] < 10 * got["winning_tickets"]

    flipped = win5.parse_day_page(combination_page(pool, carry_in, combos[::-1], carry_out), D)
    assert flipped == got  # the order of the combinations on the page does not matter


def test_history_keeps_the_payout_and_the_number_of_combinations(data_dir):
    days = [date(2019, 3, 3) + timedelta(weeks=i) for i in range(len(COMBO_CASES))]
    rows = [
        win5.parse_day_page(combination_page(*case), d)
        for d, case in zip(days, COMBO_CASES.values(), strict=True)
    ]
    table = win5.build_history(rows[::-1])
    assert {"payout_yen", "winning_combinations"} <= set(table.columns)
    win5.write_history(table)
    for frame in (table, win5.read_history()):
        got = frame.set_index(pd.to_datetime(frame["race_date"]).dt.date)
        for d, case in zip(days, COMBO_CASES.values(), strict=True):
            want = combination_figures(d, *case)
            assert int(got.loc[d, "payout_yen"]) == want["payout_yen"], d
            assert int(got.loc[d, "winning_combinations"]) == want["winning_combinations"], d
            assert int(got.loc[d, "winning_tickets"]) == want["winning_tickets"], d


def test_history_derives_the_payout_of_a_row_without_one():
    """Rows in the older six-figure shape (parse_results_page): one combination per hit day."""
    six = [
        {k: v for k, v in figures(DAYS[i], *case).items()
         if k not in ("payout_yen", "winning_combinations")}
        for i, case in enumerate(DAY_CASES)
    ]  # fmt: skip
    table = win5.build_history(six)
    for row, case in zip(table.to_dict("records"), DAY_CASES, strict=True):
        _, _, tickets, dividend, _ = case
        assert int(row["payout_yen"]) == (0 if dividend is None else dividend * tickets), case
        assert int(row["winning_combinations"]) == (1 if tickets else 0), case


def test_a_day_given_twice_with_another_payout_is_refused():
    pool, carry_in, combos, carry_out = COMBO_CASES["two combinations, larger first"]
    full = win5.parse_day_page(combination_page(pool, carry_in, combos, carry_out), D)
    first_only = dict(full, payout_yen=6_500_000 * 120)
    with pytest.raises(ValueError, match=D.isoformat()):
        win5.build_history([full, first_only])


# ---------------------------------------------------------------- parse_year_list (synthetic)


@pytest.mark.parametrize(
    "days",
    [
        [],
        [date(2011, 4, 24)],
        [date(2020, 1, 5), date(2020, 1, 12), date(2020, 12, 27)],
        [date(2026, 11, 22), date(2026, 1, 4), date(2026, 6, 7)],
    ],
)
def test_synthetic_year_list_names_its_win5_days_sorted(days):
    assert win5.parse_year_list(year_list(days)) == sorted(days)


def test_synthetic_year_list_counts_a_repeated_link_once_and_reads_extra_query_fields():
    body = (
        b'<a href="/top/win5.html?date=20250105">a</a>'
        b'<a href="https://race.netkeiba.com/top/win5.html?date=20250105">b</a>'
        b'<a href="../top/win5.html?date=20250112&amp;rf=list">c</a>'
        b'<a href="/top/win5.html?date=2025011">short</a>'
    )
    assert win5.parse_year_list(body) == [date(2025, 1, 5), date(2025, 1, 12)]


# ---------------------------------------------------------------- carryover_chain_breaks

A, B, C = 812_345_670, 1_034_000_000, 120_000_000
DAYS = [date(2024, 1, 7) + timedelta(weeks=i) for i in range(6)]


def table_of(*rows):
    """rows: (tickets, carry_in, carry_out) per consecutive WIN5 day of DAYS."""
    out = []
    for d, (tickets, carry_in, carry_out) in zip(DAYS, rows, strict=False):
        dividend = None if tickets == 0 else 1_000_000
        out.append(figures(d, 1_500_000_000, carry_in, tickets, dividend, carry_out))
    return win5.build_history(out)


CHAIN_CASES = {
    "empty": ((), []),
    "one hit day": (((10, 0, 0),), []),
    "one no-winner day with no carryover out": (((0, 0, 0),), [DAYS[0]]),
    "clean chain": (((5, 0, 0), (0, 0, A), (0, A, B), (2, B, 0), (9, 0, 0)), []),
    "first row starts mid-chain": (((0, A, B), (1, B, 0)), []),
    "capped winner carries over": (((0, 0, A), (3, A, C), (7, C, 0)), []),
    "capped winner's carryover lost": (((3, 0, C), (7, 0, 0)), [DAYS[1]]),
    "carryover in differs from the day before's out": (
        ((0, 0, A), (0, A - 1, B), (4, B, 0)),
        [DAYS[1]],
    ),
    "carryover in appears from nowhere": (((4, 0, 0), (6, C, 0)), [DAYS[1]]),
    "all carryovers zero (a label mismatch)": (
        ((5, 0, 0), (0, 0, 0), (0, 0, 0), (8, 0, 0)),
        [DAYS[1], DAYS[2]],
    ),
    "a day breaking both rules is named once": (((0, 0, A), (0, 0, 0), (2, 0, 0)), [DAYS[1]]),
    "several breaks, sorted": (
        ((0, 0, A), (1, B, 0), (0, 0, C), (5, 0, 0), (0, 0, 0)),
        [DAYS[1], DAYS[3], DAYS[4]],
    ),
}


@pytest.mark.parametrize("rows, want", CHAIN_CASES.values(), ids=CHAIN_CASES.keys())
def test_carryover_chain_breaks_names_the_days_that_break_the_chain(rows, want):
    assert win5.carryover_chain_breaks(table_of(*rows)) == want


def test_carryover_chain_breaks_reads_the_table_as_written_and_read_back():
    table = table_of((0, 0, A), (0, A, B), (1, B + 5, 0), (0, 0, 0))
    win5.write_history(table)
    assert win5.carryover_chain_breaks(win5.read_history()) == [DAYS[2], DAYS[3]]


def test_carryover_labels_the_parser_does_not_know_never_give_a_clean_table():
    """A page whose carryover labels changed: either the parser refuses it, or the chain check
    flags the no-winner day that would otherwise show no carryover."""
    days = DAYS[:3]
    cases = [(1_200_000_000, 0, 4, 2_000_000, 0), (900_000_000, 0, 0, None, A),
             (1_400_000_000, A, 2, 9_000_000, 0)]  # fmt: skip
    rename = {"先週までのキャリーオーバー": "キャリーオーバー(前回)",
              "次週へのキャリーオーバー": "キャリーオーバー(次回)"}  # fmt: skip
    rows = []
    try:
        for d, case in zip(days, cases, strict=True):
            cells = {rename.get(k, k): v for k, v in day_cells(*case).items()}
            rows.append(win5.parse_day_page(day_page(cells), d))
    except ValueError:
        return
    assert days[1] in win5.carryover_chain_breaks(win5.build_history(rows))


# ---------------------------------------------------------------- the backfill: year-list cache


class Site:
    """netkeiba double: year lists and day pages by query, every request counted by URL kind."""

    def __init__(self, lists: dict[int, list[date]]):
        self.lists = lists
        self.year_hits: Counter[int] = Counter()

    def handler(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if request.url.path.endswith("win5_results.html") and "year" in params:
            year = int(params["year"])
            self.year_hits[year] += 1
            return httpx.Response(200, content=year_list(self.lists.get(year, [])))
        if request.url.path.endswith("win5.html") and "date" in params:
            return httpx.Response(200, content=day_page(day_cells(1_000_000_000, 0, 3, 5, 0)))
        return httpx.Response(404, content=b"not found")

    def fetcher(self) -> Fetcher:
        return Fetcher("jra", http=HTTP, transport=httpx.MockTransport(self.handler),
                       sleep=lambda s: None)  # fmt: skip


def cache_year_list(year: int, days: list[date], fetched_utc: datetime) -> Path:
    """A year list in the ingest cache as the Fetcher writes it, fetched at `fetched_utc`."""
    path = paths.raw_page_path("jra", wb.YEAR_PAGE, year, str(year))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(year_list(days)))
    stamp = fetched_utc.timestamp()
    os.utime(path, (stamp, stamp))
    return path


def run_backfill(site: Site, years: range, today: date) -> list[date]:
    with site.fetcher() as fetcher:
        out = wb.backfill(years=years, today=today, fetcher=fetcher, quiet=lambda: None,
                          echo=lambda s: None)  # fmt: skip
    return sorted(out[1])


def sundays(year: int, month_days: list[tuple[int, int]]) -> list[date]:
    return [date(year, m, d) for m, d in month_days]


EARLY = {
    2025: sundays(2025, [(1, 5), (6, 1), (11, 23)]),
    2026: sundays(2026, [(1, 4), (6, 7), (11, 22)]),
}
LATE = {  # what the site lists once the year is over
    2025: EARLY[2025] + sundays(2025, [(11, 30), (12, 28)]),
    2026: EARLY[2026] + sundays(2026, [(11, 29), (12, 6), (12, 27)]),
}

STALE_CASES = {
    # year, cached at (UTC), today -> refetched?
    "fetched mid-year, read the next January": (2026, datetime(2026, 11, 25, 3, tzinfo=UTC),
                                                date(2027, 1, 10), True),
    "fetched mid-year, read years later": (2025, datetime(2025, 6, 2, 3, tzinfo=UTC),
                                           date(2027, 3, 1), True),
    "fetched on 31 December at noon JST": (2026, datetime(2026, 12, 31, 3, tzinfo=UTC),
                                           date(2027, 2, 1), True),
    "fetched on 1 January at 15:00 JST": (2026, datetime(2027, 1, 1, 6, tzinfo=UTC),
                                          date(2027, 2, 1), False),
    "fetched weeks after the year ended": (2025, datetime(2026, 2, 3, 3, tzinfo=UTC),
                                           date(2027, 3, 1), False),
}  # fmt: skip


@pytest.mark.parametrize(
    "year, cached_at, today, refetched", STALE_CASES.values(), ids=STALE_CASES.keys()
)
def test_a_year_list_cached_before_its_year_ended_is_fetched_again(
    year, cached_at, today, refetched
):
    cache_year_list(year, EARLY[year], cached_at)
    site = Site({year: LATE[year]})
    expected = run_backfill(site, range(year, year + 1), today)
    want = LATE[year] if refetched else EARLY[year]
    assert expected == want
    assert site.year_hits[year] == (1 if refetched else 0)


def test_a_stale_list_and_a_settled_one_in_the_same_run():
    """2025 settled (fetched in 2026), 2026 stale (fetched in November 2026), 2027 current: only
    the stale and the current lists go to the site, and every day before today is expected."""
    today = date(2027, 1, 10)
    cache_year_list(2025, EARLY[2025], datetime(2026, 1, 20, 3, tzinfo=UTC))
    cache_year_list(2026, EARLY[2026], datetime(2026, 11, 25, 3, tzinfo=UTC))
    jan = sundays(2027, [(1, 4), (1, 9), (1, 18)])
    cache_year_list(2027, jan[:1], datetime(2027, 1, 5, 3, tzinfo=UTC))
    site = Site({2025: LATE[2025], 2026: LATE[2026], 2027: jan})
    expected = run_backfill(site, range(2025, 2028), today)
    assert expected == EARLY[2025] + LATE[2026] + jan[:2]
    assert site.year_hits == Counter({2026: 1, 2027: 1})


def test_the_backfill_expects_the_late_december_days_after_a_stale_list_is_refetched():
    """The review's case: the 2026 list cached on 2026-11-25; on 2027-01-10 the backfill must ask
    for 2026-11-29 .. 2026-12-27 and the summary must not report them as present."""
    cache_year_list(2026, EARLY[2026], datetime(2026, 11, 25, 3, tzinfo=UTC))
    site = Site({2026: LATE[2026]})
    expected = run_backfill(site, range(2026, 2027), date(2027, 1, 10))
    late = [d for d in LATE[2026] if d > date(2026, 11, 25)]
    assert set(late) <= set(expected)


# ---------------------------------------------------------------- the command's summary

CLI_DAYS = [date(2020, 1, 5), date(2020, 1, 12), date(2020, 1, 19), date(2020, 1, 26)]
CLI_CASES = {
    # per day: (tickets, carry_in, carry_out) -> breaks
    "consistent": (((4, 0, 0), (0, 0, A), (0, A, B), (2, B, 0)), 0),
    "no-winner day with no carryover out": (((4, 0, 0), (0, 0, 0), (0, 0, B), (2, B, 0)), 1),
    "carryover in differs": (((4, 0, 0), (0, 0, A), (0, C, B), (2, A, 0)), 2),
}


@pytest.mark.parametrize("chain, breaks", CLI_CASES.values(), ids=CLI_CASES.keys())
def test_the_command_prints_the_number_of_carryover_chain_breaks(monkeypatch, chain, breaks):
    pages = {}
    for d, (tickets, carry_in, carry_out) in zip(CLI_DAYS, chain, strict=True):
        dividend = None if tickets == 0 else 2_500_000
        pages[f"{d:%Y%m%d}"] = day_page(day_cells(1_300_000_000, carry_in, tickets, dividend,
                                                  carry_out))  # fmt: skip

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "year" in params:
            return httpx.Response(200, content=year_list(CLI_DAYS))
        if params.get("date") in pages:
            return httpx.Response(200, content=pages[params["date"]])
        return httpx.Response(404, content=b"not found")

    def make_fetcher(*a, **k):
        return Fetcher("jra", http=HTTP, transport=httpx.MockTransport(handler),
                       sleep=lambda s: None)  # fmt: skip

    monkeypatch.setattr(wb, "make_fetcher", make_fetcher)
    monkeypatch.setattr(wb, "_quiet_check", lambda now: lambda: None)
    result = CliRunner().invoke(
        app, ["japan", "win5-history", "--start-year", "2020", "--end-year", "2020"]
    )
    assert "missing_days=0" in result.output, result.output
    assert f"carryover_breaks={breaks}" in result.output, result.output
    if breaks == 0:
        assert result.exit_code == 0, result.output
