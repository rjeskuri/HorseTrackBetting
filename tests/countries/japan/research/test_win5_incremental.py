"""Growing the WIN5 carryover sample: the incremental history append (ticket #325).

Contract pinned here, modules `horsetrack.countries.japan.research.win5` and `.win5_backfill`:

1. `win5.append_history(existing, rows) -> (table, added)`. `existing` is a history table (as
   `build_history` / `read_history` give it) or None; `rows` are parsed rows (`parse_day_page`
   shape). The merged table is `build_history` over the existing rows plus the new ones: one row
   per race_date, ascending, every existing day kept whether or not this call has a row for it. A
   day present in both with identical figures is not added again; a day with other figures raises
   ValueError naming the day (as `build_history` does). `added` is the list of `date`s that were
   not in `existing`, ascending. `existing` itself is not changed.

2. `win5.bin_counts(table, spec) -> {"bins": [{"lo_ratio", "hi_ratio", "n_days"}, ...],
   "holdout_days_excluded": int}`: the development days per carryover bin, binned exactly as
   `score_study` bins them (ratio = carryover_in / (break_even_takeout x net pool), edges from the
   spec; a day with no carryover is in no bin), and the number of holdout days skipped
   (`is_holdout_day("japan", day)`, looked up when called). A holdout day is counted and skipped
   before any outcome field (payout, tickets, dividend, carryover out) is read. ValueError for a
   spec `score_study` cannot score.

3. `horsetrack japan win5-history` appends to the table already in the data dir instead of
   rebuilding it: the days in the table that this run did not read stay; the days the run reads
   that are new are added; a conflicting figure for a day in the table stops the run (non-zero
   exit, table untouched). Holdout days are not fetched and not added. Besides its existing
   lines it prints `added_days=<n>` (the days new to the table), one `bin_<lo>_<hi>=<n_days>` token
   per bin of config/win5_study.yaml (`bin_0_0.5`, `bin_0.5_1`, `bin_1_2`, `bin_2_inf`; the number
   formatted as in the yaml) counting the whole table's development days, and
   `holdout_days=<n>`, the holdout days found in the table (not counted in any bin).

Every page, row and figure is synthetic; HORSETRACK_DATA_DIR points at tmp_path; the network is an
httpx.MockTransport behind the real ingest Fetcher.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
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
SPEC = {
    "family_id": "F-test-win5-325",
    "coverage": "pool-level",
    "pool": "net-of-refunds",
    "comparator": 1.0,
    "ratio_edges": [0, 0.5, 1, 2],
    "break_even_takeout": 0.30,
    "one_sided_level": 0.95,
    "payout_cap_yen": 600_000_000,
    "seed": 20261125,
    "n_boot": 200,
}
HOLDOUT_A, HOLDOUT_B = date(2026, 11, 29), date(2026, 12, 1)  # holdout days (test_win5_study)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def sundays(n: int, start: date = date(2016, 1, 3)) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def row(d, pool, carry_in, tickets, payout, carry_out) -> dict:
    """A parsed row with every figure given."""
    return {
        "race_date": d,
        "pool_yen": pool,
        "carryover_in_yen": carry_in,
        "carryover_out_yen": carry_out,
        "winning_tickets": tickets,
        "dividend_yen": payout // tickets // 10 * 10 if tickets else None,
        "payout_yen": payout,
        "winning_combinations": 1 if tickets else 0,
    }


TAKEOUT_PCT = {date(2024, 9, 16): 15}  # an all-pool boosted day; every other day from
GROSS_END = date(2014, 6, 7)  # 2014-06-07 is 30%, before it the pool is the gross pool_yen


def day(d, net, carry_in, value) -> dict:
    """A day whose pool-level return is `value` on a net pool of `net` yen: the money adds up at
    the day's own takeout (carry out = carry in + net x (1 - t) - payout)."""
    payout = round(value * net)
    tickets = 7 if payout else 0
    if d < GROSS_END:
        return row(d, net, carry_in, tickets, payout, max(0, carry_in + net * 7 // 10 - payout))
    t = TAKEOUT_PCT.get(d, 30)
    carry_out = carry_in + net * (100 - t) // 100 - payout
    assert carry_out >= 0
    return row(d, net * 103 // 100, carry_in, tickets, payout, carry_out)


def expected_bin(carry_in, net, edges, break_even) -> int | None:
    """The bin of a day from the spec's definition, written out independently of the code."""
    if carry_in <= 0:
        return None
    ratio = carry_in / (break_even * net)
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else float("inf")
        if (ratio > lo if i == 0 else ratio >= lo) and ratio < hi:
            return i
    return None


# ---------------------------------------------------------------- append_history


def old_new_rows():
    days = sundays(6)
    rows = [day(d, 1_000_000_000, c, v) for d, (c, v) in zip(days, [
        (0, 0.7), (75_000_000, 0.0), (225_000_000, 0.0), (450_000_000, 0.9),
        (0, 0.6), (900_000_000, 0.0)], strict=True)]  # fmt: skip
    return rows[:4], rows[4:]


def test_append_to_no_table_is_the_history_of_the_rows():
    old, new = old_new_rows()
    rows = (old + new)[::-1]  # any order
    table, added = win5.append_history(None, rows)
    pd.testing.assert_frame_equal(table, win5.build_history(rows))
    assert added == sorted(r["race_date"] for r in rows)
    assert all(type(d) is date for d in added)
    empty, none_added = win5.append_history(None, [])
    assert len(empty) == 0 and none_added == []
    assert list(empty.columns) == list(win5.COLUMNS)


def test_append_adds_only_the_new_days_and_keeps_every_old_row():
    old, new = old_new_rows()
    existing = win5.build_history(old)
    before = existing.copy()
    # the run read one old day again (identical) and the two new ones, in another order
    table, added = win5.append_history(existing, [new[1], old[2], new[0]])
    assert added == [new[0]["race_date"], new[1]["race_date"]]
    pd.testing.assert_frame_equal(table, win5.build_history(old + new))
    assert list(table["race_date"]) == sorted(r["race_date"] for r in old + new)
    pd.testing.assert_frame_equal(existing, before)  # the argument is left alone


def test_append_keeps_old_days_when_the_run_read_none_of_them(data_dir):
    old, new = old_new_rows()
    win5.write_history(win5.build_history(old))
    table, added = win5.append_history(win5.read_history(), new)  # as the command reads it
    assert added == [r["race_date"] for r in new]
    pd.testing.assert_frame_equal(table, win5.build_history(old + new), check_dtype=False)
    assert [d for d in table["race_date"]][:4] == [r["race_date"] for r in old]


@pytest.mark.parametrize("which", ["nothing", "old rows again", "a day repeated"])
def test_a_run_with_nothing_new_adds_nothing_and_changes_nothing(which):
    old, _ = old_new_rows()
    existing = win5.build_history(old)
    rows = {"nothing": [], "old rows again": old[::-1], "a day repeated": [old[1], dict(old[1])]}
    table, added = win5.append_history(existing, rows[which])
    assert added == []
    pd.testing.assert_frame_equal(table, existing)


def test_a_day_repeated_in_the_new_rows_is_added_once():
    old, new = old_new_rows()
    table, added = win5.append_history(win5.build_history(old), [new[0], dict(new[0])])
    assert added == [new[0]["race_date"]]
    assert len(table) == len(old) + 1


@pytest.mark.parametrize(
    "change",
    [{"pool_yen": 1}, {"carryover_in_yen": 1}, {"carryover_out_yen": 1}, {"winning_tickets": 99},
     {"payout_yen": 1}, {"winning_combinations": 2}],
)  # fmt: skip
def test_a_conflicting_figure_for_a_day_in_the_table_raises(change):
    old, new = old_new_rows()
    existing = win5.build_history(old)
    target = old[1]
    clash = {**target, **{k: target[k] + v for k, v in change.items()}}
    with pytest.raises(ValueError, match=target["race_date"].isoformat()):
        win5.append_history(existing, [new[0], clash])


# ---------------------------------------------------------------- bin_counts

# (date, net pool, carryover in, value). The same carryover lands in another bin on another pool.
CASES = [
    (sundays(16)[0], 1_000_000_000, 0, 0.7),  # no carryover: in no bin
    (sundays(16)[1], 1_000_000_000, 75_000_000, 0.0),  # ratio 0.25
    (sundays(16)[2], 1_000_000_000, 225_000_000, 0.9),  # 0.75
    (sundays(16)[3], 1_000_000_000, 450_000_000, 0.0),  # 1.5
    (sundays(16)[4], 1_000_000_000, 900_000_000, 1.1),  # 3.0
    (sundays(16)[5], 2_000_000_000, 450_000_000, 0.0),  # 0.75: same carryover, a bigger pool
    (sundays(16)[6], 400_000_000, 450_000_000, 0.5),  # 3.75: the same, a smaller pool
    (sundays(16)[7], 6_000_000_000, 450_000_000, 0.0),  # 0.25
    (sundays(16)[8], 2_500_000_000, 450_000_000, 0.8),  # 0.6
    (sundays(16)[9], 1_000_000_000, 30_000_000, 0.6),  # 0.1
    (sundays(16)[10], 1_000_000_000, 0, 0.0),  # no carryover, no hit
    (date(2024, 9, 16), 1_000_000_000, 450_000_000, 0.9),  # boosted takeout: bin by 30% still
    (date(2012, 1, 8), 1_000_000_000, 225_000_000, 0.0),  # gross pool (before 2014-06-07)
    (date(2012, 1, 15), 500_000_000, 225_000_000, 0.0),  # gross pool, 1.5
]
# no day sits on a bin edge: the implied pool is a float, so an edge day would be a coin toss


def table_of(cases) -> pd.DataFrame:
    return win5.build_history([day(*c) for c in cases])


def want_counts(cases, edges=(0, 0.5, 1, 2), break_even=0.30) -> list[int]:
    out = [0] * len(edges)
    for _day, net, carry, _ in cases:
        i = expected_bin(carry, net, list(edges), break_even)
        if i is not None:
            out[i] += 1
    return out


def test_bin_counts_match_the_study_binning_day_for_day():
    got = win5.bin_counts(table_of(CASES), SPEC)
    assert [(b["lo_ratio"], b["hi_ratio"]) for b in got["bins"]] == [(0, 0.5), (0.5, 1), (1, 2),
                                                                      (2, None)]  # fmt: skip
    counts = [b["n_days"] for b in got["bins"]]
    assert counts == want_counts(CASES)
    assert all(n > 0 for n in counts)  # every bin is exercised
    assert got["holdout_days_excluded"] == 0


def test_bin_counts_agree_with_score_study(tmp_path):
    registry = Registry(tmp_path / "r.jsonl", holdout_log=tmp_path / "h.jsonl")
    registry.freeze_family(SPEC["family_id"], SPEC)
    table = table_of(CASES)
    scored = win5.score_study(table, SPEC, registry=registry, family_id=SPEC["family_id"])
    got = win5.bin_counts(table, SPEC)
    assert [b["n_days"] for b in got["bins"]] == [b["n_days"] for b in scored["bins"]]
    assert got["holdout_days_excluded"] == scored["holdout_days_excluded"]


def test_bin_counts_follow_the_spec_not_a_fixed_binning():
    spec = dict(SPEC, ratio_edges=[0, 1, 4], break_even_takeout=0.25)
    got = win5.bin_counts(table_of(CASES), spec)
    assert [(b["lo_ratio"], b["hi_ratio"]) for b in got["bins"]] == [(0, 1), (1, 4), (4, None)]
    assert [b["n_days"] for b in got["bins"]] == want_counts(CASES, (0, 1, 4), 0.25)
    assert [b["n_days"] for b in got["bins"]] != want_counts(CASES, (0, 1, 4), 0.30)


def test_bin_counts_of_an_empty_table_and_one_without_carryover():
    empty = win5.bin_counts(win5.build_history([]), SPEC)
    assert [b["n_days"] for b in empty["bins"]] == [0, 0, 0, 0]
    plain = win5.bin_counts(table_of([c for c in CASES if c[2] == 0]), SPEC)
    assert [b["n_days"] for b in plain["bins"]] == [0, 0, 0, 0]


def test_bin_counts_refuse_a_spec_the_scorer_cannot_score():
    bad = {k: v for k, v in SPEC.items() if k != "ratio_edges"}
    with pytest.raises(ValueError):
        win5.bin_counts(table_of(CASES), bad)


def hostile(d, carry_in=450_000_000) -> dict:
    """A holdout day whose outcome figures cannot be read: `net_pool` and `day_value` raise on
    them (negative payout and carryover out) and a ratio from them would be nonsense."""
    return {**row(d, 1_000_000_000, carry_in, 3, -5, -7), "dividend_yen": 100}


def test_holdout_days_are_counted_and_skipped_before_any_outcome_is_read():
    assert is_holdout_day("japan", HOLDOUT_A) and is_holdout_day("japan", HOLDOUT_B)
    bad = hostile(HOLDOUT_A)
    with pytest.raises(ValueError):  # the figures are unreadable: reading them is an error
        win5.net_pool(bad)
    rows = [day(*c) for c in CASES] + [bad, hostile(HOLDOUT_B, carry_in=75_000_000)]
    got = win5.bin_counts(win5.build_history(rows), SPEC)
    assert got["holdout_days_excluded"] == 2
    assert [b["n_days"] for b in got["bins"]] == want_counts(CASES)  # the 2 days are in no bin


def test_bin_counts_ask_is_holdout_day_and_skip_what_it_names(monkeypatch):
    marked = {CASES[1][0], CASES[3][0]}  # development days by the real definition
    assert not any(is_holdout_day("japan", d) for d in marked)
    monkeypatch.setattr(win5, "is_holdout_day", lambda country, d: d in marked)
    rows = [day(*c) for c in CASES]
    for r in rows:
        if r["race_date"] in marked:  # outcomes of a marked day must not be read
            r.update(payout_yen=-5, carryover_out_yen=-7, winning_tickets=3, dividend_yen=100)
    got = win5.bin_counts(win5.build_history(rows), SPEC)
    assert got["holdout_days_excluded"] == 2
    kept = [c for c in CASES if c[0] not in marked]
    assert [b["n_days"] for b in got["bins"]] == want_counts(kept)
    assert [b["n_days"] for b in got["bins"]] != want_counts(CASES)


# ---------------------------------------------------------------- the command


def oku_free(v: int) -> str:
    return f"{v:,}円"


def day_page(carry_in, tickets, dividend, carry_out, pool=1_030_000_000) -> bytes:
    cells = {
        "発売金額": oku_free(pool),
        "先週までのキャリーオーバー": oku_free(carry_in),
        "的中票数": f"{tickets:,}票",
        "払戻金": "-円" if dividend is None else oku_free(dividend),
        "次週へのキャリーオーバー": oku_free(carry_out),
    }
    rows = "".join(f"<tr><th> {k} </th><td>{v}</td></tr>" for k, v in cells.items())
    return f'<html><body><table class="Win5_Table">{rows}</table></body></html>'.encode()


def year_list(days: list[date]) -> bytes:
    links = "".join(f'<li><a href="/top/win5.html?date={d:%Y%m%d}">x</a></li>' for d in days)
    return f"<html><body><ul>{links}</ul></body></html>".encode()


# 2020 days on a 1e9 net pool at 30%: (carry in, hit?) -> bin by carry_in / 3e8
SITE_DAYS = [
    (date(2020, 1, 5), 0, True),  # no carryover
    (date(2020, 1, 12), 75_000_000, False),  # bin 0
    (date(2020, 1, 19), 225_000_000, False),  # bin 1
    (date(2020, 1, 26), 450_000_000, True),  # bin 2
    (date(2020, 2, 2), 900_000_000, False),  # bin 3
    (date(2020, 2, 9), 450_000_000, False),  # bin 2
]
NET = 1_000_000_000


def site_figures(carry_in: int, hit: bool) -> tuple[int, int | None, int]:
    """(tickets, dividend, carry out) of a day on a 1e9 net pool; a hit pays 6e8 in all."""
    if hit:
        return 10, 60_000_000, carry_in + NET * 7 // 10 - 600_000_000
    return 0, None, carry_in + NET * 7 // 10


def site_row(d, carry_in, hit) -> dict:
    tickets, dividend, carry_out = site_figures(carry_in, hit)
    return row(d, 1_030_000_000, carry_in, tickets, 600_000_000 if hit else 0, carry_out)


class Site:
    """netkeiba double: the year list names `listed`; day pages by date, requests recorded."""

    def __init__(self, listed: list[date], figures=SITE_DAYS):
        self.listed = listed
        self.pages = {}
        for d, carry_in, hit in figures:
            tickets, dividend, carry_out = site_figures(carry_in, hit)
            self.pages[d] = day_page(carry_in, tickets, dividend, carry_out)
        self.day_requests: list[date] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "year" in params:
            year = int(params["year"])
            return httpx.Response(
                200, content=year_list([d for d in self.listed if d.year == year])
            )
        if "date" in params:
            d = datetime.strptime(params["date"], "%Y%m%d").date()
            self.day_requests.append(d)
            if d in self.pages:
                return httpx.Response(200, content=self.pages[d])
        return httpx.Response(404, content=b"not found")


def run(monkeypatch, site: Site, *args: str):
    def make_fetcher(*a, **k):
        return Fetcher("jra", http=HTTP, transport=httpx.MockTransport(site.handler),
                       sleep=lambda s: None)  # fmt: skip

    monkeypatch.setattr(wb, "make_fetcher", make_fetcher)
    monkeypatch.setattr(wb, "_quiet_check", lambda now: lambda: None)
    argv = ["japan", "win5-history", "--start-year", "2020", "--end-year", "2020", *args]
    return CliRunner().invoke(app, argv)


def age_year_list(year: int) -> None:
    """Make the cached year list look fetched mid-year, so the next run asks for it again."""
    path = paths.raw_page_path("jra", wb.YEAR_PAGE, year, str(year))
    stamp = datetime(year, 6, 1, 3, tzinfo=UTC).timestamp()
    os.utime(path, (stamp, stamp))


def token(output: str, name: str) -> int:
    found = re.findall(rf"(?<![\w]){re.escape(name)}=(\d+)", output)
    assert len(found) == 1, f"{name} not printed exactly once in:\n{output}"
    return int(found[0])


def bin_tokens(output: str) -> dict[str, int]:
    return {
        k: int(v) for k, v in re.findall(r"\b(bin_[0-9.]+_[0-9.]+|bin_[0-9.]+_inf)=(\d+)", output)
    }


def bins_of(figures) -> list[int]:
    out = [0, 0, 0, 0]
    for _, carry_in, _ in figures:
        i = expected_bin(carry_in, NET, [0, 0.5, 1, 2], 0.30)
        if i is not None:
            out[i] += 1
    return out


BIN_NAMES = ["bin_0_0.5", "bin_0.5_1", "bin_1_2", "bin_2_inf"]


def test_the_spec_the_command_reports_bins_for_is_the_frozen_study_spec():
    spec = load_country_yaml("japan", "win5_study")
    assert list(spec["ratio_edges"]) == [0, 0.5, 1, 2]  # the tokens below are its bins


def test_the_command_appends_new_days_run_after_run_and_reports_the_counts(monkeypatch):
    first = Site([d for d, *_ in SITE_DAYS[:3]])
    out = run(monkeypatch, first)
    assert out.exit_code == 0, out.output
    assert token(out.output, "added_days") == 3
    assert token(out.output, "days") == 3
    assert [bin_tokens(out.output)[n] for n in BIN_NAMES] == bins_of(SITE_DAYS[:3]) == [1, 1, 0, 0]
    assert token(out.output, "holdout_days") == 0
    table_1 = win5.read_history()
    assert list(table_1["race_date"]) == [d for d, *_ in SITE_DAYS[:3]]

    # three more WIN5 days run: the year list is fetched again (it was cached mid-year)
    age_year_list(2020)
    second = Site([d for d, *_ in SITE_DAYS])
    out = run(monkeypatch, second)
    assert out.exit_code == 0, out.output
    assert token(out.output, "added_days") == 3
    assert token(out.output, "days") == 6
    assert [bin_tokens(out.output)[n] for n in BIN_NAMES] == bins_of(SITE_DAYS) == [1, 1, 2, 1]
    table_2 = win5.read_history()
    want = win5.build_history([site_row(*s) for s in SITE_DAYS])
    pd.testing.assert_frame_equal(table_2, want, check_dtype=False)
    # the first three rows are the rows of the first run, untouched
    pd.testing.assert_frame_equal(
        table_2.iloc[:3].reset_index(drop=True), table_1, check_dtype=False
    )

    # nothing new: no days added, the table is the same, the counts are still printed
    age_year_list(2020)
    third = Site([d for d, *_ in SITE_DAYS])
    out = run(monkeypatch, third)
    assert out.exit_code == 0, out.output
    assert token(out.output, "added_days") == 0
    assert token(out.output, "days") == 6
    assert [bin_tokens(out.output)[n] for n in BIN_NAMES] == [1, 1, 2, 1]
    pd.testing.assert_frame_equal(win5.read_history(), table_2)


def test_a_run_over_later_years_keeps_the_days_of_the_earlier_ones(monkeypatch):
    old = [day(date(2018, 3, 4), NET, 225_000_000, 0.0), day(date(2018, 3, 11), NET, 0, 0.7),
           day(date(2012, 4, 8), NET, 75_000_000, 0.0)]  # fmt: skip
    win5.write_history(win5.build_history(old))
    site = Site([d for d, *_ in SITE_DAYS])
    out = run(monkeypatch, site)  # --start-year 2020: the 2012 and 2018 days are not read
    assert out.exit_code == 0, out.output
    assert token(out.output, "added_days") == 6
    assert token(out.output, "days") == 9
    assert sorted(site.day_requests) == [d for d, *_ in SITE_DAYS]
    table = win5.read_history()
    want = win5.build_history(old + [site_row(*s) for s in SITE_DAYS])
    pd.testing.assert_frame_equal(table, want, check_dtype=False)
    # the 2018 day with 0.75 and the 2012 day with 0.25 count in the bins beside the 2020 days
    assert [bin_tokens(out.output)[n] for n in BIN_NAMES] == [
        a + b for a, b in zip(bins_of(SITE_DAYS), [1, 1, 0, 0], strict=True)
    ]


def test_a_conflicting_day_stops_the_run_and_leaves_the_table_as_it_was(monkeypatch, data_dir):
    d, carry_in, hit = SITE_DAYS[1]
    other = site_row(d, carry_in, hit)
    other["payout_yen"] = 123_000  # the site says no hit and 0 paid
    other["winning_tickets"] = 1
    other["dividend_yen"] = 123_000
    win5.write_history(win5.build_history([other]))
    before = win5.history_path().read_bytes()
    out = run(monkeypatch, Site([d for d, *_ in SITE_DAYS]))
    assert out.exit_code != 0
    assert win5.history_path().read_bytes() == before


def test_the_command_does_not_fetch_or_add_holdout_days_and_does_not_read_their_outcomes(
    monkeypatch,
):
    held_in_table, held_on_site = SITE_DAYS[2][0], SITE_DAYS[3][0]
    marked = {held_in_table, held_on_site}
    for target in (win5, wb):
        monkeypatch.setattr(target, "is_holdout_day", lambda c, d: d in marked, raising=False)
    monkeypatch.setattr("horsetrack.core.eval.holdout.is_holdout_day", lambda c, d: d in marked)
    # a holdout day already in the table (an older run wrote it) whose outcomes cannot be read
    win5.write_history(win5.build_history([hostile(held_in_table)]))
    site = Site([d for d, *_ in SITE_DAYS])
    out = run(monkeypatch, site)
    assert out.exit_code == 0, out.output
    assert not marked & set(site.day_requests)  # holdout day pages are not asked for
    assert sorted(site.day_requests) == [d for d, *_ in SITE_DAYS if d not in marked]
    table = win5.read_history()
    assert held_on_site not in set(table["race_date"])  # not added
    assert held_in_table in set(table["race_date"])  # what was there stays
    assert token(out.output, "added_days") == 4
    assert token(out.output, "holdout_days") == 1
    dev = [s for s in SITE_DAYS if s[0] not in marked]
    assert [bin_tokens(out.output)[n] for n in BIN_NAMES] == bins_of(dev) == [1, 0, 1, 1]
