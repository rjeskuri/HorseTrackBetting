"""WIN5 per-day takeout, the payout cap, the payout check and the frozen carryover study (#135).

The owner's answers (ticket #135, 2026-11-25), which the study is frozen with:

    (1) coverage = the pool-level return (dividend x winning tickets / pool; 0 on a no-hit day);
        a live ticket rule (top-k per leg) is a later follow-up once WIN5 legs are logged (#130);
    (2) metric = per-day value vs 1.0;
    (3) carryover bins = multiples of the break-even carryover 0.30 x pool (edges: just above 0,
        0.5x, 1x, 2x);
    (4) model the 6億円 per 100 yen payout cap (excess carries over even on a hit); boosted-takeout
        days at their own rate from config/takeout_boosts.yaml;
    (5) day-block bootstrap; a positive edge = one-sided 95% lower bound above 1.0, stated before
        scoring.

Contract pinned here, module `horsetrack.countries.japan.research.win5` (round 2):

B. Takeout, cap, consistency.
- `win5_takeout(day, *, calendar=None) -> float | None`: WIN5's takeout on `day`, 1 - its payout
  rate. None before 2014-06-07 (jra.yaml `takeout_unknown_before_history`); else 0.30, changed by
  the boost calendar (`countries.japan.takeout.BoostCalendar`; None means the shipped
  takeout_boosts.yaml) exactly as `takeout.race_takeout` combines entries: a super_premium or
  ultra_premium entry sets the rate to SUPER_PREMIUM_RATE (0.80), a premium or ultra_premium entry
  adds PREMIUM_TOP_UP (0.05). Only whole-day entries whose pools include WIN5 (or name no pools)
  count; the 2024 GI-day entries, which list every pool but WIN5, do not. Shipped: 2024-09-16 and
  2025-05-04 give 0.15.
- `PAYOUT_CAP_YEN = 600_000_000` (per 100-yen ticket) and `capped_payout(pool_yen, carryover_yen,
  winning_tickets, *, takeout=TAKEOUT, cap_yen=PAYOUT_CAP_YEN) -> (payout_yen, carryover_out_yen)`:
  net = pool x (1 - takeout) + carryover; no ticket: (0, net); net / tickets <= cap: (net, 0);
  above the cap: (tickets x cap, net - tickets x cap), the excess carrying over on a hit.
  ValueError for negative tickets and the pool_ev refusals.
- `payout_mismatches(table, *, calendar=None) -> list[date]`, sorted: the days of a history table
  (`build_history`) with a known takeout where |payout_yen + carryover_out_yen - (pool x (1 -
  win5_takeout(day)) + carryover_in_yen)| exceeds the rounding tolerance, 10 yen per winning
  ticket (10 yen on a no-hit day): each dividend is rounded down to 10 yen per 100-yen ticket.
  Days before 2014-06-07 are skipped, never flagged.

C. The study.
- The spec is committed at src/horsetrack/countries/japan/config/win5_study.yaml, read with
  `core.config.load_country_yaml("japan", "win5_study")`, and holds: `family_id` (non-blank),
  `coverage: pool-level`, `pool: net-of-refunds`, `comparator: 1.0`, `ratio_edges: [0, 0.5, 1, 2]`,
  `break_even_takeout: 0.30`, `one_sided_level: 0.95`, `payout_cap_yen: 600000000`, an int `seed`
  and an int `n_boot` (>= 1000). docs/specs/win5-carryover-study.md is no longer a draft and names
  the family id.
- The pool is net of refunds (owner, #135, 2026-11-25, answer B): netkeiba's 発売金額 (pool_yen) is
  gross sales; on some days the payout falls short of pool x (1 - t_day) + carryover in, never
  over (hypothesis: refunds of combinations with a scratched horse). `net_pool(row, *,
  calendar=None) -> float`: from 2014-06-07, the pool the money identity implies, (payout_yen +
  carryover_out_yen - carryover_in_yen) / (1 - win5_takeout(day, calendar=calendar)), a boosted day
  at its own rate, pool_yen not entering it; on a no-hit day that is (carryover_out -
  carryover_in) / (1 - t), the history's carryover out holding the carryover in plus the day's net
  sales after takeout. Before 2014-06-07 (takeout unknown) it is pool_yen, the gross sales.
  ValueError when the implied pool is not positive (money that cannot add up).
- `day_value(row) -> float`: payout_yen / net_pool(row) (0 on a no-hit day); a dead-heat day counts
  every combination's payout. A hit day that carries nothing out scores exactly (1 - t_day) +
  carryover_in / net_pool, whatever the refunds.
- The spec names the pool: `pool: net-of-refunds`; a spec without it, or with another value, is
  refused by `check_study_spec` and `score_study` (ValueError), so the frozen spec records it.
- `score_study(history, spec, *, registry, family_id) -> dict`: NotPreRegistered unless `spec` is
  the spec frozen for `family_id`; ValueError for a spec it cannot score (no, empty, unsorted or
  negative ratio_edges, n_boot < 1, one_sided_level outside (0, 1), coverage other than
  pool-level, pool other than net-of-refunds). Holdout days (`is_holdout_day("japan", d)`) are
  dropped first and counted. A day's ratio r = carryover_in_yen / (break_even_takeout x
  net_pool(row)), with the spec's literal 0.30 (not the day's boosted takeout). Bins from edges
  e0 = 0 < e1 < ... < en: (e0, e1) open at 0, then [e_i, e_i+1), then [en, inf); r = 0 is in no
  bin. Result: `{family_id, spec_hash, holdout_days_excluded, gross_pool_carryover_days, bins}`:
  `gross_pool_carryover_days` counts the development days before 2014-06-07 with a carryover in,
  scored at gross sales; bins in edge order, each `{lo_ratio, hi_ratio (None for the open
  bin), n_days, mean_value, lower_bound, positive_edge}`: mean_value the mean day_value;
  lower_bound the (1 - one_sided_level) quantile of n_boot bootstrap means, the bin's days
  resampled with replacement (the WIN5 day is the unit) under the spec's seed; positive_edge
  `lower_bound > comparator`. An empty bin: n_days 0, mean_value and lower_bound None,
  positive_edge False. The same spec gives the same lower bound.
- `horsetrack japan win5-study freeze`: `default_registry("japan").freeze_family(family_id, spec)`
  with the shipped spec as loaded. `horsetrack japan win5-study score`: refuses (non-zero exit, no
  freeze, no scored run) before the freeze; else scores `read_history()` with the shipped spec and
  appends a RunRecord (hypothesis_id `P<n>`, the spec's family_id, ticket 135, status "scored")
  whose metrics hold, per non-empty bin, a FoldMetric with n = n_days, value = mean_value and lo =
  lower_bound; a scoring run that fails after the freeze appends a "failed" record and exits
  non-zero.

Every figure here is invented; HORSETRACK_DATA_DIR points at tmp_path, so the command's registry
(`default_registry`) and history live there; no test reads the real data dir.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import hashlib
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
from horsetrack.countries.japan.research import win5
from horsetrack.countries.japan.takeout import (
    PREMIUM_TOP_UP,
    SUPER_PREMIUM_RATE,
    Boost,
    BoostCalendar,
    load_boosts,
)

REPO = Path(__file__).resolve().parents[4]
CAP = 600_000_000


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def row(d, pool, carry_in, tickets, payout, carry_out, *, dividend=None, combos=None) -> dict:
    """A history row with every figure given (payout_yen as the pool paid it)."""
    if dividend is None and tickets:
        dividend = payout // tickets // 10 * 10
    return {
        "race_date": d,
        "pool_yen": pool,
        "carryover_in_yen": carry_in,
        "carryover_out_yen": carry_out,
        "winning_tickets": tickets,
        "dividend_yen": dividend if tickets else None,
        "payout_yen": payout,
        "winning_combinations": (1 if tickets else 0) if combos is None else combos,
    }


def spec_hash(spec: dict) -> str:
    """The registry's freeze hash, from its documented definition (sorted, compact JSON)."""
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- B: the day's takeout

ORDINARY = [date(2014, 6, 7), date(2014, 6, 8), date(2016, 5, 8), date(2019, 3, 10),
            date(2023, 6, 4), date(2026, 11, 22)]  # fmt: skip
ALL_POOL_BOOSTS = [date(2024, 9, 16), date(2025, 5, 4)]
GI_DAYS_WITHOUT_WIN5 = [date(2024, 2, 18), date(2024, 4, 7), date(2024, 12, 22)]
UNKNOWN = [date(2011, 4, 24), date(2013, 12, 22), date(2014, 6, 6)]


@pytest.mark.parametrize("day", ORDINARY, ids=str)
def test_win5_takeout_is_thirty_percent_on_an_ordinary_day_since_2014_06_07(day):
    assert win5.win5_takeout(day) == pytest.approx(0.30, abs=1e-9)


@pytest.mark.parametrize("day", ALL_POOL_BOOSTS, ids=str)
def test_win5_takeout_on_an_all_pool_ultra_premium_day_is_its_own_rate(day):
    want = 1 - (SUPER_PREMIUM_RATE + PREMIUM_TOP_UP)
    assert want == pytest.approx(0.15)
    assert win5.win5_takeout(day) == pytest.approx(want, abs=1e-9)
    # the rate comes from the calendar: without its entries the day is ordinary
    assert win5.win5_takeout(day, calendar=BoostCalendar(())) == pytest.approx(0.30, abs=1e-9)


@pytest.mark.parametrize("day", GI_DAYS_WITHOUT_WIN5, ids=str)
def test_a_boost_that_lists_pools_without_win5_leaves_win5_at_thirty_percent(day):
    entries = [e for e in load_boosts().entries if e.date == day]
    assert entries and all(e.pools is not None and "WIN5" not in e.pools for e in entries)
    assert win5.win5_takeout(day) == pytest.approx(0.30, abs=1e-9)


@pytest.mark.parametrize("day", UNKNOWN, ids=str)
def test_win5_takeout_before_2014_06_07_is_unknown(day):
    assert win5.win5_takeout(day) is None


SYN_DAY = date(2020, 3, 1)
CALENDAR_CASES = {
    "no entry": ((), 0.30),
    "super premium, every pool": ((Boost(kind="super_premium", date=SYN_DAY),), 0.20),
    "premium on WIN5": ((Boost(kind="premium", date=SYN_DAY, pools=("WIN5",)),), 0.25),
    "premium on other pools": ((Boost(kind="premium", date=SYN_DAY, pools=("WIN", "QUINELLA")),),
                               0.30),
    "super premium plus premium on WIN5": ((Boost(kind="super_premium", date=SYN_DAY),
                                            Boost(kind="premium", date=SYN_DAY, pools=("WIN5",))),
                                           0.15),
    "ultra premium, every pool": ((Boost(kind="ultra_premium", date=SYN_DAY),), 0.15),
    "ultra premium over a range": ((Boost(kind="ultra_premium", date=date(2020, 2, 23),
                                          until=date(2020, 3, 8)),), 0.15),
    "ultra premium another day": ((Boost(kind="ultra_premium", date=date(2020, 3, 8)),), 0.30),
}  # fmt: skip


@pytest.mark.parametrize("entries, want", CALENDAR_CASES.values(), ids=CALENDAR_CASES.keys())
def test_win5_takeout_reads_the_boost_calendar_it_is_given(entries, want):
    got = win5.win5_takeout(SYN_DAY, calendar=BoostCalendar(tuple(entries)))
    assert got == pytest.approx(want, abs=1e-9)


# ---------------------------------------------------------------- B: the payout cap

CAP_CASES = {
    # (pool, carryover, winning tickets, takeout, cap or None) -> (payout, carryover out)
    "no hit": ((1e9, 0, 0, 0.30, None), (0, 7e8)),
    "no hit after a carryover": ((1e9, 5e8, 0, 0.30, None), (0, 1.2e9)),
    "many winners": ((1e9, 0, 100, 0.30, None), (7e8, 0)),
    "share below the cap": ((3e9, 0, 4, 0.30, None), (2.1e9, 0)),
    "share exactly the cap": ((1e9, 5e8, 2, 0.30, None), (1.2e9, 0)),
    "one winner above the cap": ((2e9, 3e8, 1, 0.30, None), (6e8, 1.1e9)),
    "three winners above the cap": ((3e9, 1e9, 3, 0.30, None), (1.8e9, 1.3e9)),
    "boosted takeout above the cap": ((2e9, 0, 2, 0.15, None), (1.2e9, 5e8)),
    "ordinary takeout, same day, above the cap": ((2e9, 0, 2, 0.30, None), (1.2e9, 2e8)),
    "boosted takeout below the cap": ((2e9, 0, 4, 0.15, None), (1.7e9, 0)),
    "a lower cap binds": ((1e9, 0, 10, 0.30, 5e7), (5e8, 2e8)),
    "a cap met exactly": ((1e9, 0, 10, 0.30, 7e7), (7e8, 0)),
}


@pytest.mark.parametrize("args, want", CAP_CASES.values(), ids=CAP_CASES.keys())
def test_capped_payout_pays_up_to_the_cap_and_carries_the_rest(args, want):
    pool, carry, tickets, takeout, cap = args
    net = pool * (1 - takeout) + carry
    assert math.isclose(want[0] + want[1], net, rel_tol=1e-12)  # the table is right
    kwargs = {"takeout": takeout} if cap is None else {"takeout": takeout, "cap_yen": cap}
    payout, carried = win5.capped_payout(pool, carry, tickets, **kwargs)
    assert payout == pytest.approx(want[0], abs=1.0), args
    assert carried == pytest.approx(want[1], abs=1.0), args


def test_capped_payout_defaults_to_thirty_percent_and_six_oku_per_ticket():
    assert win5.PAYOUT_CAP_YEN == CAP
    payout, carried = win5.capped_payout(2e9, 3e8, 1)
    assert (payout, carried) == (pytest.approx(6e8, abs=1.0), pytest.approx(1.1e9, abs=1.0))
    payout, carried = win5.capped_payout(2e9, 3e8, 3)  # 1.7e9 / 3 per ticket, under the cap
    assert (payout, carried) == (pytest.approx(1.7e9, abs=30.0), pytest.approx(0, abs=30.0))


@pytest.mark.parametrize(
    "pool, carry, tickets, takeout",
    [(1e9, 0, -1, 0.30), (0, 0, 1, 0.30), (-1e9, 0, 1, 0.30), (1e9, -5, 1, 0.30),
     (1e9, 0, 1, 1.0), (1e9, 0, 1, -0.1)],
)  # fmt: skip
def test_capped_payout_refuses_impossible_figures(pool, carry, tickets, takeout):
    with pytest.raises(ValueError):
        win5.capped_payout(pool, carry, tickets, takeout=takeout)


# ---------------------------------------------------------------- B: the payout check

DEAD_HEAT = date(2019, 3, 17)
MISMATCH_CASES = {
    # dead heat: 1.8e9 * 0.7 + 3e8 = 1.56e9 split over two combinations (120 and 900 tickets)
    "dead heat, both combinations paid": (
        row(DEAD_HEAT, 1_800_000_000, 300_000_000, 1_020, 1_559_994_000, 0,
            dividend=6_500_000, combos=2), False),
    "dead heat, only the first combination's payout": (
        row(DEAD_HEAT, 1_800_000_000, 300_000_000, 1_020, 780_000_000, 0,
            dividend=6_500_000, combos=2), True),
    "no hit, the whole net carried": (row(date(2016, 5, 8), 987_654_300, 0, 0, 0, 691_358_010),
                                      False),
    "no hit, carryover short": (row(date(2016, 5, 8), 987_654_300, 0, 0, 0, 681_358_010), True),
    "capped hit, excess carried": (row(date(2018, 4, 1), 2_000_000_000, 300_000_000, 1, CAP,
                                       1_100_000_000), False),
    "capped hit, excess lost": (row(date(2018, 4, 1), 2_000_000_000, 300_000_000, 1, CAP, 0),
                                True),
    "rounding just under 10 yen a ticket": (row(date(2017, 6, 4), 1_000_000_000, 0, 500,
                                                700_000_000 - 4_990, 0), False),
    "shortfall well over 10 yen a ticket": (row(date(2017, 6, 4), 1_000_000_000, 0, 500,
                                                700_000_000 - 10_010, 0), True),
    "paid more than the pool": (row(date(2017, 6, 4), 1_000_000_000, 0, 500,
                                    700_000_000 + 10_010, 0), True),
    "boosted day at its own rate": (row(date(2025, 5, 4), 2_000_000_000, 0, 400,
                                        1_700_000_000, 0), False),
    "boosted day at the ordinary rate": (row(date(2025, 5, 4), 2_000_000_000, 0, 400,
                                             1_400_000_000, 0), True),
    "ordinary day at a boosted rate": (row(date(2025, 5, 11), 2_000_000_000, 0, 400,
                                           1_700_000_000, 0), True),
    "2013, takeout unknown: skipped": (row(date(2013, 12, 22), 1_000_000_000, 0, 10,
                                           500_000_000, 0), False),
    "2014-06-01, takeout unknown: skipped": (row(date(2014, 6, 1), 1_000_000_000, 0, 10,
                                                 500_000_000, 0), False),
    "2014-06-08, takeout known: checked": (row(date(2014, 6, 8), 1_000_000_000, 0, 10,
                                               500_000_000, 0), True),
}  # fmt: skip


@pytest.mark.parametrize("day, flagged", MISMATCH_CASES.values(), ids=MISMATCH_CASES.keys())
def test_payout_mismatches_flags_a_day_whose_money_does_not_add_up(day, flagged):
    table = win5.build_history([day])
    assert win5.payout_mismatches(table) == ([day["race_date"]] if flagged else [])


def test_payout_mismatches_names_every_bad_day_once_sorted():
    picks = [
        "capped hit, excess lost",  # 2018-04-01, flagged
        "dead heat, both combinations paid",
        "2013, takeout unknown: skipped",
        "boosted day at the ordinary rate",  # 2025-05-04, flagged
        "no hit, the whole net carried",
        "shortfall well over 10 yen a ticket",  # 2017-06-04, flagged
    ]
    rows = [MISMATCH_CASES[k][0] for k in picks]
    table = win5.build_history(rows[::-1])
    assert win5.payout_mismatches(table) == [date(2017, 6, 4), date(2018, 4, 1), date(2025, 5, 4)]


def test_payout_mismatches_uses_the_calendar_it_is_given():
    day = MISMATCH_CASES["boosted day at its own rate"][0]
    table = win5.build_history([day])
    assert win5.payout_mismatches(table, calendar=BoostCalendar(())) == [date(2025, 5, 4)]
    boosted = BoostCalendar((Boost(kind="ultra_premium", date=date(2025, 5, 11)),))
    other = win5.build_history([MISMATCH_CASES["ordinary day at a boosted rate"][0]])
    assert win5.payout_mismatches(other, calendar=boosted) == []


# ---------------------------------------------------------------- C: the shipped spec

SHIPPED_KEYS = {"family_id", "coverage", "pool", "comparator", "ratio_edges", "break_even_takeout",
                "one_sided_level", "payout_cap_yen", "seed", "n_boot"}  # fmt: skip
NET_POOL = "net-of-refunds"


def shipped_spec() -> dict:
    return dict(load_country_yaml("japan", "win5_study"))


def test_the_shipped_spec_holds_the_owners_answers():
    spec = shipped_spec()
    assert SHIPPED_KEYS <= set(spec)
    assert isinstance(spec["family_id"], str) and spec["family_id"].strip()
    assert spec["coverage"] == "pool-level"
    assert spec["pool"] == NET_POOL
    assert spec["comparator"] == 1.0
    assert spec["ratio_edges"] == [0, 0.5, 1, 2]
    assert spec["break_even_takeout"] == 0.30
    assert spec["one_sided_level"] == 0.95
    assert spec["payout_cap_yen"] == CAP
    assert type(spec["seed"]) is int
    assert type(spec["n_boot"]) is int and spec["n_boot"] >= 1000
    json.dumps(spec, allow_nan=False)  # plain JSON, so it can be frozen


def test_the_study_doc_is_signed_and_names_the_family():
    text = (REPO / "docs" / "specs" / "win5-carryover-study.md").read_text(encoding="utf-8")
    assert "DRAFT" not in text
    assert shipped_spec()["family_id"] in text


# ---------------------------------------------------------------- C: the net pool and day value

KNOWN_FROM = date(2014, 6, 7)  # jra.yaml: WIN5 takeout unknown before
BOOSTED = {date(2024, 9, 16), date(2025, 5, 4)}  # shipped ultra premium days: takeout 15%


def takeout_pct(d: date) -> int | None:
    """WIN5's takeout in percent on a day, from the facts above (not from the module)."""
    if d < KNOWN_FROM:
        return None
    return 15 if d in BOOSTED else 30


def netted(d, net, gross, carry_in, tickets, payout, *, combos=None, dividend=None) -> dict:
    """A day whose money adds up on the net pool `net` at the day's own takeout: payout + carry out
    = net x (1 - t) + carry in, while the page's pool_yen is the gross sales `gross`."""
    carry_out = carry_in + net * (100 - takeout_pct(d)) // 100 - payout
    assert carry_out >= 0 and net * (100 - takeout_pct(d)) % 100 == 0, (d, net, payout)
    return row(d, gross, carry_in, tickets, payout, carry_out, combos=combos, dividend=dividend)


# id -> (row, expected net pool); gross sales above the net pool on every day from 2014-06-07
NET_POOL_CASES = {
    "hit, no carryover, refunds": (
        netted(date(2016, 5, 8), 1_000_000_000, 1_031_234_500, 0, 40, 700_000_000),
        1_000_000_000),
    "hit after a carryover, refunds": (
        netted(date(2019, 3, 3), 1_234_567_800, 1_300_000_000, 250_000_000, 25, 1_114_197_460),
        1_234_567_800),
    "no hit after a carryover: (out - in) / (1 - t)": (
        netted(date(2016, 5, 15), 987_654_300, 1_000_000_000, 300_000_000, 0, 0), 987_654_300),
    "capped hit, excess carried": (
        netted(date(2018, 4, 1), 2_000_000_000, 2_050_000_000, 300_000_000, 1, CAP),
        2_000_000_000),
    "dead heat, both combinations": (
        netted(DEAD_HEAT, 1_799_990_000, 1_800_000_000, 300_000_000, 1_020, 1_559_993_000,
               combos=2, dividend=6_500_000), 1_799_990_000),
    "first day of a known takeout": (
        netted(KNOWN_FROM, 800_000_000, 820_000_000, 0, 12, 560_000_000), 800_000_000),
    "boosted day, its own 15%": (
        netted(date(2025, 5, 4), 2_000_000_000, 2_100_000_000, 0, 400, 1_700_000_000),
        2_000_000_000),
    "boosted day after a carryover, its own 15%": (
        netted(date(2024, 9, 16), 1_500_000_000, 1_560_000_000, 200_000_000, 3, 1_475_000_000),
        1_500_000_000),
    # takeout unknown: the gross sales, whatever the money says at 30%
    "2014-06-01, gross": (row(date(2014, 6, 1), 900_000_000, 100_000_000, 10, 400_000_000, 0),
                          900_000_000),
    "2013 no hit, gross": (row(date(2013, 12, 22), 1_000_000_000, 200_000_000, 0, 0, 800_000_000),
                           1_000_000_000),
    "the first WIN5 day, gross": (row(date(2011, 4, 24), 1_200_000_000, 0, 9, 500_000_000, 0),
                                  1_200_000_000),
}  # fmt: skip


@pytest.mark.parametrize("day, want", NET_POOL_CASES.values(), ids=NET_POOL_CASES.keys())
def test_net_pool_is_implied_by_the_money_from_2014_06_07_and_gross_before(day, want):
    record = win5.build_history([day]).to_dict("records")[0]
    assert win5.net_pool(record) == pytest.approx(want, rel=1e-9)
    assert win5.net_pool(day) == pytest.approx(want, rel=1e-9)  # a plain dict row too
    if day["race_date"] >= KNOWN_FROM:
        # the gross sales do not enter it
        tripled = dict(day, pool_yen=3 * day["pool_yen"])
        assert win5.net_pool(tripled) == pytest.approx(want, rel=1e-9)
    else:
        # the gross sales are the pool, not what the money implies at 30%
        implied = (day["payout_yen"] + day["carryover_out_yen"] - day["carryover_in_yen"]) / 0.7
        assert not math.isclose(implied, want, rel_tol=1e-3)


def test_net_pool_takes_the_takeout_from_the_calendar_it_is_given():
    boosted = NET_POOL_CASES["boosted day, its own 15%"][0]  # 2025-05-04, 1.7e9 paid out
    plain = BoostCalendar(())
    assert win5.net_pool(boosted, calendar=plain) == pytest.approx(1.7e9 / 0.7, rel=1e-9)
    ordinary = row(date(2025, 5, 11), 2_100_000_000, 0, 400, 1_700_000_000, 0)
    assert win5.net_pool(ordinary) == pytest.approx(1.7e9 / 0.7, rel=1e-9)
    ultra = BoostCalendar((Boost(kind="ultra_premium", date=date(2025, 5, 11)),))
    assert win5.net_pool(ordinary, calendar=ultra) == pytest.approx(2e9, rel=1e-9)


@pytest.mark.parametrize(
    "carry_in, carry_out",
    [(300_000_000, 300_000_000), (300_000_000, 0), (300_000_000, 100_000_000)],
    ids=["nothing-added", "carryover-lost", "carryover-shrank"],
)
def test_net_pool_refuses_money_that_implies_no_pool(carry_in, carry_out):
    day = row(date(2017, 6, 4), 1_000_000_000, carry_in, 0, 0, carry_out)
    with pytest.raises(ValueError):
        win5.net_pool(day)


def test_day_value_is_the_payout_over_the_net_pool_and_counts_every_combination():
    for name, (day, net) in NET_POOL_CASES.items():
        got = win5.day_value(win5.build_history([day]).to_dict("records")[0])
        want = day["payout_yen"] / net if day["winning_tickets"] else 0.0
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-15), name
        t = takeout_pct(day["race_date"])
        if t is not None and day["winning_tickets"] and day["carryover_out_yen"] == 0:
            # a refund day scores the pool's own return, not the smaller gross figure
            assert math.isclose(got, 1 - t / 100 + day["carryover_in_yen"] / net, rel_tol=1e-9)
            assert got > day["payout_yen"] / day["pool_yen"] + 1e-6, name
    # the dead heat's largest dividend times all its tickets would be 3.68, not 0.87
    assert 6_500_000 * 1_020 / 1_800_000_000 > 3


# ---------------------------------------------------------------- C: scoring

HOLDOUT_A, HOLDOUT_B = date(2026, 11, 29), date(2026, 12, 1)
SPEC = {
    "family_id": "F-test-win5",
    "coverage": "pool-level",
    "pool": NET_POOL,
    "comparator": 1.0,
    "ratio_edges": [0, 0.5, 1, 2],
    "break_even_takeout": 0.30,
    "one_sided_level": 0.95,
    "payout_cap_yen": CAP,
    "seed": 20261125,
    "n_boot": 2000,
}
EDGES = [(0, 0.5), (0.5, 1), (1, 2), (2, None)]
P = 1_000_000_000
# carryovers on a 1e9 net pool: ratio 0.25, 0.75, 1.5 and 3 (break-even carryover 3e8); a hit day
# that carries nothing out scores 0.7 + 0.3 x ratio, the most a day of the bin can score:
# 0.775, 0.925, 1.15 and 1.6
C_BIN = [75_000_000, 225_000_000, 450_000_000, 900_000_000]
REFUNDS = 0.03  # gross sales above the net pool, on every day from 2014-06-07 unless given
GROSS_START = date(2012, 1, 8)  # Sundays before 2014-06-07: the gross pool, takeout unknown


def sundays(n: int, start: date = date(2016, 1, 3)) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def valued(d, pool, carry, value, *, gross=None) -> dict:
    """A day whose pool-level return is `value` (payout = value x pool). From 2014-06-07 `pool` is
    the net pool: the money adds up on it at the day's own takeout and the page's pool_yen is the
    gross sales (`gross`, else 3% above). Before, `pool` is the gross pool_yen itself."""
    payout = round(value * pool)
    tickets = 0 if payout == 0 else 7
    t = takeout_pct(d)
    if t is None:
        return row(d, pool, carry, tickets, payout, max(0, carry + pool * 7 // 10 - payout))
    carry_out = carry + pool * (100 - t) // 100 - payout
    assert carry_out >= 0, f"{value} is above what a day of ratio {carry / (0.3 * pool)} can pay"
    gross = round(pool * (1 + REFUNDS)) if gross is None else gross
    return row(d, gross, carry, tickets, payout, carry_out)


def history_of(
    groups: list[tuple[int, int, list[float]]],
    extra: list[dict] = (),
    start: date = date(2016, 1, 3),
) -> list[dict]:
    """groups: (pool, carryover, values), one development day per value, consecutive Sundays from
    `start`."""
    flat = [(pool, carry, v) for pool, carry, values in groups for v in values]
    rows = [valued(d, *f) for d, f in zip(sundays(len(flat), start), flat, strict=True)]
    return rows + list(extra)


def frozen(tmp_path: Path, spec: dict, family: str) -> Registry:
    registry = Registry(tmp_path / "r.jsonl", holdout_log=tmp_path / "h.jsonl")
    registry.freeze_family(family, spec)
    return registry


def score(rows, spec, registry, family) -> dict:
    return win5.score_study(win5.build_history(rows), spec, registry=registry, family_id=family)


def test_the_holdout_days_used_here_are_holdout_days():
    assert is_holdout_day("japan", HOLDOUT_A) and is_holdout_day("japan", HOLDOUT_B)
    assert not any(is_holdout_day("japan", d) for d in sundays(60))
    assert not any(is_holdout_day("japan", d) for d in sundays(30, start=GROSS_START))


def test_scoring_refuses_a_spec_that_is_not_the_frozen_one(tmp_path):
    rows = history_of([(P, C_BIN[0], [0.7, 0.75])])
    registry = Registry(tmp_path / "r.jsonl", holdout_log=tmp_path / "h.jsonl")
    with pytest.raises(win5.NotPreRegistered):
        score(rows, SPEC, registry, "F-test-win5")  # nothing frozen
    registry.freeze_family("F-test-win5", dict(SPEC, n_boot=2001))
    with pytest.raises(win5.NotPreRegistered):
        score(rows, SPEC, registry, "F-test-win5")  # frozen with another spec
    registry.freeze_family("F-other", SPEC)
    with pytest.raises(win5.NotPreRegistered):
        score(rows, SPEC, registry, "F-test-win5")  # another family's freeze


def test_a_frozen_study_reports_each_ratio_bin_with_its_bootstrap_lower_bound(tmp_path):
    below = [0.3, 0.5, 0.7, 0.6, 0.4]
    lucky = [0, 0, 0, 0, 0, 0.9]  # most resamples hold no hit
    above = [1.45, 1.5, 1.55, 1.58, 1.48, 1.52]
    rows = history_of(
        [(P, C_BIN[0], below), (P, C_BIN[1], lucky), (P, C_BIN[3], above), (P, 0, [0.7] * 3)],
        extra=[valued(HOLDOUT_A, P, C_BIN[2], 1.1), valued(HOLDOUT_B, P, C_BIN[0], 0.0)],
    )
    registry = frozen(tmp_path, SPEC, "F-test-win5")
    same_spec = json.loads(json.dumps(SPEC, sort_keys=True))  # other key order, same spec

    got = score(rows[::-1], same_spec, registry, "F-test-win5")

    assert got["family_id"] == "F-test-win5"
    assert got["spec_hash"] == spec_hash(SPEC)
    assert got["holdout_days_excluded"] == 2
    assert got["gross_pool_carryover_days"] == 0  # every day here is from 2014-06-07
    bins = got["bins"]
    assert [(b["lo_ratio"], b["hi_ratio"]) for b in bins] == EDGES
    # the no-carryover days in no bin; the holdout day that would fill [1, 2) dropped
    assert [b["n_days"] for b in bins] == [5, 6, 0, 6]
    for b, values in zip((bins[0], bins[1], bins[3]), (below, lucky, above), strict=True):
        mean = sum(values) / len(values)
        assert math.isclose(b["mean_value"], mean, rel_tol=1e-8), b
        assert min(values) - 1e-9 <= b["lower_bound"] <= mean + 1e-9, b
    assert bins[0]["lower_bound"] < 1.0 and bins[0]["positive_edge"] is False
    # a third of the resamples miss the one hit: the 5th percentile of the means is 0
    assert bins[1]["lower_bound"] == pytest.approx(0.0, abs=1e-12)
    assert bins[1]["positive_edge"] is False
    assert bins[2]["mean_value"] is None and bins[2]["lower_bound"] is None
    assert bins[2]["positive_edge"] is False
    assert bins[3]["lower_bound"] > 1.0 and bins[3]["positive_edge"] is True


EDGE_CASES = {
    "all exactly 1.0": ([1.0] * 5, 1.0, False),
    "all at 1.2": ([1.2] * 5, 1.0, True),
    "all just below 1": ([0.99] * 8, 1.0, False),
    "spread well above 1": ([1.4, 1.5, 1.6] * 3, 1.0, True),
    "all at 1.2, comparator 1.3": ([1.2] * 5, 1.3, False),
    "spread well above 1.3, comparator 1.3": ([1.4, 1.5, 1.6] * 3, 1.3, True),
}


@pytest.mark.parametrize("values, comparator, want", EDGE_CASES.values(), ids=EDGE_CASES.keys())
def test_a_positive_edge_needs_the_lower_bound_above_the_comparator(
    tmp_path, values, comparator, want
):
    spec = dict(SPEC, comparator=comparator, n_boot=500)
    registry = frozen(tmp_path, spec, "F-edge")
    # on days before 2014-06-07 (the gross pool) a value of exactly 1.0 is exact in floats
    days = history_of([(P, C_BIN[0], values)], start=GROSS_START)
    (first, *rest) = score(days, spec, registry, "F-edge")["bins"]
    assert first["n_days"] == len(values)
    assert first["positive_edge"] is want, first
    if len(set(values)) == 1:  # every resample has the same mean
        assert first["lower_bound"] == pytest.approx(values[0], abs=1e-9)
    assert all(b["n_days"] == 0 and b["positive_edge"] is False for b in rest)


# (pool, carryover, day, expected bin index or None, gross sales if given), the ratio carryover
# over 0.30 x pool on every day; `pool` is the gross pool_yen on a "gross" day (before
# 2014-06-07), else the net pool (gross sales 3% above unless given)
RATIO_CASES = [
    # before 2014-06-07, the gross pool: the edges exactly
    (P, 0, "gross", None),  # no carryover
    (P, 3_000_000, "gross", 0),  # r = 0.01
    (P, 149_999_000, "gross", 0),  # r just under 0.5
    (P, 150_000_000, "gross", 1),  # r = 0.5 exactly
    (2 * P, 300_000_000, "gross", 1),  # the same yen on twice the pool: r = 0.5
    (P, 300_000_000, "gross", 2),  # r = 1 exactly
    (P, 597_000_000, "gross", 2),  # r = 1.99
    (P, 600_000_000, "gross", 3),  # r = 2 exactly
    # from 2014-06-07, the net pool
    (P, 0, None, None),  # no carryover
    (P, 90_000_000, None, 0),  # r = 0.3
    (P, 149_000_000, None, 0),  # r = 0.497
    (P, 151_000_000, None, 1),  # r = 0.503
    (2 * P, 302_000_000, None, 1),  # the same ratio on twice the pool
    (P, 301_000_000, None, 2),  # r = 1.003
    (400_000_000, 300_000_000, None, 3),  # r = 2.5
    (P, 9_000_000_000, None, 3),  # r = 30
    # the net pool, not the gross sales: r = 0.53, 1.05, 2.1 (on the gross: 0.48, 0.95, 1.9)
    (950_000_000, 150_000_000, None, 1, 1_050_000_000),
    (950_000_000, 300_000_000, None, 2, 1_050_000_000),
    (950_000_000, 600_000_000, None, 3, 1_050_000_000),
    # boosted days: the net pool at the day's own 15%, the ratio still at 0.30 x pool
    # r = 0.53 (0.44 on the pool the money implies at 30%, 0.48 on the gross, 1.07 at 15%)
    (P, 160_000_000, date(2025, 5, 4), 1, 1_100_000_000),
    (P, 450_000_000, date(2024, 9, 16), 2),  # r = 1.5 (3.0 at the day's rate)
]


def test_days_go_to_the_bin_of_their_carryover_over_the_break_even_carryover(tmp_path):
    spec = dict(SPEC, n_boot=200)
    registry = frozen(tmp_path, spec, "F-ratio")
    gross_days, net_days = iter(sundays(30, start=GROSS_START)), iter(sundays(30))
    rows, want = [], [[] for _ in EDGES]
    for i, (pool, carry, day, where, *gross) in enumerate(RATIO_CASES):
        value = 0.02 * (i + 1)  # each day's value names it
        d = next(gross_days) if day == "gross" else day or next(net_days)
        rows.append(valued(d, pool, carry, value, gross=gross[0] if gross else None))
        if where is not None:
            want[where].append(value)
    # a no-hit day is binned on its net pool too: (carry out - carry in) / 0.7 = 9.5e8, r = 0.53
    rows.append(valued(next(net_days), 950_000_000, 150_000_000, 0.0, gross=1_050_000_000))
    want[1].append(0.0)

    got = score(rows, spec, registry, "F-ratio")
    bins = got["bins"]

    assert [b["n_days"] for b in bins] == [len(w) for w in want] == [4, 7, 5, 4]
    for b, values in zip(bins, want, strict=True):
        assert math.isclose(b["mean_value"], sum(values) / len(values), rel_tol=1e-8), b
    assert got["gross_pool_carryover_days"] == 7


@pytest.mark.parametrize("n_gross", [0, 2, 5])
def test_carryover_days_before_2014_06_07_score_at_gross_sales_and_are_counted(tmp_path, n_gross):
    spec = dict(SPEC, n_boot=200)
    registry = frozen(tmp_path, spec, f"F-gross-{n_gross}")
    # before 2014-06-07: a carryover of 4e8 on 1e9 gross sales (r = 1.33) paying 0.9 of them, a
    # no-hit day among them; at 30% the money would imply a pool of 7.1e8 and a value of 1.26
    early = [0.0 if i % 3 == 1 else 0.9 for i in range(n_gross)]
    gross_days = sundays(n_gross + 2, start=GROSS_START)
    rows = [row(d, P, 400_000_000, 7 if v else 0, round(v * P), 0 if v else 1_100_000_000)
            for d, v in zip(gross_days[:n_gross], early, strict=True)]  # fmt: skip
    rows += [valued(d, P, 0, 0.6) for d in gross_days[n_gross:]]  # no carryover: not counted
    late = [1.0, 1.1, 0.0]  # from 2014-06-07, on the net pool, the same bin
    rows += history_of([(P, 400_000_000, late)], extra=[valued(HOLDOUT_A, P, 400_000_000, 1.1)])

    got = score(rows, spec, registry, f"F-gross-{n_gross}")

    assert got["gross_pool_carryover_days"] == n_gross
    assert got["holdout_days_excluded"] == 1
    scored = early + late
    assert [b["n_days"] for b in got["bins"]] == [0, 0, len(scored), 0]
    assert math.isclose(got["bins"][2]["mean_value"], sum(scored) / len(scored), rel_tol=1e-8)


def test_the_lower_bound_is_the_fifth_percentile_of_the_day_bootstrap_and_repeats(tmp_path):
    values = [0.6 + 0.02 * i for i in range(40)]
    spec = dict(SPEC, n_boot=4000)
    registry = frozen(tmp_path, spec, "F-boot")
    rows = history_of([(P, C_BIN[3], values)])

    first = score(rows, spec, registry, "F-boot")["bins"][3]
    again = score(rows, spec, registry, "F-boot")["bins"][3]

    n = len(values)
    mean = sum(values) / n
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / n) / math.sqrt(n)  # plug-in
    # the means of 4000 day resamples are close to normal here: the 5th percentile sits 1.645
    # standard errors below the mean (2.5th: 1.96, 10th: 1.28)
    assert first["lower_bound"] == pytest.approx(mean - 1.645 * se, abs=0.15 * se)
    assert again["lower_bound"] == first["lower_bound"]  # the spec's seed fixes the draw


def test_a_dead_heat_day_counts_every_combination_in_the_study(tmp_path):
    spec = dict(SPEC, n_boot=200)
    registry = frozen(tmp_path, spec, "F-dead-heat")
    dead_heat = row(DEAD_HEAT, 1_800_000_000, 300_000_000, 1_020, 1_559_994_000, 0,
                    dividend=6_500_000, combos=2)  # fmt: skip
    net = (1_559_994_000 - 300_000_000) / 0.7  # the money identity: 1.79999e9, not the 1.8e9 sold
    # r = 3e8 / (0.3 x net) = 0.56: the [0.5, 1) bin, alone
    bins = score([dead_heat], spec, registry, "F-dead-heat")["bins"]
    assert [b["n_days"] for b in bins] == [0, 1, 0, 0]
    assert bins[1]["mean_value"] == pytest.approx(1_559_994_000 / net, rel=1e-9)
    assert bins[1]["mean_value"] != pytest.approx(1_559_994_000 / 1_800_000_000, rel=1e-7)


@pytest.mark.parametrize(
    "change",
    [
        {"ratio_edges": None},
        {"ratio_edges": []},
        {"ratio_edges": [0, 1, 0.5, 2]},
        {"ratio_edges": [-1, 0.5, 1]},
        {"n_boot": 0},
        {"one_sided_level": 1.5},
        {"coverage": "top-k per leg"},
        {"pool": None},
        {"pool": "gross"},
    ],
    ids=["no-edges", "empty-edges", "unsorted", "negative", "no-resamples", "bad-level",
         "other-coverage", "no-pool", "gross-pool"],
)  # fmt: skip
def test_a_frozen_spec_the_scorer_cannot_score_is_refused(tmp_path, change):
    spec = {k: v for k, v in {**SPEC, **change}.items() if v is not None}
    with pytest.raises(ValueError):
        win5.check_study_spec(spec)  # refused before a freeze
    registry = frozen(tmp_path, spec, "F-bad")
    with pytest.raises(ValueError):
        score(history_of([(P, C_BIN[0], [0.7, 0.75])]), spec, registry, "F-bad")


# ---------------------------------------------------------------- C: the command

# development days by bin (ratio 0.25, 0.75, 1.5), no-carryover days, and one holdout day that
# would fill the open bin if it were scored
CLI_GROUPS = [(P, C_BIN[0], [0.5, 0.6, 0.7]), (P, C_BIN[1], [0.5, 0.9]), (P, C_BIN[2], [1.1]),
              (P, 0, [0.7, 0.7, 0.7, 0.7])]  # fmt: skip
CLI_BINS = [(3, 0.6), (2, 0.7), (1, 1.1)]  # (n_days, mean) per non-empty bin
CLI_HOLDOUT_VALUE = 1.55


def registry_at(data_dir: Path, tmp_path: Path) -> Registry:
    """The registry the command writes (default_registry under the tmp data dir)."""
    return Registry(data_dir / "registry" / "japan.jsonl", holdout_log=tmp_path / "h.jsonl")


def write_cli_history() -> None:
    rows = history_of(CLI_GROUPS, extra=[valued(HOLDOUT_A, P, C_BIN[3], CLI_HOLDOUT_VALUE)])
    win5.write_history(win5.build_history(rows))


def test_the_command_refuses_to_score_before_the_freeze_then_freezes_and_scores(data_dir, tmp_path):
    spec = shipped_spec()
    family = spec["family_id"]
    write_cli_history()
    registry = registry_at(data_dir, tmp_path)
    runner = CliRunner()

    early = runner.invoke(app, ["japan", "win5-study", "score"])
    assert early.exit_code != 0, early.output
    assert registry.frozen_spec_hash(family) is None
    assert [r for r in registry.records(family) if r.status == "scored"] == []

    froze = runner.invoke(app, ["japan", "win5-study", "freeze"])
    assert froze.exit_code == 0, froze.output
    assert registry.frozen_spec_hash(family) == spec_hash(spec)
    again = runner.invoke(app, ["japan", "win5-study", "freeze"])  # freezing again is a no-op
    assert again.exit_code == 0, again.output

    scored = runner.invoke(app, ["japan", "win5-study", "score"])
    assert scored.exit_code == 0, scored.output
    runs = [r for r in registry.records(family) if r.status == "scored"]
    assert len(runs) == 1
    run = runs[0]
    assert re.fullmatch(r"P[0-9]+", run.hypothesis_id), run.hypothesis_id
    assert run.family_id == family and run.ticket == 135
    for n_days, mean in CLI_BINS:
        assert any(
            m.n == n_days
            and math.isclose(m.value, mean, rel_tol=1e-9)
            and m.lo is not None
            and m.lo <= m.value + 1e-12
            for m in run.metrics
        ), (n_days, mean, run.metrics)
    assert not any(math.isclose(m.value, CLI_HOLDOUT_VALUE) for m in run.metrics)  # holdout
    registry.verify()


def test_a_scoring_run_that_fails_after_the_freeze_is_recorded(data_dir, tmp_path):
    family = shipped_spec()["family_id"]
    registry = registry_at(data_dir, tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["japan", "win5-study", "freeze"]).exit_code == 0

    failed = runner.invoke(app, ["japan", "win5-study", "score"])  # no history table written

    assert failed.exit_code != 0, failed.output
    statuses = [r.status for r in registry.records(family)]
    assert statuses == ["failed"], statuses
