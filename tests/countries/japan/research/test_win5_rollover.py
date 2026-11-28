"""The WIN5 no-hit (rollover) rate and the rollover-adjusted EV of the carryover bins (ticket #323).

Follow-up of #135: its [1, 2) bin scored a mean of 1.059 on days that all hit, and its day bootstrap
cannot see a rollover that never happened in the sample. A no-hit day pays 0 and passes all the
money on, so with `won_value` the mean day value of the won days of a bin and `q` the no-hit rate:

    rollover-adjusted EV = (1 - q) x won_value          (won_value = 0.70 + 0.30 x ratio at 30%)
    break-even no-hit rate q* = 1 - 1 / won_value       (0.0 when won_value <= 1: no q helps)

Contract pinned here, new module `horsetrack.countries.japan.research.win5_rollover` (every figure
is a plain float, int, None or dict; the spec is config/win5_rollover.yaml, frozen with
`freeze_family` under its own family_id before any scored read):

A. Arithmetic.
- `rollover_adjusted_ev(won_value, no_hit_rate) -> float` = (1 - q) x won_value. ValueError for a
  rate outside [0, 1], a negative or non-finite value, a bool or non-number.
- `break_even_no_hit_rate(won_value) -> float` = 1 - 1 / won_value, and 0.0 for 0 <= won_value <= 1.
  ValueError for a negative or non-finite value.
- `is_no_hit(row) -> bool`: a history row with no winning ticket. A day that paid and passed the
  excess over the 6億円 cap on (carryover out > 0) is a hit, not a rollover of the pool.

B. The interval.
- `no_hit_interval(n_no_hit, n_days, *, confidence=0.95) -> (lower, upper)`: the exact two-sided
  Clopper-Pearson interval (lower 0.0 when n_no_hit = 0, upper 1.0 when n_no_hit = n_days).
  ValueError for n_days < 1, n_no_hit outside [0, n_days], a confidence outside (0, 1).

C. The study, `check_rollover_spec(spec)` (ValueError unless scorable) and
`score_rollover(history, spec, *, registry, family_id) -> dict`: NotPreRegistered
(`research.win5.NotPreRegistered`) unless `spec` is the spec frozen for `family_id`; holdout days
dropped first (D14) and counted. Bins as `win5.score_study` (ratio = carryover in /
(break_even_takeout x net pool), edges from the spec; a day with no carryover is in no bin), no-hit
days included.
Result: family_id, spec_hash, holdout_days_excluded, `overall` (all development days, with or
without carryover: n_days, n_no_hit, no_hit_rate, no_hit_lower, no_hit_upper) and `bins`, each
lo_ratio, hi_ratio, n_days, n_no_hit, no_hit_rate, no_hit_lower, no_hit_upper, won_mean_value (mean
`win5.day_value` of its won days), break_even_no_hit_rate, adjusted_ev (at the bin's own rate),
adjusted_ev_at_upper (at its upper bound), adjusted_ev_pooled and adjusted_ev_pooled_at_upper (at
the overall rate and its upper bound). A bin with no day has n_days = n_no_hit = 0 and None for the
rest; a bin whose days are all no-hit has a rate and interval but None for the won-value figures.

D. The command `horsetrack japan win5-rollover freeze|score` (ticket 323): `freeze` freezes the
shipped spec, `score` refuses before the freeze, then scores the WIN5 history and appends one
RunRecord (a failed run too). One FoldMetric per non-empty bin and one for all development days,
metric "no_hit_rate": value the rate, lo and hi the interval, n the days.
"""

from __future__ import annotations  # noqa: I001

import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest
from scipy.stats import beta
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
from horsetrack.countries.japan.research import win5
from horsetrack.countries.japan.research import win5_rollover as ro

CAP = 600_000_000
P = 1_000_000_000  # the net pool; break-even carryover 3e8
# carryovers at ratio 0.25, 0.75, 1.5 and 3; a won day of a bin scores at most 0.775, 0.925, 1.15
# and 1.6
C_BIN = [75_000_000, 225_000_000, 450_000_000, 900_000_000]
HOLDOUT_A, HOLDOUT_B = date(2026, 11, 29), date(2026, 12, 1)
SPEC = {
    "family_id": "F-test-rollover",
    "interval": "clopper-pearson",
    "confidence": 0.95,
    "pool": "net-of-refunds",
    "ratio_edges": [0, 0.5, 1, 2],
    "break_even_takeout": 0.30,
    "payout_cap_yen": CAP,
}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def spec_hash(spec: dict) -> str:
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cp(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson from the beta quantiles (scipy, not the module under test)."""
    a = (1 - confidence) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------- A: the arithmetic

ADJUSTED = {  # (won_value, q) -> (1 - q) x won_value
    "no rollover keeps the won value": (1.059, 0.0, 1.059),
    "the [1, 2) bin at the break-even rate": (1.059, 1 - 1 / 1.059, 1.0),
    "a tenth of the days roll over": (1.5, 0.1, 1.35),
    "a quarter": (1.2, 0.25, 0.9),
    "every day rolls over": (1.7, 1.0, 0.0),
    "a bin that never pays": (0.0, 0.3, 0.0),
    "below one, half": (0.775, 0.5, 0.3875),
}


@pytest.mark.parametrize("won, q, want", ADJUSTED.values(), ids=ADJUSTED.keys())
def test_adjusted_ev_is_the_won_value_times_the_share_of_days_that_pay(won, q, want):
    assert ro.rollover_adjusted_ev(won, q) == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_adjusted_ev_falls_with_the_rate_and_rises_with_the_won_value():
    rates = [0.0, 0.05, 0.2, 0.5, 0.9]
    values = [ro.rollover_adjusted_ev(1.3, q) for q in rates]
    assert values == sorted(values, reverse=True) and len(set(values)) == len(values)
    bigger = [ro.rollover_adjusted_ev(w, 0.2) for w in (0.8, 1.0, 1.4)]
    assert bigger == sorted(bigger) and len(set(bigger)) == 3


@pytest.mark.parametrize("won, q", [(1.0, -0.01), (1.0, 1.01), (-0.5, 0.1), (1.0, float("nan")),
                                    (float("inf"), 0.1), (None, 0.1), (1.0, None), (1.0, "x"),
                                    (True, 0.1), (1.0, True)])  # fmt: skip
def test_adjusted_ev_refuses_figures_that_are_not_a_value_and_a_rate(won, q):
    with pytest.raises(ValueError):
        ro.rollover_adjusted_ev(won, q)


BREAK_EVEN = {  # won_value -> q* = 1 - 1 / won_value
    1.059: 1 - 1 / 1.059,  # the [1, 2) bin of #135: about 5.6%
    1.15: 1 - 1 / 1.15,
    1.25: 0.2,
    1.5: 1 / 3,
    2.0: 0.5,
    4.0: 0.75,
}


@pytest.mark.parametrize("won, want", BREAK_EVEN.items())
def test_break_even_no_hit_rate_is_where_the_adjusted_ev_reaches_one(won, want):
    q = ro.break_even_no_hit_rate(won)
    assert q == pytest.approx(want, rel=1e-12)
    assert ro.rollover_adjusted_ev(won, q) == pytest.approx(1.0, rel=1e-12)
    assert (
        ro.rollover_adjusted_ev(won, min(1.0, q + 0.01))
        < 1.0
        < ro.rollover_adjusted_ev(won, q - 0.01)
    )


def test_the_135_bin_breaks_even_at_about_five_point_six_percent():
    assert 0.0555 < ro.break_even_no_hit_rate(1.059) < 0.0565


@pytest.mark.parametrize("won", [0.0, 0.3, 0.775, 0.99, 1.0])
def test_break_even_is_zero_when_the_won_value_cannot_beat_one(won):
    assert ro.break_even_no_hit_rate(won) == 0.0


@pytest.mark.parametrize("won", [-0.1, float("nan"), float("inf"), None, "1.2", True])
def test_break_even_refuses_a_value_that_is_not_a_non_negative_number(won):
    with pytest.raises(ValueError):
        ro.break_even_no_hit_rate(won)


def hist_row(tickets, carry_out, payout=0) -> dict:
    return {"race_date": date(2016, 1, 3), "pool_yen": P, "carryover_in_yen": 0,
            "carryover_out_yen": carry_out, "winning_tickets": tickets, "dividend_yen": None,
            "payout_yen": payout, "winning_combinations": 1 if tickets else 0}  # fmt: skip


def test_a_no_hit_day_is_a_day_without_a_winning_ticket():
    assert ro.is_no_hit(hist_row(0, 700_000_000)) is True
    assert ro.is_no_hit(hist_row(0, 0)) is True  # the figure is the tickets, not the carry
    assert ro.is_no_hit(hist_row(1, 0, payout=700_000_000)) is False
    # a capped win passes the excess on yet paid: a hit
    assert ro.is_no_hit(hist_row(1, 100_000_000, payout=CAP)) is False
    assert ro.is_no_hit(hist_row(1_020, 0, payout=700_000_000)) is False


# ---------------------------------------------------------------- B: the interval

CP_CASES = [(0, 4), (0, 10), (0, 40), (1, 4), (1, 10), (3, 20), (7, 23), (12, 24), (19, 20),
            (4, 4), (10, 10), (0, 1), (1, 1), (5, 100)]  # fmt: skip


@pytest.mark.parametrize("k, n", CP_CASES)
def test_the_interval_is_the_exact_clopper_pearson_interval(k, n):
    lo, hi = ro.no_hit_interval(k, n)
    want_lo, want_hi = cp(k, n)
    assert lo == pytest.approx(want_lo, abs=1e-9)
    assert hi == pytest.approx(want_hi, abs=1e-9)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_no_hit_in_n_days_still_leaves_an_upper_bound_that_shrinks_with_n():
    # the rollover that never occurred in the sample: 0 of 4 days leaves 1 - 0.025 ** (1 / 4)
    lo, hi = ro.no_hit_interval(0, 4)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 4), abs=1e-9)
    assert 0.60 < hi < 0.61
    uppers = [ro.no_hit_interval(0, n)[1] for n in (4, 8, 16, 32, 64)]
    assert uppers == sorted(uppers, reverse=True) and uppers[-1] > 0
    for n, hi_n in zip((4, 8, 16, 32, 64), uppers, strict=True):
        assert hi_n == pytest.approx(1 - 0.025 ** (1 / n), abs=1e-9)


def test_every_day_a_no_hit_day_has_upper_one_and_a_positive_lower_bound():
    lo, hi = ro.no_hit_interval(5, 5)
    assert hi == 1.0
    assert lo == pytest.approx(0.025 ** (1 / 5), abs=1e-9)


@pytest.mark.parametrize("confidence", [0.5, 0.8, 0.9, 0.99])
def test_a_higher_confidence_gives_a_wider_interval(confidence):
    lo, hi = ro.no_hit_interval(3, 20, confidence=confidence)
    want_lo, want_hi = cp(3, 20, confidence)
    assert (lo, hi) == (pytest.approx(want_lo, abs=1e-9), pytest.approx(want_hi, abs=1e-9))
    assert (lo, hi) != (
        pytest.approx(cp(3, 20)[0], abs=1e-6),
        pytest.approx(cp(3, 20)[1], abs=1e-6),
    )


def test_the_interval_widens_with_the_confidence_level():
    widths = [
        hi - lo for lo, hi in (ro.no_hit_interval(3, 20, confidence=c) for c in (0.5, 0.9, 0.99))
    ]
    assert widths == sorted(widths) and len(set(widths)) == 3


@pytest.mark.parametrize("k, n, confidence", [(0, 0, 0.95), (-1, 5, 0.95), (6, 5, 0.95),
                                              (1, 5, 0.0), (1, 5, 1.0), (1, 5, 1.5),
                                              (1.5, 5, 0.95), (1, 5.5, 0.95)])  # fmt: skip
def test_the_interval_refuses_counts_and_levels_that_make_no_sense(k, n, confidence):
    with pytest.raises(ValueError):
        ro.no_hit_interval(k, n, confidence=confidence)


# ---------------------------------------------------------------- C: the study


def sundays(n: int, start: date = date(2016, 1, 3)) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def valued(d: date, carry: int, value: float) -> dict:
    """A day from 2014-06-07 whose money adds up at 30% on the net pool P: pays value x P (0: no
    winning ticket, everything passes on); the page's gross sales are 3% above the net pool."""
    payout = round(value * P)
    carry_out = carry + P * 7 // 10 - payout
    assert carry_out >= 0, (value, carry)
    return {"race_date": d, "pool_yen": round(P * 1.03), "carryover_in_yen": carry,
            "carryover_out_yen": carry_out, "winning_tickets": 0 if payout == 0 else 7,
            "dividend_yen": None if payout == 0 else payout // 7 // 10 * 10,
            "payout_yen": payout, "winning_combinations": 1 if payout else 0}  # fmt: skip


# the values of the days of each bin, 0 = a no-hit day; the days with no carryover are in no bin
BIN_VALUES = [
    [0.7, 0.75, 0.0, 0.72],  # ratio 0.25: 4 days, 1 no-hit
    [0.8, 0.9, 0.0, 0.0],  # ratio 0.75: 4 days, 2 no-hit
    [1.05, 1.1, 1.0, 1.05, 1.1],  # ratio 1.5: 5 days, 0 no-hit, won mean 1.06
    [1.5, 1.4, 0.0, 1.6],  # ratio 3: 4 days, 1 no-hit
]
NO_CARRY = [0.7, 0.7, 0.7, 0.7, 0.7, 0.0]  # 6 days, 1 no-hit
N_DAYS = sum(map(len, BIN_VALUES)) + len(NO_CARRY)  # 23
N_NO_HIT = sum(v == 0 for vs in [*BIN_VALUES, NO_CARRY] for v in vs)  # 5


def history(extra=(), *, shuffled=False) -> list[dict]:
    flat = [(c, v) for c, vs in zip(C_BIN, BIN_VALUES, strict=True) for v in vs]
    flat += [(0, v) for v in NO_CARRY]
    rows = [valued(d, c, v) for d, (c, v) in zip(sundays(len(flat)), flat, strict=True)]
    rows += list(extra)
    return rows[::-1] if shuffled else rows


def frozen(tmp_path: Path, spec: dict = SPEC) -> Registry:
    registry = Registry(tmp_path / "r.jsonl", holdout_log=tmp_path / "h.jsonl")
    registry.freeze_family(spec["family_id"], spec)
    return registry


def score(rows, spec, registry, family=None) -> dict:
    return ro.score_rollover(
        win5.build_history(rows), spec, registry=registry, family_id=family or spec["family_id"]
    )


def test_the_days_used_here_are_development_or_holdout_days_as_named():
    assert is_holdout_day("japan", HOLDOUT_A) and is_holdout_day("japan", HOLDOUT_B)
    assert not any(is_holdout_day("japan", d) for d in sundays(N_DAYS))


def test_scoring_refuses_a_spec_that_is_not_the_frozen_one(tmp_path):
    rows = history()
    registry = Registry(tmp_path / "r.jsonl", holdout_log=tmp_path / "h.jsonl")
    with pytest.raises(win5.NotPreRegistered):
        score(rows, SPEC, registry)  # nothing frozen
    registry.freeze_family(SPEC["family_id"], dict(SPEC, confidence=0.9))
    with pytest.raises(win5.NotPreRegistered):
        score(rows, SPEC, registry)  # frozen with another spec
    registry.freeze_family("F-other", SPEC)
    with pytest.raises(win5.NotPreRegistered):
        score(rows, SPEC, registry)  # another family's freeze


def test_each_bin_reports_the_no_hit_rate_its_interval_and_the_adjusted_ev(tmp_path):
    holdout = [valued(HOLDOUT_A, C_BIN[2], 0.0), valued(HOLDOUT_B, C_BIN[2], 1.1)]
    got = score(history(holdout, shuffled=True), SPEC, frozen(tmp_path))

    assert got["family_id"] == SPEC["family_id"]
    assert got["spec_hash"] == spec_hash(SPEC)
    assert got["holdout_days_excluded"] == 2

    overall = got["overall"]
    assert (overall["n_days"], overall["n_no_hit"]) == (N_DAYS, N_NO_HIT)
    assert overall["no_hit_rate"] == pytest.approx(N_NO_HIT / N_DAYS)
    lo_all, hi_all = cp(N_NO_HIT, N_DAYS)
    assert overall["no_hit_lower"] == pytest.approx(lo_all, abs=1e-9)
    assert overall["no_hit_upper"] == pytest.approx(hi_all, abs=1e-9)

    bins = got["bins"]
    assert [(b["lo_ratio"], b["hi_ratio"]) for b in bins] == [(0, 0.5), (0.5, 1), (1, 2), (2, None)]
    for b, values in zip(bins, BIN_VALUES, strict=True):
        n, k = len(values), sum(v == 0 for v in values)
        won = [v for v in values if v > 0]
        won_mean = sum(won) / len(won)
        lo, hi = cp(k, n)
        assert (b["n_days"], b["n_no_hit"]) == (n, k), b
        assert b["no_hit_rate"] == pytest.approx(k / n)
        assert b["no_hit_lower"] == pytest.approx(lo, abs=1e-9)
        assert b["no_hit_upper"] == pytest.approx(hi, abs=1e-9)
        assert b["won_mean_value"] == pytest.approx(won_mean, rel=1e-9)
        assert b["break_even_no_hit_rate"] == pytest.approx(max(0.0, 1 - 1 / won_mean), abs=1e-9)
        assert b["adjusted_ev"] == pytest.approx((1 - k / n) * won_mean, rel=1e-9)
        assert b["adjusted_ev"] == pytest.approx(sum(values) / n, rel=1e-9)  # #135's mean
        assert b["adjusted_ev_at_upper"] == pytest.approx((1 - hi) * won_mean, rel=1e-9)
        assert b["adjusted_ev_pooled"] == pytest.approx(
            (1 - N_NO_HIT / N_DAYS) * won_mean, rel=1e-9
        )
        assert b["adjusted_ev_pooled_at_upper"] == pytest.approx((1 - hi_all) * won_mean, rel=1e-9)


def test_the_135_bin_with_no_rollover_in_the_sample_is_not_safe_at_the_interval_bound(tmp_path):
    b = score(history(), SPEC, frozen(tmp_path))["bins"][2]
    assert b["n_days"] == 5 and b["n_no_hit"] == 0 and b["no_hit_rate"] == 0.0
    assert b["won_mean_value"] == pytest.approx(1.06)
    assert b["break_even_no_hit_rate"] == pytest.approx(1 - 1 / 1.06)  # about 5.7%
    assert b["adjusted_ev"] == pytest.approx(1.06)  # the sample's EV, as #135 scored it
    assert b["no_hit_lower"] == 0.0
    assert b["no_hit_upper"] == pytest.approx(1 - 0.025 ** (1 / 5), abs=1e-9)
    assert b["no_hit_upper"] > b["break_even_no_hit_rate"]  # the sample cannot rule it out
    assert b["adjusted_ev_at_upper"] < 1.0 < b["adjusted_ev"]


def test_a_won_value_that_cannot_beat_one_has_a_zero_break_even_rate(tmp_path):
    first = score(history(), SPEC, frozen(tmp_path))["bins"][0]
    assert first["won_mean_value"] == pytest.approx((0.7 + 0.75 + 0.72) / 3)
    assert first["break_even_no_hit_rate"] == 0.0
    assert first["adjusted_ev"] < 1.0


def test_a_higher_confidence_widens_every_bins_interval_in_the_score(tmp_path):
    narrow_spec, wide_spec = dict(SPEC, confidence=0.8), dict(SPEC, confidence=0.99)
    narrow = score(history(), narrow_spec, _fz(tmp_path / "a", narrow_spec))
    wide = score(history(), wide_spec, _fz(tmp_path / "b", wide_spec))
    for n_b, w_b in zip(narrow["bins"], wide["bins"], strict=True):
        assert n_b["no_hit_rate"] == w_b["no_hit_rate"]
        assert (
            n_b["no_hit_upper"] <= w_b["no_hit_upper"]
            and n_b["no_hit_lower"] >= w_b["no_hit_lower"]
        )
    assert narrow["bins"][2]["no_hit_upper"] == pytest.approx(1 - 0.1 ** (1 / 5), abs=1e-9)
    assert wide["bins"][2]["no_hit_upper"] == pytest.approx(1 - 0.005 ** (1 / 5), abs=1e-9)


def _fz(directory: Path, spec: dict) -> Registry:
    directory.mkdir()
    return frozen(directory, spec)


def test_the_bins_follow_the_specs_edges(tmp_path):
    spec = dict(SPEC, ratio_edges=[0, 1])
    got = score(history(), spec, _fz(tmp_path / "e", spec))
    assert [(b["lo_ratio"], b["hi_ratio"]) for b in got["bins"]] == [(0, 1), (1, None)]
    assert [(b["n_days"], b["n_no_hit"]) for b in got["bins"]] == [(8, 3), (9, 1)]


def test_a_capped_win_is_a_hit_and_a_bin_with_only_no_hit_days_has_no_won_figures(tmp_path):
    capped = valued(date(2017, 1, 1), C_BIN[3], 0.0)  # filled below: a hit that passes money on
    half = P // 2
    passed_on = C_BIN[3] + P * 7 // 10 - half
    capped.update(winning_tickets=1, payout_yen=half, dividend_yen=half,
                  carryover_out_yen=passed_on, winning_combinations=1)  # fmt: skip
    rows = [valued(d, C_BIN[1], 0.0) for d in sundays(3)] + [capped]
    got = score(rows, SPEC, frozen(tmp_path))
    only_no_hit, empty, capped_bin = got["bins"][1], got["bins"][2], got["bins"][3]
    assert (only_no_hit["n_days"], only_no_hit["n_no_hit"], only_no_hit["no_hit_rate"]) == (
        3,
        3,
        1.0,
    )
    assert only_no_hit["no_hit_upper"] == 1.0 and only_no_hit["no_hit_lower"] > 0
    for key in ("won_mean_value", "break_even_no_hit_rate", "adjusted_ev", "adjusted_ev_at_upper",
                "adjusted_ev_pooled", "adjusted_ev_pooled_at_upper"):  # fmt: skip
        assert only_no_hit[key] is None, key
    assert (capped_bin["n_days"], capped_bin["n_no_hit"]) == (1, 0)  # paid: not a no-hit day
    assert got["overall"]["n_days"] == 4 and got["overall"]["n_no_hit"] == 3
    assert empty["n_days"] == empty["n_no_hit"] == 0
    for key in ("no_hit_rate", "no_hit_lower", "no_hit_upper", "won_mean_value",
                "break_even_no_hit_rate", "adjusted_ev", "adjusted_ev_at_upper"):  # fmt: skip
        assert empty[key] is None, key


def test_holdout_days_change_no_figure(tmp_path):
    base = score(history(), SPEC, _fz(tmp_path / "base", SPEC))
    holdout = [valued(HOLDOUT_A, C_BIN[2], 0.0), valued(HOLDOUT_B, 0, 0.0)]
    more = score(history(holdout), SPEC, _fz(tmp_path / "h", SPEC))
    assert more["holdout_days_excluded"] == 2 and base["holdout_days_excluded"] == 0
    assert more["overall"] == base["overall"] and more["bins"] == base["bins"]


@pytest.mark.parametrize("change", [
    {"interval": "wilson"}, {"interval": None}, {"confidence": 1.0}, {"confidence": 0.0},
    {"confidence": "0.95"}, {"ratio_edges": []}, {"ratio_edges": [0, 1, 1]},
    {"ratio_edges": [2, 1]}, {"ratio_edges": [-1, 1]}, {"break_even_takeout": 0.0},
    {"break_even_takeout": 1.0}, {"pool": "gross"}, {"payout_cap_yen": 0}])  # fmt: skip
def test_a_frozen_spec_the_scorer_cannot_score_is_refused(tmp_path, change):
    spec = dict(SPEC, **change)
    with pytest.raises(ValueError):
        ro.check_rollover_spec(spec)
    registry = _fz(tmp_path / "bad", spec)
    with pytest.raises(ValueError):
        score(history(), spec, registry)


def test_a_scorable_spec_passes_the_check():
    ro.check_rollover_spec(SPEC)
    ro.check_rollover_spec(dict(SPEC, confidence=0.9, ratio_edges=[0, 1, 3]))


# ---------------------------------------------------------------- D: shipped spec, command


def shipped_spec() -> dict:
    return dict(load_country_yaml("japan", "win5_rollover"))


def test_the_shipped_spec_is_its_own_family_on_the_same_bins_as_135():
    spec = shipped_spec()
    ro.check_rollover_spec(spec)
    assert isinstance(spec["family_id"], str) and spec["family_id"].strip()
    assert spec["family_id"] != load_country_yaml("japan", "win5_study")["family_id"]
    assert spec["interval"] == "clopper-pearson" and spec["confidence"] == 0.95
    assert spec["ratio_edges"] == load_country_yaml("japan", "win5_study")["ratio_edges"]
    assert spec["break_even_takeout"] == 0.30
    json.dumps(spec, allow_nan=False)  # plain JSON, so it can be frozen


def registry_at(data_dir: Path, tmp_path: Path) -> Registry:
    return Registry(data_dir / "registry" / "japan.jsonl", holdout_log=tmp_path / "h.jsonl")


def test_the_command_refuses_before_the_freeze_then_freezes_and_scores(data_dir, tmp_path):
    spec = shipped_spec()
    family = spec["family_id"]
    win5.write_history(win5.build_history(history([valued(HOLDOUT_A, C_BIN[2], 0.0)])))
    registry = registry_at(data_dir, tmp_path)
    runner = CliRunner()

    early = runner.invoke(app, ["japan", "win5-rollover", "score"])
    assert early.exit_code != 0, early.output
    assert registry.frozen_spec_hash(family) is None
    assert [r for r in registry.records(family) if r.status == "scored"] == []

    froze = runner.invoke(app, ["japan", "win5-rollover", "freeze"])
    assert froze.exit_code == 0, froze.output
    assert registry.frozen_spec_hash(family) == spec_hash(spec)

    scored = runner.invoke(app, ["japan", "win5-rollover", "score"])
    assert scored.exit_code == 0, scored.output
    (run,) = [r for r in registry.records(family) if r.status == "scored"]
    assert run.ticket == 323 and run.family_id == family
    assert re.fullmatch(r"P[0-9]+", run.hypothesis_id), run.hypothesis_id
    assert {m.metric for m in run.metrics} == {"no_hit_rate"}
    by_n = {m.n: m for m in run.metrics}
    assert len(run.metrics) == 5 and set(by_n) == {N_DAYS, 4, 5}  # all days, the bins (4, 4, 5, 4)
    all_days = by_n[N_DAYS]
    assert all_days.value == pytest.approx(N_NO_HIT / N_DAYS)
    assert (all_days.lo, all_days.hi) == (
        pytest.approx(cp(N_NO_HIT, N_DAYS)[0], abs=1e-9),
        pytest.approx(cp(N_NO_HIT, N_DAYS)[1], abs=1e-9),
    )
    five = by_n[5]  # the [1, 2) bin: 0 of 5
    assert five.value == 0.0 and five.lo == 0.0
    assert five.hi == pytest.approx(1 - 0.025 ** (1 / 5), abs=1e-9)
    registry.verify()
    for line in scored.output.splitlines():  # aggregates only, nothing licensed to print
        assert "race_date" not in line


def test_a_scoring_run_that_fails_after_the_freeze_is_recorded(data_dir, tmp_path):
    family = shipped_spec()["family_id"]
    registry = registry_at(data_dir, tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["japan", "win5-rollover", "freeze"]).exit_code == 0

    failed = runner.invoke(app, ["japan", "win5-rollover", "score"])  # no history table written

    assert failed.exit_code != 0, failed.output
    assert [r.status for r in registry.records(family)] == ["failed"]
