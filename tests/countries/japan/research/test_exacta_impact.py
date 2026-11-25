"""The ¥100 exacta own-impact check (ticket #283): the payout identity on logged FORECAST pools.

#131 assumed that ¥100 per race has negligible own price impact on the exacta pool. Contract for
`horsetrack.countries.japan.research.exacta_impact` (planned):

- Payout identity, before breakage: dividend per ¥100 = 100 (1 - t) P / W, P the pool's money and
  W the money on the combination. Adding a stake s to each of n combinations (the winner among
  them) gives 100 (1 - t) (P + n s) / (W + s). The relative own impact is
  (P + n s) / P x W / (W + s) - 1.
- `stake_on_combination(pool, odds, takeout=TAKEOUT) -> float`: W = (1 - t) P / O, with O the
  decimal odds (dividend per ¥1).
- `own_impact(pool, odds, stake=STAKE, n_combinations=1, takeout=TAKEOUT) -> float`: that relative
  change (negative: the dividend falls). ValueError for pool <= 0, odds <= 0, n_combinations < 1,
  a takeout outside [0, 1) or a non-finite figure. `TAKEOUT = 0.25`, `STAKE = 100`.
- `POOL_BIN_EDGES`: ascending pool-size edges in yen, the first 0; a bin is [lo, hi) and the last
  is open (hi None).
- `summarize_impact(*, stake=STAKE, takeout=TAKEOUT, bin_edges=POOL_BIN_EDGES) -> dict` reads
  `$HORSETRACK_DATA_DIR/odds_snapshots/jra/<day>/*.jsonl.gz` of development days only (a holdout
  day, `is_holdout_day("japan", d)`, is never opened). For each race with a FORECAST `__pool__`
  investment (latest, `odds_log.latest_totals`) it takes the latest FORECAST prices
  (`odds_log.latest_prices`), skipping a null price (scratched), and computes `own_impact` of
  `stake` on each combination alone at its latest odds. Result:
  {days_read, days_with_forecast, days_holdout_excluded, races, races_skipped_no_pool, bins};
  `days_read` counts the development day folders read, `days_with_forecast` those with a FORECAST
  row, `races` the races used, `races_skipped_no_pool` the races with FORECAST prices and no
  FORECAST pool total. `bins` holds one dict per bin, in edge order, empty ones too:
  {lo, hi, n_races, n_combinations, median_abs_impact, max_abs_impact} (the last two None for an
  empty bin; the median is `statistics.median` of |impact| over the bin's combinations).
- `horsetrack japan exacta-impact`: runs the summary over the data dir, prints it, and appends a
  `default_registry("japan")` record (hypothesis `P<n>`, ticket 283, status "scored") with one
  FoldMetric per non-empty bin: n = n_races, value = median_abs_impact, hi = max_abs_impact.
  The same run again exits 0 and adds nothing. No days: exit 0, a record with no metrics.

Every figure here is invented; HORSETRACK_DATA_DIR points at tmp_path.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import math
import re
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.research import exacta_impact as ei

UTC = timezone.utc
POOL = odds_log.POOL_TOTAL


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


# ---------------------------------------------------------------- the identity, on hand figures


def reference(pool, odds, stake=100, n=1, takeout=0.25):
    """The relative dividend change straight from the payout identity (dividends compared)."""
    w = (1 - takeout) * pool / odds
    before = 100 * (1 - takeout) * pool / w
    after = 100 * (1 - takeout) * (pool + n * stake) / (w + stake)
    return after / before - 1


# (pool, odds, stake, n, takeout, impact) with the impact worked by hand from the identity
HAND = {
    "typical": (1_000_000, 10.0, 100, 1, 0.25, -0.001231691078561914),
    "box of six": (1_000_000, 10.0, 100, 6, 0.25, -0.0007323568575232642),
    "small pool, long odds": (20_000, 200.0, 100, 1, 0.25, -0.5692857142857144),
    "big pool": (1_000_000_000, 10.0, 100, 1, 0.25, -1.2333316888879509e-06),
    "short odds, wide box": (1_000_000, 1.5, 100, 10, 0.25, 0.0007998400319935151),
    "takeout 0.20": (1_000_000, 10.0, 100, 1, 0.20, -0.0011485642946317265),
    "stake 1000": (1_000_000, 10.0, 1000, 1, 0.25, -0.012171052631579138),
}


@pytest.mark.parametrize("pool, odds, stake, n, takeout, want", HAND.values(), ids=HAND.keys())
def test_own_impact_matches_the_hand_worked_identity(pool, odds, stake, n, takeout, want):
    got = ei.own_impact(pool, odds, stake=stake, n_combinations=n, takeout=takeout)
    assert got == pytest.approx(want, rel=1e-9)
    assert got == pytest.approx(reference(pool, odds, stake, n, takeout), rel=1e-9)


def test_defaults_are_a_hundred_yen_one_combination_and_the_exacta_takeout():
    assert ei.TAKEOUT == 0.25 and ei.STAKE == 100
    assert ei.own_impact(1_000_000, 10.0) == pytest.approx(HAND["typical"][5], rel=1e-9)


@pytest.mark.parametrize(
    "pool, odds, takeout, want",
    [
        (1_000_000, 10.0, 0.25, 75_000.0),
        (1_000_000, 10.0, 0.20, 80_000.0),
        (20_000, 200.0, 0.25, 75.0),
        (4_000_000, 2.0, 0.25, 1_500_000.0),
    ],
)
def test_stake_on_a_combination_is_the_payout_identity_solved_for_w(pool, odds, takeout, want):
    assert ei.stake_on_combination(pool, odds, takeout=takeout) == pytest.approx(want)
    if takeout == 0.25:
        assert ei.stake_on_combination(pool, odds) == pytest.approx(want)


def test_the_impact_shrinks_with_pool_size_and_grows_with_the_odds():
    pools = [20_000, 200_000, 2_000_000, 20_000_000, 200_000_000]
    at_30 = [abs(ei.own_impact(p, 30.0)) for p in pools]
    assert at_30 == sorted(at_30, reverse=True) and at_30[0] > 100 * at_30[-1]
    odds = [2.0, 10.0, 50.0, 300.0, 1000.0]
    at_pool = [abs(ei.own_impact(2_000_000, o)) for o in odds]
    assert at_pool == sorted(at_pool) and at_pool[0] < at_pool[-1]


def test_a_single_combination_always_lowers_its_own_dividend():
    for pool in (50_000, 1_000_000, 80_000_000):
        for odds in (1.0, 3.5, 40.0, 900.0):
            assert ei.own_impact(pool, odds) < 0


def test_a_zero_stake_changes_nothing():
    assert ei.own_impact(1_000_000, 10.0, stake=0) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pool": 0, "odds": 10.0},
        {"pool": -5_000, "odds": 10.0},
        {"pool": 1_000_000, "odds": 0},
        {"pool": 1_000_000, "odds": -3.0},
        {"pool": float("nan"), "odds": 10.0},
        {"pool": 1_000_000, "odds": float("inf")},
        {"pool": 1_000_000, "odds": 10.0, "n_combinations": 0},
        {"pool": 1_000_000, "odds": 10.0, "takeout": 1.0},
        {"pool": 1_000_000, "odds": 10.0, "takeout": -0.1},
    ],
)
def test_invalid_inputs_are_refused(kwargs):
    with pytest.raises(ValueError):
        ei.own_impact(**kwargs)


@pytest.mark.parametrize("pool, odds", [(0, 10.0), (-1.0, 10.0), (1_000_000, 0), (1_000_000, -2.0)])
def test_stake_on_a_combination_refuses_invalid_inputs(pool, odds):
    with pytest.raises(ValueError):
        ei.stake_on_combination(pool, odds)


# ---------------------------------------------------------------- synthetic logs

EDGES = (0, 1_000_000, 10_000_000)  # bins [0, 1M), [1M, 10M), [10M, inf)
DEV_DAYS = [date(2026, 11, 21), date(2026, 11, 22)]


def holdout_and_dev_after_start() -> tuple[date, date]:
    holdout = dev = None
    d = date(2026, 11, 28)
    while holdout is None or dev is None:
        if is_holdout_day("japan", d):
            holdout = holdout or d
        else:
            dev = dev or d
        d += timedelta(days=1)
    return holdout, dev


HOLDOUT_DAY, DEV_AFTER = holdout_and_dev_after_start()


def at(day: date, hh: int, mm: int) -> str:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=UTC).isoformat()


def rid(day: date, venue: str, no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{no:02d}"


def price(day, venue, no, combo, odds, hh, mm, pool="FORECAST") -> dict:
    return {
        "snapshot_utc": at(day, hh, mm),
        "race_id": rid(day, venue, no),
        "pool": pool,
        "combination": combo,
        "odds": odds,
        "status_raw": None,
    }


def total(day, venue, no, investment, hh, mm, pool="FORECAST") -> dict:
    return {
        "snapshot_utc": at(day, hh, mm),
        "race_id": rid(day, venue, no),
        "pool": pool,
        "combination": POOL,
        "odds": None,
        "status_raw": None,
        "investment": investment,
    }


def write(data: Path, day: date, venue: str, members: list[list[dict]]) -> Path:
    log = data / "odds_snapshots" / "jra" / day.isoformat() / f"{venue}.jsonl.gz"
    for rows in members:
        odds_log.append_capture(log, rows)
    return log


def write_day(data: Path, day: date, venue: str, races: dict[int, tuple[float | None, dict]]):
    """One member per race at 04:30 UTC (latest) after an earlier, different member at 03:30."""
    early, late = [], []
    for no, (pool, combos) in races.items():
        for combo, odds in combos.items():
            if odds is not None:
                early.append(price(day, venue, no, combo, odds * 3 + 1, 3, 30))  # to be superseded
            late.append(price(day, venue, no, combo, odds, 4, 30))
        if pool is not None:
            early.append(total(day, venue, no, pool / 7, 3, 30))
            late.append(total(day, venue, no, pool, 4, 30))
    write(data, day, venue, [early, late])


# the races: pool in yen -> combinations at their latest odds (None: scratched)
RACES_D1_TOKYO = {
    1: (500_000, {"1,2": 20.0, "2,1": 50.0, "3,1": None}),  # bin 0
    2: (5_000_000, {"1,2": 5.0, "2,1": 100.0, "4,5": 300.0}),  # bin 1
    3: (1_000_000, {"6,7": 40.0}),  # exactly on an edge: bin 1
    4: (None, {"1,2": 12.0, "2,1": 30.0}),  # prices but no pool total: skipped
}
RACES_D1_KYOTO = {
    1: (50_000_000, {"1,2": 3.0, "2,1": 12.0}),  # bin 2
    2: (250_000, {"8,9": 700.0, "9,8": 60.0}),  # bin 0
}
RACES_D2_TOKYO = {1: (12_000_000, {"2,3": 8.0, "3,2": 9.5, "1,3": 25.0})}  # bin 2


def build_data(data: Path) -> None:
    write_day(data, DEV_DAYS[0], "TOKYO", RACES_D1_TOKYO)
    write_day(data, DEV_DAYS[0], "KYOTO", RACES_D1_KYOTO)
    write_day(data, DEV_DAYS[1], "TOKYO", RACES_D2_TOKYO)
    # tier-1 rows and another pool's total must not count as exacta
    write(
        data,
        DEV_DAYS[1],
        "KYOTO",
        [[price(DEV_DAYS[1], "KYOTO", 1, "1", 3.0, 4, 30, pool="WIN"),
          total(DEV_DAYS[1], "KYOTO", 1, 9_999_999_999, 4, 30, pool="WIN")]],
    )  # fmt: skip


ALL_RACES = {
    **{(DEV_DAYS[0], "TOKYO", n): v for n, v in RACES_D1_TOKYO.items()},
    **{(DEV_DAYS[0], "KYOTO", n): v for n, v in RACES_D1_KYOTO.items()},
    **{(DEV_DAYS[1], "TOKYO", n): v for n, v in RACES_D2_TOKYO.items()},
}


def expected_bins(edges, stake=100, takeout=0.25) -> list[dict]:
    """The by-pool-size summary worked independently of the module, from ALL_RACES."""
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        impacts, races = [], 0
        for pool, combos in ALL_RACES.values():
            if pool is None or pool < lo or (hi is not None and pool >= hi):
                continue
            races += 1
            impacts += [
                abs(reference(pool, o, stake, 1, takeout)) for o in combos.values() if o is not None
            ]
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n_races": races,
                "n_combinations": len(impacts),
                "median_abs_impact": statistics.median(impacts) if impacts else None,
                "max_abs_impact": max(impacts) if impacts else None,
            }
        )
    return out


def assert_bins(got: list[dict], want: list[dict]) -> None:
    assert len(got) == len(want), got
    for g, w in zip(got, want, strict=True):
        assert (g["lo"], g["hi"]) == (w["lo"], w["hi"])
        assert g["n_races"] == w["n_races"] and g["n_combinations"] == w["n_combinations"], (g, w)
        for key in ("median_abs_impact", "max_abs_impact"):
            if w[key] is None:
                assert g[key] is None, (g, w)
            else:
                assert g[key] == pytest.approx(w[key], rel=1e-9), (g, w, key)


# ---------------------------------------------------------------- the summary


def test_the_summary_groups_logged_exacta_races_by_pool_size(data_dir):
    build_data(data_dir)

    result = ei.summarize_impact(bin_edges=EDGES)

    assert result["days_read"] == 2
    assert result["days_with_forecast"] == 2
    assert result["days_holdout_excluded"] == 0
    assert result["races"] == 6
    assert result["races_skipped_no_pool"] == 1
    want = expected_bins(EDGES)
    assert [b["n_races"] for b in want] == [2, 2, 2]  # the spread the test relies on
    assert_bins(result["bins"], want)
    # the small pools are hit hardest, the big ones barely: the figure moves with the bin
    medians = [b["median_abs_impact"] for b in result["bins"]]
    assert medians == sorted(medians, reverse=True) and medians[0] > 10 * medians[-1]


@pytest.mark.parametrize("stake", [100, 500])
def test_the_stake_and_takeout_arguments_move_the_summary(data_dir, stake):
    build_data(data_dir)
    assert_bins(
        ei.summarize_impact(stake=stake, bin_edges=EDGES), expected_bins(EDGES, stake=stake)
    )
    assert_bins(
        ei.summarize_impact(takeout=0.20, bin_edges=EDGES), expected_bins(EDGES, takeout=0.20)
    )


def test_other_bin_edges_regroup_the_same_races(data_dir):
    build_data(data_dir)
    edges = (0, 300_000, 6_000_000, 100_000_000)
    result = ei.summarize_impact(bin_edges=edges)
    assert_bins(result["bins"], expected_bins(edges))
    assert result["races"] == 6


def test_a_pool_exactly_on_an_edge_falls_in_the_upper_bin(data_dir):
    build_data(data_dir)  # DEV_DAYS[0] TOKYO race 3 has a pool of exactly 1,000,000
    result = ei.summarize_impact(bin_edges=(0, 1_000_000, 1_000_001))
    assert [b["n_races"] for b in result["bins"]] == [2, 1, 3]


def test_the_latest_capture_decides_the_odds_and_the_pool(data_dir):
    day = DEV_DAYS[0]
    write(
        data_dir,
        day,
        "TOKYO",
        [
            [price(day, "TOKYO", 1, "1,2", 500.0, 3, 0), total(day, "TOKYO", 1, 200_000, 3, 0)],
            [price(day, "TOKYO", 1, "1,2", 10.0, 5, 0), total(day, "TOKYO", 1, 2_000_000, 5, 0)],
            [price(day, "TOKYO", 1, "1,2", 80.0, 2, 0), total(day, "TOKYO", 1, 20_000, 2, 0)],
        ],
    )  # logged last, stamped earliest: the 05:00 capture is the latest
    result = ei.summarize_impact(bin_edges=EDGES)
    only = [b for b in result["bins"] if b["n_races"]]
    assert len(only) == 1 and (only[0]["lo"], only[0]["hi"]) == (1_000_000, 10_000_000)
    assert only[0]["max_abs_impact"] == pytest.approx(abs(reference(2_000_000, 10.0)), rel=1e-9)


def test_a_race_without_an_investment_figure_is_skipped_and_counted(data_dir):
    day = DEV_DAYS[0]
    no_figure = total(day, "TOKYO", 2, None, 4, 30)
    write(
        data_dir,
        day,
        "TOKYO",
        [[price(day, "TOKYO", 1, "1,2", 9.0, 4, 30), total(day, "TOKYO", 1, 3_000_000, 4, 30),
          price(day, "TOKYO", 2, "1,2", 9.0, 4, 30), no_figure]],
    )  # fmt: skip
    result = ei.summarize_impact(bin_edges=EDGES)
    assert result["races"] == 1 and result["races_skipped_no_pool"] == 1


# ---------------------------------------------------------------- holdout days, no days


def test_a_holdout_day_is_never_read_and_a_development_day_after_the_start_is(
    data_dir, monkeypatch
):
    races = {1: (3_000_000, {"1,2": 15.0, "2,1": 45.0})}
    write_day(data_dir, DEV_AFTER, "TOKYO", races)
    holdout_log = write(
        data_dir,
        HOLDOUT_DAY,
        "TOKYO",
        [[price(HOLDOUT_DAY, "TOKYO", 1, "1,2", 2.0, 4, 30),
          total(HOLDOUT_DAY, "TOKYO", 1, 90_000_000, 4, 30)]],
    )  # fmt: skip
    holdout_log.chmod(0)  # opening it would raise
    opened: list[Path] = []
    for name in ("read_rows", "iter_members"):
        real = getattr(odds_log, name)

        def spy(path, *a, _real=real, **k):
            opened.append(Path(path))
            return _real(path, *a, **k)

        monkeypatch.setattr(odds_log, name, spy)
    try:
        result = ei.summarize_impact(bin_edges=EDGES)
    finally:
        holdout_log.chmod(0o600)

    assert result["days_holdout_excluded"] == 1
    assert result["days_read"] == 1 and result["races"] == 1
    assert holdout_log not in opened
    assert_bins(
        result["bins"],
        [
            {"lo": 0, "hi": 1_000_000, "n_races": 0, "n_combinations": 0,
             "median_abs_impact": None, "max_abs_impact": None},
            {"lo": 1_000_000, "hi": 10_000_000, "n_races": 1, "n_combinations": 2,
             "median_abs_impact": statistics.median(
                 [abs(reference(3_000_000, 15.0)), abs(reference(3_000_000, 45.0))]),
             "max_abs_impact": abs(reference(3_000_000, 45.0))},
            {"lo": 10_000_000, "hi": None, "n_races": 0, "n_combinations": 0,
             "median_abs_impact": None, "max_abs_impact": None},
        ],
    )  # fmt: skip


@pytest.mark.parametrize("layout", ["no odds_snapshots folder", "a day with tier-1 rows only"])
def test_no_exacta_days_gives_an_empty_well_formed_summary(data_dir, layout):
    if layout != "no odds_snapshots folder":
        day = DEV_DAYS[0]
        win_rows = [
            price(day, "TOKYO", 1, "1", 3.0, 4, 30, pool="WIN"),
            total(day, "TOKYO", 1, 8_000_000, 4, 30, pool="WIN"),
        ]
        write(data_dir, day, "TOKYO", [win_rows])
    result = ei.summarize_impact(bin_edges=EDGES)
    assert result["days_with_forecast"] == 0 and result["races"] == 0
    assert result["races_skipped_no_pool"] == 0 and result["days_holdout_excluded"] == 0
    assert result["days_read"] == (0 if layout == "no odds_snapshots folder" else 1)
    assert [(b["lo"], b["hi"], b["n_races"], b["n_combinations"]) for b in result["bins"]] == [
        (0, 1_000_000, 0, 0),
        (1_000_000, 10_000_000, 0, 0),
        (10_000_000, None, 0, 0),
    ]
    assert all(
        b["median_abs_impact"] is None and b["max_abs_impact"] is None for b in result["bins"]
    )


def test_the_default_bins_are_ascending_from_zero_and_cover_every_pool(data_dir):
    edges = ei.POOL_BIN_EDGES
    assert edges[0] == 0 and list(edges) == sorted(set(edges)) and len(edges) >= 3
    build_data(data_dir)
    result = ei.summarize_impact()
    assert len(result["bins"]) == len(edges)
    assert sum(b["n_races"] for b in result["bins"]) == result["races"] == 6


# ---------------------------------------------------------------- the command


def registry_at(data_dir: Path, tmp_path: Path) -> Registry:
    return Registry(data_dir / "registry" / "japan.jsonl", holdout_log=tmp_path / "h.jsonl")


def test_the_command_records_the_summary_and_a_repeat_adds_nothing(data_dir, tmp_path):
    build_data(data_dir)
    registry = registry_at(data_dir, tmp_path)
    runner = CliRunner()

    first = runner.invoke(app, ["japan", "exacta-impact"])

    assert first.exit_code == 0, first.output
    runs = registry.records()
    assert len(runs) == 1
    run = runs[0]
    assert re.fullmatch(r"P[0-9]+", run.hypothesis_id) and run.ticket == 283
    assert run.status == "scored"
    want = [b for b in expected_bins(tuple(ei.POOL_BIN_EDGES)) if b["n_races"]]
    assert len(run.metrics) == len(want) >= 2
    for metric, w in zip(sorted(run.metrics, key=lambda m: m.value, reverse=True),
                         sorted(want, key=lambda b: b["median_abs_impact"], reverse=True),
                         strict=True):  # fmt: skip
        assert metric.n == w["n_races"]
        assert math.isclose(metric.value, w["median_abs_impact"], rel_tol=1e-9)
        assert metric.hi is not None and math.isclose(metric.hi, w["max_abs_impact"], rel_tol=1e-9)
    registry.verify()

    again = runner.invoke(app, ["japan", "exacta-impact"])
    assert again.exit_code == 0, again.output
    assert len(registry.records()) == 1


def test_the_command_with_no_exacta_days_exits_cleanly_and_records_no_metrics(data_dir, tmp_path):
    registry = registry_at(data_dir, tmp_path)

    result = CliRunner().invoke(app, ["japan", "exacta-impact"])

    assert result.exit_code == 0, result.output
    runs = registry.records()
    assert len(runs) == 1 and runs[0].metrics == () and runs[0].ticket == 283
    assert runs[0].status == "scored"
