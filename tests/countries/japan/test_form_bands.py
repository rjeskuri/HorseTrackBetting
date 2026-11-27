"""#218: does the meld's negative weight on form hold within T-2 odds bands and favourite ranks?

Module under test: `horsetrack.countries.japan.studies.form_bands`. The approved pre-registration
(owner, 2026-11-25, T-2 correction; reframed 2026-11-27) is in the ticket. Contract pinned here:

Constants
- `VARIANTS == ("P0", "B1", "B2", "B3", "B4", "B5", "R1", "R2", "R3", "R4", "BA", "RA")`.
- `FAMILY_ID` (str), `TICKET == 218`, `BOOTSTRAP_SEED == 218`.

The T-2 board
- `t2_win_odds(day, *, minutes_before=2.0) -> pd.DataFrame`: `day` is one JRA-VAN odds_timeseries
  day (`logger.compact.SCHEMA` columns). One row per runner on each race's Win board at T-2, read as
  `studies.cross_pool.boards_at` reads it (rows with minutes_to_post >= minutes_before, the latest
  per runner; scratched and unpriced runners leave the board): columns `race_id`, `horse_no` (int)
  and `win_odds_t2` (float).
- `assign_groups(board) -> pd.DataFrame`: `board` has `race_id` and `win_odds_t2` (NaN: no T-2
  price) and any other columns, which are kept, as are the index and row order. Adds `band` (1..5:
  [1, 3), [3, 7), [7, 15), [15, 50), [50, inf), left-closed), `fav_rank` (1 + the number of runners
  of the same race priced strictly shorter: a tie shares the better rank) and `rank_group` (1: rank
  1, 2: ranks 2-3, 3: ranks 4-6, 4: rank 7+). Ranks are within each race, over every row of that
  race with a T-2 price. A row without a T-2 price has all three missing (`pd.isna`).

The meld
- `design(rows, variant) -> pd.DataFrame`: the ln h columns of `variant`, indexed like `rows`
  (which has `h`, `band`, `rank_group`). Column names: P0 `["all"]`; Bj `[f"band{j}"]`; Rj
  `[f"rank{j}"]`; BA `band1..band5`; RA `rank1..rank4`. Column g is 1[runner in g] * ln h, exactly
  0.0 outside g.
- `fit_meld(rows, variant) -> MeldFit`: the unpenalised conditional-logit MLE of
  u = a * ln p_cal + design(rows, variant) @ b within race (`models.condlogit`), rows with
  `race_id`, `won`, `p_cal`, `h`, `band`, `rank_group`. `MeldFit` has `a` (float), `b` (dict:
  design column name -> float) and `n_races` (int), and can be built as
  `MeldFit(a=..., b=..., n_races=...)` (any further field has a default).
- `score_races(rows, variant, fit) -> pd.DataFrame`: rows as for `fit_meld` plus `meeting_id` and
  `scratched`. Scratched rows are removed and p_cal renormalised over the starters; the meld is the
  softmax of u over the starters. One row per race with a winner: `race_id`, `meeting_id`, `ll_v0`
  and `ll_variant` (-(1/k) * sum over the k winners of ln p; D6, #69) and `gain = ll_v0 -
  ll_variant`.
- `walk_forward_meld(rows, folds, variant) -> MeldRun`: rows as for `score_races` plus
  `race_date` (datetime.date), `fold` (an index into `folds`) and `win_odds_t2`, with `band` and
  `rank_group` from `assign_groups`. A race whose starter (scratched False) has no T-2 price is
  dropped from fit and score and counted in `n_unpriced_dropped`; a race with no winner is dropped
  and counted in `n_void_dropped`. Fold k is fitted on the rows with fold < k and race_date <=
  folds[k].train_end and scores fold k's races; a fold with no such training rows (the earliest
  fold present) only trains. `MeldRun.per_race` is `score_races`' frame plus `fold`; `fits` maps
  each scored fold k to its `MeldFit`.
- `gain_interval(per_race, *, n_boot, seed) -> GainInterval`: per_race has `race_id`,
  `meeting_id`, `gain`. `mean` is the per-race mean gain, `lower` the one-sided 95% lower bound
  and `ci95` the two-sided 95% interval (lo, hi), both from D7's meeting bootstrap
  (`core.eval.bootstrap.paired_bootstrap` with the meeting as cluster and `seed`: `lower` at level
  0.95, `ci95` the (lo, hi) at level 0.975); `n_meetings` the meetings resampled.

The run
- `run_family(rows, folds, *, registry, out_dir, n_boot=2000) -> dict`: rows as for
  `walk_forward_meld` without the group columns (it calls `assign_groups`). It refuses rows holding
  a holdout day (`core.eval.holdout.is_holdout_day`) with a ValueError before anything is
  written; freezes the family spec with `registry.freeze_family(FAMILY_ID, spec)` before any
  `append` (another frozen spec: `FamilyAlreadyFrozen` and no record); scores the 12 variants
  (a variant that cannot be fitted, e.g. a group with no runner, is a `failed` record, never
  dropped); appends one record per variant with a JRA-VAN source among its `sources`; runs SPA and
  StepM (`core.eval.snooping`, 5% FWE) over the 11 non-reference variants (not P0) against v0;
  and writes the report through `core.eval.reports.write_report` with the JRA-VAN source, so an
  `out_dir` inside the repository is refused. The result dict holds `family_id`, `spec_hash`,
  `report_path` (str), `records_appended`, `variants` (name -> {`status`: "scored" | "failed",
  and when scored `gain`, `lower`, `ci95`, `n_races`, `n_meetings` and `groups`: design column ->
  {`b_median`, `b_min`, `b_max` over the scored folds' fits, `runners` (the group's starters in
  the scored races) and `races` (scored races with one or more of them)}}), `spa` ({`pvalue`,
  `n_rules`}) and `stepm` ({`superior`: list of variant names}).

Added after the round-1 review of PR #473 (the family stays frozen; reporting and rerun rules only):
- Reruns on one registry: the family test runs over the current run's scored non-reference
  variants, whatever an earlier run recorded, so a variant that scored before and fails now is a
  failed record, the run finishes and its report is written. Each run writes its own private
  report (a run-specific name: an earlier report is never overwritten) and `run_ids` lists the
  16-character prefixes of the ids of the registry records holding this run's results: the new
  ones, or for an identical rerun the existing ones it reproduced.
- The private report has a section headed `## Primary question`: the per-group reading of R1-R4
  and B1-B5 and the sentence "the frozen spec defines no favourite-vs-longshot comparison rule",
  with the answer "inconclusive" (no rule is invented after the run).
- The report calls each gain the "gain of the registered meld over v0" and states that b_g
  confounds a group intercept with form ("group intercept") and that the family test credits the
  market refit as well ("market refit").

Rulings assumed here (reversible implementation choices):
- Ruling: favourite ranks count every runner priced on the T-2 board, including one scratched
  after T-2; a runner off the T-2 board has no rank.
- Ruling: fold k trains on rows of folds < k up to folds[k].train_end (#70's embargo), as
  `studies.same_day_bias`; the earliest fold holding rows only trains.
- Ruling: a variant whose group holds no runner is a failed registry record, not a b = 0 fit.
- Ruling: `ci95` is the 2.5% / 97.5% quantile pair of D7's bootstrap with the same seed.

Data: synthetic and in memory (tmp_path for the registry and report). Expected values come from
the stated formulas, `models.condlogit.fit_conditional_logit` (the reference fit on columns built
here) and `core.eval.bootstrap.paired_bootstrap`; never from the module under test.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from horsetrack.core import env, paths
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import FamilyAlreadyFrozen, Registry
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_withheld
from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.studies import form_bands as fb

VARIANTS = ("P0", "B1", "B2", "B3", "B4", "B5", "R1", "R2", "R3", "R4", "BA", "RA")
NON_REFERENCE = tuple(v for v in VARIANTS if v != "P0")
BAND_EDGES = (3.0, 7.0, 15.0, 50.0)
# variant -> its design columns, each (column name, "band" | "rank" | "all", group number)
COLUMNS = {
    "P0": [("all", "all", 0)],
    **{f"B{j}": [(f"band{j}", "band", j)] for j in range(1, 6)},
    **{f"R{j}": [(f"rank{j}", "rank", j)] for j in range(1, 5)},
    "BA": [(f"band{j}", "band", j) for j in range(1, 6)],
    "RA": [(f"rank{j}", "rank", j) for j in range(1, 5)],
}
MONTHLY = FoldScheme(
    unit="month", test_length=1, step=1, min_train=0, window="expanding", embargo_days=7,
    anchor_month=1,
)  # fmt: skip
N_BOOT = 200
COEF_TOL = 1e-6


# ---------------------------------------------------------------- references (first principles)


def ref_band(odds: float) -> int:
    return 1 + sum(odds >= edge for edge in BAND_EDGES)


def ref_rank_group(rank: int) -> int:
    return 1 if rank == 1 else 2 if rank <= 3 else 3 if rank <= 6 else 4


def ref_ranks(odds: list[float]) -> list[int]:
    return [1 + sum(other < mine for other in odds) for mine in odds]


def ref_columns(rows: pd.DataFrame, variant: str) -> np.ndarray:
    ln_h = np.log(rows["h"].to_numpy(dtype=float))
    out = []
    for _, kind, group in COLUMNS[variant]:
        if kind == "all":
            inside = np.ones(len(rows), dtype=bool)
        else:
            inside = rows["band" if kind == "band" else "rank_group"].to_numpy() == group
        out.append(np.where(inside, ln_h, 0.0))
    return np.column_stack(out)


def ref_fit(rows: pd.DataFrame, variant: str) -> dict[str, float]:
    """The reference meld fit: X = [ln p_cal, the variant's columns], no offset, no penalty."""
    X = np.column_stack([np.log(rows["p_cal"].to_numpy(dtype=float)), ref_columns(rows, variant)])
    fit = fit_conditional_logit(X, rows["race_id"].to_numpy(), rows["won"].to_numpy(dtype=bool))
    names = [name for name, _, _ in COLUMNS[variant]]
    return {
        "a": float(fit.coef[0]),
        **{n: float(c) for n, c in zip(names, fit.coef[1:], strict=True)},
    }


def ref_scores(rows: pd.DataFrame, variant: str, coef: dict[str, float]) -> pd.DataFrame:
    """Per race with a winner: ll_v0 and ll_variant over the starters, a dead heat at 1/k."""
    rows = rows[~rows["scratched"].astype(bool)]
    names = [name for name, _, _ in COLUMNS[variant]]
    race = rows["race_id"].to_numpy()
    p_cal = rows["p_cal"].to_numpy(dtype=float)
    p_cal = p_cal / pd.Series(p_cal).groupby(race).transform("sum").to_numpy()
    u = coef["a"] * np.log(p_cal) + ref_columns(rows, variant) @ np.array([coef[n] for n in names])
    e = np.exp(u)
    p = e / pd.Series(e).groupby(race).transform("sum").to_numpy()
    won = rows["won"].to_numpy(dtype=bool)
    frame = pd.DataFrame(
        {
            "race_id": race[won],
            "meeting_id": rows["meeting_id"].to_numpy()[won],
            "ll_v0": -np.log(p_cal[won]),
            "ll_variant": -np.log(p[won]),
        }
    )
    out = frame.groupby(["race_id", "meeting_id"], as_index=False, sort=True).mean()
    return out.assign(gain=out["ll_v0"] - out["ll_variant"])


def ref_train(rows: pd.DataFrame, folds, k: int) -> pd.DataFrame:
    dates = pd.to_datetime(rows["race_date"])
    return rows[(rows["fold"] < k) & (dates <= pd.Timestamp(folds[k].train_end))]


# ---------------------------------------------------------------- synthetic races


def synthetic_races(
    seed: int,
    *,
    weeks: int = 17,
    meetings: int = 3,
    races: int = 10,
    a: float = 1.0,
    b_band: dict[int, float] | None = None,
    b_rank: dict[int, float] | None = None,
    max_odds: float = 999.9,
    start: date = date(2025, 1, 1),
) -> pd.DataFrame:
    """Races whose winners are drawn from p ∝ exp(a ln p_cal + sum_g b_g 1[g] ln h), the groups
    read from a T-2 board that is a noisy copy of the market. Group columns are the references'."""
    rng = np.random.default_rng(seed)
    b_band, b_rank = b_band or {}, b_rank or {}
    out = []
    for week in range(weeks):
        day = start + timedelta(weeks=week)
        for m in range(meetings):
            meeting = f"{day.isoformat()}:V{m}"
            for r in range(races):
                n = int(rng.integers(8, 15))
                s = rng.normal(0.0, 1.3, n)
                p_cal = np.exp(s) / np.exp(s).sum()
                s_t2 = s + rng.normal(0.0, 0.15, n)
                p_t2 = np.exp(s_t2) / np.exp(s_t2).sum()
                odds = np.clip(np.round(0.8 / p_t2, 1), 1.0, max_odds)
                s_h = 0.6 * s + rng.normal(0.0, 0.8, n)
                h = np.exp(s_h) / np.exp(s_h).sum()
                bands = [ref_band(o) for o in odds]
                groups = [ref_rank_group(k) for k in ref_ranks(list(odds))]
                b = np.array(
                    [
                        b_band.get(bd, 0.0) + b_rank.get(g, 0.0)
                        for bd, g in zip(bands, groups, strict=True)
                    ]
                )
                u = a * np.log(p_cal) + b * np.log(h)
                p = np.exp(u - u.max())
                winner = rng.choice(n, p=p / p.sum())
                for i in range(n):
                    out.append(
                        {
                            "race_id": f"{meeting}:{r + 1:02d}",
                            "meeting_id": meeting,
                            "race_date": day,
                            "horse_no": i + 1,
                            "won": i == winner,
                            "scratched": False,
                            "p_cal": float(p_cal[i]),
                            "h": float(h[i]),
                            "win_odds_t2": float(odds[i]),
                            "band": bands[i],
                            "rank_group": groups[i],
                        }
                    )
    return pd.DataFrame(out)


def with_folds(rows: pd.DataFrame, scheme: FoldScheme = MONTHLY):
    folds = walk_forward(rows["race_date"].tolist(), scheme=scheme)
    fold = np.full(len(rows), -1)
    for k, f in enumerate(folds):
        fold[f.test_idx] = k
    assert (fold >= 0).all()
    return rows.assign(fold=fold), folds


def move_winners(rows: pd.DataFrame, in_folds: set[int]) -> pd.DataFrame:
    """Each race of `in_folds` won by the next runner instead."""
    out = rows.copy()
    for _, race in out[out["fold"].isin(in_folds)].groupby("race_id"):
        idx = race.index.to_numpy()
        won = race["won"].to_numpy()
        out.loc[idx, "won"] = np.roll(won, 1)
    return out


def assert_fit_equal(got, want: dict[str, float], tol: float = COEF_TOL) -> None:
    assert got.a == pytest.approx(want["a"], abs=tol)
    assert set(got.b) == set(want) - {"a"}
    for name, value in got.b.items():
        assert value == pytest.approx(want[name], abs=tol), name


def fits_differ(x, y, tol: float = 1e-4) -> bool:
    return abs(x.a - y.a) > tol or any(abs(x.b[g] - y.b[g]) > tol for g in x.b)


# ---------------------------------------------------------------- 1. bands and ranks at T-2


def test_constants_are_the_registered_family():
    assert tuple(fb.VARIANTS) == VARIANTS
    assert fb.TICKET == 218
    assert fb.BOOTSTRAP_SEED == 218
    assert isinstance(fb.FAMILY_ID, str) and fb.FAMILY_ID.strip()


EDGE_ODDS = [1.0, 2.999, 3.0, 6.99, 7.0, 14.9, 15.0, 49.9, 50.0, 999.9]
EDGE_BANDS = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_bands_are_left_closed_at_each_edge_and_ranks_share_the_better_rank_per_race():
    # race A: one runner per edge case, all distinct odds -> ranks 1..10
    # race B: ties at the shortest odds, at rank 3 (three runners) and at rank 6, and one runner
    #         off the T-2 board (no rank, no band)
    # race C: ranked on its own (a pooled rank would put both runners after race A's 1.0)
    b_odds = [2.5, 2.5, 4.0, 4.0, 4.0, 9.0, 9.0, 12.0, math.nan]
    c_odds = [5.0, 3.0]
    board = pd.DataFrame(
        {
            "race_id": ["A"] * 10 + ["B"] * 9 + ["C"] * 2,
            "horse_no": list(range(1, 11)) + list(range(1, 10)) + [1, 2],
            "win_odds_t2": EDGE_ODDS + b_odds + c_odds,
            "keep_me": np.arange(21) * 10,
        }
    )
    # a shuffled order and a non-default index: rows must stay aligned
    board = board.sample(frac=1.0, random_state=3)
    board.index = board.index + 100

    got = fb.assign_groups(board)

    assert list(got.index) == list(board.index)
    assert got["keep_me"].tolist() == board["keep_me"].tolist()
    want_band = {("A", h): b for h, b in zip(range(1, 11), EDGE_BANDS, strict=True)}
    want_band |= {("B", h): ref_band(o) for h, o in zip(range(1, 9), b_odds[:8], strict=True)}
    want_band |= {("C", 1): 2, ("C", 2): 2}
    want_rank = {("A", h): h for h in range(1, 11)}
    want_rank |= {("B", h): r for h, r in zip(range(1, 9), [1, 1, 3, 3, 3, 6, 6, 8], strict=True)}
    want_rank |= {("C", 1): 2, ("C", 2): 1}
    for row in got.itertuples():
        key = (row.race_id, row.horse_no)
        if key == ("B", 9):
            assert pd.isna(row.band) and pd.isna(row.fav_rank) and pd.isna(row.rank_group)
            continue
        assert int(row.band) == want_band[key], key
        assert int(row.fav_rank) == want_rank[key], key
        assert int(row.rank_group) == ref_rank_group(want_rank[key]), key
    # the 7+ group and the shared rank 3 landing in 2-3 are both exercised
    groups_b = got[got["race_id"] == "B"].set_index("horse_no")["rank_group"]
    assert groups_b.loc[[3, 4, 5]].astype(int).tolist() == [2, 2, 2]
    assert groups_b.loc[[6, 7]].astype(int).tolist() == [3, 3]
    assert int(groups_b.loc[8]) == 4


def test_groups_on_random_boards_match_the_reference():
    rng = np.random.default_rng(218)
    frames = []
    for r in range(40):
        n = int(rng.integers(5, 17))
        odds = np.round(rng.choice([1.0, 1.5, 2.9, 3.0, 5.0, 7.0, 15.0, 50.0, 120.0], n) *
                        rng.choice([1.0, 1.0, 1.1], n), 1)  # fmt: skip
        frames.append(pd.DataFrame({"race_id": f"R{r}", "horse_no": range(1, n + 1),
                                    "win_odds_t2": odds}))  # fmt: skip
    board = pd.concat(frames, ignore_index=True)
    got = fb.assign_groups(board)
    for rid, race in board.groupby("race_id"):
        mine = got.loc[race.index]
        ranks = ref_ranks(race["win_odds_t2"].tolist())
        assert mine["band"].astype(int).tolist() == [ref_band(o) for o in race["win_odds_t2"]], rid
        assert mine["fav_rank"].astype(int).tolist() == ranks, rid
        assert mine["rank_group"].astype(int).tolist() == [ref_rank_group(k) for k in ranks], rid


# ---------------------------------------------------------------- the T-2 Win board

UTC = timezone.utc
POST = datetime(2025, 6, 1, 6, 40, tzinfo=UTC)


def ts_row(rid: str, horse, minutes: float, odds, *, status="15時38分現在", investment=None):
    fetched = POST - timedelta(minutes=minutes)
    return {
        "race_id": rid,
        "venue": "TOK",
        "race_no": int(rid[-2:]),
        "pool": "WIN",
        "combination": str(horse),
        "snapshot_utc": fetched - timedelta(seconds=4),
        "fetched_utc": fetched,
        "post_utc": POST,
        "minutes_to_post": float(minutes),
        "odds": odds,
        "odds_hi": None,
        "investment": investment,
        "status_raw": status,
        "is_final_capture": False,
        "dividend": None,
        "is_refund": None,
    }


def capture(rid: str, minutes: float, odds: dict, *, status: dict | None = None) -> list[dict]:
    status = status or {}
    rows = [
        ts_row(rid, h, minutes, o, status=status.get(h, "15時38分現在")) for h, o in odds.items()
    ]
    rows.append(ts_row(rid, "__pool__", minutes, None, status=None, investment=1.0e7))
    return rows


def test_t2_win_odds_reads_the_board_two_minutes_before_post():
    a, b = "jra:2025-06-01:TOK:11", "jra:2025-06-01:TOK:12"
    rows = (
        capture(a, 10.0, {1: 3.0, 2: 5.0, 3: 8.0, 4: 20.0, 5: 40.0})
        + capture(a, 3.0, {1: 2.8, 2: 5.5, 3: 8.0, 4: 21.0, 5: 44.0})
        + capture(a, 2.0, {1: 2.9, 2: 6.0, 3: None, 4: 22.0}, status={4: "SCR"})
        + capture(a, 1.0, {1: 1.5, 2: 9.9, 3: 7.0, 4: 22.0, 5: 60.0})  # after T-2: never read
        + capture(b, 5.0, {1: 4.4, 2: 1.9})
        + capture(b, 0.5, {1: 2.0, 2: 3.0})
    )
    day = pa.Table.from_pylist(rows, schema=SCHEMA).to_pandas()
    day = day.sample(frac=1.0, random_state=11).reset_index(drop=True)

    got = fb.t2_win_odds(day)

    pairs = {(r.race_id, int(r.horse_no)): float(r.win_odds_t2) for r in got.itertuples()}
    # horse 3 is unpriced and horse 4 scratched at T-2: both off the board; horse 5 keeps its
    # 3-minute price (its latest row at or before T-2)
    assert pairs == {(a, 1): 2.9, (a, 2): 6.0, (a, 5): 44.0, (b, 1): 4.4, (b, 2): 1.9}
    later = fb.t2_win_odds(day, minutes_before=1.0)
    later_pairs = {(r.race_id, int(r.horse_no)): float(r.win_odds_t2) for r in later.itertuples()}
    assert later_pairs[(a, 1)] == 1.5 and later_pairs[(a, 3)] == 7.0


# ---------------------------------------------------------------- 2. the 12 variants' designs


def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["X"] * 4 + ["Y"] * 4,
            "h": [0.40, 0.25, 0.20, 0.15, 0.05, 0.30, 0.45, 0.20],
            "band": [1, 2, 3, 5, 4, 3, 1, 2],
            "rank_group": [1, 2, 3, 4, 4, 2, 1, 3],
        },
        index=[7, 3, 9, 1, 4, 8, 2, 6],
    )


def test_each_variant_puts_ln_h_on_its_own_groups_only():
    rows = small_frame()
    ln_h = np.log(rows["h"].to_numpy())
    band, rank = rows["band"].to_numpy(), rows["rank_group"].to_numpy()
    want = {
        "P0": {"all": ln_h},
        **{f"B{j}": {f"band{j}": np.where(band == j, ln_h, 0.0)} for j in range(1, 6)},
        **{f"R{j}": {f"rank{j}": np.where(rank == j, ln_h, 0.0)} for j in range(1, 5)},
        "BA": {f"band{j}": np.where(band == j, ln_h, 0.0) for j in range(1, 6)},
        "RA": {f"rank{j}": np.where(rank == j, ln_h, 0.0) for j in range(1, 5)},
    }
    for variant in VARIANTS:
        X = fb.design(rows, variant)
        assert list(X.columns) == list(want[variant]), variant
        assert list(X.index) == list(rows.index), variant
        for column, values in want[variant].items():
            np.testing.assert_allclose(X[column].to_numpy(dtype=float), values, atol=1e-12)
    # by hand: B3 holds only the two band-3 runners (labels 9 and 8)
    b3 = fb.design(rows, "B3")["band3"]
    assert b3[b3 != 0].index.tolist() == [9, 8]
    assert b3.loc[9] == pytest.approx(math.log(0.20)) and b3.loc[8] == pytest.approx(math.log(0.30))
    # the joint variants: every runner in exactly one column, the row sum ln h
    for joint in ("BA", "RA"):
        X = fb.design(rows, joint).to_numpy(dtype=float)
        assert ((X != 0).sum(axis=1) == 1).all(), joint
        np.testing.assert_allclose(X.sum(axis=1), ln_h, atol=1e-12)


# ---------------------------------------------------------------- 3. the fit recovers the plant


@pytest.mark.parametrize(
    ("variant", "a", "b_band", "b_rank", "want", "tol"),
    [
        ("BA", 1.0, {1: -0.3}, {}, {"band1": -0.3, "band2": 0, "band3": 0, "band4": 0, "band5": 0},
         0.12),
        ("BA", 0.8, {3: 0.4, 5: -0.2}, {},
         {"band1": 0, "band2": 0, "band3": 0.4, "band4": 0, "band5": -0.2}, 0.12),
        ("RA", 1.1, {}, {1: -0.4, 4: 0.15}, {"rank1": -0.4, "rank2": 0, "rank3": 0, "rank4": 0.15},
         0.12),
    ],
)  # fmt: skip
def test_the_meld_fit_recovers_planted_group_coefficients(variant, a, b_band, b_rank, want, tol):
    rows = synthetic_races(5, weeks=100, meetings=3, races=20, a=a, b_band=b_band, b_rank=b_rank)
    fit = fb.fit_meld(rows, variant)
    # exactly the unpenalised conditional logit on [ln p_cal, the variant's columns]
    assert_fit_equal(fit, ref_fit(rows, variant))
    assert fit.n_races == rows["race_id"].nunique()
    # and near the plant (6000 races; the tolerance is about 3 standard errors)
    assert fit.a == pytest.approx(a, abs=0.1)
    assert set(fit.b) == set(want)
    for group, value in want.items():
        assert fit.b[group] == pytest.approx(value, abs=tol), group


def test_single_group_fits_leave_every_other_runner_at_a_ln_p_cal():
    rows = synthetic_races(9, weeks=30, meetings=3, races=10, b_band={2: -0.5})
    for variant in VARIANTS:
        fit = fb.fit_meld(rows, variant)
        assert_fit_equal(fit, ref_fit(rows, variant))


# ---------------------------------------------------------------- 4. walk-forward discipline


@pytest.fixture(scope="module")
def folded():
    return with_folds(synthetic_races(21, b_band={1: -0.4}))


def test_each_fold_is_fitted_on_earlier_folds_only_and_the_first_fold_only_trains(folded):
    rows, folds = folded
    assert len(folds) == 4
    run = fb.walk_forward_meld(rows, folds, "BA")
    assert sorted(run.fits) == [1, 2, 3]
    for k in (1, 2, 3):
        assert_fit_equal(run.fits[k], ref_fit(ref_train(rows, folds, k), "BA"))
    per_race = run.per_race
    assert set(per_race["fold"]) == {1, 2, 3}
    scored = rows[rows["fold"] >= 1]
    assert sorted(per_race["race_id"]) == sorted(scored["race_id"].unique())
    # each fold's races are scored with that fold's fit
    for k in (1, 2, 3):
        fit = run.fits[k]
        want = ref_scores(rows[rows["fold"] == k], "BA", {"a": fit.a, **fit.b})
        got = per_race[per_race["fold"] == k].set_index("race_id").sort_index()
        want = want.set_index("race_id").sort_index()
        np.testing.assert_allclose(got["ll_v0"], want["ll_v0"], rtol=1e-9)
        np.testing.assert_allclose(got["ll_variant"], want["ll_variant"], rtol=1e-9)
        np.testing.assert_allclose(got["gain"], got["ll_v0"] - got["ll_variant"], atol=1e-12)


def test_outcomes_of_fold_k_or_later_never_move_fold_ks_fit(folded):
    rows, folds = folded
    base = fb.walk_forward_meld(rows, folds, "BA").fits
    late = fb.walk_forward_meld(move_winners(rows, {2, 3}), folds, "BA").fits
    for k in (1, 2):
        assert_fit_equal(late[k], {"a": base[k].a, **base[k].b}, tol=1e-9)
    assert fits_differ(late[3], base[3])  # fold 2's outcomes train fold 3
    early = fb.walk_forward_meld(move_winners(rows, {1}), folds, "BA").fits
    assert_fit_equal(early[1], {"a": base[1].a, **base[1].b}, tol=1e-9)
    assert fits_differ(early[2], base[2]) and fits_differ(early[3], base[3])


def test_the_earliest_fold_present_only_trains(folded):
    rows, folds = folded
    later = rows[rows["fold"] >= 1].reset_index(drop=True)
    run = fb.walk_forward_meld(later, folds, "P0")
    assert sorted(run.fits) == [2, 3]
    assert set(run.per_race["fold"]) == {2, 3}
    assert_fit_equal(run.fits[2], ref_fit(ref_train(later, folds, 2), "P0"))


@pytest.fixture(scope="module")
def embargoed():
    """`folded`'s weekly Wednesdays plus a Friday meeting day, 2025-02-28: fold 1's races that
    day fall inside fold 2's embargo (train_end = 2025-03-05 - 7 d = 2025-02-26)."""
    weekly = synthetic_races(21, b_band={1: -0.4})
    friday = synthetic_races(22, weeks=1, b_band={1: -0.4}, start=date(2025, 2, 28))
    return with_folds(pd.concat([weekly, friday], ignore_index=True))


def test_rows_inside_the_embargo_never_train(embargoed):
    rows, folds = embargoed
    dates = pd.to_datetime(rows["race_date"])
    inside = (rows["fold"] == 1) & (dates > pd.Timestamp(folds[2].train_end))
    assert inside.any()  # the fixture has fold-1 races in fold 2's embargo week
    base = fb.walk_forward_meld(rows, folds, "P0").fits
    moved = move_winners(rows, set())
    moved.loc[inside, "won"] = (
        moved.loc[inside].groupby("race_id")["won"].transform(lambda w: np.roll(w.to_numpy(), 1))
    )
    got = fb.walk_forward_meld(moved, folds, "P0").fits
    assert_fit_equal(got[2], {"a": base[2].a, **base[2].b}, tol=1e-9)
    assert fits_differ(got[3], base[3])  # the same rows do train fold 3: the change is real


# ---------------------------------------------------------------- 5. scoring


def test_score_races_by_hand_dead_heat_and_scratched_runner():
    rows = pd.DataFrame(
        {
            "race_id": ["S"] * 4 + ["D"] * 3,
            "meeting_id": ["M1"] * 4 + ["M2"] * 3,
            "won": [False, True, False, False, True, True, False],
            "scratched": [False, False, False, True, False, False, False],
            "p_cal": [0.40, 0.30, 0.10, 0.20, 0.5, 0.3, 0.2],
            "h": [0.30, 0.40, 0.20, 0.10, 0.2, 0.5, 0.3],
            "band": [2, 2, 4, 3, 1, 2, 3],
            "rank_group": [1, 2, 3, 2, 1, 2, 3],
        }
    )
    fit = fb.MeldFit(a=0.9, b={"band2": -0.5}, n_races=0)
    got = fb.score_races(rows, "B2", fit).set_index("race_id")

    # race S: the scratched runner leaves, p_cal renormalised over 0.4, 0.3, 0.1
    p_cal = np.array([0.4, 0.3, 0.1]) / 0.8
    u = 0.9 * np.log(p_cal) + np.array([-0.5 * math.log(0.3), -0.5 * math.log(0.4), 0.0])
    p = np.exp(u) / np.exp(u).sum()
    assert got.loc["S", "ll_v0"] == pytest.approx(-math.log(0.3 / 0.8))
    assert got.loc["S", "ll_variant"] == pytest.approx(-math.log(p[1]))
    # race D: a dead heat between runners 1 and 2, each at weight 1/2
    u = 0.9 * np.log([0.5, 0.3, 0.2]) + np.array([0.0, -0.5 * math.log(0.5), 0.0])
    p = np.exp(u) / np.exp(u).sum()
    assert got.loc["D", "ll_v0"] == pytest.approx(-(math.log(0.5) + math.log(0.3)) / 2)
    assert got.loc["D", "ll_variant"] == pytest.approx(-(math.log(p[0]) + math.log(p[1])) / 2)
    for rid in ("S", "D"):
        assert got.loc[rid, "gain"] == pytest.approx(
            got.loc[rid, "ll_v0"] - got.loc[rid, "ll_variant"]
        )
    assert got.loc["D", "meeting_id"] == "M2"


def test_unpriced_starters_and_void_races_are_dropped_and_counted(folded):
    rows, folds = folded
    rows = rows.copy()
    fold2 = rows[rows["fold"] == 2]["race_id"].unique()
    unpriced, void, scratched_early = fold2[0], fold2[1], fold2[2]
    # a starter with no T-2 price: the race cannot be grouped
    first = rows.index[rows["race_id"] == unpriced][0]
    rows.loc[first, ["win_odds_t2", "band", "rank_group"]] = np.nan
    # no winner: void
    rows.loc[rows["race_id"] == void, "won"] = False
    # a runner scratched before T-2 (off the board) does not drop its race
    loser = rows.index[(rows["race_id"] == scratched_early) & ~rows["won"]][0]
    rows.loc[loser, ["win_odds_t2", "band", "rank_group"]] = np.nan
    rows.loc[loser, "scratched"] = True

    run = fb.walk_forward_meld(rows, folds, "B1")

    assert run.n_unpriced_dropped == 1
    assert run.n_void_dropped == 1
    scored = set(run.per_race["race_id"])
    assert unpriced not in scored and void not in scored and scratched_early in scored
    want = set(rows.loc[rows["fold"] >= 1, "race_id"]) - {unpriced, void}
    assert scored == want


def test_gain_interval_is_d7s_seeded_meeting_bootstrap():
    rng = np.random.default_rng(4)
    n_meet = 30
    meeting = np.repeat([f"M{i:02d}" for i in range(n_meet)], 12)
    shift = np.repeat(rng.normal(0.0, 0.02, n_meet), 12)  # gains correlated within a meeting
    per_race = pd.DataFrame(
        {
            "race_id": [f"r{i:03d}" for i in range(len(meeting))],
            "meeting_id": meeting,
            "gain": 0.004 + shift + rng.normal(0.0, 0.05, len(meeting)),
        }
    )
    got = fb.gain_interval(per_race, n_boot=500, seed=218)
    again = fb.gain_interval(per_race, n_boot=500, seed=218)
    other = fb.gain_interval(per_race, n_boot=500, seed=219)

    gain = per_race.set_index("race_id")["gain"]
    clusters = per_race.set_index("race_id")["meeting_id"]
    one_sided = paired_bootstrap(gain, clusters, n_boot=500, level=0.95, seed=218)
    two_sided = paired_bootstrap(gain, clusters, n_boot=500, level=0.975, seed=218)
    assert got.mean == pytest.approx(per_race["gain"].mean(), rel=1e-12)
    assert got.lower == pytest.approx(one_sided.lo, rel=1e-12)
    assert tuple(got.ci95) == pytest.approx((two_sided.lo, two_sided.hi), rel=1e-12)
    assert got.n_meetings == n_meet
    assert got.ci95[0] <= got.lower <= got.mean <= got.ci95[1]
    assert (again.mean, again.lower, tuple(again.ci95)) == (got.mean, got.lower, tuple(got.ci95))
    assert other.lower != got.lower


# ---------------------------------------------------------------- 6. the run and its governance


def run_rows(seed: int, **kwargs) -> tuple[pd.DataFrame, list]:
    """The run's input (no group columns: the run assigns them); `reference_rows` keeps them."""
    rows, folds = with_folds(synthetic_races(seed, **kwargs))
    return rows.drop(columns=["band", "rank_group"]), folds


def reference_rows(seed: int, **kwargs) -> pd.DataFrame:
    return with_folds(synthetic_races(seed, **kwargs))[0]


def tmp_registry(base: Path) -> Registry:
    return Registry(base / "registry" / "japan.jsonl", holdout_log=base / "holdout_log.jsonl")


def registry_lines(registry: Registry) -> list[dict]:
    return [json.loads(line) for line in registry.path.read_text().splitlines()]


def run(base: Path, rows, folds, **kwargs) -> tuple[dict, Registry]:
    registry = tmp_registry(base)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(base / "data"))
        result = fb.run_family(
            rows, folds, registry=registry, out_dir=base / "out", n_boot=N_BOOT, **kwargs
        )
    return result, registry


@pytest.fixture(scope="module")
def noise_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("noise")
    rows, folds = run_rows(2)  # h carries nothing beyond the market: every b planted 0
    result, registry = run(base, rows, folds)
    return result, registry, rows, folds, base


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("planted")
    rows, folds = run_rows(1, b_band={1: -1.0})
    result, registry = run(base, rows, folds)
    return result, registry, rows, folds, base


def test_the_run_freezes_the_spec_first_and_records_all_twelve_variants(noise_run):
    result, registry, _, _, base = noise_run
    lines = registry_lines(registry)
    assert lines[0]["kind"] == "freeze" and lines[0]["family_id"] == fb.FAMILY_ID
    runs = lines[1:]
    assert [line["kind"] for line in runs] == ["run"] * 12
    assert result["family_id"] == fb.FAMILY_ID
    assert result["spec_hash"] == registry.frozen_spec_hash(fb.FAMILY_ID)
    assert result["records_appended"] == 12
    records = registry.records(fb.FAMILY_ID)
    assert sorted(r.variant for r in records) == sorted(VARIANTS)
    assert all(r.ticket == 218 for r in records)
    assert all(is_withheld(r.sources) for r in records)  # JRA-VAN bands: values withheld
    assert registry.family_size(fb.FAMILY_ID) == 12
    # the frozen spec cannot be swapped afterwards; the same spec again is a no-op
    spec = lines[0]["spec"]
    assert registry.freeze_family(fb.FAMILY_ID, spec) == result["spec_hash"]
    with pytest.raises(FamilyAlreadyFrozen):
        registry.freeze_family(fb.FAMILY_ID, {**spec, "changed_after_freeze": True})
    report = Path(result["report_path"])
    assert report.is_file() and report.resolve().parent == (base / "out").resolve()
    text = report.read_text()
    assert all(variant in text for variant in VARIANTS)


def test_the_run_reports_each_variants_gain_bounds_and_group_fits(noise_run):
    result, _, _, folds, _ = noise_run
    variants = result["variants"]
    assert set(variants) == set(VARIANTS)
    assert all(v["status"] == "scored" for v in variants.values())

    groups = reference_rows(2)  # the same races with the reference's bands and rank groups
    scored_rows = groups[groups["fold"] >= 1]
    for variant in ("P0", "B3", "RA"):
        fits = {k: ref_fit(ref_train(groups, folds, k), variant) for k in (1, 2, 3)}
        per_race = pd.concat(
            [ref_scores(groups[groups["fold"] == k], variant, fits[k]) for k in (1, 2, 3)]
        )
        got = variants[variant]
        assert got["gain"] == pytest.approx(per_race["gain"].mean(), abs=1e-7), variant
        assert got["n_races"] == len(per_race)
        assert got["n_meetings"] == per_race["meeting_id"].nunique()
        boot = paired_bootstrap(
            per_race.set_index("race_id")["gain"],
            per_race.set_index("race_id")["meeting_id"],
            n_boot=N_BOOT,
            level=0.95,
            seed=218,
        )
        assert got["lower"] == pytest.approx(boot.lo, abs=1e-6), variant
        assert got["ci95"][0] <= got["lower"] <= got["gain"] <= got["ci95"][1]
        names = [name for name, _, _ in COLUMNS[variant]]
        assert set(got["groups"]) == set(names)
        for name, kind, number in COLUMNS[variant]:
            b = [fits[k][name] for k in (1, 2, 3)]
            g = got["groups"][name]
            assert g["b_median"] == pytest.approx(float(np.median(b)), abs=1e-6), name
            assert g["b_min"] == pytest.approx(min(b), abs=1e-6), name
            assert g["b_max"] == pytest.approx(max(b), abs=1e-6), name
            if kind == "all":
                inside = scored_rows
            else:
                column = "band" if kind == "band" else "rank_group"
                inside = scored_rows[scored_rows[column].astype(int) == number]
            assert g["runners"] == len(inside), name
            assert g["races"] == inside["race_id"].nunique(), name


def test_on_pure_noise_the_family_test_finds_nothing(noise_run):
    result = noise_run[0]
    assert 0.0 <= result["spa"]["pvalue"] <= 1.0
    assert result["spa"]["n_rules"] == len(NON_REFERENCE)
    assert result["stepm"]["superior"] == []


def test_a_planted_favourite_band_effect_is_found_by_the_family_test(planted_run):
    result = planted_run[0]
    variants = result["variants"]
    assert variants["B1"]["lower"] > 0 and variants["BA"]["lower"] > 0
    assert result["spa"]["pvalue"] < 0.05
    assert result["spa"]["n_rules"] == len(NON_REFERENCE)
    superior = set(result["stepm"]["superior"])
    assert {"B1", "BA"} <= superior
    assert superior <= set(NON_REFERENCE)  # P0, the reference, is not in the family test
    groups = variants["BA"]["groups"]
    assert groups["band1"]["b_median"] < -0.5
    assert all(abs(groups[f"band{j}"]["b_median"]) < 0.5 for j in (2, 3, 4, 5))


def test_a_variant_that_cannot_be_fitted_is_a_failed_record(tmp_path):
    rows, folds = run_rows(3, max_odds=45.0)  # no runner in band 5
    result, registry = run(tmp_path, rows, folds)
    records = {r.variant: r for r in registry.records(fb.FAMILY_ID)}
    assert set(records) == set(VARIANTS)
    assert records["B5"].status == "failed"
    assert result["variants"]["B5"]["status"] == "failed"
    for variant in ("P0", "B1", "R1", "RA"):
        assert records[variant].status == "scored", variant


def test_a_family_frozen_with_another_spec_refuses_the_run(tmp_path):
    rows, folds = run_rows(2, weeks=10)
    registry = tmp_registry(tmp_path)
    registry.freeze_family(fb.FAMILY_ID, {"not": "the approved pre-registration"})
    with pytest.raises(FamilyAlreadyFrozen):
        fb.run_family(rows, folds, registry=registry, out_dir=tmp_path / "out", n_boot=N_BOOT)
    assert registry.records(fb.FAMILY_ID) == []
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_a_report_inside_the_repository_is_refused(tmp_path):
    rows, folds = run_rows(2, weeks=10)
    inside = env.repo_root() / "docs" / "experiments" / "t218-must-not-exist"
    assert not inside.exists()
    with pytest.raises((LicensedOutputInRepo, paths.DataDirInsideRepo)):
        fb.run_family(rows, folds, registry=tmp_registry(tmp_path), out_dir=inside, n_boot=N_BOOT)
    assert not inside.exists()


def test_rows_on_a_holdout_day_are_refused_before_anything_is_written(tmp_path):
    rows, folds = run_rows(2, weeks=10)
    day = date(2026, 11, 28)
    while not is_holdout_day("japan", day):
        day += timedelta(days=1)
    last = rows["race_id"].iloc[-1]
    rows.loc[rows["race_id"] == last, "race_date"] = day
    registry = tmp_registry(tmp_path)
    with pytest.raises(ValueError):
        fb.run_family(rows, folds, registry=registry, out_dir=tmp_path / "out", n_boot=N_BOOT)
    assert registry.records() == []
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


# ---------------------------------------------------------------- 7. reruns and the private report


def run_on(registry: Registry, base: Path, rows, folds) -> dict:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(base / "data"))
        return fb.run_family(rows, folds, registry=registry, out_dir=base / "out", n_boot=N_BOOT)


def ids_named(result: dict, ids: list[str]) -> set[str]:
    """The full registry ids `result["run_ids"]` names, each prefix naming exactly one."""
    out = set()
    for prefix in result["run_ids"]:
        hits = [i for i in ids if i.startswith(prefix)]
        assert len(hits) == 1, prefix
        out |= set(hits)
    return out


def test_a_rerun_in_which_a_variant_now_fails_still_reports_and_tests_the_current_run(tmp_path):
    registry = tmp_registry(tmp_path)
    first = run_on(registry, tmp_path, *run_rows(2))
    assert first["variants"]["B5"]["status"] == "scored"

    second = run_on(registry, tmp_path, *run_rows(3, max_odds=45.0))  # no band-5 runner now

    assert second["variants"]["B5"]["status"] == "failed"
    history = {(r.variant, r.status) for r in registry.records(fb.FAMILY_ID)}
    assert {("B5", "scored"), ("B5", "failed")} <= history
    scored_now = [v for v in NON_REFERENCE if second["variants"][v]["status"] == "scored"]
    assert "B5" not in scored_now and len(scored_now) >= 9
    assert second["spa"]["n_rules"] == len(scored_now)
    assert set(second["stepm"]["superior"]) <= set(scored_now)
    report = Path(second["report_path"])
    assert report.is_file() and "B5" in report.read_text()


def test_each_run_keeps_its_own_private_report_listing_its_records(tmp_path):
    registry = tmp_registry(tmp_path)
    rows, folds = run_rows(2)
    a = run_on(registry, tmp_path, rows, folds)
    a_text = Path(a["report_path"]).read_text()
    b = run_on(registry, tmp_path, rows, folds)  # identical rerun: nothing new to append
    c = run_on(registry, tmp_path, *run_rows(4))  # new inputs: 12 new records

    ids = [line["run_id"] for line in registry_lines(registry) if line["kind"] == "run"]
    assert len(ids) == 24
    assert len({a["report_path"], b["report_path"], c["report_path"]}) == 3
    assert Path(a["report_path"]).read_text() == a_text  # never overwritten
    assert ids_named(a, ids) == set(ids[:12])
    assert ids_named(b, ids) == set(ids[:12])  # the records it reproduced
    assert ids_named(c, ids) == set(ids[12:])
    for result in (a, b, c):
        text = Path(result["report_path"]).read_text()
        assert all(prefix in text for prefix in result["run_ids"])


def section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def test_the_report_answers_the_primary_question_without_a_new_rule(noise_run):
    text = Path(noise_run[0]["report_path"]).read_text()
    primary = section(text, "## Primary question")
    assert "the frozen spec defines no favourite-vs-longshot comparison rule" in primary.lower()
    assert "inconclusive" in primary.lower()
    for variant in ("R1", "R2", "R3", "R4", "B1", "B2", "B3", "B4", "B5"):
        assert variant in primary, variant


def test_the_report_labels_gains_as_the_registered_melds_and_states_the_confound(noise_run):
    text = Path(noise_run[0]["report_path"]).read_text().lower()
    assert "gain of the registered meld over v0" in text
    assert "group intercept" in text
    assert "market refit" in text
