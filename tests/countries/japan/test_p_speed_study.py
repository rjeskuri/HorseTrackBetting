"""E39: #97's p_speed standalone and marginal at the final price (ticket #347, epic #73).

#97's definition-of-done item 2. Contract pinned here: `horsetrack.countries.japan.studies.
p_speed`, imported as `ps`; the coder implements what these tests call.

- `TICKET == 347`; `FAMILY_ID`, a non-blank family id the registry can freeze; every variant's
  `hypothesis_id` is "E39". `GRID`: the pre-registered grid, `{"drift_var_per_day": [...],
  "t_df": [...]}` (at least two values each; a drift >= 0, a t_df None (normal) or > 0). The
  methods are "pl" and "henery".
- `predict_folds(runs, table, *, spec, scheme, drift_var_per_day, t_df, decision_min) ->
  DataFrame`. `runs` is `models.speed_runs.build_runs`'s frame (HorseSpeed's runs), `table`
  `studies.speed_run.build_race_table`'s (SpeedModel's races), `spec` a `SpeedSpec`. Rows on a
  holdout day (`core.eval.holdout.is_holdout_day("japan", day)`, the JST race day) are dropped
  from both frames before anything else: no holdout run or race is fitted, observed or scored,
  whatever the frames hold. Then #70's folds, `walk_forward` over the remaining race days under
  `scheme`. Per fold k: `SpeedModel(spec)` fitted on its training races and observing its test
  races; `HorseSpeed(drift_var_per_day=..., t_df=...)` fitted on the training races' runs with
  that speed model and observing the test races' runs; each test race's runners predicted at
  its decision time t = race_utc - decision_min minutes. So a test race reads every run and
  race-level result published strictly before its t, and nothing published at or after it, its
  own result included. One row per run of a test race of every fold, fold 0 included:
  `race_id`, `horse_id`, `fold` (k, the position among those folds), `train_end` (the fold's),
  `decision_utc` (t), `mu_s` and `sd_s` (`HorseSpeed.predict`'s); other columns may follow.
- `fit_scale(rows, *, method="pl") -> float` on rows with race_id, won and mu_s: "pl" is 1 / the
  coefficient of `models.condlogit.fit_conditional_logit` of the winner on -mu_s (a dead heat's
  winners at 1/k, condlogit's rule); "henery" is the scale maximising the winners' Henery
  log-likelihood, `models.speed_horse.p_speed(mu, scale=s, method="henery")` per race (a dead
  heat's winners averaged).
- `score_folds(rows, *, method) -> DataFrame` on rows with race_id, horse_id, race_date, fold,
  train_end, won and mu_s (`predict_folds`' rows with the outcome). Fold k's scale is `fit_scale`
  on the rows of earlier folds whose race_date is on or before fold k's train_end; a fold with
  none (fold 0) only trains. One row per row of a scored fold, the input's columns kept, plus
  `scale` (the fold's) and `p_speed`, `speed_horse.p_speed(the race's mu_s, scale, method)`.
- `standalone(rows, spec) -> (metrics, tables)`, rows one per runner of the scored races with
  race_id, meeting_id, fold, won, p_speed and p_cal (D4's calibrated market out of fold). The
  "all_folds" FoldMetric "delta_ll_vs_calibrated_market": value the mean over races of
  ll(p_cal) - ll(p_speed) (#69's winner log loss: p renormalised within the race, a dead heat's
  winners averaged; positive when p_speed is better), lo/hi `core.eval.bootstrap.
  paired_bootstrap` of those per-race differences over meeting clusters at spec["bootstrap"]'s
  n_boot, level and seed (D7), n the races. Only spec["bootstrap"] is read.
- `marginal(rows, spec) -> (metrics, tables)`, rows as standalone's plus race_date, train_end and
  q (the final Win price normalised within race). p_speed is a member of #87's meta-model next
  to the final price: `models.meta.fit_meta` with MetaSpec(["p_speed"], quadratic=False,
  segment_cols=[], form="stage2", ridge=0.0), the member column ln p_speed (within race) and
  ln_q_final ln q: u = alpha * ln p_speed + beta * ln q, both unbounded; alpha = 0 is the market
  alone, the null and the shrinkage target. "all_folds" metrics:
  - "coef[p_speed]": alpha fitted on every row, n the races; lo/hi alpha -/+ z * its
    meeting-clustered se (fit_meta's se_cluster on meeting_id), z the two-sided 95% normal
    quantile or D7's one-sided one at spec["bootstrap"]["level"] (either is accepted);
  - "delta_ll_vs_market_only": for each fold of the rows but the first, the member's meta and the
    market-only meta (u = beta * ln q, beta by conditional logit) fitted on the rows of earlier
    folds whose race_date is on or before the fold's train_end, predicting the fold's rows; the
    mean over those races of ll(market only) - ll(member's meta), lo/hi by paired_bootstrap over
    meetings at spec["bootstrap"], n those races.
- `build_spec(*, scheme, bootstrap, grid=None) -> dict`: the family's pre-registered spec, plain
  JSON, `spec["grid"]` the grid as given (None: GRID) and not validated: a grid point the models
  refuse is a failed variant, not a refused spec. Any further parameter is an optional keyword.
- `compute_study(runs, table, market, *, spec, scheme) -> studies.calibration.Study`: `market` is
  `studies.meta.store_frame`'s frame (priced starters with q, won, dead_heat, meeting_id). Holdout
  days are dropped from every input, the market's too. For every grid point of spec["grid"] (the
  product) and every method, variants whose params carry `drift_var_per_day`, `t_df` (None:
  normal) and `method`, as numbers and strings; between them a scored point's variants carry the
  three "all_folds" metrics above, with finite values and bounds (standalone against D4 out of
  fold on the same development folds). A grid point whose fit raises makes its variants failed
  (error set, no metrics); the other points still run.
- `load_inputs(...) -> (runs, table, market)`: the store read; replaced in these tests.
- `run_study(*, out_dir=None, scheme=None) -> dict`: an out_dir inside the repository raises
  `core.paths.DataDirInsideRepo` with nothing written or recorded. Otherwise it builds the spec
  with the default grid and freezes it (`default_registry("japan").freeze_family`) before
  `load_inputs` is called (before any outcome is read), runs `compute_study` on the inputs,
  appends one RunRecord per variant (family FAMILY_ID, ticket 347, the variant's hypothesis,
  name, params and metrics; status "failed" when it has an error, else "scored"), and writes the
  report through `core.eval.reports.write_report` under out_dir. Its data builds are the store
  tables' (`studies.calibration.store_builds()`). The result has `family_id`, `spec_hash` (the
  frozen hash) and `report_path`.

Rulings assumed here (reversible implementation choices; the ticket does not fix them):
- Ruling: fold k's PL scale is fitted on the earlier folds' out-of-fold predictions within its
  training window (fold 0 only trains), as E32's walk-forward coefficient (#348): a HorseSpeed fit
  refuses to predict at or before its own training results, so the training races have no
  as-of mu of their own.
- Ruling: Henery's scale is its own one-parameter maximum likelihood on the same rows (Henery's
  normal SD is not the PL scale).
- Ruling: each fold's HorseSpeed observes the fold's test runs, so a later test race reads the
  runs published before its t; the global terms stay the training runs'.
- Ruling: the marginal is #92's C4 marginal: the stage-2 meta (beta free, ridge 0) against the
  market-only stage-2 meta on the same rows and folds, from the second scored fold; alpha pooled
  over the scored rows with its meeting-clustered se.

Data: synthetic only (p_speed_synth.py), in memory or under tmp_path; expected values from #96's
SpeedModel, #97's HorseSpeed and p_speed, #87's fit_meta and condlogit, core.eval's bootstrap and
first principles, never from the study module. An autouse guard fails a test that changes the
committed registry summary or a holdout log.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import paths
from horsetrack.core.eval import registry as registry_mod
from horsetrack.core.eval import reports
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import FoldMetric, Registry, default_registry
from horsetrack.countries.japan.models.speed_horse import HorseSpeed, p_speed
from horsetrack.countries.japan.studies import p_speed as ps
from horsetrack.countries.japan.studies.calibration import Study, Variant

# isort: split
import p_speed_synth as syn

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_SUMMARY = REPO_ROOT / "docs" / "experiments" / "registry"
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"
DATA_DIR_VAR = "HORSETRACK_DATA_DIR"

WEEKLY = FoldScheme(unit="week", test_length=1, step=1, min_train=2, window="expanding",
                    embargo_days=1, anchor_month=1)  # fmt: skip
BOOT = {"cluster_unit": "meeting", "sided": "one-sided", "level": 0.95, "n_boot": 300, "seed": 7}
SPEC = {"bootstrap": BOOT}
Z_TWO_SIDED = NormalDist().inv_cdf(0.975)
Z_ONE_SIDED = NormalDist().inv_cdf(BOOT["level"])
STANDALONE = "delta_ll_vs_calibrated_market"
MARGINAL = "delta_ll_vs_market_only"
ALPHA = "coef[p_speed]"
METHODS = ("pl", "henery")
COLUMNS = {"race_id", "horse_id", "fold", "train_end", "decision_utc", "mu_s", "sd_s"}

EARLY_DAYS = syn.weekends(date(2025, 1, 4), 6)  # every day a development day
LATE_DAYS = syn.weekends(date(2026, 10, 31), 10)  # holdout days among them from 2026-11-28
HELD = [d for d in LATE_DAYS if is_holdout_day("japan", d)]
DEV = [d for d in LATE_DAYS if d not in HELD]


@lru_cache(maxsize=1)
def early_world() -> syn.World:
    return syn.world(EARLY_DAYS, seed=347, n_horses=60, races_per_venue=3)


@lru_cache(maxsize=1)
def late_world() -> syn.World:
    return syn.world(LATE_DAYS, seed=3471, n_horses=130, races_per_venue=6)


def snapshot() -> dict[Path, bytes]:
    files = [*COMMITTED_SUMMARY.rglob("*"), *REAL_COUNTRIES.glob("*/holdout_log.jsonl")]
    return {p: p.read_bytes() for p in files if p.is_file()}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """The data directory is under tmp_path; no test changes the committed summary or a log."""
    monkeypatch.setenv(DATA_DIR_VAR, str(tmp_path / "data"))
    before = snapshot()
    yield
    assert snapshot() == before, "a test changed the committed registry summary or a holdout log"


def predict(w: syn.World, *, drift=0.004, t_df=None, decision_min=0.0) -> pd.DataFrame:
    got = ps.predict_folds(
        w.runs.copy(), w.table.copy(), spec=syn.SPEC, scheme=WEEKLY, drift_var_per_day=drift,
        t_df=t_df, decision_min=decision_min,
    )  # fmt: skip
    assert isinstance(got, pd.DataFrame)
    assert not COLUMNS - set(got.columns), f"predict_folds lacks {COLUMNS - set(got.columns)}"
    assert not got.duplicated(["race_id", "horse_id"]).any(), "one row per run"
    return got


def keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(["race_id", "horse_id"]).sort_index()


def same(got, want, *, atol: float) -> None:
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    np.testing.assert_allclose(got, want, rtol=0, atol=atol)


def one(metrics, name: str, fold: str = "all_folds") -> FoldMetric:
    found = [m for m in metrics if m.fold == fold and m.metric == name]
    assert len(found) == 1, f"one {fold} {name} metric expected, got {found}"
    return found[0]


def meetings(rows: pd.DataFrame) -> pd.Series:
    return rows.groupby("race_id")["meeting_id"].first()


def boot(per_race: pd.Series, clusters: pd.Series):
    """D7's bounds at the test bootstrap settings."""
    n_boot, level, seed = BOOT["n_boot"], BOOT["level"], BOOT["seed"]
    return paired_bootstrap(per_race, clusters, n_boot=n_boot, level=level, seed=seed)


def training_rows(rows: pd.DataFrame, k: int) -> pd.DataFrame:
    end = rows.loc[rows["fold"] == k, "train_end"].iloc[0]
    return rows[(rows["fold"] < k) & (rows["race_date"] <= end)]


# ---------------------------------------------------------------------------------------------
# The fold loop: #96 and #97 fitted on each fold's training races, each test race as of its t
# ---------------------------------------------------------------------------------------------


def test_each_fold_fits_horse_speed_on_its_training_runs_and_predicts_its_test_races():
    w = early_world()
    figures = {}
    for drift, t_df in ((0.0, None), (0.02, 5.0)):
        want = keyed(syn.expected_predictions(w, scheme=WEEKLY, drift=drift, t_df=t_df,
                                              decision_min=0))  # fmt: skip
        got = keyed(predict(w, drift=drift, t_df=t_df, decision_min=0))
        assert sorted(set(want["fold"])) == [0, 1, 2], "the fixture has three folds"
        # every run of every fold's test races, fold 0 included, and nothing else
        assert got.index.equals(want.index)
        assert (got["fold"].to_numpy() == want["fold"].to_numpy()).all()
        assert (pd.to_datetime(got["train_end"]).dt.date.to_numpy()
                == want["train_end"].to_numpy()).all()  # fmt: skip
        decided = pd.to_datetime(got["decision_utc"], utc=True).reset_index(drop=True)
        wanted = pd.to_datetime(want["decision_utc"], utc=True).reset_index(drop=True)
        assert (decided == wanted).all()
        for col in ("mu_s", "sd_s"):
            same(got[col], want[col], atol=1e-9)
        figures[(drift, t_df)] = got["mu_s"].to_numpy(dtype=float)
    # the grid point reaches the fit: another (drift, t_df) gives other figures
    assert np.abs(figures[(0.0, None)] - figures[(0.02, 5.0)]).max() > 1e-3
    # and the predictions carry the planted ability, within race
    rows = got.reset_index()
    rel = rows["mu_s"] - rows.groupby("race_id")["mu_s"].transform("mean")
    ability = rows["horse_id"].map(w.ability)
    truth = ability - ability.groupby(rows["race_id"]).transform("mean")
    assert np.corrcoef(rel, truth)[0, 1] > 0.5


@pytest.mark.parametrize("decision_min", [0.0, 2.0])
def test_no_test_race_result_is_read_before_its_decision_time(decision_min):
    """The target, Tokyo R2 on the last Sunday (the last fold's test week), against the race
    before it at its own venue and day (whose result feeds its same-day variant) and the horses'
    Saturday runs (their history)."""
    w = early_world()
    sat, sun = EARLY_DAYS[-2], EARLY_DAYS[-1]
    target, before = syn.race_id(sun, "TOK", 1), syn.race_id(sun, "TOK", 0)
    off = w.table.loc[w.table["race_id"] == target, "race_utc"].iloc[0]
    t = off - pd.Timedelta(minutes=decision_min)
    published = w.table.loc[w.table["race_id"] == target, "result_available_utc"].iloc[0]

    def run(runs=None, table=None) -> pd.DataFrame:
        world = syn.World(w.runs if runs is None else runs, w.table if table is None else table,
                          w.market, w.ability)  # fmt: skip
        return keyed(predict(world, decision_min=decision_min))

    def target_mu(frame: pd.DataFrame) -> pd.Series:
        return frame.xs(target, level="race_id")["mu_s"].astype(float)

    def with_before(pub, *, wrecked: bool) -> pd.Series:
        """The target's mu with the race before it published at `pub`, its result as run or
        wrecked (every time 4 s slower)."""
        runs, table = w.runs.copy(), w.table.copy()
        for frame in (runs, table):
            frame.loc[frame["race_id"] == before, "result_available_utc"] = pub
        if wrecked:
            runs.loc[runs["race_id"] == before, "finish_time_s"] += 4.0
            table.loc[table["race_id"] == before, "top3_time_s"] += 4.0
        return target_mu(run(runs, table))

    base = run()
    mine = base.xs(target, level="race_id")
    assert (pd.to_datetime(mine["decision_utc"], utc=True) == t).all(), "t = the off - the lead"

    # the target's own result (every finish time and its top-3 time) is never read at its t
    runs, table = w.runs.copy(), w.table.copy()
    runs.loc[runs["race_id"] == target, "finish_time_s"] += 6.0
    table.loc[table["race_id"] == target, "top3_time_s"] += 6.0
    own = run(runs, table)
    early = (pd.to_datetime(base["decision_utc"], utc=True) <= published).to_numpy()
    assert early.sum() > len(mine)
    for col in ("mu_s", "sd_s"):
        same(own.loc[early, col], base.loc[early, col], atol=1e-12)

    # the race before it, published strictly before t, is read: its result moves the target
    just_before = t - pd.Timedelta(seconds=1)
    moved = with_before(just_before, wrecked=True) - with_before(just_before, wrecked=False)
    assert moved.abs().min() > 1e-3, "a result published before t is read"
    # ... and published at t or after it (before the off, when the lead allows), it is not
    late = [t, t + pd.Timedelta(hours=1)]
    if decision_min > 0:
        late.append(off - pd.Timedelta(minutes=1))
    for pub in late:
        diff = with_before(pub, wrecked=True) - with_before(pub, wrecked=False)
        assert diff.abs().max() <= 1e-12, f"a result published at {pub} (t = {t}) was read"

    # a horse's Saturday run (published the day before) is read: it moves that horse alone
    day = w.day_of()
    saturday = w.runs["race_id"].map(day).eq(sat).to_numpy()
    runners = set(w.runs.loc[w.runs["race_id"] == target, "horse_id"])
    horse = sorted(runners & set(w.runs.loc[saturday, "horse_id"]))[0]
    runs = w.runs.copy()
    runs.loc[saturday & runs["horse_id"].eq(horse).to_numpy(), "finish_time_s"] -= 3.0
    shift = target_mu(run(runs)) - target_mu(base)
    assert abs(shift[horse]) > 1e-3
    assert shift.drop(horse).abs().max() <= 1e-12


def test_runs_never_include_holdout_days():
    w = late_world()
    day = w.day_of()
    # the fixture: holdout days in every input frame, among them a holdout Saturday before a
    # development Sunday of the same test week, with horses running on both
    assert len(HELD) >= 4 and date(2027, 1, 2) in HELD and date(2027, 1, 3) not in HELD
    on = {d: set(w.runs.loc[w.runs["race_id"].map(day).eq(d).to_numpy(), "horse_id"])
          for d in (date(2027, 1, 2), date(2027, 1, 3))}  # fmt: skip
    assert len(on[date(2027, 1, 2)] & on[date(2027, 1, 3)]) > 10
    assert w.table["race_date"].isin(HELD).any()
    assert w.market["race_id"].map(day).isin(HELD).any()

    clean = keyed(predict(syn.on_days(w, DEV)))
    # every result on a holdout day wrecked: whatever reached a fit, a history or a score shows
    got = predict(syn.poisoned(w, HELD))
    assert not got["race_id"].map(day).isin(HELD).any(), "a holdout race was scored"
    got = keyed(got)
    assert got.index.equals(clean.index)
    assert (got["fold"].to_numpy() == clean["fold"].to_numpy()).all()
    for col in ("mu_s", "sd_s"):
        same(got[col], clean[col], atol=1e-9)
    scored_days = set(clean.index.get_level_values("race_id").map(day))
    assert scored_days == {d for d in DEV if d >= date(2026, 11, 21)}, "every development test day"

    # the wreck shows when it is on a development day: later predictions move
    wrecked_dev = keyed(predict(syn.poisoned(syn.on_days(w, DEV), [date(2026, 12, 5)])))
    later = np.asarray(clean.index.get_level_values("race_id").map(day) > date(2026, 12, 5))
    after, before = wrecked_dev.loc[later, "mu_s"], clean.loc[later, "mu_s"]
    assert np.abs(after.to_numpy(float) - before.to_numpy(float)).max() > 0.1


# ---------------------------------------------------------------------------------------------
# The scale: PL's s = 1 / the condlogit coefficient on -mu; Henery's its own likelihood's
# ---------------------------------------------------------------------------------------------

SCALES = (0.5, 1.5, 0.8, 2.0, 0.6)  # the planted PL scale of each fold's winners
CUT = 3  # fold 3's train_end leaves fold 2's Sunday out


def test_p_speed_scale_is_one_over_the_condlogit_coefficient():
    fitted = []
    for k, s in enumerate((0.3, 0.9, 2.5)):
        rows = syn.scale_rows(40 + k, (s, s, s), cut_fold=-1)
        got = ps.fit_scale(rows, method="pl")
        assert got == pytest.approx(syn.pl_scale(rows), rel=1e-9)
        assert ps.fit_scale(rows) == pytest.approx(got, rel=1e-12), "pl is the default"
        assert abs(got / s - 1.0) < 0.35, (got, s)  # the rows carry the planted scale
        fitted.append(got)
    assert fitted == sorted(fitted)

    rows = syn.scale_rows(5, SCALES, cut_fold=CUT)
    got = ps.score_folds(rows, method="pl")
    assert isinstance(got, pd.DataFrame)
    assert not {"race_id", "horse_id", "fold", "scale", "p_speed"} - set(got.columns)
    assert sorted(set(got["fold"])) == [1, 2, 3, 4], "fold 0 only trains"
    assert len(got) == int((rows["fold"] >= 1).sum())
    got = got[["race_id", "horse_id", "fold", "scale", "p_speed"]].merge(
        rows[["race_id", "horse_id", "mu_s"]], on=["race_id", "horse_id"], validate="one_to_one"
    )
    scales = {}
    for k in range(1, len(SCALES)):
        want = syn.pl_scale(training_rows(rows, k))
        mine = got[got["fold"] == k]
        np.testing.assert_allclose(mine["scale"].to_numpy(dtype=float), want, rtol=1e-9)
        scales[k] = want
        for _, race in mine.groupby("race_id"):
            mu = race["mu_s"].to_numpy(dtype=float)
            p = np.exp(-(mu - mu.min()) / want)
            np.testing.assert_allclose(race["p_speed"].to_numpy(dtype=float), p / p.sum(),
                                       rtol=0, atol=1e-12)  # fmt: skip
    # each fold's own training rows: the embargo cut is read, and no one pooled value passes
    assert abs(syn.pl_scale(rows[rows["fold"] < CUT]) - scales[CUT]) > 1e-3
    assert max(scales.values()) - min(scales.values()) > 0.1


def test_henery_scale_is_the_maximum_likelihood_scale_on_the_training_rows():
    rows = syn.scale_rows(5, SCALES, cut_fold=CUT)
    got = ps.score_folds(rows, method="henery")
    assert sorted(set(got["fold"])) == [1, 2, 3, 4]
    got = got[["race_id", "horse_id", "fold", "scale", "p_speed"]].merge(
        rows[["race_id", "horse_id", "mu_s"]], on=["race_id", "horse_id"], validate="one_to_one"
    )
    for k in range(1, len(SCALES)):
        train = training_rows(rows, k)
        want = syn.henery_mle(train)
        mine = got[got["fold"] == k]
        s = float(mine["scale"].iloc[0])
        np.testing.assert_allclose(mine["scale"].to_numpy(dtype=float), s, rtol=1e-12)
        assert s == pytest.approx(want, rel=2e-3), k
        assert ps.fit_scale(train, method="henery") == pytest.approx(want, rel=2e-3)
        ll = syn.henery_loglik(train, s)
        assert ll >= syn.henery_loglik(train, 0.98 * s) and ll >= syn.henery_loglik(train, 1.02 * s)
        assert abs(s - syn.pl_scale(train)) > 0.05 * s, "Henery's scale is its own"
        for _, race in mine.groupby("race_id"):
            want_p = p_speed(race["mu_s"].to_numpy(dtype=float), scale=s, method="henery")
            np.testing.assert_allclose(race["p_speed"].to_numpy(dtype=float), want_p,
                                       rtol=0, atol=1e-12)  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Standalone against D4, marginal in #87's meta next to the final price
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_standalone_is_the_paired_delta_against_the_calibrated_market_with_meeting_bounds(alpha):
    rows = syn.member_rows(11, alpha=alpha, beta=0.6)
    per_race = syn.winner_ll(rows["p_cal"], rows) - syn.winner_ll(rows["p_speed"], rows)
    iv = boot(per_race, meetings(rows))
    assert (rows.groupby("race_id")["won"].sum() > 1).sum() >= 5, "the fixture has dead heats"

    metrics, _ = ps.standalone(rows, SPEC)

    got = one(metrics, STANDALONE)
    assert got.n == rows["race_id"].nunique() == len(per_race)
    assert got.value == pytest.approx(per_race.mean(), abs=1e-9)
    assert got.lo == pytest.approx(iv.lo, abs=1e-9)
    assert got.hi == pytest.approx(iv.hi, abs=1e-9)
    assert (got.value > 0) == (alpha > 0), "positive when p_speed beats the calibrated market"
    # the bounds are the meetings': resampling races alone gives others
    by_race = boot(per_race, pd.Series(per_race.index, index=per_race.index))
    assert abs(by_race.lo - iv.lo) > 1e-6


def test_member_enters_meta_model_at_alpha_zero():
    # a member with nothing beyond the price: it tracks ln q, and the winners follow q^0.6
    null = syn.member_rows(11, alpha=0.0, beta=0.6)
    alpha, se = syn.pooled_alpha(null)
    offset_alpha, offset_se = syn.pooled_alpha(null, syn.OFFSET)
    # the fixture can tell: with ln q's coefficient fixed at 1 the member would earn weight
    assert abs(offset_alpha) > 3 * offset_se

    metrics, _ = ps.marginal(null, SPEC)

    coef = one(metrics, ALPHA)
    assert coef.value == pytest.approx(alpha, abs=1e-7)
    assert coef.n == null["race_id"].nunique()
    assert (coef.lo + coef.hi) / 2 == pytest.approx(coef.value, abs=1e-9)
    half = (coef.hi - coef.lo) / 2
    assert any(half == pytest.approx(z * se, rel=1e-6) for z in (Z_TWO_SIDED, Z_ONE_SIDED)), (
        "the bounds are alpha -/+ z * its meeting-clustered se"
    )
    assert coef.lo < 0.0 < coef.hi, "alpha = 0, the market alone, is inside the bounds"
    gain = one(metrics, MARGINAL)
    assert gain.lo <= 0.0 and abs(gain.value) < 0.01, "no gain over the market-only meta"

    # a member with weight: alpha is recovered and the gain shows
    signal = syn.member_rows(11, alpha=0.7, beta=0.6)
    alpha, se = syn.pooled_alpha(signal)
    metrics, _ = ps.marginal(signal, SPEC)
    coef = one(metrics, ALPHA)
    assert coef.value == pytest.approx(alpha, abs=1e-7)
    assert abs(coef.value - 0.7) < 4 * se
    assert coef.lo > 0.0
    assert one(metrics, MARGINAL).lo > 0.0


@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_marginal_is_the_paired_out_of_fold_delta_against_the_market_only_meta(alpha):
    rows = syn.member_rows(23, alpha=alpha, beta=0.8, cut_fold=4)
    member, market = syn.meta_oof(rows)
    scored = ~np.isnan(member)
    assert set(rows.loc[~scored, "fold"]) == {1}, "the first fold only trains the meta"
    kept = rows[scored].reset_index(drop=True)
    per_race = syn.winner_ll(market[scored], kept) - syn.winner_ll(member[scored], kept)
    iv = boot(per_race, meetings(kept))
    # the embargo cut is read: fold 4's metas on every earlier row would predict otherwise
    everything = rows.assign(train_end=rows["race_date"].max())
    assert np.abs(syn.meta_oof(everything)[0] - member)[rows["fold"].eq(4).to_numpy()].max() > 1e-6

    metrics, _ = ps.marginal(rows, SPEC)

    got = one(metrics, MARGINAL)
    assert got.n == len(per_race) == kept["race_id"].nunique()
    assert got.value == pytest.approx(per_race.mean(), abs=1e-7)
    assert got.lo == pytest.approx(iv.lo, abs=1e-7)
    assert got.hi == pytest.approx(iv.hi, abs=1e-7)


# ---------------------------------------------------------------------------------------------
# The grid: every point and method a variant, failures included; no holdout day read
# ---------------------------------------------------------------------------------------------

TEST_GRID = {"drift_var_per_day": [0.004, -1.0], "t_df": [None, 6.0]}  # -1: HorseSpeed refuses it


@pytest.fixture(scope="module")
def studies(tmp_path_factory):
    """compute_study on the late world's development days alone, and on every day with each
    holdout day's results wrecked, at a test grid with a point the models refuse."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_DIR_VAR, str(tmp_path_factory.mktemp("data")))
        spec = ps.build_spec(scheme=WEEKLY, bootstrap=BOOT, grid=TEST_GRID)
        w = late_world()
        dev, wrecked = syn.on_days(w, DEV), syn.poisoned(w, HELD)
        clean = ps.compute_study(dev.runs, dev.table, dev.market, spec=spec, scheme=WEEKLY)
        held = ps.compute_study(wrecked.runs, wrecked.table, wrecked.market, spec=spec,
                                scheme=WEEKLY)  # fmt: skip
    return spec, clean, held


def of_point(variants, drift, t_df, method) -> list:
    return [
        v for v in variants
        if v.params.get("drift_var_per_day") == drift and v.params.get("t_df") == t_df
        and v.params.get("method") == method
    ]  # fmt: skip


def test_every_grid_point_and_method_is_a_variant_failures_included(studies):
    spec, study, _ = studies
    assert spec["grid"] == TEST_GRID, "the grid as given"
    assert study.variants and all(v.hypothesis_id == "E39" for v in study.variants)
    standalone = {}
    for drift in TEST_GRID["drift_var_per_day"]:
        for t_df in TEST_GRID["t_df"]:
            for method in METHODS:
                mine = of_point(study.variants, drift, t_df, method)
                assert mine, f"no variant for drift {drift}, t_df {t_df}, {method}"
                if drift < 0:  # the models refuse it: recorded as failed, the others still run
                    assert all(v.error for v in mine) and not any(v.metrics for v in mine)
                    continue
                assert not any(v.error for v in mine), [v.error for v in mine]
                found = {(m.fold, m.metric): m for v in mine for m in v.metrics}
                for name in (STANDALONE, MARGINAL, ALPHA):
                    m = found.get(("all_folds", name))
                    assert m is not None, f"{name} missing at drift {drift}, t_df {t_df}, {method}"
                    assert m.lo is not None and m.hi is not None, f"{name} has no bounds"
                    assert all(math.isfinite(x) for x in (m.value, m.lo, m.hi))
                    assert m.lo <= m.hi and m.n > 0
                coef = found[("all_folds", ALPHA)]
                assert coef.lo < coef.value < coef.hi
                standalone[(t_df, method)] = found[("all_folds", STANDALONE)].value
    # each grid point and each method has its own figures
    assert standalone[(None, "pl")] != pytest.approx(standalone[(6.0, "pl")], abs=1e-9)
    assert standalone[(None, "pl")] != pytest.approx(standalone[(None, "henery")], abs=1e-9)


def test_the_study_reads_no_holdout_day_even_when_its_inputs_hold_them(studies):
    _, clean, held = studies
    assert sorted(v.name for v in held.variants) == sorted(v.name for v in clean.variants)
    by_name = {v.name: v for v in held.variants}
    for v in clean.variants:
        other = by_name[v.name]
        assert bool(other.error) == bool(v.error), v.name
        mine = {(m.fold, m.metric): m for m in v.metrics}
        theirs = {(m.fold, m.metric): m for m in other.metrics}
        assert set(mine) == set(theirs), v.name
        for key, m in mine.items():
            o = theirs[key]
            assert o.n == m.n, (v.name, key)
            for a, b in ((m.value, o.value), (m.lo, o.lo), (m.hi, o.hi)):
                assert (a is None) == (b is None), (v.name, key)
                if a is not None:
                    assert b == pytest.approx(a, abs=1e-9), (v.name, key)


def test_the_pre_registered_grid_spans_drift_and_t_df(tmp_path):
    scheme = load_country_scheme("japan")
    spec = ps.build_spec(scheme=scheme, bootstrap=BOOT)
    assert spec == ps.build_spec(scheme=scheme, bootstrap=BOOT, grid=ps.GRID)
    drifts, dfs = spec["grid"]["drift_var_per_day"], spec["grid"]["t_df"]
    assert len(set(drifts)) >= 2 and len(set(dfs)) >= 2
    for drift in drifts:
        assert isinstance(drift, (int, float)) and not isinstance(drift, bool)
        assert math.isfinite(drift) and drift >= 0
    for t_df in dfs:
        assert t_df is None or (not isinstance(t_df, bool) and math.isfinite(t_df) and t_df > 0)
    for drift in drifts:  # every point is one the model takes
        for t_df in dfs:
            HorseSpeed(drift_var_per_day=drift, t_df=t_df)
    # plain JSON the registry freezes, the same on a second build
    registry = Registry(tmp_path / "registry.jsonl", holdout_log=tmp_path / "no.jsonl")
    frozen = registry.freeze_family(ps.FAMILY_ID, spec)
    again = ps.build_spec(scheme=scheme, bootstrap=BOOT)
    assert registry.freeze_family(ps.FAMILY_ID, again) == frozen


# ---------------------------------------------------------------------------------------------
# The run: frozen first, one record per variant, the report through write_report
# ---------------------------------------------------------------------------------------------


def fake_store(data: Path) -> None:
    for name in ("runners", "races"):
        path = data / "processed" / "jra" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {name}".encode())


def fake_study() -> Study:
    scheme = "walk_forward:synthetic"
    components = {"study": "studies/p_speed.py"}
    study = Study(counts={"races": 40})
    study.variants += [
        Variant("E39", "pl:drift=0.004:t_df=none", scheme, components,
                {"drift_var_per_day": 0.004, "t_df": None, "method": "pl"},
                metrics=(FoldMetric("all_folds", STANDALONE, -0.05, -0.07, -0.03, 40),
                         FoldMetric("all_folds", ALPHA, 0.12, -0.2, 0.44, 40))),
        Variant("E39", "henery:drift=0.004:t_df=6", scheme, components,
                {"drift_var_per_day": 0.004, "t_df": 6.0, "method": "henery"},
                metrics=(FoldMetric("all_folds", MARGINAL, -0.001, -0.004, 0.002, 30),)),
        Variant("E39", "pl:drift=-1:t_df=none", scheme, components,
                {"drift_var_per_day": -1.0, "t_df": None, "method": "pl"},
                error="ValueError: drift_var_per_day must be a finite number >= 0: -1.0"),
    ]  # fmt: skip
    return study


def test_run_freezes_the_family_before_any_read_and_records_every_variant(tmp_path, monkeypatch):
    assert ps.TICKET == 347
    assert isinstance(ps.FAMILY_ID, str) and ps.FAMILY_ID.strip()
    fake_store(tmp_path / "data")
    seen = []

    def load_inputs(*args, **kwargs):
        seen.append(default_registry("japan").frozen_spec_hash(ps.FAMILY_ID))
        w = early_world()
        return w.runs.copy(), w.table.copy(), w.market.copy()

    study = fake_study()
    monkeypatch.setattr(ps, "load_inputs", load_inputs)
    monkeypatch.setattr(ps, "compute_study", lambda *args, **kwargs: study)
    written = []
    real_write = reports.write_report

    def spy(path, text, **kwargs):
        written.append(Path(path))
        return real_write(path, text, **kwargs)

    monkeypatch.setattr(reports, "write_report", spy)
    monkeypatch.setattr(ps, "write_report", spy, raising=False)
    summary = tmp_path / "summary"
    monkeypatch.setattr(registry_mod, "summary_dir", lambda country: summary)
    monkeypatch.setattr(ps, "summary_dir", lambda country: summary, raising=False)

    inside = REPO_ROOT / "p-speed-report-must-not-be-written"
    with pytest.raises(paths.DataDirInsideRepo):
        ps.run_study(out_dir=inside)
    assert not inside.exists()
    assert default_registry("japan").records(ps.FAMILY_ID) == []

    result = ps.run_study(out_dir=tmp_path / "out")

    registry = default_registry("japan")
    assert seen and all(frozen is not None for frozen in seen), "frozen before any read"
    assert registry.frozen_spec_hash(ps.FAMILY_ID) == result["spec_hash"] == seen[-1]
    assert result["family_id"] == ps.FAMILY_ID
    records = registry.records(ps.FAMILY_ID)
    by_name = {r.variant: r for r in records}
    assert len(records) == len(by_name) == len(study.variants), "one record per variant"
    for v in study.variants:
        r = by_name[v.name]
        assert (r.ticket, r.family_id, r.hypothesis_id) == (347, ps.FAMILY_ID, v.hypothesis_id)
        assert r.status == ("failed" if v.error else "scored"), v.name
        assert all(r.params.get(key) == value for key, value in v.params.items()), v.name
        assert r.metrics == v.metrics, v.name
    report = Path(result["report_path"])
    assert report.is_file() and report.resolve().parent == (tmp_path / "out").resolve()
    assert any(p.resolve() == report.resolve() for p in written), "written through write_report"
