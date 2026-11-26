"""E32: the same-day draw and position bias as of t, at the final price (ticket #348, epic #73).

The owner's answer on #97 (2026-11-26, item 4): E32 uses "one shrunk bias column from #138's
within_day_posterior". Contract pinned here (`horsetrack.countries.japan.studies.same_day_bias`):

- `DECISION_LEAD`: a timedelta; a race's decision time is t = post - DECISION_LEAD, post being
  post_time_local on the Asia/Tokyo wall clock of race_date. It is D2's primary decision time
  (config/decision.yaml `primary_min`, T-2).
- `runner_bias(prior, races, runners, *, race_id, decision_time, published, styles=None) ->
  pl.DataFrame`: one row per declared runner of `race_id` with `race_id`, `horse_no` (Int64) and
  `bias`. `prior`, `races`, `runners`, `decision_time` and `published` are #138's
  (`track_bias.track_bias_features`). `styles` is a frame of #129's running style per runner
  (`race_id`, `horse_no`, `style` in front / stalker / midpack / closer, null for none), the
  style read from earlier races; None means no runner has one. The column is the runner's draw
  group posterior mean (`draw_adj`) plus the posterior mean of its own style (`style_adj_<style>`),
  the style term 0 without a style.
- `day_rows(prior, races, runners, styles=None) -> pl.DataFrame`: for every race of `races` with a
  post time, its declared runners with `race_id`, `meeting_id`, `race_date`, `horse_no`, `won`
  (finish_pos == 1), `win_odds` (the final Win price) and `bias`, each race's bias taken at its own
  t = post - DECISION_LEAD with `published = track_bias_report.published_times(races)` (the logged
  result_published_utc, else post + 30 min).
- `fit_coefficient(rows, *, ridge=0.0) -> fit` on a pandas frame with `race_id`, `meeting_id`,
  `won`, `win_odds`, `bias`: the conditional logit u = ln q + coef * bias, q the final Win price
  normalised per race (q = (1/odds) / sum over the race of 1/odds), ln q the offset at
  coefficient 1. The fit has `coef`, `se_cluster` (the cluster sandwich over `meeting_id`, D7's
  unit), `lower` / `upper` (coef -/+ 1.96 se_cluster, 95% two-sided), `races` and `meetings`.
- `score_folds(rows, folds) -> pd.DataFrame`: `rows` as above plus `race_date` and `fold` (the
  index into `folds` of the fold whose test window holds the row and whose prior built its bias).
  For each fold k >= 1 the coefficient is `fit_coefficient` on the rows with fold < k and
  race_date <= folds[k].train_end, and it scores the rows with fold == k; fold 0 only trains. One
  row per scored race: `race_id`, `meeting_id`, `fold`, `coef` (the fold's), `ll_market`
  (-ln q of the winner, the market baseline: the offset alone) and `ll_model` (-ln p of the
  winner, p = softmax over the race of ln q + coef * bias).
- `run_study(date_from, date_to, *, scheme=None, out_dir=None) -> dict`: freezes `FAMILY_ID`'s
  spec in the run registry before any store row is read, reads the JRA store (holdout days never
  read), appends registry records (`TICKET == 348`) and writes the report through `write_report`
  under `out_dir`. The result has `family_id`, `spec_hash`, `report_path`, `records_appended`,
  `failed_variants`, `coefficient` (estimate, se_cluster, lower, upper, races, meetings) and
  `delta_ll` (mean, races, meetings); some registry record carries the coefficient with its CI as
  a metric (value, lo, hi).

Rulings assumed here (reversible):
- Ruling: the E32 column is draw_adj + style_adj of the runner's own #129 style (0 without one),
  the owner's "one shrunk bias column" read as one column per runner.
- Ruling: the coefficient is the unpenalised maximum likelihood (ridge 0), so its interval is the
  plain meeting sandwich.
- Ruling: ΔLL is against the market q (the offset alone), the nested model's own baseline.

Data: synthetic only (handicap_store.py), in memory or under tmp_path. Expected values come from
#138's `track_bias_features` (the feature's parts), `models.condlogit.fit_conditional_logit` (the
reference fit, with q normalised here by hand), and the planted outcomes; never from the module
under test.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import copy
import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset
import pytest

from handicap_store import R, Store, T, at_jst, weekend_days
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import default_registry
from horsetrack.countries.japan.handicap.track_bias import fit_bias_prior, track_bias_features
from horsetrack.countries.japan.handicap.track_bias_report import published_times
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.studies import same_day_bias as e32

STYLES = ("front", "stalker", "midpack", "closer")
Z95 = 1.959963984540054
TRAIN_END = date(2025, 6, 8)
HISTORY = weekend_days(date(2025, 1, 4), TRAIN_END)[-20:]
CELL = dict(venue="TOK", surface="turf", distance=1600, going="GOOD", course="A", rail=None)
INNER_FRONT = ({"inner": 1.5, "outer": -1.5}, {"front": 1.2, "stalker": 0.4, "midpack": -0.4,
                                               "closer": -1.2})  # fmt: skip
OUTER_CLOSER = ({"inner": -1.5, "outer": 1.5}, {"front": -1.2, "stalker": -0.4, "midpack": 0.4,
                                                "closer": 1.2})  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Synthetic race days (the #138 tests' generator, written out again here)
# ---------------------------------------------------------------------------------------------


def draw_group(draw: int, n: int) -> str:
    if 3 * (draw - 1) <= n - 1:
        return "inner"
    if 3 * (draw - 1) <= 2 * (n - 1):
        return "middle"
    return "outer"


def style_of(p: int, n: int) -> str:
    if p == 1:
        return "front"
    if 3 * (p - 1) <= n - 1:
        return "stalker"
    if 3 * (p - 1) <= 2 * (n - 1):
        return "midpack"
    return "closer"


def post_hhmm(race_no: int) -> str:
    """Race k posts at 09:00 + 30k local."""
    minutes = 9 * 60 + 30 * race_no
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def post_of(day: date, race_no: int) -> datetime:
    return at_jst(day, 9) + timedelta(minutes=30 * race_no)


def biased_race(s: Store, rng: random.Random, day: date, plant, *, field=12, race_no=None,
                **cell) -> str:  # fmt: skip
    """Finishing order Plackett-Luce from exp(draw effect + style effect); corners random."""
    cell = {**CELL, **cell}
    draw_beta, style_beta = plant
    corners = list(range(1, field + 1))
    rng.shuffle(corners)
    nos = list(range(1, field + 1))
    strength = [
        math.exp(draw_beta.get(draw_group(no, field), 0.0)
                 + style_beta.get(style_of(corners[i], field), 0.0))
        for i, no in enumerate(nos)
    ]  # fmt: skip
    left, order = list(range(field)), []
    while left:
        u = rng.random() * sum(strength[i] for i in left)
        for i in left:
            u -= strength[i]
            if u <= 0:
                break
        order.append(i)
        left.remove(i)
    pos = {i: k + 1 for k, i in enumerate(order)}
    base = len(s.runners)
    runners = [
        R(f"h:{base + i}", pos=pos[i], corners=[corners[i], corners[i]], no=no)
        for i, no in enumerate(nos)
    ]
    rid = s.race(
        day, runners, race_no=race_no, venue=cell["venue"], surface=cell["surface"],
        distance=cell["distance"], going=cell["going"], course=cell["course"], rail=cell["rail"],
    )  # fmt: skip
    for race in s.races:
        if race["race_id"] == rid:
            race["post_time_local"] = post_hhmm(race["race_no"])
            race["result_published_utc"] = None
    return rid


def frames(s: Store) -> tuple[pl.DataFrame, pl.DataFrame]:
    overrides = {"rail_setting": pl.String, "result_published_utc": pl.Datetime("us", "UTC")}
    races = pl.DataFrame(s.races, infer_schema_length=None, schema_overrides=overrides)
    return races, pl.DataFrame(s.runners, infer_schema_length=None)


def reversed_result(s: Store, race_id: str) -> Store:
    """A copy of `s` with the race's finishing order and first-corner order reversed."""
    other = copy.deepcopy(s)
    rows = [r for r in other.runners if r["race_id"] == race_id and r["finish_pos"] is not None]
    m = len(rows)
    for r in rows:
        r["finish_pos"] = m + 1 - r["finish_pos"]
        r["running_positions"] = [m + 1 - p for p in r["running_positions"]]
    return other


def styles_frame(s: Store, race_id: str, pattern) -> pl.DataFrame:
    """Each declared runner of the race gets pattern[k % len(pattern)] (None: no style)."""
    rows = [r for r in s.runners if r["race_id"] == race_id and not r["is_scratched"]]
    return pl.DataFrame(
        {
            "race_id": [r["race_id"] for r in rows],
            "horse_no": [r["horse_no"] for r in rows],
            "style": [pattern[k % len(pattern)] for k in range(len(rows))],
        },
        schema={"race_id": pl.String, "horse_no": pl.Int64, "style": pl.String},
    )


@pytest.fixture(scope="module")
def day_store():
    """20 history days with an inner-front bias, then T with six races under the opposite bias."""
    rng = random.Random(348)
    s = Store()
    for day in HISTORY:
        for k in range(1, 5):
            biased_race(s, rng, day, INNER_FRONT, race_no=k)
    today = [biased_race(s, rng, T, OUTER_CLOSER, race_no=k) for k in range(1, 7)]
    races, runners = frames(s)
    prior = fit_bias_prior(races, runners, train_end=TRAIN_END, day_sd=0.3)
    return s, today, prior


def pub_time(race_no: int) -> datetime:
    """Race k's official result: 12 minutes after its post."""
    return post_of(T, race_no) + timedelta(minutes=12)


def decision(race_no: int) -> datetime:
    return post_of(T, race_no) - timedelta(minutes=2)


def bias_of(prior, s: Store, target: str, *, decision_time, published, styles=None) -> dict:
    races, runners = frames(s)
    got = e32.runner_bias(
        prior, races, runners, race_id=target, decision_time=decision_time, published=published,
        styles=styles,
    )  # fmt: skip
    assert isinstance(got, pl.DataFrame)
    missing = {"race_id", "horse_no", "bias"} - set(got.columns)
    assert not missing, f"runner_bias lacks {missing}"
    assert got.get_column("race_id").unique().to_list() == [target]
    return dict(zip(got.get_column("horse_no").to_list(), got.get_column("bias").to_list(),
                    strict=True))  # fmt: skip


def differs(a: dict, b: dict, tol: float = 1e-9) -> bool:
    assert set(a) == set(b)
    return any(abs(a[k] - b[k]) > tol for k in a)


def same(a: dict, b: dict) -> bool:
    assert set(a) == set(b)
    return all(a[k] == pytest.approx(b[k], abs=1e-12) for k in a)


# ---------------------------------------------------------------------------------------------
# The feature: one shrunk column from #138's posterior, read only from races resulted before t
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("target_no", [3, 5, 6])
@pytest.mark.parametrize(
    "pattern",
    [(None,), ("front", "stalker", "midpack", "closer", None), ("closer", "front", None, None)],
)
def test_e32_feature_is_the_draw_posterior_plus_the_own_style_posterior(day_store, target_no,
                                                                        pattern):  # fmt: skip
    s, today, prior = day_store
    target = today[target_no - 1]
    published = {rid: pub_time(k + 1) for k, rid in enumerate(today)}
    races, runners = frames(s)
    styles = styles_frame(s, target, pattern)
    # a style row of another race with the same horse numbers must not apply to this race
    given = pl.concat([styles, styles_frame(s, today[0], ("front",))])
    parts = track_bias_features(
        prior, races, runners, race_id=target, decision_time=decision(target_no),
        published=published,
    ).to_dicts()  # fmt: skip
    style = dict(zip(styles.get_column("horse_no").to_list(), styles.get_column("style").to_list(),
                     strict=True))  # fmt: skip
    want = {
        p["horse_no"]: p["draw_adj"] + (p[f"style_adj_{style[p['horse_no']]}"]
                                        if style[p["horse_no"]] is not None else 0.0)
        for p in parts
    }  # fmt: skip
    got = bias_of(prior, s, target, decision_time=decision(target_no), published=published,
                  styles=given)  # fmt: skip
    assert set(got) == set(want)
    for no, value in want.items():
        assert got[no] == pytest.approx(value, abs=1e-12), (target, no, style[no])
    # the style term is not constant across runners when styles differ, so it is really read
    if len({v for v in pattern}) > 1:
        assert len({round(v, 12) for v in got.values()}) > 3
    # without styles the column is the draw posterior alone
    alone = bias_of(prior, s, target, decision_time=decision(target_no), published=published)
    for p in parts:
        assert alone[p["horse_no"]] == pytest.approx(p["draw_adj"], abs=1e-12)


def test_e32_feature_reads_only_races_resulted_before_t(day_store):
    s, today, prior = day_store
    target_no = 5
    target = today[target_no - 1]
    t = decision(target_no)
    styles = styles_frame(s, target, ("front", "stalker", "midpack", "closer", None))
    published = {rid: pub_time(k + 1) for k, rid in enumerate(today)}
    assert all(published[today[k]] < t for k in range(4))  # races 1-4 resulted before t

    def feature(store: Store, pub: dict) -> dict:
        return bias_of(prior, store, target, decision_time=t, published=pub, styles=styles)

    base = feature(s, published)

    # an earlier race resulted strictly before t moves the feature
    for k in (1, 2, 4):
        flipped = reversed_result(s, today[k - 1])
        assert differs(feature(flipped, published), base), f"race {k}'s result is not read"

    # race 4 published exactly at t, or after it, is not read: its result changes nothing ...
    for when in (t, t + timedelta(seconds=1), t + timedelta(minutes=40)):
        late = {**published, today[3]: when}
        without = {rid: p for rid, p in published.items() if rid != today[3]}
        assert same(feature(s, late), feature(s, without)), when
        assert same(feature(reversed_result(s, today[3]), late), feature(s, late)), when
    # ... while one second before t it is
    just_before = {**published, today[3]: t - timedelta(seconds=1)}
    assert differs(feature(reversed_result(s, today[3]), just_before), feature(s, just_before))
    # instants compare whatever the time zone: t - 1 s written in UTC is still before t
    in_utc = {**published, today[3]: (t - timedelta(seconds=1)).astimezone(timezone.utc)}
    assert same(feature(s, in_utc), feature(s, just_before))

    # a later race of the day is never read, even with a result stamped before t
    early_later = {**published, today[5]: t - timedelta(hours=1)}
    assert same(feature(reversed_result(s, today[5]), early_later), feature(s, early_later))
    # nor the target race's own result (finish and first corner), even stamped before t
    early_own = {**published, target: t - timedelta(hours=1)}
    assert same(feature(reversed_result(s, target), early_own), feature(s, early_own))
    assert same(feature(s, early_own), base)


def test_day_rows_take_each_race_at_post_minus_the_primary_lead(day_store):
    s, today, prior = day_store
    primary = load_country_yaml("japan", "decision")["primary_min"]
    assert e32.DECISION_LEAD == timedelta(minutes=primary)
    target_no = 5
    target = today[target_no - 1]
    t = post_of(T, target_no) - e32.DECISION_LEAD
    day = frames(s)[0].filter(pl.col("race_date") == T)
    runners = frames(s)[1].filter(pl.col("race_date") == T)

    def rows_with(logged: dict) -> pl.DataFrame:
        logged = {rid: when.astimezone(timezone.utc) for rid, when in logged.items()}
        races = day.with_columns(
            pl.col("race_id")
            .replace_strict(logged, default=None, return_dtype=pl.Datetime("us", "UTC"))
            .alias("result_published_utc")
        )
        got = e32.day_rows(prior, races, runners)
        assert isinstance(got, pl.DataFrame)
        need = {"race_id", "meeting_id", "race_date", "horse_no", "won", "win_odds", "bias"}
        assert not need - set(got.columns), f"day_rows lacks {need - set(got.columns)}"
        return got

    def target_bias(rows: pl.DataFrame) -> dict:
        rows = rows.filter(pl.col("race_id") == target)
        return dict(zip(rows.get_column("horse_no").to_list(), rows.get_column("bias").to_list(),
                        strict=True))  # fmt: skip

    def direct(published: dict) -> dict:
        return bias_of(prior, s, target, decision_time=t, published=published)

    logged = {rid: pub_time(k + 1).astimezone(timezone.utc) for k, rid in enumerate(today)}
    upto3 = {rid: p for rid, p in logged.items() if rid in today[:3]}
    upto4 = {rid: p for rid, p in logged.items() if rid in today[:4]}
    assert differs(direct(upto3), direct(upto4))  # the fixture can tell race 4 in from out

    # race 4 logged exactly at t: out; one second before t: in
    at_t = target_bias(rows_with({**logged, today[3]: t.astimezone(timezone.utc)}))
    assert same(at_t, direct(upto3))
    before = target_bias(rows_with({**logged, today[3]: (t - timedelta(seconds=1))}))
    assert same(before, direct({**upto3, today[3]: t - timedelta(seconds=1)}))
    assert differs(before, at_t)

    # nothing logged: the proxy post + 30 min; race 3's (11:00) is before t, race 4's (11:30) not
    proxy = rows_with({})
    assert same(target_bias(proxy), direct({rid: post_of(T, k + 1) + timedelta(minutes=30)
                                            for k, rid in enumerate(today[:3])}))  # fmt: skip
    # every race of the day is there at its own t, with its outcome and final price
    assert sorted(proxy.get_column("race_id").unique().to_list()) == sorted(today)
    for k, rid in enumerate(today):
        mine = proxy.filter(pl.col("race_id") == rid)
        got = dict(zip(mine.get_column("horse_no").to_list(), mine.get_column("bias").to_list(),
                       strict=True))  # fmt: skip
        pub = published_times(day)
        want = bias_of(prior, s, rid, decision_time=post_of(T, k + 1) - e32.DECISION_LEAD,
                       published=pub)  # fmt: skip
        assert same(got, want), rid
    src = {(r["race_id"], r["horse_no"]): r for r in s.runners if r["race_date"] == T}
    for row in proxy.to_dicts():
        r = src[(row["race_id"], row["horse_no"])]
        assert row["won"] == (r["finish_pos"] == 1)
        assert row["win_odds"] == pytest.approx(r["win_odds"])
        assert row["meeting_id"] == f"jra:{T}:TOK"


# ---------------------------------------------------------------------------------------------
# The conditional logit: ln q of the final price is the offset; meetings are the clusters
# ---------------------------------------------------------------------------------------------


def market_rows(seed: int, *, beta=0.0, meeting_sd=0.0, n_weeks=15, races=8, field=8,
                on_lnq=0.5, overround=1.25) -> tuple[pd.DataFrame, np.ndarray]:  # fmt: skip
    """Two meetings (TOK, NAK) on each Saturday and Sunday of `n_weeks` weeks. Per runner a market
    q (softmax of N(0, 1)) and a bias = on_lnq * ln q + N(0, 0.6^2); the winner is drawn from
    softmax(ln q + slope * bias), slope = beta (a float, or a callable of the week) plus a
    meeting effect N(0, meeting_sd^2). The odds carry the overround: 1/odds = overround * q.
    Returns the rows and the true q per row."""
    rng = np.random.default_rng(seed)
    out, qs = [], []
    first = date(2025, 1, 4)
    for w in range(n_weeks):
        for d in (0, 1):
            day = first + timedelta(days=7 * w + d)
            for venue in ("TOK", "NAK"):
                meeting = f"jra:{day.isoformat()}:{venue}"
                slope = (beta(w) if callable(beta) else beta) + meeting_sd * rng.standard_normal()
                for k in range(1, races + 1):
                    s = rng.standard_normal(field)
                    q = np.exp(s) / np.exp(s).sum()
                    bias = on_lnq * np.log(q) + 0.6 * rng.standard_normal(field)
                    u = np.log(q) + slope * bias
                    p = np.exp(u - u.max())
                    p /= p.sum()
                    winner = rng.choice(field, p=p)
                    for i in range(field):
                        out.append({
                            "race_id": f"{meeting}:{k:02d}", "meeting_id": meeting,
                            "race_date": day, "horse_no": i + 1, "won": bool(i == winner),
                            "win_odds": float(1.0 / (overround * q[i])), "bias": float(bias[i]),
                        })  # fmt: skip
                        qs.append(q[i])
    return pd.DataFrame(out), np.asarray(qs)


def ln_q(rows: pd.DataFrame) -> np.ndarray:
    """ln of the normalised final price, by hand."""
    inv = 1.0 / rows["win_odds"]
    return np.log((inv / inv.groupby(rows["race_id"]).transform("sum")).to_numpy())


def reference(rows: pd.DataFrame, *, offset=True, cluster="meeting_id"):
    return fit_conditional_logit(
        rows[["bias"]].to_numpy(), rows["race_id"].to_numpy(), rows["won"].to_numpy(),
        offset=ln_q(rows) if offset else None, cluster=rows[cluster].to_numpy(),
    )  # fmt: skip


@pytest.mark.parametrize("beta", [0.0, 1.5, -1.0, 0.6])
def test_e32_offset_is_the_market(beta):
    rows, q = market_rows(31 + int(10 * beta), beta=beta)
    assert np.allclose(np.exp(ln_q(rows)), q)  # the hand normalisation recovers the planted q
    fit = e32.fit_coefficient(rows)
    ref = reference(rows)
    # the fit is u = ln q + coef * bias: the offset at coefficient 1, no free market weight
    assert fit.coef == pytest.approx(ref.coef[0], abs=1e-8)
    assert fit.se_cluster == pytest.approx(ref.se_cluster[0], rel=1e-6)
    assert abs(fit.coef - beta) < 4 * fit.se_cluster, (fit.coef, beta)
    if beta == 0.0:
        # winners follow q exactly: the bias adds nothing over the market
        assert fit.lower < 0.0 < fit.upper
        # and the fixture can tell: without the offset the bias (it tracks ln q) looks strong
        free = reference(rows, offset=False)
        assert free.coef[0] > 6 * free.se_cluster[0]

    # the overround is not the market: a race's odds scaled by any constant change nothing
    scale = rows["race_id"].map({rid: 0.8 + 0.05 * (k % 9) for k, rid in
                                 enumerate(rows["race_id"].unique())})  # fmt: skip
    rescaled = e32.fit_coefficient(rows.assign(win_odds=rows["win_odds"] * scale))
    assert rescaled.coef == pytest.approx(fit.coef, abs=1e-9)
    assert rescaled.se_cluster == pytest.approx(fit.se_cluster, rel=1e-9)


def test_e32_cluster_unit_is_the_meeting():
    # each meeting has its own slope, so a meeting's races move together
    rows, _ = market_rows(77, beta=0.4, meeting_sd=1.2, races=12)
    by_meeting = reference(rows, cluster="meeting_id").se_cluster[0]
    by_race = reference(rows, cluster="race_id").se_cluster[0]
    by_day = reference(rows.assign(day=rows["race_date"].astype(str)), cluster="day").se_cluster[0]
    assert by_meeting > 1.3 * by_race  # the fixture's meetings are correlated
    assert abs(by_day - by_meeting) > 0.02 * by_meeting  # a day is not a meeting

    fit = e32.fit_coefficient(rows)
    assert fit.se_cluster == pytest.approx(by_meeting, rel=1e-6)
    assert fit.lower == pytest.approx(fit.coef - Z95 * fit.se_cluster, rel=1e-4, abs=1e-9)
    assert fit.upper == pytest.approx(fit.coef + Z95 * fit.se_cluster, rel=1e-4, abs=1e-9)
    assert (fit.upper - fit.lower) / (2 * fit.se_cluster) == pytest.approx(Z95, rel=1e-4)
    assert fit.races == rows["race_id"].nunique()
    assert fit.meetings == rows["meeting_id"].nunique()

    # the clusters are the rows' meeting ids, whatever their labels
    codes = {m: f"m{k:03d}" for k, m in enumerate(sorted(rows["meeting_id"].unique(),
                                                         reverse=True))}  # fmt: skip
    relabelled = e32.fit_coefficient(rows.assign(meeting_id=rows["meeting_id"].map(codes)))
    assert relabelled.se_cluster == pytest.approx(fit.se_cluster, rel=1e-9)
    # pairing up meetings is another clustering, and another interval
    merged = rows.assign(meeting_id=rows["race_date"].astype(str))
    assert e32.fit_coefficient(merged).se_cluster == pytest.approx(by_day, rel=1e-6)
    assert e32.fit_coefficient(merged).meetings == rows["race_date"].nunique()


# ---------------------------------------------------------------------------------------------
# Walk-forward: each fold's coefficient from its training rows, scored on its test races
# ---------------------------------------------------------------------------------------------

WEEKLY_EMBARGO = FoldScheme(unit="week", test_length=1, step=1, min_train=3, window="expanding",
                            embargo_days=7, anchor_month=1)  # fmt: skip


def folded(rows: pd.DataFrame):
    folds = walk_forward(rows["race_date"].to_list(), scheme=WEEKLY_EMBARGO)
    label = np.full(len(rows), -1)
    for k, fold in enumerate(folds):
        label[fold.test_idx] = k
    rows = rows.assign(fold=label)
    return rows[rows["fold"] >= 0].reset_index(drop=True), folds


def test_each_fold_coefficient_is_fitted_on_its_training_rows_and_scored_on_its_test_races():
    # the planted slope changes week to week, so each fold's training rows give another value
    raw, _ = market_rows(5, beta=lambda w: (2.0, -1.0, 0.5)[w % 3], n_weeks=12)
    rows, folds = folded(raw)
    assert len(folds) >= 5
    got = e32.score_folds(rows, folds)
    assert isinstance(got, pd.DataFrame)
    need = {"race_id", "meeting_id", "fold", "coef", "ll_market", "ll_model"}
    assert not need - set(got.columns), f"score_folds lacks {need - set(got.columns)}"

    scored = rows[rows["fold"] >= 1]
    assert sorted(got["race_id"]) == sorted(scored["race_id"].unique())  # once each; fold 0 trains
    coefs = {}
    for k in range(1, len(folds)):
        train = rows[(rows["fold"] < k) & (rows["race_date"] <= folds[k].train_end)]
        # the embargo keeps some earlier-fold rows out of training
        assert len(train) < (rows["fold"] < k).sum()
        coefs[k] = reference(train).coef[0]
        mine = got[got["fold"] == k]
        assert len(mine) == scored[scored["fold"] == k]["race_id"].nunique()
        assert np.allclose(mine["coef"], coefs[k], atol=1e-8), k
    assert max(coefs.values()) - min(coefs.values()) > 0.1  # one pooled value cannot pass

    lq = pd.Series(ln_q(rows), index=rows.index)
    for row in got.itertuples():
        race = rows[rows["race_id"] == row.race_id]
        w = race["won"].to_numpy()
        u = lq[race.index].to_numpy() + row.coef * race["bias"].to_numpy()
        logp = u - (np.log(np.exp(u - u.max()).sum()) + u.max())
        assert row.ll_market == pytest.approx(-lq[race.index].to_numpy()[w].mean(), abs=1e-9)
        assert row.ll_model == pytest.approx(-logp[w].mean(), abs=1e-9)
        assert row.meeting_id == race["meeting_id"].iloc[0]
        assert row.fold == race["fold"].iloc[0]


# ---------------------------------------------------------------------------------------------
# The run: the family frozen first, holdout days never read, E32 recorded with its CI
# ---------------------------------------------------------------------------------------------

RUN_DAYS = weekend_days(date(2025, 1, 4), date(2025, 3, 2))
HELD = date(2026, 11, 29)  # a holdout race day of the committed definition (holdout.yaml)
WEEKLY = FoldScheme(unit="week", test_length=1, step=1, min_train=4, window="expanding",
                    embargo_days=1, anchor_month=1)  # fmt: skip


def run_store(days) -> Store:
    rng = random.Random(9)
    s = Store()
    for day in days:
        for venue in ("TOK", "NAK"):
            for k in range(1, 5):
                biased_race(s, rng, day, INNER_FRONT, field=10, race_no=k, venue=venue)
    for r in s.runners:  # a final price per runner, the same whatever else is in the store
        r["win_odds"] = round(random.Random(f"{r['race_id']}:{r['horse_no']}").uniform(1.5, 80), 1)
    return s


def write_run_store(s: Store, tmp_path: Path, monkeypatch, sub: str) -> Path:
    data = s.write(tmp_path, monkeypatch, sub=sub)
    style_dir = data / "features" / "jra" / "handicap"
    style_dir.mkdir(parents=True)
    for day in sorted({r["race_date"] for r in s.runners}):
        rows = [r for r in s.runners if r["race_date"] == day]
        pl.DataFrame(
            {
                "race_id": [r["race_id"] for r in rows],
                "horse_no": [r["horse_no"] for r in rows],
                "style": [(*STYLES, None)[random.Random(r["horse_id"]).randrange(5)] for r in rows],
            },
            schema={"race_id": pl.String, "horse_no": pl.Int64, "style": pl.String},
        ).write_parquet(style_dir / f"{day.isoformat()}.parquet")
    return data


def spy_on_store_reads(monkeypatch) -> list:
    seen = []
    real = pyarrow.dataset.dataset

    def dataset(*args, **kwargs):
        seen.append(default_registry("japan").frozen_spec_hash(e32.FAMILY_ID))
        return real(*args, **kwargs)

    monkeypatch.setattr(pyarrow.dataset, "dataset", dataset)
    return seen


def test_run_freezes_the_family_first_and_records_e32_with_its_coefficient_ci(
    tmp_path, monkeypatch
):
    assert isinstance(e32.FAMILY_ID, str) and e32.FAMILY_ID.strip()
    assert e32.TICKET == 348
    write_run_store(run_store(RUN_DAYS), tmp_path, monkeypatch, "data")
    seen = spy_on_store_reads(monkeypatch)

    result = e32.run_study(RUN_DAYS[0], RUN_DAYS[-1], scheme=WEEKLY, out_dir=tmp_path / "out")

    assert seen and all(frozen is not None for frozen in seen)  # frozen before every read
    assert result["family_id"] == e32.FAMILY_ID
    assert result["failed_variants"] == []
    registry = default_registry("japan")
    assert registry.frozen_spec_hash(e32.FAMILY_ID) == result["spec_hash"]
    records = registry.records(e32.FAMILY_ID)
    assert len(records) == result["records_appended"] > 0
    assert all(r.ticket == 348 for r in records)

    c = result["coefficient"]
    assert all(math.isfinite(c[k]) for k in ("estimate", "se_cluster", "lower", "upper"))
    assert c["se_cluster"] > 0
    assert c["lower"] == pytest.approx(c["estimate"] - Z95 * c["se_cluster"], rel=1e-4)
    assert c["upper"] == pytest.approx(c["estimate"] + Z95 * c["se_cluster"], rel=1e-4)
    assert 0 < c["races"] <= 8 * len(RUN_DAYS)
    assert 1 < c["meetings"] <= 2 * len(RUN_DAYS)
    metrics = [m for r in records if r.status == "scored" for m in r.metrics]
    assert any(
        m.value == pytest.approx(c["estimate"], abs=1e-12)
        and m.lo == pytest.approx(c["lower"], abs=1e-12)
        and m.hi == pytest.approx(c["upper"], abs=1e-12)
        for m in metrics
    ), "no scored record carries the coefficient with its CI"

    d = result["delta_ll"]
    assert math.isfinite(d["mean"])
    assert 0 < d["races"] < 8 * len(RUN_DAYS)  # the first fold only trains
    assert 1 < d["meetings"]

    report = Path(result["report_path"])
    assert report.resolve().parent == (tmp_path / "out").resolve()
    assert result["spec_hash"] in report.read_text(encoding="utf-8")


def test_run_never_reads_a_holdout_day(tmp_path, monkeypatch):
    assert is_holdout_day("japan", HELD)
    assert not any(is_holdout_day("japan", d) for d in RUN_DAYS)
    runs = {}
    for name, days in (("plain", RUN_DAYS), ("with_holdout", [*RUN_DAYS, HELD])):
        with monkeypatch.context() as mp:
            write_run_store(run_store(days), tmp_path, mp, name)
            runs[name] = e32.run_study(RUN_DAYS[0], HELD + timedelta(days=1), scheme=WEEKLY,
                                       out_dir=tmp_path / f"out-{name}")  # fmt: skip
    plain, held = runs["plain"], runs["with_holdout"]
    for key in ("estimate", "se_cluster", "lower", "upper", "races", "meetings"):
        assert held["coefficient"][key] == pytest.approx(plain["coefficient"][key], abs=1e-12), key
    for key in ("mean", "races", "meetings"):
        assert held["delta_ll"][key] == pytest.approx(plain["delta_ll"][key], abs=1e-12), key
