"""Synthetic worlds for the p_speed study tests (ticket #347, `test_p_speed_study.py`).

Nothing here imports the module under test, so the fixtures' own properties can be checked on
their own. Every frame is synthetic and every RNG seeded; nothing reads a store.

- `world(days, ...)`: race days with two venues (two meetings a day), each race a field of eight
  from a pool of horses with a fixed planted ability. Finish times are par + the (day, venue)
  variant + ability + 0.15 s/kg * (weight - 56) + noise; the winner is the quickest. Three
  frames, in the columns the study's real builders give:
  - `runs`, `models.speed_runs.build_runs`'s frame (HorseSpeed's runs: race_id, race_utc,
    horse_id, style, early_pos_mean3, weight_kg, draw, #305's par keys, finish_time_s,
    result_available_utc = post + 30 min);
  - `table`, `studies.speed_run.build_race_table`'s frame (SpeedModel's races: race_id,
    meeting_id, race_date, post_utc, race_utc, result_available_utc, top3_time_s,
    complete_timing and the par keys);
  - `market`, `studies.meta.store_frame`'s frame (priced starters: race_id, race_date, horse_id,
    win_odds, won, dead_heat, meeting_id, q), the final price informative about ability.
- `scale_rows` and `member_rows`: per-runner rows for the pure scoring functions (planted
  expected times with a PL scale per fold; a member and a market with planted weights).
- Reference computations from existing library code only (#96's `SpeedModel`, #97's
  `HorseSpeed` and `p_speed`, #87's `fit_meta` and `fit_conditional_logit`) and first
  principles, never from the study module: `expected_predictions`, `winner_ll`, `pooled_alpha`,
  `meta_oof`, `pl_scale`, `henery_loglik` and `henery_mle`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from horsetrack.core.eval.folds import walk_forward
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.meta import MetaSpec, fit_meta
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.models.speed_horse import HorseSpeed, p_speed
from horsetrack.countries.japan.studies.speed_run import PAR_KEYS, VARIANT_KEYS

STYLES = ("front", "stalker", "midpack", "closer")
VENUES = ("TOK", "NAK")
DISTANCES = (1600, 2000)
PAR = {("TOK", 1600): 96.0, ("TOK", 2000): 120.5, ("NAK", 1600): 95.4, ("NAK", 2000): 120.0}
WEIGHTS = (54.0, 55.0, 56.0, 57.0, 58.0)
W_PER_KG = 0.15
EPS_SD = 0.4  # s, run noise
VARIANT_SD = 0.5  # s, (day, venue) variant spread
FIELD = 8
FIRST_POST_UTC = (1, 0)  # 10:00 JST
RACE_GAP = timedelta(minutes=40)  # a venue's races: each result (post + 30) precedes the next
VENUE_GAP = timedelta(minutes=10)
RESULT_LAG = timedelta(minutes=30)  # the same-day dividends rule, as the real builders
MARKET_SCALE = 0.35  # s: the market's log strength is -(ability + weight term) / this, plus noise
MARKET_NOISE = 0.5
OVERROUND = 1.25

# The study's speed spec as #305 keys it (the synthetic frames carry every one of its keys).
SPEC = SpeedSpec(par_keys=PAR_KEYS, variant_keys=VARIANT_KEYS, same_day_variant=True, t_df=None)


def utc(day: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=timezone.utc)


def weekends(first_saturday: date, n_weeks: int) -> list[date]:
    assert first_saturday.weekday() == 5
    return [first_saturday + timedelta(days=7 * w + d) for w in range(n_weeks) for d in (0, 1)]


def post_time(day: date, venue_index: int, race_index: int) -> datetime:
    return utc(day, *FIRST_POST_UTC) + race_index * RACE_GAP + venue_index * VENUE_GAP


def race_id(day: date, venue: str, race_index: int) -> str:
    return f"syn-{day:%Y%m%d}-{venue}-{race_index + 1:02d}"


def meeting_id(day: date, venue: str) -> str:
    return f"syn:{day:%Y%m%d}:{venue}"


@dataclass(frozen=True)
class World:
    runs: pd.DataFrame
    table: pd.DataFrame
    market: pd.DataFrame
    ability: dict[str, float]

    def day_of(self) -> dict[str, date]:
        """race_id -> race day (JST), from the table."""
        return dict(zip(self.table["race_id"], self.table["race_date"], strict=True))


def world(days: list[date], *, seed: int, n_horses: int, races_per_venue: int) -> World:
    rng = np.random.default_rng(seed)
    ability = rng.normal(0.8, 0.8, n_horses)
    style = [STYLES[k] for k in rng.integers(0, len(STYLES), n_horses)]
    early = rng.uniform(0.1, 0.9, n_horses)
    runs, table, market = [], [], []
    for day in days:
        pool = rng.permutation(n_horses)
        slot = 0
        for v, venue in enumerate(VENUES):
            variant = rng.normal(0.0, VARIANT_SD)
            for r in range(races_per_venue):
                post = post_time(day, v, r)
                distance = DISTANCES[r % len(DISTANCES)]
                rid = race_id(day, venue, r)
                field = pool[FIELD * slot : FIELD * (slot + 1)]
                slot += 1
                weights = np.array([WEIGHTS[k] for k in rng.integers(0, len(WEIGHTS), FIELD)])
                terms = ability[field] + W_PER_KG * (weights - 56.0)
                times = PAR[(venue, distance)] + variant + terms + rng.normal(0.0, EPS_SD, FIELD)
                strength = -terms / MARKET_SCALE + rng.normal(0.0, MARKET_NOISE, FIELD)
                q_true = np.exp(strength - strength.max())
                q_true /= q_true.sum()
                odds = 1.0 / (OVERROUND * q_true)
                q = (1.0 / odds) / (1.0 / odds).sum()
                winner = int(np.argmin(times))
                common = {
                    "race_id": rid,
                    "race_utc": post,
                    "track": venue,
                    "surface": "turf",
                    "distance": distance,
                    "layout": "A",
                    "rail": "none",
                    "class_level": "C1",
                    "result_available_utc": post + RESULT_LAG,
                }
                for i, h in enumerate(field):
                    horse = f"syn-h{h:03d}"
                    runs.append(
                        common
                        | {
                            "horse_id": horse,
                            "style": style[h],
                            "early_pos_mean3": float(early[h]),
                            "weight_kg": float(weights[i]),
                            "draw": i + 1,
                            "finish_time_s": float(times[i]),
                        }
                    )
                    market.append(
                        {
                            "race_id": rid,
                            "race_date": pd.Timestamp(day),
                            "horse_id": horse,
                            "win_odds": float(odds[i]),
                            "won": i == winner,
                            "dead_heat": False,
                            "meeting_id": meeting_id(day, venue),
                            "q": float(q[i]),
                        }
                    )
                table.append(
                    common
                    | {
                        "meeting_id": meeting_id(day, venue),
                        "race_date": day,
                        "post_utc": post,
                        "top3_time_s": float(np.mean(np.sort(times)[:3])),
                        "complete_timing": True,
                    }
                )
    runs, table, market = pd.DataFrame(runs), pd.DataFrame(table), pd.DataFrame(market)
    for frame in (runs, table):
        for col in ("race_utc", "result_available_utc"):
            frame[col] = pd.to_datetime(frame[col], utc=True)
    table["post_utc"] = pd.to_datetime(table["post_utc"], utc=True)
    planted = {f"syn-h{h:03d}": float(ability[h]) for h in range(n_horses)}
    return World(runs=runs, table=table, market=market, ability=planted)


def on_days(w: World, days) -> World:
    """The world's rows on `days` only (the same rows, nothing re-drawn)."""
    keep = set(days)
    day = w.day_of()
    in_runs = w.runs["race_id"].map(day).isin(keep).to_numpy()
    in_table = w.table["race_date"].isin(keep).to_numpy()
    in_market = w.market["race_id"].map(day).isin(keep).to_numpy()
    return World(
        runs=w.runs.loc[in_runs].reset_index(drop=True),
        table=w.table.loc[in_table].reset_index(drop=True),
        market=w.market.loc[in_market].reset_index(drop=True),
        ability=w.ability,
    )


def poisoned(w: World, days, *, by: float = 3.0) -> World:
    """The world with every result on `days` wrecked: finish and top-3 times `by` s slower in
    reverse order of merit, and the market's winner and prices turned around."""
    keep = set(days)
    day = w.day_of()
    runs, table, market = w.runs.copy(), w.table.copy(), w.market.copy()
    hit = runs["race_id"].map(day).isin(keep).to_numpy()
    ranks = runs.loc[hit].groupby("race_id")["finish_time_s"].rank(ascending=False)
    runs.loc[hit, "finish_time_s"] = runs.loc[hit, "finish_time_s"] + by + 0.5 * ranks
    hit_t = table["race_date"].isin(keep).to_numpy()
    table.loc[hit_t, "top3_time_s"] = table.loc[hit_t, "top3_time_s"] + by
    hit_m = market["race_id"].map(day).isin(keep).to_numpy()
    part = market.loc[hit_m]
    inv = (1.0 / part["q"]) / (1.0 / part["q"]).groupby(part["race_id"]).transform("sum")
    market.loc[hit_m, "q"] = inv.to_numpy()
    market.loc[hit_m, "win_odds"] = (1.0 / (OVERROUND * inv)).to_numpy()
    longest = part.groupby("race_id")["q"].transform("min").eq(part["q"]).to_numpy()
    market.loc[hit_m, "won"] = longest
    return World(runs=runs, table=table, market=market, ability=w.ability)


# --------------------------------------------------------------------------- reference fits


def expected_predictions(
    w: World, *, scheme, drift: float, t_df: float | None, decision_min: float, spec=SPEC
) -> pd.DataFrame:
    """#97's models by hand on each #70 fold of the table's race days: SpeedModel fitted on the
    training races and observing the test races; HorseSpeed(drift, t_df) fitted on the training
    races' runs and observing the test races' runs; each test race's runners predicted at its
    off less `decision_min` minutes."""
    folds = walk_forward(w.table["race_date"], scheme=scheme)
    lead = pd.Timedelta(minutes=decision_min)
    out = []
    for k, fold in enumerate(folds):
        train, test = w.table.iloc[fold.train_idx], w.table.iloc[fold.test_idx]
        speed = SpeedModel(spec).fit(train).observe(test)
        train_runs = w.runs[w.runs["race_id"].isin(train["race_id"])]
        test_runs = w.runs[w.runs["race_id"].isin(test["race_id"])]
        horse = HorseSpeed(drift_var_per_day=drift, t_df=t_df).fit(train_runs, speed)
        horse.observe(test_runs)
        for rid, card in test_runs.groupby("race_id", sort=False):
            t = card["race_utc"].iloc[0] - lead
            got = horse.predict(card, at_utc=t.to_pydatetime())
            out.append(
                pd.DataFrame(
                    {
                        "race_id": rid,
                        "horse_id": got["horse_id"].to_numpy(),
                        "fold": k,
                        "train_end": fold.train_end,
                        "decision_utc": t,
                        "mu_s": got["mu_s"].to_numpy(dtype=float),
                        "sd_s": got["sd_s"].to_numpy(dtype=float),
                    }
                )
            )
    return pd.concat(out, ignore_index=True)


def winner_ll(p, rows: pd.DataFrame) -> pd.Series:
    """#69's winner log loss per race (p renormalised within the race; a dead heat's winners
    averaged), by race id."""
    race = rows["race_id"].to_numpy(dtype=object)
    p = pd.Series(np.asarray(p, dtype=float))
    p = p / p.groupby(race).transform("sum")
    won = rows["won"].to_numpy(dtype=bool)
    return pd.Series(-np.log(p.to_numpy()[won])).groupby(race[won]).mean()


def ln_within(p, rows: pd.DataFrame) -> np.ndarray:
    """ln of p normalised within each race."""
    race = rows["race_id"].to_numpy(dtype=object)
    p = pd.Series(np.asarray(p, dtype=float))
    return np.log((p / p.groupby(race).transform("sum")).to_numpy())


STAGE2 = MetaSpec(["p_speed"], quadratic=False, segment_cols=[], form="stage2", ridge=0.0)
OFFSET = MetaSpec(["p_speed"], quadratic=False, segment_cols=[], form="offset", ridge=0.0)


def meta_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """#87's meta frame: the member column is ln p_speed, the market ln q (both within race)."""
    return pd.DataFrame(
        {
            "race_id": rows["race_id"].to_numpy(dtype=object),
            "meeting_id": rows["meeting_id"].to_numpy(dtype=object),
            "won": rows["won"].to_numpy(dtype=bool),
            "p_speed": ln_within(rows["p_speed"], rows),
            "ln_q_final": ln_within(rows["q"], rows),
        }
    )


def pooled_alpha(rows: pd.DataFrame, spec: MetaSpec = STAGE2) -> tuple[float, float]:
    """The member's coefficient and meeting-clustered se in #87's meta over every row."""
    fit = fit_meta(meta_frame(rows), spec, cluster_col="meeting_id")
    return float(fit.coef["p_speed"]), float(fit.se_cluster["p_speed"])


def meta_oof(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(member meta, market-only meta) out of fold: for each fold of `rows` but the first, both
    fitted on the rows of earlier folds whose race_date is on or before the fold's train_end,
    P(win) on the fold's rows; NaN on the first fold's."""
    frame = meta_frame(rows)
    fold = rows["fold"].to_numpy()
    day = pd.to_datetime(rows["race_date"]).to_numpy()
    member, market = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for k in sorted(set(fold))[1:]:
        end = pd.to_datetime(rows.loc[fold == k, "train_end"]).iloc[0].to_datetime64()
        train = (fold < k) & (day <= end)
        test = fold == k
        fit = fit_meta(frame[train], STAGE2)
        member[test] = fit.predict(frame[test], np.exp(frame.loc[test, "ln_q_final"].to_numpy()))
        tr = frame[train]
        beta = fit_conditional_logit(
            tr[["ln_q_final"]].to_numpy(), tr["race_id"].to_numpy(), tr["won"].to_numpy()
        ).coef[0]
        te = frame[test]
        market[test] = softmax_within(beta * te["ln_q_final"].to_numpy(), te["race_id"].to_numpy())
    return member, market


def pl_scale(rows: pd.DataFrame) -> float:
    """1 / the conditional-logit coefficient of the winner on -mu."""
    fit = fit_conditional_logit(
        -rows[["mu_s"]].to_numpy(dtype=float), rows["race_id"].to_numpy(), rows["won"].to_numpy()
    )
    return 1.0 / float(fit.coef[0])


def henery_loglik(rows: pd.DataFrame, scale: float) -> float:
    """The winners' Henery log-likelihood at `scale` (#97's p_speed; a dead heat's at 1/k)."""
    total = 0.0
    for _, race in rows.groupby("race_id", sort=False):
        won = race["won"].to_numpy(dtype=bool)
        if not won.any():
            continue
        p = p_speed(race["mu_s"].to_numpy(dtype=float), scale=scale, method="henery")
        total += float(np.log(p[won]).mean())
    return total


def henery_mle(rows: pd.DataFrame) -> float:
    found = minimize_scalar(
        lambda x: -henery_loglik(rows, math.exp(x)),
        bounds=(math.log(0.01), math.log(100.0)),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return math.exp(found.x)


# --------------------------------------------------------------------------- pure rows


def softmax_within(u: np.ndarray, race: np.ndarray) -> np.ndarray:
    codes = pd.factorize(np.asarray(race, dtype=object))[0]
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / np.bincount(codes, weights=e)[codes]


FIRST_SATURDAY = date(2025, 1, 4)


def fold_days(k: int) -> tuple[date, date]:
    sat = FIRST_SATURDAY + timedelta(weeks=k)
    return sat, sat + timedelta(days=1)


def scale_rows(seed: int, scales: tuple[float, ...], *, cut_fold: int, races: int = 4,
               field: int = 8) -> pd.DataFrame:  # fmt: skip
    """Weekly folds k = 0.. (a Saturday and a Sunday, two venues, `races` races each). mu_s is
    100 + N(0, 1) per runner; the winner is drawn from softmax(-mu / scales[k]). Fold k's
    train_end is the Friday before it, except `cut_fold`'s: the Saturday before, so the previous
    fold's Sunday is embargoed from its training. One race of fold 1 is a dead heat."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, s in enumerate(scales):
        sat, sun = fold_days(k)
        train_end = sat - timedelta(days=7 if k == cut_fold else 1)
        for day in (sat, sun):
            for venue in ("v1", "v2"):
                for r in range(races):
                    rid = f"sc-{day:%Y%m%d}-{venue}-{r}"
                    mu = 100.0 + rng.normal(0.0, 1.0, field)
                    p = np.exp(-(mu - mu.min()) / s)
                    p /= p.sum()
                    order = rng.choice(field, size=2, replace=False, p=p)
                    winners = {int(order[0])}
                    if k == 1 and day == sat and venue == "v1" and r == 0:
                        winners.add(int(order[1]))  # a dead heat for first
                    for i in range(field):
                        rows.append(
                            {
                                "race_id": rid,
                                "horse_id": f"{rid}-{i}",
                                "meeting_id": f"sc:{day:%Y%m%d}:{venue}",
                                "race_date": day,
                                "fold": k,
                                "train_end": train_end,
                                "mu_s": float(mu[i]),
                                "won": i in winners,
                            }
                        )
    return pd.DataFrame(rows)


def member_rows(seed: int, *, alpha: float, beta: float, folds: range = range(1, 7),
                races: int = 10, on_lnq: float = 1.0, noise: float = 0.5,
                cut_fold: int | None = None) -> pd.DataFrame:  # fmt: skip
    """Weekly folds (a Saturday and a Sunday, two venues, `races` races of 8-12 runners). The
    market q is a softmax of N(0, 1); the member's log-probability ln p_speed is ln softmax(on_lnq
    * ln q + N(0, noise^2)) (it tracks the price); the winner is drawn from softmax(beta * ln q +
    alpha * ln p_speed). p_cal, the D4 stand-in, is softmax(0.9 ln q): not the truth. Fold k's
    train_end is the Friday before it, except `cut_fold`'s: the Saturday before, so the previous
    fold's Sunday is embargoed from its training. The first race of each fold is a dead heat."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in folds:
        sat, sun = fold_days(k)
        train_end = sat - timedelta(days=7 if k == cut_fold else 1)
        first = True
        for day in (sat, sun):
            for venue in ("v1", "v2"):
                for r in range(races):
                    rid = f"mb-{day:%Y%m%d}-{venue}-{r:02d}"
                    n = int(rng.integers(8, 13))
                    s = rng.normal(0.0, 1.0, n)
                    ln_q = s - s.max() - np.log(np.exp(s - s.max()).sum())
                    m = on_lnq * ln_q + rng.normal(0.0, noise, n)
                    ln_p = m - m.max() - np.log(np.exp(m - m.max()).sum())
                    u = beta * ln_q + alpha * ln_p + rng.gumbel(size=n)
                    order = np.argsort(-u)
                    winners = {int(order[0])} | ({int(order[1])} if first else set())
                    first = False
                    p_cal = np.exp(0.9 * ln_q)
                    p_cal /= p_cal.sum()
                    for i in range(n):
                        rows.append(
                            {
                                "race_id": rid,
                                "horse_id": f"{rid}-{i}",
                                "meeting_id": f"mb:{day:%Y%m%d}:{venue}",
                                "race_date": day,
                                "fold": k,
                                "train_end": train_end,
                                "won": i in winners,
                                "q": float(np.exp(ln_q[i])),
                                "p_speed": float(np.exp(ln_p[i])),
                                "p_cal": float(p_cal[i]),
                            }
                        )
    return pd.DataFrame(rows)
