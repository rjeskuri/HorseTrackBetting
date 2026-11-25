"""Synthetic Plackett-Luce races and first-principles references for test_ability.py (#92).

Nothing here calls the code under test (`models.ability`). The references are a direct loop over
races and stages (`first_principles_loglik`) and the exploded conditional logit: a top-k
Plackett-Luce ranking with unit discounts is the conditional logit over k stacked choice sets per
race (stage s: the runners not placed 1..s-1, the chosen one placed s), fitted with the existing
`models.condlogit.fit_conditional_logit`.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from horsetrack.countries.japan.models.condlogit import CLFit, fit_conditional_logit


def field(rng: np.random.Generator, n_races: int, sizes=(6, 12)) -> np.ndarray:
    """Race codes 0..n_races-1, one per runner row, with field sizes drawn from `sizes`."""
    n = rng.integers(sizes[0], sizes[1] + 1, size=n_races)
    return np.repeat(np.arange(n_races), n)


def labels(codes: np.ndarray, prefix: str = "jra:r") -> np.ndarray:
    names = np.array([f"{prefix}{c:05d}" for c in range(codes.max() + 1)], dtype=object)
    return names[codes]


def centre_within_race(X: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """X less its race mean, column by column."""
    X = np.asarray(X, dtype=float).reshape(len(codes), -1)
    return X - pd.DataFrame(X).groupby(codes).transform("mean").to_numpy()


def rank_desc(score: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """1 for the highest score of each race, 2 for the next, ...; no ties."""
    return pd.Series(score).groupby(codes).rank(ascending=False, method="first").to_numpy()


def pl_positions(rng: np.random.Generator, u: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """A full Plackett-Luce finishing order per race on u (Gumbel: u + noise, sorted down)."""
    return rank_desc(np.asarray(u, dtype=float) + rng.gumbel(size=len(u)), codes)


def winner_then_reversed(rng: np.random.Generator, u: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """The winner drawn from softmax(u), every later place from Plackett-Luce on -u: the places
    point the wrong way, so training on them pulls the fit away from the winner model."""
    u = np.asarray(u, dtype=float)
    g_win = u + rng.gumbel(size=len(u))
    top = pd.Series(g_win).groupby(codes).transform("max").to_numpy()
    winner = g_win == top
    rest = np.where(winner, np.inf, -u + rng.gumbel(size=len(u)))
    return rank_desc(rest, codes)


def explode(codes: np.ndarray, pos: np.ndarray, k: int):
    """(rows, pseudo-race, chosen, stage) of the top-k explosion: stage s of a race holds its
    runners not placed 1..s-1 and exists when some runner is placed s; ties are not handled."""
    codes = np.asarray(codes)
    pos = np.asarray(pos, dtype=float)
    out_rows, out_race, out_chosen, out_stage = [], [], [], []
    for s in range(1, k + 1):
        has = pd.Series(pos == s).groupby(codes).transform("any").to_numpy()
        rows = np.flatnonzero(has & ~(pos < s))  # not placed before s (an unplaced NaN stays in)
        out_rows.append(rows)
        out_race.append(codes[rows].astype(np.int64) * 100 + s)
        out_chosen.append(pos[rows] == s)
        out_stage.append(np.full(len(rows), s))
    return (
        np.concatenate(out_rows),
        np.concatenate(out_race),
        np.concatenate(out_chosen),
        np.concatenate(out_stage),
    )


def exploded_fit(X, codes, pos, k: int, *, offset=None, ridge=0.0) -> CLFit:
    """The unit-discount top-k Plackett-Luce maximum likelihood via the conditional logit."""
    X = np.asarray(X, dtype=float).reshape(len(codes), -1)
    rows, race, chosen, _ = explode(codes, pos, k)
    off = None if offset is None else np.asarray(offset, dtype=float)[rows]
    return fit_conditional_logit(X[rows], race, chosen, offset=off, ridge=ridge)


def first_principles_loglik(u, race_id, finish_pos, k: int, discounts) -> float:
    """sum over races, over stages s = 1..k with a runner placed s, of discounts[s-1] *
    (u_s - logsumexp of u over the runners not placed 1..s-1). No ties."""
    u = np.asarray(u, dtype=float)
    race_id = np.asarray(race_id, dtype=object)
    pos = np.asarray(finish_pos, dtype=float)
    total = 0.0
    for r in pd.unique(race_id):
        mine = race_id == r
        ur, pr = u[mine], pos[mine]
        for s in range(1, k + 1):
            chosen = np.flatnonzero(pr == s)
            if len(chosen) == 0:
                continue
            assert len(chosen) == 1, "first_principles_loglik takes no ties"
            alive = ~(pr < s)
            top = ur[alive].max()
            lse = top + np.log(np.exp(ur[alive] - top).sum())
            total += float(discounts[s - 1]) * (ur[chosen[0]] - lse)
    return total


def softmax_within(u, codes) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / pd.Series(e).groupby(codes).transform("sum").to_numpy()


def race_dates(codes: np.ndarray, races_per_day: int, first_day: date) -> np.ndarray:
    """One date per row: race c runs on day c // races_per_day after first_day."""
    return np.array([first_day + timedelta(days=int(c // races_per_day)) for c in codes])


def information(X, race, coef) -> np.ndarray:
    """The conditional-logit information at coef over the choice sets `race` (one label per row):
    the sum over sets of the p-weighted covariance of the set's X rows."""
    X = np.asarray(X, dtype=float).reshape(len(race), -1)
    codes = pd.factorize(np.asarray(race, dtype=object))[0]
    p = softmax_within(X @ np.asarray(coef, dtype=float), codes)
    xbar = np.stack([np.bincount(codes, weights=p * X[:, j]) for j in range(X.shape[1])], axis=1)
    return (X * p[:, None]).T @ X - xbar.T @ xbar


def exploded_information(X, codes, pos, k: int, coef) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(len(codes), -1)
    rows, race, _, _ = explode(codes, pos, k)
    return information(X[rows], race, coef)


def move_to_place(pos: np.ndarray, codes: np.ndarray, row: int, place: int) -> None:
    """Swap finishing positions so that `row` finishes `place` in its race (in place)."""
    other = np.flatnonzero((codes == codes[row]) & (pos == place))[0]
    pos[row], pos[other] = pos[other], pos[row]


def centred(effects: np.ndarray) -> np.ndarray:
    return np.asarray(effects, dtype=float) - np.mean(effects)


# The pooled-entity design: 600 races of 7 to 9 runners. J_BIG rides one runner in 500 races,
# J_ZERO one in 500 others' slots, J_SPARSE one in 3; 40 more jockeys share every other ride.
# Every planted jockey effect together has mean 0 (the level a prior at 0 pins), as do the 30
# trainers' and the 25 sires'. J_SPARSE's three runners are placed 1st, 1st and 6th by hand.
J_BIG, J_ZERO, J_SPARSE = "j:big", "j:zero", "j:sparse"
BIG_EFFECT = 0.6
X_COEF = 0.7
SPARSE_PLACES = (1, 1, 6)


def pooled_frame(seed: int):
    """(frame, planted): frame has race_id, horse_id, race_date, finish_pos, x1, jockey_id,
    trainer_id, sire_id; planted has the true utility `u` per row and the effect dicts."""
    rng = np.random.default_rng(seed)
    n_races = 600
    codes = field(rng, n_races, sizes=(7, 9))
    n = len(codes)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    others = rng.normal(0.0, 0.5, size=40)
    others = others - others.mean() - BIG_EFFECT / 40  # all 43 planted jockey effects: mean 0
    jockey_effect = {J_BIG: BIG_EFFECT, J_ZERO: 0.0, J_SPARSE: 0.0}
    jockey_effect |= {f"j:{i:02d}": float(others[i]) for i in range(40)}
    jockey = np.array([f"j:{i:02d}" for i in rng.integers(0, 40, size=n)], dtype=object)
    jockey[starts[rng.choice(n_races, 500, replace=False)]] = J_BIG
    jockey[starts[rng.choice(n_races, 500, replace=False)] + 1] = J_ZERO
    sparse_races = rng.choice(n_races, 3, replace=False)
    jockey[starts[sparse_races] + 2] = J_SPARSE
    trainer_effect = centred(rng.normal(0.0, 0.3, size=30))
    sire_effect = centred(rng.normal(0.0, 0.3, size=25))
    t_idx = rng.integers(0, 30, size=n)
    s_idx = rng.integers(0, 25, size=n)
    x1 = rng.normal(size=n)
    u = (
        X_COEF * x1
        + np.array([jockey_effect[j] for j in jockey])
        + trainer_effect[t_idx]
        + sire_effect[s_idx]
    )
    pos = pl_positions(rng, u, codes)
    for race, place in zip(sparse_races, SPARSE_PLACES, strict=True):
        move_to_place(pos, codes, starts[race] + 2, place)
    frame = pd.DataFrame(
        {
            "race_id": labels(codes),
            "horse_id": np.array([f"h:{i:05d}" for i in range(n)], dtype=object),
            "race_date": race_dates(codes, 12, date(2024, 1, 6)),
            "finish_pos": pos.astype(np.int64),
            "x1": x1,
            "jockey_id": jockey,
            "trainer_id": np.array([f"t:{i:02d}" for i in t_idx], dtype=object),
            "sire_id": np.array([f"s:{i:02d}" for i in s_idx], dtype=object),
        }
    )
    planted = {"u": u, "codes": codes, "jockey": jockey_effect}
    return frame, planted


# The training-window design: 500 races of 6 to 10 runners, two a day from 2024-01-06. J_FLIP
# rides in 250 of the first 300 races (the training window) at +FLIP and in 150 of the last 200
# (the test window) at -FLIP; 12 more jockeys, planted with mean 0, fill the other rides.
J_FLIP = "j:flip"
FLIP = 0.8
WINDOW_TRAIN_RACES = 300


def window_frame(seed: int):
    """(frame, planted) for the training-window test; frame as `pooled_frame` without trainer and
    sire, plus `window` ('train' or 'test')."""
    rng = np.random.default_rng(seed)
    n_races = 500
    codes = field(rng, n_races, sizes=(6, 10))
    n = len(codes)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    others = centred(rng.normal(0.0, 0.3, size=12))
    jockey = np.array([f"j:{i:02d}" for i in rng.integers(0, 12, size=n)], dtype=object)
    train_races = rng.choice(WINDOW_TRAIN_RACES, 250, replace=False)
    test_races = WINDOW_TRAIN_RACES + rng.choice(n_races - WINDOW_TRAIN_RACES, 150, replace=False)
    jockey[starts[train_races]] = J_FLIP
    jockey[starts[test_races]] = J_FLIP
    in_train = codes < WINDOW_TRAIN_RACES
    flip = np.where(in_train, FLIP, -FLIP) * (jockey == J_FLIP)
    effect = np.array([0.0 if j == J_FLIP else others[int(j[2:])] for j in jockey])
    x1 = rng.normal(size=n)
    u = X_COEF * x1 + effect + flip
    pos = pl_positions(rng, u, codes)
    frame = pd.DataFrame(
        {
            "race_id": labels(codes),
            "horse_id": np.array([f"h:{i:05d}" for i in range(n)], dtype=object),
            "race_date": race_dates(codes, 2, date(2024, 1, 6)),
            "finish_pos": pos.astype(np.int64),
            "x1": x1,
            "jockey_id": jockey,
            "window": np.where(in_train, "train", "test"),
        }
    )
    return frame, {"u": u, "codes": codes, "flip": flip}


# The discount design: ten features (FEATURES_10) at planted coefficients, a few jockeys, and
# 1,500-row inner training folds, so a winner-only fit is noisy and the places' information pays
# out of fold. `places` is "pl" (the full order is Plackett-Luce on u: the places carry the
# winner model's information) or "reversed" (the winner from softmax(u), the places from
# Plackett-Luce on -u: training on them corrupts the winner model).
FEATURES_10 = [f"f{j}" for j in range(10)]


def discount_frame(seed: int, n_races: int, places: str, first_day: date, *, code_from: int = 0):
    rng = np.random.default_rng(seed)
    codes = field(rng, n_races, sizes=(6, 10))
    n = len(codes)
    beta = np.random.default_rng(9299).normal(0.0, 0.4, size=10)  # one planted beta for all
    X = rng.normal(size=(n, 10))
    jockey_effect = centred(np.random.default_rng(9298).normal(0.0, 0.4, size=8))
    j_idx = rng.integers(0, 8, size=n)
    u = X @ beta + jockey_effect[j_idx]
    if places == "pl":
        pos = pl_positions(rng, u, codes)
    elif places == "reversed":
        pos = winner_then_reversed(rng, u, codes)
    else:
        raise ValueError(places)
    frame = pd.DataFrame(X, columns=FEATURES_10)
    frame.insert(0, "race_id", labels(codes + code_from))
    frame.insert(1, "horse_id", [f"h:{code_from + i:06d}" for i in range(n)])
    frame.insert(2, "race_date", race_dates(codes, 2, first_day))
    frame.insert(3, "finish_pos", pos.astype(np.int64))
    frame["jockey_id"] = np.array([f"j:{i}" for i in j_idx], dtype=object)
    return frame
