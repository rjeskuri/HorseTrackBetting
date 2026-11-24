"""Per-race scoring as signed in docs/problem-statement.md, section 3 (Metric), decision D6.

Section 3 fixes the loss as log loss and the reported gain as the log-loss (pseudo-R2) gain over
the final-price baseline, per race and paired: the same races on both sides. D6 makes winner log
loss the primary score of a Win claim, reports pseudo-R2 against equal chances for every runner,
and puts the gain over the final price in nats per race alongside it.

This module scores the winner only (the Win pool). The top-3 order score D6 names for Place and
Wide is not here. The cases section 3 leaves open are settled by the owner's answer on #69
(2026-11-24), and each is counted in the score's result so nothing disappears silently:

- a dead heat with k co-winners scores each co-winner at weight 1/k;
- a race with no winner (void) is dropped and counted;
- when the two sides cover different races (a model and a baseline, or a score and its
  reference), only the common races are scored and the races dropped from each side are counted;
- scratched runners are removed and the probabilities renormalised over the starters.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RaceLogLoss:
    """Per-race log loss with the counts behind it (owner's answer on #69, 2026-11-24).

    `per_race` is indexed by race id over the scored races, `n_races` is their number,
    `n_dead_heats` the scored races with two or more winners, `n_void_dropped` the races with no
    winner (dropped), and `n_scratched_removed` the runner rows removed as scratched.
    """

    per_race: pd.Series
    n_races: int
    n_dead_heats: int
    n_void_dropped: int
    n_scratched_removed: int


@dataclass(frozen=True)
class PseudoR2:
    """Pseudo-R2 over the common races, with the races only on each side counted."""

    value: float
    n_common: int
    n_only_ll: int
    n_only_reference: int


@dataclass(frozen=True)
class Gain:
    """Gain per common race (ll_baseline - ll_model), with the races only on each side counted."""

    per_race: pd.Series
    n_common: int
    n_only_model: int
    n_only_baseline: int


def per_race_log_loss(
    p: pd.Series, won: pd.Series, race_id: pd.Series, scratched: pd.Series
) -> RaceLogLoss:
    """Winner log loss per race in nats (section 3 Metric, D6; owner's answer on #69, 2026-11-24).

    `p`, `won`, `race_id` and `scratched` hold one row per runner and must share one index. `p` is
    the forecast probability of the runner winning, `won` is True for the race's winner (or each
    co-winner of a dead heat), `race_id` names the race and `scratched` is True for a runner that
    did not start. Rows do not need to be grouped by race. A scratched runner's `p` may be missing
    (the final-price baseline has no price for a horse that did not start); every other value
    must be present.

    Scratched rows are removed first and every race's `p` is renormalised over its starters. A
    race's loss is then -(1/k) * sum ln p over its k winners: -ln p(winner) for k = 1, each
    co-winner at weight 1/k for a dead heat. A race with no winner (a void race, including one
    with no starters) is dropped and counted in `n_void_dropped`.

    Raises ValueError when the inputs do not line up, when `won`, `race_id` or `scratched` has a
    missing value or a starter's `p` is missing, when a scratched runner is marked as a winner,
    when a starter's probability is negative or not finite, or when a winner's probability is
    zero.
    """
    if not all(p.index.equals(s.index) for s in (won, race_id, scratched)):
        raise ValueError("p, won, race_id and scratched must share one index")
    if any(s.isna().any() for s in (won, race_id, scratched)):
        raise ValueError("won, race_id and scratched must not contain missing values")

    won = won.astype(bool)
    scratched = scratched.astype(bool)
    if (won & scratched).any():
        raise ValueError(
            f"{int((won & scratched).sum())} scratched runner(s) are marked as winners"
        )

    race = race_id.to_numpy()
    n_winners = won.groupby(race).sum()
    void = n_winners[n_winners == 0].index

    starters = ~scratched
    if p[starters].isna().any():  # a scratched runner's p may be missing: it is removed here
        raise ValueError("a starter's probability must not be missing")
    p_start = p[starters].astype(float)
    if not np.isfinite(p_start).all() or (p_start < 0).any():
        raise ValueError("a starter's probability must be finite and non-negative")
    race_start = race[starters.to_numpy()]
    p_norm = p_start / p_start.groupby(race_start).transform("sum")

    winner = won[starters].to_numpy()
    p_winner = p_norm[winner].to_numpy()
    if not (p_winner > 0).all():  # also catches 0/0 when every starter of the race has p = 0
        raise ValueError("a winner's probability must be positive")

    result = pd.Series(-np.log(p_winner)).groupby(race_start[winner]).mean()
    result.index.name = "race_id"
    result.name = "log_loss"
    return RaceLogLoss(
        per_race=result,
        n_races=len(result),
        n_dead_heats=int((n_winners >= 2).sum()),
        n_void_dropped=len(void),
        n_scratched_removed=int(scratched.sum()),
    )


def pseudo_r2(ll: pd.Series, ll_reference: pd.Series) -> PseudoR2:
    """Pseudo-R2 of per-race losses against a reference over their common races (D6).

    Defined as 1 - sum(ll) / sum(ll_reference), summed over the races both Series cover. D6 signs
    the reference as equal chances for every runner, whose loss in a race of n runners is ln n;
    the caller passes those losses as `ll_reference`. Both Series are indexed by race id. Section
    3 pairs the two sides on the same races; by the owner's answer on #69 (2026-11-24) races on
    one side only are dropped and counted in `n_only_ll` and `n_only_reference`.

    Raises ValueError when there are no common races or the reference's summed loss over them is
    not positive.
    """
    a, b, n_only_ll, n_only_reference = _paired(ll, ll_reference, "ll", "ll_reference")
    total_reference = float(b.sum())
    if total_reference <= 0:
        raise ValueError("the reference's summed log loss must be positive")
    return PseudoR2(
        value=1.0 - float(a.sum()) / total_reference,
        n_common=len(a),
        n_only_ll=n_only_ll,
        n_only_reference=n_only_reference,
    )


def gain(ll_model: pd.Series, ll_baseline: pd.Series) -> Gain:
    """Gain over the baseline per race, in nats: ll_baseline - ll_model (section 3 Metric, D6).

    Positive when the model's loss is lower, that is when the model is better. Section 3 signs
    the gain as paired per race over the final-price baseline, so both Series are indexed by race
    id and matched by label. By the owner's answer on #69 (2026-11-24) only the common races are
    scored, and the races only in `ll_model` or only in `ll_baseline` (for example a race with no
    priced baseline) are counted in `n_only_model` and `n_only_baseline`.

    `per_race` is indexed by race id in the order of `ll_model`. Raises ValueError when there are
    no common races.
    """
    a, b, n_only_model, n_only_baseline = _paired(ll_model, ll_baseline, "ll_model", "ll_baseline")
    result = b - a
    result.name = "gain"
    return Gain(
        per_race=result,
        n_common=len(result),
        n_only_model=n_only_model,
        n_only_baseline=n_only_baseline,
    )


def _paired(
    a: pd.Series, b: pd.Series, a_name: str, b_name: str
) -> tuple[pd.Series, pd.Series, int, int]:
    """`a` and `b` on their common races in `a`'s order, and the counts only in `a` and only in `b`.

    Raises ValueError on duplicate race ids, missing values or no common races.
    """
    for name, s in ((a_name, a), (b_name, b)):
        if not s.index.is_unique:
            raise ValueError(f"{name} has duplicate race ids")
        if s.isna().any():
            raise ValueError(f"{name} has missing values")
    common = a.index[a.index.isin(b.index)]
    if len(common) == 0:
        raise ValueError(f"{a_name} and {b_name} have no races in common")
    n_only_a = len(a) - len(common)
    n_only_b = len(b) - len(common)
    return a.loc[common].astype(float), b.loc[common].astype(float), n_only_a, n_only_b
