"""Per-race scoring as signed in docs/problem-statement.md, section 3 (Metric), decision D6.

Section 3 fixes the loss as log loss and the reported gain as the log-loss (pseudo-R2) gain over
the final-price baseline, per race and paired: the same races on both sides. D6 makes winner log
loss the primary score of a Win claim, reports pseudo-R2 against equal chances for every runner,
and puts the gain over the final price in nats per race alongside it.

This module scores the winner only (the Win pool). The top-3 order score D6 names for Place and
Wide is not here. Cases the signed statement does not settle (dead heats, races with no winner,
races with no priced baseline, scratched runners) are refused with a ValueError and not handled
by any policy here.
"""

import numpy as np
import pandas as pd


def per_race_log_loss(p: pd.Series, won: pd.Series, race_id: pd.Series) -> pd.Series:
    """Winner log loss per race, -ln p(winner), in nats (section 3 Metric, D6).

    `p`, `won` and `race_id` hold one row per runner and must share one index. `p` is the
    forecast probability of the runner winning, `won` is True for the race's winner, `race_id`
    names the race. Rows do not need to be grouped by race.

    Returns a Series indexed by race id. Raises ValueError when a race does not have exactly one
    winner (a dead heat or no result: the signed statement does not say how to score these),
    when the winner's probability is not in (0, 1], or when the inputs do not line up.
    """
    if not (p.index.equals(won.index) and p.index.equals(race_id.index)):
        raise ValueError("p, won and race_id must share one index")
    if p.isna().any() or won.isna().any() or race_id.isna().any():
        raise ValueError("p, won and race_id must not contain missing values")

    won = won.astype(bool)
    winners = won.groupby(race_id.to_numpy()).sum()
    bad = winners[winners != 1]
    if not bad.empty:
        raise ValueError(
            f"every race needs exactly one winner; {len(bad)} race(s) do not, "
            f"e.g. {bad.index[0]!r} with {int(bad.iloc[0])}"
        )

    p_winner = pd.Series(p[won].to_numpy(dtype=float), index=race_id[won].to_numpy())
    if ((p_winner <= 0) | (p_winner > 1)).any():
        raise ValueError("the winner's probability must be in (0, 1]")

    result = -np.log(p_winner)
    result.index.name = "race_id"
    result.name = "log_loss"
    return result


def pseudo_r2(ll: pd.Series, ll_reference: pd.Series) -> float:
    """Pseudo-R2 of per-race losses against a reference over the same races (D6).

    Defined as 1 - sum(ll) / sum(ll_reference), summed over races. D6 signs the reference as
    equal chances for every runner, whose loss in a race of n runners is ln n; the caller passes
    those losses as `ll_reference`. Both Series are indexed by race id and must cover the same
    races (section 3: paired, the same races); otherwise a ValueError is raised.
    """
    ll, ll_reference = _paired(ll, ll_reference, "ll", "ll_reference")
    total_reference = float(ll_reference.sum())
    if total_reference <= 0:
        raise ValueError("the reference's summed log loss must be positive")
    return 1.0 - float(ll.sum()) / total_reference


def gain(ll_model: pd.Series, ll_baseline: pd.Series) -> pd.Series:
    """Gain over the baseline per race, in nats: ll_baseline - ll_model (section 3 Metric, D6).

    Positive when the model's loss is lower, that is when the model is better. Section 3 signs
    the gain as paired per race over the final-price baseline, so both Series are indexed by race
    id, matched by label, and must cover the same races. A race with no priced baseline is a case
    the signed statement does not settle, so differing race sets raise a ValueError.

    Returns a Series indexed by race id in the order of `ll_model`.
    """
    ll_model, ll_baseline = _paired(ll_model, ll_baseline, "ll_model", "ll_baseline")
    result = ll_baseline - ll_model
    result.name = "gain"
    return result


def _paired(a: pd.Series, b: pd.Series, a_name: str, b_name: str) -> tuple[pd.Series, pd.Series]:
    """`a` and `b` with `b` reordered to `a`'s races; ValueError unless both have the same races."""
    for name, s in ((a_name, a), (b_name, b)):
        if not s.index.is_unique:
            raise ValueError(f"{name} has duplicate race ids")
        if s.isna().any():
            raise ValueError(f"{name} has missing values")
    if len(a) == 0:
        raise ValueError(f"{a_name} is empty")
    if len(a) != len(b) or not a.index.isin(b.index).all():
        only_a = a.index.difference(b.index)
        only_b = b.index.difference(a.index)
        raise ValueError(
            f"{a_name} and {b_name} must cover the same races: "
            f"{len(only_a)} only in {a_name}, {len(only_b)} only in {b_name}"
        )
    return a.astype(float), b.reindex(a.index).astype(float)
