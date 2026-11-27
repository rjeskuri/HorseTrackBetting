"""The tail-return curve: realised return per unit stake by band of claimed value (#433, N5).

A claim is a bet's claimed value at the final price, ``p x O`` (1 is fair). Within each segment
the bets are banded by the empirical quantiles of their claims, and each band reports its realised
return per unit stake with one-sided bounds from resampling whole clusters (D7's meeting), so a
model whose top band pays can be told from one whose tail is overconfident.

Bands. ``quantiles`` ``q_1 < ... < q_K`` in (0, 1) give K + 1 bands; band k covers the claim
quantiles ``[q_k, q_k+1)`` with ``q_0 = 0`` and ``q_K+1 = 1``. Its claim edges are the segment's
``np.quantile`` (linear interpolation) at ``q_k`` and ``q_k+1``, so the first band starts at the
segment's minimum claim and the last ends at its maximum. A bet is in band k when
``claim_lo <= claim < claim_hi``; the last band is closed, so the maximum is in it, and every bet
of a segment is in exactly one band.

Bounds. ``realised`` is ``sum(return) / n_bets`` over the band's bets; ``lo`` and ``hi`` are
``core.eval.bootstrap.paired_bootstrap`` on the band's per-bet returns with their clusters: the
(1 - level) and the ``level`` quantiles of the replicate means. A band with no bets (tied claims
can leave one empty) reports ``n_bets`` 0 and NaN statistics.

Pool-agnostic, flat stakes. ``level``, ``n_boot`` and ``seed`` have no defaults: callers pass D7's
values from the country's ``config/scoring.yaml``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from horsetrack.core.eval.bootstrap import paired_bootstrap

ALL = "all"
COLUMNS = [
    "segment",
    "band",
    "q_lo",
    "q_hi",
    "claim_lo",
    "claim_hi",
    "n_bets",
    "n_clusters",
    "mean_claim",
    "realised",
    "lo",
    "hi",
]


def tail_curve(
    bets: pd.DataFrame,
    *,
    claim_col: str,
    return_col: str,
    cluster_col: str,
    quantiles: Sequence[float],
    by: str | None = None,
    n_boot: int,
    level: float,
    seed: int,
) -> pd.DataFrame:
    """Realised return per unit stake by quantile band of the claim, per segment.

    One row per (segment, band) with the columns of ``COLUMNS``. The ``'all'`` segment covers every
    bet; with ``by`` set, one more segment per level of that column follows, labelled by the level.
    Within a segment the rows go from the lowest band to the highest (``band`` 0, 1, ...).

    Raises KeyError for a column not in ``bets``; ValueError for no bets, a missing claim, return,
    cluster or ``by`` value, or ``quantiles`` empty or not strictly increasing inside (0, 1). Each
    band's bootstrap uses ``seed``, so the same seed reproduces the whole curve.
    """
    columns = [claim_col, return_col, cluster_col] + ([] if by is None else [by])
    missing = [c for c in columns if c not in bets.columns]
    if missing:
        raise KeyError(f"bets has no column(s) {missing}")
    qs = np.asarray(quantiles, dtype=float)
    if qs.ndim != 1 or len(qs) == 0:
        raise ValueError("quantiles must be a non-empty sequence of fractions")
    if not (np.isfinite(qs).all() and (qs > 0).all() and (qs < 1).all()):
        raise ValueError(f"quantiles must lie strictly inside (0, 1), got {list(quantiles)}")
    if (np.diff(qs) <= 0).any():
        raise ValueError(f"quantiles must be strictly increasing, got {list(quantiles)}")
    if len(bets) == 0:
        raise ValueError("bets is empty")
    for column, what in [(claim_col, "claim"), (return_col, "return"), (cluster_col, "cluster")]:
        if bets[column].isna().any():
            raise ValueError(f"{bets[column].isna().sum()} bet(s) have no {what} ({column!r})")
    if by is not None and bets[by].isna().any():
        raise ValueError(f"{bets[by].isna().sum()} bet(s) have no segment ({by!r})")

    claims = bets[claim_col].to_numpy(dtype=float)
    returns = bets[return_col].to_numpy(dtype=float)
    clusters = bets[cluster_col].to_numpy()
    if not np.isfinite(claims).all() or not np.isfinite(returns).all():
        raise ValueError("claims and returns must be finite")
    edges_q = np.concatenate([[0.0], qs, [1.0]])

    segments = [(ALL, np.arange(len(bets)))]
    if by is not None:
        groups = pd.Series(np.arange(len(bets))).groupby(bets[by].to_numpy(), sort=True)
        segments += [(label, idx.to_numpy()) for label, idx in groups]

    rows = []
    for label, idx in segments:
        seg_claims, seg_returns, seg_clusters = claims[idx], returns[idx], clusters[idx]
        edges = np.quantile(seg_claims, edges_q)
        # inner edges only: equal to an edge goes up a band, and the maximum stays in the last
        band_of = np.searchsorted(edges[1:-1], seg_claims, side="right")
        for k in range(len(edges_q) - 1):
            member = band_of == k
            rows.append(
                {
                    "segment": label,
                    "band": k,
                    "q_lo": edges_q[k],
                    "q_hi": edges_q[k + 1],
                    "claim_lo": edges[k],
                    "claim_hi": edges[k + 1],
                    **_band_stats(
                        seg_claims[member],
                        seg_returns[member],
                        seg_clusters[member],
                        n_boot=n_boot,
                        level=level,
                        seed=seed,
                    ),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def _band_stats(
    claims: np.ndarray,
    returns: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int,
    level: float,
    seed: int,
) -> dict:
    if len(claims) == 0:
        nan = float("nan")
        return {
            "n_bets": 0,
            "n_clusters": 0,
            "mean_claim": nan,
            "realised": nan,
            "lo": nan,
            "hi": nan,
        }
    interval = paired_bootstrap(
        pd.Series(returns), pd.Series(clusters), n_boot=n_boot, level=level, seed=seed
    )
    return {
        "n_bets": len(claims),
        "n_clusters": interval.n_clusters,
        "mean_claim": float(claims.mean()),
        "realised": float(returns.sum() / len(returns)),
        "lo": interval.lo,
        "hi": interval.hi,
    }
