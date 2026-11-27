"""Ticket #433 (N5): the tail-return curve by claimed value, `core.eval.tail.tail_curve`.

Contract pinned here (the coder implements what these tests call):

- ``tail_curve(bets, *, claim_col, return_col, cluster_col, quantiles, by=None, n_boot, level,
  seed) -> pd.DataFrame`` with exactly the columns ``segment, band, q_lo, q_hi, claim_lo,
  claim_hi, n_bets, n_clusters, mean_claim, realised, lo, hi``. ``n_boot``, ``level`` and ``seed``
  have no defaults (D7's values come from ``config/scoring.yaml``).
- Within each segment, rows are its bands in increasing order of claim, highest band last.
  Band k covers the claim quantiles ``[q_k, q_k+1)`` with ``q_0 = 0`` and the last band closed at 1.
- Claim edges are the segment's empirical quantiles of the claim, numpy's default
  (``np.quantile``, linear interpolation): ``claim_lo = np.quantile(claims, q_lo)`` and
  ``claim_hi = np.quantile(claims, q_hi)``, so the first band's ``claim_lo`` is the segment's
  minimum claim and the last band's ``claim_hi`` its maximum. A bet is in band k when
  ``claim_lo_k <= claim < claim_hi_k`` (the last band also takes ``claim == claim_hi``).
- ``realised`` is ``sum(return) / n_bets`` over the band's bets; ``mean_claim`` their mean claim;
  ``n_clusters`` the number of distinct clusters among them; ``lo`` and ``hi`` the one-sided
  bounds at ``level`` from resampling the band's whole clusters (``paired_bootstrap`` semantics:
  ``lo`` the (1 - level) quantile of the replicate means, ``hi`` the ``level`` quantile).
- ``by=None`` gives one curve labelled ``'all'``; ``by=col`` gives ``'all'`` plus one curve per
  level of ``col``, labelled by the level. The order of segments is not pinned; the band label's
  values are not pinned (row order and ``q_lo`` identify a band).

Every dataset is synthetic, built here with a seeded numpy RNG; expected values come from the
data directly (numpy and pandas), never from the code under test.
"""

from __future__ import annotations  # noqa: I001 (isort counts eval.tail third-party until it exists)

import inspect
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.tail import tail_curve

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
KW = {"claim_col": "claim", "return_col": "ret", "cluster_col": "meeting"}


def _curve(bets, quantiles, *, by=None, n_boot=400, level=0.95, seed=7):
    return tail_curve(bets, **KW, quantiles=quantiles, by=by, n_boot=n_boot, level=level, seed=seed)


def _segment(curve: pd.DataFrame, label) -> pd.DataFrame:
    """The rows of one segment, in the order the function returned them."""
    return curve[curve["segment"] == label].reset_index(drop=True)


def _expected_bands(claims: np.ndarray, returns: np.ndarray, meetings: np.ndarray, quantiles):
    """Independent reference: edges, membership, counts and means per band."""
    qs = np.concatenate([[0.0], np.asarray(quantiles, dtype=float), [1.0]])
    edges = np.quantile(claims, qs)
    rows = []
    for k in range(len(qs) - 1):
        last = k == len(qs) - 2
        upper = claims <= edges[k + 1] if last else claims < edges[k + 1]
        member = (claims >= edges[k]) & upper
        rows.append(
            {
                "q_lo": qs[k],
                "q_hi": qs[k + 1],
                "claim_lo": edges[k],
                "claim_hi": edges[k + 1],
                "n_bets": int(member.sum()),
                "n_clusters": len(set(meetings[member])),
                "mean_claim": float(claims[member].mean()),
                "realised": float(returns[member].sum() / member.sum()),
                "member": member,
            }
        )
    return rows


def _check_bands(got: pd.DataFrame, claims, returns, meetings, quantiles):
    expected = _expected_bands(claims, returns, meetings, quantiles)
    assert len(got) == len(expected)
    for k, exp in enumerate(expected):
        row = got.iloc[k]
        assert row["q_lo"] == pytest.approx(exp["q_lo"], abs=1e-12)
        assert row["q_hi"] == pytest.approx(exp["q_hi"], abs=1e-12)
        assert row["claim_lo"] == pytest.approx(exp["claim_lo"], rel=1e-12, abs=1e-12)
        assert row["claim_hi"] == pytest.approx(exp["claim_hi"], rel=1e-12, abs=1e-12)
        assert int(row["n_bets"]) == exp["n_bets"], f"band {k}"
        assert int(row["n_clusters"]) == exp["n_clusters"], f"band {k}"
        assert row["mean_claim"] == pytest.approx(exp["mean_claim"], rel=1e-10)
        assert row["realised"] == pytest.approx(exp["realised"], rel=1e-10, abs=1e-12)
        assert row["lo"] <= row["realised"] + 1e-12 <= row["hi"] + 2e-12
    return expected


def _shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    order = np.random.default_rng(seed).permutation(len(frame))
    return frame.iloc[order].reset_index(drop=True)


# --- the profitable tail ---------------------------------------------------------------------


def _profitable_top(seed=433):
    """20,000 bets in 200 meetings of 100, claims 0.5 + U(0, 1).

    The top 5% of claims by rank (the 1,000 highest) are Bernoulli wins at a 2.0 dividend with
    p = 0.7 (mean return 1.4, sd 0.92, se over 1,000 bets 0.029); every other bet wins at 2.0
    with p = 0.375 (mean 0.75, sd 0.97). The band [0.9, 0.95) has 1,000 bets, so its upper bound
    sits near 0.75 + 1.645 * 0.031 = 0.80, far below 1; the top band's lower bound near
    1.4 - 0.05, far above 1."""
    rng = np.random.default_rng(seed)
    n = 20_000
    meeting = np.repeat([f"M{m:03d}" for m in range(200)], 100)
    claim = 0.5 + rng.random(n)
    top = np.zeros(n, dtype=bool)
    top[np.argsort(claim)[-n // 20 :]] = True
    p = np.where(top, 0.7, 0.375)
    ret = 2.0 * (rng.random(n) < p)
    return pd.DataFrame({"claim": claim, "ret": ret, "meeting": meeting}), top


def test_a_profitable_top_band_is_found():
    bets, top = _profitable_top()
    # premise, from the data alone
    assert bets.loc[top, "ret"].mean() == pytest.approx(1.4, abs=0.06)
    assert bets.loc[~top, "ret"].mean() == pytest.approx(0.75, abs=0.03)

    curve = _curve(bets, (0.5, 0.9, 0.95), n_boot=500, seed=11)

    assert list(curve.columns) == COLUMNS
    assert set(curve["segment"]) == {"all"}
    got = _segment(curve, "all")
    _check_bands(
        got,
        bets["claim"].to_numpy(),
        bets["ret"].to_numpy(),
        bets["meeting"].to_numpy(),
        (0.5, 0.9, 0.95),
    )
    top_band = got.iloc[-1]
    assert top_band["q_lo"] == pytest.approx(0.95)
    assert int(top_band["n_bets"]) == 1000
    assert abs(top_band["realised"] - 1.4) < 0.1
    assert top_band["lo"] > 1.0
    for k in range(len(got) - 1):
        assert got.iloc[k]["hi"] < 1.0, f"band {k} upper bound {got.iloc[k]['hi']}"


def test_the_same_curve_finds_the_reverse_tail():
    """The reverse: a losing top band among break-even bets (the claim is overconfident there)."""
    rng = np.random.default_rng(5)
    n = 10_000
    claim = rng.uniform(0.8, 1.6, n)
    top = claim >= np.quantile(claim, 0.9)
    ret = 2.0 * (rng.random(n) < np.where(top, 0.2, 0.5))
    bets = pd.DataFrame(
        {"claim": claim, "ret": ret, "meeting": np.repeat([f"M{m}" for m in range(100)], 100)}
    )

    got = _segment(_curve(bets, (0.5, 0.9), n_boot=400, seed=3), "all")

    assert got.iloc[-1]["hi"] < 0.6
    assert got.iloc[-1]["realised"] == pytest.approx(bets.loc[top, "ret"].mean(), rel=1e-10)
    assert got.iloc[0]["lo"] > 0.85 and got.iloc[0]["hi"] > 0.95


# --- the bands ------------------------------------------------------------------------------


def test_bands_split_the_bets_exactly():
    """1,003 lognormal claims, rows shuffled. (n - 1) * 0.5 is an integer, so the 0.5 edge is a
    data point: it belongs to the upper band ([q_k, q_k+1))."""
    rng = np.random.default_rng(17)
    n = 1003
    claim = np.exp(rng.normal(0.0, 0.6, n))
    ret = np.where(rng.random(n) < 0.3, np.round(1.0 / 0.3 * rng.uniform(0.5, 1.5, n), 1), 0.0)
    meeting = np.array([f"V{k % 7}|d{k % 41:02d}" for k in range(n)])
    bets = _shuffled(pd.DataFrame({"claim": claim, "ret": ret, "meeting": meeting}), 2)
    quantiles = (0.5, 0.9, 0.99)

    curve = _curve(bets, quantiles)

    assert list(curve.columns) == COLUMNS
    got = _segment(curve, "all")
    assert len(got) == 4
    assert int(got["n_bets"].sum()) == n
    expected = _check_bands(
        got, bets["claim"].to_numpy(), bets["ret"].to_numpy(), bets["meeting"].to_numpy(), quantiles
    )
    # every bet in exactly one band, and the edges are the empirical quantiles
    membership = np.vstack([e["member"] for e in expected]).sum(axis=0)
    assert (membership == 1).all()
    assert got["q_lo"].tolist() == pytest.approx([0.0, 0.5, 0.9, 0.99])
    assert got["q_hi"].tolist() == pytest.approx([0.5, 0.9, 0.99, 1.0])
    assert got.iloc[0]["claim_lo"] == pytest.approx(claim.min())
    assert got.iloc[-1]["claim_hi"] == pytest.approx(claim.max())
    assert got["claim_hi"].iloc[:-1].tolist() == pytest.approx(got["claim_lo"].iloc[1:].tolist())
    assert got.iloc[1]["claim_lo"] == pytest.approx(np.median(claim))
    assert (np.diff(got["claim_lo"].to_numpy()) > 0).all(), "highest band last"


@pytest.mark.parametrize(
    "quantiles",
    [(0.25, 0.5, 0.75), (0.1,), (0.33, 0.66, 0.8, 0.9, 0.97), (0.6, 0.95)],
)
def test_bands_follow_any_given_quantiles(quantiles):
    rng = np.random.default_rng(len(quantiles) * 101)
    n = 2_400 + 37 * len(quantiles)
    claim = rng.gamma(2.0, 0.5, n)
    ret = 3.0 * (rng.random(n) < 0.3 * claim / (1 + claim))
    meeting = rng.integers(0, 60, n).astype(str)
    bets = pd.DataFrame({"claim": claim, "ret": ret, "meeting": meeting})

    got = _segment(_curve(bets, quantiles), "all")

    assert len(got) == len(quantiles) + 1
    assert int(got["n_bets"].sum()) == n
    _check_bands(got, claim, ret, meeting, quantiles)


def test_curves_by_segment_use_each_segment_own_quantiles():
    """Three classes with disjoint claim ranges and different hit rates: the global edges would
    put each class wholly inside one or two bands, so only per-class quantiles pass."""
    rng = np.random.default_rng(29)
    spec = {"maiden": (0.3, 0.9, 0.2), "allowance": (1.0, 1.5, 0.35), "graded": (1.6, 4.0, 0.5)}
    parts = []
    for i, (label, (low, high, p)) in enumerate(spec.items()):
        n = 1_500 + 400 * i
        parts.append(
            pd.DataFrame(
                {
                    "claim": rng.uniform(low, high, n),
                    "ret": 2.0 * (rng.random(n) < p),
                    "meeting": rng.integers(0, 50, n).astype(str),
                    "race_class": label,
                }
            )
        )
    bets = _shuffled(pd.concat(parts, ignore_index=True), 4)
    quantiles = (0.5, 0.8, 0.95)

    curve = _curve(bets, quantiles, by="race_class")

    assert list(curve.columns) == COLUMNS
    assert set(curve["segment"]) == {"all", *spec}
    assert len(curve) == 4 * (len(spec) + 1)
    for label in ["all", *spec]:
        sub = bets if label == "all" else bets[bets["race_class"] == label]
        got = _segment(curve, label)
        assert int(got["n_bets"].sum()) == len(sub)
        _check_bands(
            got,
            sub["claim"].to_numpy(),
            sub["ret"].to_numpy(),
            sub["meeting"].to_numpy(),
            quantiles,
        )
    # the per-class realised returns differ, so a class curve is not the 'all' curve relabelled
    assert _segment(curve, "maiden")["realised"].mean() < 0.6
    assert _segment(curve, "graded")["realised"].mean() > 0.8

    # by=None is the 'all' curve alone, with the same bands
    alone = _curve(bets, quantiles)
    assert set(alone["segment"]) == {"all"}
    deterministic = ["q_lo", "q_hi", "claim_lo", "claim_hi", "n_bets", "n_clusters", "realised"]
    pd.testing.assert_frame_equal(
        alone[deterministic].reset_index(drop=True),
        _segment(curve, "all")[deterministic],
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("levels", "missing"),
    [
        (("maiden", "allowance", "graded"), (None, np.nan)),
        ((1.0, 2.0, 3.0), (np.nan, np.nan)),
    ],
    ids=["labels-none-nan", "codes-nan"],
)
def test_bets_without_a_segment_stay_in_all_and_leave_the_named_curves(levels, missing):
    """A race whose class code did not map keeps a null class: its bets count in the 'all' curve
    (exactly the by=None curve, bounds included) and in no class curve; each class curve is the
    curve of that class's bets alone, and no segment is labelled by the null."""
    rng = np.random.default_rng(444)
    n = 3_000
    claim = rng.uniform(0.4, 3.0, n)
    # the unmapped bets pay best, so dropping them from 'all' or folding them into a class shows
    unmapped = rng.random(n) < 0.15
    p = np.where(unmapped, 0.7, 0.25)
    bets = pd.DataFrame(
        {
            "claim": claim,
            "ret": 2.0 * (rng.random(n) < p),
            "meeting": rng.integers(0, 60, n).astype(str),
            "race_class": pd.Series([levels[i % len(levels)] for i in range(n)], dtype=object),
        }
    )
    where = np.flatnonzero(unmapped)
    bets.loc[where[::2], "race_class"] = missing[0]
    bets.loc[where[1::2], "race_class"] = missing[1]
    quantiles = (0.5, 0.8, 0.95)

    curve = _curve(bets, quantiles, by="race_class", n_boot=300, seed=21)

    assert list(curve.columns) == COLUMNS
    assert curve["segment"].notna().all()
    assert set(curve["segment"]) == {"all", *levels}
    assert len(curve) == 4 * (len(levels) + 1)

    alone = _curve(bets, quantiles, n_boot=300, seed=21).reset_index(drop=True)
    pd.testing.assert_frame_equal(_segment(curve, "all"), alone, check_dtype=False)
    assert int(_segment(curve, "all")["n_bets"].sum()) == n

    named = bets[~unmapped]
    for level in levels:
        sub = named[named["race_class"] == level].reset_index(drop=True)
        got = _segment(curve, level)
        assert int(got["n_bets"].sum()) == len(sub)
        _check_bands(
            got,
            sub["claim"].to_numpy(),
            sub["ret"].to_numpy(),
            sub["meeting"].to_numpy(),
            quantiles,
        )
        own = _curve(sub, quantiles, n_boot=300, seed=21)
        cols = [c for c in COLUMNS if c != "segment"]
        pd.testing.assert_frame_equal(got[cols], own[cols], check_dtype=False)
    assert sum(int(_segment(curve, lv)["n_bets"].sum()) for lv in levels) == len(named)


# --- the bounds -----------------------------------------------------------------------------


def test_bounds_resample_whole_clusters():
    """Every top-decile bet is in meeting M-top, with varied returns: a bet-level resample would
    spread its mean, a cluster resample of one cluster reproduces it exactly."""
    rng = np.random.default_rng(41)
    n = 3_000
    claim = rng.uniform(0.5, 2.0, n)
    top = claim >= np.quantile(claim, 0.9)
    meeting = np.where(top, "M-top", rng.integers(0, 40, n).astype(str))
    ret = np.where(rng.random(n) < 0.4, rng.uniform(1.5, 4.0, n), 0.0)
    bets = pd.DataFrame({"claim": claim, "ret": ret, "meeting": meeting})
    assert bets.loc[top, "ret"].std() > 0.5  # premise: the band's returns vary

    got = _segment(_curve(bets, (0.5, 0.9), n_boot=500, level=0.95, seed=2), "all")

    top_band = got.iloc[-1]
    assert int(top_band["n_clusters"]) == 1
    assert top_band["realised"] == pytest.approx(bets.loc[top, "ret"].mean(), rel=1e-12)
    assert top_band["lo"] == pytest.approx(top_band["realised"], rel=1e-12)
    assert top_band["hi"] == pytest.approx(top_band["realised"], rel=1e-12)
    # the other bands span the 40 other meetings and have a real interval
    expected = _expected_bands(claim, ret, meeting, (0.5, 0.9))
    for k in (0, 1):
        assert int(got.iloc[k]["n_clusters"]) == expected[k]["n_clusters"] > 30
        assert got.iloc[k]["hi"] - got.iloc[k]["lo"] > 0.01


def _clustered(seed=8, n_meetings=150, per_meeting=40):
    """Meeting-level hit rates U(0.1, 0.7): returns are correlated within a meeting, so the
    cluster standard error is about twice the bet-level one in the bottom band."""
    rng = np.random.default_rng(seed)
    n = n_meetings * per_meeting
    meeting = np.repeat([f"V{m % 5}|day{m:03d}" for m in range(n_meetings)], per_meeting)
    p = np.repeat(rng.uniform(0.1, 0.7, n_meetings), per_meeting)
    claim = rng.uniform(0.6, 1.8, n)
    ret = 2.0 * (rng.random(n) < p)
    return pd.DataFrame({"claim": claim, "ret": ret, "meeting": meeting})


def _cluster_se(ret: np.ndarray, meeting: np.ndarray) -> float:
    """Standard error of the per-bet mean (a ratio) under whole-cluster resampling, linearised:
    sqrt(sum_c (S_c - r * N_c)^2) / N."""
    frame = pd.DataFrame({"ret": ret, "meeting": meeting})
    r = ret.mean()
    by = frame.groupby("meeting")["ret"].agg(["sum", "size"])
    return float(np.sqrt(((by["sum"] - r * by["size"]) ** 2).sum()) / len(ret))


@pytest.mark.parametrize("level", [0.95, 0.8])
def test_bounds_are_the_one_sided_cluster_bootstrap_quantiles(level):
    bets = _clustered()
    quantiles = (0.5, 0.9)
    claims, rets, meetings = (bets[c].to_numpy() for c in ("claim", "ret", "meeting"))

    got = _segment(_curve(bets, quantiles, n_boot=1500, level=level, seed=19), "all")

    expected = _expected_bands(claims, rets, meetings, quantiles)
    z = NormalDist().inv_cdf(level)
    for k, exp in enumerate(expected):
        se = _cluster_se(rets[exp["member"]], meetings[exp["member"]])
        row = got.iloc[k]
        assert row["lo"] == pytest.approx(exp["realised"] - z * se, abs=0.3 * se), f"band {k}"
        assert row["hi"] == pytest.approx(exp["realised"] + z * se, abs=0.3 * se), f"band {k}"
    # premise: in the bottom band the bet-level bound would be far off the cluster one
    bottom = expected[0]["member"]
    se_bets = rets[bottom].std() / np.sqrt(bottom.sum())
    assert _cluster_se(rets[bottom], meetings[bottom]) > 1.6 * se_bets


def test_a_higher_level_widens_the_bounds():
    bets = _clustered(seed=12)
    quantiles = (0.5, 0.9)

    narrow = _segment(_curve(bets, quantiles, n_boot=1000, level=0.8, seed=4), "all")
    wide = _segment(_curve(bets, quantiles, n_boot=1000, level=0.99, seed=4), "all")

    pd.testing.assert_series_equal(narrow["realised"], wide["realised"])
    assert (wide["lo"] < narrow["lo"]).all()
    assert (wide["hi"] > narrow["hi"]).all()


# --- reproducibility and the signature ------------------------------------------------------


def test_a_seed_makes_the_curve_reproducible():
    bets = _clustered(seed=3, n_meetings=60, per_meeting=30)
    bets["race_class"] = np.where(np.arange(len(bets)) % 3 == 0, "maiden", "open")

    first = _curve(bets, (0.5, 0.9), by="race_class", n_boot=300, seed=99)
    again = _curve(bets, (0.5, 0.9), by="race_class", n_boot=300, seed=99)
    pd.testing.assert_frame_equal(first, again)

    others = [_curve(bets, (0.5, 0.9), by="race_class", n_boot=300, seed=s) for s in (1, 2)]
    for other in others:
        pd.testing.assert_series_equal(other["realised"], first["realised"])
    los = {tuple(np.round(frame["lo"].to_numpy(), 12)) for frame in [first, *others]}
    assert len(los) > 1, "the seed must drive the resampling"


def test_level_n_boot_and_seed_are_keyword_only_without_defaults():
    params = inspect.signature(tail_curve).parameters
    for name in ("level", "n_boot", "seed", "claim_col", "return_col", "cluster_col", "quantiles"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    for name in ("level", "n_boot", "seed"):
        assert params[name].default is inspect.Parameter.empty, name
    assert params["by"].default is None

    bets = _clustered(seed=1, n_meetings=10, per_meeting=10)
    with pytest.raises(TypeError):
        tail_curve(bets, **KW, quantiles=(0.5,), n_boot=100, seed=1)
    with pytest.raises(TypeError):
        tail_curve(bets, **KW, quantiles=(0.5,), level=0.95, seed=1)
    with pytest.raises(TypeError):
        tail_curve(bets, **KW, quantiles=(0.5,), n_boot=100, level=0.95)


# --- bad input ------------------------------------------------------------------------------


def _small():
    rng = np.random.default_rng(6)
    n = 200
    return pd.DataFrame(
        {
            "claim": rng.uniform(0.5, 1.5, n),
            "ret": 2.0 * (rng.random(n) < 0.4),
            "meeting": rng.integers(0, 10, n).astype(str).astype(object),
            "race_class": np.where(rng.random(n) < 0.5, "maiden", "open"),
        }
    )


def test_bad_input_raises():
    """A NaN claim or return, a bet without a cluster, or quantiles not strictly increasing in
    (0, 1): ValueError. An unknown column: KeyError."""
    # the valid frame is accepted, so each failure below is its bad value's
    assert len(_curve(_small(), (0.5, 0.9), by="race_class", n_boot=50)) == 9

    bad_values = [("claim", np.nan), ("ret", np.nan), ("meeting", None), ("meeting", np.nan)]
    for column, value in bad_values:
        bets = _small()
        bets.loc[17, column] = value
        with pytest.raises(ValueError):
            _curve(bets, (0.5, 0.9), n_boot=50)

    bad_quantiles = [
        (0.5, 0.5),
        (0.9, 0.5),
        (0.0, 0.5),
        (0.5, 1.0),
        (-0.1, 0.5),
        (0.5, 1.2),
        (0.2, float("nan")),
    ]
    for quantiles in bad_quantiles:
        with pytest.raises(ValueError):
            _curve(_small(), quantiles, n_boot=50)

    for override in ({"claim_col": "nope"}, {"return_col": "nope"}, {"cluster_col": "nope"}):
        with pytest.raises(KeyError):
            tail_curve(
                _small(), **{**KW, **override}, quantiles=(0.5,), n_boot=50, level=0.95, seed=1
            )
    with pytest.raises(KeyError):
        _curve(_small(), (0.5,), by="nope", n_boot=50)
