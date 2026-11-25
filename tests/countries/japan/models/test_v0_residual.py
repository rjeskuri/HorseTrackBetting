"""The v0 shrunk segment residual on the calibrated market (ticket #125, epic #73 C4 offset form).

Contract pinned here (`horsetrack.countries.japan.models.v0.residual`):
- Segment names, the whole vocabulary (anything else is a ValueError):
    `field_16_plus`  the race has 16 or more starters;
    `race_12`        race number 12;
    `races_10_12`    race number 10, 11 or 12;
    `venue:<CODE>`   the race is at venue CODE, an upper-case JRA code from jra.yaml `venues`.
- `ResidualFit(base: WinCalibration, segments: tuple[str, ...], gamma: tuple[float, ...],
  ridge: float, fit_id: str)`, a frozen dataclass; an unknown segment name, or len(gamma) !=
  len(segments), is a ValueError at construction.
- `ResidualFit.predict(q, race)`: P(i wins) proportional to exp(f(ln q_i) + g * ln q_i), f the
  base's spline, g = sum_s gamma_s * z_s(race). Equivalently p_i proportional to
  base.predict(q)_i * q_i ** g. NaN entries stay NaN, the rest sum to 1. The indicators come from
  the RaceContext only: `starters` (never len(q)), `race_no`, `venue`. With no segments `race` may
  be None; with segments a None race is a ValueError.
- `fit_residual(frame, *, base, segments, ridge)`: frame columns race_id, race_date, venue,
  race_no, win_odds, won (horse_no may be present; there is NO starters column). q comes from
  each race's odds as in #114; a race's field size is its number of rows in the frame. gamma is
  the conditional-logit maximum likelihood with base's f(ln q) as a fixed offset and a ridge
  penalty toward 0 (scale is the coder's: the tests only use ridge 1e-6 as "negligible" and
  compare penalties). Returns `base` unchanged, `segments` as a tuple, `ridge` as given.
- `oof_predict(frame, *, folds, segments, ridge, n_knots=5)`: for fold k (positions into
  `frame`), base_k = calibration.fit_win_calibration(train rows, n_knots=n_knots) and
  fit_k = fit_residual(train rows, base=base_k, segments=..., ridge=...): the base is refitted in
  fold too. Columns race_id, horse_no, fold, q, p_cal (base_k), p_res (fit_k, with the race's
  RaceContext built from the frame: venue, race_no, starters = rows of the race).
- `read_segments(path=CONFIG_PATH) -> tuple[str, ...]`: the `segments` key of
  countries/japan/config/residual_v0.yaml; a missing file is FileNotFoundError, a missing key or
  an unknown name a ValueError, `segments: []` gives ().
- `to_json` / `from_json` round-trip (the base is embedded).

Expected values are derived here from first principles (planted p from base.predict * q^g, the
standard errors from the planted model's Fisher information) or, for the out-of-fold contract,
from the separately pinned `fit_residual` and #114's `fit_win_calibration`. All data is
synthetic; HORSETRACK_DATA_DIR points at an empty tmp folder, so nothing can come from a store.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import paths
from horsetrack.core.config import load_source
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.countries.japan.board.snapshot import RaceContext
from horsetrack.countries.japan.models.v0.calibration import WinCalibration, fit_win_calibration
from horsetrack.countries.japan.models.v0.residual import (
    ResidualFit,
    fit_residual,
    oof_predict,
    read_segments,
)

TAKEOUT = 0.8
VENUES = ("TOK", "NAK", "KYO")
ALL_FOUR = ("field_16_plus", "race_12", "races_10_12", "venue:TOK")

# A base that is visibly not the identity (slope 1.12 and a bend), so a fit that took ln q itself
# as the offset would be biased.
BASE = WinCalibration(
    knots=(-4.5, -2.5, -0.8),
    coef=(1.12, 0.04),
    fit_id="base-125",
    train_start=date(2015, 1, 3),
    train_end=date(2024, 12, 28),
    n_races=20_000,
)

PLANTED_SEGMENTS = ("field_16_plus", "race_12", "venue:TOK")
PLANTED_GAMMA = np.array([0.1, 0.0, -0.1])


@pytest.fixture(autouse=True)
def empty_data_dir(tmp_path, monkeypatch):
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


# ---------------------------------------------------------------------------------------------
# Synthetic races and independent references
# ---------------------------------------------------------------------------------------------


def indicator(name, venue, race_no, starters):
    """z_s for arrays of race attributes, written from the segment vocabulary above."""
    venue, race_no, starters = np.asarray(venue), np.asarray(race_no), np.asarray(starters)
    if name == "field_16_plus":
        return (starters >= 16).astype(float)
    if name == "race_12":
        return (race_no == 12).astype(float)
    if name == "races_10_12":
        return ((race_no >= 10) & (race_no <= 12)).astype(float)
    assert name.startswith("venue:")
    return (venue == name.split(":", 1)[1]).astype(float)


def ref_predict(base, q, g):
    """p proportional to base.predict(q) * q**g over the priced entries; NaN stays NaN."""
    q = np.asarray(q, dtype=float)
    w = base.predict(q) * q**g
    return w / np.nansum(w)


def simulate_q(rng, n_races, n_runners, alpha=0.9, lo=0.005, hi=0.6):
    q = rng.dirichlet(np.full(n_runners, alpha), size=n_races)
    for _ in range(3):
        q = np.clip(q, lo, hi)
        q /= q.sum(axis=1, keepdims=True)
    return q


def make_races(
    rng,
    n_races,
    *,
    segments,
    gamma,
    base=BASE,
    races_per_day=6,
    first_day=date(2015, 1, 3),
    sizes=(10, 18),
):
    """Races with mixed field sizes, venues and race numbers, winners drawn from
    p proportional to base.predict(q) * q ** (sum_s gamma_s z_s). Returns the fit frame and the
    flat arrays the references need (codes, q, p_true, Z per race)."""
    k = np.arange(n_races)
    day, slot = k // races_per_day, k % races_per_day
    venue = np.array(VENUES)[slot % 3]
    race_no = 1 + (day * 5 + 6 * (slot // 3)) % 12  # unique per (day, venue) for 6 races a day
    starters = rng.integers(sizes[0], sizes[1] + 1, size=n_races)
    offsets = np.r_[0, np.cumsum(starters)[:-1]]
    n_rows = int(starters.sum())
    codes = np.repeat(k, starters)
    q = np.empty(n_rows)
    for n in np.unique(starters):
        idx = np.flatnonzero(starters == n)
        pos = offsets[idx][:, None] + np.arange(n)[None, :]
        q[pos] = simulate_q(rng, len(idx), int(n))

    z = np.zeros((n_races, len(segments)))
    for j, s in enumerate(segments):
        z[:, j] = indicator(s, venue, race_no, starters)
    g = z @ np.asarray(gamma, dtype=float)
    w = base.predict_many(q, codes) * q ** g[codes]
    p = w / np.bincount(codes, weights=w)[codes]

    u = np.minimum(rng.random(n_races), 1 - 1e-12)
    within = np.cumsum(p) - (np.cumsum(p)[offsets] - p[offsets])[codes]
    won = (within > u[codes]) & (within - p <= u[codes])
    won[offsets + starters - 1] |= np.bincount(codes, weights=won)[k] == 0  # rounding at 1.0
    assert np.all(np.bincount(codes, weights=won) == 1)

    days = [first_day + timedelta(days=int(d)) for d in day]
    ids = [f"jra:{d.isoformat()}:{v}:{r:02d}" for d, v, r in zip(days, venue, race_no, strict=True)]
    frame = pd.DataFrame(
        {
            "race_id": np.repeat(np.array(ids, dtype=object), starters),
            "race_date": np.repeat(np.array(days, dtype=object), starters),
            "venue": np.repeat(venue, starters),
            "race_no": np.repeat(race_no, starters),
            "horse_no": np.concatenate([np.arange(1, n + 1) for n in starters]),
            "win_odds": TAKEOUT / q,
            "won": won,
        }
    )
    return frame, dict(codes=codes, q=q, p=p, z=z)


def fisher_se(arrays):
    """Standard errors of gamma at the planted model: I = sum_r z_r z_r' Var_p(ln q)."""
    codes, q, p, z = arrays["codes"], arrays["q"], arrays["p"], arrays["z"]
    ln_q = np.log(q)
    mean = np.bincount(codes, weights=p * ln_q)
    var = np.bincount(codes, weights=p * ln_q**2) - mean**2
    info = (z * var[:, None]).T @ z
    return np.sqrt(np.diag(np.linalg.inv(info)))


def context(race_id="jra:2026-11-28:TOK:12", venue="TOK", race_no=12, starters=16):
    return RaceContext(race_id=race_id, venue=venue, race_no=race_no, starters=starters)


# ---------------------------------------------------------------------------------------------
# Heavy fits, shared: 20,000 races of 10 to 18 runners
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recovery():
    rng = np.random.default_rng(125)
    frame, arrays = make_races(rng, 20_000, segments=PLANTED_SEGMENTS, gamma=PLANTED_GAMMA)
    assert "starters" not in frame.columns
    return frame, arrays


@pytest.fixture(scope="module")
def ridge_path(recovery):
    frame, _ = recovery
    return {
        ridge: fit_residual(frame, base=BASE, segments=list(PLANTED_SEGMENTS), ridge=ridge)
        for ridge in (1e-6, 10.0, 1e5)
    }


# ---------------------------------------------------------------------------------------------
# DoD: with gamma = 0 the predictions equal #114's calibrated market
# ---------------------------------------------------------------------------------------------

PROBES = [
    np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]),
    np.array([0.5, 0.3, 0.2]),
    np.array([0.3, np.nan, 0.15, 0.12, 0.1, np.nan, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]),
]


def test_zero_residual_equals_the_calibrated_market():
    rng = np.random.default_rng(0)
    probes = [*PROBES, rng.dirichlet(np.full(16, 0.8)), rng.dirichlet(np.full(18, 0.5))]
    contexts = [
        context(),
        context(venue="NAK", race_no=3, starters=9, race_id="jra:2026-11-28:NAK:03"),
        context(venue="TOK", race_no=11, starters=18, race_id="jra:2026-11-28:TOK:11"),
    ]
    for base in (BASE, dataclasses_replace_coef(BASE, (0.9, -0.03))):
        fit = ResidualFit(
            base=base, segments=ALL_FOUR, gamma=(0.0,) * 4, ridge=1.0, fit_id="zero-125"
        )
        for q in probes:
            want = base.predict(q)
            for ctx in contexts:
                got = fit.predict(q, ctx)
                assert got.shape == q.shape
                np.testing.assert_array_equal(np.isnan(got), np.isnan(q))
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def dataclasses_replace_coef(base, coef):
    return dataclasses.replace(base, coef=coef, fit_id=f"{base.fit_id}-b")


# ---------------------------------------------------------------------------------------------
# DoD: planted segment slopes are recovered, and the ridge shrinks them toward 0
# ---------------------------------------------------------------------------------------------


def test_planted_segment_slopes_are_recovered(recovery, ridge_path):
    """gamma(field of 16 or more) = 0.1, gamma(race 12) = 0, gamma(TOK) = -0.1 planted on top of a
    non-identity base; with a negligible ridge each is within 3 standard errors, the SEs from the
    planted model's Fisher information. Field size comes from the rows per race (no column)."""
    _, arrays = recovery
    se = fisher_se(arrays)
    # The data can tell 0.1 from 0, so returning no residual cannot pass.
    assert 3 * se[0] < 0.1 and 3 * se[2] < 0.1
    fit = ridge_path[1e-6]
    assert isinstance(fit, ResidualFit)
    assert fit.segments == PLANTED_SEGMENTS
    assert fit.base == BASE
    assert fit.ridge == 1e-6
    assert isinstance(fit.fit_id, str) and fit.fit_id
    got = np.asarray(fit.gamma, dtype=float)
    assert got.shape == (3,)
    assert np.all(np.abs(got - PLANTED_GAMMA) < 3 * se), (got.tolist(), se.tolist())


def test_the_ridge_shrinks_toward_zero(ridge_path):
    ridges = sorted(ridge_path)
    gammas = [np.asarray(ridge_path[r].gamma, dtype=float) for r in ridges]
    norms = [float(np.linalg.norm(g)) for g in gammas]
    # A larger penalty gives a smaller |gamma|, strictly, and the largest one shrinks it hard.
    assert norms[0] > norms[1] > norms[2], norms
    assert norms[2] < 0.5 * norms[0], norms
    small, big = gammas[0], gammas[-1]
    for s in (0, 2):  # the planted non-zero slopes: same sign or zero, smaller in size
        assert abs(big[s]) < 0.5 * abs(small[s])
        assert big[s] * small[s] >= 0
    assert all(ridge_path[r].ridge == r for r in ridges)
    assert len({ridge_path[r].fit_id for r in ridges}) == len(ridges)


# ---------------------------------------------------------------------------------------------
# DoD: an empty segment list gives no residual
# ---------------------------------------------------------------------------------------------


def test_an_empty_segment_list_gives_no_residual():
    rng = np.random.default_rng(3)
    frame, _ = make_races(rng, 2_000, segments=("field_16_plus",), gamma=(0.3,))
    for segments in ([], ()):
        fit = fit_residual(frame, base=BASE, segments=segments, ridge=1.0)
        assert fit.segments == ()
        assert tuple(fit.gamma) == ()
        assert fit.base == BASE
        for q in PROBES:
            want = BASE.predict(q)
            for race in (None, context(), context(venue="KYO", race_no=1, starters=5)):
                np.testing.assert_allclose(fit.predict(q, race), want, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# DoD: segments are read from the race context at predict time
# ---------------------------------------------------------------------------------------------

CONTEXT_CASES = {
    # (venue, race_no, starters, len(q)) -> which of ALL_FOUR are on
    "TOK R12, 16 starters": (("TOK", 12, 16, 16), (1, 1, 1, 1)),
    "15 starters": (("TOK", 12, 15, 15), (0, 1, 1, 1)),
    "race 11": (("TOK", 11, 16, 16), (1, 0, 1, 1)),
    "race 9": (("TOK", 9, 18, 18), (1, 0, 0, 1)),
    "race 10, other venue": (("NAK", 10, 16, 16), (1, 0, 1, 0)),
    "nothing on": (("KYO", 3, 8, 8), (0, 0, 0, 0)),
    "starters from the context, not len(q): 16 starters, 10 prices": (
        ("TOK", 12, 16, 10),
        (1, 1, 1, 1),
    ),
    "starters from the context, not len(q): 15 starters, 17 prices": (
        ("NAK", 12, 15, 17),
        (0, 1, 1, 0),
    ),
}
GAMMA_FOUR = (0.3, -0.2, 0.15, 0.25)


@pytest.mark.parametrize("case", list(CONTEXT_CASES.values()), ids=list(CONTEXT_CASES))
def test_segments_are_read_from_the_race_context(case):
    (venue, race_no, starters, n), on = case
    fit = ResidualFit(base=BASE, segments=ALL_FOUR, gamma=GAMMA_FOUR, ridge=1.0, fit_id="ctx-125")
    race = RaceContext(
        race_id=f"jra:2026-11-28:{venue}:{race_no:02d}",
        venue=venue,
        race_no=race_no,
        starters=starters,
    )
    g = float(np.dot(on, GAMMA_FOUR))
    rng = np.random.default_rng(n * 100 + race_no)
    for q in (rng.dirichlet(np.full(n, 0.9)), rng.dirichlet(np.full(n, 3.0))):
        got = fit.predict(q, race)
        want = ref_predict(BASE, q, g)
        assert got.sum() == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)
        if g != 0:  # the case can fail: the residual moves p away from the base
            assert np.abs(want - BASE.predict(q)).max() > 1e-3
    # NaN entries stay NaN, the priced rest follow the same rule.
    q = np.r_[rng.dirichlet(np.full(n, 0.9)) * 0.9, np.nan]
    got = fit.predict(q, race)
    assert np.isnan(got[-1])
    np.testing.assert_allclose(got[:-1], ref_predict(BASE, q, g)[:-1], rtol=0, atol=1e-10)


def test_the_official_example_context_turns_every_segment_on():
    race = RaceContext(race_id="jra:2026-11-28:TOK:12", venue="TOK", race_no=12, starters=16)
    q = np.random.default_rng(12).dirichlet(np.full(16, 0.9))
    for seg, gamma in zip(ALL_FOUR, GAMMA_FOUR, strict=True):
        fit = ResidualFit(base=BASE, segments=(seg,), gamma=(gamma,), ridge=0.5, fit_id=seg)
        np.testing.assert_allclose(fit.predict(q, race), ref_predict(BASE, q, gamma), atol=1e-10)


def test_a_missing_race_context_is_refused_when_there_are_segments():
    fit = ResidualFit(base=BASE, segments=("race_12",), gamma=(0.2,), ridge=1.0, fit_id="x")
    with pytest.raises(ValueError):
        fit.predict(PROBES[0], None)
    with pytest.raises(ValueError):
        fit.predict(PROBES[0])


BAD_NAMES = ["bogus", "field_15_plus", "race_13", "venue:", "venue:tok", "venue:XYZ", "Venue:TOK"]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_an_unknown_segment_name_is_refused(name):
    with pytest.raises(ValueError):
        ResidualFit(base=BASE, segments=(name,), gamma=(0.1,), ridge=1.0, fit_id="bad")
    with pytest.raises(ValueError):
        ResidualFit(base=BASE, segments=("race_12", name), gamma=(0.1, 0.1), ridge=1.0, fit_id="b")
    rng = np.random.default_rng(4)
    frame, _ = make_races(rng, 300, segments=(), gamma=())
    with pytest.raises(ValueError):
        fit_residual(frame, base=BASE, segments=["field_16_plus", name], ridge=1.0)


def test_every_jra_venue_and_the_three_race_segments_are_accepted():
    venues = tuple(f"venue:{code}" for code in load_source("japan", "jra").venues)
    assert len(venues) == 10
    names = ("field_16_plus", "race_12", "races_10_12", *venues)
    fit = ResidualFit(base=BASE, segments=names, gamma=(0.0,) * len(names), ridge=1.0, fit_id="v")
    assert fit.segments == names
    with pytest.raises(ValueError):  # one gamma per segment
        ResidualFit(base=BASE, segments=names, gamma=(0.0,), ridge=1.0, fit_id="v")


# ---------------------------------------------------------------------------------------------
# Config: the owner's list, from residual_v0.yaml only
# ---------------------------------------------------------------------------------------------


def test_the_committed_config_holds_the_owners_list():
    got = read_segments()
    assert isinstance(got, tuple)
    assert {"field_16_plus", "race_12", "races_10_12"} <= set(got)
    rest = [s for s in got if s not in {"field_16_plus", "race_12", "races_10_12"}]
    codes = set(load_source("japan", "jra").venues)
    assert rest, "the owner's list includes venue segments"
    assert all(s.startswith("venue:") and s.split(":", 1)[1] in codes for s in rest), rest
    assert len(set(got)) == len(got)


READ_CASES = {
    "two names": ("segments: [field_16_plus, venue:NAK]\n", ("field_16_plus", "venue:NAK")),
    "block list": ("segments:\n  - races_10_12\n  - race_12\n", ("races_10_12", "race_12")),
    "empty list": ("segments: []\n", ()),
}


@pytest.mark.parametrize("case", list(READ_CASES.values()), ids=list(READ_CASES))
def test_read_segments_reads_the_segments_key(tmp_path, case):
    text, want = case
    path = tmp_path / "residual_v0.yaml"
    path.write_text(text, encoding="utf-8")
    assert read_segments(path) == want


@pytest.mark.parametrize(
    "text,error",
    [
        (None, FileNotFoundError),
        ("other: 1\n", ValueError),
        ("segments: [field_16_plus, nope]\n", ValueError),
    ],
    ids=["missing file", "no segments key", "unknown name"],
)
def test_read_segments_refuses_a_bad_config(tmp_path, text, error):
    path = tmp_path / "residual_v0.yaml"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    with pytest.raises(error):
        read_segments(path)


# ---------------------------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------------------------


def test_the_fit_round_trips_through_json(ridge_path):
    built = ResidualFit(base=BASE, segments=ALL_FOUR, gamma=GAMMA_FOUR, ridge=2.5, fit_id="rt-125")
    for original in (built, ridge_path[1e-6]):
        text = original.to_json()
        json.loads(text)
        back = ResidualFit.from_json(text)
        assert back == original
        assert isinstance(back.base, WinCalibration) and back.base == original.base
        for q in PROBES:
            for race in (context(), context(venue="NAK", race_no=10, starters=12)):
                assert np.array_equal(back.predict(q, race), original.predict(q, race), True)


# ---------------------------------------------------------------------------------------------
# Out of fold over #70's folds: base and residual refitted on each fold's training rows
# ---------------------------------------------------------------------------------------------

MONTHLY = FoldScheme(
    unit="month",
    test_length=1,
    step=1,
    min_train=6,
    window="expanding",
    embargo_days=0,
    anchor_month=1,
)
FOLD_J = 3
OOF_SEGMENTS = ["field_16_plus", "races_10_12", "venue:TOK"]
OOF_RIDGE = 1e-3


@pytest.fixture(scope="module")
def oof_case():
    rng = np.random.default_rng(2125)
    frame, _ = make_races(
        rng, 3_000, segments=tuple(OOF_SEGMENTS), gamma=(0.15, -0.1, 0.1), races_per_day=6
    )
    folds = walk_forward(frame["race_date"], scheme=MONTHLY)
    assert len(folds) >= FOLD_J + 3
    base = oof_predict(frame, folds=folds, segments=OOF_SEGMENTS, ridge=OOF_RIDGE, n_knots=4)
    return frame, folds, base


def keyed(oof):
    return oof.set_index(["race_id", "horse_no"]).sort_index()


def rows_of(frame, fold):
    return frame.iloc[fold.test_idx].set_index(["race_id", "horse_no"]).index


def test_oof_p_res_is_the_fold_training_fit_applied_to_each_race(oof_case):
    frame, folds, base = oof_case
    assert {"race_id", "horse_no", "fold", "q", "p_cal", "p_res"} <= set(base.columns)
    got = keyed(base)
    assert not got.index.duplicated().any()
    assert len(got) == sum(len(f.test_idx) for f in folds)
    for k in (0, FOLD_J, len(folds) - 1):
        fold = folds[k]
        train = frame.iloc[fold.train_idx]
        cal = fit_win_calibration(train, n_knots=4)
        res = fit_residual(train, base=cal, segments=OOF_SEGMENTS, ridge=OOF_RIDGE)
        assert res.base == cal
        test = frame.iloc[fold.test_idx]
        for race_id, g in test.groupby("race_id", sort=False):
            inv = 1.0 / g["win_odds"].to_numpy()
            q = inv / inv.sum()
            race = RaceContext(
                race_id=race_id,
                venue=str(g["venue"].iloc[0]),
                race_no=int(g["race_no"].iloc[0]),
                starters=len(g),
            )
            keys = list(zip([race_id] * len(g), g["horse_no"], strict=True))
            assert (got.loc[keys, "fold"] == k).all()
            np.testing.assert_allclose(got.loc[keys, "q"].to_numpy(), q, atol=1e-12)
            np.testing.assert_allclose(got.loc[keys, "p_cal"].to_numpy(), cal.predict(q), atol=1e-8)
            np.testing.assert_allclose(
                got.loc[keys, "p_res"].to_numpy(), res.predict(q, race), atol=1e-6
            )


def test_oof_predictions_do_not_see_their_own_fold(oof_case):
    frame, folds, base = oof_case
    flipped = frame.copy()
    test_rows = folds[FOLD_J].test_idx
    # Every race of fold j is now won by its longest shot.
    sub = flipped.iloc[test_rows]
    longest = sub.groupby("race_id")["win_odds"].transform("max").to_numpy()
    flipped.loc[flipped.index[test_rows], "won"] = sub["win_odds"].to_numpy() == longest
    assert (flipped["won"] != frame["won"]).sum() > 100

    again = oof_predict(flipped, folds=folds, segments=OOF_SEGMENTS, ridge=OOF_RIDGE, n_knots=4)
    a, b = keyed(base), keyed(again)
    for k in range(FOLD_J + 1):
        idx = rows_of(frame, folds[k])
        for col in ("p_res", "p_cal"):
            np.testing.assert_allclose(
                b.loc[idx, col].to_numpy(), a.loc[idx, col].to_numpy(), rtol=0, atol=1e-12
            )
    # A later fold trains on fold j, so the flip must reach its residual (the test can fail).
    later = rows_of(frame, folds[FOLD_J + 1])
    assert np.abs(b.loc[later, "p_res"].to_numpy() - a.loc[later, "p_res"].to_numpy()).max() > 1e-6
