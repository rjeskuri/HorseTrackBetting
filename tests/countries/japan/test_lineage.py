"""C2 lineage: the pedigree relationship matrix and the sire x surface x distance-band prior
(ticket #95, epic #73; 08 §8 B5, §4 E37; 05 §4.1-4.3).

Contract pinned here (`horsetrack.countries.japan.models.lineage`):
- `relationship_matrix(pedigree, horses) -> pd.DataFrame`: the additive relationship A among
  `horses`, its index and columns both `horses` in the order given, floats. `pedigree` has
  horse_id, sire_id and dam_id (other columns ignored, rows in any order, a missing parent, None
  or NA, unknown). A horse with no row, or named only as a parent, is a founder: unrelated to the
  other founders and not inbred. A(i, i) = 1 + A(sire_i, dam_i) / 2 and, for i not an ancestor of
  j, A(i, j) = (A(sire_i, j) + A(dam_i, j)) / 2, an unknown parent counting 0. Only the horses'
  ancestors enter: a row for a descendant or for an unrelated horse never changes an entry.
- `sire_cells(runs, pedigree, *, before_utc) -> pd.DataFrame`: one row per (sire_id, surface,
  dist_band) with at least one run of the sire's progeny whose race_utc is strictly before
  before_utc, with n_runs (those runs) and n_horses (the distinct progeny among them). A run at
  before_utc or later never counts, nor a run of a horse without a known sire. Row order is not
  pinned.
- `LineagePrior(*, tau_cell, damsire_weight=0.25)`; `.fit(runs, pedigree, *, before_utc,
  trainer_effects=None)` returns the fitted LineagePrior. runs: race_id, race_utc (tz-aware UTC),
  horse_id, finish_pos (1 = winner), surface and dist_band (other columns ignored); only runs
  with race_utc strictly before before_utc are read. Surfaces and distance bands are the caller's
  opaque labels. The effects are on the utility scale of a Plackett-Luce model of the finishing
  order (#92's u): each sire's overall effect g_sire, shrunk toward 0, and its cell effects
  g(sire, surface, band) ~ N(g_sire, tau_cell^2), so a cell with few progeny runs stays near
  g_sire and one with many follows its data. `trainer_effects` maps a trainer id to its effect
  (for instance #92's fitted trainer effects on the same window).
- `.prior_mean(horse_id, surface, dist_band, *, trainer_id=None) -> float`, the debut prior mean
  of the horse's ability a_h: trainer_effects[trainer_id] (0 without a trainer, for a trainer the
  mapping lacks, or without trainer_effects) + g(sire, surface, band) + damsire_weight *
  g(damsire, surface, band). One table g serves both roles: the damsire's term is its own cell
  effect as a sire. The damsire is the sire of the horse's dam (every frame here gives the dam's
  own row and a dam_sire_id that agrees with it, so either source works). g of a cell in which a
  stallion's progeny never ran before before_utc is its g_sire; g of a stallion without a progeny
  run before before_utc, or of an unknown one, is 0; a horse without a pedigree row has lineage 0.

Not pinned: the prior on g_sire and how it is set; the Plackett-Luce k and discounts; the
optimiser; Henderson's sparse inverse; how the prior enters #92's model and #93's aptitudes; the
distance bands' cut points (E37's frozen spec); the E37 run, its report, registry records and
leaderboard rows, which read the local stores. Expected values come from first principles (the
recursive method in lineage_synth.py, hand counts) or are planted, never from the code under
test. Every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.models.lineage import (
    LineagePrior,
    relationship_matrix,
    sire_cells,
)

# isort: split
from lineage_synth import (
    FIRST_POST,
    LONG,
    MIDDLE,
    MILE,
    SPRINT,
    Cell,
    ancestors,
    lineage_runs,
    pedigree,
    random_pedigree,
    reference_relationship,
)

CELL_COLUMNS = {"sire_id", "surface", "dist_band", "n_runs", "n_horses"}

# ---------------------------------------------------------------------------------------------
# DoD: relationship coefficients are right on known pedigrees and use ancestors only
# ---------------------------------------------------------------------------------------------

# The ticket's pedigree: unrelated founders S, D1 and D2; X and Y out of S x D1, Z out of S x D2.
TICKET_ROWS = [("X", "S", "D1"), ("Y", "S", "D1"), ("Z", "S", "D2")]
FOUNDER_ROWS = [("S", None, None), ("D1", None, None), ("D2", None, None)]
SIX = ["S", "D1", "D2", "X", "Y", "Z"]
TICKET_A = np.array(
    [
        # S    D1   D2   X     Y     Z
        [1.0, 0.0, 0.0, 0.5, 0.5, 0.5],  # S
        [0.0, 1.0, 0.0, 0.5, 0.5, 0.0],  # D1
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.5],  # D2
        [0.5, 0.5, 0.0, 1.0, 0.5, 0.25],  # X
        [0.5, 0.5, 0.0, 0.5, 1.0, 0.25],  # Y
        [0.5, 0.0, 0.5, 0.25, 0.25, 1.0],  # Z
    ]
)


def checked_matrix(ped: pd.DataFrame, horses) -> pd.DataFrame:
    """relationship_matrix(ped, horses), checked for its shape and labels."""
    horses = list(horses)
    got = relationship_matrix(ped, horses)
    assert isinstance(got, pd.DataFrame)
    assert list(got.index) == horses
    assert list(got.columns) == horses
    values = got.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    return got


def assert_matrix(got: pd.DataFrame, horses, want) -> None:
    sub = got.loc[list(horses), list(horses)].to_numpy(dtype=float)
    np.testing.assert_allclose(sub, np.asarray(want, dtype=float), rtol=0, atol=1e-12)


def test_relationship_coefficients_of_known_pedigrees():
    """The ticket's pedigree: A(X, Y) = 0.5 (full sibs), A(X, Z) = 0.25 (half sibs), A(S, X) =
    0.5 (parent), A(X, X) = 1 (not inbred), and the whole 6 x 6 matrix by hand. The same with the
    founders given rows or not, with NA in place of None, with rows not in birth order, and in
    another order of `horses` (the index follows it). A horse with no row at all is a founder."""
    variants = {
        "founders with rows": pedigree(FOUNDER_ROWS + TICKET_ROWS),
        "founders without rows": pedigree(TICKET_ROWS),
        "rows not in birth order": pedigree((TICKET_ROWS + FOUNDER_ROWS)[::-1]),
        "string dtype (NA parents)": pedigree(TICKET_ROWS + FOUNDER_ROWS).astype("string"),
    }
    for name, ped in variants.items():
        a = checked_matrix(ped, SIX)
        assert a.loc["X", "Y"] == pytest.approx(0.5, abs=1e-12), name
        assert a.loc["X", "Z"] == pytest.approx(0.25, abs=1e-12), name
        assert a.loc["S", "X"] == pytest.approx(0.5, abs=1e-12), name
        assert a.loc["X", "X"] == pytest.approx(1.0, abs=1e-12), name
        assert_matrix(a, SIX, TICKET_A)

        order = ["Z", "X", "D2", "S", "Y", "D1"]
        perm = [SIX.index(h) for h in order]
        assert_matrix(checked_matrix(ped, order), order, TICKET_A[np.ix_(perm, perm)])

        lone = checked_matrix(ped, ["X", "U:not-in-pedigree", "D2"])
        assert_matrix(lone, ["X", "U:not-in-pedigree", "D2"], np.eye(3))


# More known pedigrees on the ticket's: D3 and D4 more founders; W out of the full sibs X x Y,
# V out of S x his daughter X, H out of the half sibs Z x X, C1 out of X x D3 and C2 out of
# Y x D4 (first cousins), K out of the cousins C1 x C2.
MATINGS = [
    ("W", "X", "Y"),
    ("V", "S", "X"),
    ("H", "Z", "X"),
    ("C1", "X", "D3"),
    ("C2", "Y", "D4"),
    ("K", "C1", "C2"),
]
# (i, j, A(i, j)) by the recursion, worked by hand
KNOWN = [
    ("W", "W", 1.25),  # F = A(X, Y) / 2 = 0.25
    ("W", "X", 0.75),  # (A(X, X) + A(Y, X)) / 2
    ("W", "S", 0.5),
    ("W", "D1", 0.5),
    ("W", "Z", 0.25),
    ("V", "V", 1.25),  # F = A(S, X) / 2 = 0.25
    ("S", "V", 0.75),  # (A(S, S) + A(X, S)) / 2
    ("X", "V", 0.75),
    ("D1", "V", 0.25),
    ("H", "H", 1.125),  # F = A(Z, X) / 2 = 0.125
    ("H", "S", 0.5),
    ("C1", "C2", 0.125),  # first cousins
    ("C1", "C1", 1.0),
    ("K", "K", 1.0625),  # F = A(C1, C2) / 2
    ("K", "X", 0.375),  # (A(C1, X) + A(C2, X)) / 2 = (0.5 + 0.25) / 2
    ("K", "D3", 0.25),
    ("W", "V", 0.625),  # (A(X, V) + A(Y, V)) / 2 = (0.75 + 0.5) / 2
]


def test_relationship_counts_inbreeding_on_known_matings():
    """Inbred matings on the ticket's pedigree give A(i, i) = 1 + F above 1: a full-sib mating
    1.25, a sire x daughter mating 1.25 (and A(S, V) = 0.75), a half-sib mating 1.125, a
    first-cousin mating 1.0625; the matrix is symmetric."""
    ped = pedigree(TICKET_ROWS + FOUNDER_ROWS + MATINGS)
    horses = sorted({h for pair in KNOWN for h in pair[:2]})
    a = checked_matrix(ped, horses)
    for i, j, want in KNOWN:
        assert a.loc[i, j] == pytest.approx(want, abs=1e-12), (i, j)
        assert a.loc[j, i] == pytest.approx(want, abs=1e-12), (j, i)
    np.testing.assert_allclose(a.to_numpy(float), a.to_numpy(float).T, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", [9501, 9502, 9503])
def test_relationship_matrix_matches_the_recursive_method_on_random_pedigrees(seed):
    """Random six-generation pedigrees with matings across generations (inbreeding arises) and
    unknown parents, rows shuffled: every entry, among every horse named, is the recursive
    method's."""
    ped = random_pedigree(seed)
    named = set(ped["horse_id"]) | set(ped["sire_id"].dropna()) | set(ped["dam_id"].dropna())
    horses = list(np.random.default_rng(seed).permutation(sorted(named)))
    want = reference_relationship(ped, horses)
    assert (np.diag(want) > 1.0).sum() >= 3, "the design should hold inbred horses"
    assert ((want > 0.5) & ~np.eye(len(horses), dtype=bool)).any()
    got = checked_matrix(ped, horses)
    np.testing.assert_allclose(got.to_numpy(float), want, rtol=0, atol=1e-12)


def test_relationship_matrix_uses_ancestors_only():
    """Adding an offspring of X (O out of X x D2), an inbred one (W out of X x Y), a later
    generation (G out of O x W) and an unrelated horse changes no entry among S, D1, D2, X, Y and
    Z, alone or alongside the new horses. On random pedigrees the matrix of a set of horses is the
    one computed from their ancestors' rows alone, and their descendants' rows never move it. A
    horse's own ancestry does: giving Z the dam D1 makes X and Z full sibs."""
    base = TICKET_ROWS + FOUNDER_ROWS
    grown = pedigree(base + [("O", "X", "D2"), ("W", "X", "Y"), ("G", "O", "W"), ("Q", "D3", "D4")])
    assert_matrix(checked_matrix(grown, SIX), SIX, TICKET_A)
    assert_matrix(checked_matrix(grown, SIX + ["O", "W", "G"]), SIX, TICKET_A)

    for seed in (9511, 9512, 9513):
        ped = random_pedigree(seed)
        rng = np.random.default_rng(seed)
        chosen = list(rng.choice(ped["horse_id"].to_numpy(dtype=object), 8, replace=False))
        full = checked_matrix(ped, chosen).to_numpy(float)
        keep = ancestors(ped, chosen)
        own = checked_matrix(ped.loc[ped["horse_id"].isin(keep)], chosen).to_numpy(float)
        np.testing.assert_allclose(own, full, rtol=0, atol=1e-12)
        np.testing.assert_allclose(full, reference_relationship(ped, chosen), rtol=0, atol=1e-12)
        kids = [(f"kid{i}", chosen[i], chosen[i + 1]) for i in range(0, 8, 2)]
        kids.append(("kid9", "kid0", "kid2"))
        rows = rows_of(ped)
        more = pedigree(rows + kids)
        grown_full = checked_matrix(more, chosen).to_numpy(float)
        np.testing.assert_allclose(grown_full, full, rtol=0, atol=1e-12)

    sibs = pedigree([("X", "S", "D1"), ("Y", "S", "D1"), ("Z", "S", "D1")])
    assert checked_matrix(sibs, ["X", "Z"]).loc["X", "Z"] == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# DoD: progeny counts and sire-cell effects at t use only runs before t
# ---------------------------------------------------------------------------------------------

DAY = pd.Timedelta(days=1)
# Six races: (race_id, post time, surface, band, runners). SA sires h1 and h2, SB sires h3 and h6;
# h4 has no pedigree row and h5's sire is unknown, so neither is ever counted. r5 runs 30 minutes
# after r4 on the same day.
CELL_RACES = [
    ("r1", FIRST_POST, "turf", MILE, ["h1", "h3", "h4"]),
    ("r2", FIRST_POST + DAY, "turf", MILE, ["h1", "h2"]),
    ("r3", FIRST_POST + 2 * DAY, "dirt", MILE, ["h2", "h5", "h3"]),
    ("r4", FIRST_POST + 3 * DAY, "turf", SPRINT, ["h1", "h6"]),
    ("r5", FIRST_POST + 3 * DAY + pd.Timedelta(minutes=30), "dirt", MILE, ["h3", "h2"]),
    ("r6", FIRST_POST + 4 * DAY, "turf", MILE, ["h2", "h6"]),
]
CELL_PEDIGREE = [
    ("h1", "SA", "m1"),
    ("h2", "SA", "m2"),
    ("h3", "SB", "m1"),
    ("h5", None, "m3"),
    ("h6", "SB", None),
    ("m1", "SC", None),
]


def cell_runs() -> pd.DataFrame:
    rows = []
    for race, post, surface, band, horses in CELL_RACES:
        for place, h in enumerate(horses, start=1):
            rows.append((race, post, h, place, surface, band, f"t:{place}"))
    columns = ["race_id", "race_utc", "horse_id", "finish_pos", "surface", "dist_band"]
    return pd.DataFrame(rows, columns=[*columns, "trainer_id"])


def counted(runs: pd.DataFrame, ped: pd.DataFrame, before) -> dict:
    """{(sire, surface, band): (n_runs, n_horses)}, by a direct loop over the runs before
    `before`."""
    sire_of = {h: s for h, s in zip(ped["horse_id"], ped["sire_id"], strict=True) if s is not None}
    n_runs: dict = defaultdict(int)
    horses: dict = defaultdict(set)
    for row in runs.itertuples(index=False):
        if row.race_utc < before and row.horse_id in sire_of:
            key = (sire_of[row.horse_id], row.surface, row.dist_band)
            n_runs[key] += 1
            horses[key].add(row.horse_id)
    return {k: (n_runs[k], len(horses[k])) for k in n_runs}


def cells_of(table: pd.DataFrame) -> dict:
    assert isinstance(table, pd.DataFrame)
    if len(table) == 0:
        return {}
    assert CELL_COLUMNS <= set(table.columns), sorted(table.columns)
    keys = list(zip(table["sire_id"], table["surface"], table["dist_band"], strict=True))
    assert len(set(keys)) == len(keys), "one row per cell"
    values = zip(table["n_runs"].astype(int), table["n_horses"].astype(int), strict=True)
    return dict(zip(keys, [(int(r), int(h)) for r, h in values], strict=True))


def test_sire_cell_counts_are_as_of_t():
    """Progeny runs and distinct progeny per sire x surface x band, as of t. At r5's post time
    (r5 itself excluded) by hand: SA turf mile 3 runs of 2 horses, SB turf mile 1 of 1, SA and
    SB dirt mile 1 of 1 each, SA and SB turf sprint 1 of 1 each; h4 (no row) and h5 (no sire)
    never count. A moment after r5, both dirt-mile cells hold 2 runs of 1 horse. Over a spread of
    cut times (before the first race, at it, a second after, at and after each later post) the
    table is the direct count of the runs strictly before t, rows in any order."""
    runs = cell_runs()
    ped = pedigree(CELL_PEDIGREE)
    r5 = CELL_RACES[4][1]
    at_r5 = {
        ("SA", "turf", MILE): (3, 2),
        ("SB", "turf", MILE): (1, 1),
        ("SA", "dirt", MILE): (1, 1),
        ("SB", "dirt", MILE): (1, 1),
        ("SA", "turf", SPRINT): (1, 1),
        ("SB", "turf", SPRINT): (1, 1),
    }
    assert cells_of(sire_cells(runs, ped, before_utc=r5)) == at_r5
    assert counted(runs, ped, r5) == at_r5
    after_r5 = at_r5 | {("SA", "dirt", MILE): (2, 1), ("SB", "dirt", MILE): (2, 1)}
    later = r5 + pd.Timedelta(seconds=1)
    assert cells_of(sire_cells(runs, ped, before_utc=later.to_pydatetime())) == after_r5

    second = pd.Timedelta(seconds=1)
    cuts = [FIRST_POST - second, FIRST_POST, FIRST_POST + second]
    for _, post, *_ in CELL_RACES[1:]:
        cuts += [post, post + pd.Timedelta(microseconds=1)]
    cuts.append(FIRST_POST + 10 * DAY)
    shuffled = runs.sample(frac=1.0, random_state=95).reset_index(drop=True)
    seen = set()
    for cut in cuts:
        want = counted(runs, ped, cut)
        seen.add(tuple(sorted(want.items())))
        assert cells_of(sire_cells(runs, ped, before_utc=cut)) == want, cut
        assert cells_of(sire_cells(shuffled, ped, before_utc=cut)) == want, cut
    assert len(seen) >= 6, "the cut times should give several different tables"


# The as-of design: one focal sire in two mile cells and a second in turf sprints, against the
# field; 420 races, cut at the post time of a race about 60% of the way through.
ASOF_CELLS = (
    Cell("s:one", "turf", MILE, 150, 0.8),
    Cell("s:one", "dirt", MILE, 150, -0.4),
    Cell("s:two", "turf", SPRINT, 120, 0.3),
)
PROBES = [
    ("p:one", "s:one", None),
    ("p:two", "s:two", None),
    ("p:late", "s:late", None),
    ("p:field", "f:03", None),
]
GRID = [("turf", MILE), ("dirt", MILE), ("turf", SPRINT), ("dirt", LONG)]


def asof_design_data():
    """(runs, pedigree with the probes and s:late's foals, t) of the as-of design; t is a race's
    post time."""
    runs, ped, _ = lineage_runs(9531, ASOF_CELLS)
    posts = np.sort(runs["race_utc"].unique())
    t = pd.Timestamp(posts[int(0.6 * len(posts))])
    late = [(f"s:late/h{i}", "s:late", None) for i in range(3)]
    return runs, pedigree(rows_of(ped) + late + PROBES), t


def rows_of(ped: pd.DataFrame) -> list[tuple]:
    return list(ped[["horse_id", "sire_id", "dam_id"]].itertuples(index=False, name=None))


def priors(model, horses=("p:one", "p:two", "p:late", "p:field"), grid=GRID) -> np.ndarray:
    return np.array([[prior(model, h, s, b) for s, b in grid] for h in horses])


def prior(model, horse, surface, band, **kw) -> float:
    value = model.prior_mean(horse, surface, band, **kw)
    assert isinstance(value, (float, np.floating)), type(value)
    assert math.isfinite(value)
    return float(value)


def fitted(runs, ped, before, *, tau_cell=0.5, **kw):
    model = LineagePrior(tau_cell=tau_cell, **kw).fit(runs, ped, before_utc=before)
    assert isinstance(model, LineagePrior)
    return model


def late_races(runs: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Races at t and after: s:late's progeny win at t exactly and later in dirt long (a cell no
    one ran in before t) and in turf mile; s:one's progeny win turf sprints after t."""
    field = [h for h in pd.unique(runs["horse_id"]) if str(h).startswith("fh")][:5]
    one = runs.loc[runs["horse_id"].str.startswith("s:one/"), "horse_id"].iloc[0]
    rows = []
    plan = [
        ("jra:late0", t, "dirt", LONG, "s:late/h0"),
        ("jra:late1", t + pd.Timedelta(hours=1), "turf", MILE, "s:late/h1"),
        ("jra:late2", t + pd.Timedelta(days=2), "dirt", LONG, "s:late/h2"),
        ("jra:late3", t, "turf", SPRINT, one),
    ]
    for race, post, surface, band, winner in plan:
        for place, h in enumerate([winner, *field], start=1):
            rows.append((race, post, h, place, surface, band, "t:9"))
    return pd.DataFrame(rows, columns=list(runs.columns))


def reversed_from(runs: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """runs with every race at or after t finishing in reverse order."""
    out = runs.copy()
    late = out["race_utc"] >= t
    size = out.loc[late].groupby("race_id")["finish_pos"].transform("max")
    out.loc[late, "finish_pos"] = size + 1 - out.loc[late, "finish_pos"]
    return out


def test_sire_cell_effects_use_only_runs_before_t():
    """Fitted at t, the prior means over every probe and cell are the same whether the runs at t
    and later are dropped, kept, kept in reverse order, or joined by races at t exactly (a new
    sire's progeny winning in a new cell, s:one's progeny winning sprints); the counts are too. A
    sire whose progeny first ran at t has prior 0. A race a minute before t does count: its result
    moves s:one's turf-mile prior."""
    runs, ped, t = asof_design_data()
    cut = runs.loc[runs["race_utc"] < t]
    assert 0 < len(cut) < len(runs) and (runs["race_utc"] == t).any()
    base = fitted(cut, ped, t)
    want = priors(base)
    assert np.abs(want[0]).max() > 0.1, "s:one's prior should move off 0"
    future = pd.concat([reversed_from(runs, t), late_races(runs, t)], ignore_index=True)
    for name, frame in {"kept": runs, "reversed and joined": future}.items():
        got = priors(fitted(frame, ped, t))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9, err_msg=name)
        assert cells_of(sire_cells(frame, ped, before_utc=t)) == cells_of(
            sire_cells(cut, ped, before_utc=t)
        ), name
    np.testing.assert_allclose(want[2], 0.0, rtol=0, atol=1e-9)
    assert cells_of(sire_cells(future, ped, before_utc=t + pd.Timedelta(days=3))) != cells_of(
        sire_cells(future, ped, before_utc=t)
    )

    focal = cut.loc[cut["horse_id"].str.startswith("s:one/"), "horse_id"].iloc[-1]
    field = [h for h in pd.unique(cut["horse_id"]) if str(h).startswith("fh")][-5:]
    minute = t - pd.Timedelta(minutes=1)
    by_place = {}
    for name, order in {"wins": [focal, *field], "last": [*field, focal]}.items():
        extra = pd.DataFrame(
            [
                ("jra:minute", minute, h, place, "turf", MILE, "t:0")
                for place, h in enumerate(order, start=1)
            ],
            columns=list(runs.columns),
        )
        model = fitted(pd.concat([cut, extra], ignore_index=True), ped, t)
        by_place[name] = prior(model, "p:one", "turf", MILE)
    assert by_place["wins"] - by_place["last"] > 1e-4, by_place


# ---------------------------------------------------------------------------------------------
# The cell effect between the sire's overall effect and the data
# ---------------------------------------------------------------------------------------------

# s:one runs 800 races in each mile cell (planted 1.0 on turf, -0.2 on dirt: 1.2 apart) and 3
# turf sprints, all won (hand-set); s:two 800 in each mile cell (-0.6 and 0.4: -1.0 apart); no
# one runs dirt sprints or long races. The field's 40 sires are planted with mean 0.
SHRINK_SEED = 2
SHRINK_CELLS = (
    Cell("s:one", "turf", MILE, 800, 1.0),
    Cell("s:one", "dirt", MILE, 800, -0.2),
    Cell("s:one", "turf", SPRINT, 3, 0.4, wins=True),
    Cell("s:two", "turf", MILE, 800, -0.6),
    Cell("s:two", "dirt", MILE, 800, 0.4),
)
UNSEEN = [("dirt", SPRINT), ("turf", LONG), ("dirt", MIDDLE), ("jump", LONG)]


@pytest.fixture(scope="module")
def shrink_design():
    runs, ped, _ = lineage_runs(SHRINK_SEED, SHRINK_CELLS)
    rows = rows_of(ped)
    ped = pedigree(rows + [("p:one", "s:one", None), ("p:two", "s:two", None)])
    return runs, ped, runs["race_utc"].max() + DAY


def test_cell_effect_shrinks_toward_the_sire_with_few_progeny_and_toward_the_data_with_many(
    shrink_design,
):
    """Every cell s:one's progeny never ran in has one value, its overall effect g_sire, which
    lies between its two mile cells. With tau_cell = 1 the mile cells follow the 800-race data
    (turf less dirt within 0.35 of the planted 1.2 for s:one and -1.0 for s:two). With tau_cell =
    0.3 the 3-race sprint cell, all won, sits above g_sire by less than 0.35, and by more with
    tau_cell = 1. As tau_cell goes to 0 every cell is g_sire while the sires still differ by
    their data."""
    runs, ped, end = shrink_design
    by_tau = {tau: fitted(runs, ped, end, tau_cell=tau) for tau in (1.0, 0.3, 1e-3)}
    few = {}
    for tau, model in by_tau.items():
        for horse in ("p:one", "p:two"):
            unseen = [prior(model, horse, s, b) for s, b in UNSEEN]
            assert max(unseen) - min(unseen) < 1e-12, (tau, horse, unseen)
        g_one = prior(model, "p:one", *UNSEEN[0])
        turf, dirt = prior(model, "p:one", "turf", MILE), prior(model, "p:one", "dirt", MILE)
        if tau >= 0.3:
            assert min(turf, dirt) < g_one < max(turf, dirt), (tau, turf, g_one, dirt)
        few[tau] = prior(model, "p:one", "turf", SPRINT) - g_one

    wide = by_tau[1.0]
    gap_one = prior(wide, "p:one", "turf", MILE) - prior(wide, "p:one", "dirt", MILE)
    gap_two = prior(wide, "p:two", "turf", MILE) - prior(wide, "p:two", "dirt", MILE)
    assert gap_one == pytest.approx(1.2, abs=0.35)
    assert gap_two == pytest.approx(-1.0, abs=0.35)

    assert 0.0 < few[0.3] < 0.35, few
    assert few[1.0] > few[0.3] + 0.3, few

    tight = by_tau[1e-3]
    for horse in ("p:one", "p:two"):
        g = prior(tight, horse, *UNSEEN[0])
        for surface, band in [("turf", MILE), ("dirt", MILE), ("turf", SPRINT)]:
            assert abs(prior(tight, horse, surface, band) - g) < 0.01, (horse, surface, band)
    assert prior(tight, "p:one", *UNSEEN[0]) - prior(tight, "p:two", *UNSEEN[0]) > 0.25


# ---------------------------------------------------------------------------------------------
# Outline 1: the damsire at a quarter weight
# ---------------------------------------------------------------------------------------------

# s:hi's progeny are strong on turf and weak on dirt, s:lo's the reverse (500 races a cell). a:hi
# and a:lo share the sire f:00 and differ in the damsire (their dams m:hi and m:lo are s:hi's and
# s:lo's daughters); p:hi and p:lo are s:hi's and s:lo's own foals; n:hi has no known sire.
DAMSIRE_CELLS = (
    Cell("s:hi", "turf", MILE, 500, 0.6),
    Cell("s:hi", "dirt", MILE, 500, -0.4),
    Cell("s:lo", "turf", MILE, 500, -0.4),
    Cell("s:lo", "dirt", MILE, 500, 0.4),
)
DAMSIRE_PROBES = [
    ("m:hi", "s:hi", None),
    ("m:lo", "s:lo", None),
    ("a:hi", "f:00", "m:hi"),
    ("a:lo", "f:00", "m:lo"),
    ("p:hi", "s:hi", None),
    ("p:lo", "s:lo", None),
    ("n:hi", None, "m:hi"),
]


def test_damsire_enters_at_a_quarter_weight():
    """Two foals of one sire whose damsires are s:hi and s:lo differ by damsire_weight times what
    a foal of s:hi and one of s:lo differ by, in every cell (turf, dirt, unseen), for the default
    weight (1/4), 1/2 and 0; a foal with no known sire carries damsire_weight times its damsire's
    foal's prior. The data put s:hi's foals clearly above s:lo's on turf and below on dirt."""
    runs, ped, _ = lineage_runs(9541, DAMSIRE_CELLS)
    rows = rows_of(ped)
    ped = pedigree(rows + DAMSIRE_PROBES)
    assert ped.set_index("horse_id").loc["a:hi", "dam_sire_id"] == "s:hi"
    end = runs["race_utc"].max() + DAY
    cells = [("turf", MILE), ("dirt", MILE), ("turf", LONG)]
    for weight in (None, 0.5, 0.0):
        kw = {} if weight is None else {"damsire_weight": weight}
        w = 0.25 if weight is None else weight
        model = fitted(runs, ped, end, **kw)
        for surface, band in cells:
            p_hi, p_lo = prior(model, "p:hi", surface, band), prior(model, "p:lo", surface, band)
            a_gap = prior(model, "a:hi", surface, band) - prior(model, "a:lo", surface, band)
            assert a_gap == pytest.approx(w * (p_hi - p_lo), abs=1e-9), (w, surface, band)
            assert prior(model, "n:hi", surface, band) == pytest.approx(w * p_hi, abs=1e-9)
        turf_gap = prior(model, "p:hi", "turf", MILE) - prior(model, "p:lo", "turf", MILE)
        dirt_gap = prior(model, "p:hi", "dirt", MILE) - prior(model, "p:lo", "dirt", MILE)
        assert turf_gap > 0.4 and dirt_gap < -0.3, (w, turf_gap, dirt_gap)


# ---------------------------------------------------------------------------------------------
# Outline 1: the debut prior mean of a_h is the trainer effect plus lineage
# ---------------------------------------------------------------------------------------------

TRAINER_EFFECTS = {"t:0": 0.35, "t:1": -0.2, "t:5": 0.05, "t:debut-only": -0.6}


def test_debut_prior_mean_is_trainer_plus_lineage():
    """For debut probes (in the pedigree, no runs) of two sires and of the field, a horse without
    a pedigree row, and every cell: the prior mean with a trainer is the lineage-only prior mean
    plus that trainer's effect, exactly; an unknown trainer, no trainer, or a fit without
    trainer effects adds 0. The lineage part is not constant: s:one's turf-mile prior is well
    above its dirt-mile prior (planted 1.2 apart), and a horse with no row has lineage 0."""
    runs, ped, t = asof_design_data()
    cut = runs.loc[runs["race_utc"] < t]
    model = LineagePrior(tau_cell=0.5).fit(cut, ped, before_utc=t, trainer_effects=TRAINER_EFFECTS)
    assert isinstance(model, LineagePrior)
    plain = fitted(cut, ped, t)
    horses = ["p:one", "p:two", "p:field", "x:no-pedigree"]
    for h in horses:
        for surface, band in GRID:
            lineage = prior(model, h, surface, band)
            assert prior(model, h, surface, band, trainer_id=None) == lineage
            for trainer, effect in TRAINER_EFFECTS.items():
                got = prior(model, h, surface, band, trainer_id=trainer)
                assert got == pytest.approx(lineage + effect, abs=1e-12), (h, trainer)
            assert prior(model, h, surface, band, trainer_id="t:unknown") == pytest.approx(
                lineage, abs=1e-12
            )
            bare = prior(plain, h, surface, band)
            assert prior(plain, h, surface, band, trainer_id="t:0") == pytest.approx(
                bare, abs=1e-12
            )
    assert prior(model, "x:no-pedigree", "turf", MILE) == pytest.approx(0.0, abs=1e-12)
    assert prior(model, "x:no-pedigree", "turf", MILE, trainer_id="t:0") == pytest.approx(
        0.35, abs=1e-12
    )
    gap = prior(model, "p:one", "turf", MILE) - prior(model, "p:one", "dirt", MILE)
    assert gap > 0.4, gap


def test_unknown_lineage_falls_back_to_zero():
    """A horse with no pedigree row, one whose sire and dam are unknown, one whose sire has no
    progeny run before t, and one whose sire is unknown and whose dam has no known sire all have
    lineage 0 in every cell, including surfaces and bands never seen; a known sire's foal with an
    unknown dam has its sire's cell effect only, so it matches a full-pedigree foal of that sire
    whose dam's sire is unknown."""
    runs, ped, t = asof_design_data()
    cut = runs.loc[runs["race_utc"] < t]
    rows = rows_of(ped)
    extra = [
        ("x:orphan", None, None),
        ("x:no-damsire", None, "x:dam-without-row"),
        ("x:one-full", "s:one", "x:dam-of-unknown-sire"),
        ("x:dam-of-unknown-sire", None, None),
    ]
    ped = pedigree(rows + extra)
    model = fitted(cut, ped, t)
    cells = [*GRID, ("jump", MIDDLE), ("turf", "a-band-never-seen")]
    for horse in ("x:no-pedigree", "x:orphan", "p:late", "x:no-damsire"):
        for surface, band in cells:
            assert prior(model, horse, surface, band) == pytest.approx(0.0, abs=1e-9), horse
    for surface, band in cells:
        assert prior(model, "x:one-full", surface, band) == pytest.approx(
            prior(model, "p:one", surface, band), abs=1e-12
        )
    assert abs(prior(model, "p:one", "turf", MILE)) > 0.1
