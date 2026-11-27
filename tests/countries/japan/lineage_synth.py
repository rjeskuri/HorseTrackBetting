"""Synthetic pedigrees and runs for test_lineage.py (#95, epic #73).

Nothing here calls the code under test (`models.lineage`, `studies.lineage`). The references are
the recursive (tabular) method for the additive relationship (`reference_relationship`, Wright's
coefficients by the parents' recursion), hand counts, and planted Plackett-Luce utilities:
finishing orders are drawn by Plackett-Luce on the planted utility (ability_synth's Gumbel
sampler). Every RNG is seeded. The designs' sizes were set so that a reference estimator (a
penalised top-1 and top-3 Plackett-Luce MAP with sire effects, sire x cell deviations at
tau_cell and the damsire at its weight) meets the tolerances by a wide margin over many seeds;
that reference is not used by any test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ability_synth import move_to_place, pl_positions

FIRST_POST = pd.Timestamp("2024-01-06 01:00", tz="UTC")
RACES_PER_DAY = 24
SURFACES = ("turf", "dirt")
# Distance bands are the caller's opaque labels: the module under test never parses them.
SPRINT, MILE, MIDDLE, LONG = "sprint", "mile", "middle", "long"


def race_times(codes) -> pd.DatetimeIndex:
    """One tz-aware UTC post time per row: race c runs on day c // RACES_PER_DAY at slot
    c % RACES_PER_DAY, 30 minutes apart, so race codes are in time order."""
    codes = np.asarray(codes)
    days = pd.to_timedelta(codes // RACES_PER_DAY, unit="D")
    slots = pd.to_timedelta((codes % RACES_PER_DAY) * 30, unit="min")
    return FIRST_POST + days + slots


# ---------------------------------------------------------------------------------------------
# Pedigrees and the additive relationship by first principles
# ---------------------------------------------------------------------------------------------


def pedigree(rows) -> pd.DataFrame:
    """horse_id, sire_id, dam_id from (horse, sire, dam) tuples (None: unknown), plus
    dam_sire_id, the sire of the dam's own row (None when the dam has no row or no sire), so a
    reader that takes the damsire from the dam's row and one that reads dam_sire_id agree."""
    frame = pd.DataFrame(list(rows), columns=["horse_id", "sire_id", "dam_id"], dtype=object)
    sire_of = dict(zip(frame["horse_id"], frame["sire_id"], strict=True))
    frame["dam_sire_id"] = [sire_of.get(d) if d is not None else None for d in frame["dam_id"]]
    return frame.astype(object).where(frame.notna(), None)


def _parents(ped: pd.DataFrame) -> dict[str, tuple[str | None, str | None]]:
    out = {}
    for h, s, d in ped[["horse_id", "sire_id", "dam_id"]].itertuples(index=False):
        out[h] = (None if pd.isna(s) else s, None if pd.isna(d) else d)
    return out


def reference_relationship(ped: pd.DataFrame, horses) -> np.ndarray:
    """The additive relationship A among `horses` by the recursive method: A(i, i) = 1 +
    A(sire, dam) / 2 (0 when a parent is unknown); for i not an ancestor of j, A(i, j) =
    (A(sire_i, j) + A(dam_i, j)) / 2, an unknown parent contributing 0. A horse with no row is
    a founder. The younger of a pair is the one deeper in the pedigree (an ancestor is always
    shallower)."""
    parents = _parents(ped)
    depth: dict[str, int] = {}

    def deep(h: str) -> int:
        if h not in depth:
            s, d = parents.get(h, (None, None))
            depth[h] = 1 + max([deep(p) for p in (s, d) if p is not None], default=-1)
        return depth[h]

    memo: dict[tuple[str, str], float] = {}

    def a(i: str | None, j: str | None) -> float:
        if i is None or j is None:
            return 0.0
        key = (i, j) if i <= j else (j, i)
        if key in memo:
            return memo[key]
        if i == j:
            s, d = parents.get(i, (None, None))
            value = 1.0 + 0.5 * a(s, d)
        else:
            young, old = (i, j) if deep(i) >= deep(j) else (j, i)
            s, d = parents.get(young, (None, None))
            value = 0.5 * (a(s, old) + a(d, old))
        memo[key] = value
        return value

    horses = list(horses)
    return np.array([[a(i, j) for j in horses] for i in horses])


def random_pedigree(seed: int, *, generations: int = 6, per_generation: int = 9) -> pd.DataFrame:
    """A pedigree over `generations` generations of `per_generation` horses (about half of each
    generation male). A horse's sire is a male and its dam a female of any earlier generation,
    so inbreeding arises (half-sib, full-sib and parent-offspring matings); about one parent in
    eight is unknown. The first generation's founders have no row; rows are shuffled, so they
    are not in birth order."""
    rng = np.random.default_rng(seed)
    males = [f"g0m{i}" for i in range(per_generation // 2)]
    females = [f"g0f{i}" for i in range(per_generation - per_generation // 2)]
    rows = []
    for g in range(1, generations):
        new_m, new_f = [], []
        for i in range(per_generation):
            sire = None if rng.random() < 0.12 else males[rng.integers(len(males))]
            dam = None if rng.random() < 0.12 else females[rng.integers(len(females))]
            male = i % 2 == 0
            name = f"g{g}{'m' if male else 'f'}{i}"
            rows.append((name, sire, dam))
            (new_m if male else new_f).append(name)
        males, females = males + new_m, females + new_f
    order = rng.permutation(len(rows))
    return pedigree([rows[i] for i in order])


def ancestors(ped: pd.DataFrame, horses) -> set[str]:
    """`horses` and every ancestor of theirs named in ped."""
    parents = _parents(ped)
    seen: set[str] = set()
    stack = list(horses)
    while stack:
        h = stack.pop()
        if h in seen:
            continue
        seen.add(h)
        stack.extend(p for p in parents.get(h, (None, None)) if p is not None)
    return seen


# ---------------------------------------------------------------------------------------------
# Runs of planted sire x surface x distance-band cells
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """`races` races at (surface, band), each with one progeny of `sire` at the planted utility
    `utility` against a field of the field sires' progeny; `wins` hand-sets every one of those
    progeny runs to win."""

    sire: str
    surface: str
    band: str
    races: int
    utility: float
    wins: bool = False


FIELD_SIRES = 40
FIELD_SD = 0.4
RUNNERS = 6
TRAINERS = 6


def lineage_runs(seed: int, cells, *, progeny_runs: int = 4, prefix: str = "jra:l"):
    """(runs, pedigree, field) for the planted cells. runs: race_id, race_utc, horse_id,
    finish_pos, surface, dist_band, trainer_id (no trainer effect is planted), one race per
    cell race in a seeded shuffle of time order, RUNNERS runners each. A cell's progeny run
    about `progeny_runs` times each. The field: FIELD_SIRES sires (f:00 ..) with planted effects
    centred at 0 (sd FIELD_SD) and no cell deviation; each field horse is a field sire's. Every
    horse's dam is unknown (no dam_id). `field` maps field sire to its planted effect."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0.0, FIELD_SD, size=FIELD_SIRES)
    effect = effect - effect.mean()
    field = {f"f:{i:02d}": float(effect[i]) for i in range(FIELD_SIRES)}
    plan = [c for c in cells for _ in range(c.races)]
    order = rng.permutation(len(plan))
    n_field_horses = max(50, len(plan) * (RUNNERS - 1) // 6)
    field_sire = rng.integers(0, FIELD_SIRES, size=n_field_horses)
    progeny: dict[str, list[str]] = {}
    for c in cells:
        n = sum(x.races for x in cells if x.sire == c.sire)
        if c.sire not in progeny:
            progeny[c.sire] = [f"{c.sire}/h{i:03d}" for i in range(max(1, -(-n // progeny_runs)))]

    rows = []
    for code, i in enumerate(order):
        c = plan[i]
        focal = progeny[c.sire][rng.integers(len(progeny[c.sire]))]
        others = rng.choice(n_field_horses, RUNNERS - 1, replace=False)
        horses = [focal] + [f"fh{j:05d}" for j in others]
        u = np.r_[c.utility, effect[field_sire[others]]]
        codes = np.zeros(RUNNERS, dtype=np.int64)
        pos = pl_positions(rng, u, codes)
        if c.wins:
            move_to_place(pos, codes, 0, 1)
        trainers = rng.integers(0, TRAINERS, size=RUNNERS)
        for h, p, t in zip(horses, pos, trainers, strict=True):
            rows.append((code, h, int(p), c.surface, c.band, f"t:{t}"))
    frame = pd.DataFrame(
        rows, columns=["code", "horse_id", "finish_pos", "surface", "dist_band", "trainer_id"]
    )
    runs = pd.DataFrame(
        {
            "race_id": [f"{prefix}{c:05d}" for c in frame["code"]],
            "race_utc": race_times(frame["code"].to_numpy()),
            "horse_id": frame["horse_id"].astype(object),
            "finish_pos": frame["finish_pos"].astype(np.int64),
            "surface": frame["surface"].astype(object),
            "dist_band": frame["dist_band"].astype(object),
            "trainer_id": frame["trainer_id"].astype(object),
        }
    )
    ped_rows = [(f"fh{j:05d}", f"f:{field_sire[j]:02d}", None) for j in range(n_field_horses)]
    ped_rows += [(h, s, None) for s, hs in progeny.items() for h in hs]
    return runs, pedigree(ped_rows), field
