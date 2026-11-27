"""C2 lineage (#95, epic #73; 08 §8 B5, §4 E37; 05 §4.1-4.3): the pedigree relationship matrix
and a sire x surface x distance-band prior on the utility scale of #92's Plackett-Luce model.

Pedigree. A frame with horse_id, sire_id and dam_id (other columns ignored, one row per horse,
rows in any order); a missing parent (None or NA) is unknown. A horse without a row, or named
only as a parent, is a founder. The damsire of a horse is its row's dam_sire_id when the frame
has that column and it is given, else the sire of the dam's own row.

`relationship_matrix` is Wright's additive relationship A by the tabular method over the
horses' ancestors only, parents before offspring: A(i, i) = 1 + A(sire_i, dam_i) / 2 and
A(i, j) = (A(sire_i, j) + A(dam_i, j)) / 2 for j earlier than i, an unknown parent counting 0.
No row of a descendant or of an unrelated horse is read, so none can change an entry. It is
dense over the ancestors (memory quadratic in their number); Henderson's sparse inverse is not
built: the prior below needs only the sire and the damsire.

`sire_cells` counts, per (sire_id, surface, dist_band), the runs of the sire's progeny strictly
before t and the distinct progeny among them. Surfaces and bands are the caller's opaque labels
(the bands' cut points belong to the study's frozen spec); a run with either missing is in no
cell, and a run of a horse without a known sire is not counted.

`LineagePrior` fits, on the runs strictly before t, the discounted top-k Plackett-Luce model of
#92 (`models.ability`: the same stages, dead heats and missing places) with

    u = g(sire, surface, band) + damsire_weight * g(damsire, surface, band),
    g(s, c) = g_sire[s] + d[s, c],  g_sire ~ N(0, tau_sire^2),  d[s, c] ~ N(0, tau_cell^2),

so g(s, c) ~ N(g_sire[s], tau_cell^2): a cell with few progeny runs stays near g_sire, one with
many follows its data. One table serves both roles (a damsire's term is its own effect as a
sire). The parameters are the stallions with a progeny run before t and their `sire_cells`
cells only: a stallion seen only as a damsire, or a cell its progeny never ran in, has no
parameter. tau_cell is the caller's; tau_sire^2 is set on the same runs by empirical Bayes
(MacKay's fixed point, `models.ability._empirical_bayes`) with the cells' prior fixed. The fit
is the penalised MAP by Newton's method. There is no per-horse effect: a run is its sire's and
damsire's evidence only, and the per-horse ability stays the ability model's. Races in which no
runner has a lineage term are dropped (their likelihood does not depend on the effects); the
rest are sorted by post time, race and horse, so a fit does not depend on the input's row
order.

`prior_mean` is the debut prior mean of a horse's ability a_h in a cell: the trainer's effect
(from `trainer_effects`; 0 without a trainer, for one the mapping lacks, or without the mapping)
plus g(sire, cell) + damsire_weight * g(damsire, cell), where g of a cell without a parameter is
the stallion's g_sire and g of a stallion without one (no progeny run before t, or unknown) is
0. `prior_means` is the same over a frame's rows, and `cell_effects_` the fitted table per
stallion x surface x band: the prior #93's structured aptitudes (b_h, e_h) can start from.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp

from horsetrack.countries.japan.models.ability import (
    _discounts,
    _empirical_bayes,
    _problem,
    _race_codes,
    _stages,
)

PEDIGREE_COLUMNS = ("horse_id", "sire_id", "dam_id")
RUN_COLUMNS = ("race_id", "race_utc", "horse_id", "finish_pos", "surface", "dist_band")
CELL_COLUMNS = ["sire_id", "surface", "dist_band", "n_runs", "n_horses"]
CELL_KEYS = ["sire_id", "surface", "dist_band"]

__all__ = ["CELL_COLUMNS", "LineagePrior", "relationship_matrix", "sire_cells"]


# --------------------------------------------------------------------------- pedigree


def _value(v) -> Hashable | None:
    return None if v is None or pd.isna(v) else v


def _rows(pedigree: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PEDIGREE_COLUMNS if c not in pedigree.columns]
    if missing:
        raise ValueError(f"pedigree lacks column(s) {missing}")
    rows = pedigree.loc[pedigree["horse_id"].notna()]
    if rows["horse_id"].duplicated().any():
        dup = rows.loc[rows["horse_id"].duplicated(), "horse_id"].iloc[0]
        raise ValueError(f"pedigree has more than one row for horse {dup!r}")
    return rows


def _parents(pedigree: pd.DataFrame) -> dict[Hashable, tuple[Hashable | None, Hashable | None]]:
    rows = _rows(pedigree)
    columns = [rows[c].to_numpy(dtype=object) for c in PEDIGREE_COLUMNS]
    return {h: (_value(s), _value(d)) for h, s, d in zip(*columns, strict=True)}


def _birth_order(parents: dict, horses: Sequence) -> list:
    """`horses` and their ancestors, every parent before its offspring (iterative DFS)."""
    order, state = [], {}  # state: 1 on the DFS path, 2 placed
    for root in horses:
        if state.get(root) == 2:
            continue
        stack = [(root, False)]
        while stack:
            h, done = stack.pop()
            if done:
                state[h] = 2
                order.append(h)
                continue
            if state.get(h) == 2:
                continue
            if state.get(h) == 1:
                raise ValueError(f"the pedigree has a cycle through {h!r}")
            state[h] = 1
            stack.append((h, True))
            for p in parents.get(h, (None, None)):
                if p is not None and state.get(p) != 2:
                    if state.get(p) == 1:
                        raise ValueError(f"the pedigree has a cycle through {p!r}")
                    stack.append((p, False))
    return order


def relationship_matrix(pedigree: pd.DataFrame, horses: Sequence[str]) -> pd.DataFrame:
    """The additive relationship A among `horses` (index and columns `horses`, in that order),
    from their ancestors' rows only; see the module docstring."""
    horses = list(horses)
    parents = _parents(pedigree)
    order = _birth_order(parents, horses)
    at = {h: i for i, h in enumerate(order)}
    a = np.zeros((len(order), len(order)))
    for k, h in enumerate(order):
        s, d = (at.get(p) if p is not None else None for p in parents.get(h, (None, None)))
        row = np.zeros(k)
        for p in (s, d):
            if p is not None:
                row += 0.5 * a[p, :k]
        a[k, :k] = row
        a[:k, k] = row
        a[k, k] = 1.0 + (0.5 * a[s, d] if s is not None and d is not None else 0.0)
    idx = [at[h] for h in horses]
    return pd.DataFrame(a[np.ix_(idx, idx)], index=horses, columns=horses, dtype=float)


def _lineage(pedigree: pd.DataFrame) -> tuple[dict, dict]:
    """(sire of, damsire of) per horse with a known one."""
    rows = _rows(pedigree)
    horse = rows["horse_id"].to_numpy(dtype=object)
    sire = [_value(v) for v in rows["sire_id"].to_numpy(dtype=object)]
    dam = [_value(v) for v in rows["dam_id"].to_numpy(dtype=object)]
    sire_of = {h: s for h, s in zip(horse, sire, strict=True) if s is not None}
    given = (
        [_value(v) for v in rows["dam_sire_id"].to_numpy(dtype=object)]
        if "dam_sire_id" in rows.columns
        else [None] * len(rows)
    )
    damsire_of = {}
    for h, d, ds in zip(horse, dam, given, strict=True):
        ds = ds if ds is not None else (sire_of.get(d) if d is not None else None)
        if ds is not None:
            damsire_of[h] = ds
    return sire_of, damsire_of


# --------------------------------------------------------------------------- runs before t


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        raise ValueError("before_utc must be timezone-aware (UTC)")
    return t


def _before(runs: pd.DataFrame, before_utc: datetime, columns=RUN_COLUMNS) -> pd.DataFrame:
    """The runs strictly before before_utc, `columns` only, sorted by post time, race and horse."""
    missing = [c for c in columns if c not in runs.columns]
    if missing:
        raise ValueError(f"runs lacks column(s) {missing}")
    times = pd.to_datetime(runs["race_utc"])
    if times.dt.tz is None:
        raise ValueError("race_utc must be timezone-aware (UTC)")
    out = runs.loc[(times < _utc(before_utc)).to_numpy(), list(columns)]
    return out.sort_values(["race_utc", "race_id", "horse_id"], kind="stable").reset_index(
        drop=True
    )


def _cells(read: pd.DataFrame, sire_of: dict) -> pd.DataFrame:
    sire = read["horse_id"].map(sire_of)
    known = read.assign(sire_id=sire).dropna(subset=CELL_KEYS)
    out = known.groupby(CELL_KEYS, sort=True, observed=True).agg(
        n_runs=("horse_id", "size"), n_horses=("horse_id", "nunique")
    )
    return out.reset_index()[CELL_COLUMNS].astype({"n_runs": np.int64, "n_horses": np.int64})


def sire_cells(runs: pd.DataFrame, pedigree: pd.DataFrame, *, before_utc: datetime) -> pd.DataFrame:
    """One row per (sire_id, surface, dist_band) with a progeny run strictly before before_utc:
    n_runs (those runs) and n_horses (the distinct progeny among them)."""
    columns = ("race_utc", "race_id", "horse_id", "surface", "dist_band")
    return _cells(_before(runs, before_utc, columns), _lineage(pedigree)[0])


# --------------------------------------------------------------------------- the prior


class LineagePrior:
    """Sire x surface x band effects with the damsire at a weight; see the module docstring."""

    def __init__(
        self,
        *,
        tau_cell: float,
        damsire_weight: float = 0.25,
        k: int = 3,
        discounts: Sequence[float] | None = None,
    ) -> None:
        tau_cell, damsire_weight = float(tau_cell), float(damsire_weight)
        if not (np.isfinite(tau_cell) and tau_cell > 0):
            raise ValueError(f"tau_cell must be positive and finite, got {tau_cell!r}")
        if not (np.isfinite(damsire_weight) and damsire_weight >= 0):
            raise ValueError(f"damsire_weight must be non-negative, got {damsire_weight!r}")
        self.tau_cell = tau_cell
        self.damsire_weight = damsire_weight
        self.k = int(k)
        self.discounts = _discounts(k, discounts)
        self.before_utc_: pd.Timestamp | None = None
        self.sire_effects_ = pd.Series(dtype=float, name="g_sire")
        self.cell_effects_ = pd.DataFrame(columns=[*CELL_COLUMNS, "deviation", "g"])
        self.sire_prior_var_: float | None = None
        self.loglik_: float | None = None
        self._sire_of: dict = {}
        self._damsire_of: dict = {}
        self._g_sire: dict = {}
        self._g_cell: dict = {}
        self._trainer: dict = {}

    # -- fit ------------------------------------------------------------------------------------

    def fit(
        self,
        runs: pd.DataFrame,
        pedigree: pd.DataFrame,
        *,
        before_utc: datetime,
        trainer_effects: Mapping[Hashable, float] | pd.Series | None = None,
    ) -> LineagePrior:
        """The penalised MAP on the runs strictly before before_utc. Returns self."""
        self.before_utc_ = _utc(before_utc)
        read = _before(runs, before_utc)
        self._sire_of, self._damsire_of = _lineage(pedigree)
        self._trainer = {} if trainer_effects is None else dict(trainer_effects.items())
        self._trainer = {t: float(v) for t, v in self._trainer.items()}
        cells = _cells(read, self._sire_of)
        stallions = pd.Index(pd.unique(cells["sire_id"].to_numpy(dtype=object)))
        cell_index = pd.MultiIndex.from_frame(cells[CELL_KEYS])
        n_s = len(stallions)
        deviation = np.zeros(len(cells))
        g_sire = np.zeros(n_s)
        self.sire_prior_var_, self.loglik_ = None, None
        if n_s:
            design, keep = self._design(read, stallions, cell_index)
            rows = read.loc[keep].reset_index(drop=True)
            design = design[np.flatnonzero(keep)]
            codes = _race_codes(rows["race_id"])
            pos = pd.to_numeric(rows["finish_pos"]).to_numpy(dtype=float)
            st = _stages(codes, pos, self.discounts)
            if len(st.rows):
                problem = _problem(design, codes, pos, st)
                base = np.r_[np.zeros(n_s), np.full(len(cells), 1.0 / self.tau_cell**2)]
                theta, loglik, _, tau2 = _empirical_bayes(problem, [np.arange(n_s)], base)
                g_sire, deviation = theta[:n_s], theta[n_s:]
                self.sire_prior_var_, self.loglik_ = float(tau2[0]), float(loglik)
        self.sire_effects_ = pd.Series(g_sire, index=stallions, dtype=float, name="g_sire")
        g_cell = g_sire[stallions.get_indexer(cells["sire_id"].to_numpy(dtype=object))] + deviation
        self.cell_effects_ = cells.assign(deviation=deviation, g=g_cell)
        self._g_sire = dict(zip(stallions, g_sire.tolist(), strict=True))
        self._g_cell = dict(zip(cell_index, g_cell.tolist(), strict=True))
        return self

    def _design(self, read: pd.DataFrame, stallions: pd.Index, cell_index: pd.MultiIndex):
        """The sparse design over [g_sire per stallion, d per cell] and the rows it touches."""
        n, n_s = len(read), len(stallions)
        horse = read["horse_id"]
        surface, band = (
            read["surface"].to_numpy(dtype=object),
            read["dist_band"].to_numpy(dtype=object),
        )
        rows, cols, vals = [], [], []
        roles = [(horse.map(self._sire_of), 1.0)]
        if self.damsire_weight > 0:
            roles.append((horse.map(self._damsire_of), self.damsire_weight))
        for who, weight in roles:
            who = who.to_numpy(dtype=object)
            s = stallions.get_indexer(who)
            has = np.flatnonzero(s >= 0)
            rows.append(has)
            cols.append(s[has])
            c = cell_index.get_indexer(pd.MultiIndex.from_arrays([who, surface, band]))
            in_cell = np.flatnonzero(c >= 0)
            rows.append(in_cell)
            cols.append(n_s + c[in_cell])
            vals.append(np.full(len(has) + len(in_cell), weight))
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        design = sp.csr_matrix((vals, (rows, cols)), shape=(n, n_s + len(cell_index)))
        touched = np.bincount(rows, minlength=n) > 0
        races = read["race_id"].to_numpy(dtype=object)
        keep = pd.Series(touched).groupby(races).transform("any").to_numpy(dtype=bool)
        return design, keep

    # -- the prior mean -------------------------------------------------------------------------

    def _g(self, stallion, surface, band) -> float:
        if stallion is None or stallion not in self._g_sire:
            return 0.0
        return self._g_cell.get((stallion, surface, band), self._g_sire[stallion])

    def prior_mean(self, horse_id, surface, dist_band, *, trainer_id=None) -> float:
        """The debut prior mean of a_h: trainer effect + g(sire, cell) + damsire_weight *
        g(damsire, cell); see the module docstring."""
        if self.before_utc_ is None:
            raise RuntimeError("fit the prior before reading it")
        horse_id, trainer_id = _value(horse_id), _value(trainer_id)
        lineage = self._g(self._sire_of.get(horse_id), surface, dist_band)
        lineage += self.damsire_weight * self._g(self._damsire_of.get(horse_id), surface, dist_band)
        trainer = self._trainer.get(trainer_id, 0.0) if trainer_id is not None else 0.0
        return float(trainer + lineage)

    def prior_means(self, frame: pd.DataFrame) -> pd.Series:
        """`prior_mean` per row of frame (horse_id, surface, dist_band and, when present,
        trainer_id), on frame's index."""
        trainers = (
            frame["trainer_id"].to_numpy(dtype=object)
            if "trainer_id" in frame.columns
            else [None] * len(frame)
        )
        values = [
            self.prior_mean(h, s, b, trainer_id=t)
            for h, s, b, t in zip(
                frame["horse_id"].to_numpy(dtype=object),
                frame["surface"].to_numpy(dtype=object),
                frame["dist_band"].to_numpy(dtype=object),
                trainers,
                strict=True,
            )
        ]
        return pd.Series(values, index=frame.index, dtype=float, name="prior_mean")
