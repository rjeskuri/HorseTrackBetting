"""C2 interactions on #92's ability model: jockey, trainer and track (#93, epic #73; 08 §3.3,
§4 E35, E36, §8 B5; 05 §4.1-4.3).

Cells. `cell_sizes` counts, per runner, the runs of its cell (equal values in every column of
the block) in races with a strictly earlier race_utc: a race at the same post time (a tie) and
the race itself never count. `n_half(tau, I)` = 1 / (I tau^2), the runs a cell needs before its
estimate is half data, half prior. A row with a missing key is in no cell (n_prior 0).

Blocks, added to #92's `AbilityModel` (its k, discounts, features and entity columns; the same
discounted top-k Plackett-Luce likelihood, `models.ability._stages`):
- 'variance_component' (cols: the cell's keys): one N(0, tau^2) effect per cell seen in training;
  horse x jockey is E35's negative control, trainer x venue E36.
- 'structured' (cols: horse, distance in metres, surface): the horse's a_h + b_h z + c_h z^2 +
  e_h 1{surface == 'dirt'}, z = (d - mean) / sd over the training rows; each of the four
  coefficients has its own N(0, tau_k^2) prior. The block's tau is the prior sd of its term
  averaged over the training rows, sqrt(mean_i sum_k tau_k^2 x_ik^2).
- 'low_rank' (cols: row entity, column entity): rank one, U_r W_c, with W centred over the
  columns seen in training and scaled to unit root mean square, so a row's term is a contrast
  between columns; U_r ~ N(0, tau^2) and tau is the block's tau. A column a row never ran at is
  predicted from the factor (E36's one-factor east/west term).
An unseen level, cell or column, or a missing id, has effect 0 (its prior mean).

Fit. The penalised MAP with empirical-Bayes prior variances, as `AbilityModel`: each prior group
(a base entity, a block, a structured coefficient) gets MacKay's evidence fixed point on ln tau^2,
secant-accelerated and floored at TAU2_FLOOR, `models.ability`'s rule. Parameters of the base and
of every block with at most DENSE_MAX parameters are solved jointly by Newton's method on
`models.ability._evaluate`'s information (exact). A larger block (horse x jockey or a per-horse
aptitude on the full store) is "local": its levels take block-Jacobi Newton steps (each level's
own curvature, cross-level terms within a race dropped), alternating with the joint step until no
step exceeds MAP_TOL, so the MAP itself is exact; only its posterior variances (the fixed point's
trace and the evidence's log-determinant) are approximate. With a local term they are those of
the information with every runner's own curvature h only (the coupling between the runners of a
race dropped, as in the local steps), where the local levels stay coupled to the dense block:
the dense parameters' covariance is the inverse of their Schur complement, A - B C^-1 B', and a
local level's variance adds the dense block's uncertainty carried through its column of B (so a
horse's level and the horse's own cells, which are confounded, share their variance as in the
dense fit, not each take all of it). Sparse: C is block-diagonal over the levels and only B's few
entries per level are used. Two local terms are not coupled to each other (each is, to the dense
block), and the dropped within-race coupling makes a variance a few percent small against the
dense fit (3% on the test fixtures; more where a prior sd is near its floor). A fit with no local
term uses the exact information. A low-rank
block's W takes its own Newton step given U (the objective is concave in W after the rescaling
that keeps W at unit scale), then U W is renormalised. Every step is halved until the penalised
objective does not worsen.

tau's interval (`InteractionFit.tau`): the profile of the Laplace evidence over the block's prior
scale (all its groups scaled together; the other groups held at their empirical-Bayes values and
each low-rank W at its fitted direction, a fixed design in the evidence),
each bound where the evidence falls TAU_LEVEL's one-sided chi-square(1) cutoff below its value at
the point estimate (D7's convention: each bound one-sided at 95%); a lower bound reaching
tau = 0 (the block removed) is 0. It is computed on first use, so a fit used only to predict
costs no profile. `Interval.n_clusters` holds the block's number of levels.

`drop_aliased` keeps the blocks the frame can identify: a block is aliased when its cells (the
distinct key combinations of its columns) are the levels of one of its columns, i.e. every level
of that column has one cell (horse x trainer where every horse keeps one trainer: 08 §3.3). A
block of one column is always aliased (it is that column's main effect). `transfer_step` marks a
run after a stable change of the horse's trainer (see its docstring).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from scipy.stats import norm

from horsetrack.core.eval.bootstrap import Interval
from horsetrack.countries.japan.models.ability import (
    EB_JUMP,
    EB_MAX,
    EB_TOL,
    HALVINGS,
    TAU2_FLOOR,
    TAU2_START,
    AbilityModel,
    _evaluate,
    _features,
    _inverse,
    _loglik,
    _problem,
    _race_codes,
    _softmax,
    _solve,
    _stages,
)

VARIANCE_COMPONENT, STRUCTURED, LOW_RANK = "variance_component", "structured", "low_rank"
KINDS = (VARIANCE_COMPONENT, STRUCTURED, LOW_RANK)
ENTITY = "entity"  # a base entity column of the AbilityModel
DENSE_MAX = 5_000  # a block with more parameters is fitted locally (block-Jacobi)
MAP_TOL = 1e-8  # the largest step on any parameter when a MAP solve stops
PROFILE_TOL = 1e-6  # the same, for the MAP at a profile point
MAX_SWEEPS = 1_000  # a solve that needs more fails (and its variant is recorded failed)
EB_SWITCH = 1e-3  # an alternating fit solves its MAP fully at each update once the residual is here
EB_LOCAL_TOL = 1e-4  # EB_TOL for a fit with local terms, whose alternating MAP resolves no finer
TAU_LEVEL = 0.95  # each bound of tau one-sided at this level (D7's convention)
TAU_ZERO = 1e-5  # the prior sd at which a block counts as removed (tau = 0)
PROFILE_BISECTIONS = 7  # per bound: tau to 1/128 of the point below it, to 1.1% above
PROFILE_GROWTH = 4.0  # the upper search's factor per step
PROFILE_DOUBLINGS = 8
QUAD_CHUNK = 2_000_000  # entries of the dense work matrix when a local term's variances are formed
OUT_COLUMNS = ["race_id", "horse_id", "u", "p_ability"]

__all__ = [
    "KINDS",
    "InteractionBlock",
    "InteractionFit",
    "cell_sizes",
    "drop_aliased",
    "fit_with_blocks",
    "n_half",
    "transfer_step",
]


@dataclass(frozen=True)
class InteractionBlock:
    """An interaction block: its name, the columns it reads and its kind (one of KINDS)."""

    name: str
    cols: tuple[str, ...]
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cols", tuple(self.cols))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a block needs a name, got {self.name!r}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not self.cols or not all(isinstance(c, str) for c in self.cols):
            raise ValueError(f"cols must be column names, got {self.cols!r}")
        need = {STRUCTURED: 3, LOW_RANK: 2}.get(self.kind)
        if need is not None and len(self.cols) != need:
            raise ValueError(f"a {self.kind} block reads {need} columns, got {self.cols!r}")


# --------------------------------------------------------------------------- cells


def _keys(frame: pd.DataFrame, cols: Sequence[str]) -> tuple[pd.Index, np.ndarray]:
    """The distinct key combinations of cols over the rows with no missing key, and each row's
    code into them (-1 for a row with a missing key)."""
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"frame lacks column(s) {missing}")
    keys = frame[list(cols)]
    ok = ~keys.isna().any(axis=1).to_numpy()
    codes = np.full(len(frame), -1, dtype=np.int64)
    if len(cols) == 1:
        values = pd.Index(keys.iloc[np.flatnonzero(ok), 0].to_numpy(dtype=object))
    else:
        values = pd.MultiIndex.from_frame(keys.iloc[np.flatnonzero(ok)])
    index = values.unique()
    codes[ok] = index.get_indexer(values)
    return index, codes


def _lookup(index: pd.Index, frame: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Each row's code into a fitted level index, -1 for an unseen level or a missing key."""
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"frame lacks column(s) {missing}")
    keys = frame[list(cols)]
    ok = ~keys.isna().any(axis=1).to_numpy()
    codes = np.full(len(frame), -1, dtype=np.int64)
    if ok.any():
        sub = keys.iloc[np.flatnonzero(ok)]
        if len(cols) == 1:
            values = pd.Index(sub.iloc[:, 0].to_numpy(dtype=object))
        else:
            values = pd.MultiIndex.from_frame(sub)
        codes[ok] = index.get_indexer(values)
    return codes


def _utc_ns(values: pd.Series) -> np.ndarray:
    """race_utc as int64 nanoseconds since the epoch."""
    when = pd.to_datetime(values, utc=True)
    if when.isna().any():
        raise ValueError("race_utc has missing values")
    return when.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def cell_sizes(frame: pd.DataFrame, *, block: tuple[str, ...]) -> pd.DataFrame:
    """Per row (frame's index): race_id, horse_id and n_prior, the rows of frame in the row's
    cell (equal values in every column of `block`) with a strictly earlier race_utc."""
    for col in ("race_id", "horse_id", "race_utc"):
        if col not in frame.columns:
            raise ValueError(f"frame lacks {col!r}")
    _, codes = _keys(frame, tuple(block))
    t = _utc_ns(frame["race_utc"])
    n_prior = np.zeros(len(frame), dtype=np.int64)
    ok = codes >= 0
    if ok.any():  # rank 'min': 1 + the rows of the cell strictly earlier
        ranks = pd.Series(t[ok]).groupby(codes[ok]).rank(method="min").to_numpy()
        n_prior[ok] = ranks.astype(np.int64) - 1
    return pd.DataFrame(
        {
            "race_id": frame["race_id"].to_numpy(),
            "horse_id": frame["horse_id"].to_numpy(),
            "n_prior": n_prior,
        },
        index=frame.index,
    )


def n_half(tau: float, info_per_run: float) -> float:
    """1 / (info_per_run * tau^2): the runs at which a cell's estimate is half data, half prior."""
    tau, info = float(tau), float(info_per_run)
    if not (tau > 0 and info > 0 and math.isfinite(tau) and math.isfinite(info)):
        raise ValueError(f"tau and info_per_run must be positive, got {tau}, {info}")
    return 1.0 / (info * tau * tau)


def _aliased(block: InteractionBlock, frame: pd.DataFrame) -> bool:
    _, cells = _keys(frame, block.cols)
    for col in block.cols:
        _, level = _keys(frame, (col,))
        ok = level >= 0
        if not ok.any():
            continue
        per_level = pd.Series(cells[ok]).groupby(level[ok]).nunique()
        if int(per_level.max()) == 1:
            return True
    return False


def drop_aliased(blocks: Sequence[InteractionBlock], frame: pd.DataFrame) -> list[InteractionBlock]:
    """The blocks frame can identify, in their order: a block is aliased when every level of one
    of its columns has a single cell (a row with a missing key counts as no cell)."""
    return [b for b in blocks if not _aliased(b, frame)]


def transfer_step(frame: pd.DataFrame, *, stable_runs: int) -> pd.Series:
    """True on a run after a stable change of the horse's trainer, one bool per row of frame
    (race_id, race_utc, horse_id, trainer_id).

    A horse's runs are taken in race_utc order. Its debut trainer is its established trainer. A
    run with another trainer starts (or continues) a spell; when that trainer has had the horse
    for `stable_runs` consecutive runs, that run included, it becomes the established trainer
    and the step is on, on that run and every later run with it. A run with the established
    trainer ends a spell, so a spell shorter than `stable_runs` (a one-off, then back) never
    turns the step on. A run with a missing trainer changes nothing and is off. Each value reads
    the horse's runs up to and including its own."""
    if isinstance(stable_runs, bool) or not isinstance(stable_runs, (int, np.integer)):
        raise ValueError(f"stable_runs must be a positive integer, got {stable_runs!r}")
    if stable_runs < 1:
        raise ValueError(f"stable_runs must be a positive integer, got {stable_runs!r}")
    for col in ("race_utc", "horse_id", "trainer_id"):
        if col not in frame.columns:
            raise ValueError(f"frame lacks {col!r}")
    horse = pd.factorize(frame["horse_id"].to_numpy(dtype=object))[0]
    trainer = pd.factorize(frame["trainer_id"].to_numpy(dtype=object))[0]
    order = np.lexsort((_utc_ns(frame["race_utc"]), horse))
    out = np.zeros(len(frame), dtype=bool)
    last_horse, settled, moved, spell, runs = -2, -1, False, -1, 0
    for i in order:
        h, t = horse[i], trainer[i]
        if h != last_horse:
            last_horse, settled, moved, spell, runs = h, -1, False, -1, 0
        if h < 0 or t < 0:
            continue
        if settled < 0:
            settled = t
        elif t == settled:
            spell, runs = -1, 0
        else:
            runs = runs + 1 if t == spell else 1
            spell = t
            if runs >= stable_runs:
                settled, moved, spell, runs = t, True, -1, 0
        out[i] = moved and t == settled
    return pd.Series(out, index=frame.index, name="transfer_step")


# --------------------------------------------------------------------------- the model's terms


@dataclass
class _Term:
    """u gains sum_k X[i, k] * theta[code_i, k] on the rows with a level (code >= 0): a base
    entity or a variance component (K = 1, X = 1), a structured block (K = 4) or a low-rank
    block's U (K = 1, X = W of the row's column)."""

    label: str
    kind: str
    key_cols: tuple[str, ...]
    index: pd.Index
    codes: np.ndarray
    X: np.ndarray
    groups: list[int] = dataclasses.field(default_factory=list)
    local: bool = False
    offset: int = -1  # its first column in the joint parameters (k-major), when not local
    theta: np.ndarray | None = None  # m x K, when local
    # structured
    dist_col: str = ""
    surface_col: str = ""
    z_mean: float = 0.0
    z_sd: float = 1.0
    # low rank
    col: str = ""
    col_index: pd.Index | None = None
    ccodes: np.ndarray | None = None
    W: np.ndarray | None = None
    started: bool = False  # W set from the data (a low-rank block's first sweep)

    @property
    def m(self) -> int:
        return len(self.index)

    @property
    def K(self) -> int:
        return self.X.shape[1]

    def covariates(self, frame: pd.DataFrame, ccodes: np.ndarray | None = None) -> np.ndarray:
        n = len(frame)
        if self.kind in (ENTITY, VARIANCE_COMPONENT):
            return np.ones((n, 1))
        if self.kind == STRUCTURED:
            d = pd.to_numeric(frame[self.dist_col], errors="coerce").to_numpy(dtype=float)
            z = np.nan_to_num((d - self.z_mean) / self.z_sd)
            dirt = frame[self.surface_col].to_numpy(dtype=object) == "dirt"
            return np.column_stack([np.ones(n), z, z * z, dirt.astype(float)])
        if ccodes is None:
            ccodes = _lookup(self.col_index, frame, (self.col,))
        return np.where(ccodes >= 0, self.W[np.maximum(ccodes, 0)], 0.0)[:, None]


@dataclass
class _Group:
    term: int
    k: int
    block: str | None


def _structured_term(label, block: InteractionBlock, frame: pd.DataFrame) -> _Term:
    horse, dist, surface = block.cols
    for col in (dist, surface):
        if col not in frame.columns:
            raise ValueError(f"frame lacks column {col!r}")
    index, codes = _keys(frame, (horse,))
    d = pd.to_numeric(frame[dist], errors="coerce").to_numpy(dtype=float)
    seen = d[np.isfinite(d)]
    if not len(seen):
        raise ValueError(f"{block.name}: no distance in {dist!r}")
    sd = float(seen.std())
    term = _Term(
        label, STRUCTURED, (horse,), index, codes, np.zeros((len(frame), 4)), dist_col=dist
    )
    term.surface_col, term.z_mean, term.z_sd = surface, float(seen.mean()), sd if sd > 0 else 1.0
    term.X = term.covariates(frame)
    return term


def _low_rank_term(label, block: InteractionBlock, frame: pd.DataFrame) -> _Term:
    row, col = block.cols
    index, codes = _keys(frame, (row,))
    col_index, ccodes = _keys(frame, (col,))
    if len(col_index) < 2:
        raise ValueError(f"{block.name}: a low-rank block needs at least two columns' levels")
    term = _Term(label, LOW_RANK, (row,), index, codes, np.zeros((len(frame), 1)), col=col)
    term.col_index, term.ccodes = col_index, ccodes
    term.W = _unit(np.arange(len(col_index), dtype=float))  # replaced before U first moves
    term.X = term.covariates(frame, ccodes)
    return term


def _unit(w: np.ndarray) -> np.ndarray:
    """w centred and scaled to unit root mean square."""
    w = np.asarray(w, dtype=float) - np.mean(w)
    norm_w = float(np.sqrt(np.mean(w * w)))
    if norm_w == 0:
        w = np.arange(len(w), dtype=float) - (len(w) - 1) / 2.0
        norm_w = float(np.sqrt(np.mean(w * w)))
    return w / norm_w


def _columns(term: _Term, codes: np.ndarray, cov: np.ndarray) -> sp.csr_matrix:
    """The term's design columns, k-major (column k * m + level): rows x (m * K), sparse."""
    ok = np.flatnonzero(codes >= 0)
    rows = np.tile(ok, term.K)
    cols = np.concatenate([k * term.m + codes[ok] for k in range(term.K)])
    data = np.concatenate([cov[ok, k] for k in range(term.K)])
    return sp.csr_matrix((data, (rows, cols)), shape=(len(codes), term.m * term.K))


def _quad_diag(G: sp.spmatrix, inv: np.ndarray) -> np.ndarray:
    """diag(G' inv G) for a sparse G (P x c) and a dense P x P inv, in column chunks so that no
    P x c dense matrix is formed."""
    G = G.tocsc()
    out = np.empty(G.shape[1])
    step = max(1, QUAD_CHUNK // max(1, G.shape[0]))
    for a in range(0, G.shape[1], step):
        part = G[:, a : a + step].T.tocsr()
        out[a : a + step] = np.asarray(part.multiply(part @ inv).sum(axis=1)).ravel()
    return out


def _centred_basis(m: int) -> np.ndarray:
    """An orthonormal basis of the vectors in R^m that sum to 0 (m x (m - 1))."""
    q, _ = np.linalg.qr(np.column_stack([np.ones(m), np.eye(m)[:, : m - 1]]))
    return q[:, 1:]


# --------------------------------------------------------------------------- the fit


class _Model:
    """The terms, the parameters and the solver of one fit."""

    def __init__(self, frame: pd.DataFrame, base: AbilityModel, blocks: list[InteractionBlock]):
        for col in ("race_id", "finish_pos"):
            if col not in frame.columns:
                raise ValueError(f"frame lacks {col!r}")
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"block names repeat: {names}")
        self.base, self.blocks = base, list(blocks)
        self.features = list(base.features)
        self.X = _features(frame, self.features)
        self.n = len(frame)
        codes = _race_codes(frame["race_id"])
        pos = pd.to_numeric(frame["finish_pos"]).to_numpy(dtype=float)
        self.st = _stages(codes, pos, base.discounts)
        if not len(self.st.rows):
            raise ValueError("no race has a placed runner to fit")

        self.terms: list[_Term] = []
        self.groups: list[_Group] = []
        for col in base.entity_cols:
            index, cc = _keys(frame, (col,))
            self._add(_Term(f"entity:{col}", ENTITY, (col,), index, cc, np.ones((self.n, 1))))
        for block in blocks:
            if block.kind == VARIANCE_COMPONENT:
                index, cc = _keys(frame, block.cols)
                term = _Term(
                    block.name, VARIANCE_COMPONENT, block.cols, index, cc, np.ones((self.n, 1))
                )
            elif block.kind == STRUCTURED:
                term = _structured_term(block.name, block, frame)
            else:
                term = _low_rank_term(block.name, block, frame)
            if not term.m:
                raise ValueError(f"block {block.name!r} has no level in the training rows")
            term.local = term.m * term.K > DENSE_MAX
            self._add(term, block.name)

        offset = len(self.features)
        for term in self.terms:
            if term.local:
                term.theta = np.zeros((term.m, term.K))
            else:
                term.offset = offset
                offset += term.m * term.K
        self.P = offset
        self.theta = np.zeros(self.P)
        self.has_local = any(t.local for t in self.terms)
        self._pb = _problem(sp.csr_matrix((self.n, self.P + self.has_local)), codes, pos, self.st)
        self._D = self._design(self.X, [(t, t.codes, t.X) for t in self.terms])
        self._at = None  # _evaluate at the current point, while it holds
        self.z = np.full(len(self.groups), math.log(TAU2_START))
        self.counts = {"rows": self.n, "races": int(codes.max()) + 1}

    def label(self, g: int) -> str:
        """A prior group's name: its term's, with the coefficient of a structured block."""
        group = self.groups[g]
        term = self.terms[group.term]
        if term.kind == STRUCTURED:
            return f"{term.label}:{('a', 'b', 'c', 'e')[group.k]}"
        return term.label

    def _add(self, term: _Term, block: str | None = None) -> None:
        if not term.m:
            raise ValueError(f"{term.label}: no level in the training rows")
        self.terms.append(term)
        for k in range(term.K):
            term.groups.append(len(self.groups))
            self.groups.append(_Group(len(self.terms) - 1, k, block))

    # -- design and u -------------------------------------------------------------------------

    def _design(self, X: np.ndarray, parts) -> sp.csr_matrix:
        """The joint design: the features, then each non-local term's columns (k-major)."""
        n = X.shape[0]
        blocks = [sp.csr_matrix(X.reshape(n, len(self.features)))]
        for term, codes, cov in parts:
            if not term.local:
                blocks.append(_columns(term, codes, cov))
        return sp.hstack(blocks, format="csr")

    def _idx(self, g: int) -> np.ndarray:
        group = self.groups[g]
        term = self.terms[group.term]
        return term.offset + group.k * term.m + np.arange(term.m)

    def _values(self, g: int) -> np.ndarray:
        group = self.groups[g]
        term = self.terms[group.term]
        return term.theta[:, group.k] if term.local else self.theta[self._idx(g)]

    def _local_u(self, term: _Term, theta: np.ndarray | None = None) -> np.ndarray:
        theta = term.theta if theta is None else theta
        ok = term.codes >= 0
        out = np.zeros(self.n)
        out[ok] = np.einsum("ik,ik->i", term.X[ok], theta[term.codes[ok]])
        return out

    def _offset(self) -> np.ndarray:
        out = np.zeros(self.n)
        for term in self.terms:
            if term.local:
                out += self._local_u(term)
        return out

    def u(self) -> np.ndarray:
        return self._D @ self.theta + self._offset()

    def _joint(self) -> tuple[object, np.ndarray]:
        """The `_Problem` and pseudo-parameters for `_evaluate`: the joint design and, with local
        terms, their sum as one more column at coefficient 1."""
        if not self.has_local:
            return dataclasses.replace(self._pb, D=self._D), self.theta
        D = sp.hstack([self._D, sp.csr_matrix(self._offset()[:, None])], format="csr")
        return dataclasses.replace(self._pb, D=D), np.r_[self.theta, 1.0]

    def _evaluate(self, theta: np.ndarray | None = None):
        pb, star = self._joint()
        if theta is not None:
            star = np.r_[theta, 1.0] if self.has_local else theta
        loglik, score, info = _evaluate(star, pb)
        return loglik, score[: self.P], info[: self.P, : self.P]

    def _w_jacobian(self, term: _Term) -> sp.csr_matrix:
        """d u / d W of a low-rank term: U of the row's level at the row's column."""
        U = self._u_values(term)
        ok = np.flatnonzero((term.codes >= 0) & (term.ccodes >= 0))
        return sp.csr_matrix(
            (U[term.codes[ok]], (ok, term.ccodes[ok])), shape=(self.n, len(term.col_index))
        )

    def _joint_w_step(self, z: np.ndarray, tol: float, terms: list[_Term]) -> float:
        """One Gauss-Newton step on the joint parameters and the W of `terms` (non-local
        low-rank terms in use), each W moved in the tangent of its unit sphere (centred, and
        orthogonal to W: the objective l(U W) - prec |U|^2 |W|^2 / (2 m) does not change along
        the scale), halved until the objective does not worsen; then W is renormalised."""
        prec = self._precision(z)
        pb, star = self._joint()
        jac = [self._w_jacobian(t) for t in terms]
        extra = [c.shape[1] for c in jac]
        parts = [self._D, *jac] + (
            [sp.csr_matrix(self._offset()[:, None])] if self.has_local else []
        )
        star = np.r_[self.theta, np.zeros(sum(extra)), [1.0] if self.has_local else []]
        loglik, score, info = _evaluate(star, dataclasses.replace(pb, D=sp.hstack(parts, "csr")))
        bases = []
        for t in terms:
            mc = len(t.col_index)
            q, _ = np.linalg.qr(np.column_stack([np.ones(mc), t.W, np.eye(mc)]))
            bases.append(q[:, 2:mc])
        P = self.P
        T = scipy.linalg.block_diag(np.eye(P), *bases)
        n_full = P + sum(extra)
        H = T.T @ info[:n_full, :n_full] @ T
        g = T.T @ np.r_[score[:P] - prec * self.theta, score[P:n_full]]
        lam = []
        for t in terms:
            sl = slice(t.offset, t.offset + t.m)
            lam.append(float(prec[sl][0] * (self.theta[sl] @ self.theta[sl])) / len(t.col_index))
        tangent = [np.full(b.shape[1], lm) for b, lm in zip(bases, lam, strict=True)]
        H = H + np.diag(np.concatenate([prec, *tangent]))
        # the exact Hessian adds -d2u/dU dW times the score: the score summed per (row, column)
        # cell, so the step is Newton's (quadratic near the MAP); Gauss-Newton's where that is
        # not positive definite
        exact = H.copy()
        _, gu, _ = self._scores(self._D @ self.theta + (self._offset() if self.has_local else 0.0))
        at = P
        for t, basis in zip(terms, bases, strict=True):
            mc = len(t.col_index)
            ok = (t.codes >= 0) & (t.ccodes >= 0)
            cell = t.codes[ok] * mc + t.ccodes[ok]
            G = np.bincount(cell, weights=gu[ok], minlength=t.m * mc).reshape(t.m, mc) @ basis
            exact[t.offset : t.offset + t.m, at : at + basis.shape[1]] -= G
            exact[at : at + basis.shape[1], t.offset : t.offset + t.m] -= G.T
            at += basis.shape[1]
        try:
            factor = scipy.linalg.cho_factor(exact, check_finite=False)
            step = scipy.linalg.cho_solve(factor, g, check_finite=False)
        except np.linalg.LinAlgError:
            step = _solve(H, g)
        d_theta, d_w = step[:P], []
        at = P
        for b in bases:
            d_w.append(b @ step[at : at + b.shape[1]])
            at += b.shape[1]
        offset = self._offset() if self.has_local else 0.0

        def objective(theta: np.ndarray, ws: list[np.ndarray]) -> float:
            u = self._D @ theta + offset
            pen = prec * theta * theta
            for t, w in zip(terms, ws, strict=True):
                ok = (t.codes >= 0) & (t.ccodes >= 0)
                U = theta[t.offset : t.offset + t.m]
                u = u + np.where(ok, U[np.maximum(t.codes, 0)] * (w - t.W)[t.ccodes], 0.0)
                pen[t.offset : t.offset + t.m] *= float(w @ w) / len(t.col_index)
            return -self._ll(u) + 0.5 * float(pen.sum())

        current = -loglik + 0.5 * float(prec @ (self.theta * self.theta))
        scale = 1.0
        for _ in range(HALVINGS):
            trial = self.theta + scale * d_theta
            ws = [t.W + scale * d for t, d in zip(terms, d_w, strict=True)]
            biggest = max(
                float(np.abs(scale * d_theta).max()),
                *(
                    float(np.abs(scale * d).max() * np.abs(trial[t.offset : t.offset + t.m]).max())
                    for t, d in zip(terms, d_w, strict=True)
                ),
            )
            if objective(trial, ws) <= current or biggest < tol:
                break
            scale /= 2.0
        self.theta = trial
        for t, w in zip(terms, ws, strict=True):
            norm_w = float(np.sqrt(np.mean(w * w)))
            t.W = w / norm_w
            self.theta[t.offset : t.offset + t.m] *= norm_w
            t.X = self._lr_cov(t)
        self._D = self._design(self.X, [(t, t.codes, t.X) for t in self.terms])
        self._at = None
        return biggest

    def _scores(self, u: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """The log-likelihood at u, its score in u and the diagonal of its information in u."""
        st = self.st
        v = u[st.rows]
        p, lse = _softmax(v, st)
        dp = np.repeat(st.discount, st.sizes) * p
        loglik = float(st.weight @ v - st.discount @ lse)
        g = np.bincount(st.rows, weights=st.weight - dp, minlength=self.n)
        h = np.bincount(st.rows, weights=dp * (1.0 - p), minlength=self.n)
        return loglik, g, h

    def _ll(self, u: np.ndarray) -> float:
        return _loglik(u[self.st.rows], self.st)

    # -- steps --------------------------------------------------------------------------------

    def _precision(self, z: np.ndarray) -> np.ndarray:
        prec = np.zeros(self.P)
        for g, zz in enumerate(z):
            if not self.terms[self.groups[g].term].local:
                prec[self._idx(g)] = math.exp(-zz)
        return prec

    def _joint_step(self, prec: np.ndarray, tol: float) -> float:
        """One Newton step on the joint parameters, halved until the objective does not worsen
        (a trial reads only the log-likelihood)."""
        if self._at is None:
            self._at = self._evaluate()
        loglik, score, info = self._at
        step = _solve(info + np.diag(prec), score - prec * self.theta)
        current = -loglik + 0.5 * float(prec @ (self.theta * self.theta))
        offset = self._offset() if self.has_local else 0.0
        for _ in range(HALVINGS):
            trial = self.theta + step
            value = -self._ll(self._D @ trial + offset) + 0.5 * float(prec @ (trial * trial))
            if value <= current or np.abs(step).max() < tol:
                break
            step = step / 2.0
        self.theta, self._at = trial, None
        return float(np.abs(step).max()) if len(step) else 0.0

    def _local_curvature(self, term: _Term, h: np.ndarray, prec_k: np.ndarray) -> np.ndarray:
        """Each level's K x K block of the information plus its prior precision."""
        ok = term.codes >= 0
        c, X, hh = term.codes[ok], term.X[ok], h[ok]
        H = np.zeros((term.m, term.K, term.K))
        for a in range(term.K):
            for b in range(a, term.K):
                H[:, a, b] = np.bincount(c, weights=hh * X[:, a] * X[:, b], minlength=term.m)
                H[:, b, a] = H[:, a, b]
        H[:, np.arange(term.K), np.arange(term.K)] += prec_k
        return H

    def _local_step(self, term: _Term, z: np.ndarray, tol: float) -> float:
        prec_k = np.exp(-z[term.groups])
        u = self.u()
        loglik, g, h = self._scores(u)
        ok = term.codes >= 0
        c, X = term.codes[ok], term.X[ok]
        grad = np.column_stack(
            [np.bincount(c, weights=X[:, a] * g[ok], minlength=term.m) for a in range(term.K)]
        )
        grad -= prec_k * term.theta
        H = self._local_curvature(term, h, prec_k)
        if term.K == 1:
            step = grad / H[:, :, 0]
        else:
            step = np.linalg.solve(H, grad[:, :, None])[:, :, 0]
        current = -loglik + 0.5 * float((prec_k * term.theta * term.theta).sum())
        base_u = u - self._local_u(term)
        for _ in range(HALVINGS):
            trial = term.theta + step
            value = -self._ll(base_u + self._local_u(term, trial))
            value += 0.5 * float((prec_k * trial * trial).sum())
            if value <= current or np.abs(step).max() < tol:
                break
            step = step / 2.0
        term.theta = trial
        self._at = None
        return float(np.abs(step).max())

    def _u_values(self, term: _Term) -> np.ndarray:
        return term.theta[:, 0] if term.local else self.theta[term.offset : term.offset + term.m]

    def _set_u(self, term: _Term, values: np.ndarray) -> None:
        if term.local:
            term.theta[:, 0] = values
        else:
            self.theta[term.offset : term.offset + term.m] = values

    def _refresh(self, term: _Term) -> None:
        term.X = self._lr_cov(term)
        self._D = self._design(self.X, [(t, t.codes, t.X) for t in self.terms])
        self._at = None

    def _lr_cov(self, term: _Term) -> np.ndarray:
        cc = term.ccodes
        return np.where(cc >= 0, term.W[np.maximum(cc, 0)], 0.0)[:, None]

    def _init_w(self, term: _Term) -> None:
        """W from the leading right singular vector of the score summed per (row, column) cell at
        the current point with U = 0: the rank-one direction the likelihood climbs fastest."""
        self._set_u(term, np.zeros(term.m))
        _, g, _ = self._scores(self.u())
        ok = (term.codes >= 0) & (term.ccodes >= 0)
        mc = len(term.col_index)
        cell = term.codes[ok] * mc + term.ccodes[ok]
        G = np.bincount(cell, weights=g[ok], minlength=term.m * mc).reshape(term.m, mc)
        B = _centred_basis(mc)
        gram = B.T @ (G.T @ G) @ B
        vals, vecs = np.linalg.eigh(gram)
        w = B @ vecs[:, -1]
        term.W = _unit(w if np.abs(w).max() > 0 else B[:, 0])
        term.started = True
        self._refresh(term)

    def _w_step(self, term: _Term, z: np.ndarray, tol: float) -> float:
        """One Newton step on W given U: maximise l(U W') - (prec |U|^2 / 2) |W'|^2 / m over
        centred W' (the objective after rescaling W' to unit scale), then renormalise."""
        U = self._u_values(term)
        prec = math.exp(-z[term.groups[0]])
        mc = len(term.col_index)
        size = float(U @ U)
        if size < 1e-24:
            return 0.0
        lam = prec * size / mc
        st = self.st
        u = self.u()
        v = u[st.rows]
        p, lse = _softmax(v, st)
        loglik = float(st.weight @ v - st.discount @ lse)
        dp = np.repeat(st.discount, st.sizes) * p
        ok = (term.codes >= 0) & (term.ccodes >= 0)
        a = np.where(ok, U[np.maximum(term.codes, 0)], 0.0)
        col = np.where(ok, term.ccodes, 0)
        g = np.bincount(st.rows, weights=st.weight - dp, minlength=self.n)
        grad = np.bincount(col, weights=a * g, minlength=mc)
        a_s, c_s = a[st.rows], col[st.rows]
        first = np.bincount(c_s, weights=dp * a_s * a_s, minlength=mc)
        set_of = np.repeat(np.arange(len(st.starts)), st.sizes)
        msum = np.bincount(set_of * mc + c_s, weights=p * a_s, minlength=len(st.starts) * mc)
        msum = msum.reshape(len(st.starts), mc)
        info = np.diag(first) - msum.T @ (st.discount[:, None] * msum)
        B = _centred_basis(mc)
        W0 = term.W
        H = B.T @ (info + lam * np.eye(mc)) @ B
        step = B @ np.linalg.solve(H, B.T @ (grad - lam * W0))
        current = -loglik + 0.5 * lam * float(W0 @ W0)
        base_u = u - a * W0[col]
        for _ in range(HALVINGS):
            trial = W0 + step
            value = -self._ll(base_u + a * trial[col]) + 0.5 * lam * float(trial @ trial)
            if value <= current or np.abs(step).max() < tol:
                break
            step = step / 2.0
        scale = float(np.sqrt(np.mean(trial * trial)))
        term.W = trial / scale
        self._set_u(term, U * scale)
        self._refresh(term)
        # the step on u: a W that U barely uses (tau near 0) has converged whatever it does
        return float(np.abs(term.W - W0).max() * np.abs(U).max() * scale)

    def sweep(self, z: np.ndarray, tol: float, *, fix_w: bool = False) -> float:
        """One step on every part: the joint Newton step, each local term and, unless `fix_w`,
        each low-rank W (a non-local term's W in the joint step, once its U is in use). Returns
        the largest step."""
        joint_w = []
        if not fix_w:
            for t in self.terms:
                if t.kind == LOW_RANK and not t.local and t.started and len(t.col_index) > 2:
                    U = self._u_values(t)
                    if float(U @ U) > 1e-12:
                        joint_w.append(t)
        if joint_w:
            biggest = self._joint_w_step(z, tol, joint_w)
        else:
            biggest = self._joint_step(self._precision(z), tol) if self.P else 0.0
        for term in self.terms:
            if term.local:
                biggest = max(biggest, self._local_step(term, z, tol))
        for term in self.terms:
            if term.kind == LOW_RANK and not fix_w and term not in joint_w:
                if not term.started:
                    self._init_w(term)
                    biggest = max(biggest, 1.0)
                else:
                    biggest = max(biggest, self._w_step(term, z, tol))
        return biggest

    def solve(self, z: np.ndarray, tol: float, *, fix_w: bool = False) -> None:
        for _ in range(MAX_SWEEPS):
            if self.sweep(z, tol, fix_w=fix_w) < tol:
                return
        raise RuntimeError(f"the interaction fit did not converge in {MAX_SWEEPS} sweeps")

    # -- empirical Bayes and the evidence -----------------------------------------------------

    def _coupled(self, z: np.ndarray, *, trace: bool):
        """The posterior of a fit with local terms: the dense parameters' variances (diagonal of
        the inverse of their Schur complement), each local term's (m x K) variances and the
        log-determinant, all of the information with each runner's own curvature h only (the
        within-race coupling between runners dropped, as in the local steps) and the local levels
        coupled to the dense block: [[A, B], [B', C]] with C block-diagonal over the levels, A and
        B from the same h. Sparse: the dense block's Schur complement A - sum_t B_t C_t^-1 B_t'
        and, per local level, B_t C_t^-1's column (a few entries) against its inverse."""
        _, _, h = self._scores(self.u())
        prec = self._precision(z)
        Dh = sp.diags(h) @ self._D if self.P else None
        schur = (self._D.T @ Dh).toarray() + np.diag(prec) if self.P else np.zeros((0, 0))
        logdet, parts = 0.0, []
        for t, term in enumerate(self.terms):
            if not term.local:
                continue
            H = self._local_curvature(term, h, np.exp(-z[term.groups]))
            logdet += float(np.linalg.slogdet(H)[1].sum())
            inv = np.linalg.inv(H)  # m x K x K
            levels = np.arange(term.m)
            rows = np.concatenate(
                [a * term.m + levels for a in range(term.K) for _ in range(term.K)]
            )
            cols = np.concatenate(
                [b * term.m + levels for _ in range(term.K) for b in range(term.K)]
            )
            data = np.concatenate([inv[:, a, b] for a in range(term.K) for b in range(term.K)])
            size = term.m * term.K
            G = None
            if self.P:  # B C^-1, P x (m K)
                B = Dh.T @ _columns(term, term.codes, term.X)
                G = B @ sp.csr_matrix((data, (rows, cols)), shape=(size, size))
                schur -= (G @ B.T).toarray()
            parts.append((t, term, inv, G))
        diag, local_var = None, {}
        if self.P:
            if trace:
                inverse = _inverse(schur)
                diag = np.diag(inverse)
            else:
                chol, _ = scipy.linalg.cho_factor(schur, check_finite=False)
                logdet += 2.0 * float(np.log(np.abs(np.diag(chol))).sum())
        if trace:
            for t, term, inv, G in parts:
                eye = np.arange(term.K)
                var = inv[:, eye, eye]
                if G is not None:  # + G' S^-1 G, G = B C^-1 (column k * m + level)
                    var = var + _quad_diag(G, inverse).reshape(term.K, term.m).T
                local_var[t] = var
        return diag, local_var, logdet

    def _posterior(self, z: np.ndarray, *, trace: bool):
        """Per group: its levels, the trace of its posterior covariance (with `trace`, else 0)
        and |theta|^2; the log-likelihood; and, without `trace`, the log-determinant of the
        penalised information (see `_coupled` for a fit with local terms)."""
        prec = self._precision(z)
        if self._at is None:
            self._at = self._evaluate()
        loglik, _, info = self._at
        logdet, diag, local_var = 0.0, None, {}
        if self.has_local:
            diag, local_var, logdet = self._coupled(z, trace=trace)
        elif self.P:
            matrix = info + np.diag(prec)
            if trace:
                diag = np.diag(_inverse(matrix))
            else:
                chol, _ = scipy.linalg.cho_factor(matrix, check_finite=False)
                logdet += 2.0 * float(np.log(np.abs(np.diag(chol))).sum())
        out = []
        for g, group in enumerate(self.groups):
            term, values = self.terms[group.term], self._values(g)
            if not trace:
                tr = 0.0
            elif term.local:
                tr = float(local_var[group.term][:, group.k].sum())
            else:
                tr = float(diag[self._idx(g)].sum())
            out.append((term.m, tr, float(values @ values)))
        return out, loglik, logdet

    def evidence(self, z: np.ndarray) -> float:
        """The Laplace evidence at the current MAP for prior variances exp(z), up to a constant
        that does not depend on z."""
        groups, loglik, logdet = self._posterior(z, trace=False)
        prec = np.exp(-z)
        penalty = sum(0.5 * p * sq for p, (_, _, sq) in zip(prec, groups, strict=True))
        prior = sum(0.5 * m * math.log(p) for p, (m, _, _) in zip(prec, groups, strict=True))
        return loglik - penalty - 0.5 * logdet + prior

    def empirical_bayes(self) -> None:
        """MacKay's fixed point on z = ln tau^2 per group, `models.ability`'s rule: one sweep per
        update until the residual is within EB_TOL (EB_SWITCH for a fit with local or low-rank
        terms, whose single sweep leaves the MAP behind), then the MAP solved to MAP_TOL at each z
        from there on, so the fit stops at a MAP_TOL MAP whose z meets the fixed point within
        EB_TOL (EB_LOCAL_TOL with local terms: their alternating MAP leaves residual noise of that
        order). Where the secant has no valid slope, a group whose residual keeps its sign doubles
        its plain step (up to EB_JUMP), so a nearly flat fixed point (a weakly identified
        variance) is crossed in a few updates instead of crawled to."""
        z = self.z
        floor = math.log(TAU2_FLOOR)
        alternating = self.has_local or any(t.kind == LOW_RANK for t in self.terms)
        switch = EB_SWITCH if alternating else EB_TOL
        tol = EB_LOCAL_TOL if self.has_local else EB_TOL
        last, tight = None, not len(self.groups)
        gain = np.ones(len(self.groups))
        for _ in range(EB_MAX):
            if tight:
                self.solve(z, MAP_TOL)
            else:
                self.sweep(z, MAP_TOL)
            if not len(self.groups):
                break
            groups, _, _ = self._posterior(z, trace=True)
            update = np.empty(len(groups))
            for g, (m, trace, square) in enumerate(groups):
                effective = m - trace * math.exp(-z[g])
                ratio = square / effective if effective > 0 and square > 0 else 0.0
                update[g] = max(math.log(ratio), floor) if ratio > 0 else floor
            r = update - z
            if tight and np.abs(r).max() < tol:
                self.z = z
                return
            if not tight and np.abs(r).max() < switch:
                tight = True
                continue
            step = r.copy()
            if last is not None:
                dz, dr = z - last[0], r - last[1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    secant = -r * dz / dr
                valid = (dz != 0) & (dr * dz < 0) & np.isfinite(secant)
                same = np.sign(r) == np.sign(last[1])
                gain = np.where(valid | ~same, 1.0, np.minimum(2.0 * gain, 1024.0))
                step = np.where(valid, secant, gain * r)
            last = (z, r)
            z = np.maximum(z + np.clip(step, -EB_JUMP, EB_JUMP), floor)
        else:
            raise RuntimeError(f"empirical Bayes did not converge in {EB_MAX} updates")
        self.z = z

    # -- state --------------------------------------------------------------------------------

    def state(self) -> tuple:
        return (
            self.theta.copy(),
            [None if t.theta is None else t.theta.copy() for t in self.terms],
            [None if t.W is None else t.W.copy() for t in self.terms],
            [t.started for t in self.terms],
        )

    def restore(self, state: tuple) -> None:
        theta, local, ws, started = state
        self.theta = theta.copy()
        for term, th, w, s in zip(self.terms, local, ws, started, strict=True):
            term.theta = None if th is None else th.copy()
            if w is not None:
                term.W = w.copy()
                term.X = self._lr_cov(term)
            term.started = s
        self._D = self._design(self.X, [(t, t.codes, t.X) for t in self.terms])
        self._at = None

    def warm(self, start: dict) -> None:
        """Start from another fit's `warm_start()`: its prior sds by group name, its feature
        coefficients, each term's values on the levels both fits share (0 elsewhere) and each
        low-rank W on the shared columns (re-centred and rescaled)."""
        for g in range(len(self.groups)):
            sd = start.get("prior_sd", {}).get(self.label(g))
            if sd is not None and math.isfinite(sd) and sd > 0:
                self.z[g] = max(2.0 * math.log(sd), math.log(TAU2_FLOOR))
        for j, feat in enumerate(self.features):
            value = start.get("coef", {}).get(feat)
            if value is not None and math.isfinite(value):
                self.theta[j] = value
        for term in self.terms:
            prev = start.get("terms", {}).get(term.label)
            if prev is None or prev[1].shape[1] != term.K:
                continue
            at = prev[0].get_indexer(term.index)
            values = np.where(at[:, None] >= 0, prev[1][np.maximum(at, 0)], 0.0)
            if term.local:
                term.theta = values
            else:
                for k in range(term.K):
                    self.theta[term.offset + k * term.m : term.offset + (k + 1) * term.m] = values[
                        :, k
                    ]
            w = start.get("W", {}).get(term.label)
            if term.kind == LOW_RANK and w is not None:
                at = w[0].get_indexer(term.col_index)
                mapped = np.where(at >= 0, w[1][np.maximum(at, 0)], 0.0)
                if np.abs(mapped - mapped.mean()).max() > 0:
                    term.W, term.started = _unit(mapped), True
                    term.X = self._lr_cov(term)
        self._D = self._design(self.X, [(t, t.codes, t.X) for t in self.terms])
        self._at = None


class InteractionFit:
    """A fitted AbilityModel with interaction blocks: `tau(name)` and `predict(frame)`."""

    def __init__(self, model: _Model) -> None:
        self._model = model
        self._state = model.state()
        self._z = model.z.copy()
        self._tau: dict[str, Interval] = {}
        self.blocks = list(model.blocks)
        self.coef_ = pd.Series(model.theta[: len(model.features)], index=model.features)
        self.prior_sd_ = {model.label(g): math.exp(0.5 * z) for g, z in enumerate(self._z)}
        self.counts = dict(model.counts)

    def _block_groups(self, name: str) -> list[int]:
        groups = [g for g, group in enumerate(self._model.groups) if group.block == name]
        if not groups:
            raise KeyError(name)
        return groups

    def _scale(self, name: str, z: np.ndarray) -> float:
        """The block's tau at prior variances exp(z)."""
        groups = self._block_groups(name)
        term = self._model.terms[self._model.groups[groups[0]].term]
        if term.kind != STRUCTURED:
            return math.exp(0.5 * z[groups[0]])
        tau2 = np.exp(z[groups])
        return float(np.sqrt(np.mean((term.X * term.X) @ tau2)))

    def warm_start(self) -> dict:
        """What `fit_with_blocks(..., start=)` reads to start another fit from this one."""
        model = self._model
        theta, local, ws, _ = self._state
        terms, w = {}, {}
        for term, th, wt in zip(model.terms, local, ws, strict=True):
            if term.local:
                values = th
            else:
                values = np.column_stack(
                    [
                        theta[term.offset + k * term.m : term.offset + (k + 1) * term.m]
                        for k in range(term.K)
                    ]
                )
            terms[term.label] = (term.index, values.copy())
            if term.kind == LOW_RANK:
                w[term.label] = (term.col_index, wt.copy())
        return {
            "prior_sd": dict(self.prior_sd_),
            "coef": {f: float(v) for f, v in self.coef_.items()},
            "terms": terms,
            "W": w,
        }

    def tau_point(self, name: str) -> float:
        """The block's tau at the empirical-Bayes prior variances (no interval)."""
        return self._scale(name, self._z)

    def tau(self, name: str) -> Interval:
        """The block's tau and its profile-evidence bounds (see the module docstring)."""
        if name in self._tau:
            return self._tau[name]
        groups = self._block_groups(name)
        model, z0 = self._model, self._z
        point = self._scale(name, z0)
        drop = 0.5 * norm.ppf(TAU_LEVEL) ** 2
        solved: dict[float, tuple] = {1.0: self._state}  # profile points' MAPs, warm starts

        def at(s: float) -> float:
            z = z0.copy()
            z[groups] += 2.0 * math.log(s)
            near = min(solved, key=lambda t: abs(math.log(t / s)))
            model.restore(solved[near])
            model.solve(z, PROFILE_TOL, fix_w=True)
            solved[s] = model.state()
            return model.evidence(z)

        model.restore(self._state)
        target = model.evidence(z0) - drop
        lo = 0.0
        if point > TAU_ZERO and at(TAU_ZERO / point) < target:
            a, b = TAU_ZERO / point, 1.0  # evidence below the target at a, above at b
            for _ in range(PROFILE_BISECTIONS):
                mid = 0.5 * (a + b)
                a, b = (mid, b) if at(mid) < target else (a, mid)
            lo = a * point
        b = 1.0
        for _ in range(PROFILE_DOUBLINGS):
            if at(PROFILE_GROWTH * b) < target:
                break
            b *= PROFILE_GROWTH
        a, b = b, PROFILE_GROWTH * b  # above the target at a, below at b (or the growth ran out)
        for _ in range(PROFILE_BISECTIONS):
            mid = math.sqrt(a * b)
            a, b = (mid, b) if at(mid) >= target else (a, mid)
        hi = b * point
        model.restore(self._state)
        iv = Interval(mean=point, lo=min(lo, point), hi=max(hi, point), n_clusters=0)
        m = model.terms[model.groups[groups[0]].term].m
        self._tau[name] = dataclasses.replace(iv, n_clusters=int(m))
        return self._tau[name]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One row per frame row (frame's index): race_id, horse_id, u and p_ability (softmax of
        u within race). Reads race_id, horse_id, the features and the terms' columns only."""
        model = self._model
        theta, local, ws, _ = self._state
        X = _features(frame, model.features)
        parts, u = [], np.zeros(len(frame))
        for term, th, w in zip(model.terms, local, ws, strict=True):
            if term.kind == LOW_RANK:
                cc = _lookup(term.col_index, frame, (term.col,))
                cov = np.where(cc >= 0, w[np.maximum(cc, 0)], 0.0)[:, None]
            else:
                cov = term.covariates(frame)
            codes = _lookup(term.index, frame, term.key_cols)
            if term.local:
                ok = codes >= 0
                u[ok] += np.einsum("ik,ik->i", cov[ok], th[codes[ok]])
            parts.append((term, codes, cov))
        u += model._design(X, parts) @ theta
        race = _race_codes(frame["race_id"])
        top = pd.Series(u).groupby(race).transform("max").to_numpy()
        e = np.exp(u - top)
        p = e / pd.Series(e).groupby(race).transform("sum").to_numpy()
        return pd.DataFrame(
            {
                "race_id": frame["race_id"].to_numpy(),
                "horse_id": frame["horse_id"].to_numpy(),
                "u": u,
                "p_ability": p,
            },
            index=frame.index,
        )[OUT_COLUMNS]


def fit_with_blocks(
    frame: pd.DataFrame,
    base: AbilityModel,
    blocks: list[InteractionBlock],
    *,
    start: dict | None = None,
) -> InteractionFit:
    """`base` (an unfitted AbilityModel: its k, discounts, features and entity columns) with the
    blocks added, fitted on frame's rows only (its training window); see the module docstring.
    `start` (another fit's `warm_start()`, or a dict with only its "prior_sd") starts the solver
    there instead of at 0 and TAU2_START: a walk-forward fit starts from the previous fold's,
    which changes the steps it takes, not the MAP and fixed point it stops at (within MAP_TOL and
    EB_TOL; a low-rank block's W from the previous fold's, so not from the score's direction)."""
    if not isinstance(base, AbilityModel):
        raise TypeError(f"base must be an AbilityModel, got {type(base).__name__}")
    for block in blocks:
        if not isinstance(block, InteractionBlock):
            raise TypeError(f"blocks must be InteractionBlocks, got {type(block).__name__}")
    model = _Model(frame, base, list(blocks))
    if start:
        model.warm(start)
    model.empirical_bayes()
    return InteractionFit(model)
