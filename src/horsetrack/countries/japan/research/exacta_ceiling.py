"""N2 (#431): the exacta look-ahead ceiling, #131's E2 cuts scored with the selection read from the
final boards (`horsetrack japan exacta-ceiling freeze|run`).

LOOK-AHEAD everywhere: a combination c of category X (fav_second or no_fav against the favourite
of the final Win board) is bought for 100 yen when r = q_c / p_c < k, q_c the final exacta board's
implied share (``implied_share``, breakage added back, NaN at or below 110 yen) and p_c the
discounted Harville probability on the normalised implied Win shares of the final Win board
(#131's rule-v5 p_c(t), read on the final board: the v2 families). The final board holds money
placed after T-2, so the result is not a betting rule and not a proven bound on the T-2 rule
(informed late money); docs/specs/exacta-ceiling-prereg.md. The v1 families (p from #114's
out-of-fold ``p_cal``, races with a #114 row) are superseded: frozen, recorded and ranked as
variant rows from the registry, never executed again.

- Return at the final dividend with own impact: stake / 100 x D x (P + n s) / P x W / (W + s), n
  combinations bought, P the pool's final money net of refunds, W the winner's final money.
- rho = sum returned / sum (staked x rate), each race's own payout rate; break-even 1 / rate. Its
  one-sided 95% bounds come from #131's meeting bootstrap (``exacta_study.rho_interval``).
- Every frozen member is a ranked row (owner reframe, 2026-11-27): rho descending, a member
  without a bet last, a v1 member with no scored record after it (unavailable, not counted as
  searched). No verdict and no kill rule.
- Descriptives: D = sum q / sum p per category and cell, and the Quinella x order split (each
  Quinella pair share split by Harville's order probabilities, against the exacta shares).

``run_ceiling`` refuses (``RunRefused``) before any store read unless every family of
``family_specs`` is frozen with that spec. It reads development days only (``is_holdout_day``,
D14; never ``LockedHoldout``), none before 2014-06-07 (D10), one year per ``load_races`` call.
A run that scores no race records its members as failed and publishes nothing
(``NoEligibleRace``). Every figure goes through ``write_report`` (jravan is withheld, D21) and
the private registry. Each run writes its own report and export, named by its commit, data build,
seed and asked days and carrying the days it scored, and never the ledger-linked ``CELLS_FILE``.
The run of record is the one run whose commit, data build, seed and asked days are
``RECORD_COMMIT``, ``RECORD_BUILD``, ``DEFAULT_SEED`` and ``RECORD_DAYS`` (the ledger's N2 row);
``record_cells`` (`exacta-ceiling record`) writes ``CELLS_FILE`` from its export alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer

from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    Registry,
    RunRecord,
    _canonical,
    _record_data,
    _sha256,
    _spec_hash,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan import takeout
from horsetrack.countries.japan.research import exacta_run, pool_depth
from horsetrack.countries.japan.research import exacta_study as study
from horsetrack.countries.japan.research.exacta_consistency import (
    CATEGORIES,
    favourite,
    harville_exacta,
)

COUNTRY = "japan"
SOURCE = "jravan"
SOURCES = (SOURCE,)
TICKET = 431
HYPOTHESIS_ID = "E43"  # as #131
FIRST_DAY = date(2014, 6, 7)  # D10
FOLD_SCHEME = "development-days"
# v2 (Sol review round 1 of 24d080f): p from the final Win board's normalised implied shares,
# every development day with final Win and exacta boards, as #131's rule-v5. The run's families.
FAMILY_ID = "japan-431-exacta-ceiling-v2"  # the 16 members at #115's full-development lambda2
PLAIN_ID = "japan-431-exacta-ceiling-plain-harville-v2"  # the same 16 at lambda2 = 1
DESCRIPTIVE_ID = "japan-431-exacta-ceiling-descriptive-v2"  # D and the Quinella x order split
MEMBER_FAMILIES = (FAMILY_ID, PLAIN_ID)
# v1, superseded: p from #114's out-of-fold p_cal on the races with a #114 row (rule-v4's
# inputs). Frozen and recorded, ranked as variant rows from the registry, never executed again.
V1_ID = "japan-431-exacta-ceiling"
V1_PLAIN_ID = "japan-431-exacta-ceiling-plain-harville"
V1_DESCRIPTIVE_ID = "japan-431-exacta-ceiling-descriptive"
V1_MEMBER_FAMILIES = (V1_ID, V1_PLAIN_ID)
RULE_CATEGORIES = ("fav_second", "no_fav")
CUTS = (0.6, 0.7, 0.8, 0.9)
CELLS = ("all", "headline")
MEMBERS = tuple((c, k, s) for c in RULE_CATEGORIES for k in CUTS for s in CELLS)
VARIANTS_SEARCHED = len(MEMBERS) * (len(MEMBER_FAMILIES) + len(V1_MEMBER_FAMILIES))  # 64
WIN_SOURCE = "final-win-board-implied-shares"  # v2's p: no fitted Win model enters
P_SOURCE = {FAMILY_ID: "board", PLAIN_ID: "board", V1_ID: "p_cal", V1_PLAIN_ID: "p_cal"}
P_SOURCES = {
    "board": "normalised implied Win shares of the final Win board (v2, as #131's rule-v5)",
    "p_cal": "#114's out-of-fold calibrated p_cal (v1, superseded, as #131's rule-v4)",
}
SPANS = {
    "board": "every development day from 2014-06-07 with final Win and exacta boards",
    "p_cal": "the development races with #114 out-of-fold Win probabilities (its test quarters)",
}
STAKE = 100.0
N_BOOT = 10_000
LEVEL = 0.95
DEFAULT_SEED = 431
POOL_ROW = "__pool__"
UNIT = 100.0  # yen per vote
WITHHELD = "withheld (D21)"
LOOK_AHEAD = "look-ahead: the selection reads the final boards"
CELLS_FILE = Path("candidate-cells") / "exacta-ceiling.json"
# The run of record (the ledger's N2 row): its commit, the sha256 of its merged data builds
# (``_build_digest``), seed DEFAULT_SEED and these asked days. Every run, this one too, writes
# only its own report and export; ``record_cells`` writes CELLS_FILE from this run's export alone.
RECORD_COMMIT = "fd4df5c079608285c07955c0efa66ef71b2da94d"
RECORD_BUILD = "3fbc963a25ea052e29fb268eaec69381a6846241eea9ec5535a48bb0d846028f"
RECORD_DAYS = (FIRST_DAY, date(2026, 11, 27))
NO_RECORD = "no scored record of this superseded member in the registry"
NO_BET = "no combination selected"
FEW_MEETINGS = "fewer than two meetings: no bootstrap bound"
EMPTY = "no scored race with a share in the cell"
NO_RACE = "no eligible race in the asked development days"


class RunRefused(RuntimeError):
    """The run refused before it read, recorded or wrote anything."""


class NoEligibleRace(RuntimeError):
    """The run scored no race: its members are recorded as failed and nothing is published."""


# --------------------------------------------------------------------------- pre-registration


def _member_spec(family_id: str, lambda2: str | float, role: str) -> dict:
    return {
        "ticket": TICKET,
        "hypothesis_id": HYPOTHESIS_ID,
        "spec": "docs/specs/exacta-ceiling-prereg.md",
        "family_id": family_id,
        "role": role,
        "label": "look-ahead",
        "rule": "buy exacta c of category X for 100 yen when q_c / p_c < k, read on the final "
        "boards (look-ahead): #131's E2 rule with its selection on the final exacta and Win "
        "boards instead of T-2",
        "members": [{"category": c, "cut": k, "cell": s} for c, k, s in MEMBERS],
        "categories": list(RULE_CATEGORIES),
        "cuts": list(CUTS),
        "cells": {
            "all": "every scored race",
            "headline": "#131's headline cell (exacta_study.segment_cells): the final Win "
            "favourite's odds >= 3.0 and less popular (race_no 1-4 or a DEBUT/MAIDEN race)",
        },
        "q": "implied_share(final exacta odds x 100, rate): rate / ((D + 5) / 100), NaN at or "
        "below 110 yen (never selected); the final exacta odds are final_odds FORECAST rows",
        "p": "discounted Harville (harville_exacta) on #114's out-of-fold calibrated p_cal over "
        "the starters, renormalised; a race whose priced final Win field is not #114's horses "
        "is dropped; a race whose final Win board's implied shares differ from #114's q by "
        "more than 1e-9 keeps its p_cal and is counted (diagnostic)",
        "lambda2": lambda2,
        "favourite": "lowest final Win odds (final_odds WIN), ties to the lowest number",
        "stake_yen": 100,
        "return": "at the final dividend with own impact: stake / 100 x D x (P + n s) / P x "
        "W / (W + s), P the exacta pool's final votes net of refund votes x 100 (pool_votes "
        "__pool__), W the paying combination's final votes x 100, not rounded again",
        "estimand": "rho = sum returned / sum (staked x rate), rate the race's exacta payout "
        "rate (takeout.race_takeout(race_id, 'FORECAST', race_class_raw=...)); break-even 1 / rate",
        "secondary": "return_per_100 = 100 x sum returned / sum staked, with its upper bound",
        "interval": "meeting bootstrap (exacta_study.rho_interval), one-sided 95% upper bound "
        "(the CI column) and lower bound, 10000 replicates, the run's seed (default 431); no "
        "bound with fewer than two meetings",
        "n_boot": N_BOOT,
        "level": LEVEL,
        "ranking": "rho descending; a member without a bet ranks after every member with one",
        "variants_searched": len(MEMBERS) * len(V1_MEMBER_FAMILIES),
        "family_size": len(MEMBERS),
        "exclusions": [
            "boost_may_be_withheld",
            "fewer than 5 starters",
            "no exacta row",
            "a refunded exacta",
            "more than one paying exacta (dead heat)",
            "segments undefined (field outside 5-18, race class unknown)",
            "no priced exacta board",
            "no exacta pool total",
            "no final money on the paying combination",
        ],
        "dropped_before_scoring": [
            "no #114 out-of-fold Win probabilities",
            "race status other than run",
            "race class outside exacta_study.CLASS_GROUPS",
            "priced final Win field other than #114's horses",
            "takeout unknown",
        ],
        "data": "jravan final_odds (WIN, QUINELLA, FORECAST), pool_votes (FORECAST), dividends "
        "(FORECAST), races; confirmed records (data_kubun 4, 5); development days only, from "
        "2014-06-07; races with #114 out-of-fold Win probabilities",
        "caveat": "a selection on the final board sees money placed after T-2; it is not a "
        "proven upper bound on the T-2 rule (informed late money): a ledger assumption",
        "verdict": "none: every member is a ranked row (owner reframe 2026-11-27)",
    }


def _descriptive_spec(family_id: str) -> dict:
    return {
        "ticket": TICKET,
        "hypothesis_id": HYPOTHESIS_ID,
        "spec": "docs/specs/exacta-ceiling-prereg.md",
        "family_id": family_id,
        "role": "descriptive, not a member: no rank",
        "label": "look-ahead",
        "descriptives": {
            "d": "sum q_c / sum p_c over the priced combinations with an identified q_c (above "
            "110 yen) of each category (fav_first, fav_second, no_fav) and cell (all, "
            "headline), per lambda2, over the members' scored races",
            "quinella_order_split": "each final Quinella pair share split into (i, j) and "
            "(j, i) by Harville's order probabilities; per category and cell sum of the exacta "
            "shares over sum of the split Quinella shares (QUINELLA payout rate for the "
            "Quinella), per lambda2",
        },
        "lambda2": ["full_development", 1.0],
        "interval": "meeting bootstrap of the ratio, two-sided 95%, 10000 replicates",
        "n_boot": N_BOOT,
    }


def superseded_specs() -> dict[str, dict]:
    """The v1 families (p from #114's p_cal), frozen and recorded on 2026-11-28 and superseded by
    v2: kept so their hashes and committed summary stay checkable; never executed again."""
    return {
        V1_ID: _member_spec(V1_ID, "full_development", "the 16 frozen members"),
        V1_PLAIN_ID: _member_spec(V1_PLAIN_ID, 1.0, "variant: plain Harville (lambda2 = 1)"),
        V1_DESCRIPTIVE_ID: _descriptive_spec(V1_DESCRIPTIVE_ID),
    }


_V2_REVISION = (
    "v2 (Sol review round 1 of 24d080f, 2026-11-28): v1 took p from #114's out-of-fold "
    "calibrated p_cal and scored only the races with a #114 row, the inputs of #131's "
    "superseded rule-v4. #131's frozen E2 is rule-v5: p_c(t) is discounted Harville at the "
    "registered lambda2 on the normalised implied Win shares of the Win board, on every day with "
    "Win and exacta boards. v2 is rule-v5's look-ahead analogue: the same shares read on the "
    "final Win board, every development day from 2014-06-07 with final Win and exacta boards. "
    "It also excludes a race whose exacta pool has refund votes (a scratch after sales: the "
    "JRA-VAN dividends carry no refund row) and records the bootstrap seed with every result. "
    "Members, cuts, cells, lambda2, favourite, stake, return, estimand, interval and ranking "
    "are v1's. v1's families and records stay in the registry as ranked variant rows."
)
_V2_P = (
    "discounted Harville (harville_exacta) at lambda2 on the normalised implied Win shares of "
    "the final Win board (final_odds WIN: 1 / odds over the race's sum of 1 / odds, over the "
    "horses with a positive final Win odds): #131's rule-v5 p_c(t) read on the final board; no "
    "fitted Win model enters"
)
_V2_EXCLUSIONS = [
    "boost_may_be_withheld",
    "fewer than 5 starters",
    "no exacta row",
    "a refunded exacta: a paying row with is_refund, or refund votes on the exacta pool's "
    "__pool__ row (a scratch after sales), so no combination on a scratched horse is scored",
    "more than one paying exacta (dead heat)",
    "segments undefined (field outside 5-18, race class unknown)",
    "no priced exacta board",
    "no exacta pool total",
    "no final money on the paying combination",
]
_V2_DROPPED = [
    "race status other than run",
    "race class outside exacta_study.CLASS_GROUPS",
    "no priced final Win board",
    "takeout unknown",
]
_V2_DATA = (
    "jravan final_odds (WIN, QUINELLA, FORECAST), pool_votes (FORECAST), dividends (FORECAST), "
    "races; confirmed records (data_kubun 4, 5); development days only, from 2014-06-07; every "
    "race with a final Win board and a final exacta board (no #114 row required)"
)
_V2_SEED = "the run's seed (default 431) in every record's params and in the private report's name"


def _member_spec_v2(family_id: str, lambda2: str | float, role: str, supersedes: str) -> dict:
    return {
        **_member_spec(family_id, lambda2, role),
        "rule": "buy exacta c of category X for 100 yen when q_c / p_c < k, read on the final "
        "boards (look-ahead): #131's E2 rule-v5 with its selection on the final exacta and Win "
        "boards instead of T-2",
        "p": _V2_P,
        "exclusions": list(_V2_EXCLUSIONS),
        "dropped_before_scoring": list(_V2_DROPPED),
        "data": _V2_DATA,
        "seed": _V2_SEED,
        "variants_searched": VARIANTS_SEARCHED,
        "variants_searched_counts": "every member family of the study: v2's two (this run) and "
        "v1's two (superseded, recorded), 16 members each",
        "supersedes": supersedes,
        "revision": _V2_REVISION,
    }


def family_specs() -> dict[str, dict]:
    """Every family the run writes (v2), as plain-JSON specs (a fresh copy on every call)."""
    descriptive = {
        **_descriptive_spec(DESCRIPTIVE_ID),
        "p": _V2_P,
        "data": _V2_DATA,
        "seed": _V2_SEED,
        "supersedes": V1_DESCRIPTIVE_ID,
        "revision": _V2_REVISION,
    }
    return {
        FAMILY_ID: _member_spec_v2(FAMILY_ID, "full_development", "the 16 frozen members", V1_ID),
        PLAIN_ID: _member_spec_v2(
            PLAIN_ID, 1.0, "variant: plain Harville (lambda2 = 1)", V1_PLAIN_ID
        ),
        DESCRIPTIVE_ID: descriptive,
    }


# --------------------------------------------------------------------------- per race


def _implied(odds: np.ndarray, rate: float) -> np.ndarray:
    """``implied_share`` of decimal odds, vectorised: NaN at or below 110 yen or unpriced."""
    dividend = np.asarray(odds, dtype=float) * 100
    with np.errstate(invalid="ignore", divide="ignore"):
        share = rate / ((dividend + 5) / 100)
    return np.where(dividend > 110, share, np.nan)


def _model(p: Mapping[int, float], lambda2: float) -> tuple[dict[int, int], np.ndarray]:
    horses = sorted(p)
    probs = np.array([p[h] for h in horses], dtype=float)
    return {h: k for k, h in enumerate(horses)}, harville_exacta(probs, lambda2)


def _codes(first: np.ndarray, second: np.ndarray, fav: int) -> np.ndarray:
    """CATEGORIES codes of (first, second) horse numbers against the favourite."""
    return np.where(first == fav, 0, np.where(second == fav, 1, 2)).astype(np.int8)


@dataclass(frozen=True)
class _Priced:
    """A final exacta board's priced combinations within the field."""

    combos: list[tuple[int, int]]
    a: np.ndarray  # first's index into the model
    b: np.ndarray
    category: np.ndarray
    q: np.ndarray


def _priced(board: Mapping, index: dict[int, int], fav: int, rate: float) -> _Priced:
    combos = sorted(tuple(c) for c in board if c[0] in index and c[1] in index and c[0] != c[1])
    odds = np.array([float(board[c]) for c in combos], dtype=float)
    keep = np.isfinite(odds) & (odds > 0)
    combos = [c for c, k in zip(combos, keep, strict=True) if k]
    horses = np.array(combos, dtype=int).reshape(-1, 2)
    return _Priced(
        combos,
        np.array([index[h] for h in horses[:, 0]], dtype=int),
        np.array([index[h] for h in horses[:, 1]], dtype=int),
        _codes(horses[:, 0], horses[:, 1], fav),
        _implied(odds[keep], rate),
    )


def _rule_code(category: str) -> int:
    if category not in RULE_CATEGORIES:
        raise ValueError(f"the rule buys {RULE_CATEGORIES} only, got {category!r}")
    return CATEGORIES.index(category)


def select_combinations(
    board: Mapping[tuple[int, int], float],
    p: Mapping[int, float],
    *,
    favourite: int,
    rate: float,
    lambda2: float,
    category: str,
    cut: float,
) -> list[tuple[int, int]]:
    """The look-ahead selection: the combinations of ``category`` with q / p_c < ``cut`` on the
    final exacta board (decimal odds). fav_first is never bought (ValueError)."""
    code = _rule_code(category)
    index, model = _model(p, lambda2)
    priced = _priced(board, index, favourite, rate)
    with np.errstate(invalid="ignore"):
        r = priced.q / model[priced.a, priced.b]
        chosen = (priced.category == code) & (r < cut)
    return [priced.combos[k] for k in np.flatnonzero(chosen)]


def _positive(name: str, value) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def race_return(
    selected,
    *,
    winner: tuple[int, int] | None,
    dividend: float | None,
    pool_yen: float,
    money_yen: Mapping[tuple[int, int], float],
    stake: float,
    rate: float,
) -> tuple[float, float]:
    """(staked, returned) of ``stake`` on each selected combination, paid at the final dividend
    with own impact: stake / 100 x D x (P + n s) / P x W / (W + s). ``rate`` is checked only: the
    dividend already carries it."""
    stake, _ = _positive("stake", stake), _positive("rate", rate)
    chosen = [tuple(c) for c in selected]
    if len(set(chosen)) != len(chosen):
        raise ValueError("a combination is selected twice")
    staked = stake * len(chosen)
    if winner is None or tuple(winner) not in chosen:
        return staked, 0.0
    pool = _positive("pool_yen", pool_yen)
    w = _positive("the winner's money", money_yen[tuple(winner)])
    d = _positive("dividend", dividend)
    return staked, stake / 100 * d * (pool + len(chosen) * stake) / pool * w / (w + stake)


def _split(quinella: Mapping, exacta: Mapping, index, model, fav, q_rate, e_rate):
    """(category codes, exacta shares, split Quinella shares) of every combination whose exacta
    share and Quinella pair share are both identified."""
    pairs = sorted({(min(c), max(c)) for c in quinella if c[0] in index and c[1] in index})
    if not pairs:
        return np.empty(0, np.int8), np.empty(0), np.empty(0)
    pair = np.array(pairs, dtype=int)
    sq = _implied([quinella.get(c, quinella.get(c[::-1])) for c in pairs], q_rate)
    a = np.array([index[h] for h in pair[:, 0]])
    b = np.array([index[h] for h in pair[:, 1]])
    h_ab, h_ba = model[a, b], model[b, a]
    nan = math.nan
    e_ab = _implied([exacta.get((i, j), nan) for i, j in pairs], e_rate)
    e_ba = _implied([exacta.get((j, i), nan) for i, j in pairs], e_rate)
    first = np.concatenate([pair[:, 0], pair[:, 1]])
    second = np.concatenate([pair[:, 1], pair[:, 0]])
    e = np.concatenate([e_ab, e_ba])
    s = np.concatenate([sq * h_ab / (h_ab + h_ba), sq * h_ba / (h_ab + h_ba)])
    ok = np.isfinite(e) & np.isfinite(s)
    return _codes(first, second, fav)[ok], e[ok], s[ok]


def quinella_order_split(
    quinella: Mapping[tuple[int, int], float],
    exacta: Mapping[tuple[int, int], float],
    p: Mapping[int, float],
    *,
    favourite: int,
    lambda2: float,
    quinella_rate: float,
    exacta_rate: float,
) -> dict[str, float]:
    """Per category: sum of the exacta shares over sum of the Harville-split Quinella shares."""
    index, model = _model(p, lambda2)
    cat, e, s = _split(quinella, exacta, index, model, favourite, quinella_rate, exacta_rate)
    out = {}
    for code, name in enumerate(CATEGORIES):
        m = cat == code
        out[name] = float(e[m].sum() / s[m].sum()) if m.any() else math.nan
    return out


# --------------------------------------------------------------------------- the store seam


RACE_COLUMNS = [
    "race_id",
    "meeting_id",
    "race_date",
    "race_no",
    "race_class",
    "race_class_raw",
    "status",
]
BOARD_POOLS = ("WIN", "QUINELLA", "FORECAST")


def _ids_filter(ids) -> ds.Expression:
    return ds.field("race_id").isin(pa.array(list(ids), pa.string()))


def _boards(frame: pd.DataFrame, value: str) -> dict[str, dict]:
    """race_id -> {combination: value}; a combination "i,j" is a tuple, "h" an int."""
    out: dict[str, dict] = {}
    for rid, combo, v in zip(frame["race_id"], frame["combination"], frame[value], strict=True):
        legs = tuple(int(x) for x in str(combo).split(","))
        out.setdefault(rid, {})[legs if len(legs) > 1 else legs[0]] = float(v)
    return out


def load_races(days: Sequence[date]) -> tuple[list[dict], dict]:
    """The scoring inputs of the development races on ``days`` from the jravan store (confirmed
    records), p the normalised implied Win shares of the final Win board, and the run's meta."""
    days = sorted({d for d in days if d >= FIRST_DAY and not is_holdout_day(COUNTRY, d)})
    _, _, ordering_id, lambda2 = exacta_run._fits()  # #115's lambda2; #114's p is not read
    meta = {
        "lambda2_full_development": lambda2,
        "win_fit": WIN_SOURCE,
        "ordering_fit": ordering_id,
        "dropped_before_scoring": {},
        "diagnostics": {},
    }
    read: dict[str, pd.DataFrame] = {}
    out: list[dict] = []
    if days:
        in_days = ds.field("race_date").isin(pa.array(days, pa.date32()))
        lo, hi = days[0], days[-1]
        races = pool_depth._scan("races", RACE_COLUMNS, lo, hi, where=in_days).to_pandas()
        read["races"] = races
        if not races.empty:
            out = _assemble(races, in_days, lo, hi, read, meta)
    names = {
        "races": "jravan_races",
        "final_odds": "jravan_final_odds",
        "pool_votes": "jravan_pool_votes",
        "dividends": "jravan_dividends",
    }
    builds = {}
    for table, name in names.items():
        frame = read.get(table, pd.DataFrame())
        builds[name] = f"processed/jravan/{table}.parquet@{exacta_run._fingerprint(frame)}"
    meta["data_builds"] = builds
    return out, meta


def _assemble(races, where, lo, hi, read, meta) -> list[dict]:
    confirmed = pool_depth._confirmed
    odds = pool_depth._scan(
        "final_odds",
        ["race_id", "pool", "combination", "odds"],
        lo,
        hi,
        where=confirmed(
            where
            & ds.field("pool").isin(list(BOARD_POOLS))
            & ds.field("odds").is_valid()
            & (ds.field("combination") != POOL_ROW)
        ),
    ).to_pandas()
    votes = pool_depth._scan(
        "pool_votes",
        ["race_id", "combination", "votes", "refund_votes"],
        lo,
        hi,
        where=confirmed(where & (ds.field("pool") == "FORECAST")),
    ).to_pandas()
    ids = sorted(set(races["race_id"]))
    dividends = pq.read_table(
        paths.processed_dir(SOURCE) / "dividends.parquet",
        columns=["race_id", "combination", "dividend", "is_refund"],
        filters=_ids_filter(ids) & (ds.field("pool") == "FORECAST"),
    ).to_pandas()
    read.update(final_odds=odds, pool_votes=votes, dividends=dividends)
    odds = odds.loc[odds["odds"].to_numpy(dtype=float) > 0]
    boards = {
        pool: _boards(odds.loc[odds["pool"].to_numpy() == pool], "odds") for pool in BOARD_POOLS
    }
    is_pool = votes["combination"].to_numpy() == POOL_ROW
    pools = votes.loc[is_pool]
    net = UNIT * (
        pools["votes"].to_numpy(dtype=float)
        - np.nan_to_num(pools["refund_votes"].to_numpy(dtype=float))
    )
    pool_yen = dict(zip(pools["race_id"], net, strict=True))
    # A scratch after sales refunds the exacta tickets on the horse: the pool row carries the
    # refund votes while the dividends (JRA-VAN writes no refund row) still pay as usual.
    refund_votes = np.nan_to_num(pools["refund_votes"].to_numpy(dtype=float), nan=0.0)
    refunded = set(pools["race_id"].to_numpy()[refund_votes > 0])
    slots = votes.loc[~is_pool].assign(yen=lambda f: UNIT * f["votes"].astype(float))
    money = _boards(slots.loc[slots["yen"].notna()], "yen")
    paid: dict[str, list] = {}
    for r in dividends.itertuples(index=False):
        combo = tuple(int(h) for h in str(r.combination).split(","))
        refund = bool(r.is_refund) or r.race_id in refunded
        paid.setdefault(r.race_id, []).append((combo, float(r.dividend), refund))
    dropped, diagnostics = Counter(), Counter()
    out = []
    for race in races.itertuples(index=False):
        rid = race.race_id
        field = {h: v for h, v in boards["WIN"].get(rid, {}).items() if math.isfinite(v) and v > 0}
        if race.status != "run":
            dropped["status_not_run"] += 1
        elif race.race_class not in study.CLASS_GROUPS:
            dropped["unknown_race_class"] += 1
        elif not field:
            dropped["no_final_win_board"] += 1
        else:
            try:
                t = takeout.race_takeout(rid, "FORECAST", race_class_raw=race.race_class_raw)
                tq = takeout.race_takeout(rid, "QUINELLA", race_class_raw=race.race_class_raw)
            except (takeout.TakeoutUnknown, ValueError):
                dropped["takeout_unknown"] += 1
                continue
            diagnostics["exacta_pool_refund_votes"] += int(rid in refunded)
            p = exacta_run._board_shares(field)
            out.append(
                {
                    "race_id": rid,
                    "race_date": race.race_date,
                    "meeting_id": race.meeting_id,
                    "race_no": int(race.race_no),
                    "race_class": race.race_class,
                    "race_class_raw": race.race_class_raw,
                    "win_odds": field,
                    "p": p,
                    "rate": t.payout_rate,
                    "quinella_rate": tq.payout_rate,
                    "boost_may_be_withheld": t.boost_may_be_withheld,
                    "board": boards["FORECAST"].get(rid, {}),
                    "quinella_board": boards["QUINELLA"].get(rid, {}),
                    "pool_yen": pool_yen.get(rid, math.nan),
                    "money_yen": money.get(rid, {}),
                    "exacta": paid.get(rid, []),
                }
            )
    meta["dropped_before_scoring"] = dict(dropped)
    meta["diagnostics"] = dict(diagnostics)
    return out


# --------------------------------------------------------------------------- the run


class _Bets:
    """One member's bets: per race with a bet, its meeting, staked, returned and rate."""

    def __init__(self) -> None:
        self.meeting: list[str] = []
        self.staked: list[float] = []
        self.returned: list[float] = []
        self.rate: list[float] = []
        self.n_selected = 0


class _Ratio:
    """A descriptive ratio's per-race numerator and denominator, with the race's meeting."""

    def __init__(self) -> None:
        self.meeting: list[str] = []
        self.num: list[float] = []
        self.den: list[float] = []


@dataclass(frozen=True)
class CeilingRow:
    family_id: str
    lambda2: str
    category: str
    cut: float
    cell: str
    rho: float
    upper: float  # one-sided 95% meeting-bootstrap upper bound (the CI column)
    lower: float  # one-sided 95% lower bound
    return_per_100: float
    return_per_100_upper: float
    n_selected: int
    n_races: int
    n_meetings: int
    run: str = ""  # the commit of the run the row comes from (a v1 row: its recorded run)
    unavailable: str = ""  # why the row has no figure: a v1 member with no scored record

    @property
    def p_source(self) -> str:
        return P_SOURCE[self.family_id]


@dataclass(frozen=True)
class CeilingResult:
    rows: tuple[CeilingRow, ...]  # FAMILY_ID's 16 members in rank order
    variant_rows: tuple[CeilingRow, ...]  # PLAIN_ID's 16, in rank order
    # every member of every member family ranked together: this run's 32 and the v1 families'
    # latest scored rows in the registry (labelled by p source and span), a v1 member with no
    # scored record as an unavailable row, ranked last
    ranked: tuple[CeilingRow, ...]
    descriptives: tuple[dict, ...]
    excluded: dict
    meta: dict
    report_path: Path
    cells_path: Path | None  # always None: a run never writes CELLS_FILE, ``record_cells`` does
    run_ids: tuple[str, ...]
    variants_searched: int  # the variants actually scored, of VARIANTS_SEARCHED frozen
    export_path: Path | None = None  # this run's own ranked export, beside its report


def _check_frozen(registry: Registry) -> None:
    for family_id, spec in family_specs().items():
        frozen = registry.frozen_spec_hash(family_id)
        if frozen is None:
            raise RunRefused(f"{family_id} is not frozen; run `exacta-ceiling freeze` first")
        if frozen != _spec_hash(spec):
            raise RunRefused(f"{family_id} is frozen with another spec ({frozen[:12]})")


def _days(date_from: date, date_to: date) -> list[date]:
    start = max(date_from, FIRST_DAY)
    span = [start + timedelta(days=k) for k in range((date_to - start).days + 1)]
    return [d for d in span if not is_holdout_day(COUNTRY, d)]


def _merge_builds(parts: list[dict[str, str]]) -> dict[str, str]:
    """One "<identity>@<digest>" per build name over every chunk's builds, in read order."""
    out = {}
    for name in sorted({n for part in parts for n in part}):
        values = [part[name] for part in parts if name in part]
        identities = {v.rpartition("@")[0] or v for v in values}
        if len(identities) != 1:
            raise ValueError(f"the chunks read {name} from {sorted(identities)}")
        digest = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:16]
        out[name] = f"{identities.pop()}@{digest}"
    return out


def _score_race(race, lambdas, bets, ratios, excluded) -> bool:
    odds = race["win_odds"]
    exacta = list(race["exacta"])
    reason = study.exclusion_reason(
        n_starters=len(odds),
        exacta=exacta,
        boost_may_be_withheld=bool(race["boost_may_be_withheld"]),
    )
    if reason is not None:
        excluded[reason] += 1
        return False
    fav = favourite(odds)
    try:
        cells = study.segment_cells(
            fav_odds=odds[fav],
            field_size=len(odds),
            race_class=race["race_class"],
            race_no=int(race["race_no"]),
        )
    except ValueError:
        excluded["segments_undefined"] += 1
        return False
    (winner, dividend, _), rate = exacta[0], float(race["rate"])
    winner, pool, money = tuple(winner), float(race["pool_yen"]), race["money_yen"]
    index = {h: k for k, h in enumerate(sorted(race["p"]))}
    priced = _priced(race["board"], index, fav, rate)
    if not priced.combos:
        excluded["no_priced_exacta_board"] += 1
        return False
    if not pool > 0:
        excluded["no_exacta_pool_total"] += 1
        return False
    if not money.get(winner, 0) > 0:
        excluded["no_money_on_the_paying_combination"] += 1
        return False
    scopes = [s for s in CELLS if s in cells]
    meeting = race["meeting_id"]
    for family, (key, lam) in lambdas.items():
        _, model = _model(race["p"], lam)
        p_c = model[priced.a, priced.b]
        with np.errstate(invalid="ignore"):
            r = priced.q / p_c
        for category in RULE_CATEGORIES:
            code = _rule_code(category)
            for cut in CUTS:
                chosen = np.flatnonzero((priced.category == code) & (r < cut))
                if not chosen.size:
                    continue
                selected = [priced.combos[k] for k in chosen]
                staked, ret = race_return(
                    selected,
                    winner=winner,
                    dividend=dividend,
                    pool_yen=pool,
                    money_yen=money,
                    stake=STAKE,
                    rate=rate,
                )
                for scope in scopes:
                    b = bets[(family, (category, cut, scope))]
                    b.meeting.append(meeting)
                    b.staked.append(staked)
                    b.returned.append(ret)
                    b.rate.append(rate)
                    b.n_selected += len(selected)
        known = np.isfinite(priced.q)
        split = _split(
            race["quinella_board"],
            race["board"],
            index,
            model,
            fav,
            float(race["quinella_rate"]),
            rate,
        )
        for code, category in enumerate(CATEGORIES):
            d = known & (priced.category == code)
            s = split[0] == code
            for scope in scopes:
                for kind, mask, num, den in (
                    ("d", d, priced.q, p_c),
                    ("quinella_order_split", s, split[1], split[2]),
                ):
                    if mask.any():
                        ratio = ratios[(kind, key, category, scope)]
                        ratio.meeting.append(meeting)
                        ratio.num.append(float(num[mask].sum()))
                        ratio.den.append(float(den[mask].sum()))
    return True


def _interval(num, den, meetings, seed: int, level: float) -> tuple[float, float, float, int]:
    """(estimate, lower, upper, meetings): one-sided bounds at ``level`` from #131's meeting
    bootstrap; NaN bounds with fewer than two meetings."""
    up = study.rho_interval(
        num, den, meetings, alternative="less", n_boot=N_BOOT, seed=seed, level=level
    )
    if up.n_meetings < 2:
        return up.rho, math.nan, math.nan, up.n_meetings
    lo = study.rho_interval(
        num, den, meetings, alternative="greater", n_boot=N_BOOT, seed=seed, level=level
    )
    return up.rho, lo.bound, up.bound, up.n_meetings


def _row(family: str, label: str, member: tuple, b: _Bets, seed: int, run: str) -> CeilingRow:
    category, cut, cell = member
    nan = math.nan
    if not b.staked:
        return CeilingRow(family, label, category, cut, cell, nan, nan, nan, nan, nan, 0, 0, 0, run)
    staked = np.asarray(b.staked)
    rho, lower, upper, meetings = _interval(
        b.returned, staked * np.asarray(b.rate), b.meeting, seed, LEVEL
    )
    r100, _, r100_up, _ = _interval(b.returned, staked, b.meeting, seed, LEVEL)
    return CeilingRow(
        family,
        label,
        category,
        cut,
        cell,
        rho,
        upper,
        lower,
        100 * r100,
        100 * r100_up,
        b.n_selected,
        len(b.staked),
        meetings,
        run,
    )


def _value(metric: FoldMetric | None, attr: str = "value") -> float:
    found = None if metric is None else getattr(metric, attr)
    return math.nan if found is None else float(found)


def _superseded_rows(registry: Registry) -> list[CeilingRow]:
    """Every member of the v1 families as a row, in spec order: its latest scored record in the
    registry (rho, its bounds, the return per 100 yen and the counts, as ``_member_metrics`` wrote
    them), or, with no scored record, an unavailable row (no figure, ranked last, not counted)."""
    latest: dict[tuple[str, str], RunRecord] = {}
    for record in registry.records():
        if record.family_id in V1_MEMBER_FAMILIES and record.status == "scored":
            latest[(record.family_id, record.variant)] = record
    specs, nan = superseded_specs(), math.nan
    rows = []
    for family in V1_MEMBER_FAMILIES:
        for category, cut, cell in MEMBERS:
            record = latest.get((family, f"{category}|k={cut}|{cell}"))
            if record is None:
                label = str(specs[family]["lambda2"])  # as the v1 run labelled it
                nothing = (nan, nan, nan, nan, nan, 0, 0, 0)
                rows.append(
                    CeilingRow(family, label, category, cut, cell, *nothing, unavailable=NO_RECORD)
                )
            else:
                rows.append(_recorded_row(family, record))
    return rows


def _recorded_row(family: str, record: RunRecord) -> CeilingRow:
    m = {x.metric: x for x in record.metrics}
    rho, r100 = m.get("rho"), m.get("return_per_100")
    p = record.params
    return CeilingRow(
        family,
        str(p["lambda2"]),  # "full_development" or "1.0", as the run labels them
        str(p["category"]),
        float(p["cut"]),
        str(p["cell"]),
        _value(rho),
        _value(rho, "hi"),
        _value(rho, "lo"),
        _value(r100),
        _value(r100, "hi"),
        int(_value(m.get("n_selected"))) if "n_selected" in m else 0,
        int(rho.n) if rho is not None else 0,
        int(_value(m.get("n_meetings"))) if "n_meetings" in m else 0,
        record.git_sha[:12],
    )


def _ranked(rows) -> tuple[CeilingRow, ...]:
    """rho descending; a member without a bet after every member with one, and an unavailable
    row (no scored record) after both."""

    def key(item):
        k, row = item
        good = row.rho is not None and math.isfinite(row.rho)
        return (bool(row.unavailable), not good, -row.rho if good else 0.0, k)

    return tuple(row for _, row in sorted(enumerate(rows), key=key))


def _available(value) -> bool:
    return value is not None and math.isfinite(value)


def _member_metrics(row: CeilingRow) -> tuple[FoldMetric, ...]:
    fold, n = "development", row.n_races
    if not n:
        return (FoldMetric(fold, f"rho:unavailable:{NO_BET}", 0.0, None, None, 0),)
    out = []
    if _available(row.upper):
        out += [
            FoldMetric(fold, "rho", row.rho, row.lower, row.upper, n),
            FoldMetric(
                fold, "return_per_100", row.return_per_100, None, row.return_per_100_upper, n
            ),
        ]
    else:
        out += [
            FoldMetric(fold, "rho", row.rho, None, None, n),
            FoldMetric(fold, "rho_bounds:unavailable:" + FEW_MEETINGS, 0.0, None, None, n),
            FoldMetric(fold, "return_per_100", row.return_per_100, None, None, n),
        ]
    out += [
        FoldMetric(fold, "n_selected", float(row.n_selected), None, None, n),
        FoldMetric(fold, "n_meetings", float(row.n_meetings), None, None, n),
    ]
    return tuple(out)


def _descriptives(ratios, labels, seed: int) -> list[dict]:
    out = []
    for kind in ("d", "quinella_order_split"):
        for key in labels:
            for category in CATEGORIES:
                for cell in CELLS:
                    r = ratios.get((kind, key, category, cell))
                    row = {
                        "descriptive": kind,
                        "lambda2": key,
                        "category": category,
                        "cell": cell,
                        "value": math.nan,
                        "lo": math.nan,
                        "hi": math.nan,
                        "n_races": 0,
                        "n_meetings": 0,
                    }
                    if r is not None and r.num:
                        v, lo, hi, m = _interval(r.num, r.den, r.meeting, seed, 0.975)
                        row.update(value=v, lo=lo, hi=hi, n_races=len(r.num), n_meetings=m)
                    out.append(row)
    return out


def _descriptive_metrics(rows: list[dict]) -> tuple[FoldMetric, ...]:
    out = []
    for r in rows:
        fold, name = f"development:{r['category']}:{r['cell']}", r["descriptive"]
        if not r["n_races"]:
            out.append(FoldMetric(fold, f"{name}:unavailable:{EMPTY}", 0.0, None, None, 0))
        elif _available(r["lo"]):
            out.append(FoldMetric(fold, name, r["value"], r["lo"], r["hi"], r["n_races"]))
        else:
            out.append(FoldMetric(fold, name, r["value"], None, None, r["n_races"]))
    return tuple(out)


def run_ceiling(
    *,
    date_from: date,
    date_to: date,
    out_dir: Path,
    seed: int,
    registry: Registry | None = None,
) -> CeilingResult:
    """Score every frozen member on the development days of [max(date_from, 2014-06-07),
    date_to]; one registry record per member and per descriptive (a failure too), and the
    private report and this run's own export under ``out_dir`` (named by its commit, data build,
    seed and asked days, carrying the days it scored, flagged ``run_of_record`` by
    ``_is_run_of_record``). No run writes the ledger-linked ``CELLS_FILE``: ``record_cells`` does,
    from the run of record's export. With the default registry, the committed summary of this
    study's families is regenerated."""
    default = registry is None
    registry = default_registry(COUNTRY) if default else registry
    _check_frozen(registry)
    out_dir = Path(out_dir)
    sha = _git_sha()
    if not sha:  # the registry takes no record without a commit: refuse before any read
        raise RunRefused("no git commit to record the run under")
    specs = family_specs()
    lambdas: dict[str, tuple[str, float]] = {}
    labels = {FAMILY_ID: "full_development", PLAIN_ID: "1.0"}
    bets = {(f, m): _Bets() for f in MEMBER_FAMILIES for m in MEMBERS}
    ratios: defaultdict[tuple, _Ratio] = defaultdict(_Ratio)
    excluded: Counter = Counter()
    builds: list[dict[str, str]] = []
    first = {"meta": None}
    try:
        days = _days(date_from, date_to)
        asked = set(days)
        scored_days = []
        for year in sorted({d.year for d in days}):
            races, meta = load_races([d for d in days if d.year == year])
            builds.append(dict(meta["data_builds"]))
            if first["meta"] is None:
                first["meta"] = meta
                lam = float(meta["lambda2_full_development"])
                lambdas.update({FAMILY_ID: ("full_development", lam), PLAIN_ID: ("1.0", 1.0)})
            for key in ("lambda2_full_development", "win_fit", "ordering_fit"):
                if meta[key] != first["meta"][key]:
                    raise ValueError(f"the chunks disagree on {key}")
            for name, n in meta.get("dropped_before_scoring", {}).items():
                excluded[f"before_scoring:{name}"] += n
            for name, n in meta.get("diagnostics", {}).items():
                excluded[f"diagnostic:{name}"] += n
            for race in races:
                if race["race_date"] not in asked:  # never expected: the seam returns asked days
                    excluded["outside_the_asked_days"] += 1
                    continue
                if _score_race(race, lambdas, bets, ratios, excluded):
                    scored_days.append(race["race_date"])
            del races
        meta = dict(first["meta"] or {"win_fit": "none", "ordering_fit": "none"})
        meta["data_builds"] = _merge_builds(builds)
        if not scored_days:  # nothing to rank: no scored record, no report, no cells file
            raise NoEligibleRace(f"{NO_RACE} ({date_from} to {date_to})")
        run = sha[:12]
        groups = {
            f: [_row(f, labels[f], m, bets[(f, m)], seed, run) for m in MEMBERS]
            for f in MEMBER_FAMILIES
        }
        descriptives = _descriptives(ratios, list(labels.values()), seed)
        rows, variant_rows = _ranked(groups[FAMILY_ID]), _ranked(groups[PLAIN_ID])
        ranked = _ranked([*groups[FAMILY_ID], *groups[PLAIN_ID], *_superseded_rows(registry)])
        span = (min(scored_days), max(scored_days))
        window = (max(date_from, FIRST_DAY), date_to)  # the asked days, as `_days` clamps them
        report_path, export_path = _write_private(
            out_dir, sha, meta, ranked, descriptives, dict(excluded), span, window, seed
        )
    except RunRefused:
        raise
    except Exception as exc:
        meta = dict(first["meta"] or {"win_fit": "none", "ordering_fit": "none"})
        meta["data_builds"] = _merge_builds(builds) if builds else {}
        reason = NO_RACE if isinstance(exc, NoEligibleRace) else type(exc).__name__
        marker = (FoldMetric("development", f"run:unavailable:{reason}", 0.0, None, None, 0),)
        _append(
            registry,
            [
                _record(f, m, marker, meta, sha, specs, "failed", seed)
                for f in MEMBER_FAMILIES
                for m in MEMBERS
            ],
        )
        raise
    records = [
        _record(
            row.family_id,
            (row.category, row.cut, row.cell),
            _member_metrics(row),
            meta,
            sha,
            specs,
            "scored",
            seed,
        )
        for row in sorted((*rows, *variant_rows), key=_spec_order)  # never in rank order
    ]
    for kind in ("d", "quinella_order_split"):
        for label in labels.values():
            part = [r for r in descriptives if r["descriptive"] == kind and r["lambda2"] == label]
            records.append(_descriptive_record(kind, label, part, meta, sha, specs, seed))
    run_ids = _append(registry, records)
    if default:
        write_committed_summary(registry)
    return CeilingResult(
        rows,
        variant_rows,
        ranked,
        tuple(descriptives),
        dict(excluded),
        meta,
        report_path,
        None,  # cells_path: a run never writes CELLS_FILE
        tuple(run_ids),
        sum(not row.unavailable for row in ranked),
        export_path,
    )


def _components(family: str, meta: dict) -> dict[str, str]:
    ordering = meta["ordering_fit"] if family == FAMILY_ID else "plain_harville"
    return {"win": str(meta["win_fit"]), "ordering": str(ordering)}


def _record(family, member, metrics, meta, sha, specs, status, seed: int) -> RunRecord:
    category, cut, cell = member
    params = {
        "category": category,
        "cut": cut,
        "cell": cell,
        "selection": "look-ahead",
        "lambda2": specs[family]["lambda2"],
        "stake_yen": 100,
        "seed": seed,
        "n_boot": N_BOOT,
        "family_spec_hash": _spec_hash(specs[family]),
    }
    return RunRecord(
        HYPOTHESIS_ID,
        family,
        f"{category}|k={cut}|{cell}",
        TICKET,
        sha,
        dict(meta["data_builds"]),
        SOURCES,
        FOLD_SCHEME,
        _components(family, meta),
        params,
        tuple(metrics),
        status,
    )


def _descriptive_record(kind, label, rows, meta, sha, specs, seed: int) -> RunRecord:
    params = {
        "descriptive": kind,
        "lambda2": label,
        "selection": "look-ahead",
        "seed": seed,
        "n_boot": N_BOOT,
        "family_spec_hash": _spec_hash(specs[DESCRIPTIVE_ID]),
    }
    family = FAMILY_ID if label == "full_development" else PLAIN_ID
    return RunRecord(
        HYPOTHESIS_ID,
        DESCRIPTIVE_ID,
        f"{kind}|lambda2={label}",
        TICKET,
        sha,
        dict(meta["data_builds"]),
        SOURCES,
        FOLD_SCHEME,
        _components(family, meta),
        params,
        _descriptive_metrics(rows),
        "scored",
    )


def _append(registry: Registry, records: list[RunRecord]) -> list[str]:
    """Append each record not already in the registry (a rerun of the same commit and data
    reuses its records); the run ids in order."""
    held = {_sha256(_canonical(_record_data(r))) for r in registry.records()}
    ids = []
    for record in records:
        run_id = _sha256(_canonical(_record_data(record)))
        if run_id not in held:
            registry.append(record)
            held.add(run_id)
        ids.append(run_id)
    return ids


# --------------------------------------------------------------------------- outputs


def _num(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if _available(value) else "n/a"


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def _status(row: CeilingRow) -> str:
    if row.unavailable:
        return f"unavailable: {row.unavailable}"
    return "scored" if _available(row.rho) else f"scored, {NO_BET}"


def _build_digest(builds: Mapping) -> str:
    """A run's data-build identity: the sha256 of its merged builds as sorted JSON."""
    return hashlib.sha256(json.dumps(builds, sort_keys=True).encode()).hexdigest()


def _stem(commit: str, digest: str, seed: int, window) -> str:
    """A run's report and export name: its commit, data build, seed and asked days, so no other
    run (another revision, seed or span) ever overwrites them."""
    days = f"{window[0]:%Y%m%d}-{window[1]:%Y%m%d}"
    return f"exacta-ceiling-japan-{commit[:12]}-{digest[:8]}-seed{seed}-{days}"


def _is_run_of_record(commit: str, builds: Mapping, seed: int, window) -> bool:
    """The recorded commit, data build, seed and asked days, read now; never when the record or
    the run lacks a commit or a data build (two blanks or two empty build maps are no record)."""
    if not (RECORD_COMMIT and RECORD_BUILD and commit and builds):
        return False
    found = (commit, _build_digest(builds), seed, tuple(window))
    return found == (RECORD_COMMIT, RECORD_BUILD, DEFAULT_SEED, tuple(RECORD_DAYS))


def _write_private(out_dir, sha, meta, ranked, descriptives, excluded, span, window, seed):
    """The report and this run's export (``<stem>.md``, ``<stem>.json``); (report, export).
    Never ``CELLS_FILE``: ``record_cells`` writes it from the run of record's export."""
    specs = family_specs()
    hashes = {f: _spec_hash(s) for f, s in specs.items()}
    digest = _build_digest(meta["data_builds"])
    tag = digest[:8]
    stem = _stem(sha, digest, seed, window)
    of_record = _is_run_of_record(sha, meta["data_builds"], seed, window)
    searched = sum(not r.unavailable for r in ranked)  # only the variants actually scored
    scored = f"{span[0]} to {span[1]}"

    def days(r: CeilingRow) -> str:
        if r.family_id in MEMBER_FAMILIES:
            return scored  # the first and last development day this run scored
        if r.unavailable:
            return "none (no scored record)"
        return f"the days scored by run {r.run} (in that run's private report)"

    leaderboard = [
        {
            "rank": k + 1,
            "option": f"{r.family_id}:{r.category}|k={r.cut}|{r.cell}",
            **asdict(r),
            "status": _status(r),
            "p_source": r.p_source,
            "p": P_SOURCES[r.p_source],
            "eligible": SPANS[r.p_source],
            "span": days(r),
            "label": "look-ahead",
            "family_size": len(MEMBERS),  # the members of the row's own family
            "variants_searched": searched,  # the variants this export actually scored
            "variants_frozen": VARIANTS_SEARCHED,  # every member family of the study
            "export": stem,
            "run_of_record": of_record,
            "seed": seed if r.family_id in MEMBER_FAMILIES else "its run's recorded seed",
            "break_even_rho": "1 / rate (rho = 1 is the payout rate)",
        }
        for k, r in enumerate(ranked)
    ]
    missing = len(ranked) - searched
    lines = [
        f"# N2 (#431): exacta look-ahead ceiling on final boards ({sha[:12]}, seed {seed})",
        "",
        f"**{LOOK_AHEAD}.** Not a betting rule and not a proven bound on the T-2 rule.",
        "",
        f"- Run `{stem}`: asked {window[0]} to {window[1]}, seed {seed}; "
        + (
            f"the run of record (`exacta-ceiling record` writes `{CELLS_FILE.as_posix()}` from "
            f"its export `{stem}.json`)"
            if of_record
            else f"not the run of record (commit `{RECORD_COMMIT[:12]}`, data build "
            f"`{RECORD_BUILD[:8]}`, seed {DEFAULT_SEED}, asked {RECORD_DAYS[0]} to "
            f"{RECORD_DAYS[1]}): `{CELLS_FILE.as_posix()}` is the run of record's alone, this "
            f"run's export is `{stem}.json`"
        ),
        f"- Span scored by this run (p source `board`): {scored}",
        *[f"- p source `{k}`: {v}; eligible: {SPANS[k]}" for k, v in P_SOURCES.items()],
        f"- Seed {seed}, {N_BOOT} meeting-bootstrap replicates, one-sided {LEVEL:.0%} bounds",
        f"- Variants searched: {searched} scored of the {VARIANTS_SEARCHED} frozen "
        f"({len(MEMBERS)} members in each member family of the study: this run's "
        f"{', '.join(MEMBER_FAMILIES)} and the superseded {', '.join(V1_MEMBER_FAMILIES)}, whose "
        "rows come from their recorded runs)"
        + (
            f"; {missing} superseded members have no scored record in this registry: unavailable "
            "rows, ranked last, not counted"
            if missing
            else ""
        ),
        f"- Win probabilities `{meta['win_fit']}`, ordering fit `{meta['ordering_fit']}`, "
        f"lambda2 full_development = {meta.get('lambda2_full_development')}",
        *[f"- Family `{f}`, spec sha256 `{h}`" for f, h in hashes.items()],
        *[f"- Build `{k}`: `{v}`" for k, v in meta["data_builds"].items()],
        "",
        "## Ranked members (rho = return over staked x rate; the CI column is the one-sided 95% "
        "upper bound; family size = the row's family, variants searched = the variants this run "
        "scored, of the study's frozen)",
        "",
        "| rank | family | p source | run | span | status | lambda2 | category | cut | cell | rho "
        "| upper (CI) | lower | return per 100 yen | its upper | n selected | n races | "
        "n meetings | family size | variants searched |",
        "|" + " --- |" * 20,
    ]
    for row in leaderboard:
        lines.append(
            f"| {row['rank']} | {row['family_id']} | {row['p_source']} | {row['run'] or 'none'} | "
            f"{row['span']} | {row['status']} | "
            f"{row['lambda2']} | {row['category']} | {row['cut']} | {row['cell']} | "
            f"{_num(row['rho'])} | {_num(row['upper'])} | {_num(row['lower'])} | "
            f"{_num(row['return_per_100'], 1)} | {_num(row['return_per_100_upper'], 1)} | "
            f"{row['n_selected']} | {row['n_races']} | {row['n_meetings']} | "
            f"{row['family_size']} | {row['variants_searched']} of {row['variants_frozen']} |"
        )
    lines += [
        "",
        "## Descriptives (two-sided 95% meeting-bootstrap intervals)",
        "",
        "| descriptive | lambda2 | category | cell | value | lo | hi | n races | n meetings |",
        "|" + " --- |" * 9,
    ]
    for r in descriptives:
        lines.append(
            f"| {r['descriptive']} | {r['lambda2']} | {r['category']} | {r['cell']} | "
            f"{_num(r['value'])} | {_num(r['lo'])} | {_num(r['hi'])} | {r['n_races']} | "
            f"{r['n_meetings']} |"
        )
    lines += ["", "## Exclusions and diagnostics (races)", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(excluded.items())] or ["- none"]
    text = "\n".join(lines) + "\n"
    report = write_report(out_dir / f"{stem}.md", text, sources=SOURCES)
    payload = {
        "run": {
            "export": stem,
            "commit": sha,
            "data_build": tag,
            "data_build_digest": digest,
            "seed": seed,
            "asked": window,
            "scored": span,
            "run_of_record": of_record,
        },
        "meta": meta,
        "span": span,
        "seed": seed,
        "family_size": len(MEMBERS),
        "variants_searched": searched,
        "variants_frozen": VARIANTS_SEARCHED,
        "unavailable": missing,
        "leaderboard": leaderboard,
        "descriptives": descriptives,
        "excluded": excluded,
        "spec_hashes": hashes,
        "superseded_spec_hashes": {f: _spec_hash(s) for f, s in superseded_specs().items()},
    }
    export = write_report(
        out_dir / f"{stem}.json", json.dumps(_json_safe(payload), indent=1) + "\n", sources=SOURCES
    )
    return report, export


def record_cells(out_dir: Path) -> Path:
    """Write ``out_dir / CELLS_FILE``, the ledger-linked ranked member file, as the leaderboard of
    the run of record's export (``RECORD_COMMIT``, ``RECORD_BUILD``, ``DEFAULT_SEED``,
    ``RECORD_DAYS``, read now) and return its path. Everything is checked before anything is
    written: no record, no export, or an export whose commit, data build (recomputed from its
    builds, never its digest field), seeds or asked days differ raises ``RunRefused`` and writes
    nothing. It appends no registry record."""
    out_dir = Path(out_dir)
    if not (RECORD_COMMIT and RECORD_BUILD):
        raise RunRefused("the run of record names no commit or no data build")
    path = out_dir / f"{_stem(RECORD_COMMIT, RECORD_BUILD, DEFAULT_SEED, RECORD_DAYS)}.json"
    if not path.is_file():
        raise RunRefused(f"no export of the run of record at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RunRefused(f"{path.name} is not JSON: {exc}") from exc
    run = payload.get("run") if isinstance(payload, dict) else None
    meta = payload.get("meta") if isinstance(payload, dict) else None
    if not isinstance(run, dict) or not isinstance(meta, dict):
        raise RunRefused(f"{path.name} has no run or meta block")
    builds, leaderboard = meta.get("data_builds"), payload.get("leaderboard")
    if not run.get("commit") or not isinstance(builds, dict) or not builds:
        raise RunRefused(f"{path.name} names no commit or no data build")
    if not isinstance(leaderboard, list) or not leaderboard:
        raise RunRefused(f"{path.name} has no ranked members")
    checks = (
        ("commit", run["commit"], RECORD_COMMIT),
        ("data build", _build_digest(builds), RECORD_BUILD),
        ("seed", run.get("seed"), DEFAULT_SEED),
        ("export seed", payload.get("seed"), DEFAULT_SEED),
        ("asked days", run.get("asked"), [d.isoformat() for d in RECORD_DAYS]),
    )
    for name, found, want in checks:
        if found != want:
            raise RunRefused(f"{path.name}: its {name} is {found!r}, the run of record's {want!r}")
    text = json.dumps(leaderboard, indent=1) + "\n"
    return write_report(out_dir / CELLS_FILE, text, sources=SOURCES)


class _Mine:
    """The registry's records of this study's families: what ``write_summary`` reads."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def records(self, family_id: str | None = None) -> list[RunRecord]:
        """In append order: ``write_summary`` lists licensed runs by identity, so a committed
        file never shows the order a run ranked its members in (D21)."""
        mine = {**superseded_specs(), **family_specs()}
        return [r for r in self._registry.records(family_id) if r.family_id in mine]


def _spec_order(row: CeilingRow) -> tuple[int, int]:
    return MEMBER_FAMILIES.index(row.family_id), MEMBERS.index((row.category, row.cut, row.cell))


_INDEX_ENTRY = re.compile(r"- \[([^\]]+)\]\(([^)]+)\.md\)")  # a family's line in the index


def write_committed_summary(registry: Registry) -> Path:
    """This study's family files by ``write_summary`` over their records alone, and the index
    with their lines added; no other family's file is written."""
    directory = summary_dir(COUNTRY)
    with tempfile.TemporaryDirectory() as scratch:
        write_summary(_Mine(registry), scratch)  # type: ignore[arg-type]
        for path in sorted(Path(scratch).glob("*.md")):
            if path.stem != "README":
                write_report(directory / path.name, path.read_text("utf-8"), sources=())
        fresh = Path(scratch, "README.md").read_text("utf-8").splitlines()
    index = directory / "README.md"
    listed = index.read_text("utf-8").splitlines() if index.exists() else []
    entries = {m[1]: line for line in listed + fresh if (m := _INDEX_ENTRY.fullmatch(line))}
    head = [line for line in fresh if not _INDEX_ENTRY.fullmatch(line)]
    write_report(
        index, "\n".join([*head, *(entries[f] for f in sorted(entries))]) + "\n", sources=()
    )
    return directory


PUBLIC_REPORT = Path("docs") / "experiments" / "exacta-ceiling-japan.md"


def render_public_report(result: CeilingResult) -> str:
    """The public report: what ran and where its figures are, and no figure of ``result`` (no
    rho, bound, count, rank or span): every figure is withheld (D21)."""
    del result  # nothing of the result is public
    hashes = "\n".join(f"  | `{f}` | `{_spec_hash(s)}` |" for f, s in family_specs().items())
    old = "\n".join(f"  | `{f}` | `{_spec_hash(s)}` |" for f, s in superseded_specs().items())
    members = ", ".join(f"`{c}|k={k}|{s}`" for c, k, s in MEMBERS)
    return f"""# Experiment report: N2, the exacta look-ahead ceiling on final boards (Japan, #431)

Ticket #431 (epic #72). **Every figure is {WITHHELD}**: the run reads the JRA-VAN store, a
licensed source, so this report says what ran and where its figures are, and gives no count, rate,
span, sign, bound or rank derived from that data. The selection is **look-ahead**: it reads the
final exacta and Win boards, which hold money placed after T-2. Nothing here is a betting rule or
a betting-edge claim, nothing is adopted, and the holdout is not opened.

- Registry families of the run (v2), each frozen with `Registry.freeze_family` before the run
  read any row (`uv run horsetrack japan exacta-ceiling freeze`; spec
  `docs/specs/exacta-ceiling-prereg.md`, code `exacta_ceiling.family_specs()`). Win
  probabilities: the normalised implied Win shares of the final Win board, as #131's E2
  rule-v5 on the T-2 board:

  | family | spec sha256 |
  | --- | --- |
{hashes}
- Superseded families (v1, `exacta_ceiling.superseded_specs()`): Win probabilities from #114's
  out-of-fold calibrated `p_cal` on the races with a #114 row, as #131's superseded rule-v4.
  Frozen and recorded; their members stay ranked variant rows, labelled with that p source and
  span; never executed again:

  | family | spec sha256 |
  | --- | --- |
{old}
- Command: `uv run horsetrack japan exacta-ceiling run --from 2014-06-07 --to <last day>`;
  development days only (`is_holdout_day`), none before 2014-06-07 (D10), every race with final
  Win and exacta boards; `LockedHoldout` is never opened. A race whose exacta pool has refund
  votes (a scratch after sales) is excluded and counted. A run that scores no race records its
  members as failed and writes no report.
- Private outputs, through `write_report` with the source `jravan`: each run's report and export
  `$HORSETRACK_DATA_DIR/reports/exacta-ceiling-japan-<commit>-<build>-seed<seed>-<from>-<to>.md`
  and `.json`, named by the run's commit, data build, seed and asked days so that no run
  overwrites another's (every ranked member with its p source, status, the days its run scored,
  family size and the variants searched; the descriptives and the exclusions); and the
  ledger-linked ranked member file `$HORSETRACK_DATA_DIR/reports/{CELLS_FILE.as_posix()}`, which
  no run writes: `uv run horsetrack japan exacta-ceiling record` writes it from the run of
  record's export (commit `{RECORD_COMMIT[:12]}`, data build `{RECORD_BUILD[:8]}`, seed
  {DEFAULT_SEED}, asked days {RECORD_DAYS[0]} to {RECORD_DAYS[1]}) and refuses every other.
- Variants frozen: {VARIANTS_SEARCHED}, every member family of the study: {len(MEMBERS)}
  members at #115's full-development lambda2 and the same {len(MEMBERS)} at plain Harville, for
  each p source (v2 and v1); family size {len(MEMBERS)} each. Each run's report and export state
  beside it the variants searched, the ones actually scored: the run's own and each v1 member
  with a scored record in the registry. A v1 member without one is an unavailable row, ranked
  last and not counted.

## Hypothesis

E43 as #131, read on the final boards (N2): some exacta combination category, fav_second or
no_fav, is underbet against the Win-derived discounted Harville probability far enough that
buying it when q / p < k returns more than break-even (rho above 1 / rate) at the final dividend.
The null, the starting point: the exacta pool prices each category like the Win pool, so every
member's rho is about 1 (below break-even, takeout and breakage in). Even a member above
break-even here is a ceiling, not a strategy: a selection on the final board is not a proven
upper bound on the T-2 rule (informed late money), a ledger assumption.

## Variants tried

The 16 members of #131's E2 unchanged: {members}; each at #115's full-development lambda2
(`{FAMILY_ID}`) and at plain Harville, lambda2 = 1 (`{PLAIN_ID}`), with p from the final Win
board's normalised implied shares; and the same 32 with p from #114's `p_cal` on the races with a
#114 row (`{V1_ID}`, `{V1_PLAIN_ID}`: superseded, recorded, ranked as variant rows). Each
member's look-ahead rho, its one-sided 95% meeting-bootstrap upper bound (the CI column), lower
bound, return per 100 yen and its upper bound, n selected, n races and n meetings: {WITHHELD}.
Descriptives (`{DESCRIPTIVE_ID}`, and v1's `{V1_DESCRIPTIVE_ID}`): D = sum q / sum p per category
and cell, and the Quinella x order split, each with a two-sided 95% meeting-bootstrap interval:
{WITHHELD}. A member with no bet keeps its row, ranked last, with an unavailable marker in the
registry; a v1 member with no scored record in the registry keeps an unavailable row, ranked after
those and not counted as searched; a failed run is recorded too (the registry summary lists every
run's status).

## Alternatives rejected and why

- A kill rule or an open/closed verdict: replaced by the owner's reframe (2026-11-27); every
  member is a ranked row and nothing closes or is parked on this result.
- The real lower tertile of the final exacta pool totals as the headline's "less popular": not
  added; #131's race-card proxy keeps the members comparable with E2 and the family small.
- #114's out-of-fold `p_cal` as the run's Win probabilities (v1): superseded by v2 because
  #131's frozen E2 is rule-v5, whose p_c(t) is the normalised implied Win shares of the Win
  board on every day with Win and exacta boards; v1's rows stay ranked (Sol review round 1 of
  24d080f).
- Own impact from the logged odds: the JRA-VAN final votes give each combination's money and the
  pool's net total directly, so the payout identity needs no odds inversion.

## What this rules out

{WITHHELD}: the private report gives each member's look-ahead rho with its one-sided 95% upper
bound, against break-even 1 / rate. Whatever it shows, the selection reads the final boards, so a
member's row is a look-ahead ceiling for the leaderboard, never a betting rule.

## Leaderboard rows

Option ids `<family>:<category>|k=<cut>|<cell>` for the 16 members above in each member family
(`{FAMILY_ID}`, `{PLAIN_ID}`, `{V1_ID}`, `{V1_PLAIN_ID}`), each a look-ahead row ranked by rho
and labelled with its p source and span; rank and figures {WITHHELD}.

## Follow-ups filed

- The same members on T-2 boards (#131's E2 v5) as #130-logged development days accrue, and in
  the holdout ranking check through #447's leaderboard.
"""


# --------------------------------------------------------------------------- the command


exacta_ceiling_app = typer.Typer(
    no_args_is_help=True, help="N2 (#431): the exacta look-ahead ceiling on final boards."
)


@exacta_ceiling_app.command("freeze")
def freeze_command() -> None:
    """Freeze every family's spec in the japan registry, the superseded v1 specs too
    (irreversible; again is a no-op)."""
    registry = default_registry(COUNTRY)
    for family_id, spec in {**superseded_specs(), **family_specs()}.items():
        typer.echo(f"frozen {family_id} {registry.freeze_family(family_id, spec)}")


@exacta_ceiling_app.command("run")
def run_command(
    date_to: str = typer.Option(..., "--to", help="Last day (YYYY-MM-DD)."),
    date_from: str = typer.Option("2014-06-07", "--from", help="First day (clamped to D10)."),
    seed: int = typer.Option(DEFAULT_SEED, help="Bootstrap seed."),
) -> None:
    """Score the frozen members on development days and record the run; prints no figure. A
    span with no eligible race records the members as failed and publishes nothing (exit 1)."""
    try:
        result = run_ceiling(
            date_from=date.fromisoformat(date_from),
            date_to=date.fromisoformat(date_to),
            out_dir=paths.reports_dir(),
            seed=seed,
        )
    except NoEligibleRace as exc:
        typer.echo(f"failed: {exc}; recorded, nothing published", err=True)
        raise typer.Exit(1) from exc
    public = write_report(env.repo_root() / PUBLIC_REPORT, render_public_report(result), sources=())
    typer.echo(
        f"report={result.report_path}\nexport={result.export_path}\n"
        f"cells=not written by a run; `exacta-ceiling record` writes {CELLS_FILE.as_posix()} from "
        f"the run of record's export\npublic={public}"
    )
    typer.echo(f"rows={len(result.rows)} ranked={len(result.ranked)} records={len(result.run_ids)}")


@exacta_ceiling_app.command("record")
def record_command() -> None:
    """Write the ledger-linked ranked member file from the run of record's export (the commit
    and data build of the ledger's N2 row); any other export, or none, writes nothing (exit 1)."""
    try:
        cells = record_cells(paths.reports_dir())
    except RunRefused as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"cells={cells}")
