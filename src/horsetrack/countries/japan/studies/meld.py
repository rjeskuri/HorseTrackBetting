"""The C4 meld of the out-of-fold members at the final Win price (#98, epic #73; 08 §3.2 C4 and
cross-cutting, §4 E41, E42): `python -m horsetrack.countries.japan.studies.meld --run [--out DIR]`.

Members (`MEMBERS`): #91's rating (logp, its sigma_before), #92's ability model (ln p_ability, its
se_u; JRA-VAN's sire, so licensed) and #93's six interaction members (ln of the softmax of each u
within the file's race: the base without the sire and its five blocks), read column- and
date-scoped from their exports under the data dir (`MEMBER_FILES`, each named by its sha256 in the
spec). #94's competition terms, #95's lineage members, #97's p_speed and E41's R_eff have no
out-of-fold export: each is a failed record saying so (`NOT_RUN`).

Level 2 is `studies.meta.store_frame`'s priced starters of development race days (holdout days
dropped by `read_store_table` and `split_off`; the holdout is never opened), each with `fold`, the
label of the D9 quarter (`core.eval.folds.walk_forward`, config/folds.yaml) that tests it.
`models.meld.member_frame` aligns every member and refuses a row whose oof_fold is not that label;
`models.stacking.join_members` and `check_level2` then check each member's provenance and each
scored fold's level-2 training window against D9's 5 seasons. Two populations (`POPULATIONS`):
every member from 2020-Q1 (rating's first out-of-fold quarter) and every member but rating from
2015-Q1. In each, the quarters before the first fold with 5 seasons of level-2 training form one
seed block that only trains; every later quarter is scored by fits on every earlier row at least
the scheme's embargo before its first race day (`models.meld.walk_forward_predictions`).

Every fit is `models.meld.fit_meld` in the stage-2 form (u = sum_k alpha_k m_k + beta ln q), ridge
`RIDGE` on every member term and none on the market's (the owner's target alpha = 0, 2026-11-24);
the comparator is the market-only stage-2 meta fitted the same way. Variants, per population: the
market only; each member standalone; the meld of all members and of all minus each member; each
member's marginal gain (the paired table of `ablation_table`, one set of meeting draws); each
member's encompassing coefficient (`models.meta.encompassing`, meeting-clustered); E41, the meld
with one precision member's alpha_i = alpha_0 / (1 + kappa sigma_i^2) at each kappa of the grid,
against the constant-alpha meld (paired ΔLL and Clark-West's adjusted squared-error difference);
E42, the meld gated on debut races, on handicaps and on both (each cell its own alpha_k), against
the pooled meld overall and per cell, with each cell's in-sample alpha; the members' correlations
and the Heskes ambiguity. Every model variant also carries N5's top claim band at the final price,
settled at the recorded Win dividends (`studies.interactions.win_paid`), as its leaderboard row.

Order: build and freeze the spec (`freeze_family`, before any row is read), read and check the
inputs, score every variant (`Study.attempt`: a failure is a failed record), write the report and
its JSON under the reports dir through `write_report` (licensed: the ability member reads
JRA-VAN), append the records (a record equal to a held one but for git_sha is not appended again)
and write the committed summary, which withholds every licensed record's values (D21). A failed
launch records every planned variant as failed. Nothing is adopted (D19: ranked, not adopted).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from horsetrack.core import paths
from horsetrack.core.config import load_source
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import Fold, FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, split_off
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.tail import tail_curve
from horsetrack.countries.japan.models import meld, stacking
from horsetrack.countries.japan.models.meta import Q, encompassing
from horsetrack.countries.japan.studies import ability
from horsetrack.countries.japan.studies.calibration import (
    Study,
    _markdown_table,
    _metric,
    bootstrap_settings,
    store_builds,
)
from horsetrack.countries.japan.studies.interactions import read_win_dividends, win_paid
from horsetrack.countries.japan.studies.meta import (
    RACE_COLUMNS,
    MetaVariant,
    _one_sided_z,
    _run_id,
    file_digest,
    read_development,
    restrict,
    store_frame,
)

COUNTRY = "japan"
TICKET = 98
FAMILY_ID = "japan-c4-meld-final-win"
STORE = ("jra",)
BOOTSTRAP_SEED = 98  # reproducibility only; scoring.yaml signs no seed
CLUSTER = "meeting_id"
MARKET_ONLY = "market_only"  # the comparator's variant: its predictions are meld.MARKET's
RIDGE = 1.0  # a N(0, 1) prior on each member term: decisive only where the races cannot inform it
KAPPA_GRID = (0.25, 1.0, 4.0)  # kappa * median(sigma^2): the weight at the median sigma
TAIL_QUANTILES = (0.5, 0.9, 0.99)  # N5's bands; the top band is the leaderboard row
D9_MIN_SEASONS = 5
GATES = {
    "debut": ("race_class", "DEBUT", "a newcomer race (race_class DEBUT)"),
    "handicap": ("weight_type", "handicap", "a handicap (weight_type handicap)"),
}
E42_GATES = (("debut",), ("handicap",), ("debut", "handicap"))
MEMBER_FILES = {  # export -> (file under features/japan, ticket, sources, study family)
    "rating": ("rating-7ec8931213c6.parquet", 91, ("jra",), "japan-c2-rating-final-win"),
    "ability": (
        "ability-japan-12b1eed01095-9a3f58d49fb3.parquet",
        92,
        ("jra", "jravan"),
        "japan-c2-ability-final-win-v2",
    ),
    "interactions": (
        "interactions-japan-2dde55836222-28bea4f0bfa3.parquet",
        93,
        ("jra",),
        "japan-c2-interactions-final-win-v2",
    ),
}
IX = (
    "base",
    "horse_x_jockey",
    "trainer_x_track",
    "horse_aptitude",
    "trainer_debut",
    "transfer_step",
)
MEMBERS = {  # member -> (export, column, transform, sigma column)
    "rating": ("rating", "logp", "as given", "sigma_before"),
    "ability": ("ability", "p_ability", "ln", "se_u"),
    **{f"ix_{b}": ("interactions", f"u_{b}", "log-softmax within race", None) for b in IX},
}
POPULATIONS = {
    "from2020": (date(2020, 1, 1), tuple(MEMBERS)),
    "from2015": (date(2015, 1, 1), tuple(m for m in MEMBERS if m != "rating")),
}
NOT_RUN = {
    "not_run:competition": "#94's competition terms (#531's study) export no out-of-fold member "
    "per runner, only its report: a member needs an export of its stage-1 u with oof_fold",
    "not_run:lineage": "#95's lineage members (#538's E37 study) export no out-of-fold member per "
    "runner; its level-1 grid takes about 1.5 hours to refit",
    "not_run:p_speed": "#97's p_speed (studies.p_speed) exports no out-of-fold member per runner; "
    "its HorseSpeed fit is the slowest of the stack",
    "e41:reff": "E41's R_eff weighting needs #94's race_comparability per scored race from each "
    "fold's comparison graph (models.competition), which no study exports",
}
DECISIONS = {
    "D1": "A: condition on the final price; every fit conditions on the final Win price",
    "D6": "winner log loss per race, a dead heat at 1/k per co-winner (#69 answer)",
    "D7": "meeting (venue x day) clusters, 95% one-sided (config/scoring.yaml)",
    "D9": "walk-forward quarters, expanding, 1-week embargo, 5 seasons minimum at level 2 "
    "(config/folds.yaml; models.stacking.check_level2)",
    "D10": "no price before 2014-06-07",
    "D11-D15": "holdout A2 from 2026-11-28, fully locked; no holdout row is read; not opened",
    "D19": "every variant a ranked row; nothing adopted or rejected",
    "D21": "no figure built on JRDB or JRA-VAN in the repository: records with the ability "
    "member carry jravan and the committed summary withholds them",
    "#98": "the owner, 2026-11-24 and 2026-11-27: shrinkage target alpha = 0, the market alone; "
    "gates and segment weights pre-registered as variants; standalone, marginal and "
    "encompassing are columns beside each ranked row, not thresholds",
}


def _members_sources(members: Sequence[str]) -> tuple[str, ...]:
    found = {s for m in members for s in MEMBER_FILES[MEMBERS[m][0]][2]}
    return tuple(sorted(found | set(STORE)))


SOURCES = _members_sources(MEMBERS)


# --------------------------------------------------------------------------- the spec


def build_spec(scheme: FoldScheme, bootstrap: dict, digests: dict[str, str]) -> dict:
    """The family's pre-registered spec, frozen in the registry before any row is read."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E40": "the C4 meld of the out-of-fold members at the final Win price: each member "
            "standalone and in the meld against the market-only meta, the marginal table, "
            "encompassing, correlations and ambiguity",
            "E41": "precision weighting alpha_i = alpha_0 / (1 + kappa sigma_i^2) beats a "
            "constant alpha out of fold (ΔLL and Clark-West)",
            "E42": "the members' weights differ between debut races, handicaps and other races",
        },
        "target": "the race's winner (Win pool), conditioned on the final Win price (D1 A)",
        "member_files": {
            name: {"file": f"features/japan/{file}", "sha256": digests[name], "ticket": ticket}
            | {"sources": list(sources), "family": family}
            for name, (file, ticket, sources, family) in MEMBER_FILES.items()
        },
        "members": {
            m: {"export": e, "column": c, "value": t, "sigma": s}
            for m, (e, c, t, s) in MEMBERS.items()
        },
        "not_run": NOT_RUN,
        "populations": {
            name: {"from": start.isoformat(), "members": list(members)}
            for name, (start, members) in POPULATIONS.items()
        },
        "level2": {
            "rows": "studies.meta.store_frame: priced starters of run races with a winner on "
            "development days of the price era, q the normalised final Win price",
            "join": "models.meld.member_frame: a member row is used only when its oof_fold is the "
            "row's D9 quarter; a race with a runner lacking a member is dropped",
            "guard": "models.stacking.join_members and check_level2 on the scored folds",
        },
        "folds": {
            "scheme": dataclasses.asdict(scheme),
            "quarters": "walk_forward over the store frame with min_train 1, every quarter "
            "labelled by its test start (studies.ability.level1_quarters' rule); a member's "
            "member_train_end is the train_end of its oof_fold's quarter",
            "scored": f"every quarter whose level-2 training window covers {D9_MIN_SEASONS} "
            "seasons (stacking.fold_windows); earlier quarters form one seed block that only "
            "trains",
            "training": "every earlier row at least embargo_days before the quarter's first race "
            "day (expanding; the same rows as walk_forward's train_idx)",
        },
        "fit": {
            "form": "stage2",
            "ridge": RIDGE,
            "quadratic": False,
            "penalised": "every member term; never the market's",
            "comparator": "the market-only stage-2 meta (u = beta ln q), unpenalised",
        },
        "variants": {
            "<pop>:market_only": "the comparator itself",
            "<pop>:standalone:<m>": "the meld of m alone",
            "<pop>:meld:all": "the meld of every member of the population",
            "<pop>:meld:all minus <m>": "the meld without m",
            "<pop>:marginal:<m>": "ll(all minus m) - ll(all) per race (models.meld.ablation_table)",
            "<pop>:encompassing:<m>": "models.meta.encompassing on the scored rows (in-sample), "
            "meeting-clustered se",
            "<pop>:e41:<m>:c[<c>]": "meld:all with m precision-weighted, kappa = c / median of "
            "sigma_m^2 over the population's rows, c in kappa_grid",
            "<pop>:e42:gates[<g>]": "meld:all gated on g (each cell its own alpha_k)",
            "<pop>:correlations": "models.meld.member_correlations on the scored rows",
            "<pop>:ambiguity": "models.meld.ambiguity on the scored rows",
        },
        "kappa_grid": list(KAPPA_GRID),
        "gates": {g: {"column": c, "value": v, "what": w} for g, (c, v, w) in GATES.items()},
        "e42": [list(g) for g in E42_GATES],
        "e41_test": "Clark-West: per race sum over runners of (y - p0)^2 - (y - p1)^2 + "
        "(p0 - p1)^2, p0 the constant-alpha meld, p1 the precision-weighted one; its mean with "
        "meeting-bootstrap bounds (positive: p1 better)",
        "e42_alpha": "each gate cell's alpha_k from fit_meld on every row of the population, "
        "meeting-clustered se, one-sided bounds at the D7 level",
        "delta_ll": "per scored race, the comparator's winner log loss minus the variant's; "
        "core.eval.bootstrap.paired_bootstrap over meetings, one seed, the same races",
        "correlations_bounds": "a meeting bootstrap of the pooled within-race cross products",
        "tail": {
            "quantiles": list(TAIL_QUANTILES),
            "claim": "the variant's out-of-fold p x the final Win odds",
            "settled": "studies.interactions.win_paid: the recorded Win dividend over the unit "
            "stake; a race without a Win dividend row fails the variant",
            "leaderboard": "the top band's return per ¥100 (development fold), kind comparator, "
            "member or model, pool WIN, selection look_ahead",
        },
        "bootstrap": bootstrap,
        "decisions": DECISIONS,
        "sources": list(SOURCES),
    }


# --------------------------------------------------------------------------- inputs


def read_store() -> tuple[pd.DataFrame, dict]:
    """The store frame with the gate columns (0/1, race-level)."""
    runners, races = read_development(race_columns=[*RACE_COLUMNS, "race_class", "weight_type"])
    frame, counts = store_frame(runners, races)
    flags = races[["race_id", "race_class", "weight_type"]].astype(object)
    keyed = (
        frame[["race_id"]]
        .astype(object)
        .merge(flags, on="race_id", how="left", validate="many_to_one")
    )
    for gate, (column, value, _) in GATES.items():
        frame[gate] = keyed[column].eq(value).to_numpy(dtype=int)
    return frame, counts


def read_members(start: date) -> dict[str, pd.DataFrame]:
    """Each member as models.meld.member_frame takes it (race_id, horse_id, logp, oof_fold) and
    each sigma as a member of its own ('<member>_sigma'), development days from `start`."""
    out = {}
    for export, (file, *_) in MEMBER_FILES.items():
        mine = {m: spec for m, spec in MEMBERS.items() if spec[0] == export}
        columns = {c for _, c, _, s in mine.values()} | {s for *_, s in mine.values() if s}
        table = pq.read_table(
            paths.data_dir() / "features" / COUNTRY / file,
            columns=["race_id", "horse_id", "race_date", "oof_fold", *sorted(columns)],
            filters=[("race_date", ">=", pd.Timestamp(start))],
        )
        rows, _ = split_off(table.to_pandas(), "race_date", COUNTRY)
        keys = {k: rows[k].to_numpy(dtype=object) for k in ("race_id", "horse_id", "oof_fold")}
        codes = pd.factorize(keys["race_id"])[0]
        for m, (_, column, transform, sigma) in mine.items():
            v = rows[column].to_numpy(dtype=float)
            v = np.log(v) if transform == "ln" else v
            v = meld._log_softmax(v, codes) if transform.startswith("log-softmax") else v
            out[m] = pd.DataFrame({**keys, "logp": v})
            if sigma:
                out[f"{m}_sigma"] = pd.DataFrame({**keys, "logp": rows[sigma].to_numpy(float)})
    return out


def quarters(store: pd.DataFrame, scheme: FoldScheme) -> list[Fold]:
    """D9's quarters over the store frame, every one labelled (min_train 1, as
    `studies.ability.level1_quarters`): each quarter's test rows and every earlier row to its
    embargo."""
    return walk_forward(store["race_date"], scheme=dataclasses.replace(scheme, min_train=1))


def population_frame(
    store: pd.DataFrame, folds: list[Fold], members: dict, pop: str
) -> tuple[pd.DataFrame, list[str], dict, list[dict]]:
    """The population's level-2 frame (store positions as index, `fold` relabelled so the seed
    block is one label), its scored fold labels, counts and D9 windows. Refuses (stacking's
    InvalidProvenance or IneligibleFold) before any fit. A member's `member_train_end` is the
    D9 train_end of its oof_fold's quarter: the member studies fit each quarter on every earlier
    row to its embargo."""
    start, names = POPULATIONS[pop]
    own = [f for f in folds if f.test_start >= start]
    label = np.full(len(store), None, dtype=object)
    for f in own:
        label[f.test_idx] = f.test_start.isoformat()
    labelled = pd.notna(label)
    base = store.loc[labelled].assign(fold=label[labelled])
    used = {k: v for k, v in members.items() if k.removesuffix("_sigma") in names}
    frame = meld.member_frame(used, base)
    train_end = {f.test_start.isoformat(): f.train_end for f in own}
    keys = frame[["race_id", "horse_id"]]
    predictions = pd.concat(
        [
            keys.assign(
                member=m, logp=frame[m].to_numpy(), member_fold=frame["fold"].to_numpy()
            ).assign(member_train_end=pd.to_datetime(frame["fold"].map(train_end)).to_numpy())
            for m in names
        ],
        ignore_index=True,
    )
    level2 = stacking.join_members(
        frame[["race_id", "horse_id", "race_date"]],
        restrict(own, frame.index.to_numpy(), len(store)),
        predictions,
        members=names,
    )
    windows = stacking.fold_windows(level2, min_seasons=D9_MIN_SEASONS)
    eligible = [i for i, w in enumerate(windows) if w.eligible and len(level2.folds[i].test_idx)]
    if not eligible:
        raise stacking.IneligibleFold(f"{pop}: no fold has {D9_MIN_SEASONS} seasons of level 2")
    first = eligible[0]
    scored = dataclasses.replace(level2, folds=level2.folds[first:])
    stacking.check_level2(scored, min_seasons=D9_MIN_SEASONS)
    seed, first_scored = own[0].test_start.isoformat(), windows[first].test_start.isoformat()
    frame["fold"] = np.where(frame["fold"].to_numpy() < first_scored, seed, frame["fold"])
    labels = sorted(set(frame["fold"]) - {seed})
    frame[Q] = np.log(frame["q"].to_numpy())
    counts = {
        f"{pop}:base_races": int(base["race_id"].nunique()),
        f"{pop}:races_with_every_member": int(frame["race_id"].nunique()),
        f"{pop}:scored_folds": len(labels),
        f"{pop}:scored_races": int(frame.loc[frame["fold"] != seed, "race_id"].nunique()),
    }
    table = [
        {"population": pop, "fold": w.test_start.isoformat(), "train from": w.train_from}
        | {"train to": w.train_to, "seasons": round(w.years, 2), "train races": w.train_races}
        | {"scored": i >= first}
        for i, w in enumerate(windows)
    ]
    return frame, labels, counts, table


# --------------------------------------------------------------------------- scoring


def _bounds(diff: pd.Series, clusters: pd.Series, spec: dict):
    boot = spec["bootstrap"]
    return paired_bootstrap(
        diff, clusters, n_boot=boot["n_boot"], level=boot["level"], seed=boot["seed"]
    )


def _iv(fold: str, metric: str, iv, n: int):
    return _metric(fold, metric, iv.mean, n, iv.lo, iv.hi)


def _block(kind: str, pop: str, name: str, spec: dict) -> dict:
    return {
        "kind": kind,
        "pool": "WIN",
        "rule": f"{pop}:{name} (C4 meld, stage 2 at the final Win price): a flat Win bet on every "
        f"scored runner in N5's top claim band (p x final odds at or above the "
        f"{TAIL_QUANTILES[-1]:g} quantile)",
        "selection": "look_ahead",
        "payout_rate": None,
        "ci_level": spec["bootstrap"]["level"],
        "n_unit": "bets",
    }


@dataclasses.dataclass
class Scored:
    """A population's scored rows, its out-of-fold predictions and per-race losses."""

    rows: pd.DataFrame
    p: pd.DataFrame
    ll: pd.DataFrame
    clusters: pd.Series
    fold_of: pd.Series  # race_id -> scored fold label


def _tail(p: pd.Series, scored: Scored, spec: dict, name: str) -> tuple[list, dict]:
    rows = scored.rows
    ret = rows["paid"].to_numpy(dtype=float)
    if np.isnan(ret).any():
        races = rows.loc[np.isnan(ret), "race_id"].drop_duplicates()
        raise ValueError(f"{len(races)} scored races have no WIN dividend row (unsettled)")
    bets = pd.DataFrame(
        {
            "claim": p.to_numpy() * rows["win_odds"].to_numpy(dtype=float),
            "ret": ret,
            CLUSTER: rows[CLUSTER].to_numpy(),
        }
    )
    boot = spec["bootstrap"]
    curve = tail_curve(
        bets,
        claim_col="claim",
        return_col="ret",
        cluster_col=CLUSTER,
        quantiles=list(TAIL_QUANTILES),
        n_boot=boot["n_boot"],
        level=boot["level"],
        seed=boot["seed"],
    )
    top = curve.iloc[-1]
    n = int(top["n_bets"])
    lo, hi = 100 * top["lo"], 100 * top["hi"]
    metrics = [
        _metric("development", "return_per_100", 100 * top["realised"], n, lo, hi),
        _metric("development", "n_meetings", top["n_clusters"], n),
    ]
    for r in curve.itertuples(index=False):
        if r.n_bets:
            band = f"claim_q[{r.q_lo:g},{r.q_hi:g}]"
            metrics.append(
                _metric(
                    "all_folds",
                    f"tail_return_per_100[{band}]",
                    100 * r.realised,
                    r.n_bets,
                    100 * r.lo,
                    100 * r.hi,
                )
            )
    return metrics, {"tail": curve.assign(variant=name).to_dict("records")}


def _model(name: str, scored: Scored, spec: dict, against: str | None = None) -> tuple[list, dict]:
    """A model variant's ΔLL against the market only (per fold and overall, with bounds), its
    mean log loss and its tail; with `against`, its paired ΔLL against that variant too."""
    ll = scored.ll
    n = len(ll)
    metrics = [_metric("all_folds", "mean_log_loss", ll[name].mean(), n)]
    row = {"variant": name, "races": n, "mean log loss": ll[name].mean()}
    if name != MARKET_ONLY:
        diff = ll[meld.MARKET] - ll[name]
        iv = _bounds(diff, scored.clusters, spec)
        metrics.append(_iv("all_folds", "delta_ll_vs_market_only", iv, n))
        for fold, part in diff.groupby(scored.fold_of.reindex(diff.index).to_numpy()):
            metrics.append(_metric(fold, "delta_ll_vs_market_only", part.mean(), len(part)))
        row |= {"ΔLL vs market": iv.mean, "lower": iv.lo, "upper": iv.hi}
    if against is not None:
        iv = _bounds(ll[against] - ll[name], scored.clusters, spec)
        metrics.append(_iv("all_folds", f"delta_ll_vs_{against}", iv, n))
        row |= {f"ΔLL vs {against}": iv.mean, "its lower": iv.lo, "its upper": iv.hi}
    tail_metrics, tables = _tail(scored.p[name], scored, spec, name)
    return metrics + tail_metrics, tables | {"delta": [row]}


def clark_west(rows: pd.DataFrame, p0: pd.Series, p1: pd.Series) -> pd.Series:
    """Per race, Clark and West's adjusted squared-error difference on the winner indicator:
    sum of (y - p0)^2 - (y - p1)^2 + (p0 - p1)^2 (positive: p1, the larger model, is better)."""
    y = rows["won"].to_numpy(dtype=float)
    a, b = p0.to_numpy(), p1.to_numpy()
    f = (y - a) ** 2 - (y - b) ** 2 + (a - b) ** 2
    return pd.Series(f).groupby(rows["race_id"].to_numpy()).sum()


def _e41(name: str, scored: Scored, spec: dict) -> tuple[list, dict]:
    metrics, tables = _model(name, scored, spec, against="meld:all")
    cw = clark_west(scored.rows, scored.p["meld:all"], scored.p[name])
    iv = _bounds(cw, scored.clusters, spec)
    metrics.append(_iv("all_folds", "clark_west_vs_meld:all", iv, len(cw)))
    tables["e41"] = [{"variant": name, "CW mean": iv.mean, "lower": iv.lo, "upper": iv.hi}]
    return metrics, tables


def _e42(
    name: str, gates: Sequence[str], scored: Scored, frame: pd.DataFrame, opts: dict, spec: dict
) -> tuple[list, dict]:
    metrics, tables = _model(name, scored, spec, against="meld:all")
    cell = pd.Series(meld._cells(scored.rows, gates), index=scored.rows.index)
    cell_of = cell.groupby(scored.rows["race_id"].to_numpy()).first()
    diff = scored.ll["meld:all"] - scored.ll[name]
    tables["e42_cells"] = []
    for c, part in diff.groupby(cell_of.reindex(diff.index).to_numpy()):
        iv = _bounds(part, scored.clusters, spec)
        metrics.append(_iv("all_folds", f"delta_ll_vs_meld:all[{c}]", iv, len(part)))
        tables["e42_cells"].append(
            {"variant": name, "cell": c, "races": len(part), "ΔLL vs meld:all": iv.mean}
            | {"lower": iv.lo, "upper": iv.hi}
        )
    fit = meld.fit_meld(frame, **opts, cluster_col=CLUSTER)
    z = _one_sided_z(spec)
    tables["e42_alpha"] = []
    for term, (member, c) in fit.terms.items():
        a, s = float(fit.coef[term]), float(fit.se_cluster[term])
        metrics.append(_metric("in_sample", f"alpha[{term}]", a, fit.n_races, a - z * s, a + z * s))
        tables["e42_alpha"].append(
            {"variant": name, "member": member, "cell": c, "alpha": a, "se (meeting)": s}
            | {"lower": a - z * s, "upper": a + z * s, "races": fit.n_races}
        )
    return metrics, tables


def _marginal(table: pd.DataFrame, member: str) -> tuple[list, dict]:
    row = table.set_index("variant").loc[f"all minus {member}"]
    m = _metric(
        "all_folds", "marginal_gain", row["gain"], int(row["races"]), row["gain_lo"], row["gain_hi"]
    )
    return [m], {
        "marginal": [
            {"member": member, "races": int(row["races"]), "gain": row["gain"]}
            | {"lower": row["gain_lo"], "upper": row["gain_hi"]}
        ]
    }


def _encompassing(scored: Scored, member: str, spec: dict) -> tuple[list, dict]:
    c, s = encompassing(scored.rows, member, cluster_col=CLUSTER)
    z, n = _one_sided_z(spec), scored.ll.shape[0]
    return [_metric("in_sample", "encompassing_coef", c, n, c - z * s, c + z * s)], {
        "encompassing": [
            {"member": member, "coef": c, "se (meeting)": s, "lower": c - z * s}
            | {"upper": c + z * s, "races": n}
        ]
    }


def correlation_bounds(
    rows: pd.DataFrame, members: list[str], clusters: pd.Series, spec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided bounds at the D7 level on each within-race correlation, from a meeting bootstrap
    of the pooled cross products of the demeaned members."""
    x = meld.demeaned(rows, members)
    unit = pd.factorize(clusters.reindex(rows["race_id"].to_numpy()).to_numpy(), sort=True)[0]
    k, g = len(members), unit.max() + 1
    cross = np.zeros((g, k * k))
    np.add.at(cross, unit, (x[:, :, None] * x[:, None, :]).reshape(len(x), -1))
    boot = spec["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    draws = []
    for start in range(0, boot["n_boot"], 500):
        size = min(500, boot["n_boot"] - start)
        idx = rng.integers(0, g, size=(size, g))
        weights = np.stack([np.bincount(row, minlength=g) for row in idx])
        s = (weights @ cross).reshape(size, k, k)
        d = np.sqrt(np.einsum("bii->bi", s))
        draws.append(s / (d[:, :, None] * d[:, None, :]))
    draws = np.concatenate(draws)
    level = boot["level"]
    return np.quantile(draws, 1 - level, axis=0), np.quantile(draws, level, axis=0)


def _correlations(scored: Scored, members: list[str], spec: dict) -> tuple[list, dict]:
    corr = meld.member_correlations(scored.rows, members)
    lo, hi = correlation_bounds(scored.rows, members, scored.clusters, spec)
    metrics, rows, n = [], [], len(scored.ll)
    for i, a in enumerate(members):
        for j in range(i + 1, len(members)):
            b, r = members[j], float(corr.iloc[i, j])
            metrics.append(_metric("all_folds", f"corr[{a},{b}]", r, n, lo[i, j], hi[i, j]))
            rows.append({"a": a, "b": b, "r": r, "lower": lo[i, j], "upper": hi[i, j]})
    return metrics, {"correlations": rows}


def _ambiguity(scored: Scored, members: list[str], spec: dict) -> tuple[list, dict]:
    terms = meld.ambiguity_terms(scored.rows, members)
    iv = _bounds(terms, scored.clusters, spec)
    point = meld.ambiguity(scored.rows, members)
    return [_metric("all_folds", "ambiguity", point, len(terms), iv.lo, iv.hi)], {
        "ambiguity": [{"members": len(members), "ambiguity": point, "lower": iv.lo, "upper": iv.hi}]
    }


def plan(pop: str, frame: pd.DataFrame | None) -> dict[str, tuple[str, dict]]:
    """The population's model variants: name -> (hypothesis, fit_meld keywords). Without a frame
    (an aborted launch's plan) kappa is not computed."""
    names = list(POPULATIONS[pop][1])
    base = {"form": "stage2", "ridge": RIDGE, "quadratic": False}
    out = {MARKET_ONLY: ("E40", {"members": [], "form": "stage2", "quadratic": False})}
    out |= {f"standalone:{m}": ("E40", meld.meld_options([m], **base)) for m in names}
    out["meld:all"] = ("E40", meld.meld_options(names, **base))
    for m in names:
        rest = [k for k in names if k != m]
        out[f"meld:all minus {m}"] = ("E40", meld.meld_options(rest, **base))
    for m in names:
        if MEMBERS[m][3] is None:
            continue
        sigma = f"{m}_sigma"
        median = None if frame is None else float(np.median(frame[sigma].to_numpy() ** 2))
        for c in KAPPA_GRID:
            kappa = {m: 0.0 if median is None else c / median}
            opts = meld.meld_options(names, **base, precision={m: sigma}, kappa=kappa)
            out[f"e41:{m}:c[{c:g}]"] = ("E41", opts)
    for gates in E42_GATES:
        out[f"e42:gates[{','.join(gates)}]"] = (
            "E42",
            meld.meld_options(names, **base) | {"gates": list(gates)},
        )
    return out


def _kind(name: str) -> str:
    if name == MARKET_ONLY:
        return "comparator"
    return "member" if name.startswith("standalone:") else "model"


def planned() -> list[tuple[str, str]]:
    out = [("E41" if n.startswith("e41") else "E40", n) for n in NOT_RUN]
    for pop, (_, names) in POPULATIONS.items():
        out += [(h, f"{pop}:{name}") for name, (h, _) in plan(pop, None).items()]
        out += [("E40", f"{pop}:marginal:{m}") for m in names]
        out += [("E40", f"{pop}:encompassing:{m}") for m in names]
        out += [("E40", f"{pop}:correlations"), ("E40", f"{pop}:ambiguity")]
    return out


def score_population(study: Study, frame: pd.DataFrame, pop: str, spec: dict, log=print) -> None:
    """Every variant of one population into `study` (pure but for `log`)."""
    names = list(POPULATIONS[pop][1])
    embargo = spec["folds"]["scheme"]["embargo_days"]
    scheme = "walk_forward:" + json.dumps(spec["folds"]["scheme"], sort_keys=True)
    variants = plan(pop, frame)
    preds, errors = {}, {}
    for name, (_, opts) in variants.items():
        started = time.monotonic()
        try:  # one variant at a time, so a failed fit fails its own record only
            got = meld.walk_forward_predictions(frame, {name: opts}, embargo_days=embargo)
            preds[name] = got[name]
            preds.setdefault(meld.MARKET, got[meld.MARKET])
        except Exception as exc:  # the variant's failed record carries it
            errors[name] = exc
        log(f"{pop}:{name} ({time.monotonic() - started:.0f} s)")
    if meld.MARKET not in preds:
        raise RuntimeError(f"{pop}: no variant was predicted: {errors}")
    p = pd.DataFrame(preds)
    rows = frame.loc[p.index]
    ll = meld.walk_forward_loss(frame, p)
    race = rows["race_id"].to_numpy()
    scored = Scored(
        rows=rows,
        p=p,
        ll=ll,
        clusters=rows[CLUSTER].groupby(race).first(),
        fold_of=rows["fold"].groupby(race).first(),
    )
    study.counts[f"{pop}:scored_meetings"] = int(scored.clusters.nunique())

    def attempt(hypothesis, name, opts, compute, *, needs=(), kind=None, members=names):
        """One record; `needs` are the predictions it reads, whose errors it fails with."""
        params = {"population": pop} | (opts or {})
        if kind is not None:
            params["leaderboard"] = _block(kind, pop, name, spec)
        variant = MetaVariant(
            hypothesis,
            f"{pop}:{name}",
            scheme,
            {"study": "studies/meld.py", "fit": "models.meld.fit_meld"},
            json.loads(json.dumps(params)),  # tuples as lists, as the registry reads them back
            sources=_members_sources(members),
        )

        def run():
            for needed in needs:
                if needed in errors:
                    raise errors[needed]
            return compute()

        study.attempt(variant, run)

    for name, (hypothesis, opts) in variants.items():
        needs, used = (name, meld.MARKET), opts.get("members") or []
        if name.startswith("e41:"):
            compute = lambda name=name: _e41(name, scored, spec)  # noqa: E731
            needs += ("meld:all",)
        elif name.startswith("e42:"):
            compute = lambda name=name, o=opts: _e42(  # noqa: E731
                name, o["gates"], scored, frame, o, spec
            )
            needs += ("meld:all",)
        else:
            compute = lambda name=name: _model(name, scored, spec)  # noqa: E731
        attempt(hypothesis, name, opts, compute, needs=needs, kind=_kind(name), members=used)

    renamed = ll.rename(columns={"meld:all": meld.ALL})
    renamed = renamed.rename(columns={f"meld:all minus {m}": f"all minus {m}" for m in names})
    table = None
    try:
        boot = spec["bootstrap"]
        table = meld.ablation_table(
            renamed,
            names,
            clusters=scored.clusters,
            n_boot=boot["n_boot"],
            level=boot["level"],
            seed=boot["seed"],
        )
    except Exception as exc:  # each marginal record fails with it
        errors["marginal"] = exc
    for m in names:

        def marginal(m=m):
            if table is None:
                raise errors["marginal"]
            return _marginal(table, m)

        attempt("E40", f"marginal:{m}", None, marginal)
        attempt(
            "E40",
            f"encompassing:{m}",
            None,
            lambda m=m: _encompassing(scored, m, spec),
            members=[m],
        )
    attempt("E40", "correlations", None, lambda: _correlations(scored, names, spec))
    attempt("E40", "ambiguity", None, lambda: _ambiguity(scored, names, spec))


def _not_run(why: str):
    raise RuntimeError(f"not run: {why}")


def compute_study(store: pd.DataFrame, counts: dict, members: dict, spec: dict, log=print):
    """Every variant: the not-run members, then each population's (see the module doc)."""
    folds = quarters(store, FoldScheme.from_mapping(spec["folds"]["scheme"]))
    study = Study(counts=dict(counts))
    fold_scheme = "walk_forward:" + json.dumps(spec["folds"]["scheme"], sort_keys=True)
    for name, why in NOT_RUN.items():
        variant = MetaVariant(
            "E41" if name.startswith("e41") else "E40",
            name,
            fold_scheme,
            {"study": "studies/meld.py"},
            {},
            sources=STORE,
        )
        study.attempt(variant, lambda why=why: _not_run(why))
    frames = {}
    for pop in POPULATIONS:
        frame, labels, pop_counts, windows = population_frame(store, folds, members, pop)
        frames[pop] = frame
        study.counts |= pop_counts
        study.tables.setdefault("d9_windows", []).extend(windows)
        log(f"{pop}: {pop_counts}, scored {labels[0]}..{labels[-1]}")
    union = sorted(set().union(*(set(f.index) for f in frames.values())))
    rows = store.loc[union]
    numbers, dividends = read_win_dividends(rows["race_id"].unique())
    unit = float(load_source(COUNTRY, "jra").unit_stake)
    paid, paid_counts = win_paid(rows, numbers, dividends, unit)
    study.counts |= paid_counts
    paid = pd.Series(paid, index=rows.index)
    for pop, frame in frames.items():
        frame["paid"] = paid.reindex(frame.index).to_numpy()
        score_population(study, frame, pop, spec, log=log)
    return study


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    ("d9_windows", "Level-2 training windows per quarter against D9's 5 seasons"),
    ("delta", "ΔLL against the market-only meta (out of fold; per race, nats)"),
    ("marginal", "Marginal gain of each member: ll(all minus m) - ll(all)"),
    ("encompassing", "Encompassing: each member beside ln q (in-sample, meeting-clustered)"),
    ("e41", "E41: Clark-West against the constant-alpha meld"),
    ("e42_cells", "E42: the gated meld against the pooled one, per cell"),
    ("e42_alpha", "E42: each cell's alpha (in-sample, meeting-clustered)"),
    ("correlations", "Member correlations (logp demeaned within race)"),
    ("ambiguity", "Heskes ambiguity (mean KL of the geometric-mean meld to each member)"),
    ("tail", "N5 tail at the final price (Win dividends; per ¥100)"),
)


def render_report(study: Study, spec: dict, about: dict, record_ids: dict[str, str]) -> str:
    lines = [f"# The C4 meld at the final Win price ({COUNTRY}, #{TICKET})", ""]
    lines += [f"- {k}: {v}" for k, v in about.items()]
    lines += ["", "## Counts", "", _markdown_table([study.counts])]
    for key, title in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_Not computed._"]
    failed = [{"variant": v.name, "error": v.error} for v in study.variants if v.error]
    lines += ["", "## Failed variants", "", _markdown_table(failed) if failed else "_None._"]
    ids = [{"variant": k, "run id": v[:16]} for k, v in record_ids.items()]
    lines += ["", "## Records", "", _markdown_table(ids)]
    return "\n".join([*lines, ""])


def artifact_stem(spec_hash: str, builds: dict) -> str:
    digest = hashlib.sha256(json.dumps(builds, sort_keys=True).encode()).hexdigest()
    return f"meld-{COUNTRY}-{spec_hash[:12]}-{digest[:12]}"


def _record(variant: MetaVariant, git_sha: str, builds: dict, spec_hash: str, *, artifacts=True):
    params = dict(variant.params) | {"spec_sha256": spec_hash}
    if variant.error:
        params["error"] = variant.error
    data_builds = dict(builds)
    if artifacts:
        data_builds["meld_artifacts"] = f"reports/{COUNTRY}@{artifact_stem(spec_hash, builds)}"
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=data_builds,
        sources=tuple(variant.sources),
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def run_study(*, out_dir: Path | None = None, log=print) -> dict:
    """Freeze the spec, read and check the inputs, score every variant, write the private report,
    append the records and write the committed summary."""
    started = time.monotonic()
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme = load_country_scheme(COUNTRY)
    features = paths.data_dir() / "features" / COUNTRY
    digests = {name: file_digest(features / file) for name, (file, *_) in MEMBER_FILES.items()}
    spec = build_spec(scheme, {**bootstrap_settings(), "seed": BOOTSTRAP_SEED}, digests)
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any row is read
    git_sha = _git_sha()
    builds = store_builds() | {
        f"member_{name}": f"features/{COUNTRY}/{file}@{digests[name][:16]}"
        for name, (file, *_) in MEMBER_FILES.items()
    }
    try:
        store, counts = read_store()
        members = read_members(min(start for start, _ in POPULATIONS.values()))
        study = compute_study(store, counts, members, spec, log=log)
    except Exception as exc:
        fold_scheme = "walk_forward:" + json.dumps(spec["folds"]["scheme"], sort_keys=True)
        for hypothesis, name in planned():
            failed = MetaVariant(
                hypothesis,
                name,
                fold_scheme,
                {"study": "studies/meld.py"},
                {},
                error=f"{type(exc).__name__}: {exc}",
                sources=SOURCES,
            )
            registry.append(_record(failed, git_sha, builds, spec_hash, artifacts=False))
        ability.write_committed_summary(registry, families=(FAMILY_ID,))
        raise

    blank = {_run_id(dataclasses.replace(r, git_sha="")): r for r in registry.records(FAMILY_ID)}
    record_ids, appended, new = {}, [], []
    for variant in study.variants:
        record = _record(variant, git_sha, builds, spec_hash)
        same = blank.get(_run_id(dataclasses.replace(record, git_sha="")))
        record_ids[variant.name] = _run_id(record if same is None else same)
        if same is None:
            new.append(record)
    stem = artifact_stem(spec_hash, builds)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "records": f"{len(new)} appended; {len(record_ids) - len(new)} already held",
        "run time": f"{time.monotonic() - started:.0f} s",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"{stem}.md", render_report(study, spec, about, record_ids), sources=SOURCES
    )
    shown = {"about": about, "counts": study.counts, "tables": study.tables}
    write_report(
        out_dir / f"{stem}.json", json.dumps(shown, indent=1, default=str), sources=SOURCES
    )
    for record in new:
        registry.append(record)
        appended.append(record.variant)
    ability.write_committed_summary(registry, families=(FAMILY_ID,))
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "artifact_stem": stem,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "records_appended": len(appended),
        "failed_variants": [v.name for v in study.variants if v.error],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.meld")
    parser.add_argument("--run", action="store_true", required=True, help="run the family")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(out_dir=args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
