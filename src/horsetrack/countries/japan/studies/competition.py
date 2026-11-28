"""The competition-terms study (#531, epic #73; #94's outline 4; 08 §3.3; 05 §3.4, H12):
`horsetrack japan competition-study`.

#94's terms (`models.competition`) added to #92's pooled C2 member (`studies.ability`), each
measured as a ranked member row. Level 1 is #92's run as is (`ability.load_inputs`, its rating
grid, pair and discount selection and pooled member: `level1_grid`, `select_discounts`,
`level1_members`). Each variant (VARIANTS: each term alone and the three together) is #92's
pooled `AbilityModel` at each outer fold's selection (its rating pair and pooled discounts) with
the variant's terms as extra features, fitted on the fold's training rows and predicting its test
quarter. The terms (`field_features`) are #94's `interaction_columns` of the before-the-race
rating (`studies.rating`'s mu_before at the fold's drift) over the other runners of the race.

Level 2 is #92's: the store's priced starters (`studies.meta.store_frame`), each member's ln p
renormalised over the race's priced starters, D9's folds, the stage-2 meta of a member
(`ability.STAGE2`'s form) and the market-only meta (`ability.market_only_oof`). `score_variants`
gives each variant the paired per-race winner ΔLL of its member alone against the pooled member
alone (standalone) and of its stage-2 meta against the market-only meta (at the final price),
with D7's meeting bounds, and its tail return at the final price (N5: the top claim band of
`core.eval.tail.tail_curve`, a flat Win bet on every scored runner), with a leaderboard block. The
REFF record splits the pooled member's stage-2 residual against the market-only meta by quantile
band of R_eff (`fold_comparability`: #94's `race_comparability` on each fold's graph of the races
before the fold's first day, 00:00 JST) and an unconnected bucket (`reff_buckets`; 05 H12, the
precision weighting itself is #98's E41).

The sire is JRA-VAN's, so every figure is withheld (D21): the report goes under the data dir
through `write_report`, the records carry the source `jravan` and the committed summary withholds
them. The spec is frozen before any row is read; holdout days are never read; nothing is adopted
(D16).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from horsetrack.core import paths
from horsetrack.core.eval.folds import load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import RunRecord, default_registry, summary_dir, write_summary
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.tail import ALL, tail_curve
from horsetrack.countries.japan.models import competition
from horsetrack.countries.japan.models.ability import AbilityModel
from horsetrack.countries.japan.models.meta import MetaSpec
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import ability
from horsetrack.countries.japan.studies import rating as c2_rating
from horsetrack.countries.japan.studies.calibration import (
    Study,
    Variant,
    _bounds,
    _markdown_table,
    _metric,
    bootstrap_settings,
    store_builds,
)
from horsetrack.countries.japan.studies.meta import (
    CLUSTER,
    Scored,
    _ll,
    _meta_oof,
    _run_id,
    _try,
    baseline_oof,
    restrict,
)

COUNTRY = "japan"
TICKET = 531
FAMILY_ID = "japan-c2-competition-final-win"
SOURCES = ("jra", "jravan")
BOOTSTRAP_SEED = 531  # reproducibility only; scoring.yaml signs no seed
ARTIFACTS = "competition_artifacts"  # the data_builds entry naming a run's report stem
POOLED = ability.POOLED  # #92's pooled member: the base every variant adds terms to
TERMS = ("field_mean", "field_max", competition.INTERACTION)
COMBINED = "field_all"
VARIANTS = {**{term: (term,) for term in TERMS}, COMBINED: TERMS}
REFF = "reff_by_bucket"
MEMBER_E, REFF_E = "E33", "E41"  # the C2 member's (#92) and the precision weighting's (#98)
PLANNED = (*((MEMBER_E, name) for name in VARIANTS), (REFF_E, REFF))
N5_QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)  # #438's Win member bands
REFF_QUANTILES = (0.2, 0.4, 0.6, 0.8)
BUCKET_COLUMNS = ["bucket", "races", "meetings", "reff_lo", "reff_hi"]
BUCKET_COLUMNS += ["residual", "lower", "upper"]
UNCONNECTED = "unconnected"
JST = "Asia/Tokyo"


def _stage2(member: str) -> MetaSpec:
    """`ability.STAGE2`'s form for any member column."""
    return dataclasses.replace(ability.STAGE2[POOLED], members=[member])


# --------------------------------------------------------------------------- the spec


def build_spec(scheme, bootstrap: dict) -> dict:
    """The family's pre-registered spec, frozen in the registry before any outcome is read."""
    base = ability.build_spec(scheme, bootstrap)
    level1_keys = ("frame", "model", "rating", "rating_fixed", "drift_grid", "scale_grid")
    level1_keys += ("entities", "discount_grid", "discount_selection")
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "base": {
            "family_id": ability.FAMILY_ID,
            "member": POOLED,
            "what": "#92's pooled member, recomputed here by #92's code on the same reads: its "
            "rating pair and discounts selected per outer fold by #92's rule",
            "level1": {key: base["level1"][key] for key in level1_keys},
        },
        "terms": {
            "field_mean": "the mean of the other runners' before-the-race rating in the race",
            "field_max": "the max of the other runners' before-the-race rating in the race",
            "field_x_rating": "field_mean x the runner's own before-the-race rating",
            "rating": "studies.rating's mu_before at the outer fold's selected drift (#91's "
            "DynamicRating: no outcome of the race enters it)",
            "field": "the race's level-1 starters (models.competition.interaction_columns); a "
            "runner with no other rated runner has field_mean and field_max 0 in the fit (its "
            "race adds nothing to the Plackett-Luce likelihood) and field_x_rating 0 (#94)",
        },
        "variants": {name: list(terms) for name, terms in VARIANTS.items()},
        "variant_model": "per outer fold: models.ability.AbilityModel(k=3) with the base's "
        "entities, discounts (1, w2, w3) and rating pair selected for the base member in that "
        "fold, features [rating, *the variant's terms], fitted on the fold's training rows and "
        "predicting its test quarter; the terms do not re-enter the selection",
        "hypotheses": {
            MEMBER_E: "a competition term added to #92's pooled member improves it standalone "
            "and adds to the market term at the final price (08 §3.3; opponent strength is "
            "implicit in Plackett-Luce, so a null is the expected result)",
            REFF_E: "05 H12: the pooled member's stage-2 error against the market-only meta "
            "rises with the field's effective resistance (R_eff), the premise of E41's "
            "precision weighting (#98)",
        },
        "hypothesis_ids": {name: h for h, name in PLANNED},
        "null": "no term changes the member's winner log loss, alone or at the final price; the "
        "residual does not vary with R_eff",
        "metrics": {
            "delta_ll_standalone": "all_folds: the paired per-race winner ΔLL (positive: the "
            "variant is better) of the variant's member alone against the base member alone, "
            "with D7's meeting-bootstrap one-sided bounds; per scored fold, points",
            "delta_ll_final_price": "all_folds: the same for the variant's stage-2 meta (stage2 "
            "on [member] and ln q, ridge 0, unbounded, out of fold) against the market-only "
            "stage-2 meta (ability.market_only_oof) on the same rows and folds",
            "return_per_100": "development: N5's top band, 100 x the realised return per unit "
            "stake and its bounds, n its bets; n_meetings its meetings",
        },
        "level2": {
            "frame": "studies.meta.store_frame: the priced starters of run races with a priced "
            "winner from the price era; q the normalised final Win price; a race with a starter "
            "lacking the base member is dropped",
            "baseline": "D4 calibrated market, in-fold on D9's folds (the scored-row filter only)",
            "member": "ln p of each level-1 member, renormalised over the race's priced starters",
            "scored": "every D9 fold with a test row",
        },
        "n5": {
            "quantiles": list(N5_QUANTILES),
            "bets": "core.eval.tail.tail_curve over every scored runner as a flat Win bet: "
            "claim = the variant's stage-2 meta p x final win_odds, return = win_odds / m for "
            "each of m co-winners (the store frame has no dead-heat dividend) and 0 otherwise, "
            "meeting clusters",
        },
        "reff": {
            "quantiles": list(REFF_QUANTILES),
            "graph": "per scored fold: models.competition.comparison_graph of every rated "
            "starter (studies.rating.read_results) of the races before 00:00 Asia/Tokyo of the "
            "fold's first day, race_comparability of the fold's test rows (+inf: an unconnected "
            "pair or a horse the graph never saw)",
            "residual": "per scored race: the winner log loss of the base member's stage-2 meta "
            "less the market-only meta's (positive: the member's meta is worse)",
            "buckets": "the finite R_eff in quantile bands (np.quantile edges, lo <= R < hi, "
            "the last band closed: core.eval.tail's rule) and one 'unconnected' bucket for "
            "+inf; each bucket's mean residual with D7's meeting-bootstrap bounds",
        },
        "leaderboard": "each VARIANTS record a member row (kind member, pool WIN, selection "
        "look_ahead: the claim reads the final price; payout_rate None: the return is at each "
        "race's paid odds), ranked by its N5 top-band return; the REFF record has no return",
        "folds": base["folds"],
        "bootstrap": bootstrap,
        "price_era_start": PRICE_ERA_START.isoformat(),
        "adoption": "none (D16): no holdout is opened and nothing is adopted",
        "sources": list(SOURCES),
    }


# --------------------------------------------------------------------------- the pure parts


def field_features(frame: pd.DataFrame, rating) -> pd.DataFrame:
    """The TERMS columns on frame's index: #94's interaction_columns of `rating` (one value per
    frame row, by index) over each race_id's other runners. Reads race_id and rating only."""
    data = pd.DataFrame(
        {"race_id": frame["race_id"], "rating": pd.Series(rating, index=frame.index, dtype=float)}
    )
    return competition.interaction_columns(data, rating_col="rating")[list(TERMS)]


def _jst_midnight(day) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(JST).tz_convert("UTC")


def fold_comparability(results: pd.DataFrame, rows: pd.DataFrame, folds) -> pd.Series:
    """Per race_id of the folds' test rows: #94's race_comparability on the comparison graph of
    the results before 00:00 Asia/Tokyo of the fold's first day."""
    parts = []
    for fold in folds:
        test = rows.iloc[fold.test_idx][["race_id", "horse_id"]]
        if len(test):
            graph = competition.comparison_graph(results, before_utc=_jst_midnight(fold.test_start))
            parts.append(competition.race_comparability(test, graph))
    return pd.concat(parts) if parts else pd.Series(dtype=float, name="race_comparability")


def _bucket(name: str, part: pd.Series, clusters: pd.Series, lo, hi, spec: dict) -> dict:
    row = {"bucket": name, "races": len(part), "meetings": 0, "reff_lo": lo, "reff_hi": hi}
    row |= {"residual": np.nan, "lower": np.nan, "upper": np.nan}
    if len(part):
        iv = _bounds(part, clusters, spec)
        row |= {"meetings": iv.n_clusters, "residual": iv.mean, "lower": iv.lo, "upper": iv.hi}
    return row


def reff_buckets(
    reff: pd.Series, residual: pd.Series, clusters: pd.Series, spec: dict
) -> pd.DataFrame:
    """The residual by quantile band of the finite R_eff (REFF_QUANTILES) and an unconnected
    bucket (R_eff +inf), over the races with a residual and an R_eff; see the module docstring."""
    res = residual.dropna()
    r = reff.reindex(res.index)
    keep = r.notna().to_numpy()
    res, r = res[keep], r[keep].to_numpy(dtype=float)
    finite = np.isfinite(r)
    qs = np.r_[0.0, REFF_QUANTILES, 1.0]
    edges, band = np.full(len(qs), np.nan), np.full(len(r), -1)
    if finite.any():
        edges = np.quantile(r[finite], qs)
        band[finite] = np.searchsorted(edges[1:-1], r[finite], side="right")  # tail's rule
    rows = [
        _bucket(f"q{k + 1}", res[band == k], clusters, edges[k], edges[k + 1], spec)
        for k in range(len(qs) - 1)
    ]
    rows.append(_bucket(UNCONNECTED, res[~finite], clusters, np.inf, np.inf, spec))
    return pd.DataFrame(rows, columns=BUCKET_COLUMNS)


def _block(terms, spec: dict) -> dict:
    return {
        "kind": "member",
        "pool": "WIN",
        "rule": f"#92's pooled C2 member with {' + '.join(terms)} at the stage-2 meta with ln "
        "q_final: a flat Win bet on every scored runner in N5's top claim band (p x final odds)",
        "selection": "look_ahead",
        "payout_rate": None,
        "ci_level": spec["bootstrap"]["level"],
        "n_unit": "bets",
    }


def _tail(p: pd.Series, scored: Scored, spec: dict, name: str) -> tuple[list, dict]:
    """N5 at the final price: the top claim band's return per ¥100 as the leaderboard's metric."""
    sr = scored.rows
    odds = sr["win_odds"].to_numpy(dtype=float)
    won = sr["won"].to_numpy(dtype=bool)
    m = sr.groupby("race_id")["won"].transform("sum").to_numpy(dtype=float)
    bets = pd.DataFrame(
        {
            "claim": p.reindex(sr.index).to_numpy() * odds,
            "ret": np.where(won, odds / np.maximum(m, 1.0), 0.0),
            CLUSTER: sr[CLUSTER].to_numpy(),
        }
    )
    boot = spec["bootstrap"]
    curve = tail_curve(
        bets,
        claim_col="claim",
        return_col="ret",
        cluster_col=CLUSTER,
        quantiles=list(N5_QUANTILES),
        n_boot=boot["n_boot"],
        level=boot["level"],
        seed=boot["seed"],
    )
    top = curve.loc[curve["segment"].eq(ALL)].iloc[-1]
    n = int(top["n_bets"])
    lo, hi = 100 * float(top["lo"]), 100 * float(top["hi"])
    metrics = [
        _metric("development", "return_per_100", 100 * float(top["realised"]), n, lo, hi),
        _metric("development", "n_meetings", top["n_clusters"], n),
    ]
    return metrics, {"tail": curve.assign(variant=name).to_dict("records")}


def score_variants(
    rows: pd.DataFrame,
    folds,
    reff: pd.Series,
    spec: dict,
    *,
    fold_scheme: str = "walk_forward",
    errors: dict[str, str] | None = None,
) -> Study:
    """One Variant per VARIANTS name and the REFF Variant over the scored races; see the module
    docstring. `errors`: a variant's level-1 failure, named in its failed Variant."""
    errors = errors or {}
    scored = Scored.of(rows, folds)
    sr = scored.rows
    study = Study(counts={"scored_folds": len(folds), "scored_races": scored.n_races})
    market = _try(lambda: ability.market_only_oof(rows, folds))
    comp = {"study": "studies/competition.py", "model": "models/ability.py", "base": POOLED}
    comp |= {"base_family": ability.FAMILY_ID, "form": "stage2", "market": "ln_q"}

    def scores(name: str) -> tuple[list, dict]:
        if name not in rows.columns:
            raise RuntimeError(
                f"the level-1 fit of {name} failed: {errors[name]}"
                if name in errors
                else f"the rows have no member column {name!r}"
            )
        alone, t_alone = ability._comparison(
            f"{name}: member alone vs {POOLED} alone",
            np.exp(sr[name]),
            np.exp(sr[POOLED]),
            scored,
            spec,
            metric="delta_ll_standalone",
        )
        p = _meta_oof(rows, folds, _stage2(name), "ln_q")[0]
        final, t_final = ability._comparison(
            f"{name}: stage-2 vs stage-2 market only",
            p,
            market.get(),
            scored,
            spec,
            metric="delta_ll_final_price",
        )
        tail, t_tail = _tail(p, scored, spec, name)
        return alone + final + tail, t_tail | {
            "comparisons": t_alone["comparisons"] + t_final["comparisons"]
        }

    for name, terms in VARIANTS.items():
        params = {"terms": list(terms), "ridge": 0.0, "bounds": "none"}
        params |= {"n5_quantiles": list(N5_QUANTILES), "leaderboard": _block(terms, spec)}
        study.attempt(
            Variant(MEMBER_E, name, fold_scheme, comp | {"member": name}, params),
            lambda n=name: scores(n),
        )

    def buckets() -> tuple[list, dict]:
        p = _meta_oof(rows, folds, _stage2(POOLED), "ln_q")[0]
        residual = _ll(p, sr) - _ll(market.get(), sr)
        table = reff_buckets(reff, residual, scored.clusters, spec)
        metrics = [
            _metric("all_folds", f"residual[{b}]", r, n, lo, hi)
            for b, n, r, lo, hi in table[
                ["bucket", "races", "residual", "lower", "upper"]
            ].itertuples(index=False)
            if n
        ]
        return metrics, {"reff_buckets": table.to_dict("records")}

    params = {"reff_quantiles": list(REFF_QUANTILES)}
    comp_reff = comp | {"member": POOLED, "score": "stage2_residual_vs_market_only"}
    study.attempt(Variant(REFF_E, REFF, fold_scheme, comp_reff, params), buckets)
    return study


# --------------------------------------------------------------------------- level 1 and 2


def variant_members(level1, members, states, outer, selection, log=print):
    """Each VARIANTS member's u on the outer folds' test rows (NaN elsewhere): the base's
    AbilityModel at the fold's selection with the variant's terms added; and each failed
    variant's error (its u is left out)."""
    numeric = pd.DataFrame(  # #92's level1_grid encoding: ids as codes
        {
            col: pd.factorize(level1[col].to_numpy(dtype=object))[0]
            for col in ("race_id", "horse_id")
        }
        | {"finish_pos": level1["finish_pos"].to_numpy(dtype=float)}
        | {col: ability._codes(level1[col]) for col in ability.ENTITIES}
    )
    field: dict[float, pd.DataFrame] = {}
    u = {name: np.full(len(level1), np.nan) for name in VARIANTS}
    errors: dict[str, str] = {}
    for fold, pick in zip(outer, selection, strict=True):
        drift, scale = pick["drift_var_per_day"], pick["scale"]
        if drift not in field:
            field[drift] = field_features(level1, states[drift]["mu_before"]).fillna(0.0)
        data = numeric.assign(
            **{ability.RATING: members[(drift, scale)].to_numpy()},
            **{t: field[drift][t].to_numpy() for t in TERMS},
        )
        train, test = data.iloc[fold.train_idx], data.iloc[fold.test_idx]
        w = (1.0, pick["pooled w2"], pick["pooled w3"])
        for name, terms in VARIANTS.items():
            if name in errors:
                continue
            model = AbilityModel(
                k=3, discounts=w, entity_cols=ability.ENTITIES, features=[ability.RATING, *terms]
            )
            try:
                u[name][fold.test_idx] = model.fit(train).predict(test)["u"].to_numpy()
            except Exception as exc:  # a failed variant is a failed record, never dropped
                errors[name] = f"{type(exc).__name__}: {exc}"
        state = f"failed {sorted(errors)}" if errors else "fitted"
        log(f"  variants on {fold.test_start}: {state}")
    return {n: v for n, v in u.items() if n not in errors}, errors


def member_rows(price: pd.DataFrame, level1: pd.DataFrame, u: dict, first) -> tuple:
    """The price frame's races whose every priced starter has every member in `u`, each member's
    ln p renormalised over the race's priced starters, and ln_q. Index: price positions."""
    keys = level1[["race_id", "horse_id"]].assign(**u)
    joined = price.reset_index().merge(
        keys, on=["race_id", "horse_id"], how="left", validate="one_to_one"
    )
    whole = joined[list(u)].notna().all(axis=1).groupby(joined["race_id"]).transform("all")
    rows = joined.loc[whole].set_index("index").rename_axis(None)
    race = rows["race_id"].to_numpy()
    for m in u:
        v = rows[m].to_numpy()
        top = pd.Series(v).groupby(race).transform("max").to_numpy()
        total = pd.Series(np.exp(v - top)).groupby(race).transform("sum").to_numpy()
        rows[m] = v - top - np.log(total)
    rows["ln_q"] = np.log(rows["q"].to_numpy())
    late = ~whole & joined["race_date"].ge(pd.Timestamp(first))
    counts = {
        "level2_races_with_every_member": int(rows["race_id"].nunique()),
        "level2_races_dropped_for_a_missing_member": int(joined.loc[late, "race_id"].nunique()),
    }
    return rows, counts


def load_inputs(log=print) -> ability.Inputs:
    """#92's reads (`ability.load_inputs`): development days only, every store read here."""
    return ability.load_inputs(log=log)


def compute_study(inputs: ability.Inputs, spec: dict, scheme, log=print) -> Study:
    """Every variant on `inputs`. Pure: no file or registry I/O."""
    level1, price = inputs.level1, inputs.price
    counts = dict(inputs.counts)
    fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
    quarters = ability.level1_quarters(level1["race_date"], scheme)
    outer = [q for q in quarters if q.test_start >= ability.LEVEL1_FROM]
    if quarters[0].test_start != ability.INNER_FROM or outer[0].test_start != ability.LEVEL1_FROM:
        raise ValueError("the level-1 quarters do not start at #92's signed dates")
    log(f"rating grid over {len(inputs.results)} starters")
    members, states = c2_rating.member_grid(inputs.results, level1)
    picks = ability.select_pairs(level1, members, outer)
    needed = ability.needed_pairs(outer, picks, quarters)
    members = {pair: members[pair] for pair in needed}
    grid = ability.level1_grid(level1, members, quarters, needed, log=log)
    selection = ability.select_discounts(level1, grid, outer, picks)
    u = {POOLED: ability.level1_members(level1, grid, outer, selection)[0][POOLED]}
    del grid
    log(f"variant fits on {len(outer)} outer folds")
    fitted, errors = variant_members(level1, members, states, outer, selection, log=log)
    u |= fitted

    folds2 = walk_forward(price["race_date"], scheme=scheme)
    price = price.assign(p_cal=baseline_oof(price, folds2))
    rows, member_counts = member_rows(price, level1, u, outer[0].test_start)
    rfolds = restrict(folds2, rows.index.to_numpy(), len(price))
    rfolds = [f for f in rfolds if len(f.test_idx)]
    rows = rows.reset_index(drop=True)
    log(f"comparison graphs for {len(rfolds)} scored folds")
    reff = fold_comparability(inputs.results, rows, rfolds)
    study = score_variants(rows, rfolds, reff, spec, fold_scheme=fold_scheme, errors=errors)
    study.counts = counts | member_counts | study.counts
    study.counts["variant_level1_failures"] = len(errors)
    study.tables["selection"] = selection
    return study


# --------------------------------------------------------------------------- the run

SECTIONS = (
    ("comparisons", "Paired ΔLL (positive: the first beats the second; meeting bootstrap)"),
    ("tail", "N5 at the final price (realised return per unit stake by claim band)"),
    ("reff_buckets", "Stage-2 residual against the market-only meta by R_eff bucket (05 H12)"),
    ("selection", "#92's selection per outer fold (the variants' rating pair and discounts)"),
)


def render_report(study: Study, spec: dict, about: dict, record_ids: dict[str, str]) -> str:
    """The private report: aggregate tables only, never a runner or race row."""
    lines = [
        f"# Competition terms on #92's pooled C2 member at the final Win price ({COUNTRY}, "
        f"#{TICKET})",
        "",
        "Built on JRA-VAN data (the sire): every figure here is withheld from the repository "
        "(D21).",
        "",
        *(f"- {k}: {v}" for k, v in about.items()),
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
        "",
        "## Variants (one registry record each)",
        "",
        _markdown_table(
            [
                {
                    "E": v.hypothesis_id,
                    "variant": v.name,
                    "status": "failed" if v.error else "scored",
                }
                | {"terms": " + ".join(VARIANTS.get(v.name, ())) or "-"}
                | {"record": record_ids.get(v.name, "")[:16], "error": v.error or ""}
                for v in study.variants
            ]
        ),
        "",
        "## Metrics",
        "",
    ]
    metrics = [
        {"variant": v.name, "fold": m.fold, "metric": m.metric, "value": m.value}
        | {"lower": m.lo, "upper": m.hi, "n": m.n}
        for v in study.variants
        for m in v.metrics
    ]
    lines.append(_markdown_table(metrics) if metrics else "_None scored._")
    for key, title in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_Not computed._"]
    return "\n".join([*lines, ""])


def artifact_stem(spec_hash: str, builds: dict) -> str:
    """A run's report stem: the spec hash and a digest of the data builds."""
    digest = hashlib.sha256(
        json.dumps(dict(builds), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"competition-{COUNTRY}-{spec_hash[:12]}-{digest[:12]}"


def _record(variant: Variant, git_sha: str, builds: dict, spec_hash: str, *, artifacts=True):
    """The variant's registry record; `artifacts=False` for an aborted run (no report)."""
    params = dict(variant.params) | {"spec_sha256": spec_hash}
    if variant.error:
        params["error"] = variant.error
    data_builds = dict(builds)
    if artifacts:
        data_builds[ARTIFACTS] = f"reports/{COUNTRY}@{artifact_stem(spec_hash, builds)}"
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=data_builds,
        sources=SOURCES,
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def write_committed_summary(registry) -> None:
    """This family's summary file by `write_summary`, and the committed index with its line
    added (#92's `write_committed_summary`, for this family)."""
    directory = summary_dir(COUNTRY)
    mine = ability._Families(registry, {FAMILY_ID})
    if not mine.records():
        return
    with tempfile.TemporaryDirectory() as scratch:
        write_summary(mine, scratch)
        for path in sorted(Path(scratch).glob("*.md")):
            if path.stem != "README":
                write_report(directory / path.name, path.read_text("utf-8"), sources=())
        fresh = Path(scratch, "README.md").read_text("utf-8").splitlines()
    index = directory / "README.md"
    committed = index.read_text("utf-8").splitlines() if index.exists() else []
    entries = {f: line for line in committed + fresh if (f := ability._index_family(line))}
    head = [line for line in fresh if not ability._index_family(line)]
    lines = [*head, *(entries[family] for family in sorted(entries))]
    write_report(index, "\n".join(lines) + "\n", sources=())


def run_study(*, out_dir: Path | None = None, log=print) -> dict:
    """Freeze the spec, read Japan's development data, score every variant, append the
    records and write the private report and the committed summary."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme = load_country_scheme(COUNTRY)
    spec = build_spec(scheme, {**bootstrap_settings(), "seed": BOOTSTRAP_SEED})
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any row is read
    git_sha = _git_sha()
    builds = {**store_builds(), **ability.pedigree_build()}
    try:
        study = compute_study(load_inputs(log=log), spec, scheme, log=log)
    except Exception as exc:
        fold_scheme = "walk_forward:" + json.dumps(spec["folds"], sort_keys=True)
        for hypothesis, name in PLANNED:
            failed = Variant(
                hypothesis,
                name,
                fold_scheme,
                {"study": "studies/competition.py"},
                {},
                error=f"{type(exc).__name__}: {exc}",
            )
            registry.append(_record(failed, git_sha, builds, spec_hash, artifacts=False))
        write_committed_summary(registry)
        raise

    blank = {_run_id(dataclasses.replace(r, git_sha="")): r for r in registry.records(FAMILY_ID)}
    record_ids, appended = {}, []
    for variant in study.variants:
        record = _record(variant, git_sha, builds, spec_hash)
        same = blank.get(_run_id(dataclasses.replace(record, git_sha="")))
        if same is None:
            record_ids[variant.name] = registry.append(record)
            appended.append(variant.name)
        else:  # the same run reproduced at another commit
            record_ids[variant.name] = _run_id(same)
    stem = artifact_stem(spec_hash, builds)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "base": f"`{ability.FAMILY_ID}`'s pooled member",
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "records": f"{len(appended)} appended; {len(record_ids) - len(appended)} already held",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"{stem}.md", render_report(study, spec, about, record_ids), sources=SOURCES
    )
    write_report(
        out_dir / f"{stem}.json",
        json.dumps(
            {"about": about, "counts": study.counts, "tables": study.tables}, indent=1, default=str
        ),
        sources=SOURCES,
    )
    write_committed_summary(registry)
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "artifact_stem": stem,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "records_appended": len(appended),
        "record_ids": {name: rid[:16] for name, rid in record_ids.items()},
        "failed_variants": [v.name for v in study.variants if v.error],
    }


def competition_study_command() -> None:
    """Measure #94's competition terms on #92's pooled member (#531); write the private report."""
    typer.echo(json.dumps(run_study(log=typer.echo), indent=2))
