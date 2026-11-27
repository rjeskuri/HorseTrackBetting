"""`horsetrack japan exacta-rules`: #131's E2 ratio-cut rules scored on #130-logged boards (#281).

docs/specs/exacta-overload-prereg.md sections 5-7, family ``exacta_study.E2_ID`` (rule-v4, frozen
hash ``FROZEN_E2_HASH``), recorded under ``E2_RULE_ID``, the same id without "/" (#525). With the
owner's reframe of 2026-11-27 nothing is refused for small n: zero logged meetings give an n = 0
result. All 16 members are decision-time rules: every selection input is on the race card or on
the boards at t.

- Selection at t = post - 2 min, from the boards at t only: p_i the normalised 1/odds of the Win
  board, p_c their discounted Harville exacta, q_c the exacta board's implied share, the favourite
  the lowest Win odds. Member (X, k, scope) buys c of category X when q_c / p_c < k.
- Scope "headline" (v4): the favourite's Win odds at t >= 3.0 and the race "less popular" on its
  race card (race_no 1-4 or DEBUT/MAIDEN, E1's popularity segment). The day's pool-total tertile
  is only the agreement diagnostic (``proxy_agreement``), never a selection input.
- Each member stakes 100 yen per selected combination, settled at the final dividend with its own
  impact from the payout identity on the race's final logged FORECAST pool (#283): the published
  dividend's 10 yen breakage is added back, and the result is rounded down to 10 yen with the 110
  yen minimum (D18). A floor-bound published dividend (100 yen, Plus-10 not funded; 110 yen,
  possibly a Plus-10 top-up) does not identify the winning investment: its impact-adjusted payout
  is unavailable (NaN), the published return is kept as ``payout_published``, and a member with
  such a hit has no impact-adjusted ROI. A race without a decision time (no post time), without
  boards at t, or whose log has no final FORECAST pool total, is not scored.
- Shrinkage (spec section 7): ``core.eval.segment_roi`` on each member's own bets alone (one
  ticket held by several nested members is evidence once per fit), fixed effects the final-board
  exacta odds bands of ``EXACTA_ODDS_BANDS`` (< 10, 10-30, 30-100, >= 100) and one constant
  segment column, the member's variant. Its status (fitted, failed, not attempted), the rows it
  was given and the member's meeting count are record metrics, never params.
- D, the full-board category ratio on the final boards, over every loaded race whose final exacta
  board prices every ordered pair of the final Win board's horses (a partial board is counted as
  incomplete and left out); the agreement of the race-card proxy with the lower final-pool
  tertile of each race day. Neither needs a board at t or a post time.
- The ordering fit is the artifact ``models/japan/v0/current.json`` names, recorded by its content
  hash; a fit whose out-of-fold training races reach the earliest scored race date is refused.

Development days only: a holdout day's race is dropped, and its log folder never opened, before
anything else of it is read (D14); LockedHoldout is not opened. Public JRA data only.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Iterable, Mapping
from datetime import date, timedelta
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.eval import snooping
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    RunRecord,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.segment_roi import fit_segment_roi
from horsetrack.countries.japan import odds_log, takeout
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.research import exacta_run
from horsetrack.countries.japan.research import exacta_study as study
from horsetrack.countries.japan.research.exacta_consistency import (
    CATEGORIES,
    combo_category,
    favourite,
    harville_exacta,
    implied_share,
    race_ratios,
)
from horsetrack.countries.japan.research.exacta_impact import own_impact

CUTS = (0.6, 0.7, 0.8, 0.9)
RULE_CATEGORIES = ("fav_second", "no_fav")
SCOPES = ("all", "headline")
MEMBERS = tuple((c, k, s) for c in RULE_CATEGORIES for k in CUTS for s in SCOPES)
STAKE = 100
TAKEOUT = 0.25
MINUTES_BEFORE_POST = 2
BLOCK_SIZE = 5
STEPM_FWER = exacta_run.STEPM_FWER
FROZEN_E2_HASH = "28929cd783cd9e25ae8c7f39f30f2b0891720b60864a38fdbb13d7dcd2a8d3bc"  # rule-v4
E2_RULE_ID = study.E2_ID.replace("/", "-")  # #525: no "/" in a registry family id
MAX_TEST_ID = f"{E2_RULE_ID}-max-test"
COUNTRY = "japan"
SOURCES = ("jra",)
FORECAST = "FORECAST"
TICKET = 281
FOLD_SCHEME = "development-days:#130-logged boards at T-2"
PAYOUT_STEP = 10.0  # FORECAST pays per 100 yen rounded down to 10 yen,
MIN_DIVIDEND = 110.0  # never below 110 yen,
PLUS10_UNFUNDED = 100.0  # unless the Plus-10 top-up to 110 yen was not funded: 100 yen published
FLOOR_BOUND = (PLUS10_UNFUNDED, MIN_DIVIDEND)  # neither identifies the winning investment
_HEADLINE = study.family_specs()[study.E2_ID]["scopes"]["headline"]
HEADLINE_FAV_ODDS = float(_HEADLINE["fav_odds_min"])
LESS_POPULAR_RACE_NO_MAX = int(_HEADLINE["less_popular"]["race_no_max"])
LESS_POPULAR_CLASSES = frozenset(_HEADLINE["less_popular"]["race_classes"])
EXACTA_ODDS_BANDS = (10.0, 30.0, 100.0)  # shrinkage fixed effects: final-board exacta odds bands
SHRINK_DRAWS = 4000
SHRINK_STATUSES = ("fitted", "failed", "not_attempted")
STORE_TABLES = ("races", "runners", "dividends")
MODEL_DIR = ("models", "japan", "v0")


def variant(category: str, cut: float, scope: str) -> str:
    return f"{category}|cut={cut}|{scope}"


def _yen(odds: float) -> float:
    """Board odds as yen per 100, float noise snapped: 1.1 is 110 yen, so the floor binds."""
    dividend = 100 * float(odds)
    whole = round(dividend)
    return float(whole) if abs(dividend - whole) < 1e-6 else dividend


def _win_p(win_odds: Mapping[int, float]) -> dict[int, float]:
    inv = {h: 1 / float(o) for h, o in win_odds.items()}
    total = sum(inv.values())
    return {h: v / total for h, v in inv.items()}


def _positive(value) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


# --------------------------------------------------------------------------- the rule


def select_combinations(
    win_odds: Mapping[int, float],
    exacta_odds: Mapping[tuple[int, int], float | None],
    *,
    rate: float,
    lambda2: float,
) -> dict[tuple[str, float], frozenset]:
    """The combinations each (category, cut) buys, from the Win and exacta boards at t."""
    p = _win_p(win_odds)
    horses = sorted(p)
    index = {h: k for k, h in enumerate(horses)}
    model = harville_exacta(np.array([p[h] for h in horses]), lambda2=lambda2)
    fav = favourite(win_odds)
    chosen: dict[tuple[str, float], set] = {(c, k): set() for c in RULE_CATEGORIES for k in CUTS}
    for combo, odds in exacta_odds.items():
        first, second = combo
        if odds is None or first == second or first not in index or second not in index:
            continue
        category = combo_category(combo, fav)
        if category not in RULE_CATEGORIES:
            continue
        q = implied_share(_yen(odds), rate=rate)
        if math.isnan(q):
            continue
        ratio = q / float(model[index[first], index[second]])
        for k in CUTS:
            if ratio < k:
                chosen[(category, k)].add((first, second))
    return {key: frozenset(value) for key, value in chosen.items()}


def own_impact_raw(
    dividend: float, *, pool: float, n_bets: int, stake: float = STAKE, takeout: float = TAKEOUT
) -> float:
    """The winner's unrounded dividend per 100 yen once `stake` is on each of `n_bets`
    combinations (the winner among them): 100 (1 - t) (P + n s) / (W + s), with the winner's money
    W = (1 - t) P / ((D + 5) / 100), the published dividend's breakage added back (half a step)."""
    unrounded = float(dividend) + PAYOUT_STEP / 2
    return unrounded * (1 + own_impact(pool, unrounded / 100, stake, n_bets, takeout))


def own_impact_dividend(
    dividend: float, *, pool: float, n_bets: int, stake: float = STAKE, takeout: float = TAKEOUT
) -> float:
    """The dividend that pays: ``own_impact_raw`` rounded down to 10 yen, never below 110 yen.
    NaN (unavailable) for a floor-bound published dividend, whose winning investment is
    unidentified (``impact_identified``)."""
    if not impact_identified(dividend):
        return math.nan
    raw = own_impact_raw(dividend, pool=pool, n_bets=n_bets, stake=stake, takeout=takeout)
    return max(MIN_DIVIDEND, PAYOUT_STEP * math.floor(raw / PAYOUT_STEP + 1e-9))


def impact_identified(dividend: float) -> bool:
    """False for a floor-bound published dividend: 100 yen (Plus-10 not funded) or 110 yen (the
    minimum, possibly a Plus-10 top-up). Either way the published figure does not identify the
    winning investment, so the payout identity cannot be inverted."""
    return float(dividend) not in FLOOR_BOUND


def settle(
    selected: Iterable[tuple[int, int]],
    winner: tuple[int, int],
    dividend: float,
    *,
    pool: float,
    stake: float = STAKE,
    takeout: float = TAKEOUT,
) -> dict:
    """One race of one member: n_bets, stake (yen), payout (yen) at the own-impact dividend (NaN on
    a hit at a floor-bound dividend, 0 on a miss), payout_published (yen at the published dividend,
    the observed return) and impact_available (False when the dividend is floor-bound)."""
    selected = {tuple(c) for c in selected}
    n = len(selected)
    payout = published = 0.0
    if n and tuple(winner) in selected:
        paid = own_impact_dividend(dividend, pool=pool, n_bets=n, stake=stake, takeout=takeout)
        payout = stake / 100 * paid
        published = stake / 100 * float(dividend)
    return {"n_bets": n, "stake": stake * n, "payout": payout, "payout_published": published,
            "impact_available": impact_identified(dividend)}  # fmt: skip


def lower_tertile_by_day(races: Iterable[Mapping], *, key: str) -> dict[str, bool]:
    """Per race_date, whether race[key] is at or below the day's 1/3 quantile (linear
    interpolation) over that day's races with a finite positive value; others are absent."""
    by_day: dict[date, list[Mapping]] = {}
    for race in races:
        if _positive(race.get(key)):
            by_day.setdefault(race["race_date"], []).append(race)
    out: dict[str, bool] = {}
    for day_races in by_day.values():
        cut = float(np.quantile([float(r[key]) for r in day_races], 1 / 3))
        out.update({r["race_id"]: float(r[key]) <= cut for r in day_races})
    return out


def less_popular(race_no: int, race_class: str | None) -> bool:
    """v4's race-card "less popular" (E1's popularity segment): race 1-4 or DEBUT/MAIDEN."""
    return int(race_no) <= LESS_POPULAR_RACE_NO_MAX or race_class in LESS_POPULAR_CLASSES


def race_scopes(
    win_odds: Mapping[int, float], *, race_no: int, race_class: str | None
) -> frozenset:
    """Always "all"; "headline" when the favourite's Win odds at t are 3.0 or more and the race
    card says less popular. Both inputs exist at t: no pool total enters the scope."""
    headline = win_odds[favourite(win_odds)] >= HEADLINE_FAV_ODDS and less_popular(
        race_no, race_class
    )
    return frozenset({"all", "headline"} if headline else {"all"})


# --------------------------------------------------------------------------- scoring


def _spec_hash(spec: dict) -> str:
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(family, name, params, components, metrics, data_builds, sha, status="scored"):
    return RunRecord(
        hypothesis_id="E43", family_id=family, variant=name, ticket=TICKET, git_sha=sha,
        data_builds=dict(data_builds), sources=SOURCES, fold_scheme=FOLD_SCHEME,
        components=dict(components), params=params, metrics=tuple(metrics), status=status,
    )  # fmt: skip


def odds_band(odds: float) -> str:
    """The shrinkage's fixed-effect band of a final-board exacta price."""
    low, mid, high = EXACTA_ODDS_BANDS
    if odds < low:
        return f"<{low:g}"
    if odds < mid:
        return f"{low:g}-{mid:g}"
    if odds < high:
        return f"{mid:g}-{high:g}"
    return f">={high:g}"


SHRINK_SEGMENT = "member"  # one constant level per fit: the member's variant


def _shrinkage(name: str, bets: list[dict], rates: Mapping[str, float], seed: int) -> dict:
    """segment_roi on one member's own bets (one row per race x selected combination with a
    final-board price), fixed effects the exacta odds bands. Each member is fitted alone, so a
    ticket other members also hold is evidence once here. segment_roi needs a segment column; a
    member's rows have no segment to shrink across, so the column is constant (the variant) and
    p_beats_takeout covers every one of its bets. Status "not_attempted" (no row to fit, n_bets
    0), "failed" or "fitted", n_bets the rows given to the fit."""
    info = {"band_edges": list(EXACTA_ODDS_BANDS), "segment_cols": [SHRINK_SEGMENT]}
    rows = [{SHRINK_SEGMENT: name, "odds": b["odds"], "won": int(b["won"]),
             "takeout": 1 - rates[b["race_id"]]}
            for b in bets if b["odds"] is not None]  # fmt: skip
    if not rows:
        reason = "no bet" if not bets else "no bet has a price on the final board"
        return {"status": "not_attempted", "reason": reason, "n_bets": 0, **info}
    frame = pd.DataFrame(rows)
    frame["band"] = frame["odds"].map(odds_band)
    try:
        model = fit_segment_roi(frame, segment_cols=info["segment_cols"], odds_band_col="band",
                                takeout_col="takeout", seed=seed, draws=SHRINK_DRAWS)  # fmt: skip
        p_beats = model.p_beats_takeout({SHRINK_SEGMENT: name})
    except (RuntimeError, ValueError, KeyError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:500],
                "n_bets": len(frame), **info}  # fmt: skip
    return {"status": "fitted", "n_bets": len(frame),
            "bands": {str(k): int(v) for k, v in frame["band"].value_counts().items()},
            "band_effects": model.band_effects, "tau": model.tau, "tau_ess": model.tau_ess,
            "tau_rejected": model.tau_rejected, "p_beats_takeout": p_beats, **info}  # fmt: skip


def _exclusion(race: Mapping) -> str | None:
    """Why E2 cannot score a (development-day) race, or None."""
    if race.get("no_decision_time"):  # no post time: no t, so no board at t
        return str(race["no_decision_time"])
    if not race["win_odds_t"] or not race["exacta_odds_t"]:
        return "no_board_at_t"
    if not _positive(race.get("pool_total")):
        return "no_final_board"
    return study.exclusion_reason(
        n_starters=int(race["n_starters"]),
        boost_may_be_withheld=bool(race["boost_may_be_withheld"]),
        exacta=list(race["exacta"]),
    )


def score_e2(
    races: Iterable[Mapping],
    *,
    registry,
    git_sha: str,
    lambda2: float,
    data_builds: Mapping[str, str],
    n_boot: int = 10000,
    seed: int = 131,
    holdout_day: Callable[[str, date], bool] = is_holdout_day,
    components: Mapping[str, str] | None = None,
    trained_through: date | None = None,
) -> dict:
    """Score the 16 E2 members on `races`, record one record per member under E2_RULE_ID and, when
    a race is scored, one SPA + StepM record under MAX_TEST_ID; a record already in the registry
    is not appended again. Refuses unless E2 is frozen at its spec's hash under E2_RULE_ID, and,
    before appending anything, when the ordering fit's training races reach (`trained_through` on
    or after) the earliest scored race date."""
    # owner, 2026-11-26 on #281: take the hash from #280's final head (rule-v4 since 2026-11-27)
    frozen = registry.frozen_spec_hash(E2_RULE_ID)
    expected = _spec_hash(study.family_specs()[study.E2_ID])
    if frozen is None or frozen != expected or frozen != FROZEN_E2_HASH:
        raise RuntimeError(f"{E2_RULE_ID} is not frozen at {FROZEN_E2_HASH} (found {frozen})")
    components = dict(components or {"ordering": f"lambda2={lambda2:.6g}"})
    append = exacta_run._appender(registry)

    excluded: dict[str, int] = {}
    rows, bets, rates, impact_unavailable = [], [], {}, 0
    for race in races:
        if holdout_day(COUNTRY, race["race_date"]):
            excluded["holdout_day"] = excluded.get("holdout_day", 0) + 1
            continue
        reason = _exclusion(race)
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        pool = race["pool_total"]
        win, final_exacta = race["win_odds_t"], race["final_exacta_odds"]
        rate = float(race["rate"])
        rates[race["race_id"]] = rate
        chosen = select_combinations(win, race["exacta_odds_t"], rate=rate, lambda2=lambda2)
        scopes = race_scopes(win, race_no=race["race_no"], race_class=race["race_class"])
        ((winner, dividend, _),) = race["exacta"]
        winner = tuple(winner)
        impact_unavailable += not impact_identified(dividend)
        row = {"race_id": race["race_id"], "meeting_id": race["meeting_id"],
               "race_date": race["race_date"]}  # fmt: skip
        for category, cut, scope in MEMBERS:
            sel = chosen[(category, cut)] if scope in scopes else frozenset()
            got = settle(sel, winner, dividend, pool=pool, takeout=1 - rate)
            name = variant(category, cut, scope)
            row[("n", name)], row[("stake", name)] = got["n_bets"], got["stake"]
            row[("payout", name)] = got["payout"]
            row[("published", name)] = got["payout_published"]
            for combo in sorted(sel):
                odds = final_exacta.get(combo)
                won = combo == winner
                bets.append({"category": category, "cut": cut, "scope": scope,
                             "race_id": race["race_id"], "meeting_id": race["meeting_id"],
                             "combination": combo, "odds": None if odds is None else float(odds),
                             "won": won, "stake": float(STAKE),
                             "payout": got["payout"] if won else 0.0,
                             "payout_published": got["payout_published"] if won else 0.0,
                             })  # fmt: skip
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) and trained_through is not None:
        first = min(frame["race_date"])
        if trained_through >= first:
            raise RuntimeError(
                f"the ordering fit's training races run through {trained_through}, on or after "
                f"the earliest scored race date {first}: look-ahead, not scored"
            )

    # owner reframe, 2026-11-27 on #281: nothing is refused for small n; n and meetings are
    # reported with every row
    members = []
    for category, cut, scope in MEMBERS:
        name = variant(category, cut, scope)
        own = [b for b in bets if (b["category"], b["cut"], b["scope"]) == (category, cut, scope)]
        m = {"category": category, "cut": cut, "scope": scope, "n_bets": 0, "n_races": 0,
             "n_meetings": 0, "stake": 0.0, "payout": 0.0, "payout_published": 0.0,
             "roi": math.nan, "lo": math.nan, "hi": math.nan, "roi_published": math.nan,
             "shrinkage": _shrinkage(name, own, rates, seed)}  # fmt: skip
        if len(frame):
            sub = frame[frame[("stake", name)] > 0]
            m.update(n_bets=int(frame[("n", name)].sum()), n_races=len(sub),
                     n_meetings=int(sub["meeting_id"].nunique()),
                     stake=float(sub[("stake", name)].sum()),
                     payout=float(sub[("payout", name)].sum(skipna=False)),
                     payout_published=float(sub[("published", name)].sum()))  # fmt: skip
            if len(sub):
                m["roi_published"] = m["payout_published"] / m["stake"]
            # a floor-bound hit leaves the member's impact-adjusted return unavailable: dropping
            # only those races would bias the ROI toward losses
            if len(sub) and math.isfinite(m["payout"]):
                args = (sub[("payout", name)], sub[("stake", name)], sub["meeting_id"])
                kw = {"n_boot": n_boot, "seed": seed}
                hi = study.rho_interval(*args, alternative="less", **kw)
                lo = study.rho_interval(*args, alternative="greater", **kw)
                m.update(roi=hi.rho, lo=lo.bound, hi=hi.bound)
        members.append(m)
        n = m["n_races"]
        shrink = m["shrinkage"]
        metrics = [FoldMetric("development", "n_bets", float(m["n_bets"]), None, None, n)]
        if m["stake"] > 0 and math.isfinite(m["roi"]):
            metrics.append(FoldMetric("development", "roi", m["roi"], m["lo"], m["hi"], n))
        if m["stake"] > 0:
            metrics.append(FoldMetric("development", "roi_published", m["roi_published"], None,
                                      None, n))  # fmt: skip
        # results, never params: a failed fit and a fitted one are the same variant
        metrics += [
            FoldMetric("development", "n_meetings", float(m["n_meetings"]), None, None, n),
            FoldMetric("development", f"shrinkage_{shrink['status']}", 1.0, None, None, n),
            FoldMetric("development", "shrinkage_n_bets", float(shrink["n_bets"]), None, None, n),
        ]
        params = {"category": category, "cut": cut, "scope": scope, "spec_hash": FROZEN_E2_HASH,
                  "lambda2": study.FITTED, "stake_yen": STAKE,
                  "minutes_before_post": MINUTES_BEFORE_POST}  # fmt: skip
        append(_record(E2_RULE_ID, name, params, components, metrics, data_builds, git_sha))

    result = {
        "n_races": len(frame),
        "n_meetings": int(frame["meeting_id"].nunique()) if len(frame) else 0,
        "excluded": excluded,
        "impact_unavailable": impact_unavailable,
        "members": members,
        "spa": None,
        "stepm": None,
        "max_test": {"status": "not_run", "reason": "no race scored"},
        "bets": bets,
    }
    if len(frame):
        try:
            result.update(_max_test(append, registry, frame, components, data_builds, git_sha,
                                    n_boot, seed, impact_unavailable))  # fmt: skip
        except snooping.FamilySizeMismatch:
            raise
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError) as exc:
            # e.g. arch needs at least 3 meetings: the failed max test is a record too
            error = f"{type(exc).__name__}: {exc}"[:500]
            params = {"family": E2_RULE_ID, "spec_hash": FROZEN_E2_HASH, "error": error,
                      "block_size": BLOCK_SIZE, "stepm_fwer": STEPM_FWER}  # fmt: skip
            append(_record(MAX_TEST_ID, "spa+stepm", params, components, [], data_builds,
                           git_sha, status="failed"))  # fmt: skip
            result["max_test"] = {"status": "failed", "error": error}
    return result


def _max_test(
    append, registry, frame, components, data_builds, sha, n_boot, seed, impact_unavailable
) -> dict:
    """SPA and StepM over the 16 decision-time members, per-race loss stake - payout, meetings as
    clusters. Where a floor-bound winner leaves the impact-adjusted payout unavailable the
    published payout stands in: a 110 yen floor still binds after our stake, and boost-withheld
    races are already excluded. How many races that touched is a metric, never a param."""
    frame = frame.set_index("race_id")
    names = [variant(*key) for key in MEMBERS]
    paid = {n: frame[("payout", n)].where(frame[("payout", n)].notna(), frame[("published", n)])
            for n in names}  # fmt: skip
    losses = pd.DataFrame({n: frame[("stake", n)] - paid[n] for n in names})
    kw = {"clusters": frame["meeting_id"], "order": pd.to_datetime(frame["race_date"]),
          "block_size": BLOCK_SIZE, "n_boot": n_boot, "seed": seed, "registry": registry,
          "family_id": E2_RULE_ID}  # fmt: skip
    bench = pd.Series(0.0, index=losses.index)
    spa = snooping.spa(bench, losses, **kw)
    stepm = snooping.stepm(bench, losses, level=STEPM_FWER, **kw)
    params = {"family": E2_RULE_ID, "spec_hash": FROZEN_E2_HASH,
              "superior": sorted(stepm.superior), "block_size": BLOCK_SIZE,
              "stepm_fwer": STEPM_FWER}  # fmt: skip
    finite = [v if math.isfinite(v) else None for v in (spa.pvalue_lower, spa.pvalue_upper)]
    metrics = [
        FoldMetric("development", "stepm_superior", float(len(stepm.superior)), None, None,
                   stepm.n_clusters),
        FoldMetric("development", "family_size", float(spa.family_size), None, None, spa.n_rules),
        FoldMetric("development", "impact_unavailable_races", float(impact_unavailable), None,
                   None, len(frame)),
    ]  # fmt: skip
    status, max_test = "scored", {"status": "scored"}
    if math.isfinite(spa.pvalue):
        metrics.insert(0, FoldMetric("development", "spa_pvalue", spa.pvalue, *finite,
                                     spa.n_clusters))  # fmt: skip
    else:  # degenerate bootstrap (e.g. one meeting): recorded as failed, not dropped
        status, params["error"] = "failed", "SPA p-value not finite"
        max_test = {"status": "failed", "error": params["error"]}
    append(_record(MAX_TEST_ID, "spa+stepm", params, components, metrics, data_builds, sha,
                   status))  # fmt: skip
    return {
        "spa": {"pvalue": spa.pvalue, "lower": spa.pvalue_lower, "upper": spa.pvalue_upper,
                "n_rules": spa.n_rules, "family_size": spa.family_size,
                "n_clusters": spa.n_clusters},
        "stepm": {"superior": stepm.superior, "family_size": stepm.family_size,
                  "fwer": STEPM_FWER},
        "max_test": max_test,
    }  # fmt: skip


# --------------------------------------------------------------------------- D and the proxy


def d_descriptive(
    races: Iterable[Mapping],
    *,
    lambda2: float,
    holdout_day: Callable[[str, date], bool] = is_holdout_day,
) -> dict:
    """D = sum_S q / sum_S p per race on the final boards, by category: n_races, mean, median;
    and ``incomplete_boards``. A race counts when its final exacta board prices every ordered pair
    of the final Win board's horses, whatever its pool total, post time and boards at t; a race
    with final prices that miss a pair is incomplete and left out of every category; a race
    without final FORECAST prices is neither. Prices for horses off the final Win board (scratched)
    are not required and not used."""
    values: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    incomplete = 0
    for race in races:
        if holdout_day(COUNTRY, race["race_date"]):
            continue
        win = race["final_win_odds"]
        board = race["final_exacta_odds"]
        if len(win) < 2 or not board:
            continue
        if any(board.get(pair) is None for pair in permutations(win, 2)):
            incomplete += 1
            continue
        p = _win_p(win)
        dividends = {tuple(c): _yen(o) for c, o in board.items()
                     if o is not None and c[0] in p and c[1] in p and c[0] != c[1]}  # fmt: skip
        ratios = race_ratios(p, dividends, favourite=favourite(win), rate=race["rate"],
                             lambda2=lambda2).categories  # fmt: skip
        for c in CATEGORIES:
            if not math.isnan(ratios[c]):
                values[c].append(ratios[c])
    out: dict = {
        c: {"n_races": len(v), "mean": statistics.mean(v) if v else math.nan,
            "median": statistics.median(v) if v else math.nan}
        for c, v in values.items()
    }  # fmt: skip
    out["incomplete_boards"] = incomplete
    return out


def proxy_agreement(
    races: Iterable[Mapping], *, holdout_day: Callable[[str, date], bool] = is_holdout_day
) -> dict:
    """The race-card proxy (race 1-4 or DEBUT/MAIDEN) against the lower tertile of the final
    pool_total within each race day (all venues together, ``lower_tertile_by_day``)."""
    kept = [r for r in races if not holdout_day(COUNTRY, r["race_date"])]
    tertile = lower_tertile_by_day(kept, key="pool_total")
    counts = dict.fromkeys(("both_less", "proxy_only", "tertile_only", "neither"), 0)
    days = set()
    for r in kept:
        if r["race_id"] not in tertile:
            continue
        days.add(r["race_date"])
        proxy = less_popular(r["race_no"], r["race_class"])
        less = tertile[r["race_id"]]
        key = ("both_less" if less else "proxy_only") if proxy else (
            "tertile_only" if less else "neither")  # fmt: skip
        counts[key] += 1
    n = sum(counts.values())
    agree = (counts["both_less"] + counts["neither"]) / n if n else math.nan
    return {"n_races": n, "n_days": len(days), **counts, "agreement": agree}


# --------------------------------------------------------------------------- the logged boards


def _combo(text) -> tuple[int, int] | None:
    try:
        first, second = (int(part) for part in str(text).split(","))
    except (TypeError, ValueError):
        return None
    return (first, second) if first != second else None


def _exacta_at(rows: list[dict], at_utc, scratched) -> dict[tuple[int, int], float]:
    """The latest FORECAST price per combination available at `at_utc` (board-at-t rules)."""
    best: dict[tuple[int, int], tuple] = {}
    for row in rows:
        if row.get("pool") != FORECAST or row.get("combination") == odds_log.POOL_TOTAL:
            continue
        available, combo = snapshot._available(row), _combo(row.get("combination"))
        if available is None or combo is None or (at_utc is not None and available > at_utc):
            continue
        if combo not in best or available >= best[combo][0]:
            best[combo] = (available, row)
    return {c: float(r["odds"]) for c, (_, r) in best.items()
            if snapshot._priced(r) and not set(c) & scratched}  # fmt: skip


def _exacta_final(rows: list[dict], scratched) -> tuple[dict, float | None]:
    """The final exacta board and pool total: the first capture final for FORECAST at or after the
    WIN anchor, else the latest one before it (as final_board takes PLACE and Wide). No such
    capture, or no total in it: ({}, None) or (board, None), never a pre-final total."""
    captures: dict = {}
    for row in rows:
        snap = odds_log._instant(row.get("snapshot_utc"))
        if snap is not None and snapshot._available(row) is not None:
            captures.setdefault(snap, []).append(row)
    snaps = sorted(captures)
    anchor = next((s for s in snaps if snapshot._win_rows_final(captures[s])), None)
    finals = [s for s in snaps if snapshot._pool_rows_final(captures[s], FORECAST)]
    if anchor is None or not finals:
        return {}, None
    source = next((s for s in finals if s >= anchor), finals[-1])
    return _exacta_at(captures[source], None, scratched), _forecast_total(captures[source])


def _forecast_total(rows: list[dict], at_utc=None) -> float | None:
    """The latest FORECAST pool total among `rows` available by `at_utc` (any time when None)."""
    best = None
    for row in rows:
        if row.get("pool") != FORECAST or row.get("combination") != odds_log.POOL_TOTAL:
            continue
        value, available = row.get("investment"), snapshot._available(row)
        if value is None or available is None or (at_utc is not None and available > at_utc):
            continue
        if best is None or available >= best[0]:
            best = (available, float(value))
    return None if best is None else best[1]


def _sha256_prefix(path: Path, end: int) -> str:
    """sha256 hex of the first `end` bytes of `path` (the whole members read), in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        left = end
        while left > 0:
            chunk = fh.read(min(left, 1 << 20))
            if not chunk:
                break
            digest.update(chunk)
            left -= len(chunk)
    return digest.hexdigest()


def _logged(minutes_before: int, holdout_day) -> tuple[dict[str, dict], dict]:
    """Per race with FORECAST rows on a development day: the final boards, and the boards at t
    when it has a decision time (``no_decision_time`` says why not); and what was read: per log
    its consumed content's hash and its resolved post times."""
    root = paths.data_dir() / "odds_snapshots" / "jra"
    info = {"days_read": 0, "days_holdout_skipped": 0, "logs": [], "dropped": {}}
    found: dict[str, dict] = {}
    if not root.is_dir():
        return found, info
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            day = date.fromisoformat(folder.name)
        except ValueError:
            continue
        if holdout_day(COUNTRY, day):  # D14: the folder is never opened
            info["days_holdout_skipped"] += 1
            continue
        info["days_read"] += 1
        for log in sorted(folder.glob("*.jsonl.gz")):
            venue = log.name.split(".")[0]
            by_race: dict[str, list[dict]] = {}
            end = 0  # the byte after the last whole member read: the consumed content
            for offset, member in odds_log.iter_members_with_offsets(log):  # one capture at a time
                end = offset
                for row in member:
                    pool = row.get("pool")
                    if pool in ("WIN", FORECAST) or (
                        pool == "PLACE" and row.get("status_raw") == snapshot.SCRATCHED
                    ):
                        by_race.setdefault(row.get("race_id"), []).append(row)
            post_times = snapshot.post_times_for(day, venue)
            resolved = json.dumps({str(k): v.isoformat() for k, v in sorted(post_times.items())})
            info["logs"].append(f"{day.isoformat()}/{log.name}:{_sha256_prefix(log, end)}:"
                                f"{hashlib.sha256(resolved.encode()).hexdigest()}")  # fmt: skip
            for race_id, rows in by_race.items():
                if not any(r.get("pool") == FORECAST for r in rows):
                    continue
                at, reason = _decision_time(race_id, post_times, minutes_before)
                pool_t = None if at is None else _forecast_total(rows, at)
                got = _race_boards(race_id, rows, at, minutes_before)
                found[race_id] = {"race_date": day, "pool_total_t": pool_t,
                                  "no_decision_time": reason, **got}  # fmt: skip
    return found, info


def _decision_time(race_id, post_times, minutes_before):
    try:
        post = post_times.get(int(str(race_id).rsplit(":", 1)[1]))
    except (IndexError, ValueError):
        return None, "unreadable_race_id"
    if post is None:
        return None, "no_post_time"
    return post - timedelta(minutes=minutes_before), None


def _race_boards(race_id, rows, at, minutes_before) -> dict:
    """The boards at t ({} when there is no t or nothing was available then: E2 excludes it) and
    the final boards (D and the proxy use them whatever the boards at t and the post time)."""
    post = None if at is None else at + timedelta(minutes=minutes_before)
    win_t, exacta_t = {}, {}
    if at is not None:
        board = snapshot.board_at(rows, race_id, at, post_utc=post)
        if board is not None and board.win:
            win_t = dict(board.win)
            exacta_t = _exacta_at(rows, at, board.scratched)
    final = snapshot.final_board(rows, race_id, post_utc=post)
    final_win, final_exacta, pool = {}, {}, None  # no final board: excluded as no_final_board
    if final is not None:
        final_win = dict(final.win)
        final_exacta, pool = _exacta_final(rows, final.scratched)  # prices kept without a total
    return {"win_odds_t": win_t, "exacta_odds_t": exacta_t,
            "final_win_odds": final_win, "final_exacta_odds": final_exacta,
            "pool_total": pool}  # fmt: skip


def _load(minutes_before: int, holdout_day) -> tuple[list[dict], dict]:
    found, info = _logged(minutes_before, holdout_day)
    if not found:
        return [], info
    base = paths.processed_dir("jra")
    ids = sorted(found)
    meta = pq.read_table(
        base / "races.parquet",
        columns=["race_id", "meeting_id", "race_no", "race_class", "race_class_raw"],
        filters=pc.field("race_id").isin(ids),
    ).to_pylist()
    dividends = pq.read_table(
        base / "dividends.parquet",
        columns=["race_id", "combination", "dividend", "is_refund"],
        filters=pc.field("race_id").isin(ids) & (pc.field("pool") == FORECAST),
    ).to_pylist()
    paying: dict[str, list] = {}
    for d in dividends:
        combo = tuple(int(h) for h in str(d["combination"]).split(","))
        paying.setdefault(d["race_id"], []).append(
            (combo, float(d["dividend"]), bool(d["is_refund"]))
        )
    starters: dict[str, int] = {}
    runners = base / "runners.parquet"
    runner_rows = []
    if runners.exists():
        runner_rows = pq.read_table(runners, columns=["race_id", "is_scratched"],
                                    filters=pc.field("race_id").isin(ids)).to_pylist()  # fmt: skip
        for r in runner_rows:
            starters[r["race_id"]] = starters.get(r["race_id"], 0) + (not r["is_scratched"])
    out = []
    for m in sorted(meta, key=lambda m: m["race_id"]):
        rid = m["race_id"]
        t = takeout.race_takeout(rid, FORECAST, race_class_raw=m["race_class_raw"])
        race = {"race_id": rid, "meeting_id": m["meeting_id"], "race_no": int(m["race_no"]),
                "race_class": m["race_class"], "rate": t.payout_rate,
                "boost_may_be_withheld": t.boost_may_be_withheld, **found[rid],
                "exacta": paying.get(rid, [])}  # fmt: skip
        if rid in starters:
            race["n_starters"] = starters[rid]
        else:  # no runner rows in the store: the runners priced on the final Win board
            race["n_starters"] = len(race["final_win_odds"])
            info["n_starters_from_final_board"] = info.get("n_starters_from_final_board", 0) + 1
        out.append(race)
    info["dropped"]["not_in_store"] = len(found) - len(out)
    meetings = sorted({r["meeting_id"] for r in out})
    card = pq.read_table(base / "races.parquet", columns=["race_id", "meeting_id"],
                         filters=pc.field("meeting_id").isin(meetings)).to_pylist()  # fmt: skip
    loaded = {r["race_id"] for r in out}
    info["meetings_with_every_race"] = sorted(
        m for m in meetings if all(c["race_id"] in loaded for c in card if c["meeting_id"] == m)
    )
    info["store_rows"] = {"races": _rows_digest([*meta, *card]),
                          "runners": _rows_digest(runner_rows),
                          "dividends": _rows_digest(dividends)}  # fmt: skip
    return out, info


def load_logged_races(*, minutes_before: int = MINUTES_BEFORE_POST) -> list[dict]:
    """The race mappings score_e2 takes, from the #130 odds log and the store (development days).
    A race without a post time is loaded with its final boards and ``no_decision_time`` set."""
    return _load(minutes_before, is_holdout_day)[0]


# --------------------------------------------------------------------------- the ordering fit


def _ordering_artifact() -> dict:
    """The ordering artifact ``current.json`` names, pinned by the sha256 of the bytes its lambda2
    is read from, and its training window: the race dates (``jra:<YYYY-MM-DD>:...``) of the
    out-of-fold file its manifest names."""
    model_dir = paths.data_dir().joinpath(*MODEL_DIR)
    names = json.loads((model_dir / "current.json").read_text(encoding="utf-8"))
    path = model_dir / names["ordering"]
    raw = path.read_bytes()
    fit = json.loads(raw)
    manifest_path = model_dir / f"manifest-ordering-{fit['fit_id']}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    oof = Path(manifest["oof"]["path"])
    if not oof.is_absolute():
        oof = model_dir / oof
    race_ids = pc.unique(pq.read_table(oof, columns=["race_id"])["race_id"]).to_pylist()
    days = {date.fromisoformat(str(rid).split(":")[1]) for rid in race_ids}
    if not days:
        raise ValueError(f"the ordering fit's out-of-fold file {oof.name} has no race")
    return {"file": path.name, "fit_id": str(fit["fit_id"]), "lambda2": float(fit["lambdas"][0]),
            "sha256": hashlib.sha256(raw).hexdigest(), "oof": oof.name,
            "trained_through": max(days)}  # fmt: skip


# --------------------------------------------------------------------------- the command


def _rows_digest(rows: Iterable[Mapping]) -> str:
    """16 hex digits of sha256 over the rows read, order-free: other rows of the table (holdout
    or unlogged races outside the loaded meetings) leave it unchanged."""
    lines = sorted(json.dumps(r, sort_keys=True, default=str) for r in rows)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def _data_builds(info: dict, ordering: Mapping | None = None) -> dict[str, str]:
    """The builds a run consumed, the same four keys whether or not any race was logged. odds_log:
    the consumed log content and resolved post times of every log read, and the content hash of
    the ordering artifact that turned them into selections; per store table a fingerprint of the
    logged development races' rows read from it."""
    digest = hashlib.sha256("\n".join(info["logs"]).encode()).hexdigest()[:16]
    refresh = digest if ordering is None else f"{digest}+ordering-{ordering['sha256'][:12]}"
    builds = {"odds_log": f"odds_snapshots/jra/{FORECAST}@{refresh}"}
    consumed = info.get("store_rows", {})
    for name in STORE_TABLES:
        rows = consumed.get(name, _rows_digest([]))
        builds[f"jra_{name}"] = f"processed/jra/{name}.parquet@{rows}"
    return builds


def _run_digest(data_builds: Mapping[str, str], components: Mapping[str, str]) -> str:
    """12 hex digits naming a run's inputs and model, so a rescore at the same commit on more logs
    (or another ordering artifact) writes its own report beside the earlier one."""
    text = json.dumps({"data_builds": dict(data_builds), "components": dict(components)},
                      sort_keys=True)  # fmt: skip
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _num(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _render(summary: dict) -> str:
    lines = [f"# #281 E2 ratio-cut rules, {E2_RULE_ID} ({summary['git_sha'][:12]})", "",
             f"Races scored {summary['n_races']}, meetings {summary['n_meetings']}; "
             f"excluded {json.dumps(summary['excluded'], sort_keys=True)}; winning dividend "
             f"floor-bound (published 100 or 110 yen: impact unavailable) "
             f"{summary['impact_unavailable']}.", "",
             "## Members", "",
             "Every member is a decision-time rule (race card and the boards at T-2). ROI is at "
             "the own-impact dividend, unavailable for a member with a floor-bound hit; published "
             "ROI is at the published dividend. Shrinkage is fitted per member (segment_roi, "
             "exacta odds bands as fixed effects).", "",
             "| category | cut | scope | n_bets | races | meetings | ROI | lo | hi | "
             "published ROI | shrinkage | P(beats takeout) |",
             "|" + " --- |" * 12]  # fmt: skip
    for m in summary["members"]:
        shrink = m["shrinkage"]
        lines.append(f"| {m['category']} | {m['cut']} | {m['scope']} | {m['n_bets']} | "
                     f"{m['n_races']} | {m['n_meetings']} | {_num(m['roi'])} | {_num(m['lo'])} | "
                     f"{_num(m['hi'])} | {_num(m['roi_published'])} | {shrink['status']} | "
                     f"{_num(shrink.get('p_beats_takeout'))} |")  # fmt: skip
    max_test, spa = summary["max_test"], summary["spa"]
    lines += ["", "## Max test (SPA + StepM over the 16 members)", "",
              f"Status: {max_test['status']}."]  # fmt: skip
    if max_test.get("error") or max_test.get("reason"):
        lines.append(f"Reason: {max_test.get('error') or max_test.get('reason')}")
    if spa:
        superior = summary["stepm"]["superior"] if summary["stepm"] else []
        lines.append(f"SPA p-value {_num(spa['pvalue'])} over {spa['n_rules']} rules (family "
                     f"size {spa['family_size']}, {spa['n_clusters']} meetings); StepM superior: "
                     f"{', '.join(superior) or 'none'}. A floor-bound winner's race enters at "
                     f"its published payout.")  # fmt: skip
    d = summary["d"]
    lines += ["", "## D, the full-board category ratio on the final boards", "",
              f"Final boards missing a priced pair of the final field (left out): "
              f"{d['incomplete_boards']}.", "",
              "| category | races | mean | median |", "| --- | --- | --- | --- |"]  # fmt: skip
    for category in CATEGORIES:
        row = d[category]
        lines.append(f"| {category} | {row['n_races']} | {_num(row['mean'])} | "
                     f"{_num(row['median'])} |")  # fmt: skip
    proxy = summary["proxy"]
    lines += ["", "## Proxy agreement (race 1-4 or maiden against the day's final-pool tertile)",
              "", f"Races {proxy['n_races']} over {proxy['n_days']} days: both less popular "
              f"{proxy['both_less']}, proxy only {proxy['proxy_only']}, tertile only "
              f"{proxy['tertile_only']}, neither {proxy['neither']}; agreement "
              f"{_num(proxy['agreement'])}."]  # fmt: skip
    edges = ", ".join(f"{e:g}" for e in EXACTA_ODDS_BANDS)
    lines += ["", "## Shrinkage per member", "",
              f"Band edges {edges} (final-board odds); one constant segment column per fit, the "
              f"member's variant."]  # fmt: skip
    for m in summary["members"]:
        shrink, name = m["shrinkage"], variant(m["category"], m["cut"], m["scope"])
        if shrink["status"] == "fitted":
            effects = ", ".join(f"{b} {_num(v)}" for b, v in shrink["band_effects"].items())
            lines.append(f"- {name}: fitted on {shrink['n_bets']} bets; band effects "
                         f"(ln omega) {effects}.")  # fmt: skip
        else:
            lines.append(f"- {name}: {shrink['status']} ({shrink['reason']}).")
    return "\n".join(lines) + "\n"


def exacta_rules_command() -> None:
    """Score #131's E2 ratio-cut rules on the #130-logged development boards and record the run."""
    sha, registry = exacta_run._git_sha(), default_registry(COUNTRY)
    append = exacta_run._appender(registry)
    data_builds: dict[str, str] = {}
    try:
        races, info = _load(MINUTES_BEFORE_POST, is_holdout_day)
        data_builds = _data_builds(info)
        ordering = _ordering_artifact()
        data_builds = _data_builds(info, ordering)
        components = {"ordering": ordering["fit_id"]}
        result = score_e2(
            races, registry=registry, git_sha=sha, lambda2=ordering["lambda2"],
            data_builds=data_builds, components=components,
            trained_through=ordering["trained_through"],
        )  # fmt: skip
        summary = {
            "git_sha": sha, "family": E2_RULE_ID, "spec": study.E2_ID,
            "spec_hash": FROZEN_E2_HASH, "lambda2_full_development": ordering["lambda2"],
            "ordering_fit": ordering["fit_id"], "ordering_artifact": ordering,
            "read": {k: v for k, v in info.items() if k not in ("logs", "store_rows")},
            "logs_read": len(info["logs"]), "data_builds": data_builds,
            "logged_meetings_with_forecast": len({r["meeting_id"] for r in races}),
            "logged_races_loaded": len(races), **result,
            "d": d_descriptive(races, lambda2=ordering["lambda2"]),
            "proxy": proxy_agreement(races),
        }  # fmt: skip
    except Exception as exc:  # the failed run is a registry record too
        error = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        append(_record(MAX_TEST_ID, "run", error, {}, [], data_builds, sha, status="failed"))
        raise
    summary = _clean(summary)
    out = paths.reports_dir() / "japan" / "281-exacta-rules"
    stem = f"e2-{sha[:12]}-{_run_digest(data_builds, components)}"
    write_report(out / f"{stem}.json", json.dumps(summary, indent=1), sources=SOURCES)
    write_report(out / f"{stem}.md", _render(summary), sources=SOURCES)
    write_summary(registry, summary_dir(COUNTRY))
    keys = ("read", "data_builds", "logged_meetings_with_forecast", "n_races", "n_meetings",
            "excluded", "impact_unavailable", "spa", "stepm", "max_test", "d", "proxy")  # fmt: skip
    typer.echo(json.dumps({k: summary[k] for k in keys}, indent=1))
