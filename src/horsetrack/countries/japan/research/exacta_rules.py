"""`horsetrack japan exacta-rules`: #131's E2 ratio-cut rules scored on #130-logged boards (#281).

docs/specs/exacta-overload-prereg.md sections 5-7, family ``japan/131-exacta-rule-v3`` (frozen
hash ``FROZEN_E2_HASH``; the favourite and the category from the board at T-2). With the owner's
reframe of 2026-11-27 nothing is refused for small n: the frozen ``min_meetings: 100`` is
superseded and zero logged meetings give an n = 0 result.

- Selection at t = post - 2 min, from the boards at t only: p_i the normalised 1/odds of the Win
  board, p_c their discounted Harville exacta, q_c the exacta board's implied share, the favourite
  the lowest Win odds. Member (X, k, scope) buys c of category X when q_c / p_c < k.
- Scope "headline": the favourite's Win odds at t >= 3.0 and the race "less popular", the lower
  tertile of its FORECAST pool total at t within its logged day (all venues; every logged race
  with a total at t, whatever its E2 eligibility). The race 1-4 / maiden proxy is only compared
  with the final-pool tertile (``proxy_agreement``).
- Each member stakes 100 yen per selected combination and is settled at the final dividend with
  its own impact from the payout identity on the race's final logged FORECAST pool (#283): the
  published dividend's 10 yen breakage is added back, and the result is rounded down to 10 yen
  with the 110 yen minimum (D18). A race whose log has no final FORECAST capture is not scored.
- Shrinkage (spec section 7): ``core.eval.segment_roi`` on the members' bets, fixed effects the
  final-board exacta odds bands of ``EXACTA_ODDS_BANDS`` (< 10, 10-30, 30-100, >= 100: roughly
  the favourite pairs, the middle of the board, the outsiders and the long shots), crossed random
  effects category, cut and scope.
- D, the full-board category ratio on the final boards, and the agreement of the "less popular"
  proxy with the lower final-pool tertile of each race day, over every loaded logged race.

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
FROZEN_E2_HASH = "9104e173371668a34b4594b6e4970546649b886cdccc4e726ec9749188701f0d"  # rule-v3
MAX_TEST_ID = f"{study.E2_ID}/max-test"
COUNTRY = "japan"
SOURCES = ("jra",)
FORECAST = "FORECAST"
TICKET = 281
FOLD_SCHEME = "development-days:#130-logged boards at T-2"
PAYOUT_STEP = 10.0  # FORECAST pays per 100 yen rounded down to 10 yen,
MIN_DIVIDEND = 110.0  # never below 110 yen
HEADLINE_FAV_ODDS = 3.0
EXACTA_ODDS_BANDS = (10.0, 30.0, 100.0)  # shrinkage fixed effects: final-board exacta odds bands
SHRINK_DRAWS = 4000
STORE_TABLES = ("races", "runners", "dividends")


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
    """The dividend that pays: ``own_impact_raw`` rounded down to 10 yen, never below 110 yen."""
    raw = own_impact_raw(dividend, pool=pool, n_bets=n_bets, stake=stake, takeout=takeout)
    return max(MIN_DIVIDEND, PAYOUT_STEP * math.floor(raw / PAYOUT_STEP + 1e-9))


def settle(
    selected: Iterable[tuple[int, int]],
    winner: tuple[int, int],
    dividend: float,
    *,
    pool: float,
    stake: float = STAKE,
    takeout: float = TAKEOUT,
) -> dict:
    """One race of one member: n_bets, stake (yen) and payout (yen) at the own-impact dividend."""
    selected = {tuple(c) for c in selected}
    n = len(selected)
    payout = 0.0
    if n and tuple(winner) in selected:
        paid = own_impact_dividend(dividend, pool=pool, n_bets=n, stake=stake, takeout=takeout)
        payout = stake / 100 * paid
    return {"n_bets": n, "stake": stake * n, "payout": payout}


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


def race_scopes(win_odds: Mapping[int, float], *, less_popular: bool) -> frozenset:
    """Always "all"; "headline" when the favourite's Win odds at t are 3.0 or more and the race is
    in the lower pool tertile of its logged day (v3's frozen scope)."""
    headline = win_odds[favourite(win_odds)] >= HEADLINE_FAV_ODDS and bool(less_popular)
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


def _shrinkage(bets: list[dict], rates: Mapping[str, float], seed: int) -> dict:
    """segment_roi on every member's bets (one row per member x race x selected combination with
    a final-board price), fixed effects the exacta odds bands, crossed random effects category,
    cut and scope. The members' bets overlap (nested cuts, headline within all), so one bet
    enters up to eight rows and the posterior is narrower than the data warrant."""
    info = {"band_edges": list(EXACTA_ODDS_BANDS), "segment_cols": ["category", "cut", "scope"]}
    rows = [{"category": b["category"], "cut": f"<{b['cut']}", "scope": b["scope"],
             "odds": b["odds"], "won": int(b["won"]), "takeout": 1 - rates[b["race_id"]]}
            for b in bets if b["odds"] is not None]  # fmt: skip
    if not rows:
        reason = "no bet" if not bets else "no bet has a price on the final board"
        return {"status": "not_run", "reason": reason, **info}
    frame = pd.DataFrame(rows)
    frame["band"] = frame["odds"].map(odds_band)
    try:
        model = fit_segment_roi(frame, segment_cols=info["segment_cols"], odds_band_col="band",
                                takeout_col="takeout", seed=seed, draws=SHRINK_DRAWS)  # fmt: skip
        p_beats = {c: model.p_beats_takeout({"category": c})
                   for c in sorted(frame["category"].unique())}  # fmt: skip
    except (RuntimeError, ValueError, KeyError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:500],
                "n_bets": len(frame), **info}  # fmt: skip
    return {"status": "fitted", "n_bets": len(frame),
            "bands": {str(k): int(v) for k, v in frame["band"].value_counts().items()},
            "band_effects": model.band_effects, "tau": model.tau, "tau_ess": model.tau_ess,
            "tau_rejected": model.tau_rejected, "p_beats_takeout": p_beats, **info}  # fmt: skip


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
) -> dict:
    """Score the 16 E2 members on `races`, record one record per member and, when a race is scored,
    one SPA + StepM record under MAX_TEST_ID; a record already in the registry is not appended
    again. Refuses unless E2 is frozen at its spec's hash."""
    frozen = registry.frozen_spec_hash(study.E2_ID)
    expected = _spec_hash(study.family_specs()[study.E2_ID])
    if frozen is None or frozen != expected or frozen != FROZEN_E2_HASH:
        raise RuntimeError(f"{study.E2_ID} is not frozen at {FROZEN_E2_HASH} (found {frozen})")
    components = dict(components or {"ordering": f"lambda2={lambda2:.6g}"})
    append = exacta_run._appender(registry)

    excluded: dict[str, int] = {}
    rows, bets, rates = [], [], {}
    for race in races:
        if holdout_day(COUNTRY, race["race_date"]):
            excluded["holdout_day"] = excluded.get("holdout_day", 0) + 1
            continue
        pool = race.get("pool_total")
        if not _positive(pool):
            reason = "no_final_board"
        else:
            reason = study.exclusion_reason(
                n_starters=int(race["n_starters"]),
                boost_may_be_withheld=bool(race["boost_may_be_withheld"]),
                exacta=list(race["exacta"]),
            )
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        win, final_exacta = race["win_odds_t"], race["final_exacta_odds"]
        rate = float(race["rate"])
        rates[race["race_id"]] = rate
        chosen = select_combinations(win, race["exacta_odds_t"], rate=rate, lambda2=lambda2)
        scopes = race_scopes(win, less_popular=race["less_popular"])
        ((winner, dividend, _),) = race["exacta"]
        winner = tuple(winner)
        row = {"race_id": race["race_id"], "meeting_id": race["meeting_id"],
               "race_date": race["race_date"]}  # fmt: skip
        for category, cut, scope in MEMBERS:
            sel = chosen[(category, cut)] if scope in scopes else frozenset()
            got = settle(sel, winner, dividend, pool=pool, takeout=1 - rate)
            name = variant(category, cut, scope)
            row[("n", name)], row[("stake", name)] = got["n_bets"], got["stake"]
            row[("payout", name)] = got["payout"]
            for combo in sorted(sel):
                odds = final_exacta.get(combo)
                bets.append({"category": category, "cut": cut, "scope": scope,
                             "race_id": race["race_id"], "meeting_id": race["meeting_id"],
                             "combination": combo, "odds": None if odds is None else float(odds),
                             "won": combo == winner, "stake": float(STAKE),
                             "payout": got["payout"] if combo == winner else 0.0})  # fmt: skip
        rows.append(row)
    frame = pd.DataFrame(rows)

    members = []
    for category, cut, scope in MEMBERS:
        name = variant(category, cut, scope)
        m = {"category": category, "cut": cut, "scope": scope, "n_bets": 0, "n_races": 0,
             "n_meetings": 0, "stake": 0.0, "payout": 0.0, "roi": math.nan, "lo": math.nan,
             "hi": math.nan}  # fmt: skip
        if len(frame):
            sub = frame[frame[("stake", name)] > 0]
            m.update(n_bets=int(frame[("n", name)].sum()), n_races=len(sub),
                     n_meetings=int(sub["meeting_id"].nunique()),
                     stake=float(sub[("stake", name)].sum()),
                     payout=float(sub[("payout", name)].sum()))  # fmt: skip
            if len(sub):
                args = (sub[("payout", name)], sub[("stake", name)], sub["meeting_id"])
                kw = {"n_boot": n_boot, "seed": seed}
                hi = study.rho_interval(*args, alternative="less", **kw)
                lo = study.rho_interval(*args, alternative="greater", **kw)
                m.update(roi=hi.rho, lo=lo.bound, hi=hi.bound)
        members.append(m)
        metrics = [FoldMetric("development", "n_bets", float(m["n_bets"]), None, None,
                              m["n_races"])]  # fmt: skip
        if m["stake"] > 0:
            metrics.append(
                FoldMetric("development", "roi", m["roi"], m["lo"], m["hi"], m["n_races"])
            )
        params = {"category": category, "cut": cut, "scope": scope, "spec_hash": FROZEN_E2_HASH,
                  "lambda2": study.FITTED, "stake_yen": STAKE,
                  "minutes_before_post": MINUTES_BEFORE_POST}  # fmt: skip
        append(_record(study.E2_ID, name, params, components, metrics, data_builds, git_sha))

    result = {
        "n_races": len(frame),
        "n_meetings": int(frame["meeting_id"].nunique()) if len(frame) else 0,
        "excluded": excluded,
        "members": members,
        "spa": None,
        "stepm": None,
        "max_test": {"status": "not_run", "reason": "no race scored"},
        "bets": bets,
        "shrinkage": _shrinkage(bets, rates, seed),
    }
    if len(frame):
        try:
            result.update(_max_test(append, registry, frame, components, data_builds, git_sha,
                                    n_boot, seed))  # fmt: skip
        except snooping.FamilySizeMismatch:
            raise
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError) as exc:
            # e.g. arch needs at least 3 meetings: the failed max test is a record too
            error = f"{type(exc).__name__}: {exc}"[:500]
            params = {"family": study.E2_ID, "spec_hash": FROZEN_E2_HASH, "error": error,
                      "block_size": BLOCK_SIZE, "stepm_fwer": STEPM_FWER}  # fmt: skip
            append(_record(MAX_TEST_ID, "spa+stepm", params, components, [], data_builds,
                           git_sha, status="failed"))  # fmt: skip
            result["max_test"] = {"status": "failed", "error": error}
    return result


def _max_test(append, registry, frame, components, data_builds, sha, n_boot, seed) -> dict:
    """SPA and StepM over the 16 members, per-race loss stake - payout, meetings as clusters."""
    frame = frame.set_index("race_id")
    names = [variant(*key) for key in MEMBERS]
    losses = pd.DataFrame({n: frame[("stake", n)] - frame[("payout", n)] for n in names})
    kw = {"clusters": frame["meeting_id"], "order": pd.to_datetime(frame["race_date"]),
          "block_size": BLOCK_SIZE, "n_boot": n_boot, "seed": seed, "registry": registry,
          "family_id": study.E2_ID}  # fmt: skip
    bench = pd.Series(0.0, index=losses.index)
    spa = snooping.spa(bench, losses, **kw)
    stepm = snooping.stepm(bench, losses, level=STEPM_FWER, **kw)
    params = {"family": study.E2_ID, "spec_hash": FROZEN_E2_HASH,
              "superior": sorted(stepm.superior), "block_size": BLOCK_SIZE,
              "stepm_fwer": STEPM_FWER}  # fmt: skip
    finite = [v if math.isfinite(v) else None for v in (spa.pvalue_lower, spa.pvalue_upper)]
    metrics = [
        FoldMetric("development", "stepm_superior", float(len(stepm.superior)), None, None,
                   stepm.n_clusters),
        FoldMetric("development", "family_size", float(spa.family_size), None, None, spa.n_rules),
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
) -> dict[str, dict]:
    """D = sum_S q / sum_S p per race on the final boards, by category: n_races, mean, median.
    A race without a final FORECAST capture (pool_total None) is left out."""
    values: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    for race in races:
        if holdout_day(COUNTRY, race["race_date"]) or not _positive(race.get("pool_total")):
            continue
        win = race["final_win_odds"]
        if len(win) < 2:
            continue
        p = _win_p(win)
        dividends = {tuple(c): _yen(o) for c, o in race["final_exacta_odds"].items()
                     if o is not None and c[0] in p and c[1] in p and c[0] != c[1]}  # fmt: skip
        ratios = race_ratios(p, dividends, favourite=favourite(win), rate=race["rate"],
                             lambda2=lambda2).categories  # fmt: skip
        for c in CATEGORIES:
            if not math.isnan(ratios[c]):
                values[c].append(ratios[c])
    return {
        c: {"n_races": len(v), "mean": statistics.mean(v) if v else math.nan,
            "median": statistics.median(v) if v else math.nan}
        for c, v in values.items()
    }  # fmt: skip


def proxy_agreement(
    races: Iterable[Mapping], *, holdout_day: Callable[[str, date], bool] = is_holdout_day
) -> dict:
    """The frozen proxy (race 1-4 or DEBUT/MAIDEN) against the lower tertile of the final
    pool_total within each race day (all venues together, ``lower_tertile_by_day``)."""
    kept = [r for r in races if not holdout_day(COUNTRY, r["race_date"])]
    tertile = lower_tertile_by_day(kept, key="pool_total")
    counts = dict.fromkeys(("both_less", "proxy_only", "tertile_only", "neither"), 0)
    days = set()
    for r in kept:
        if r["race_id"] not in tertile:
            continue
        days.add(r["race_date"])
        proxy = r["race_no"] <= 4 or study.CLASS_GROUPS.get(r["race_class"]) == "maiden"
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


def _logged(minutes_before: int, holdout_day) -> tuple[dict[str, dict], dict]:
    """Per race with FORECAST rows on a development day: the boards, less_popular set over every
    logged race of the day with a pool total at t; and what was read."""
    root = paths.data_dir() / "odds_snapshots" / "jra"
    info = {"days_read": 0, "days_holdout_skipped": 0, "logs": [], "dropped": {}}
    found: dict[str, dict] = {}
    totals_t: list[dict] = []  # every logged race's pool total at t, for the day tertile
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
            info["logs"].append(f"{day.isoformat()}/{log.name}:{log.stat().st_size}")
            by_race: dict[str, list[dict]] = {}
            for member in odds_log.iter_members(log):  # one capture held at a time
                for row in member:
                    pool = row.get("pool")
                    if pool in ("WIN", FORECAST) or (
                        pool == "PLACE" and row.get("status_raw") == snapshot.SCRATCHED
                    ):
                        by_race.setdefault(row.get("race_id"), []).append(row)
            post_times = snapshot.post_times_for(day, venue)
            for race_id, rows in by_race.items():
                if not any(r.get("pool") == FORECAST for r in rows):
                    continue
                at, reason = _decision_time(race_id, post_times, minutes_before)
                pool_t = None if at is None else _forecast_total(rows, at)
                totals_t.append({"race_id": race_id, "race_date": day, "pool_total_t": pool_t})
                if reason is None:
                    reason, got = _race_boards(race_id, rows, at, minutes_before)
                if reason:
                    info["dropped"][reason] = info["dropped"].get(reason, 0) + 1
                    continue
                found[race_id] = {"race_date": day, "pool_total_t": pool_t, **got}
    less = lower_tertile_by_day(totals_t, key="pool_total_t")
    for race_id, race in found.items():
        race["less_popular"] = less.get(race_id, False)
    return found, info


def _decision_time(race_id, post_times, minutes_before):
    try:
        post = post_times.get(int(str(race_id).rsplit(":", 1)[1]))
    except (IndexError, ValueError):
        return None, "unreadable_race_id"
    if post is None:
        return None, "no_post_time"
    return post - timedelta(minutes=minutes_before), None


def _race_boards(race_id, rows, at, minutes_before) -> tuple[str | None, dict]:
    post = at + timedelta(minutes=minutes_before)
    board = snapshot.board_at(rows, race_id, at, post_utc=post)
    if board is None or not board.win:
        return "no_win_board_at_t", {}
    exacta_t = _exacta_at(rows, at, board.scratched)
    if not exacta_t:
        return "no_exacta_board_at_t", {}
    final = snapshot.final_board(rows, race_id, post_utc=post)
    final_win, final_exacta, pool = {}, {}, None  # no final board: excluded as no_final_board
    if final is not None:
        final_win = dict(final.win)
        final_exacta, pool = _exacta_final(rows, final.scratched)
        if pool is None:
            final_exacta = {}
    return None, {"win_odds_t": dict(board.win), "exacta_odds_t": exacta_t,
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
    if runners.exists():
        for r in pq.read_table(runners, columns=["race_id", "is_scratched"],
                               filters=pc.field("race_id").isin(ids)).to_pylist():  # fmt: skip
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
    return out, info


def load_logged_races(*, minutes_before: int = MINUTES_BEFORE_POST) -> list[dict]:
    """The race mappings score_e2 takes, from the #130 odds log and the store (development days)."""
    return _load(minutes_before, is_holdout_day)[0]


# --------------------------------------------------------------------------- the command


def _file_digest(path: Path) -> str:
    """16 hex digits of sha256 over the file's bytes; "absent" when there is no file."""
    if not path.is_file():
        return "absent"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _data_builds(info: dict) -> dict[str, str]:
    """The builds a run read, the same keys whether or not any race was logged."""
    digest = hashlib.sha256("\n".join(info["logs"]).encode()).hexdigest()[:16]
    builds = {"odds_log": f"odds_snapshots/jra/{FORECAST}@{digest}"}
    base = paths.processed_dir("jra")
    for name in STORE_TABLES:
        table = f"{name}.parquet"
        builds[f"jra_{name}"] = f"processed/jra/{table}@{_file_digest(base / table)}"
    return builds


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _num(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _render(summary: dict) -> str:
    lines = [f"# #281 E2 ratio-cut rules, {study.E2_ID} ({summary['git_sha'][:12]})", "",
             f"Races scored {summary['n_races']}, meetings {summary['n_meetings']}; "
             f"excluded {json.dumps(summary['excluded'], sort_keys=True)}.", "",
             "## Members", "",
             "| category | cut | scope | n_bets | races | meetings | ROI | lo | hi |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for m in summary["members"]:
        lines.append(f"| {m['category']} | {m['cut']} | {m['scope']} | {m['n_bets']} | "
                     f"{m['n_races']} | {m['n_meetings']} | {_num(m['roi'])} | {_num(m['lo'])} | "
                     f"{_num(m['hi'])} |")  # fmt: skip
    max_test, spa = summary["max_test"], summary["spa"]
    lines += ["", "## Max test (SPA + StepM over the 16 members)", "",
              f"Status: {max_test['status']}."]  # fmt: skip
    if max_test.get("error") or max_test.get("reason"):
        lines.append(f"Reason: {max_test.get('error') or max_test.get('reason')}")
    if spa:
        superior = summary["stepm"]["superior"] if summary["stepm"] else []
        lines.append(f"SPA p-value {_num(spa['pvalue'])} over {spa['n_rules']} rules (family "
                     f"size {spa['family_size']}, {spa['n_clusters']} meetings); StepM superior: "
                     f"{', '.join(superior) or 'none'}.")  # fmt: skip
    lines += ["", "## D, the full-board category ratio on the final boards", "",
              "| category | races | mean | median |", "| --- | --- | --- | --- |"]  # fmt: skip
    for category, d in summary["d"].items():
        lines.append(f"| {category} | {d['n_races']} | {_num(d['mean'])} | "
                     f"{_num(d['median'])} |")  # fmt: skip
    proxy = summary["proxy"]
    lines += ["", "## Proxy agreement (race 1-4 or maiden against the day's final-pool tertile)",
              "", f"Races {proxy['n_races']} over {proxy['n_days']} days: both less popular "
              f"{proxy['both_less']}, proxy only {proxy['proxy_only']}, tertile only "
              f"{proxy['tertile_only']}, neither {proxy['neither']}; agreement "
              f"{_num(proxy['agreement'])}."]  # fmt: skip
    shrink = summary["shrinkage"]
    edges = ", ".join(f"{e:g}" for e in shrink["band_edges"])
    lines += ["", "## Shrinkage (segment_roi, exacta odds bands as fixed effects)", "",
              f"Status: {shrink['status']}; band edges {edges} (final-board odds)."]  # fmt: skip
    if shrink["status"] == "fitted":
        effects = ", ".join(f"{b} {_num(v)}" for b, v in shrink["band_effects"].items())
        lines.append(f"Band effects (ln omega): {effects}.")
    else:
        lines.append(f"Reason: {shrink['reason']}")
    return "\n".join(lines) + "\n"


def exacta_rules_command() -> None:
    """Score #131's E2 ratio-cut rules on the #130-logged development boards and record the run."""
    sha, registry = exacta_run._git_sha(), default_registry(COUNTRY)
    append = exacta_run._appender(registry)
    data_builds: dict[str, str] = {}
    try:
        races, info = _load(MINUTES_BEFORE_POST, is_holdout_day)
        data_builds = _data_builds(info)
        _, _, ordering_id, lambda2 = exacta_run._fits()
        result = score_e2(
            races, registry=registry, git_sha=sha, lambda2=lambda2, data_builds=data_builds,
            components={"ordering": ordering_id},
        )  # fmt: skip
        summary = {
            "git_sha": sha, "family": study.E2_ID, "spec_hash": FROZEN_E2_HASH,
            "lambda2_full_development": lambda2, "ordering_fit": ordering_id,
            "read": {k: v for k, v in info.items() if k != "logs"}, "logs_read": len(info["logs"]),
            "logged_meetings_with_forecast": len({r["meeting_id"] for r in races}),
            "logged_races_loaded": len(races), **result,
            "d": d_descriptive(races, lambda2=lambda2), "proxy": proxy_agreement(races),
        }  # fmt: skip
    except Exception as exc:  # the failed run is a registry record too
        error = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        append(_record(MAX_TEST_ID, "run", error, {}, [], data_builds, sha, status="failed"))
        raise
    summary = _clean(summary)
    out = paths.reports_dir() / "japan" / "281-exacta-rules"
    write_report(out / f"e2-{sha[:12]}.json", json.dumps(summary, indent=1), sources=SOURCES)
    write_report(out / f"e2-{sha[:12]}.md", _render(summary), sources=SOURCES)
    write_summary(registry, summary_dir(COUNTRY))
    keys = ("read", "logged_meetings_with_forecast", "n_races", "n_meetings", "excluded", "spa",
            "stepm", "max_test", "shrinkage", "d", "proxy")  # fmt: skip
    typer.echo(json.dumps({k: summary[k] for k in keys}, indent=1))
