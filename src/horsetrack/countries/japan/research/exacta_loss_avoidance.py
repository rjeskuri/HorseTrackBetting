"""Family L on the exacta (#282): skip a fav_first exacta when q_c / p_c at T-2 is above a cut.

`python -m horsetrack.countries.japan.research.exacta_loss_avoidance {freeze|score|summary}`

The D18 loss-avoidance rule of #131's spec (docs/specs/exacta-overload-prereg.md, family L),
approved by the owner 2026-11-26 and reframed 2026-11-27 on #282. At T-2, every combination c of
the T-2 exacta board whose first horse is the T-2 Win favourite is a fav_first combination; it is
skipped when ``r_c = q_c / p_c`` is above a cut k. Both sets are reported, nothing is dropped:

- the comparator buys every fav_first combination at 100 yen on the same races;
- the kept set (r_c <= k, or q not identified) and the skipped set (r_c > k) are each scored as
  the return per yen at the final dividend minus the comparator's, with a meeting-cluster
  bootstrap and a max-statistic over the 8 members (4 cuts x scopes all and headline) of the set.

q_c is the exacta board's implied share (``exacta_consistency.implied_share``), p_c the
discounted Harville exacta of the T-2 Win board with #115's full-development lambda2. #131's
code is reused, not copied. There is no meeting gate: every run scores the logged development
meetings it finds, with n and the interval; none gives n = 0 records (data pending).

D14: a holdout day's race is dropped (and its log folder never opened) before anything else of it
is read; ``LockedHoldout`` is never opened. Public JRA data only (sources: jra).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    RunRecord,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan import odds_log, takeout
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.research import exacta_run
from horsetrack.countries.japan.research import exacta_study as study
from horsetrack.countries.japan.research.exacta_consistency import (
    favourite,
    harville_exacta,
    implied_share,
)

COUNTRY = "japan"
SOURCES = ("jra",)
TICKET = 282
FAMILY_ID = "japan/282-exacta-fav-first-avoidance-v1"
CUTS = (1.1, 1.2, 1.3, 1.5)
SCOPES = ("all", "headline")
SETS = ("kept", "skipped")
STAKE = 100
LEVEL = 0.95
N_BOOT = 10000
SEED = 282
MINUTES_BEFORE_POST = 2
FORECAST = "FORECAST"
FOLD_SCHEME = "development-days:#130-logged boards at T-2"
Combo = tuple[int, int]


def _yen(odds: float) -> float:
    """Board odds as yen per 100, float noise snapped: 1.1 is 110 yen, so the floor binds."""
    dividend = 100 * float(odds)
    whole = round(dividend)
    return float(whole) if abs(dividend - whole) < 1e-6 else dividend


def _combo(value) -> Combo:
    """A combination as (first, second) from a tuple or a 'first,second' text."""
    if isinstance(value, str):
        value = value.split(",")
    first, second = value
    return int(first), int(second)


# --------------------------------------------------------------------------- the rule


def fav_first_ratios(
    win_board: Mapping[int, float],
    exacta_board: Mapping[Combo, float],
    *,
    rate: float,
    lambda2: float = 1.0,
) -> dict[Combo, float]:
    """``r_c = q_c / p_c`` of each exacta combination led by the T-2 favourite (NaN if q is not
    identified); p_c from the Win board's 1 / odds renormalised over its starters."""
    fav = favourite(win_board)
    horses = sorted(win_board)
    index = {h: k for k, h in enumerate(horses)}
    inverse = np.array([1.0 / win_board[h] for h in horses])
    model = harville_exacta(inverse / inverse.sum(), lambda2=lambda2)
    out: dict[Combo, float] = {}
    for combo, odds in exacta_board.items():
        first, second = _combo(combo)
        if first != fav or second == first or second not in index:
            continue
        q = implied_share(_yen(odds), rate=rate)
        out[(first, second)] = q / float(model[index[first], index[second]])
    return out


def split_by_cut(ratios: Mapping[Combo, float], cut: float) -> tuple[tuple, tuple]:
    """(kept, skipped), each sorted: skipped when r_c > cut strictly; a NaN ratio is kept (the
    rule skips only on evidence that the pool overloads c)."""
    skipped = tuple(sorted(c for c, r in ratios.items() if r > cut))
    kept = tuple(sorted(c for c, r in ratios.items() if not r > cut))
    return kept, skipped


@dataclass(frozen=True)
class Selection:
    favourite: int
    ratios: dict[Combo, float]
    kept: tuple
    skipped: tuple


def select(win_board, exacta_board, *, rate: float, cut: float, lambda2: float = 1.0) -> Selection:
    ratios = fav_first_ratios(win_board, exacta_board, rate=rate, lambda2=lambda2)
    kept, skipped = split_by_cut(ratios, cut)
    return Selection(favourite(win_board), ratios, kept, skipped)


def less_popular_races(rows: Iterable[tuple[str, date, float]]) -> set[str]:
    """Race ids in the lower tertile of the exacta pool total within their day: the ceil(n / 3)
    smallest of the day's n races (spec section 6, #130 days)."""
    by_day: dict[date, list[tuple[float, str]]] = {}
    for race_id, day, total in rows:
        by_day.setdefault(day, []).append((float(total), race_id))
    out: set[str] = set()
    for races in by_day.values():
        out |= {race_id for _, race_id in sorted(races)[: math.ceil(len(races) / 3)]}
    return out


# --------------------------------------------------------------------------- per race


def score_races(
    races: Iterable[Mapping],
    *,
    holdout_day: Callable[[str, date], bool],
    lambda2: float = 1.0,
) -> pd.DataFrame:
    """One row per scored race and cut: the comparator's (``n_all``, ``ret_all``) and the kept and
    skipped sets' tickets (100 yen each) and returns (yen: the final dividend when the winning
    exacta is in the set). A holdout day's race is dropped before any other field is read;
    excluded races (#131's ``exclusion_reason``, a board without a fav_first combination) are
    dropped. The drops by reason are in ``frame.attrs["excluded"]``."""
    rows: list[dict] = []
    excluded: dict[str, int] = {}

    def drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for race in races:
        if holdout_day(COUNTRY, race["race_date"]):
            drop("holdout_day")
            continue
        win, board = race["win_board_t2"], race["exacta_board_t2"]
        exacta = list(race["exacta"])
        reason = study.exclusion_reason(
            n_starters=len(win), boost_may_be_withheld=bool(race["boost_may_be_withheld"]),
            exacta=exacta,
        )  # fmt: skip
        ratios = {} if reason else fav_first_ratios(win, board, rate=race["rate"], lambda2=lambda2)
        if reason is None and not ratios:
            reason = "no_fav_first_combination"
        if reason is not None:
            drop(reason)
            continue
        (winner, dividend, _) = exacta[0]
        winner = _combo(winner)
        headline = win[favourite(win)] >= 3.0 and bool(race["less_popular"])
        for cut in CUTS:
            kept, skipped = split_by_cut(ratios, cut)
            paid = float(dividend) if winner in ratios else 0.0
            rows.append({
                "race_id": race["race_id"], "meeting_id": race["meeting_id"],
                "race_date": race["race_date"], "cut": cut, "headline": headline,
                "n_all": len(ratios), "n_kept": len(kept), "n_skipped": len(skipped),
                "ret_all": paid, "ret_kept": paid if winner in kept else 0.0,
                "ret_skipped": paid if winner in skipped else 0.0,
            })  # fmt: skip
    columns = ["race_id", "meeting_id", "race_date", "cut", "headline", "n_all", "n_kept",
               "n_skipped", "ret_all", "ret_kept", "ret_skipped"]  # fmt: skip
    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs["excluded"] = excluded
    return frame


# --------------------------------------------------------------------------- the members

_COLUMNS = ["cut", "scope", "set", "n_races", "n_meetings", "stake", "ret", "per_yen",
            "all_per_yen", "diff", "se", "lo", "hi", "adj_lo", "adj_hi"]  # fmt: skip
_CHUNK_DRAWS = 2_000_000


def _ratio(ret, stake):
    """ret / stake, NaN where the stake is 0 (works on scalars and arrays)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(stake > 0, ret / np.where(stake > 0, stake, 1.0), np.nan)


def member_table(
    rows: pd.DataFrame, *, n_boot: int, seed: int = SEED, level: float = LEVEL
) -> pd.DataFrame:
    """Per member (cut, scope) and set (kept, skipped): the return per yen, the comparator's, their
    difference, and its meeting-cluster bootstrap (SE, one-sided bounds at ``level`` as in
    core.eval.bootstrap). ``adj_lo`` and ``adj_hi`` widen those bounds by the max over the set's
    8 members of the studentised deviation, resampled on the same meeting draws."""
    codes, meetings = pd.factorize(rows["meeting_id"], sort=True)
    n_meet = len(meetings)
    cells = []  # one entry per (cut, scope, set)
    for cut in CUTS:
        in_cut = (rows["cut"] == cut).to_numpy()
        for scope in SCOPES:
            mask = in_cut & (rows["headline"].to_numpy(dtype=bool) if scope == "headline" else True)
            sub = rows[mask]
            sums = {
                name: np.bincount(codes[mask], weights=sub[name].to_numpy(dtype=float),
                                  minlength=n_meet)
                for name in ("n_all", "ret_all", "n_kept", "ret_kept", "n_skipped", "ret_skipped")
            }  # fmt: skip
            for name in SETS:
                cells.append({
                    "cut": cut, "scope": scope, "set": name, "n_races": len(sub),
                    "n_meetings": sub["meeting_id"].nunique(),
                    # per meeting: set stake, set return, comparator stake, comparator return
                    "parts": np.column_stack([
                        STAKE * sums[f"n_{name}"], sums[f"ret_{name}"],
                        STAKE * sums["n_all"], sums["ret_all"],
                    ]),
                })  # fmt: skip
    parts = np.concatenate([c["parts"] for c in cells], axis=1)  # (meetings, 4 per cell)
    totals = parts.sum(axis=0).reshape(len(cells), 4)
    point = _ratio(totals[:, 1], totals[:, 0]) - _ratio(totals[:, 3], totals[:, 2])

    reps = np.full((n_boot, len(cells)), np.nan)
    if n_meet:
        rng = np.random.default_rng(seed)
        chunk = max(1, _CHUNK_DRAWS // n_meet)
        for start in range(0, n_boot, chunk):
            size = min(chunk, n_boot - start)
            draws = rng.multinomial(n_meet, np.full(n_meet, 1.0 / n_meet), size=size)
            drawn = (draws @ parts).reshape(size, len(cells), 4)
            reps[start : start + size] = _ratio(drawn[..., 1], drawn[..., 0]) - _ratio(
                drawn[..., 3], drawn[..., 2]
            )  # fmt: skip

    out = []
    for k, cell in enumerate(cells):
        stake, ret, all_stake, all_ret = totals[k]
        col = reps[:, k]
        finite = col[np.isfinite(col)]
        have = len(finite) > 1 and np.isfinite(point[k])
        out.append({
            "cut": cell["cut"], "scope": cell["scope"], "set": cell["set"],
            "n_races": cell["n_races"], "n_meetings": cell["n_meetings"],
            "stake": stake, "ret": ret, "per_yen": float(_ratio(ret, stake)),
            "all_per_yen": float(_ratio(all_ret, all_stake)), "diff": float(point[k]),
            "se": float(finite.std(ddof=1)) if have else np.nan,
            "lo": float(np.quantile(finite, 1 - level)) if have else np.nan,
            "hi": float(np.quantile(finite, level)) if have else np.nan,
            "adj_lo": np.nan, "adj_hi": np.nan,
        })  # fmt: skip
    table = pd.DataFrame(out, columns=_COLUMNS)
    for name in SETS:
        _widen(table, reps, np.array([c["set"] == name for c in cells]), level)
    return table


def _widen(table: pd.DataFrame, reps: np.ndarray, in_set: np.ndarray, level: float) -> None:
    """Fill adj_lo and adj_hi of the set's members: the ``level`` quantile, over replicates, of the
    largest studentised deviation among the members; never narrower than lo and hi."""
    idx = np.flatnonzero(in_set & table["se"].notna().to_numpy())
    if not len(idx):
        return
    diff, se = table["diff"].to_numpy()[idx], table["se"].to_numpy()[idx]
    scale = np.where(se > 0, se, np.inf)  # a degenerate member deviates by 0
    for sign, bound, adj in ((1.0, "lo", "adj_lo"), (-1.0, "hi", "adj_hi")):
        dev = sign * (diff - reps[:, idx]) / scale  # replicates x members
        worst = np.where(np.isnan(dev), -np.inf, dev).max(axis=1)
        worst = worst[np.isfinite(worst)]
        c = float(np.quantile(worst, level)) if len(worst) else 0.0
        wide = diff - sign * c * se
        base = table[bound].to_numpy()[idx]
        table.loc[table.index[idx], adj] = np.minimum(base, wide) if sign > 0 else np.maximum(
            base, wide
        )  # fmt: skip


# --------------------------------------------------------------------------- the family


def family_specs() -> dict[str, dict]:
    """The pre-registered family as plain JSON, a fresh copy on every call."""
    members = [{"cut": k, "scope": s} for k in CUTS for s in SCOPES]
    return {
        FAMILY_ID: {
            "ticket": TICKET,
            "hypothesis_id": "E43",
            "spec": "docs/specs/exacta-overload-prereg.md, family L",
            "owner_signoff": "2026-11-26, #282 (family L as drafted); reframed 2026-11-27, #282",
            "category": "fav_first",
            "selection_time": "T-2",
            "rule": "skip fav_first exacta c at T-2 when q_c / p_c > k (strictly above; a ratio "
            "equal to k, or a q_c not identified at or below 110 yen, is kept); q_c the exacta "
            "board's implied share at T-2 (implied_share(odds * 100, rate)), p_c the discounted "
            "Harville exacta of the Win board at T-2 (1 / odds renormalised over its starters)",
            "favourite": "lowest Win odds on the Win board at T-2, ties to the lowest number; "
            "the fav_first combinations are those of the T-2 exacta board led by it",
            "members": members,
            "cuts": list(CUTS),
            "scopes": list(SCOPES),
            "scope_definitions": {
                "all": "every scored race",
                "headline": "favourite's Win odds at T-2 >= 3.0 and less popular: the lower "
                "tertile, ceil(n / 3) of the day's n logged races, of the exacta pool total at "
                "T-2 within the logged day (spec section 6)",
            },
            "family_size": len(members),
            "stake_yen": STAKE,
            "exacta_takeout": 0.25,
            "lambda2": study.FITTED,
            "comparator": "buy every fav_first combination of the T-2 exacta board at 100 yen on "
            "the same races",
            "primary_metric": "return per yen of the kept set minus that of all fav_first "
            "combinations, at the final dividends (takeout and breakage in), with its CI",
            "report_sets": list(SETS),
            "judged_at": "final dividends, no own impact: the ledger assumption that 100 yen per "
            "combination is negligible against the exacta pool (spec section 7), the same for "
            "the kept set, the skipped set and the comparator",
            "exclusions": [
                "boost_may_be_withheld",
                "fewer than 5 starters",
                "more than one paying exacta (dead heat for 1st or 2nd)",
                "a refunded exacta",
                "no exacta row",
                "no fav_first combination on the T-2 exacta board",
            ],  # fmt: skip
            "data": "#130-logged development days only (is_holdout_day False, checked before "
            "any board or dividend is read); no gate on the number of meetings: every run "
            "scores the meetings logged, with n and the interval",
            "sources": list(SOURCES),
            "pool": FORECAST,
            "cluster": "meeting",
            "interval": "meeting-cluster bootstrap of the difference, 95% one-sided bounds in "
            "each direction; adj_lo and adj_hi widen them by a max-statistic over the set's 8 "
            "members",
            "n_boot": N_BOOT,
            "level": LEVEL,
            "seed": SEED,
        }
    }


def freeze(registry) -> dict[str, str]:
    """Freeze the family's spec; a freeze is not a run, so the committed summary is untouched."""
    return {fid: registry.freeze_family(fid, spec) for fid, spec in family_specs().items()}


def _num(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def score(
    registry,
    races: Iterable[Mapping],
    meta: Mapping,
    sha: str,
    *,
    holdout_day: Callable[[str, date], bool] = is_holdout_day,
    n_boot: int = N_BOOT,
) -> dict:
    """Score the 8 members on ``races`` (any number of meetings) and append one registry record
    per member holding its kept and skipped metrics. ``RunRefused`` before anything is computed
    or written unless the family is frozen with the spec in use and its scored records hold
    ``meta``'s ordering fit. A rerun on the same inputs appends nothing."""
    spec = family_specs()[FAMILY_ID]
    frozen, in_use = registry.frozen_spec_hash(FAMILY_ID), exacta_run._spec_hash(spec)
    if frozen is None:
        raise exacta_run.RunRefused(f"{FAMILY_ID} is not frozen; run `freeze` first")
    if frozen != in_use:
        raise exacta_run.RunRefused(
            f"{FAMILY_ID} is frozen with spec {frozen[:12]}, not the spec in use "
            f"({in_use[:12]}): a changed design is a new family"
        )
    for r in registry.records(FAMILY_ID):
        held = r.components.get("ordering")
        if r.status == "scored" and held not in (None, meta["ordering_fit"]):
            raise exacta_run.RunRefused(
                f"{FAMILY_ID} holds scored records under ordering fit {held}, not "
                f"{meta['ordering_fit']}: a refreshed fit needs its own frozen family"
            )
    frame = score_races(races, holdout_day=holdout_day, lambda2=meta["lambda2_full_development"])
    table = member_table(frame, n_boot=n_boot)
    append = exacta_run._appender(registry)
    by_member = {(r.cut, r.scope, r.set): r for r in table.itertuples(index=False)}
    for cut in CUTS:
        for scope in SCOPES:
            kept, skipped = by_member[(cut, scope, "kept")], by_member[(cut, scope, "skipped")]
            n = int(kept.n_races)
            metrics = [FoldMetric("development", "n_races", float(n), None, None, n)]
            for name, row in (("kept", kept), ("skipped", skipped)):
                if row.stake > 0:
                    metrics += [
                        FoldMetric("development", f"{name}_per_yen", row.per_yen, None, None, n),
                        FoldMetric("development", f"{name}_minus_all", row.diff, _num(row.lo),
                                   _num(row.hi), n),
                        FoldMetric("development", f"{name}_minus_all_max_stat", row.diff,
                                   _num(row.adj_lo), _num(row.adj_hi), n),
                    ]  # fmt: skip
            if math.isfinite(kept.all_per_yen):
                metrics.append(
                    FoldMetric("development", "all_per_yen", kept.all_per_yen, None, None, n)
                )
            append(RunRecord(
                hypothesis_id="E43", family_id=FAMILY_ID, variant=f"cut={cut}|{scope}",
                ticket=TICKET, git_sha=sha, data_builds=dict(meta["data_builds"]),
                sources=SOURCES, fold_scheme=FOLD_SCHEME,
                components={"ordering": meta["ordering_fit"]},
                params={"cut": cut, "scope": scope}, metrics=tuple(metrics), status="scored",
            ))  # fmt: skip
    return {
        "members": table,
        "meta": {
            **meta,
            "scored_races": int(frame["race_id"].nunique()),
            "scored_meetings": int(frame["meeting_id"].nunique()),
            "excluded": dict(frame.attrs["excluded"]),
        },
    }


# --------------------------------------------------------------------------- the logged boards


def _race_no(race_id) -> int | None:
    try:
        return int(str(race_id).rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def _boards_at(win_rows: list[dict], held: dict, race_id: str, at_utc, post) -> tuple[str, dict]:
    """The race's boards at ``at_utc``: ``("", boards)`` or (why not, {}). ``held`` maps each
    FORECAST combination (and the pool total) to its latest row available by ``at_utc``."""
    board = snapshot.board_at(win_rows, race_id, at_utc, post_utc=post)
    if board is None or not board.win:
        return "no_win_board_at_t", {}
    priced: dict[Combo, float] = {}
    for key, row in held.items():
        if key == odds_log.POOL_TOTAL:
            continue
        try:
            combo = _combo(key)
        except ValueError:
            continue
        if snapshot._priced(row) and combo[0] != combo[1] and not set(combo) & board.scratched:
            priced[combo] = float(row["odds"])
    if not priced:
        return "no_exacta_board_at_t", {}
    if odds_log.POOL_TOTAL not in held:
        return "no_pool_total_at_t", {}
    return "", {"win_board_t2": dict(board.win), "exacta_board_t2": priced,
                "pool_total": float(held[odds_log.POOL_TOTAL]["investment"])}  # fmt: skip


def _logged_boards(minutes_before: int, holdout_day) -> tuple[dict[str, dict], dict]:
    """Per race with FORECAST rows on a development day: its boards at T-2 (a Win board and the
    exacta board with its pool total). A holdout day's folder is never opened. A log is streamed
    one capture at a time and only the latest FORECAST row per key available by T-2 is held."""
    root = paths.data_dir() / "odds_snapshots" / "jra"
    info = {"days_read": 0, "days_holdout_skipped": 0, "logs": [], "dropped": {}}
    found: dict[str, dict] = {}
    folders = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for folder in folders:
        try:
            day = date.fromisoformat(folder.name)
        except ValueError:
            continue
        if holdout_day(COUNTRY, day):
            info["days_holdout_skipped"] += 1
            continue
        info["days_read"] += 1
        for log in sorted(folder.glob("*.jsonl.gz")):
            info["logs"].append(f"{day.isoformat()}/{log.name}:{log.stat().st_size}")
            times = snapshot.post_times_for(day, log.name.split(".")[0])
            cutoff = {n: t - timedelta(minutes=minutes_before) for n, t in times.items()}
            win_rows: dict[str, list[dict]] = {}
            held: dict[str, dict] = {}  # race_id -> {combination or total: (available, row)}
            for member in odds_log.iter_members(log):
                for row in member:
                    race_id, pool = row.get("race_id"), row.get("pool")
                    if pool == "WIN" or (pool == "PLACE" and row.get("status_raw") == "SCR"):
                        win_rows.setdefault(race_id, []).append(row)
                    elif pool == FORECAST:
                        slot = held.setdefault(race_id, {})
                        at, available = cutoff.get(_race_no(race_id)), snapshot._available(row)
                        if at is None or available is None or available > at:
                            continue
                        key = row["combination"]
                        if key == odds_log.POOL_TOTAL and row.get("investment") is None:
                            continue
                        if key not in slot or available >= slot[key][0]:
                            slot[key] = (available, row)
            for race_id, slot in held.items():
                post = times.get(_race_no(race_id))
                if post is None:
                    reason, got = "no_post_time", {}
                else:
                    reason, got = _boards_at(
                        win_rows.get(race_id, []), {k: r for k, (_, r) in slot.items()},
                        race_id, post - timedelta(minutes=minutes_before), post,
                    )  # fmt: skip
                if reason:
                    info["dropped"][reason] = info["dropped"].get(reason, 0) + 1
                else:
                    found[race_id] = {"race_date": day, **got}
    return found, info


def load_logged_races(
    *, minutes_before: int = MINUTES_BEFORE_POST, holdout_day=is_holdout_day
) -> tuple[list[dict], dict, dict]:
    """The `races` input of ``score`` from the #130 odds logs and the store, and the run's meta:
    (races, meta, info). Development days only; ``meta`` carries the ordering fit, its lambda2
    and the data builds."""
    found, info = _logged_boards(minutes_before, holdout_day)
    _, _, ordering_id, lambda2 = exacta_run._fits()
    digest = hashlib.sha256("\n".join(info["logs"]).encode()).hexdigest()[:16]
    builds = {"odds_log": f"odds_snapshots/jra/{FORECAST}@{digest}"}
    races: list[dict] = []
    if found:
        base, ids = paths.processed_dir("jra"), sorted(found)
        meta_rows = pq.read_table(
            base / "races.parquet",
            columns=["race_id", "meeting_id", "race_no", "race_class_raw"],
            filters=pc.field("race_id").isin(ids),
        ).to_pandas()
        dividends = pq.read_table(
            base / "dividends.parquet",
            columns=["race_id", "combination", "dividend", "is_refund"],
            filters=pc.field("race_id").isin(ids) & (pc.field("pool") == FORECAST),
        ).to_pandas()
        paying: dict[str, list] = {}
        for r in dividends.itertuples(index=False):
            paying.setdefault(r.race_id, []).append(
                (_combo(r.combination), float(r.dividend), bool(r.is_refund))
            )
        for m in meta_rows.sort_values("race_id").itertuples(index=False):
            t = takeout.race_takeout(m.race_id, FORECAST, race_class_raw=m.race_class_raw)
            races.append({"race_id": m.race_id, "meeting_id": m.meeting_id,
                          "race_no": int(m.race_no), "rate": t.payout_rate,
                          "boost_may_be_withheld": t.boost_may_be_withheld,
                          **found[m.race_id], "exacta": paying.get(m.race_id, [])})  # fmt: skip
        popular = less_popular_races((r["race_id"], r["race_date"], r["pool_total"]) for r in races)
        for r in races:
            r["less_popular"] = r["race_id"] in popular
        builds["jra_races"] = f"processed/jra/races.parquet@{exacta_run._fingerprint(meta_rows)}"
        builds["jra_dividends"] = (
            f"processed/jra/dividends.parquet@{exacta_run._fingerprint(dividends)}"
        )
    info["dropped"]["not_in_store"] = len(found) - len(races)
    meta = {"ordering_fit": ordering_id, "lambda2_full_development": lambda2,
            "data_builds": builds}  # fmt: skip
    return races, meta, info


# --------------------------------------------------------------------------- the command


def _render(table: pd.DataFrame, meta: Mapping, sha: str) -> str:
    def f(v) -> str:
        return "-" if pd.isna(v) else f"{v:.3f}"

    lines = [f"# #282 family L: fav_first loss avoidance ({sha[:12]})", "",
             f"Races scored {meta['scored_races']}, meetings {meta['scored_meetings']}, "
             f"excluded {meta['excluded']}.", "",
             "| cut | scope | set | races | meetings | stake | return per yen | all fav_first | "
             "difference | lo | hi | max-stat lo | max-stat hi |",
             "|" + " --- |" * 13]  # fmt: skip
    for r in table.itertuples(index=False):
        lines.append(
            f"| {r.cut} | {r.scope} | {r.set} | {r.n_races} | {r.n_meetings} | {r.stake:.0f} | "
            f"{f(r.per_yen)} | {f(r.all_per_yen)} | {f(r.diff)} | {f(r.lo)} | {f(r.hi)} | "
            f"{f(r.adj_lo)} | {f(r.adj_hi)} |"
        )  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.research.exacta_loss_avoidance"
    )
    parser.add_argument("action", choices=["freeze", "score", "summary"])
    args = parser.parse_args(argv)
    registry = default_registry(COUNTRY)
    if args.action == "freeze":
        result = freeze(registry)
    elif args.action == "summary":  # the committed summary, regenerated from the registry
        write_summary(registry, summary_dir(COUNTRY))
        result = {"summary": str(summary_dir(COUNTRY))}
    else:
        sha = exacta_run._git_sha()
        races, meta, info = load_logged_races()
        try:
            out = score(registry, races, meta, sha)
        except exacta_run.RunRefused:
            raise
        except Exception as exc:  # a failed run is a registry record too
            exacta_run._appender(registry)(RunRecord(
                hypothesis_id="E43", family_id=FAMILY_ID, variant="run", ticket=TICKET,
                git_sha=sha, data_builds=meta["data_builds"], sources=SOURCES,
                fold_scheme=FOLD_SCHEME, components={}, status="failed", metrics=(),
                params={"error": f"{type(exc).__name__}: {exc}"[:500]},
            ))  # fmt: skip
            raise
        table, run_meta = out["members"], out["meta"]
        folder = paths.reports_dir() / COUNTRY / "282-exacta-loss-avoidance"
        write_report(folder / f"l-{sha[:12]}.md", _render(table, run_meta, sha), sources=SOURCES)
        write_report(
            folder / f"l-{sha[:12]}.json",
            json.dumps({"meta": run_meta, "read": {k: v for k, v in info.items() if k != "logs"},
                        "members": table.to_dict(orient="records")}, indent=1, default=str),
            sources=SOURCES,
        )  # fmt: skip
        result = {"meta": run_meta, "read": {k: v for k, v in info.items() if k != "logs"},
                  "members": table.to_dict(orient="records")}  # fmt: skip
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
