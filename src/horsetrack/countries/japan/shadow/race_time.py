"""Race-time forecasts at the primary cutoff, their timing and their coverage (ticket #414).

A research recorder for #305's race-time model (`models.speed`), independent of the V0 shadow
loader. Nothing here is adopted (D16) and nothing reads an outcome of the race forecast.

- `record_race(race, *, model, results, model_version, card_version, now, replay=False)`: one
  JSON-ready record for one card race. The input cutoff is the post known at the cutoff less the
  primary decision minutes (config/decision.yaml). A deep copy of the fitted model observes only
  the day's results first seen (#225's first observed publication) at or before the cutoff, so the
  caller's model never changes and a result seen later, or never, cannot move the forecast
  whatever the clock reads. par_s is the cell's par, same_day_s the par plus the same-day
  variant's mean at the cutoff. `now()` is read once the computation is done: completion_utc is
  that reading. mode is "live" when completion_utc is at or before the cutoff, "late" when it is
  after the cutoff but before the post, and "replay" at or after the post or whenever replay=True.
  A race without a post or in a cell the fit never saw is "skipped", with its own reason.
- `coverage(records, card)`: per venue and "all", every race of the latest card in the
  denominator. A race counts once; readiness is judged against both the post recorded at the
  cutoff and the latest card's post, so a post moved earlier after the record cannot leave it on
  time.
- `verify_replay(records, *, card, results, model, model_version, card_version)`: recomputes each
  ok record from the card row, the source results and the versions; [] when everything matches,
  else one problem per mismatch, each naming its race id.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import Any

import pandas as pd

from horsetrack.countries.japan.board.snapshot import load_decision_grid
from horsetrack.countries.japan.models.speed import JST, SpeedModel

OK, SKIPPED = "ok", "skipped"
LIVE, LATE, REPLAY = "live", "late", "replay"
NO_POST = "no scheduled post time on the card at the cutoff"
UNSEEN_CELL = "the race's cell has no par in the fitted model"
REFUSED = "the model refused the day's results"
COUNTS = (
    "scheduled",
    "forecast",
    "skipped",
    "missing",
    "ready_by_cutoff",
    "ready_by_post",
    "replay",
)
TOLERANCE_S = 1e-9

__all__ = ["coverage", "record_race", "verify_replay"]


def _stamp(value: Any) -> pd.Timestamp | None:
    """An aware UTC Timestamp, or None for a missing value; a naive time is refused."""
    if value is None or (not isinstance(value, datetime) and pd.isna(value)):
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        raise ValueError(f"timestamps must be tz-aware (UTC): {value!r}")
    return stamp.tz_convert("UTC")


def _iso(stamp: pd.Timestamp | None) -> str | None:
    return None if stamp is None else stamp.isoformat()


def _primary() -> pd.Timedelta:
    return pd.Timedelta(minutes=load_decision_grid().primary)


def _eligible(results: pd.DataFrame, race: Mapping, model: SpeedModel, cutoff: pd.Timestamp):
    """The results the same-day variant may use: another race of the target's JST day and variant
    group, timed, first seen at or before the cutoff; renamed for `SpeedModel.observe`."""
    seen = pd.to_datetime(results["first_seen_utc"], utc=True)
    posts = pd.to_datetime(results["post_utc"], utc=True)
    keep = seen.notna() & (seen <= cutoff) & posts.notna() & results["top3_time_s"].notna()
    keep &= results["race_id"] != race["race_id"]
    day = _stamp(race["post_utc"]).tz_convert(JST).date()
    keep &= posts.dt.tz_convert(JST).dt.date == day
    for key in model.spec.variant_keys:
        keep &= results[key] == race[key]
    frame = results[keep].copy()
    frame["race_utc"], frame["result_available_utc"] = posts[keep], seen[keep]
    return frame.sort_values(["result_available_utc", "race_id"], kind="stable")


def _mode(completion: pd.Timestamp, cutoff: pd.Timestamp | None, post, replay: bool) -> str | None:
    if replay:
        return REPLAY
    if cutoff is None:
        return None  # no post: the timing cannot be judged
    if completion >= post:
        return REPLAY
    return LIVE if completion <= cutoff else LATE


def record_race(
    race: Mapping,
    *,
    model: SpeedModel,
    results: pd.DataFrame,
    model_version: str,
    card_version: str,
    now: Callable[[], datetime],
    replay: bool = False,
) -> dict[str, Any]:
    """One record for one card race (see the module docstring)."""
    post = _stamp(race.get("post_utc"))
    cutoff = None if post is None else post - _primary()
    par_s = same_day_s = reason = None
    used: list[dict[str, str]] = []
    if post is None:
        reason = NO_POST
    else:
        local = copy.deepcopy(model)  # the caller's model observes nothing
        cell = {key: race[key] for key in model.spec.par_keys}
        try:
            par_s = float(local.par(**cell))
        except KeyError:
            reason = UNSEEN_CELL
        if reason is None:
            frame = _eligible(results, race, model, cutoff)
            try:
                local.observe(frame)
            except ValueError as exc:
                reason, par_s = f"{REFUSED}: {exc}", None
            else:
                target = {**race, "race_utc": post}
                at = cutoff + pd.Timedelta(1, "ns")  # the variant counts times strictly before
                same_day_s = float(local.expected_top3_time(target, at_utc=at))
                used = [
                    {"race_id": rid, "first_seen_utc": _iso(_stamp(seen))}
                    for rid, seen in zip(
                        frame["race_id"], frame["result_available_utc"], strict=True
                    )
                ]
    completion = _stamp(now())  # read after the work
    if completion is None:
        raise ValueError("now() gave no time")
    return {
        "race_id": race["race_id"],
        "venue": race["venue"],
        "status": SKIPPED if reason else OK,
        "reason": reason,
        "model_version": model_version,
        "card_version": card_version,
        "input_cutoff_utc": _iso(cutoff),
        "post_utc_at_cutoff": _iso(post),
        "completion_utc": _iso(completion),
        "par_s": par_s,
        "same_day_s": same_day_s,
        "results_used": used,
        "mode": _mode(completion, cutoff, post, replay),
    }


def _ready(rec: Mapping, latest_post: pd.Timestamp | None, primary: pd.Timedelta):
    """(ready by the cutoff, ready by the post) of one ok record against the latest post."""
    done = _stamp(rec.get("completion_utc"))
    cutoff, post = _stamp(rec.get("input_cutoff_utc")), _stamp(rec.get("post_utc_at_cutoff"))
    if None in (done, cutoff, post, latest_post):
        return False, False
    mode = rec.get("mode")
    by_cutoff = mode == LIVE and done <= cutoff and done <= latest_post - primary
    by_post = mode in (LIVE, LATE) and done < post and done < latest_post
    return by_cutoff, by_post


def coverage(records: Iterable[Mapping], card: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Readiness per venue and "all" over every race of the latest card (module docstring)."""
    primary = _primary()
    by_race: dict[Any, list[Mapping]] = {}
    for rec in records:
        by_race.setdefault(rec["race_id"], []).append(rec)
    out: dict[str, dict[str, int]] = {"all": dict.fromkeys(COUNTS, 0)}
    for race in card.drop_duplicates("race_id").to_dict("records"):
        counts = out.setdefault(str(race["venue"]), dict.fromkeys(COUNTS, 0))
        recs = by_race.get(race["race_id"], [])
        ok = [r for r in recs if r.get("status") == OK]
        flags = {"scheduled": True, "forecast": bool(ok)}
        flags["skipped"] = bool(recs) and not ok
        flags["missing"] = not recs
        latest = _stamp(race.get("post_utc"))
        ready = [_ready(r, latest, primary) for r in ok]
        flags["ready_by_cutoff"] = any(c for c, _ in ready)
        flags["ready_by_post"] = any(p for _, p in ready)
        flags["replay"] = bool(ok) and not flags["ready_by_post"]
        for name, on in flags.items():
            counts[name] += int(on)
            out["all"][name] += int(on)
    return out


def _same_time(a: Any, b: Any) -> bool:
    return _stamp(a) == _stamp(b)


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOLERANCE_S)


def verify_replay(
    records: Iterable[Mapping],
    *,
    card: pd.DataFrame,
    results: pd.DataFrame,
    model: SpeedModel,
    model_version: str,
    card_version: str,
) -> list[str]:
    """The problems found recomputing each ok record from its recorded inputs; [] when none."""
    rows = {r["race_id"]: r for r in card.to_dict("records")}
    problems: list[str] = []
    for rec in records:
        if rec.get("status") != OK:
            continue
        race_id = rec["race_id"]
        for name, want in (("model_version", model_version), ("card_version", card_version)):
            if rec.get(name) != want:
                problems.append(f"{race_id}: {name} {rec.get(name)!r} is not {want!r}")
        if race_id not in rows:
            problems.append(f"{race_id}: not on the card")
            continue
        again = record_race(
            rows[race_id],
            model=model,
            results=results,
            model_version=model_version,
            card_version=card_version,
            now=lambda: pd.Timestamp.now("UTC"),
            replay=True,
        )
        if again["status"] != OK:
            problems.append(f"{race_id}: recomputes as skipped ({again['reason']})")
            continue
        for name in ("post_utc_at_cutoff", "input_cutoff_utc"):
            if not _same_time(rec.get(name), again[name]):
                problems.append(f"{race_id}: {name} {rec.get(name)} is not {again[name]}")
        recorded = {(u["race_id"], _stamp(u["first_seen_utc"])) for u in rec["results_used"]}
        source = {(u["race_id"], _stamp(u["first_seen_utc"])) for u in again["results_used"]}
        for rid, seen in sorted(recorded - source, key=repr):
            problems.append(
                f"{race_id}: results_used has {rid} first seen {_iso(seen)}, not in "
                "the source within the cutoff"
            )
        for rid, seen in sorted(source - recorded, key=repr):
            problems.append(f"{race_id}: results_used lacks {rid} first seen {_iso(seen)}")
        for name in ("par_s", "same_day_s"):
            if not _close(rec.get(name), again[name]):
                problems.append(f"{race_id}: {name} {rec.get(name)} recomputes as {again[name]}")
    return problems
