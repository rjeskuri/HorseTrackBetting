"""Race-time forecasts at the primary cutoff, their timing and their coverage (ticket #414).

A research recorder for #305's race-time model (`models.speed`), independent of the V0 shadow
loader. Nothing here is adopted (D16) and nothing reads an outcome of the race forecast.

- `record_race(race, *, model, results, model_version, card_version, now, replay=False,
  latest_post=None, inputs_utc=None)`: one JSON-ready record for one card race. The input cutoff
  is the post known at the cutoff less the primary decision minutes (config/decision.yaml). The
  model must hold no observations (ValueError otherwise: a residual it cached could move
  same_day_s unlisted). `now()` is read before the computation and after it (completion_utc);
  input_snapshot_utc is the first read, or `inputs_utc` (the instant the caller began reading
  `results`) when that is earlier, so a sighting written after the inputs were read is never one
  the snapshot saw. A deep copy
  of the model observes only the day's results first seen (#225's first observed publication) at
  or before min(cutoff, snapshot) in a cell with a fitted par, so the caller's model never changes
  and a result seen later, or never, cannot move the forecast whatever the clock reads; each is
  listed in results_used with its sighting and top-3 time. A result of that set without a top-3
  time at the snapshot is listed in results_pending (id and sighting; [] when none) and feeds
  nothing. Index labels of `results` carry no meaning. par_s is the cell's par, same_day_s the
  par plus the same-day variant's mean at the cutoff. `latest_post()` (default: the post at the
  cutoff) is read after the computation: latest_post_utc. mode is "live" when completion_utc is at
  or before both cutoffs (each post less the primary minutes), "replay" at or after either post or
  whenever replay=True, else "late". The provenance fields (post_utc_at_cutoff, input_cutoff_utc)
  stay those of the post the inputs were taken for. A race without a post or in a cell the fit
  never saw is "skipped", with its own reason.
- `coverage(records, card)`: per venue and "all", every race of the latest card in the
  denominator. A race counts once; readiness is judged against both the post recorded at the
  cutoff and the latest card's post, so a post moved earlier after the record cannot leave it on
  time.
- `verify_replay(records, *, card, results, model, model_version, card_version)`: [] when every
  ok record matches, else one problem per mismatch, each naming its race id. `card` is the card
  version's snapshot: the venue, post and model keys of every race, as the recorder had them.
  The record's own inputs (its results_used, keyed by the card, and its input snapshot) must
  recompute its par_s and same_day_s, and each result it used must be one the cutoff allowed.
  `results` is the independent source of sightings and finish times, cross-checked both ways:
  each result used or pending must be in it with the same first sighting (and, when used, the
  same top-3 time), and each result it shows eligible by min(cutoff, snapshot), keyed by the card
  where the card lists the race, must be used or pending. A changed input value is a problem even
  when the forecast it gives is unchanged.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import Any

import numpy as np
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


def _has_par(model: SpeedModel, row: Mapping) -> bool:
    try:
        model.par(**{key: row[key] for key in model.spec.par_keys})
    except KeyError:
        return False
    return True


def _eligible(results: pd.DataFrame, race: Mapping, model: SpeedModel, as_of: pd.Timestamp):
    """The results the same-day variant may use: another race of the target's JST day and variant
    group, in a cell with a fitted par, first seen at or before `as_of`; renamed for
    `SpeedModel.observe`. Returns (timed, pending): pending have no top-3 time yet. One positional
    mask selects the rows, posts and sightings, whatever the index labels are."""
    seen = pd.to_datetime(results["first_seen_utc"], utc=True).reset_index(drop=True)
    posts = pd.to_datetime(results["post_utc"], utc=True).reset_index(drop=True)
    keep = (seen.notna() & (seen <= as_of) & posts.notna()).to_numpy()
    keep &= (results["race_id"] != race["race_id"]).to_numpy()
    day = _stamp(race["post_utc"]).tz_convert(JST).date()
    keep &= (posts.dt.tz_convert(JST).dt.date == day).to_numpy()
    for key in model.spec.variant_keys:
        keep &= (results[key] == race[key]).to_numpy()
    rows = results.to_dict("records")
    for i in np.flatnonzero(keep):
        keep[i] = _has_par(model, rows[i])
    frame = results.iloc[keep].copy()
    frame["race_utc"], frame["result_available_utc"] = posts[keep].array, seen[keep].array
    frame = frame.sort_values(["result_available_utc", "race_id"], kind="stable")
    timed = frame["top3_time_s"].notna().to_numpy()
    return frame.iloc[timed], frame.iloc[~timed]


def _mode(completion: pd.Timestamp, posts: tuple, replay: bool) -> str | None:
    """Judged against every post given (the post at the cutoff and the latest one)."""
    if replay:
        return REPLAY
    if posts[0] is None:
        return None  # no post: the timing cannot be judged
    if completion >= min(posts):
        return REPLAY
    return LIVE if completion <= min(posts) - _primary() else LATE


def _refuse_observed(model: SpeedModel) -> None:
    if model.n_observed or model.n_unseen:
        raise ValueError(
            f"the model already holds {model.n_observed + model.n_unseen} observed result(s): a "
            "residual it cached could move same_day_s without being in results_used; pass the "
            "fitted model before any observe()"
        )


def record_race(
    race: Mapping,
    *,
    model: SpeedModel,
    results: pd.DataFrame,
    model_version: str,
    card_version: str,
    now: Callable[[], datetime],
    replay: bool = False,
    latest_post: Callable[[], datetime | None] | None = None,
    inputs_utc: datetime | None = None,
) -> dict[str, Any]:
    """One record for one card race (see the module docstring)."""
    _refuse_observed(model)
    snapshot = _stamp(now())  # read before the work: when the inputs were taken
    if snapshot is None:
        raise ValueError("now() gave no time")
    read = _stamp(inputs_utc)
    snapshot = snapshot if read is None else min(snapshot, read)
    post = _stamp(race.get("post_utc"))
    cutoff = None if post is None else post - _primary()
    par_s = same_day_s = reason = None
    used: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
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
            frame, waiting = _eligible(results, race, model, min(cutoff, snapshot))
            try:
                local.observe(frame)
            except ValueError as exc:
                reason, par_s = f"{REFUSED}: {exc}", None
            else:
                target = {**race, "race_utc": post}
                at = cutoff + pd.Timedelta(1, "ns")  # the variant counts times strictly before
                same_day_s = float(local.expected_top3_time(target, at_utc=at))
                used = [
                    {"race_id": rid, "first_seen_utc": _iso(_stamp(seen)), "top3_time_s": float(t)}
                    for rid, seen, t in zip(
                        frame["race_id"],
                        frame["result_available_utc"],
                        frame["top3_time_s"],
                        strict=True,
                    )
                ]
                pending = [
                    {"race_id": rid, "first_seen_utc": _iso(_stamp(seen))}
                    for rid, seen in zip(
                        waiting["race_id"], waiting["result_available_utc"], strict=True
                    )
                ]
    latest = None if latest_post is None else _stamp(latest_post())  # read after the work
    latest = post if latest is None else latest
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
        "latest_post_utc": _iso(latest),
        "input_snapshot_utc": _iso(snapshot),
        "completion_utc": _iso(completion),
        "par_s": par_s,
        "same_day_s": same_day_s,
        "results_used": used,
        "results_pending": pending,
        "mode": _mode(completion, (post, latest), replay),
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
    """The problems found checking each ok record against its recorded inputs and the source;
    [] when none (see the module docstring)."""
    _refuse_observed(model)
    keys = ["venue", "post_utc", *dict.fromkeys([*model.spec.par_keys, *model.spec.variant_keys])]
    rows = {r["race_id"]: r for r in card.to_dict("records")}
    source = _keyed_by_card(results, rows, keys)
    sightings: dict[Any, tuple[pd.Timestamp | None, Any]] = {}
    for rid, seen, t in zip(
        source["race_id"], source["first_seen_utc"], source["top3_time_s"], strict=True
    ):
        sightings.setdefault(rid, (_stamp(seen), t))
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
        snapshot = _stamp(rec.get("input_snapshot_utc"))
        if snapshot is None:
            problems.append(f"{race_id}: no input snapshot recorded")
            continue
        used = list(rec.get("results_used") or ())
        pending = list(rec.get("results_pending") or ())
        # 1. the record's own inputs give its forecast, and the cutoff allowed each of them
        inputs, unlisted = _recorded_inputs(used, rows, keys)
        problems += [f"{race_id}: results_used has {rid}, not on the card" for rid in unlisted]
        again = record_race(
            rows[race_id],
            model=model,
            results=inputs,
            model_version=model_version,
            card_version=card_version,
            now=lambda at=snapshot: at,  # the inputs as the recorded snapshot saw them
            replay=True,
        )
        if again["status"] != OK:
            problems.append(f"{race_id}: recomputes as skipped ({again['reason']})")
            continue
        for name in ("post_utc_at_cutoff", "input_cutoff_utc"):
            if not _same_time(rec.get(name), again[name]):
                problems.append(f"{race_id}: {name} {rec.get(name)} is not {again[name]}")
        allowed = {(u["race_id"], _stamp(u["first_seen_utc"])) for u in again["results_used"]}
        for u in used:
            if u["race_id"] in unlisted:
                continue
            if (u["race_id"], _stamp(u.get("first_seen_utc"))) not in allowed:
                problems.append(
                    f"{race_id}: results_used has {u['race_id']} first seen "
                    f"{u.get('first_seen_utc')}, which the cutoff does not allow with a time"
                )
        for name in ("par_s", "same_day_s"):
            if not _close(rec.get(name), again[name]):
                problems.append(f"{race_id}: {name} {rec.get(name)} recomputes as {again[name]}")
        # 2. the source agrees with every result listed, used or pending
        for kind, entries in (("results_used", used), ("results_pending", pending)):
            for u in entries:
                rid, seen = u["race_id"], _stamp(u.get("first_seen_utc"))
                if rid not in sightings:
                    problems.append(f"{race_id}: {kind} has {rid}, not in the source")
                    continue
                truth, t = sightings[rid]
                if seen != truth:
                    problems.append(
                        f"{race_id}: {kind} has {rid} first seen {_iso(seen)}, the source "
                        f"{_iso(truth)}"
                    )
                if kind == "results_used" and not _close(u.get("top3_time_s"), _time(t)):
                    problems.append(
                        f"{race_id}: results_used has {rid} at top3_time_s "
                        f"{u.get('top3_time_s')}, the source has {_time(t)}"
                    )
        # 3. every result the source shows eligible by the snapshot is listed
        cutoff = _stamp(again["input_cutoff_utc"])
        timed, waiting = _eligible(source, rows[race_id], model, min(cutoff, snapshot))
        listed = {u["race_id"] for u in (*used, *pending)}
        for rid, seen in zip(
            [*timed["race_id"], *waiting["race_id"]],
            [*timed["result_available_utc"], *waiting["result_available_utc"]],
            strict=True,
        ):
            if rid not in listed:
                problems.append(f"{race_id}: results_used lacks {rid} first seen {_iso(seen)}")
    return problems


def _time(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _keyed_by_card(results: pd.DataFrame, rows: Mapping, keys: list[str]) -> pd.DataFrame:
    """The source with each race's venue, post and model keys taken from the card where the card
    lists it (what the recorder keyed it by), its own elsewhere; sightings and times its own."""
    out = []
    for row in results.to_dict("records"):
        on = rows.get(row["race_id"])
        out.append(row if on is None else {**row, **{k: on[k] for k in keys if k in on}})
    frame = pd.DataFrame(out, columns=list(results.columns))
    for col in ("post_utc", "first_seen_utc"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def _recorded_inputs(used: list, rows: Mapping, keys: list[str]) -> tuple[pd.DataFrame, set]:
    """The record's results_used as a results frame keyed by the card, and the ids not on it."""
    out, unlisted = [], set()
    for u in used:
        on = rows.get(u["race_id"])
        if on is None:
            unlisted.add(u["race_id"])
            continue
        out.append(
            {
                "race_id": u["race_id"],
                **{k: on.get(k) for k in keys},
                "top3_time_s": _time(u.get("top3_time_s")),
                "first_seen_utc": _stamp(u.get("first_seen_utc")),
            }
        )
    frame = pd.DataFrame(out, columns=["race_id", *keys, "top3_time_s", "first_seen_utc"])
    frame["top3_time_s"] = frame["top3_time_s"].astype(float)
    for col in ("post_utc", "first_seen_utc"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame, unlisted
