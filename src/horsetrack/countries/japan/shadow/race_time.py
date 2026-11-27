"""Race-time forecasts at the primary cutoff, their timing and their coverage (ticket #414).

A research recorder for #305's race-time model (`models.speed`), independent of the V0 shadow
loader. Nothing here is adopted (D16) and nothing reads an outcome of the race forecast.

- `build_inputs(race, results, *, model, read_utc)`: the frozen, JSON-ready snapshot a forecast
  is computed from: "read_utc" (when the inputs were read), "target" (race_id, venue, post_utc,
  post_source, the race's values of the model's par and variant keys), "rail" ({"value",
  "available_utc"}: the race's rail and `race["rail_available_utc"]`, the instant it was
  published, null when not given) and "results" (every row of `results` on the target's JST day,
  or without a post (its day unknown; never eligible), but the target's own, with its own
  race_id, venue, post_utc, model keys, first_seen_utc and top3_time_s, null where missing).
  Index labels of `results` carry no meaning.
- `forecast(inputs, model)`: pure (no clock, no I/O, neither argument changed): status, reason,
  par_s, same_day_s, results_used and results_pending from `inputs` alone. The input cutoff is
  the target's post less the primary decision minutes (config/decision.yaml). The model must hold
  no observations (ValueError otherwise: a residual it cached could move same_day_s unlisted). A
  deep copy of the model observes only the results first seen (#225's first observed
  publication) at or before min(cutoff, read_utc), of another race of the target's JST day and
  variant group, in a cell with a fitted par; each is listed in results_used with its sighting
  and top-3 time. A result of that set without a top-3 time is listed in results_pending (id and
  sighting) and feeds nothing. par_s is the cell's par, same_day_s the par plus the same-day
  variant's mean at the cutoff. A race without a post, in a cell the fit never saw, or whose
  results the model refuses is "skipped", with its own reason.
- `record_race(race, *, model, results, model_version, card_version, now, replay=False,
  latest_post=None, inputs_utc=None)`: one JSON-ready record for one card race: build_inputs at
  the input snapshot, forecast, the post check and the completion read; the record carries the
  snapshot as "inputs". `now()` is read before the computation and after it (completion_utc);
  input_snapshot_utc is the first read, or `inputs_utc` (the instant the caller began reading
  `results`) when that is earlier. `latest_post()` (default: the post at the cutoff) is read
  after the computation: latest_post_utc. mode is "live" when completion_utc is at or before both
  cutoffs (each post less the primary minutes), "replay" at or after either post or whenever
  replay=True, else "late". mode is provisional: `coverage` rejudges it against the latest card.
- `coverage(records, card)`: per venue and "all", every race of the latest card in the
  denominator. A race counts once; readiness is judged against both the post recorded at the
  cutoff and the latest card's post, so a post moved earlier after the record cannot leave it on
  time.
- `verify_replay(records, *, model, model_version)`: [] when every ok record matches, else one
  problem per mismatch, each naming its race id. Each ok record is recomputed from its own
  `inputs` only (no card, no store, no sidecar): its model version, inputs for its own race, its
  snapshot, post and cutoff, par_s, same_day_s, results_used and results_pending must agree with
  them, and every sighting it lists and a known rail's publication time must be at or before
  min(cutoff, read_utc); a rail of "unknown" needs no time. A changed input value is a problem
  even when the forecast it gives is unchanged.
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
RAIL_UNKNOWN = "unknown"
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

__all__ = ["build_inputs", "coverage", "forecast", "record_race", "verify_replay"]


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


def _native(value: Any) -> Any:
    """A JSON-ready scalar: ISO text for a time, a Python number for a numpy one, None for NA."""
    if isinstance(value, datetime):
        return _iso(_stamp(value))
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if not isinstance(value, (str, int, float, bool)) and pd.isna(value):
        return None
    return value


def _time(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _primary() -> pd.Timedelta:
    return pd.Timedelta(minutes=load_decision_grid().primary)


def _model_keys(model: SpeedModel) -> list[str]:
    return list(dict.fromkeys([*model.spec.par_keys, *model.spec.variant_keys]))


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


def build_inputs(
    race: Mapping, results: pd.DataFrame, *, model: SpeedModel, read_utc: datetime
) -> dict[str, Any]:
    """The frozen snapshot one forecast is computed from (see the module docstring)."""
    read = _stamp(read_utc)
    if read is None:
        raise ValueError("read_utc gave no time")
    keys = _model_keys(model)
    post = _stamp(race.get("post_utc"))
    target = {
        "race_id": _native(race["race_id"]),
        "venue": _native(race["venue"]),
        "post_utc": _iso(post),
        "post_source": _native(race.get("post_source")),
        **{key: _native(race.get(key)) for key in keys},
    }
    rail = {
        "value": _native(race.get("rail")),
        "available_utc": _iso(_stamp(race.get("rail_available_utc"))),
    }
    rows: list[dict[str, Any]] = []
    if post is not None:  # without a post the target has no day: no result is one of its day
        day = post.tz_convert(JST).date()
        posts = pd.to_datetime(results["post_utc"], utc=True)
        seen = pd.to_datetime(results["first_seen_utc"], utc=True)
        for row, at, first in zip(results.to_dict("records"), posts, seen, strict=True):
            if row["race_id"] == race["race_id"]:
                continue
            if pd.notna(at) and at.tz_convert(JST).date() != day:  # no post: kept, never eligible
                continue
            rows.append(
                {
                    "race_id": _native(row["race_id"]),
                    "venue": _native(row.get("venue")),
                    "post_utc": _iso(_stamp(at)),
                    **{key: _native(row.get(key)) for key in keys},
                    "first_seen_utc": _iso(_stamp(first)),
                    "top3_time_s": _time(row.get("top3_time_s")),
                }
            )
    return {"read_utc": _iso(read), "target": target, "rail": rail, "results": rows}


def _results_frame(entries: list, keys: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(["race_id", "venue", "post_utc", *keys]))
    frame = pd.DataFrame([{c: e.get(c) for c in columns} for e in entries], columns=columns).astype(
        object
    )
    frame["first_seen_utc"] = pd.to_datetime([e.get("first_seen_utc") for e in entries], utc=True)
    frame["top3_time_s"] = pd.Series([_time(e.get("top3_time_s")) for e in entries], dtype=float)
    frame["post_utc"] = pd.to_datetime(frame["post_utc"], utc=True)
    return frame


def forecast(inputs: Mapping, model: SpeedModel) -> dict[str, Any]:
    """par_s, same_day_s and the results they used, from `inputs` alone (module docstring)."""
    _refuse_observed(model)
    target = inputs["target"]
    post = _stamp(target.get("post_utc"))
    out: dict[str, Any] = {
        "status": SKIPPED,
        "reason": None,
        "par_s": None,
        "same_day_s": None,
        "results_used": [],
        "results_pending": [],
    }
    if post is None:
        out["reason"] = NO_POST
        return out
    read = _stamp(inputs["read_utc"])
    if read is None:
        raise ValueError("the inputs have no read_utc")
    cutoff = post - _primary()
    local = copy.deepcopy(model)  # the caller's model observes nothing
    try:
        par_s = float(local.par(**{key: target.get(key) for key in model.spec.par_keys}))
    except KeyError:
        out["reason"] = UNSEEN_CELL
        return out
    race = {**target, "post_utc": post}
    frame = _results_frame(list(inputs.get("results") or ()), _model_keys(model))
    timed, waiting = _eligible(frame, race, model, min(cutoff, read))
    try:
        local.observe(timed)
    except ValueError as exc:
        out["reason"] = f"{REFUSED}: {exc}"
        return out
    at = cutoff + pd.Timedelta(1, "ns")  # the variant counts times strictly before
    same_day_s = float(local.expected_top3_time({**race, "race_utc": post}, at_utc=at))
    used = [
        {"race_id": rid, "first_seen_utc": _iso(_stamp(seen)), "top3_time_s": float(t)}
        for rid, seen, t in zip(
            timed["race_id"], timed["result_available_utc"], timed["top3_time_s"], strict=True
        )
    ]
    pending = [
        {"race_id": rid, "first_seen_utc": _iso(_stamp(seen))}
        for rid, seen in zip(waiting["race_id"], waiting["result_available_utc"], strict=True)
    ]
    out.update(status=OK, par_s=par_s, same_day_s=same_day_s)
    out.update(results_used=used, results_pending=pending)
    return out


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
    inputs = build_inputs(race, results, model=model, read_utc=snapshot)
    got = forecast(inputs, model)
    post = _stamp(race.get("post_utc"))
    cutoff = None if post is None else post - _primary()
    latest = None if latest_post is None else _stamp(latest_post())  # read after the work
    latest = post if latest is None else latest
    completion = _stamp(now())  # read after the work
    if completion is None:
        raise ValueError("now() gave no time")
    return {
        "race_id": race["race_id"],
        "venue": race["venue"],
        "status": got["status"],
        "reason": got["reason"],
        "model_version": model_version,
        "card_version": card_version,
        "input_cutoff_utc": _iso(cutoff),
        "post_utc_at_cutoff": _iso(post),
        "latest_post_utc": _iso(latest),
        "input_snapshot_utc": _iso(snapshot),
        "completion_utc": _iso(completion),
        "par_s": got["par_s"],
        "same_day_s": got["same_day_s"],
        "results_used": got["results_used"],
        "results_pending": got["results_pending"],
        "mode": _mode(completion, (post, latest), replay),
        "inputs": inputs,
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


def _same_entries(listed: list, again: list, timed: bool) -> bool:
    """The listed results are the recomputed ones: ids, sightings and, when timed, times."""

    def norm(entries: list) -> list:
        return sorted((e.get("race_id"), _stamp(e.get("first_seen_utc"))) for e in entries)

    if norm(listed) != norm(again):
        return False
    if not timed:
        return True
    key = lambda e: (e.get("race_id"), _stamp(e.get("first_seen_utc")))  # noqa: E731
    return all(
        _close(a.get("top3_time_s"), b.get("top3_time_s"))
        for a, b in zip(sorted(listed, key=key), sorted(again, key=key), strict=True)
    )


def _check(rec: Mapping, inputs: Mapping, model: SpeedModel) -> list[str]:
    """The problems of one ok record against its own inputs, without its race id."""
    problems: list[str] = []
    target = inputs.get("target") or {}
    if target.get("race_id") != rec["race_id"]:
        problems.append(f"its inputs are for {target.get('race_id')!r}")
    read = _stamp(inputs.get("read_utc"))
    if read is None:
        return [*problems, "its inputs have no read_utc"]
    if not _same_time(rec.get("input_snapshot_utc"), read):
        problems.append(f"input_snapshot_utc {rec.get('input_snapshot_utc')} is not {_iso(read)}")
    post = _stamp(target.get("post_utc"))
    cutoff = None if post is None else post - _primary()
    for name, want in (("post_utc_at_cutoff", post), ("input_cutoff_utc", cutoff)):
        if not _same_time(rec.get(name), want):
            problems.append(f"{name} {rec.get(name)} is not {_iso(want)} from its inputs")
    again = forecast(inputs, model)
    if again["status"] != OK:
        return [*problems, f"its inputs recompute as skipped ({again['reason']})"]
    for name in ("par_s", "same_day_s"):
        if not _close(rec.get(name), again[name]):
            problems.append(f"{name} {rec.get(name)} recomputes as {again[name]}")
    used = list(rec.get("results_used") or ())
    pending = list(rec.get("results_pending") or ())
    for kind, listed, timed in (("results_used", used, True), ("results_pending", pending, False)):
        if not _same_entries(listed, again[kind], timed):
            problems.append(f"{kind} {listed} recomputes from its inputs as {again[kind]}")
    as_of = min(cutoff, read)
    for kind, listed in (("results_used", used), ("results_pending", pending)):
        for entry in listed:
            seen = _stamp(entry.get("first_seen_utc"))
            if seen is None or seen > as_of:
                problems.append(
                    f"{kind} has {entry.get('race_id')} first seen {_iso(seen)}, not by "
                    f"{_iso(as_of)}"
                )
    rail = inputs.get("rail") or {}
    value = rail.get("value")
    if "rail" in target and target["rail"] != value:
        problems.append(f"its target's rail {target['rail']!r} is not the input rail {value!r}")
    if value is not None and value != RAIL_UNKNOWN:
        published = _stamp(rail.get("available_utc"))
        if published is None:
            problems.append(f"rail {value!r} has no publication time")
        elif published > as_of:
            problems.append(f"rail {value!r} published {_iso(published)}, after {_iso(as_of)}")
    return problems


def verify_replay(
    records: Iterable[Mapping], *, model: SpeedModel, model_version: str
) -> list[str]:
    """The problems found recomputing each ok record from its own inputs; [] when none (see the
    module docstring)."""
    _refuse_observed(model)
    problems: list[str] = []
    for rec in records:
        if rec.get("status") != OK:
            continue
        race_id = rec["race_id"]
        if rec.get("model_version") != model_version:
            problems.append(
                f"{race_id}: model_version {rec.get('model_version')!r} is not {model_version!r}"
            )
        inputs = rec.get("inputs")
        if not isinstance(inputs, Mapping) or not inputs:
            problems.append(f"{race_id}: no inputs recorded")
            continue
        try:
            found = _check(rec, inputs, model)
        except (KeyError, TypeError, ValueError) as exc:
            found = [f"its inputs cannot be recomputed ({exc!r})"]
        problems += [f"{race_id}: {p}" for p in found]
    return problems
