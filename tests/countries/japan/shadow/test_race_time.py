"""Race-time forecasts at the primary cutoff, their timing and their coverage (ticket #414).

Contract pinned here (`horsetrack.countries.japan.shadow.race_time`; the coder follows it):

- `record_race(race, *, model, results, model_version, card_version, now, replay=False,
  latest_post=None) -> dict`: one record for one card race.
  - `race`: a mapping with race_id, venue, post_utc (the scheduled post known at the cutoff, an
    aware UTC datetime, or None when the card has none) and the model's par keys (track, surface,
    distance, layout, rail, class_level).
  - `model`: a fitted #305 `SpeedModel` (`models.speed`) holding no observations, never changed by
    the call (a copy observes the results), so the same model serves any number of calls in any
    order. A model that already holds observations (`observe` was called on it, e.g. with the
    #305 study's post + 30 min proxy) is refused with ValueError: a residual it cached could carry
    a result seen after the cutoff into same_day_s without being in results_used.
  - `results`: a DataFrame of the day's results seen so far: race_id, venue, post_utc, the par
    keys, top3_time_s and first_seen_utc (the first observed publication, #225; NaT when never
    seen).
  - The input cutoff is post_utc - the primary decision minutes (`load_decision_grid().primary`,
    config/decision.yaml, D2: T-2).
  - `now()` is read twice: before the computation (input_snapshot_utc, when the inputs were taken)
    and after it (completion_utc). Only results first seen at or before min(input cutoff, input
    snapshot) feed the same-day variant, whatever the clock says afterwards and whatever
    post_utc + 30 min says.
  - `latest_post`: an optional callable read after the computation, giving the latest known post
    (aware UTC); without it the latest post is the post at the cutoff.
  - Fields: race_id, venue, status ("ok" or "skipped"), reason (None when ok, a non-empty text
    when skipped, one per cause), model_version, card_version (as given), input_cutoff_utc,
    post_utc_at_cutoff (the post the inputs were taken for, kept for provenance), latest_post_utc
    (the post read after the computation), input_snapshot_utc, completion_utc (ISO-8601 UTC text,
    null where there is no post), par_s (the cell's par, seconds), same_day_s (par + the same-day
    variant's mean at the cutoff, seconds; both null when skipped), results_used (a list of
    {"race_id", "first_seen_utc", "top3_time_s"}: the results the variant used, only those the
    model accepted, i.e. in a cell with a fitted par) and mode, judged against both the post at
    the cutoff and the latest post:
      "live"   completion_utc <= both cutoffs (each post - primary), in a live run;
      "late"   after either cutoff but before both posts, in a live run;
      "replay" at or after either post, or in any run with replay=True (a retrospective
               calculation).
  - A race whose cell the fit never saw, or without a post time, gives a skipped record.
  - Every record is JSON-serialisable as is (timestamps are text).
- `coverage(records, card) -> dict[str, dict[str, int]]`: per venue and "all", over every race of
  `card` (the latest known card, a DataFrame of race rows; post_utc may have moved since a record
  was made): scheduled, forecast (an ok record), skipped (no ok record, a skipped one), missing (no
  record), ready_by_cutoff (an ok live record completed by its input cutoff and by the latest
  post - primary), ready_by_post (an ok live or late record completed before its recorded post and
  the latest post), replay (forecast but not ready_by_post). A race counts once whatever its
  number of records. Records read back from JSON give the same counts.
- `verify_replay(records, *, card, results, model, model_version, card_version) -> list[str]`:
  the problems found recomputing each ok record from its recorded inputs (card row, the recorded
  input snapshot, results_used against `results` by id, sighting and top3_time_s value, versions):
  an empty list when inputs and predictions match; a problem about a race names its race_id. A
  changed input value is a problem even when the forecast it gives is unchanged.

Every race, fit and result is synthetic (a planted par per cell and variant per day); expected
forecasts come from a separate copy of the #305 model fed only the eligible results, never from
the recorder.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import copy
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.board.snapshot import load_decision_grid
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.shadow.race_time import coverage, record_race, verify_replay

UTC = timezone.utc
PAR_KEYS = ("track", "surface", "distance", "layout", "rail", "class_level")
VARIANT_KEYS = ("track", "surface")
SPEC = SpeedSpec(par_keys=PAR_KEYS, variant_keys=VARIANT_KEYS, same_day_variant=True, t_df=None)
PRIMARY = timedelta(minutes=load_decision_grid().primary)  # D2's T-2, read from decision.yaml
MV = "speed-c3:fit-0414a"
CV = "card:2025-02-15:sha-7f3c"
FIELDS = {
    "race_id",
    "venue",
    "status",
    "reason",
    "model_version",
    "card_version",
    "input_cutoff_utc",
    "post_utc_at_cutoff",
    "latest_post_utc",
    "input_snapshot_utc",
    "completion_utc",
    "par_s",
    "same_day_s",
    "results_used",
    "mode",
}

# --------------------------------------------------------------------------- the planted world

T16 = ("TOK", "turf", 1600, "outer", "B", 3)
T20 = ("TOK", "turf", 2000, "outer", "B", 3)
TD14 = ("TOK", "dirt", 1400, "outer", "unknown", 2)
K16 = ("KYO", "turf", 1600, "inner", "B", 3)
KD18 = ("KYO", "dirt", 1800, "outer", "unknown", 3)
UNSEEN = ("TOK", "turf", 1800, "outer", "B", 3)  # never in the fit frame
PAR = {T16: 96.0, T20: 120.5, TD14: 85.2, K16: 95.4, KD18: 112.7}

CELLS = {"TOK": (T16, TD14, T20, TD14, T16, T20), "KYO": (KD18, K16, KD18, K16, KD18, K16)}
POSTS = {  # UTC (hh, mm); 10:00-13:05 JST, so the JST and UTC dates agree
    "TOK": ((1, 0), (1, 35), (2, 10), (2, 45), (3, 20), (3, 55)),
    "KYO": ((1, 10), (1, 45), (2, 20), (2, 55), (3, 30), (4, 5)),
}
TRAIN_DAYS = tuple(date(2025, 1, 25) + timedelta(days=k) for k in range(6))
DAY = date(2025, 2, 15)  # the development day replayed; none of its races is in the fit frame

V_DAY = {("TOK", "turf"): 0.9, ("TOK", "dirt"): -0.6, ("KYO", "turf"): 0.4, ("KYO", "dirt"): -1.1}
EPS_DAY = {"TOK": (0.2, -0.1, 0.5, 0.3, -0.2, 0.1), "KYO": (-0.3, 0.2, 0.1, -0.2, 0.3, 0.0)}
# minutes from post to the result's first sighting; TOK R5's inquiry (40) outlasts TOK R6's cutoff
LAG_MIN = {"TOK": (11, 14, 9, 22, 40, 13), "KYO": (12, 10, 26, 15, 9, 19)}


def rid(venue: str, no: int, day: date = DAY) -> str:
    return f"jra:{day.isoformat()}:{venue}:{no:02d}"


def at(day: date, hh: int, mm: int) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=UTC)


def keys_of(cell: tuple) -> dict:
    return dict(zip(PAR_KEYS, cell, strict=True))


def fit_frame() -> pd.DataFrame:
    rng = np.random.default_rng(414)
    groups = sorted({c[:2] for c in PAR})
    shifts = {g: rng.normal(0.0, 0.6, len(TRAIN_DAYS)) for g in groups}
    rows = []
    for k, day in enumerate(TRAIN_DAYS):
        for venue, cells in CELLS.items():
            for no, (cell, (hh, mm)) in enumerate(zip(cells, POSTS[venue], strict=True), 1):
                post = at(day, hh, mm)
                rows.append(
                    {
                        "race_id": rid(venue, no, day),
                        "race_utc": post,
                        **keys_of(cell),
                        "top3_time_s": PAR[cell] + shifts[cell[:2]][k] + rng.normal(0.0, 0.25),
                        "result_available_utc": post + timedelta(minutes=30),
                    }
                )
    return pd.DataFrame(rows)


def card_rows() -> dict[str, dict]:
    """race_id -> card row: R1-R6 at each venue, TOK R7 in an unseen cell, KYO R7 without a post."""
    rows = {}
    for venue, cells in CELLS.items():
        for no, (cell, (hh, mm)) in enumerate(zip(cells, POSTS[venue], strict=True), 1):
            rows[rid(venue, no)] = {
                "race_id": rid(venue, no),
                "venue": venue,
                "post_utc": at(DAY, hh, mm),
                **keys_of(cell),
            }
    rows[rid("TOK", 7)] = {
        "race_id": rid("TOK", 7),
        "venue": "TOK",
        "post_utc": at(DAY, 4, 30),
        **keys_of(UNSEEN),
    }
    rows[rid("KYO", 7)] = {
        "race_id": rid("KYO", 7),
        "venue": "KYO",
        "post_utc": None,
        **keys_of(K16),
    }
    return rows


CARD = card_rows()
FORECASTABLE = tuple(rid(v, n) for v in CELLS for n in range(1, 7))


def card_frame(rows=None) -> pd.DataFrame:
    rows = list(CARD.values()) if rows is None else rows
    frame = pd.DataFrame(rows)
    frame["post_utc"] = pd.to_datetime(frame["post_utc"], utc=True)
    return frame


def day_results() -> pd.DataFrame:
    rows = []
    for venue, cells in CELLS.items():
        for no, cell in enumerate(cells, 1):
            race = CARD[rid(venue, no)]
            rows.append(
                {
                    **race,
                    "top3_time_s": PAR[cell] + V_DAY[cell[:2]] + EPS_DAY[venue][no - 1],
                    "first_seen_utc": race["post_utc"] + timedelta(minutes=LAG_MIN[venue][no - 1]),
                }
            )
    frame = pd.DataFrame(rows)
    for col in ("post_utc", "first_seen_utc"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


@pytest.fixture(scope="module")
def model() -> SpeedModel:
    return SpeedModel(SPEC).fit(fit_frame())


@pytest.fixture
def results() -> pd.DataFrame:
    return day_results()


# --------------------------------------------------------------------------- helpers


class Clock:
    """A fake UTC clock: reading it never moves it."""

    def __init__(self, start: datetime) -> None:
        self.at = start

    def __call__(self) -> datetime:
        return self.at

    def advance(self, seconds: float) -> None:
        self.at += timedelta(seconds=seconds)


def utc(text: str) -> datetime:
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0), text
    return stamp


def cutoff_of(race_id: str) -> datetime:
    return CARD[race_id]["post_utc"] - PRIMARY


def eligible(results: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    seen = results["first_seen_utc"]
    return results[seen.notna() & (seen <= pd.Timestamp(cutoff))]


def expected(model: SpeedModel, race: dict, results: pd.DataFrame, cutoff: datetime):
    """(par, par + same-day variant) from a separate copy of the #305 model that observes only the
    results first seen by the cutoff."""
    ref = copy.deepcopy(model)
    seen = eligible(results, cutoff).rename(
        columns={"post_utc": "race_utc", "first_seen_utc": "result_available_utc"}
    )
    ref.observe(seen)
    cell = {k: race[k] for k in PAR_KEYS}
    target = {"race_id": race["race_id"], "race_utc": race["post_utc"], **cell}
    late = cutoff + timedelta(days=30)  # only eligible results were observed
    return model.par(**cell), ref.expected_top3_time(target, at_utc=late)


def same_group(results: pd.DataFrame, race: dict) -> pd.DataFrame:
    mask = (results["track"] == race["track"]) & (results["surface"] == race["surface"])
    return results[mask & (results["race_id"] != race["race_id"])]


def used_ids(record: dict) -> set[str]:
    return {u["race_id"] for u in record["results_used"]}


def record(model, results, race_id, when, *, replay=False, race=None, mv=MV, cv=CV, **extra):
    return record_race(
        CARD[race_id] if race is None else race,
        model=model,
        results=results,
        model_version=mv,
        card_version=cv,
        now=Clock(when),
        replay=replay,
        **extra,
    )


def by_id(records: list[dict]) -> dict[str, dict]:
    return {r["race_id"]: r for r in records}


# --------------------------------------------------------------------------- the forecasts


def test_both_forecasts_in_seconds_at_the_primary_cutoff(model, results):
    """DoD 3 and outline 1-2: every race of both venues, recorded just before its cutoff, carries
    the par and par + same-day forecasts in seconds, its versions, cutoff, post, completion time
    and the results it could use; nothing first seen after the cutoff is among them."""
    with_variant = without_variant = 0
    for race_id in FORECASTABLE:
        race = CARD[race_id]
        cutoff = cutoff_of(race_id)
        when = cutoff - timedelta(seconds=30)
        mv, cv = f"{MV}-{race_id[-2:]}", f"{CV}-{race['venue']}"
        rec = record(model, results, race_id, when, mv=mv, cv=cv)
        assert FIELDS <= set(rec), race_id
        json.dumps(rec)  # storable as is

        par, same_day = expected(model, race, results, cutoff)
        assert rec["race_id"] == race_id and rec["venue"] == race["venue"]
        assert rec["status"] == "ok" and rec["reason"] is None
        assert (rec["model_version"], rec["card_version"]) == (mv, cv)
        assert utc(rec["input_cutoff_utc"]) == cutoff
        assert utc(rec["post_utc_at_cutoff"]) == race["post_utc"]
        assert utc(rec["latest_post_utc"]) == race["post_utc"]  # no latest_post: the same post
        assert utc(rec["input_snapshot_utc"]) == when
        assert utc(rec["completion_utc"]) == when
        assert rec["mode"] == "live"
        assert isinstance(rec["par_s"], float) and isinstance(rec["same_day_s"], float)
        assert rec["par_s"] == pytest.approx(par, abs=1e-9)
        assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)

        group = same_group(results, race)
        could = set(eligible(group, cutoff)["race_id"])
        later = set(group["race_id"]) - could
        assert could <= used_ids(rec), race_id
        assert not used_ids(rec) & (later | {race_id}), race_id
        sightings = dict(zip(results["race_id"], results["first_seen_utc"], strict=True))
        times = dict(zip(results["race_id"], results["top3_time_s"], strict=True))
        for used in rec["results_used"]:
            seen = utc(used["first_seen_utc"])
            assert seen == sightings[used["race_id"]] and seen <= cutoff, (race_id, used)
            assert isinstance(used["top3_time_s"], float), (race_id, used)
            assert used["top3_time_s"] == pytest.approx(times[used["race_id"]], abs=1e-12)
        if could:
            with_variant += 1
            assert abs(rec["same_day_s"] - rec["par_s"]) > 1e-3, race_id
        else:
            without_variant += 1
            assert rec["same_day_s"] == pytest.approx(rec["par_s"], abs=1e-9), race_id
    assert with_variant >= 6 and without_variant >= 4  # the day exercises both


def test_skipped_races_carry_a_reason_per_cause(model, results):
    """Outline 2: a race in a cell the fit never saw and a race without a post time give skipped
    records, each with its own reason and no forecast."""
    unseen = record(model, results, rid("TOK", 7), cutoff_of(rid("TOK", 7)) - timedelta(minutes=1))
    no_post = record(model, results, rid("KYO", 7), at(DAY, 3, 0))
    for rec in (unseen, no_post):
        assert FIELDS <= set(rec)
        json.dumps(rec)
        assert rec["status"] == "skipped"
        assert isinstance(rec["reason"], str) and rec["reason"].strip()
        assert rec["par_s"] is None and rec["same_day_s"] is None
    assert unseen["reason"] != no_post["reason"]
    assert utc(unseen["input_cutoff_utc"]) == cutoff_of(rid("TOK", 7))
    assert no_post["input_cutoff_utc"] is None and no_post["post_utc_at_cutoff"] is None


def test_the_model_passed_in_is_never_changed(model, results):
    """Recording in any order gives the same forecasts, and the caller's model observes nothing."""
    pars = {cell: model.par(**keys_of(cell)) for cell in PAR}
    forward = [record(model, results, r, cutoff_of(r) - timedelta(seconds=5)) for r in FORECASTABLE]
    backward = [
        record(model, results, r, cutoff_of(r) - timedelta(seconds=5))
        for r in reversed(FORECASTABLE)
    ]
    a, b = by_id(forward), by_id(backward)
    for race_id in FORECASTABLE:
        assert a[race_id]["same_day_s"] == pytest.approx(b[race_id]["same_day_s"], abs=1e-12)
    assert {cell: model.par(**keys_of(cell)) for cell in PAR} == pars
    assert model.n_observed == 0
    mean, sd = model.variant(DAY, "TOK", "turf", at_utc=at(DAY, 23, 0))
    assert (mean, sd) == (0.0, model.tau)


# --------------------------------------------------------------------------- timing


@pytest.mark.parametrize("race_id", [rid("TOK", 3), rid("KYO", 5)])
@pytest.mark.parametrize(
    ("anchor", "offset_s", "mode"),
    [
        ("cutoff", -600, "live"),
        ("cutoff", -1, "live"),
        ("cutoff", 0, "live"),
        ("cutoff", 1, "late"),
        ("cutoff", 60, "late"),
        ("post", -1, "late"),
        ("post", 0, "replay"),
        ("post", 3600, "replay"),
        ("post", 3 * 86400, "replay"),
    ],
)
def test_synthetic_clocks_before_after_cutoff_and_after_post(
    model, results, race_id, anchor, offset_s, mode
):
    """Tests that must pass (1) and DoD 2: completion before the cutoff is live, after it late, at
    or after the scheduled post replay; the forecast itself stays the one as of the cutoff."""
    race = CARD[race_id]
    base = cutoff_of(race_id) if anchor == "cutoff" else race["post_utc"]
    when = base + timedelta(seconds=offset_s)
    rec = record(model, results, race_id, when)
    assert rec["mode"] == mode
    assert utc(rec["completion_utc"]) == when
    par, same_day = expected(model, race, results, cutoff_of(race_id))
    assert rec["par_s"] == pytest.approx(par, abs=1e-9)
    assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)


@pytest.mark.parametrize("offset_s", [-600, -1, 30, 3 * 86400])
def test_a_declared_replay_is_replay_whatever_the_clock(model, results, offset_s):
    """Outline 3: a retrospective calculation is marked replay even when its clock reads early."""
    race_id = rid("KYO", 4)
    rec = record(
        model, results, race_id, cutoff_of(race_id) + timedelta(seconds=offset_s), replay=True
    )
    assert rec["mode"] == "replay"


class SlowModel(SpeedModel):
    """The #305 model whose every read takes STEP seconds on the shared fake clock (a class
    attribute, so a copy of the model moves the same clock)."""

    clock: Clock | None = None
    step: float = 0.0

    def _tick(self) -> None:
        type(self).clock.advance(type(self).step)

    def observe(self, results):
        self._tick()
        return super().observe(results)

    def par(self, **keys):
        self._tick()
        return super().par(**keys)

    def variant(self, race_date, *group, at_utc):
        self._tick()
        return super().variant(race_date, *group, at_utc=at_utc)

    def expected_top3_time(self, race, *, at_utc):
        self._tick()
        return super().expected_top3_time(race, at_utc=at_utc)


@pytest.mark.parametrize(
    ("start_before_cutoff_s", "step_s", "live"),
    [(10, 20.0, False), (5, 400.0, False), (600, 1.0, True)],
)
def test_completion_is_measured_after_the_computation(
    model, results, start_before_cutoff_s, step_s, live
):
    """DoD 2: a computation that starts before the cutoff but ends after it is not on time; the
    completion time is the clock after the work, not before it."""
    race_id = rid("TOK", 5)
    slow = copy.deepcopy(model)
    slow.__class__ = SlowModel
    start = cutoff_of(race_id) - timedelta(seconds=start_before_cutoff_s)
    clock = Clock(start)
    SlowModel.clock, SlowModel.step = clock, step_s
    try:
        rec = record_race(
            CARD[race_id],
            model=slow,
            results=results,
            model_version=MV,
            card_version=CV,
            now=clock,
        )
    finally:
        SlowModel.clock, SlowModel.step = None, 0.0
    done = utc(rec["completion_utc"])
    assert done == clock.at  # read after the last piece of work
    assert done >= start + timedelta(seconds=step_s)
    assert utc(rec["input_snapshot_utc"]) == start  # read before the work
    if live:
        assert rec["mode"] == "live"
    else:
        assert done > cutoff_of(race_id)
        assert rec["mode"] == ("late" if done < CARD[race_id]["post_utc"] else "replay")


# --------------------------------------------------------------------------- as of the cutoff


@pytest.mark.parametrize(
    "seen_after_cutoff", [timedelta(seconds=1), timedelta(minutes=5), timedelta(hours=2), None]
)
@pytest.mark.parametrize("clock_after_cutoff", [timedelta(seconds=-30), timedelta(hours=3)])
def test_a_result_first_seen_after_the_cutoff_does_not_move_the_forecast(
    model, results, seen_after_cutoff, clock_after_cutoff
):
    """Tests that must pass (2): TOK R3's result (same group as TOK R5, run 70 minutes before R5's
    cutoff, so post + 30 min is long past) first seen after R5's cutoff, or never, gives R5 the
    forecast it has without that result, even when the computation runs after the sighting."""
    target, moved = rid("TOK", 5), rid("TOK", 3)
    cutoff = cutoff_of(target)
    assert CARD[moved]["post_utc"] + timedelta(minutes=30) < cutoff
    when = cutoff + clock_after_cutoff

    on_time = record(model, results, target, when)  # R3 seen at 02:19, well before the cutoff
    without = record(model, results[results["race_id"] != moved], target, when)
    assert moved in used_ids(on_time) and moved not in used_ids(without)
    assert abs(on_time["same_day_s"] - without["same_day_s"]) > 1e-3  # R3 matters when seen

    late = results.copy()
    seen = pd.NaT if seen_after_cutoff is None else pd.Timestamp(cutoff + seen_after_cutoff)
    late.loc[late["race_id"] == moved, "first_seen_utc"] = seen
    rec = record(model, late, target, when)
    assert moved not in used_ids(rec)
    assert rec["same_day_s"] == pytest.approx(without["same_day_s"], abs=1e-12)
    assert rec["par_s"] == pytest.approx(on_time["par_s"], abs=1e-12)


def _proxy_rows(results: pd.DataFrame, race_id: str, days: int = 0) -> pd.DataFrame:
    """One result as the #305 study gives it to `observe`: available at post + 30 min."""
    rows = results[results["race_id"] == race_id].copy()
    rows["race_utc"] = rows["post_utc"] + pd.Timedelta(days=days)
    rows["race_id"] = rows["race_id"] + ("" if days == 0 else f"+{days}d")
    rows["result_available_utc"] = rows["race_utc"] + pd.Timedelta(minutes=30)
    return rows.drop(columns=["post_utc", "first_seen_utc"])


@pytest.mark.parametrize("days", [0, 1], ids=["same day, post + 30 min proxy", "another day"])
def test_a_model_that_already_holds_observations_is_refused(model, results, days):
    """Review finding 3: a fitted model that already observed results (here TOK R3 by the post +
    30 min proxy, while its real first sighting is after TOK R5's cutoff) is refused, so a cached
    residual can never move same_day_s; the recorder and the verifier both refuse it."""
    target, leaked = rid("TOK", 5), rid("TOK", 3)
    cutoff = cutoff_of(target)
    late = results.copy()
    late.loc[late["race_id"] == leaked, "first_seen_utc"] = pd.Timestamp(
        cutoff + timedelta(minutes=5)
    )
    pre = copy.deepcopy(model).observe(_proxy_rows(late, leaked, days))
    assert pre.n_observed == 1
    if days == 0:  # the cached residual would move the variant at the cutoff
        assert pre.variant(DAY, "TOK", "turf", at_utc=cutoff)[0] != 0.0

    when = cutoff - timedelta(seconds=30)
    clean = record(model, late, target, when)
    assert leaked not in used_ids(clean)
    with pytest.raises(ValueError):
        record(pre, late, target, when)
    with pytest.raises(ValueError):
        verify_replay(
            [clean], card=card_frame(), results=late, model=pre, model_version=MV, card_version=CV
        )


@pytest.mark.parametrize(
    ("snapshot_vs_cutoff_s", "included"),
    [(-10, False), (-6, False), (-5, True), (-1, True), (0, True)],
)
def test_the_inputs_are_those_seen_by_the_snapshot_before_the_cutoff(
    model, results, snapshot_vs_cutoff_s, included
):
    """Review finding 4: a computation that takes its inputs before the cutoff uses only results
    first seen by then (TOK R3 first seen 5 s before TOK R5's cutoff); the record keeps that
    snapshot, and replay verification rebuilds the same inputs and accepts it."""
    target, early = rid("TOK", 5), rid("TOK", 3)
    cutoff = cutoff_of(target)
    source = results.copy()
    source.loc[source["race_id"] == early, "first_seen_utc"] = pd.Timestamp(
        cutoff - timedelta(seconds=5)
    )
    when = cutoff + timedelta(seconds=snapshot_vs_cutoff_s)
    rec = record(model, source, target, when)
    assert rec["mode"] == "live"
    assert utc(rec["input_snapshot_utc"]) == when
    assert (early in used_ids(rec)) is included
    par, same_day = expected(model, CARD[target], source, min(cutoff, when))
    assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)
    problems = verify_replay(
        [rec], card=card_frame(), results=source, model=model, model_version=MV, card_version=CV
    )
    assert problems == []


def test_only_results_the_model_accepted_are_listed_as_used(model, results):
    """Review (low): a same-group result in a cell the fit never saw cannot feed the variant, so
    it is not in results_used and leaves the forecast as it was."""
    target, unseen = rid("TOK", 5), rid("TOK", 9)
    post = at(DAY, 1, 20)
    extra = {
        "race_id": unseen,
        "venue": "TOK",
        "post_utc": post,
        **keys_of(UNSEEN),
        "top3_time_s": 109.0,
        "first_seen_utc": post + timedelta(minutes=12),
    }
    source = pd.concat([results, pd.DataFrame([extra])], ignore_index=True)
    for col in ("post_utc", "first_seen_utc"):
        source[col] = pd.to_datetime(source[col], utc=True)
    when = cutoff_of(target) - timedelta(seconds=30)
    base = record(model, results, target, when)
    rec = record(model, source, target, when)
    assert unseen not in used_ids(rec)
    assert used_ids(rec) == used_ids(base) and used_ids(base)
    assert rec["same_day_s"] == pytest.approx(base["same_day_s"], abs=1e-12)
    problems = verify_replay(
        [rec], card=card_frame(), results=source, model=model, model_version=MV, card_version=CV
    )
    assert problems == []


# --------------------------------------------------------------------------- coverage


def test_coverage_by_venue_keeps_every_scheduled_race(model, results):
    """DoD 1 and outline 3: readiness by cutoff and by post per venue and overall, with skipped and
    unrecorded races in the denominator and a moved post re-judged against the latest card."""
    s = timedelta(seconds=1)

    def rec(name, no, anchor, offset_s, *, replay=False):
        race_id = rid(name, no)
        base = cutoff_of(race_id) if anchor == "cutoff" else CARD[race_id]["post_utc"]
        return record(model, results, race_id, base + offset_s * s, replay=replay)

    records = [
        rec("TOK", 1, "cutoff", -30),  # live
        rec("TOK", 2, "cutoff", -90),  # live
        rec("TOK", 3, "cutoff", 30),  # late
        rec("TOK", 4, "post", 600),  # replay
        rec("TOK", 5, "cutoff", -20),  # live ...
        rec("TOK", 5, "post", 7200),  # ... and recomputed later: counted once
        # TOK R6: never recorded
        record(model, results, rid("TOK", 7), cutoff_of(rid("TOK", 7)) - 60 * s),  # skipped
        rec("KYO", 1, "cutoff", -45),  # live
        rec("KYO", 2, "cutoff", -300, replay=True),  # declared replay
        rec("KYO", 3, "cutoff", -20),  # live, then the post moves 3 min earlier
        rec("KYO", 4, "cutoff", -20),  # live, then the post moves 1 min earlier
        rec("KYO", 5, "cutoff", 90),  # late
        rec("KYO", 6, "cutoff", -10),  # live
        record(model, results, rid("KYO", 7), at(DAY, 3, 0)),  # skipped: no post
    ]
    latest = copy.deepcopy(CARD)
    latest[rid("KYO", 3)]["post_utc"] -= timedelta(minutes=3)
    latest[rid("KYO", 4)]["post_utc"] -= timedelta(minutes=1)
    card = card_frame(list(latest.values()))

    want = {
        "TOK": dict(
            scheduled=7,
            forecast=5,
            skipped=1,
            missing=1,
            ready_by_cutoff=3,
            ready_by_post=4,
            replay=1,
        ),
        "KYO": dict(
            scheduled=7,
            forecast=6,
            skipped=1,
            missing=0,
            ready_by_cutoff=2,
            ready_by_post=4,
            replay=2,
        ),
    }
    want["all"] = {k: want["TOK"][k] + want["KYO"][k] for k in want["TOK"]}
    got = coverage(records, card)
    for scope, counts in want.items():
        assert {k: got[scope][k] for k in counts} == counts, scope
    stored = json.loads(json.dumps(records))
    again = coverage(stored, card)
    for scope, counts in want.items():
        assert {k: again[scope][k] for k in counts} == counts, scope


@pytest.mark.parametrize(
    ("moved_s", "by_cutoff", "by_post"),
    [(-10, 1, 1), (-60, 0, 1), (-180, 0, 0), (-3600, 0, 0)],
)
@pytest.mark.parametrize("race_id", [rid("KYO", 3), rid("TOK", 4)])
def test_a_post_moved_earlier_after_the_record_cannot_stay_on_time(
    model, results, race_id, moved_s, by_cutoff, by_post
):
    """DoD 2: a record made live 20 s before its cutoff is re-judged against the latest post: a
    move that leaves it before the new cutoff keeps it ready, a larger one does not."""
    old = CARD[race_id]
    rec = record(model, results, race_id, cutoff_of(race_id) - timedelta(seconds=20))
    assert rec["mode"] == "live"
    moved = {**old, "post_utc": old["post_utc"] + timedelta(seconds=moved_s)}
    got = coverage([rec], card_frame([moved]))[old["venue"]]
    assert (got["ready_by_cutoff"], got["ready_by_post"]) == (by_cutoff, by_post)
    assert got["replay"] == 1 - by_post and got["scheduled"] == got["forecast"] == 1


@pytest.mark.parametrize(("moved_s", "mode"), [(-60, "late"), (-180, "replay")])
def test_a_post_moved_before_the_computation_moves_the_cutoff(model, results, moved_s, mode):
    """DoD 2: when the card known at the cutoff already carries the earlier post, the cutoff is
    taken from it, so a clock that was on time for the old post is late or after the post."""
    race_id = rid("KYO", 3)
    old = CARD[race_id]
    moved = {**old, "post_utc": old["post_utc"] + timedelta(seconds=moved_s)}
    rec = record(model, results, race_id, cutoff_of(race_id) - timedelta(seconds=20), race=moved)
    assert utc(rec["input_cutoff_utc"]) == moved["post_utc"] - PRIMARY
    assert utc(rec["post_utc_at_cutoff"]) == moved["post_utc"]
    assert rec["mode"] == mode


P_S = PRIMARY.total_seconds()


@pytest.mark.parametrize("race_id", [rid("TOK", 3), rid("KYO", 5)])
@pytest.mark.parametrize(
    ("done_vs_cutoff_s", "latest_shift_s", "mode"),
    [
        (-30, 0, "live"),  # the post did not move
        (-30, -25, "live"),  # moved earlier, completion still by the new cutoff
        (-30, -60, "late"),  # past the new cutoff, before the new post
        (-30, -(P_S + 30), "replay"),  # the new post is the completion instant
        (-30, -(P_S + 60), "replay"),  # 12:00 -> 11:57 while computing, done 11:57:30
        (-30, 300, "live"),  # moved later: the original cutoff was met
        (30, 300, "late"),  # moved later, but done after the cutoff the inputs were taken for
    ],
)
def test_the_mode_is_judged_against_the_latest_post_too(
    model, results, race_id, done_vs_cutoff_s, latest_shift_s, mode
):
    """Review finding 2 (DoD 2): the latest post, read after the computation, can only make a
    record less on time; the record keeps the post its inputs were taken for and the cutoff and
    forecast of that post, and records the latest post beside it."""
    race = CARD[race_id]
    post = race["post_utc"]
    latest = post + timedelta(seconds=latest_shift_s)
    when = cutoff_of(race_id) + timedelta(seconds=done_vs_cutoff_s)
    rec = record(model, results, race_id, when, latest_post=lambda: latest)
    json.dumps(rec)
    assert rec["mode"] == mode
    assert utc(rec["post_utc_at_cutoff"]) == post
    assert utc(rec["latest_post_utc"]) == latest
    assert utc(rec["input_cutoff_utc"]) == post - PRIMARY
    par, same_day = expected(model, race, results, cutoff_of(race_id))
    assert rec["par_s"] == pytest.approx(par, abs=1e-9)
    assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)


def test_a_post_moved_while_the_forecast_runs_is_read_after_it(model, results):
    """Review finding 2: the post moves earlier while the computation runs (the feed changes once
    the clock passes the start); the latest post is the one read after the computation, so the
    record is not live although it was on time for the post read before it."""
    race_id = rid("TOK", 3)
    post = CARD[race_id]["post_utc"]
    start = cutoff_of(race_id) - timedelta(seconds=60)
    moved = start + PRIMARY - timedelta(seconds=30)  # its cutoff is 30 s before the start
    slow = copy.deepcopy(model)
    slow.__class__ = SlowModel
    clock = Clock(start)

    def feed() -> datetime:
        return post if clock.at < start + timedelta(seconds=0.5) else moved

    SlowModel.clock, SlowModel.step = clock, 1.0
    try:
        rec = record_race(
            CARD[race_id],
            model=slow,
            results=results,
            model_version=MV,
            card_version=CV,
            now=clock,
            latest_post=feed,
        )
    finally:
        SlowModel.clock, SlowModel.step = None, 0.0
    done = utc(rec["completion_utc"])
    assert done > start
    assert utc(rec["latest_post_utc"]) == moved
    assert utc(rec["post_utc_at_cutoff"]) == post
    assert rec["mode"] == ("late" if done < moved else "replay")


# --------------------------------------------------------------------------- replay


REPLAY_AT = datetime(2025, 2, 18, 0, 0, tzinfo=UTC)  # days after the development day


def replay_day(model, results):
    return [record(model, results, r, REPLAY_AT, replay=True) for r in CARD]


def test_a_replay_of_the_day_verifies_and_keeps_the_coverage_facts(model, results):
    """DoD 1 and 3: a replay days later labels every calculation replay, forecasts as of each
    cutoff, verifies against its recorded inputs (stored or not), and still reports coverage."""
    records = replay_day(model, results)
    assert {r["mode"] for r in records if r["status"] == "ok"} == {"replay"}
    for race_id in FORECASTABLE:
        par, same_day = expected(model, CARD[race_id], results, cutoff_of(race_id))
        assert by_id(records)[race_id]["same_day_s"] == pytest.approx(same_day, abs=1e-9)

    kwargs = dict(
        card=card_frame(), results=results, model=model, model_version=MV, card_version=CV
    )
    assert verify_replay(records, **kwargs) == []
    assert verify_replay(json.loads(json.dumps(records)), **kwargs) == []
    live = [record(model, results, r, cutoff_of(r) - timedelta(seconds=15)) for r in FORECASTABLE]
    assert verify_replay(live, **kwargs) == []

    got = coverage(records, card_frame())
    for venue in ("TOK", "KYO"):
        assert got[venue]["scheduled"] == 7
        assert (got[venue]["forecast"], got[venue]["skipped"]) == (6, 1)
        assert (got[venue]["ready_by_cutoff"], got[venue]["ready_by_post"]) == (0, 0)
        assert got[venue]["replay"] == 6
    assert got["all"]["scheduled"] == 14 and got["all"]["replay"] == 12


def _shift(text: str, **delta) -> str:
    return (utc(text) + timedelta(**delta)).isoformat()


def _bump_same_day(records, results):
    by_id(records)[rid("TOK", 3)]["same_day_s"] += 0.05
    return rid("TOK", 3)


def _bump_par(records, results):
    by_id(records)[rid("KYO", 2)]["par_s"] -= 0.05
    return rid("KYO", 2)


def _move_a_sighting(records, results):
    rec = by_id(records)[rid("TOK", 5)]
    entry = next(u for u in rec["results_used"] if u["race_id"] == rid("TOK", 1))
    entry["first_seen_utc"] = _shift(entry["first_seen_utc"], minutes=1)
    return rid("TOK", 5)


def _drop_a_result(records, results):
    rec = by_id(records)[rid("KYO", 4)]
    rec["results_used"] = [u for u in rec["results_used"] if u["race_id"] != rid("KYO", 2)]
    return rid("KYO", 4)


def _add_a_late_result(records, results):
    rec = by_id(records)[rid("TOK", 6)]
    seen = results.loc[results["race_id"] == rid("TOK", 5), "first_seen_utc"].iloc[0]
    rec["results_used"].append({"race_id": rid("TOK", 5), "first_seen_utc": seen.isoformat()})
    return rid("TOK", 6)


def _correct_a_result(records, results):
    results.loc[results["race_id"] == rid("TOK", 1), "top3_time_s"] += 0.4
    return rid("TOK", 3)


def _move_the_recorded_post(records, results):
    rec = by_id(records)[rid("KYO", 5)]
    rec["post_utc_at_cutoff"] = _shift(rec["post_utc_at_cutoff"], minutes=5)
    return rid("KYO", 5)


def _move_the_snapshot_before_a_used_sighting(records, results):
    rec = by_id(records)[rid("TOK", 5)]
    assert rid("TOK", 3) in used_ids(rec)  # first seen 02:19
    rec["input_snapshot_utc"] = at(DAY, 2, 18).isoformat()
    return rid("TOK", 5)


def _change_a_recorded_value(records, results):
    rec = by_id(records)[rid("KYO", 6)]
    rec["results_used"][0]["top3_time_s"] += 0.2
    return rid("KYO", 6)


def _correct_two_results_by_equal_and_opposite_amounts(records, results):
    results.loc[results["race_id"] == rid("TOK", 1), "top3_time_s"] += 0.3
    results.loc[results["race_id"] == rid("TOK", 3), "top3_time_s"] -= 0.3
    return rid("TOK", 5)


@pytest.mark.parametrize(
    "tamper",
    [
        _bump_same_day,
        _bump_par,
        _move_a_sighting,
        _drop_a_result,
        _add_a_late_result,
        _correct_a_result,
        _move_the_recorded_post,
        _move_the_snapshot_before_a_used_sighting,
        _change_a_recorded_value,
        _correct_two_results_by_equal_and_opposite_amounts,
    ],
)
def test_replay_verification_finds_mismatched_inputs_and_predictions(model, results, tamper):
    """DoD 1: a changed prediction, a recorded input that does not match the source, a result
    the cutoff did not allow, or a source corrected afterwards is a problem naming the race."""
    records = replay_day(model, results)
    changed = copy.deepcopy(records)
    source = results.copy()
    race_id = tamper(changed, source)
    problems = verify_replay(
        changed, card=card_frame(), results=source, model=model, model_version=MV, card_version=CV
    )
    assert problems and all(isinstance(p, str) for p in problems)
    assert any(race_id in p for p in problems)


def test_a_changed_input_value_fails_verification_even_with_an_unchanged_forecast(model, results):
    """Review finding 5: two finish times corrected by equal and opposite amounts leave the normal
    model's variant mean, hence the forecast, where it was; verification still reports the races
    that used them, from the input values the records keep."""
    records = replay_day(model, results)
    source = results.copy()
    _correct_two_results_by_equal_and_opposite_amounts(records, source)
    corrected = {rid("TOK", 1), rid("TOK", 3)}
    for target in (rid("TOK", 5), rid("TOK", 6)):
        rec = by_id(records)[target]
        assert corrected <= used_ids(rec)
        again = record(model, source, target, REPLAY_AT, replay=True)
        assert again["same_day_s"] == pytest.approx(rec["same_day_s"], abs=1e-9)  # blind to it
    problems = verify_replay(
        records, card=card_frame(), results=source, model=model, model_version=MV, card_version=CV
    )
    for target in (rid("TOK", 5), rid("TOK", 6)):
        assert any(target in p for p in problems), (target, problems)


@pytest.mark.parametrize("which", ["model_version", "card_version"])
def test_replay_verification_refuses_another_model_or_card(model, results, which):
    records = replay_day(model, results)
    kwargs = dict(model_version=MV, card_version=CV)
    kwargs[which] = kwargs[which] + "-other"
    problems = verify_replay(records, card=card_frame(), results=results, model=model, **kwargs)
    assert problems
