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
    seen). Its index labels carry no meaning (round-2 finding G: duplicate labels, as from a
    concat of venue frames, give the same records as a reindexed frame).
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
- Round-2 finding D: each record also carries results_pending, a list of {"race_id",
  "first_seen_utc"}: the results of the same eligible set (same JST day and variant group, a cell
  with a fitted par, first seen by min(cutoff, snapshot), not the target) whose top3_time_s was
  NaN at the snapshot; they do not feed the forecast.

Round 5 (the design review's plan, section 2): the record carries its own frozen inputs.
- `build_inputs(race, results, *, model, read_utc) -> dict`, JSON-ready (timestamps ISO-8601 UTC
  text, null where missing):
  - "read_utc": the instant the inputs were read;
  - "target": race_id, venue, post_utc, post_source (`race.get("post_source")`: "sidecar" or
    "card" from the day runner, null when the race does not say) and the race's values of the
    model's par and variant keys;
  - "rail": {"value": the race's rail, "available_utc": `race.get("rail_available_utc")`, the
    instant that rail was published, null when the race gives none};
  - "results": one entry per row of `results` on the target's JST day (another day's row never),
    each with its own race_id, venue, post_utc, model keys, first_seen_utc (null when never seen)
    and top3_time_s (a float, null while NaN).
- `forecast(inputs, model) -> dict`: pure (no clock, no I/O, neither argument changed), reading
  only `inputs`: status, reason, par_s, same_day_s, results_used and results_pending exactly as
  `record_race` gives them for the same inputs (eligible: first seen at or before min(cutoff,
  read_utc), the cutoff being the target's post less the primary minutes).
- `record_race` = build_inputs (read_utc = the input snapshot), forecast, the post check and the
  completion read; the record carries the snapshot as "inputs", and results_used and
  results_pending are what forecast derives from it.
- `verify_replay(records, *, model, model_version) -> list[str]`: recomputes each ok record from
  its `inputs` only (no card, no source: an earlier result missing from any card verifies on its
  own keys). A problem, naming the race id, for: another model_version; no inputs; inputs that
  recompute as skipped or to another par_s, same_day_s, results_used or results_pending (ids,
  sightings, times); a post_utc_at_cutoff or input_cutoff_utc that is not the inputs' post (less
  the primary minutes); an input_snapshot_utc that is not inputs.read_utc; any sighting listed
  in results_used or results_pending, or a known rail's available_utc, missing or later than
  min(cutoff, read_utc). A rail of "unknown" needs no time. A changed input value is a problem
  even when the forecast it gives is unchanged. A model holding observations is refused
  (ValueError).

Round 6 (review round 5 on PR #420):
- Mixed precision: the inputs' and the results' timestamps may mix whole-second and
  fractional-second ISO text (`...T01:12:00+00:00` beside `...T01:47:00.250000+00:00`); every
  record is still written, each instant keeping its sub-second value when compared.
- Standalone rail check: `forecast` (hence `record_race`) never uses a rail whose available_utc is
  missing or later than min(cutoff, read_utc); that rail counts as "unknown" when the par is
  chosen. Without a fitted par for the unknown cell the race is skipped with a reason; with one,
  the forecast is the unknown cell's, and the honest record verifies. A rail published at or
  before the bound is used.
- Format version: every new record carries "format": 2. `verify_replay` now returns a result with
  `problems` (the list of texts above, [] when clean) and `legacy` (the race ids, in log order,
  of ok rows written by the earlier recorder: no "format" and no "inputs"). A legacy row is not
  verifiable from the record: it is listed in `legacy` and is never a problem, so a log holding
  both formats verifies its new rows as before. A format-2 row without inputs is still a problem.

Every race, fit and result is synthetic (a planted par per cell and variant per day); expected
forecasts come from a separate copy of the #305 model fed only the eligible results, never from
the recorder. Each turf race of the card carries rail B published at 00:00 UTC (09:00 JST).
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
from horsetrack.countries.japan.shadow import race_time
from horsetrack.countries.japan.shadow.race_time import coverage, record_race, verify_replay


def build_inputs(race, results, *, model, read_utc):
    """`race_time.build_inputs`, looked up at call time (round 5 adds it)."""
    return race_time.build_inputs(race, results, model=model, read_utc=read_utc)


def forecast(inputs, model):
    """`race_time.forecast`, looked up at call time (round 5 adds it)."""
    return race_time.forecast(inputs, model)


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
    "results_pending",
    "mode",
    "format",  # round 6: the record format version, 2 for every new record
}
FORMAT = 2
OK_FIELDS = FIELDS | {"inputs"}  # round 5: an ok record carries its frozen inputs
MODEL_KEYS = tuple(dict.fromkeys([*PAR_KEYS, *VARIANT_KEYS]))

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


RAIL_AT = at(DAY, 0, 0)  # the day's turf rail published at 09:00 JST


def rail_time(cell: tuple) -> datetime | None:
    return None if cell[4] == "unknown" else RAIL_AT


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
                "rail_available_utc": rail_time(cell),
            }
    rows[rid("TOK", 7)] = {
        "race_id": rid("TOK", 7),
        "venue": "TOK",
        "post_utc": at(DAY, 4, 30),
        **keys_of(UNSEEN),
        "rail_available_utc": RAIL_AT,
    }
    rows[rid("KYO", 7)] = {
        "race_id": rid("KYO", 7),
        "venue": "KYO",
        "post_utc": None,
        **keys_of(K16),
        "rail_available_utc": RAIL_AT,
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
                    **{k: v for k, v in race.items() if k != "rail_available_utc"},
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


def verify(records, model, mv=MV) -> list[str]:
    """The problems of a log of new-format records only (round 6: none of them is legacy)."""
    got = verify_replay(records, model=model, model_version=mv)
    assert list(got.legacy) == [], got.legacy
    return list(got.problems)


def same_time(text, want) -> bool:
    """An ISO text (or null) read back equals a datetime (or None/NaT)."""
    if want is None or pd.isna(want):
        return text is None
    return text is not None and utc(text) == pd.Timestamp(want)


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
        assert OK_FIELDS <= set(rec), race_id
        json.dumps(rec)  # storable as is
        assert utc(rec["inputs"]["read_utc"]) == when
        assert rec["inputs"]["target"]["race_id"] == race_id

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


# --------------------------------------------------------------------------- round 5: the inputs

OTHER_DAY = DAY - timedelta(days=7)


def with_gaps(results: pd.DataFrame) -> pd.DataFrame:
    """The day's results with TOK R2's time not yet in, KYO R3 never seen and a race of another
    day (TOK R1 a week before, seen 12 minutes after its post)."""
    source = results.copy()
    source.loc[source["race_id"] == rid("TOK", 2), "top3_time_s"] = np.nan
    source.loc[source["race_id"] == rid("KYO", 3), "first_seen_utc"] = pd.NaT
    week = {
        **{k: v for k, v in CARD[rid("TOK", 1)].items() if k != "rail_available_utc"},
        "race_id": rid("TOK", 1, OTHER_DAY),
        "post_utc": at(OTHER_DAY, 1, 0),
        "top3_time_s": 96.7,
        "first_seen_utc": at(OTHER_DAY, 1, 12),
    }
    source = pd.concat([source, pd.DataFrame([week])], ignore_index=True)
    for col in ("post_utc", "first_seen_utc"):
        source[col] = pd.to_datetime(source[col], utc=True)
    return source


@pytest.mark.parametrize(
    ("race_id", "read_vs_cutoff_s", "post_source"),
    [
        (rid("TOK", 3), -30, "sidecar"),
        (rid("TOK", 4), -600, "card"),  # dirt: rail unknown, no publication time
        (rid("KYO", 5), 0, None),
        (rid("TOK", 6), 3 * 3600, "sidecar"),
    ],
)
def test_build_inputs_freezes_the_target_its_rail_and_every_same_day_result(
    model, results, race_id, read_vs_cutoff_s, post_source
):
    """Plan section 2: the snapshot holds the read instant, the target (ids, post, post source,
    the model's keys), its rail with the instant it was published, and every result row of the
    target's JST day with its own keys, post, first sighting and time (null while missing);
    another day's row is left out, and the snapshot is JSON as is."""
    source = with_gaps(results)
    race = dict(CARD[race_id])
    if post_source is not None:
        race["post_source"] = post_source
    read = cutoff_of(race_id) + timedelta(seconds=read_vs_cutoff_s)
    got = build_inputs(race, source, model=model, read_utc=read)
    assert json.loads(json.dumps(got)) == got

    assert utc(got["read_utc"]) == read
    target = got["target"]
    assert (target["race_id"], target["venue"]) == (race_id, race["venue"])
    assert utc(target["post_utc"]) == race["post_utc"]
    assert target["post_source"] == post_source
    assert {k: target[k] for k in MODEL_KEYS} == {k: race[k] for k in MODEL_KEYS}
    assert got["rail"]["value"] == race["rail"]
    assert same_time(got["rail"]["available_utc"], race["rail_available_utc"])

    rows = {r["race_id"]: r for r in got["results"]}
    assert len(rows) == len(got["results"])
    day_rows = source[source["race_id"].str.contains(DAY.isoformat())]
    assert set(rows) - {race_id} == set(day_rows["race_id"]) - {race_id}
    assert rid("TOK", 1, OTHER_DAY) not in rows
    for row in day_rows.to_dict("records"):
        if row["race_id"] == race_id and race_id not in rows:
            continue
        entry = rows[row["race_id"]]
        assert entry["venue"] == row["venue"]
        assert same_time(entry["post_utc"], row["post_utc"]), entry
        assert {k: entry[k] for k in MODEL_KEYS} == {k: row[k] for k in MODEL_KEYS}, entry
        assert same_time(entry["first_seen_utc"], row["first_seen_utc"]), entry
        if pd.isna(row["top3_time_s"]):
            assert entry["top3_time_s"] is None, entry
        else:
            assert isinstance(entry["top3_time_s"], float), entry
            assert entry["top3_time_s"] == pytest.approx(row["top3_time_s"], abs=1e-12)


@pytest.mark.parametrize("read_vs_cutoff_s", [-3600, -600, -30, 0, 3 * 3600])
@pytest.mark.parametrize(
    "race_id", [rid("TOK", 3), rid("TOK", 5), rid("TOK", 6), rid("KYO", 4), rid("KYO", 6)]
)
def test_forecast_of_built_inputs_is_the_independent_model_as_of_the_read_or_cutoff(
    model, results, race_id, read_vs_cutoff_s
):
    """Plan section 2: forecast(build_inputs(...)) is the separate #305 copy fed only the results
    first seen by min(cutoff, read_utc); the used results are exactly that same-group set."""
    race = CARD[race_id]
    cutoff = cutoff_of(race_id)
    read = cutoff + timedelta(seconds=read_vs_cutoff_s)
    got = forecast(build_inputs(race, results, model=model, read_utc=read), model)
    as_of = min(cutoff, read)
    par, same_day = expected(model, race, results, as_of)
    assert (got["status"], got["reason"]) == ("ok", None)
    assert got["par_s"] == pytest.approx(par, abs=1e-9)
    assert got["same_day_s"] == pytest.approx(same_day, abs=1e-9)
    want = set(eligible(same_group(results, race), as_of)["race_id"])
    assert {u["race_id"] for u in got["results_used"]} == want
    assert got["results_pending"] == []


def test_forecast_reads_only_its_inputs_and_changes_nothing(model, results):
    """Plan section 2: forecast is pure. The same inputs (also read back from JSON) give the same
    answer twice, the inputs and the model are left as they were, and what it forecasts follows
    the inputs: the target relabelled as a 1600 m race takes that cell's par, and a result taken
    out of the inputs no longer feeds it."""
    target, early = rid("TOK", 5), rid("TOK", 3)
    rec = record(model, results, target, cutoff_of(target) - timedelta(seconds=30))
    inputs = json.loads(json.dumps(rec["inputs"]))
    frozen = copy.deepcopy(inputs)
    got = forecast(inputs, model)
    assert forecast(inputs, model) == got
    assert inputs == frozen and model.n_observed == 0
    for name in ("status", "reason", "par_s", "same_day_s", "results_used", "results_pending"):
        assert got[name] == rec[name], name
    assert early in {u["race_id"] for u in got["results_used"]}

    relabelled = copy.deepcopy(inputs)
    relabelled["target"]["distance"] = 1600  # T20 -> T16
    assert forecast(relabelled, model)["par_s"] == pytest.approx(model.par(**keys_of(T16)))

    fewer = copy.deepcopy(inputs)
    fewer["results"] = [r for r in fewer["results"] if r["race_id"] != early]
    less = forecast(fewer, model)
    assert early not in {u["race_id"] for u in less["results_used"]}
    without = results[results["race_id"] != early]
    _, same_day = expected(model, CARD[target], without, cutoff_of(target))
    assert less["same_day_s"] == pytest.approx(same_day, abs=1e-9)
    assert abs(less["same_day_s"] - got["same_day_s"]) > 1e-3


@pytest.mark.parametrize("replay", [False, True], ids=["live", "replay"])
def test_every_record_carries_the_inputs_its_forecast_is_derived_from(model, results, replay):
    """Plan section 2: record_race writes its snapshot as "inputs" (read_utc the input snapshot,
    the target's post the post at the cutoff), and its par_s, same_day_s, results_used and
    results_pending are what forecast gives for those inputs."""
    for race_id in FORECASTABLE:
        when = REPLAY_AT if replay else cutoff_of(race_id) - timedelta(seconds=30)
        rec = record(model, with_gaps(results), race_id, when, replay=replay)
        assert OK_FIELDS <= set(rec), race_id
        inputs = rec["inputs"]
        assert utc(inputs["read_utc"]) == utc(rec["input_snapshot_utc"]) == when
        assert utc(inputs["target"]["post_utc"]) == utc(rec["post_utc_at_cutoff"])
        assert inputs["rail"]["value"] == CARD[race_id]["rail"]
        again = forecast(json.loads(json.dumps(inputs)), model)
        for name in ("par_s", "same_day_s", "results_used", "results_pending"):
            assert again[name] == rec[name], (race_id, name)


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
        verify_replay([clean], model=pre, model_version=MV)


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
    assert verify([rec], model) == []


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
    assert unseen in {r["race_id"] for r in rec["inputs"]["results"]}  # in the snapshot, unused
    assert verify([rec], model) == []


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

    assert verify(records, model) == []
    assert verify(json.loads(json.dumps(records)), model) == []
    live = [record(model, results, r, cutoff_of(r) - timedelta(seconds=15)) for r in FORECASTABLE]
    assert verify(live, model) == []
    results.loc[:, "top3_time_s"] += 5.0  # the source changes afterwards: the records keep theirs
    assert verify(records + live, model) == []

    got = coverage(records, card_frame())
    for venue in ("TOK", "KYO"):
        assert got[venue]["scheduled"] == 7
        assert (got[venue]["forecast"], got[venue]["skipped"]) == (6, 1)
        assert (got[venue]["ready_by_cutoff"], got[venue]["ready_by_post"]) == (0, 0)
        assert got[venue]["replay"] == 6
    assert got["all"]["scheduled"] == 14 and got["all"]["replay"] == 12


def _shift(text: str, **delta) -> str:
    return (utc(text) + timedelta(**delta)).isoformat()


def _input(rec: dict, race_id: str) -> dict:
    return next(r for r in rec["inputs"]["results"] if r["race_id"] == race_id)


def _bump_same_day(records):
    by_id(records)[rid("TOK", 3)]["same_day_s"] += 0.05
    return rid("TOK", 3)


def _bump_par(records):
    by_id(records)[rid("KYO", 2)]["par_s"] -= 0.05
    return rid("KYO", 2)


def _move_a_used_sighting(records):
    rec = by_id(records)[rid("TOK", 5)]
    entry = next(u for u in rec["results_used"] if u["race_id"] == rid("TOK", 1))
    entry["first_seen_utc"] = _shift(entry["first_seen_utc"], minutes=1)
    return rid("TOK", 5)


def _drop_a_used_result(records):
    rec = by_id(records)[rid("KYO", 4)]
    rec["results_used"] = [u for u in rec["results_used"] if u["race_id"] != rid("KYO", 2)]
    return rid("KYO", 4)


def _add_a_late_result(records):
    rec = by_id(records)[rid("TOK", 6)]
    late = _input(rec, rid("TOK", 5))  # first seen 04:00, after TOK R6's 03:53 cutoff
    rec["results_used"].append({k: late[k] for k in ("race_id", "first_seen_utc", "top3_time_s")})
    return rid("TOK", 6)


def _change_a_used_time(records):
    rec = by_id(records)[rid("KYO", 6)]
    rec["results_used"][0]["top3_time_s"] += 0.2
    return rid("KYO", 6)


def _correct_an_input_time(records):
    _input(by_id(records)[rid("TOK", 3)], rid("TOK", 1))["top3_time_s"] += 0.4
    return rid("TOK", 3)


def _correct_two_input_times_by_equal_and_opposite_amounts(records):
    rec = by_id(records)[rid("TOK", 5)]
    _input(rec, rid("TOK", 1))["top3_time_s"] += 0.3
    _input(rec, rid("TOK", 3))["top3_time_s"] -= 0.3
    return rid("TOK", 5)


def _move_an_input_sighting_past_the_cutoff(records):
    rec = by_id(records)[rid("TOK", 5)]
    late = _shift(rec["input_cutoff_utc"], minutes=1)
    _input(rec, rid("TOK", 3))["first_seen_utc"] = late
    next(u for u in rec["results_used"] if u["race_id"] == rid("TOK", 3))["first_seen_utc"] = late
    return rid("TOK", 5)


def _move_the_recorded_post(records):
    rec = by_id(records)[rid("KYO", 5)]
    rec["post_utc_at_cutoff"] = _shift(rec["post_utc_at_cutoff"], minutes=5)
    return rid("KYO", 5)


def _move_the_input_post(records):
    rec = by_id(records)[rid("KYO", 5)]
    rec["inputs"]["target"]["post_utc"] = _shift(rec["inputs"]["target"]["post_utc"], minutes=5)
    return rid("KYO", 5)


def _move_the_read_before_a_used_sighting(records):
    rec = by_id(records)[rid("TOK", 5)]
    assert rid("TOK", 3) in used_ids(rec)  # first seen 02:19
    rec["input_snapshot_utc"] = rec["inputs"]["read_utc"] = at(DAY, 2, 18).isoformat()
    return rid("TOK", 5)


def _publish_the_rail_after_the_cutoff(records):
    rec = by_id(records)[rid("TOK", 3)]
    rec["inputs"]["rail"]["available_utc"] = _shift(rec["input_cutoff_utc"], minutes=1)
    return rid("TOK", 3)


def _drop_the_rail_time(records):
    rec = by_id(records)[rid("KYO", 4)]  # turf, rail B
    rec["inputs"]["rail"]["available_utc"] = None
    return rid("KYO", 4)


def _swap_in_another_races_inputs(records):
    rec = by_id(records)[rid("TOK", 5)]
    rec["inputs"] = copy.deepcopy(by_id(records)[rid("TOK", 3)]["inputs"])
    return rid("TOK", 5)


def _drop_the_inputs(records):
    del by_id(records)[rid("KYO", 6)]["inputs"]
    return rid("KYO", 6)


@pytest.mark.parametrize(
    "tamper",
    [
        _bump_same_day,
        _bump_par,
        _move_a_used_sighting,
        _drop_a_used_result,
        _add_a_late_result,
        _change_a_used_time,
        _correct_an_input_time,
        _correct_two_input_times_by_equal_and_opposite_amounts,
        _move_an_input_sighting_past_the_cutoff,
        _move_the_recorded_post,
        _move_the_input_post,
        _move_the_read_before_a_used_sighting,
        _publish_the_rail_after_the_cutoff,
        _drop_the_rail_time,
        _swap_in_another_races_inputs,
        _drop_the_inputs,
    ],
)
def test_replay_verification_finds_mismatched_inputs_and_predictions(model, results, tamper):
    """DoD 1 and plan section 2: verified from the records alone, a changed prediction, a listed
    result its inputs do not give, a changed input, an input sighting or rail published after
    min(cutoff, read_utc), a known rail without a publication time, another race's inputs or no
    inputs is a problem naming the race; the untouched records stay clean."""
    records = replay_day(model, results)
    assert verify(records, model) == []
    changed = copy.deepcopy(records)
    race_id = tamper(changed)
    problems = verify(changed, model)
    assert problems and all(isinstance(p, str) for p in problems)
    assert any(race_id in p for p in problems)
    others = [r for r in changed if r["race_id"] != race_id]
    assert verify(others, model) == []


def test_a_changed_input_value_fails_verification_even_with_an_unchanged_forecast(model, results):
    """Review finding 5: two finish times in a record's inputs changed by equal and opposite
    amounts leave the normal model's variant mean, hence the forecast, where it was; the record
    is still reported, because the results it lists no longer match its inputs."""
    records = replay_day(model, results)
    target = _correct_two_input_times_by_equal_and_opposite_amounts(records)
    rec = by_id(records)[target]
    again = forecast(rec["inputs"], model)
    assert again["same_day_s"] == pytest.approx(rec["same_day_s"], abs=1e-9)  # blind to it
    assert {rid("TOK", 1), rid("TOK", 3)} <= used_ids(rec)
    problems = verify(records, model)
    assert any(target in p for p in problems), problems


def test_replay_verification_refuses_another_model_version(model, results):
    records = replay_day(model, results)
    problems = verify(records, model, mv=MV + "-other")
    for race_id in FORECASTABLE:
        assert any(race_id in p for p in problems), race_id


def test_results_with_duplicate_index_labels_give_the_same_records(model, results):
    """Round-2 finding G: the two venues' results concatenated with their own default indices
    (labels 0-5 twice) give every race the record the reindexed frame gives, and verify."""
    parts = [results[results["venue"] == v].reset_index(drop=True) for v in ("KYO", "TOK")]
    stacked = pd.concat(parts)
    assert not stacked.index.is_unique
    for race_id in FORECASTABLE:
        when = cutoff_of(race_id) - timedelta(seconds=30)
        want = record(model, results, race_id, when)
        got = record(model, stacked, race_id, when)
        assert got["results_used"] == want["results_used"], race_id
        assert got["par_s"] == pytest.approx(want["par_s"], abs=1e-12), race_id
        assert got["same_day_s"] == pytest.approx(want["same_day_s"], abs=1e-12), race_id
        key = lambda r: r["race_id"]  # noqa: E731
        assert sorted(got["inputs"]["results"], key=key) == sorted(
            want["inputs"]["results"], key=key
        ), race_id
    records = [record(model, stacked, r, REPLAY_AT, replay=True) for r in CARD]
    assert verify(records, model) == []


def _pending(record: dict) -> set:
    return {(u["race_id"], utc(u["first_seen_utc"])) for u in record["results_pending"]}


@pytest.mark.parametrize(
    ("target", "pending", "elsewhere"),
    [
        (rid("TOK", 5), rid("TOK", 3), rid("TOK", 2)),  # TOK turf; TOK R2 is TOK dirt
        (rid("KYO", 4), rid("KYO", 2), rid("KYO", 1)),  # KYO turf; KYO R1 is KYO dirt
    ],
)
def test_a_result_seen_without_its_finish_time_is_pending_and_its_record_still_verifies(
    model, results, target, pending, elsewhere
):
    """Round-2 finding D: a result first seen by the cutoff whose finish time was not yet there
    (NaN) at the snapshot is listed in results_pending, not used, and leaves the forecast as it
    is without it; the record keeps the null time in its inputs, so it verifies from itself
    whatever the store holds later. With results_pending emptied the same record is reported, and
    moving a used result into results_pending cannot make a forecast that used it pass."""
    cutoff = cutoff_of(target)
    when = cutoff - timedelta(seconds=30)
    at_snapshot = results.copy()
    gap = at_snapshot["race_id"].isin([pending, elsewhere])
    at_snapshot.loc[gap, "top3_time_s"] = np.nan  # the store had not got their times yet
    seen = dict(zip(results["race_id"], results["first_seen_utc"], strict=True))
    assert seen[pending] <= pd.Timestamp(cutoff) and seen[elsewhere] <= pd.Timestamp(cutoff)

    rec = record(model, at_snapshot, target, when)
    json.dumps(rec)
    assert rec["mode"] == "live"
    assert _pending(rec) == {(pending, seen[pending])}  # another group's gap is not listed
    assert pending not in used_ids(rec)
    without = results[results["race_id"] != pending]
    _, same_day = expected(model, CARD[target], without, cutoff)
    assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)

    pending_input = next(r for r in rec["inputs"]["results"] if r["race_id"] == pending)
    assert pending_input["top3_time_s"] is None
    assert verify([rec], model) == []

    hidden = copy.deepcopy(rec)
    hidden["results_pending"] = []
    assert any(target in p for p in verify([hidden], model))

    rebuilt = results.copy()  # every time filled in
    on_time = record(model, rebuilt, target, when)  # the time was there: the result is used
    assert pending in used_ids(on_time) and on_time["results_pending"] == []
    assert verify([on_time], model) == []
    moved = copy.deepcopy(on_time)
    entry = next(u for u in moved["results_used"] if u["race_id"] == pending)
    moved["results_used"].remove(entry)
    moved["results_pending"].append(
        {"race_id": entry["race_id"], "first_seen_utc": entry["first_seen_utc"]}
    )
    assert any(target in p for p in verify([moved], model))


# --------------------------------------------------------------------------- round 3


@pytest.mark.parametrize(
    "contradict",
    ["listed after the cutoff", "input seen after the cutoff", "input never seen", "not an input"],
)
def test_a_pending_sighting_its_inputs_do_not_allow_is_reported(model, results, contradict):
    """Round-3 finding 4, from the record alone: a live record lists TOK R3 as pending (seen by
    TOK R5's cutoff, no time yet) and verifies; a pending entry listed after the cutoff, or inputs
    that put that sighting after the cutoff, never see it or lack the race, is a problem naming
    TOK R5."""
    target, pending = rid("TOK", 5), rid("TOK", 3)
    cutoff = cutoff_of(target)
    at_snapshot = results.copy()
    at_snapshot.loc[at_snapshot["race_id"] == pending, "top3_time_s"] = np.nan
    rec = record(model, at_snapshot, target, cutoff - timedelta(seconds=30))
    seen = results.loc[results["race_id"] == pending, "first_seen_utc"].iloc[0]
    assert _pending(rec) == {(pending, seen)} and seen <= pd.Timestamp(cutoff)
    assert verify([rec], model) == []

    bad = copy.deepcopy(rec)
    late = (cutoff + timedelta(minutes=1)).isoformat()
    row = _input(bad, pending)
    if contradict == "listed after the cutoff":
        bad["results_pending"][0]["first_seen_utc"] = late
    elif contradict == "input seen after the cutoff":
        row["first_seen_utc"] = late
    elif contradict == "input never seen":
        row["first_seen_utc"] = None
    else:
        bad["inputs"]["results"].remove(row)
    problems = verify([bad], model)
    assert any(target in p for p in problems), problems


def test_an_earlier_result_missing_from_the_card_verifies_on_its_own_keys(model, results):
    """Round-4 finding rt:315: a standalone record_race given an earlier same-venue result that no
    card lists (TOK R9, a 1600 m turf race at 01:20, seen 01:32) uses it by its own keys; the
    record keeps those keys in its inputs and verifies from them, with no card consulted."""
    target, extra_id = rid("TOK", 5), rid("TOK", 9)
    assert extra_id not in CARD
    post = at(DAY, 1, 20)
    extra = {
        "race_id": extra_id,
        "venue": "TOK",
        "post_utc": post,
        **keys_of(T16),
        "top3_time_s": PAR[T16] + V_DAY[("TOK", "turf")] + 0.4,
        "first_seen_utc": post + timedelta(minutes=12),
    }
    source = pd.concat([results, pd.DataFrame([extra])], ignore_index=True)
    for col in ("post_utc", "first_seen_utc"):
        source[col] = pd.to_datetime(source[col], utc=True)
    cutoff = cutoff_of(target)
    rec = record(model, source, target, cutoff - timedelta(seconds=30))
    assert extra_id in used_ids(rec)
    _, same_day = expected(model, CARD[target], source, cutoff)
    assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)
    _, without = expected(model, CARD[target], results, cutoff)
    assert abs(same_day - without) > 1e-3  # TOK R9 moves the forecast
    entry = _input(rec, extra_id)
    assert {k: entry[k] for k in MODEL_KEYS} == keys_of(T16)
    assert verify([rec], model) == []
    assert verify(json.loads(json.dumps([rec])), model) == []


RAIL_CASES = (  # (race, rail publication vs its cutoff in s (None: no time), read vs cutoff, used)
    (rid("TOK", 3), -90, -60, True),  # published before the read
    (rid("TOK", 3), -60, -60, True),  # published at the read: at the bound
    (rid("TOK", 3), -30, 3 * 3600, True),  # a replay read: the cutoff bounds it
    (rid("TOK", 3), 0, 3 * 3600, True),  # published at the cutoff: at the bound
    (rid("KYO", 4), -600, -1, True),
    (rid("TOK", 3), -30, -60, False),  # after the read, before the cutoff
    (rid("TOK", 3), 1, 3 * 3600, False),  # published after the cutoff
    (rid("KYO", 4), None, -60, False),  # no publication time
    (rid("TOK", 1), 60, -60, False),  # post 10:00 JST, read 09:57, rail B published 09:59
)
RAIL_ARGS = ("race_id", "rail_vs_cutoff_s", "read_vs_cutoff_s", "used")
RAIL_IDS = [f"{r[-6:]} rail {p} read {q} {'used' if u else 'unknown'}" for r, p, q, u in RAIL_CASES]


def rail_case(race_id, rail_vs_cutoff_s, read_vs_cutoff_s) -> tuple[dict, datetime]:
    cutoff = cutoff_of(race_id)
    published = None if rail_vs_cutoff_s is None else cutoff + timedelta(seconds=rail_vs_cutoff_s)
    return {**CARD[race_id], "rail_available_utc": published}, cutoff + timedelta(
        seconds=read_vs_cutoff_s
    )


@pytest.mark.parametrize(RAIL_ARGS, RAIL_CASES, ids=RAIL_IDS)
def test_a_rail_counts_only_when_published_by_the_cutoff_and_the_read(
    model, results, race_id, rail_vs_cutoff_s, read_vs_cutoff_s, used
):
    """Plan section 2 and round-5 finding rt:249 (unknown unless proven, standalone): a turf rail
    published at or before min(cutoff, read_utc) gives rail B's par; one published later, or
    without a time, counts as "unknown", whose turf cell the fit never saw, so the race is
    skipped with a reason, never an ok (let alone live) record made with B."""
    race, when = rail_case(race_id, rail_vs_cutoff_s, read_vs_cutoff_s)
    rec = record(model, results, race_id, when, race=race, replay=read_vs_cutoff_s > 0)
    json.dumps(rec)
    if used:
        assert rec["status"] == "ok", rec
        par, same_day = expected(model, race, results, min(cutoff_of(race_id), when))
        assert rec["par_s"] == pytest.approx(par, abs=1e-9)
        assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)
        assert same_time(rec["inputs"]["rail"]["available_utc"], race["rail_available_utc"])
        assert verify([rec], model) == []
    else:
        assert rec["status"] == "skipped", rec
        assert isinstance(rec["reason"], str) and rec["reason"].strip()
        assert rec["par_s"] is None and rec["same_day_s"] is None
        assert rec["results_used"] == [] and rec["results_pending"] == []
    assert forecast(rec["inputs"], model)["status"] == rec["status"]


def unknown_rail(cell: tuple) -> tuple:
    return (*cell[:4], "unknown", cell[5])


UNKNOWN_RAIL_PAR = {unknown_rail(T16): 99.0, unknown_rail(T20): 123.9, unknown_rail(K16): 98.1}


@pytest.fixture(scope="module")
def unknown_rail_model() -> SpeedModel:
    """The planted fit plus a fitted par for each turf cell with its rail unknown, 2.6-3.4 s off
    the rail-B par."""
    rng = np.random.default_rng(4140)
    rows = [
        {
            "race_id": f"jra:{day.isoformat()}:{cell[0]}:{20 + n:02d}",
            "race_utc": at(day, 5, 10 * n),
            **keys_of(cell),
            "top3_time_s": par + rng.normal(0.0, 0.2),
            "result_available_utc": at(day, 5, 10 * n) + timedelta(minutes=30),
        }
        for day in TRAIN_DAYS
        for n, (cell, par) in enumerate(UNKNOWN_RAIL_PAR.items())
    ]
    return SpeedModel(SPEC).fit(pd.concat([fit_frame(), pd.DataFrame(rows)], ignore_index=True))


@pytest.mark.parametrize(RAIL_ARGS, RAIL_CASES, ids=RAIL_IDS)
def test_an_unproven_rail_forecasts_with_the_unknown_cell_when_it_has_a_par(
    unknown_rail_model, results, race_id, rail_vs_cutoff_s, read_vs_cutoff_s, used
):
    """Round-5 finding rt:249: with a fitted par for the turf cell of unknown rail, a rail
    published after min(cutoff, read_utc) (or without a time) is forecast as that cell, not as
    rail B; a rail published by the bound is forecast as B. Either honest record verifies from
    its own inputs, read back from JSON too."""
    m = unknown_rail_model
    race, when = rail_case(race_id, rail_vs_cutoff_s, read_vs_cutoff_s)
    rec = record(m, results, race_id, when, race=race, replay=read_vs_cutoff_s > 0)
    assert rec["status"] == "ok", rec
    as_seen = race if used else {**race, "rail": "unknown"}
    par, same_day = expected(m, as_seen, results, min(cutoff_of(race_id), when))
    other = m.par(**{**{k: race[k] for k in PAR_KEYS}, "rail": "unknown" if used else "B"})
    assert abs(par - other) > 2.0  # the two cells' pars are far apart
    assert rec["par_s"] == pytest.approx(par, abs=1e-9)
    assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)
    assert forecast(rec["inputs"], m)["par_s"] == pytest.approx(par, abs=1e-9)
    assert verify([rec], m) == []
    assert verify(json.loads(json.dumps([rec])), m) == []


@pytest.mark.parametrize(
    ("inputs_vs_now_s", "snapshot_is", "seen_counts"),
    [(-30, "inputs", False), (15, "now", True), (None, "now", True)],
    ids=["inputs read before now", "inputs after now", "no inputs time"],
)
def test_the_input_snapshot_is_the_earlier_of_the_read_and_now(
    model, results, inputs_vs_now_s, snapshot_is, seen_counts
):
    """Round-3 finding 1: `inputs_utc`, the instant the caller began reading the results, is the
    input snapshot when it comes before `now()`, so TOK R3, first seen 15 s before `now()` but
    after that read, does not feed TOK R5's forecast; a later or absent `inputs_utc` leaves the
    snapshot at `now()`, and TOK R3 is used. Each record verifies against the same source."""
    target, early = rid("TOK", 5), rid("TOK", 3)
    cutoff = cutoff_of(target)
    when = cutoff - timedelta(seconds=60)
    source = results.copy()
    source.loc[source["race_id"] == early, "first_seen_utc"] = pd.Timestamp(
        when - timedelta(seconds=15)
    )
    extra = {}
    if inputs_vs_now_s is not None:
        extra["inputs_utc"] = when + timedelta(seconds=inputs_vs_now_s)
    rec = record(model, source, target, when, **extra)
    snapshot = extra["inputs_utc"] if snapshot_is == "inputs" else when
    assert utc(rec["input_snapshot_utc"]) == snapshot
    assert utc(rec["completion_utc"]) == when
    assert rec["mode"] == "live"
    assert (early in used_ids(rec)) is seen_counts
    assert early not in {u["race_id"] for u in rec["results_pending"]}
    _, same_day = expected(model, CARD[target], source, min(cutoff, snapshot))
    assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)
    assert utc(rec["inputs"]["read_utc"]) == snapshot
    assert verify([rec], model) == []


# --------------------------------------------------------------------------- round 6

MIXED = {  # case -> (result, column, its new instant, used by TOK R5's replay at its cutoff)
    "a fractional sighting used": (rid("TOK", 3), "first_seen_utc", at(DAY, 2, 19), True),
    "a fractional late sighting": (rid("TOK", 6), "first_seen_utc", at(DAY, 4, 8), False),
    "0.25 s after the cutoff": (rid("TOK", 3), "first_seen_utc", None, False),
    "0.25 s before the cutoff": (rid("TOK", 3), "first_seen_utc", None, True),
    "a fractional post": (rid("TOK", 1), "post_utc", at(DAY, 1, 0), True),
}
FRACTION = {
    "a fractional sighting used": timedelta(milliseconds=250),
    "a fractional late sighting": timedelta(milliseconds=500),
    "0.25 s after the cutoff": timedelta(milliseconds=250),
    "0.25 s before the cutoff": timedelta(milliseconds=-250),
    "a fractional post": timedelta(milliseconds=500),
}


def mixed_source(results: pd.DataFrame, case: str) -> tuple[pd.DataFrame, str, str, datetime]:
    """The day's results with one instant moved by a fraction of a second; every other instant
    stays on a whole second, so the ISO texts mix both precisions."""
    target = rid("TOK", 5)
    race_id, column, base, _ = MIXED[case]
    base = cutoff_of(target) if base is None else base
    instant = base + FRACTION[case]
    assert instant.microsecond != 0
    source = results.copy()
    source.loc[source["race_id"] == race_id, column] = pd.Timestamp(instant)
    for col in ("post_utc", "first_seen_utc"):
        source[col] = pd.to_datetime(source[col], utc=True)
        whole = source.loc[source["race_id"] != race_id, col].dropna()
        assert (whole.dt.microsecond == 0).all()  # the rest stay whole-second
    return source, race_id, column, instant


def as_text(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with its instants as ISO text (a snapshot read back from JSON)."""
    out = frame.copy()
    for col in ("post_utc", "first_seen_utc"):
        out[col] = [None if pd.isna(v) else v.isoformat() for v in frame[col]]
    return out


@pytest.mark.parametrize("text", [False, True], ids=["instants", "ISO text"])
@pytest.mark.parametrize("case", list(MIXED))
def test_mixed_whole_and_fractional_second_instants_still_give_every_record(
    model, results, case, text
):
    """Round-5 finding rt:221: results whose sightings or posts mix whole-second and
    fractional-second ISO text (as given, or after the inputs freeze them as text) give every
    record of the day and its coverage; the fractional instant keeps its sub-second value, so a
    sighting 0.25 s after TOK R5's cutoff is not used and one 0.25 s before it is. Each record
    verifies from its own inputs."""
    source, changed, column, instant = mixed_source(results, case)
    given = as_text(source) if text else source
    records = [record(model, given, r, REPLAY_AT, replay=True) for r in CARD]
    assert all(r["format"] == FORMAT for r in records)
    got = coverage(records, card_frame())
    for venue in ("TOK", "KYO"):
        assert (got[venue]["forecast"], got[venue]["skipped"]) == (6, 1), (venue, got)
    target = rid("TOK", 5)
    rec = by_id(records)[target]
    assert rec["status"] == "ok", rec
    _, same_day = expected(model, CARD[target], source, cutoff_of(target))
    assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9)
    assert (changed in used_ids(rec)) is MIXED[case][3]
    entry = _input(rec, changed)
    assert utc(entry[column]) == instant  # the sub-second value kept
    if column == "first_seen_utc" and MIXED[case][3]:
        listed = next(u for u in rec["results_used"] if u["race_id"] == changed)
        assert utc(listed["first_seen_utc"]) == instant
    assert forecast(rec["inputs"], model)["same_day_s"] == pytest.approx(same_day, abs=1e-9)
    assert verify(records, model) == []
    assert verify(json.loads(json.dumps(records)), model) == []


def test_every_new_record_carries_format_2(model, results):
    """Round-5 finding rt:455: ok and skipped records, live and replay, carry "format": 2."""
    live = [record(model, results, r, cutoff_of(r) - timedelta(seconds=20)) for r in FORECASTABLE]
    skipped = [
        record(model, results, rid("TOK", 7), at(DAY, 4, 0)),
        record(model, results, rid("KYO", 7), at(DAY, 3, 0)),
    ]
    records = replay_day(model, results) + live + skipped
    assert {r["status"] for r in skipped} == {"skipped"}
    assert [r["format"] for r in records] == [FORMAT] * len(records)
    assert all(json.loads(json.dumps(r))["format"] == FORMAT for r in records)


def legacy_of(records: list[dict]) -> list[dict]:
    """The rows the earlier recorder wrote for the same races: no format, no inputs."""
    old = copy.deepcopy(records)
    for rec in old:
        rec.pop("inputs", None)
        rec.pop("format", None)
    return old


def test_rows_of_the_earlier_format_are_reported_as_legacy_not_as_problems(model, results):
    """Round-5 finding rt:455: a rehearsal log from the earlier recorder (rows without format or
    inputs) followed by a replay appended after the upgrade: the earlier ok rows are listed as
    legacy, in log order, and are no problem; the new rows verify as before. A log of legacy
    rows alone is not rejected either."""
    old = legacy_of(replay_day(model, results))
    new = replay_day(model, results)
    old_ok = [r["race_id"] for r in old if r["status"] == "ok"]
    assert len(old_ok) == len(FORECASTABLE)
    for log in (old + new, json.loads(json.dumps(old + new)), old):
        got = verify_replay(log, model=model, model_version=MV)
        assert list(got.problems) == [], got.problems
        assert list(got.legacy) == old_ok
    got = verify_replay(new, model=model, model_version=MV)
    assert list(got.problems) == [] and list(got.legacy) == []


@pytest.mark.parametrize(
    "tamper", [_bump_same_day, _publish_the_rail_after_the_cutoff, _drop_the_inputs]
)
def test_a_mismatched_new_row_still_fails_beside_legacy_rows(model, results, tamper):
    """Round-5 finding rt:455: beside legacy rows of the same races, a changed new row is still a
    problem naming its race (a format-2 row without inputs included: it is not legacy), the
    legacy rows stay legacy and are no problem."""
    old = legacy_of(replay_day(model, results))
    new = replay_day(model, results)
    race_id = tamper(new)
    got = verify_replay(old + new, model=model, model_version=MV)
    problems = list(got.problems)
    assert problems and all(race_id in p for p in problems), problems
    assert list(got.legacy) == [r["race_id"] for r in old if r["status"] == "ok"]
