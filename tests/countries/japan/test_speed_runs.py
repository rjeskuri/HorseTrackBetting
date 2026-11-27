"""The horse-runs frame and race cards for #97's `HorseSpeed`, from the store (#346, epic #73).

Contract pinned here (`horsetrack.countries.japan.models.speed_runs`):

- `build_runs(races, runners, *, decision_min) -> (frame, counts)`. `races` and `runners` are
  store-shaped (docs/specs/store.md): races race_id, meeting_id, race_date (the JST race day),
  venue, surface, distance_m, course_config, rail_setting, race_class, post_time_local ("H:MM" on
  the JST wall clock), field_size; runners race_id, race_date, horse_id, draw, actual_wt_kg,
  running_positions (a list, the first corner first; null when none), finish_pos (null when not
  placed), finish_time_s, is_scratched. No JRDB table (D21).
  - `frame`: one row per run (a non-scratched runner of a kept race) with race_id, race_utc
    (= post_utc: post_time_local on the Asia/Tokyo wall clock of race_date, tz-aware UTC),
    horse_id, finish_time_s (null when the runner has none: unplaced runners are kept), weight_kg
    (= actual_wt_kg, the exact carried kg the JRA adapters store; not converted from lb), draw,
    style, result_available_utc (= race_utc + 30 min, the owner's rule, docs/specs/as-of.md), and
    #305's par keys: track (venue), surface, distance (distance_m), layout (course_config), rail
    (rail_setting), class_level (race_class), a null key -> "unknown" (#305's convention).
  - style is #129's first-corner rule (`handicap.factors._style_code` / `_finish_style`): a start
    at first-corner position p in a field of n (races.field_size) is front if p == 1, stalker if
    3(p-1) <= n-1, midpack if 3(p-1) <= 2(n-1), else closer; a run's style is the most frequent
    over the horse's newest 5 starts that have a first-corner position, ties to the more forward
    style, null when there is none. The starts a run's style reads are the horse's runs in other
    races whose result_available_utc is strictly before that run's decision time,
    race_utc - `decision_min` minutes: never the run's own race, never a race published at or
    after its decision time (a same-day race published before it counts).
  - Jump races (surface "jump", owner on #96) are excluded; races on holdout days
    (`core.eval.holdout.is_holdout_day("japan", race_date)`) are never read: their runs are in
    no row and feed no style. Races without a post time are excluded. Scratched runners are no
    run.
  - `counts`, a dict: n_races (races in), n_holdout, n_jump, n_no_post_time (races excluded,
    each counted once), n_kept (races kept), n_scratched (runner rows dropped), n_runs
    (= len(frame)). In these fixtures a race has at most one reason to be excluded, so
    n_holdout + n_jump + n_no_post_time + n_kept == n_races.
- `build_card(races, runners, race_id, *, at_utc) -> DataFrame`: the target race's non-scratched
  runners with the frame's columns minus the result (no finish_time_s, no result_available_utc,
  no finish column at all). Each runner's style reads the horse's starts in other races whose
  result_available_utc is strictly before at_utc, never the target race.
- `read_inputs(*, start, end) -> (races, runners)`: the JRA store's races and runners with
  start <= race_date <= end, column-scoped (no price column), holdout days never returned.

A card asked at an at_utc after the target's result is published may refuse it (ValueError); if it
does not, the target is still not read. Not pinned: the order of exclusions when a race has two
reasons, a jump race's starts in a flat horse's style window, a card for a jump race, extra
columns. The
unseen-cell drop belongs to #97's `HorseSpeed.fit` (its `n_unseen`: a run whose cell has no par in
the fitted `SpeedModel`); a builder without the fitted model cannot know it, so it is not here.
Every frame is synthetic and built here; expected values come from the planted races (zoneinfo
instants, hand-written style tables, an independent reference of #129's rule), never from the code
under test.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
import random
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.models.speed_runs import build_card, build_runs, read_inputs

JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
RESULT_LAG = timedelta(minutes=30)  # the owner's rule (docs/specs/as-of.md)
STYLES = ("front", "stalker", "midpack", "closer")  # #129
PAR_KEYS = ("track", "surface", "distance", "layout", "rail", "class_level")
CARD_COLUMNS = ("race_id", "race_utc", "horse_id", "style", "weight_kg", "draw", *PAR_KEYS)
RESULT_COLUMNS = ("finish_time_s", "result_available_utc")
RUN_COLUMNS = (*CARD_COLUMNS, *RESULT_COLUMNS)
COUNT_KEYS = (
    "n_races",
    "n_holdout",
    "n_jump",
    "n_no_post_time",
    "n_kept",
    "n_scratched",
    "n_runs",
)
DEC = 2  # minutes before post (config/decision.yaml's primary); each test passes its own
LB_PER_KG = 2.20462262

# ---------------------------------------------------------------------------------------------
# Store-shaped frames
# ---------------------------------------------------------------------------------------------


def rid(day: date, venue: str, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{race_no:02d}"


def post_instant(day: date, hhmm: str) -> pd.Timestamp:
    """The post as a UTC instant, from first principles: the JST wall clock of the race day."""
    hh, mm = (int(x) for x in hhmm.split(":"))
    return pd.Timestamp(datetime.combine(day, time(hh, mm), tzinfo=JST).astimezone(UTC))


def run(horse, corner=None, pos=None, secs=None, kg=55.0, draw=None, scratched=False):
    """One runner: horse id, first-corner position (None: no corner positions), finish_pos and
    finish_time_s (None: not placed / no time), carried kg, draw, scratched."""
    return {
        "horse": horse,
        "corner": corner,
        "pos": pos,
        "secs": secs,
        "kg": kg,
        "draw": draw,
        "scratched": scratched,
    }


def make(specs):
    """Store-shaped races and runners frames. Each spec: day, race_no, and optionally venue,
    surface, distance, course, rail, cls, post (None: no post time), and `runners` (`run` dicts).
    field_size is the number of non-scratched runners."""
    races, runners = [], []
    for spec in specs:
        day, venue, no = spec["day"], spec.get("venue", "TOK"), spec["race_no"]
        race_id = rid(day, venue, no)
        field = [r for r in spec["runners"] if not r["scratched"]]
        races.append(
            {
                "jurisdiction": "jra",
                "race_id": race_id,
                "meeting_id": f"jra:{day.isoformat()}:{venue}",
                "race_date": day,
                "venue": venue,
                "race_no": no,
                "race_class": spec.get("cls", "C1"),
                "distance_m": spec.get("distance", 1600),
                "surface": spec.get("surface", "turf"),
                "course_config": spec.get("course", "outer"),
                "rail_setting": spec.get("rail"),
                "post_time_local": spec.get("post", f"{10 + (no - 1) % 7}:05"),
                "field_size": len(field),
                "status": "run",
            }
        )
        for horse_no, r in enumerate(spec["runners"], start=1):
            corner = None if r["corner"] is None else [r["corner"], r["corner"], r["corner"]]
            runners.append(
                {
                    "jurisdiction": "jra",
                    "race_id": race_id,
                    "meeting_id": f"jra:{day.isoformat()}:{venue}",
                    "race_date": day,
                    "venue": venue,
                    "race_no": no,
                    "horse_id": r["horse"],
                    "horse_no": horse_no,
                    "draw": r["draw"] if r["draw"] is not None else horse_no,
                    "actual_wt_kg": r["kg"],
                    "actual_wt_lb": round(r["kg"] * LB_PER_KG),
                    "running_positions": corner,
                    "finish_pos": r["pos"],
                    "finish_time_s": r["secs"],
                    "win_odds": 5.0,
                    "is_scratched": r["scratched"],
                }
            )
    races = pd.DataFrame(races)
    runners = pd.DataFrame(runners)
    races["field_size"] = races["field_size"].astype("Int64")
    runners["finish_pos"] = runners["finish_pos"].astype("Int64")
    runners["finish_time_s"] = runners["finish_time_s"].astype(float)
    runners["draw"] = runners["draw"].astype("Int64")
    runners["is_scratched"] = runners["is_scratched"].astype("boolean")
    # nothing may rely on the frames' row order
    return (
        races.sample(frac=1.0, random_state=346).reset_index(drop=True),
        runners.sample(frac=1.0, random_state=346).reset_index(drop=True),
    )


def field(horses, *, first=None, base=90.0):
    """Runners in finishing order 1..n with first-corner position = finishing position (or the
    `first` list of corners), 0.2 s apart."""
    corners = first or list(range(1, len(horses) + 1))
    return [
        run(h, corner=c, pos=k, secs=round(base + 0.2 * (k - 1), 1))
        for k, (h, c) in enumerate(zip(horses, corners, strict=True), start=1)
    ]


def row(frame, race_id, horse):
    hit = frame[(frame["race_id"] == race_id) & (frame["horse_id"] == horse)]
    assert len(hit) == 1, (race_id, horse, len(hit))
    return hit.iloc[0]


def style_of(frame, race_id, horse):
    value = row(frame, race_id, horse)["style"]
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def is_null(value) -> bool:
    return value is None or pd.isna(value)


def check_counts(counts):
    assert set(COUNT_KEYS) <= set(counts), sorted(counts)


# ---------------------------------------------------------------------------------------------
# The four named tests
# ---------------------------------------------------------------------------------------------

D0 = date(2026, 10, 3)  # history
D1 = date(2026, 10, 10)  # the target's day
D2 = date(2026, 10, 17)  # after the target


def style_history():
    """Horse H: a closer on D0; front in race 1 of D1 (post 10:05, published 10:35); front again
    in the target, race 3 of D1 (post 10:50); front three times after D1."""
    others = [f"o{k}" for k in range(1, 8)]
    specs = [
        {"day": D0, "race_no": 1, "post": "13:05", "runners": field([*others, "H"])},  # H 8th of 8
        {"day": D1, "race_no": 1, "post": "10:05", "runners": field(["H", *others])},
        {"day": D1, "race_no": 3, "post": "10:50", "runners": field(["H", *others])},
    ]
    for day in (D2, D2 + timedelta(days=1), D2 + timedelta(days=7)):
        specs.append({"day": day, "race_no": 2, "post": "11:05", "runners": field(["H", *others])})
    return make(specs)


def test_runs_style_uses_only_earlier_races():
    races, runners = style_history()
    target = rid(D1, "TOK", 3)
    # the target's decision time against D1 race 1's publication (10:35 JST): before it at T-16
    # and at T-15 (10:35, not strictly before), after it at T-14 and T-2
    expected = {16: "closer", 15: "closer", 14: "front", DEC: "front"}
    for decision_min, style in expected.items():
        frame, _ = build_runs(races, runners, decision_min=decision_min)
        # D0's closer alone, or D0's closer and D1 race 1's front (a tie: the forward style).
        # Reading the target itself or any later race would tip T-16 and T-15 to front.
        assert style_of(frame, target, "H") == style, decision_min

    frame, _ = build_runs(races, runners, decision_min=DEC)
    # the horse's first start has nothing before it
    assert style_of(frame, rid(D0, "TOK", 1), "H") is None
    # D1 race 1 at T-2 (10:03): only D0 published before it
    assert style_of(frame, rid(D1, "TOK", 1), "H") == "closer"
    # a later run reads the target and the races published before it: one closer, three fronts
    assert style_of(frame, rid(D2, "TOK", 2), "H") == "front"
    # o7 was 7th of 8 on D0 (3*6 = 18 > 2*7: closer) and 8th in D1 race 1 (closer)
    assert style_of(frame, target, "o7") == "closer"
    # o1 was 1st on D0 (front) and 2nd of 8 in D1 race 1 (3*1 <= 7: stalker): tie -> front
    assert style_of(frame, target, "o1") == "front"


def test_runs_result_available_is_post_plus_30_min():
    posts = [
        (date(2026, 10, 3), "TOK", 1, "9:50"),
        (date(2026, 10, 3), "TOK", 2, "10:20"),
        (date(2026, 10, 3), "KYO", 11, "15:45"),
        (date(2026, 10, 4), "NAK", 12, "16:25"),
        (date(2026, 7, 18), "SAP", 6, "23:55"),
        (date(2026, 1, 1), "TOK", 1, "0:10"),  # the UTC instant falls on the previous day
    ]
    specs = [
        {
            "day": day,
            "venue": venue,
            "race_no": no,
            "post": hhmm,
            "runners": field([f"{venue}{no}a", f"{venue}{no}b", f"{venue}{no}c"]),
        }
        for day, venue, no, hhmm in posts
    ]
    specs.append({"day": date(2026, 10, 4), "race_no": 3, "post": None, "runners": field(["x"])})
    races, runners = make(specs)
    frame, counts = build_runs(races, runners, decision_min=DEC)
    check_counts(counts)
    assert counts["n_no_post_time"] == 1
    assert rid(date(2026, 10, 4), "TOK", 3) not in set(frame["race_id"])
    assert set(frame["race_id"]) == {rid(d, v, n) for d, v, n, _ in posts}
    for day, venue, no, hhmm in posts:
        part = frame[frame["race_id"] == rid(day, venue, no)]
        assert len(part) == 3
        post = post_instant(day, hhmm)
        for _, r in part.iterrows():
            assert pd.Timestamp(r["race_utc"]).tzinfo is not None
            assert pd.Timestamp(r["race_utc"]) == post, (day, hhmm)
            assert pd.Timestamp(r["result_available_utc"]) == post + RESULT_LAG, (day, hhmm)


def test_runs_exclude_jump_races():
    specs = []
    jump_ids, flat_ids = set(), set()
    for k in range(6):
        day = date(2026, 10, 3) + timedelta(days=7 * (k // 3))
        surface = ("jump", "turf", "dirt")[k % 3]
        spec = {
            "day": day,
            "race_no": k + 1,
            "surface": surface,
            "distance": 3000 if surface == "jump" else 1200 + 200 * k,
            "runners": field([f"h{k}{j}" for j in range(5)]),
        }
        specs.append(spec)
        (jump_ids if surface == "jump" else flat_ids).add(rid(day, "TOK", k + 1))
    races, runners = make(specs)
    frame, counts = build_runs(races, runners, decision_min=DEC)
    check_counts(counts)
    assert counts["n_jump"] == len(jump_ids) == 2
    assert counts["n_races"] == 6
    assert counts["n_kept"] == len(flat_ids) == 4
    assert set(frame["race_id"]) == flat_ids
    assert not set(frame["race_id"]) & jump_ids
    assert "jump" not in set(frame["surface"])
    assert counts["n_runs"] == len(frame) == 5 * len(flat_ids)


def test_runs_keep_unplaced_as_null_time():
    day = date(2026, 10, 10)
    specs = [
        {
            "day": day,
            "race_no": 1,
            "runners": [
                run("a", corner=2, pos=1, secs=95.3),
                run("b", corner=1, pos=2, secs=95.4),
                run("c", corner=3, pos=3, secs=95.9),
                run("d", corner=4, pos=None, secs=None),  # pulled up: not placed, no time
                run("e", corner=None, pos=None, secs=None),  # fell before the first corner
                run("f", corner=5, pos=4, secs=None),  # placed, time missing
                run("g", scratched=True),  # withdrawn: no run
            ],
        },
        {
            "day": day,
            "race_no": 2,
            "runners": [
                run("p", corner=1, pos=1, secs=71.0),
                run("q", corner=2, pos=None, secs=None),
            ],
        },
    ]
    races, runners = make(specs)
    frame, counts = build_runs(races, runners, decision_min=DEC)
    check_counts(counts)
    r1, r2 = rid(day, "TOK", 1), rid(day, "TOK", 2)
    assert set(zip(frame["race_id"], frame["horse_id"], strict=True)) == {
        *((r1, h) for h in "abcdef"),
        *((r2, h) for h in "pq"),
    }
    expected = {
        (r1, "a"): 95.3,
        (r1, "b"): 95.4,
        (r1, "c"): 95.9,
        (r1, "d"): None,
        (r1, "e"): None,
        (r1, "f"): None,
        (r2, "p"): 71.0,
        (r2, "q"): None,
    }
    for (race_id, horse), secs in expected.items():
        got = row(frame, race_id, horse)["finish_time_s"]
        if secs is None:
            assert is_null(got), (race_id, horse, got)
        else:
            assert got == pytest.approx(secs, abs=1e-9), (race_id, horse)
    assert counts["n_scratched"] == 1
    assert counts["n_runs"] == len(frame) == 8


# ---------------------------------------------------------------------------------------------
# The frame's columns and values
# ---------------------------------------------------------------------------------------------


def test_runs_carry_weight_kg_draw_and_the_par_keys_of_each_race():
    day = date(2026, 10, 11)
    cells = [
        # venue, surface, distance, course, rail, class
        ("TOK", "turf", 1600, "outer", "C", "G3"),
        ("NAK", "dirt", 1800, "inner", None, "C2"),
        ("KYO", "turf", 2400, None, "A", "OP"),
    ]
    kgs = [[54.0, 55.5, 57.0, 58.5], [52.0, 56.0, 53.5, 57.5], [55.0, 54.5, 58.0, 56.5]]
    draws = [[3, 1, 4, 2], [2, 4, 1, 3], [4, 3, 2, 1]]
    specs = []
    for k, (venue, surface, dist, course, rail, cls) in enumerate(cells):
        specs.append(
            {
                "day": day,
                "venue": venue,
                "race_no": k + 1,
                "surface": surface,
                "distance": dist,
                "course": course,
                "rail": rail,
                "cls": cls,
                "runners": [
                    run(f"{venue}{j}", corner=j + 1, pos=j + 1, secs=100.0 + j, kg=kg, draw=dr)
                    for j, (kg, dr) in enumerate(zip(kgs[k], draws[k], strict=True))
                ],
            }
        )
    races, runners = make(specs)
    frame, counts = build_runs(races, runners, decision_min=DEC)
    assert set(RUN_COLUMNS) <= set(frame.columns), sorted(set(RUN_COLUMNS) - set(frame.columns))
    check_counts(counts)
    assert counts["n_races"] == counts["n_kept"] == 3
    assert counts["n_holdout"] == counts["n_jump"] == counts["n_no_post_time"] == 0
    for k, (venue, surface, dist, course, rail, cls) in enumerate(cells):
        race_id = rid(day, venue, k + 1)
        for j in range(4):
            r = row(frame, race_id, f"{venue}{j}")
            # the exact carried kg; a conversion from the rounded lb column would miss it
            assert r["weight_kg"] == pytest.approx(kgs[k][j], abs=1e-9)
            assert int(r["draw"]) == draws[k][j]
            assert r["track"] == venue
            assert r["surface"] == surface
            assert int(r["distance"]) == dist
            assert r["layout"] == (course if course is not None else "unknown")
            assert r["rail"] == (rail if rail is not None else "unknown")
            assert r["class_level"] == cls


@pytest.mark.parametrize(
    ("n", "table"),
    [
        # first-corner position -> style, by hand from #129's rule
        (8, {1: "front", 2: "stalker", 3: "stalker", 4: "midpack", 5: "midpack", 6: "closer"}),
        (
            16,
            {
                1: "front",
                2: "stalker",
                6: "stalker",
                7: "midpack",
                11: "midpack",
                12: "closer",
                16: "closer",
            },
        ),
    ],
)
def test_runs_style_follows_the_first_corner_rule_by_field_size(n, table):
    horses = [f"h{k}" for k in range(1, n + 1)]
    corners = list(range(n, 0, -1))  # first-corner order is not the finishing order
    history = {"day": D0, "race_no": 1, "runners": field(horses, first=corners)}
    later = {"day": D1, "race_no": 1, "runners": field(horses)}
    races, runners = make([history, later])
    frame, _ = build_runs(races, runners, decision_min=DEC)
    by_corner = dict(zip(corners, horses, strict=True))
    for p, style in table.items():
        assert style_of(frame, rid(D1, "TOK", 1), by_corner[p]) == style, (n, p)


def test_runs_style_is_the_newest_five_starts_with_a_corner():
    """Seven earlier starts, oldest first: midpack, closer, closer, closer, midpack, midpack, then
    one with no corner position. The newest five with a corner (midpack, midpack, closer, closer,
    closer) give closer. A window of the newest five starts that then drops the corner-less one
    reads four (a tie: midpack); all six with a corner give midpack 3, closer 3: midpack."""
    others = [f"o{k}" for k in range(1, 8)]
    places = [5, 8, 8, 8, 5, 5, None]  # in a field of 8: 5th is midpack, 8th closer
    specs = []
    for k, p in enumerate(places):
        day = date(2026, 9, 5) + timedelta(days=7 * k)
        if p is None:
            runners = [*field(others), run("H", corner=None, pos=8, secs=99.0)]
        else:
            runners = field(others[: p - 1] + ["H"] + others[p - 1 :])
        specs.append({"day": day, "race_no": 1, "runners": runners})
    target_day = date(2026, 9, 5) + timedelta(days=7 * len(places))
    specs.append({"day": target_day, "race_no": 1, "runners": field(["H", *others])})
    races, runners = make(specs)
    frame, _ = build_runs(races, runners, decision_min=DEC)
    assert style_of(frame, rid(target_day, "TOK", 1), "H") == "closer"


def reference_style(starts, horse, own_race, cutoff):
    """#129's rule from first principles: the horse's starts in other races published strictly
    before `cutoff`, with a corner, newest 5, most frequent, ties forward."""
    seen = [
        s
        for s in starts
        if s["horse"] == horse
        and s["race_id"] != own_race
        and s["available"] < cutoff
        and s["corner"] is not None
    ]
    seen.sort(key=lambda s: s["post"], reverse=True)
    codes = []
    for s in seen[:5]:
        p, n = s["corner"], s["n"]
        if p == 1:
            codes.append(0)
        elif 3 * (p - 1) <= n - 1:
            codes.append(1)
        elif 3 * (p - 1) <= 2 * (n - 1):
            codes.append(2)
        else:
            codes.append(3)
    if not codes:
        return None
    tally = Counter(codes)
    best = max(tally.values())
    return STYLES[min(k for k, v in tally.items() if v == best)]


def test_runs_style_matches_the_rule_over_a_random_calendar():
    rng = random.Random(346)
    horses = [f"h{k:02d}" for k in range(30)]
    specs, starts = [], []
    for d in range(6):
        day = date(2026, 9, 5) + timedelta(days=d)
        for no in range(1, 8):
            hh, mm = divmod(10 * 60 + 5 + 25 * (no - 1), 60)  # races 25 min apart
            hhmm = f"{hh}:{mm:02d}"
            post = post_instant(day, hhmm)
            n = rng.randint(5, 14)
            chosen = rng.sample(horses, n)
            corners = list(range(1, n + 1))
            rng.shuffle(corners)
            runners = []
            for k, (h, c) in enumerate(zip(chosen, corners, strict=True), start=1):
                corner = None if rng.random() < 0.15 else c
                runners.append(run(h, corner=corner, pos=k, secs=90.0 + 0.1 * k))
                starts.append(
                    {
                        "horse": h,
                        "race_id": rid(day, "TOK", no),
                        "post": post,
                        "available": post + RESULT_LAG,
                        "corner": corner,
                        "n": n,
                    }
                )
            specs.append({"day": day, "race_no": no, "post": hhmm, "runners": runners})
    races, runners = make(specs)
    for decision_min in (2, 16, 60):
        frame, counts = build_runs(races, runners, decision_min=decision_min)
        assert len(frame) == len(starts)
        lag = timedelta(minutes=decision_min)
        styled = 0
        for s in starts:
            want = reference_style(starts, s["horse"], s["race_id"], s["post"] - lag)
            assert style_of(frame, s["race_id"], s["horse"]) == want, (s, decision_min)
            styled += want is not None
        assert styled > len(starts) // 2  # the calendar exercises the rule, not only nulls


# ---------------------------------------------------------------------------------------------
# Drops: every one counted, holdout days never read
# ---------------------------------------------------------------------------------------------


def dev_and_held_days():
    """The first holdout day on or after the holdout start, and the first development day after
    it, from the committed definition (holdout.yaml)."""
    day = date(2026, 11, 28)
    while not is_holdout_day("japan", day):
        day += timedelta(days=1)
    held = day
    while is_holdout_day("japan", day):
        day += timedelta(days=1)
    return held, day


def test_runs_never_read_holdout_days_and_count_every_drop():
    held, after = dev_and_held_days()
    before = date(2026, 11, 21)
    assert not is_holdout_day("japan", before) and not is_holdout_day("japan", after)
    others = [f"o{k}" for k in range(1, 8)]
    specs = [
        # H is a closer on a development day, front twice on the holdout day
        {"day": before, "race_no": 1, "runners": field([*others, "H"])},
        {"day": held, "race_no": 1, "runners": field(["H", *others])},
        {"day": held, "race_no": 5, "post": "15:05", "runners": field(["H", *others])},
        {"day": after, "race_no": 1, "runners": field(["H", *others])},
        {"day": after, "race_no": 2, "surface": "jump", "runners": field(["J", *others])},
        {"day": after, "race_no": 3, "post": None, "runners": field(["N", *others])},
        {
            "day": after,
            "race_no": 4,
            "post": "14:05",
            "runners": [*field(["S1", *others]), run("S2", scratched=True)],
        },
    ]
    races, runners = make(specs)
    frame, counts = build_runs(races, runners, decision_min=DEC)
    check_counts(counts)
    days = {pd.Timestamp(r).tz_convert(JST).date() for r in frame["race_utc"]}
    assert held not in days
    assert not {rid(held, "TOK", 1), rid(held, "TOK", 5)} & set(frame["race_id"])
    # the holdout day's fronts feed no style: H is the closer it was on the development day
    assert style_of(frame, rid(after, "TOK", 1), "H") == "closer"
    assert counts["n_races"] == 7
    assert counts["n_holdout"] == 2
    assert counts["n_jump"] == 1
    assert counts["n_no_post_time"] == 1
    assert counts["n_kept"] == 3
    assert counts["n_scratched"] == 1
    assert counts["n_runs"] == len(frame) == 3 * 8
    total = sum(counts[k] for k in ("n_holdout", "n_jump", "n_no_post_time", "n_kept"))
    assert total == counts["n_races"]
    assert "S2" not in set(frame["horse_id"])


# ---------------------------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------------------------


def test_card_has_the_run_columns_minus_the_result():
    races, runners = style_history()
    target = rid(D1, "TOK", 3)
    at = post_instant(D1, "10:48")
    card = build_card(races, runners, target, at_utc=at)
    assert set(CARD_COLUMNS) <= set(card.columns), sorted(set(CARD_COLUMNS) - set(card.columns))
    for column in RESULT_COLUMNS:
        assert column not in card.columns
    assert not [c for c in card.columns if "finish" in c or "running_positions" in c]
    assert set(card["race_id"]) == {target}
    assert sorted(card["horse_id"]) == sorted(["H", *(f"o{k}" for k in range(1, 8))])
    for _, r in card.iterrows():
        assert pd.Timestamp(r["race_utc"]) == post_instant(D1, "10:50")
        assert r["track"] == "TOK" and r["surface"] == "turf" and int(r["distance"]) == 1600
        assert r["weight_kg"] == pytest.approx(55.0)


def test_card_style_uses_only_results_published_before_at_utc():
    races, runners = style_history()
    target = rid(D1, "TOK", 3)
    # D1 race 1 is published at 10:35 JST: strictly before at_utc counts, at it does not
    cases = {
        "10:20": "closer",
        "10:35": "closer",
        "10:36": "front",
        "10:49": "front",
    }
    for hhmm, style in cases.items():
        card = build_card(races, runners, target, at_utc=post_instant(D1, hhmm))
        got = card.set_index("horse_id")["style"]
        assert got["H"] == style, hhmm
    # the target's own front and the later fronts are never read: at 10:20 H is D0's closer
    # only; o7 (7th of 8 on D0, 8th of 8 in race 1) is a closer either way
    card = build_card(races, runners, target, at_utc=post_instant(D1, "10:20"))
    assert card.set_index("horse_id")["style"]["o7"] == "closer"


def test_card_style_never_reads_the_target_race():
    """H is a closer on D0 and front in the target. A card asked after the target's result is
    published either refuses the decision time (ValueError: after the off) or still reads only
    D0: reading the target would tie closer with front and tip H to front."""
    others = [f"o{k}" for k in range(1, 8)]
    specs = [
        {"day": D0, "race_no": 1, "runners": field([*others, "H"])},
        {"day": D1, "race_no": 3, "post": "10:50", "runners": field(["H", *others])},
    ]
    races, runners = make(specs)
    target = rid(D1, "TOK", 3)
    before = build_card(races, runners, target, at_utc=post_instant(D1, "10:48"))
    assert before.set_index("horse_id")["style"]["H"] == "closer"
    try:
        late = build_card(races, runners, target, at_utc=post_instant(D1, "11:30"))
    except ValueError:
        return  # refusing a decision time after the off never reads the target either
    assert late.set_index("horse_id")["style"]["H"] == "closer"


def test_card_drops_scratched_runners_and_takes_weight_and_draw_from_the_target():
    others = [f"o{k}" for k in range(1, 5)]
    history = {"day": D0, "race_no": 1, "runners": field(["H", *others])}
    target_runners = [
        run("H", kg=57.5, draw=4),
        run("o1", kg=54.0, draw=1),
        run("o2", kg=56.5, draw=3),
        run("o3", kg=55.0, draw=2, scratched=True),
        run("o4", kg=53.0, draw=5),
    ]
    target = {"day": D1, "race_no": 2, "post": "11:30", "runners": target_runners}
    races, runners = make([history, target])
    card = build_card(races, runners, rid(D1, "TOK", 2), at_utc=post_instant(D1, "11:28"))
    by = card.set_index("horse_id")
    assert set(by.index) == {"H", "o1", "o2", "o4"}
    for r in target_runners:
        if r["scratched"]:
            continue
        assert by.loc[r["horse"], "weight_kg"] == pytest.approx(r["kg"], abs=1e-9)
        assert int(by.loc[r["horse"], "draw"]) == r["draw"]
    assert by.loc["H", "style"] == "front"  # 1st of 5 on D0
    assert by.loc["o4", "style"] == "closer"  # 5th of 5: 3*4 = 12 > 2*4


# ---------------------------------------------------------------------------------------------
# The store read: column- and date-scoped, holdout days never returned
# ---------------------------------------------------------------------------------------------


def write_store(tmp_path, monkeypatch, races: pd.DataFrame, runners: pd.DataFrame) -> None:
    """The two tables as the store writes them, under a fresh data directory with no JRDB table."""
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    jra = data / "processed" / "jra"
    jra.mkdir(parents=True)
    for name, frame in (("races", races), ("runners", runners)):
        spec = schemas.TABLES[name].columns
        schema = pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in spec.items()])
        frame = frame[[c for c in frame.columns if c in spec]]
        records = [
            {k: (None if not isinstance(v, list) and pd.isna(v) else v) for k, v in r.items()}
            for r in frame.astype(object).to_dict("records")
        ]
        pq.write_table(pa.Table.from_pylist(records, schema=schema), jra / f"{name}.parquet")
    assert not any("jrdb" in p.name for p in data.rglob("*"))


def test_read_inputs_scopes_by_date_and_columns_and_never_returns_a_holdout_day(
    tmp_path, monkeypatch
):
    held, after = dev_and_held_days()
    early, start = date(2026, 11, 1), date(2026, 11, 21)
    others = [f"o{k}" for k in range(1, 6)]
    specs = [
        {"day": early, "race_no": 1, "runners": field(["H", *others])},  # before start: front
        {"day": start, "race_no": 1, "runners": field([*others, "H"])},  # closer
        {"day": held, "race_no": 1, "runners": field(["H", *others])},  # holdout: front
        {"day": after, "race_no": 1, "runners": field(["H", *others])},
        {"day": after + timedelta(days=7), "race_no": 1, "runners": field(["H", *others])},
    ]
    races, runners = make(specs)
    write_store(tmp_path, monkeypatch, races, runners)

    got_races, got_runners = read_inputs(start=start, end=after)
    want_days = {start, after}
    for frame in (got_races, got_runners):
        assert {pd.Timestamp(d).date() for d in frame["race_date"]} == want_days
    assert "win_odds" not in got_runners.columns  # no price column is read

    frame, counts = build_runs(got_races, got_runners, decision_min=DEC)
    assert counts["n_holdout"] == 0  # the reader already left the holdout day out
    assert set(frame["race_id"]) == {rid(start, "TOK", 1), rid(after, "TOK", 1)}
    # the early front is out of range and the holdout front is never read: H is a closer
    assert style_of(frame, rid(after, "TOK", 1), "H") == "closer"
    assert row(frame, rid(start, "TOK", 1), "H")["finish_time_s"] == pytest.approx(91.0)
