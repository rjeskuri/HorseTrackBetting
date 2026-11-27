"""The race-time recorder's day entry point (ticket #414, review finding 1).

Contract pinned here (`horsetrack.countries.japan.shadow.race_time_day`; the coder follows it):

- `run_race_time_day(race_date, *, model, model_version, races, now, sleep, replay=False,
  day_dir=None) -> RaceTimeRun`: records every race of the day once through
  `race_time.record_race` and appends each record, as it is made, to the day's log.
  - `model`: a fitted #305 `SpeedModel` holding no observations; `model_version` its label.
  - `races()`: re-readable; a DataFrame of the day's card, one row per race: race_id
    (`jra:<YYYY-MM-DD>:<VENUE>:<NN>`), venue, the model's par keys, top3_time_s (NaN until the
    race has a result) and optionally post_utc (the card's own post, aware UTC, NaT when none).
    It carries no sighting.
  - Post times come from each venue's `<VENUE>.races.json` sidecar, first sightings (#225) from
    its `<VENUE>.results.json` sidecar, both in `board.snapshot.jra_day_dir(race_date, day_dir)`
    (the odds logger's layout). Round-2 finding C: a race the sidecar lacks (or a venue without
    a sidecar yet) takes the card's post_utc from `races()`; the sidecar wins where it has the
    race, and a sidecar written after the run started is used from the next pass on. A race with
    no post in either is recorded as skipped (no post).
  - Prospective (replay=False): each race with a post is recorded at most 5 minutes before its
    input cutoff (post - primary) and not after it when the run starts in time, waiting with
    `sleep(seconds)` between races; a race whose cutoff has already passed is recorded at once
    (late or replay by the clock). `latest_post` re-reads the races sidecar after the
    computation. The run returns once every race of the card has its record.
  - replay=True: every race is recorded at once with replay=True; it never waits (no `sleep`).
  - card_version: chosen by the runner from the card inputs (a non-empty text, the same for an
    unchanged card). Round-2 finding F: every card version a run records under is saved, and
    `load_card_snapshots(race_date) -> dict[str, DataFrame]` gives each version's card (race_id,
    venue, post_utc aware UTC or NaT, the par keys; one row per race), kept under
    `paths.data_dir()` apart from the log, a rerun adding its versions to the earlier ones. A
    record verifies against the snapshot its card_version names.
  - RaceTimeRun: records (this run's records, one per race of the card), coverage
    (`race_time.coverage` of this run's records against the latest card, per venue and "all"),
    path (the log file).
- `race_time_log_path(race_date) -> Path`: the day's log under `paths.data_dir()` (read on every
  call), one JSON object per line, append-only: a rerun adds its records after the earlier ones.
- CLI: `horsetrack japan race-time --date YYYY-MM-DD [--replay] [--day-dir PATH]` on the japan
  group. Module attributes looked up at call time: `utc_now()` (the clock), `sleep(seconds)`,
  `load_speed_model(race_date) -> (SpeedModel, model_version)` and `load_day_races(race_date) ->
  DataFrame` (the `races()` frame). It prints the coverage, one line per venue naming the venue
  and its counts; exit 0.
- Round-2 finding B, `load_day_races(race_date)`: the scheduled races come from the upcoming card,
  `paths.processed_dir("jra") / "upcoming" / "jra_<YYYY-MM-DD>_<VENUE>" / "races.parquet"` (every
  venue folder of the date; the card has meeting_id, race_no, surface, distance_m, course_config,
  race_class and post_time_local as ISO +09:00, and no venue, race_date or rail_setting), with the
  model's keys from the card (venue from meeting_id; a missing value or column is "unknown", as
  `studies.speed_run` fills it) and post_utc from post_time_local. Results are joined from the
  store's races and runners: top3_time_s, NaN until resulted. Every card race is in the frame
  whatever the store holds (nothing of the day, some races or all); a store race of the day that
  no card lists is added with the store's keys.
- Round-2 finding E, `load_speed_model(race_date)`: the version derives from the model spec and the
  training rows it is fitted on (the races before race_date), so rows of race_date or later added
  to the store leave it unchanged, and a changed training row changes it.
- D14 (holdout days fully locked, outcomes and odds paths alike): on a Japan holdout day
  (`core.eval.holdout.is_holdout_day("japan", d)`) `run_race_time_day` (prospective and replay)
  and `load_day_races` raise an exception whose message names the date and the holdout, before
  any read: no `races()` call, no sidecar read, no store table read, no log written. The command
  exits non-zero with that message and writes no log. Development days (also those on or after
  the holdout start) run as before.

Every race, fit, sidecar and clock is synthetic; the fake sleep only moves the fake clock.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import copy
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.snapshot import load_decision_grid
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.shadow import race_time_day
from horsetrack.countries.japan.shadow.race_time import verify_replay

UTC = timezone.utc
PAR_KEYS = ("track", "surface", "distance", "layout", "rail", "class_level")
SPEC = SpeedSpec(
    par_keys=PAR_KEYS, variant_keys=("track", "surface"), same_day_variant=True, t_df=None
)
PRIMARY = timedelta(minutes=load_decision_grid().primary)
MV = "speed-c3:fit-0414-day"
DAY = date(2025, 2, 15)
TRAIN_DAYS = tuple(date(2025, 1, 25) + timedelta(days=k) for k in range(6))
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

T16 = ("TOK", "turf", 1600, "outer", "B", 3)
T20 = ("TOK", "turf", 2000, "outer", "B", 3)
TD14 = ("TOK", "dirt", 1400, "outer", "unknown", 2)
K16 = ("KYO", "turf", 1600, "inner", "B", 3)
KD18 = ("KYO", "dirt", 1800, "outer", "unknown", 3)
UNSEEN = ("TOK", "turf", 1800, "outer", "B", 3)
PAR = {T16: 96.0, T20: 120.5, TD14: 85.2, K16: 95.4, KD18: 112.7}
V_DAY = {("TOK", "turf"): 0.9, ("TOK", "dirt"): -0.6, ("KYO", "turf"): 0.4, ("KYO", "dirt"): -1.1}


def at(hh: int, mm: int, ss: int = 0, day: date = DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=UTC)


# (venue, no, cell, post or None, first sighting or None, top3 time offset or None)
DAY_RACES = (
    ("TOK", 1, T16, at(1, 0), at(1, 12), 0.2),
    ("TOK", 2, TD14, at(1, 35), at(1, 47), -0.1),
    ("TOK", 3, T20, at(2, 10), at(2, 42, 50), 0.5),  # seen 10 s before TOK R4's cutoff
    ("TOK", 4, T16, at(2, 45), at(2, 58), 0.3),
    ("TOK", 5, UNSEEN, at(3, 20), None, None),  # a cell the fit never saw
    ("KYO", 1, KD18, at(1, 10), at(1, 21), -0.3),
    ("KYO", 2, K16, at(1, 45), at(1, 58), 0.2),
    ("KYO", 3, K16, None, None, None),  # no post time in the sidecar
)
VENUES = ("TOK", "KYO")
SCHEDULED = {"TOK": 5, "KYO": 3}


def rid(venue: str, no: int, day: date = DAY) -> str:
    return f"jra:{day.isoformat()}:{venue}:{no:02d}"


def keys_of(cell: tuple) -> dict:
    return dict(zip(PAR_KEYS, cell, strict=True))


def fit_frame(pars: dict = PAR) -> pd.DataFrame:
    rng = np.random.default_rng(4141)
    rows = []
    for day in TRAIN_DAYS:
        shift = {g: rng.normal(0.0, 0.6) for g in V_DAY}
        for n, cell in enumerate(pars):
            for rep in range(2):
                post = at(1 + n, 30 * rep, day=day)
                rows.append(
                    {
                        "race_id": f"jra:{day.isoformat()}:{cell[0]}:{10 * rep + n + 1:02d}",
                        "race_utc": post,
                        **keys_of(cell),
                        "top3_time_s": pars[cell] + shift[cell[:2]] + rng.normal(0.0, 0.25),
                        "result_available_utc": post + timedelta(minutes=30),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model() -> SpeedModel:
    return SpeedModel(SPEC).fit(fit_frame())


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def sidecar(venue: str, suffix: str) -> Path:
    return paths.odds_snapshot_path("jra", DAY.isoformat(), venue).with_name(f"{venue}{suffix}")


def write_sidecars() -> None:
    """What the odds logger leaves in the day folder: post times and first sightings."""
    for venue in VENUES:
        posts = {no: post for v, no, _, post, _, _ in DAY_RACES if v == venue and post}
        odds_log.write_post_times(sidecar(venue, ".races.json"), posts, at(0, 0))
        for v, no, _, _, seen, _ in DAY_RACES:
            if v == venue and seen is not None:
                odds_log.record_published(sidecar(venue, ".results.json"), [no], seen)


def store_races() -> pd.DataFrame:
    """The store's card for the day: no post, no sighting."""
    rows = []
    for venue, no, cell, _, _, eps in DAY_RACES:
        time = np.nan if eps is None or cell not in PAR else PAR[cell] + V_DAY[cell[:2]] + eps
        rows.append(
            {"race_id": rid(venue, no), "venue": venue, **keys_of(cell), "top3_time_s": time}
        )
    return pd.DataFrame(rows)


def card_frame() -> pd.DataFrame:
    rows = [
        {"race_id": rid(v, no), "venue": v, "post_utc": post, **keys_of(cell)}
        for v, no, cell, post, _, _ in DAY_RACES
    ]
    frame = pd.DataFrame(rows)
    frame["post_utc"] = pd.to_datetime(frame["post_utc"], utc=True)
    return frame


def source_results() -> pd.DataFrame:
    rows = [
        {
            "race_id": rid(v, no),
            "venue": v,
            "post_utc": post,
            **keys_of(cell),
            "top3_time_s": PAR[cell] + V_DAY[cell[:2]] + eps,
            "first_seen_utc": seen,
        }
        for v, no, cell, post, seen, eps in DAY_RACES
        if eps is not None
    ]
    frame = pd.DataFrame(rows)
    for col in ("post_utc", "first_seen_utc"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


POSTS = {rid(v, no): post for v, no, _, post, _, _ in DAY_RACES}
CELL = {rid(v, no): cell for v, no, cell, _, _, _ in DAY_RACES}


def expected(model: SpeedModel, race_id: str, as_of: datetime) -> tuple[float, float, set[str]]:
    """(par, par + variant, the same-group results used) from a separate copy of the #305 model
    that observes only the results first seen by `as_of`."""
    src = source_results()
    seen = src[src["first_seen_utc"] <= pd.Timestamp(as_of)]
    ref = copy.deepcopy(model)
    ref.observe(
        seen.rename(columns={"post_utc": "race_utc", "first_seen_utc": "result_available_utc"})
    )
    cell = CELL[race_id]
    target = {"race_id": race_id, "race_utc": POSTS[race_id], **keys_of(cell)}
    same = ref.expected_top3_time(target, at_utc=as_of + timedelta(days=30))
    group = seen[(seen["track"] == cell[0]) & (seen["surface"] == cell[1])]
    used = {r for r in group["race_id"] if r != race_id and CELL[r] in PAR}
    return model.par(**keys_of(cell)), same, used


class FakeTime:
    """A fake UTC clock; `sleep` moves it and nothing really waits. Each event (instant, action)
    runs once, when a sleep first takes the clock to or past its instant (a sidecar the odds
    logger writes meanwhile)."""

    def __init__(self, start: datetime, events=()) -> None:
        self.at = start
        self.events = sorted(events, key=lambda e: e[0])

    def now(self) -> datetime:
        return self.at

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0, seconds
        self.at += timedelta(seconds=seconds)
        assert self.at < at(12, 0), "the run waits past the day's end"
        while self.events and self.events[0][0] <= self.at:
            self.events.pop(0)[1]()


def no_sleep(seconds: float) -> None:
    raise AssertionError(f"a replay waited {seconds} s")


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def read_log() -> list[dict]:
    path = race_time_day.race_time_log_path(DAY)
    assert path.is_relative_to(paths.data_dir())
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(model, start: datetime, *, replay: bool = False):
    clock = FakeTime(start)
    write_sidecars()
    return race_time_day.run_race_time_day(
        DAY,
        model=model,
        model_version=MV,
        races=store_races,
        now=clock.now,
        sleep=no_sleep if replay else clock.sleep,
        replay=replay,
    )


def check_forecasts(model, records: list[dict]) -> None:
    for rec in records:
        if rec["status"] != "ok":
            continue
        race_id = rec["race_id"]
        cutoff = POSTS[race_id] - PRIMARY
        assert utc(rec["input_cutoff_utc"]) == cutoff
        as_of = min(cutoff, utc(rec["input_snapshot_utc"]))
        par, same_day, used = expected(model, race_id, as_of)
        assert rec["par_s"] == pytest.approx(par, abs=1e-9), race_id
        assert rec["same_day_s"] == pytest.approx(same_day, abs=1e-9), race_id
        assert {u["race_id"] for u in rec["results_used"]} == used, race_id


# --------------------------------------------------------------------------- prospective


@pytest.mark.parametrize(
    ("start", "want"),
    [
        (  # in time for every race
            at(0, 30),
            {
                "TOK": dict(forecast=4, skipped=1, missing=0, ready_by_cutoff=4, replay=0),
                "KYO": dict(forecast=2, skipped=1, missing=0, ready_by_cutoff=2, replay=0),
            },
        ),
        (  # started after four posts: those are recorded at once, the rest at their cutoffs
            at(2, 0),
            {
                "TOK": dict(forecast=4, skipped=1, missing=0, ready_by_cutoff=2, replay=2),
                "KYO": dict(forecast=2, skipped=1, missing=0, ready_by_cutoff=0, replay=2),
            },
        ),
    ],
    ids=["before the first cutoff", "after four posts"],
)
def test_a_prospective_day_records_every_race_at_its_cutoff(model, start, want):
    """Finding 1 (DoD 1 and 3): one record per scheduled race, appended to the day's log as JSON
    lines, each race with a post taken at its cutoff from the sidecars' post times and first
    sightings, and the coverage reported by venue with every scheduled race counted."""
    got = run(model, start)
    records = got.records
    assert sorted(r["race_id"] for r in records) == sorted(rid(v, n) for v, n, *_ in DAY_RACES)
    assert read_log() == json.loads(json.dumps(records))
    assert got.path == race_time_day.race_time_log_path(DAY)

    by_id = {r["race_id"]: r for r in records}
    assert by_id[rid("KYO", 3)]["status"] == "skipped"  # no post in the sidecar
    assert by_id[rid("TOK", 5)]["status"] == "skipped"  # unseen cell
    assert by_id[rid("TOK", 5)]["reason"] != by_id[rid("KYO", 3)]["reason"]
    for race_id, rec in by_id.items():
        if rec["status"] != "ok":
            continue
        assert rec["model_version"] == MV
        assert isinstance(rec["card_version"], str) and rec["card_version"]
        cutoff = POSTS[race_id] - PRIMARY
        snapshot, done = utc(rec["input_snapshot_utc"]), utc(rec["completion_utc"])
        assert start <= snapshot <= done
        if cutoff >= start + timedelta(minutes=5):
            assert cutoff - timedelta(minutes=5) <= snapshot <= cutoff, race_id
            assert rec["mode"] == "live", race_id
        elif cutoff < start:
            assert rec["mode"] != "live", race_id
    check_forecasts(model, records)

    for venue, counts in want.items():
        counts = {"scheduled": SCHEDULED[venue], **counts}
        assert {k: got.coverage[venue][k] for k in counts} == counts, venue
    assert got.coverage["all"]["scheduled"] == sum(SCHEDULED.values())

    versions = {r["card_version"] for r in records if r["status"] == "ok"}
    assert len(versions) == 1  # the card did not change during the run
    problems = verify_replay(
        read_log(),
        card=card_frame(),
        results=source_results(),
        model=model,
        model_version=MV,
        card_version=versions.pop(),
    )
    assert problems == []


# --------------------------------------------------------------------------- replay


def test_a_replay_records_only_replays_and_a_rerun_appends(model):
    """Finding 1 (DoD 1 and 3): a replay days later writes only replay records, as of each
    cutoff, never waits, and appends after the prospective run's records instead of replacing
    them."""
    live = run(model, at(0, 30))
    before = race_time_day.race_time_log_path(DAY).read_bytes()

    again = run(model, datetime(2025, 2, 18, 0, 0, tzinfo=UTC), replay=True)
    assert len(again.records) == len(DAY_RACES)
    assert {r["mode"] for r in again.records if r["status"] == "ok"} == {"replay"}
    check_forecasts(model, again.records)

    after = race_time_day.race_time_log_path(DAY).read_bytes()
    assert after.startswith(before) and len(after) > len(before)
    stored = read_log()
    assert stored == json.loads(json.dumps(live.records + again.records))
    for venue in VENUES:
        counts = again.coverage[venue]
        assert counts["scheduled"] == SCHEDULED[venue]
        assert counts["forecast"] == SCHEDULED[venue] - 1
        assert (counts["ready_by_cutoff"], counts["ready_by_post"]) == (0, 0)
        assert counts["replay"] == SCHEDULED[venue] - 1


# --------------------------------------------------------------------------- the command


def test_the_race_time_command_runs_a_day_and_a_replay(model, monkeypatch):
    """Finding 1: `horsetrack japan race-time` loads the model and the day's card through its
    seams, writes the day's log, prints the coverage by venue; `--replay` appends replays."""
    write_sidecars()
    clock = FakeTime(at(0, 30))
    monkeypatch.setattr(race_time_day, "utc_now", clock.now)
    monkeypatch.setattr(race_time_day, "sleep", clock.sleep)
    monkeypatch.setattr(race_time_day, "load_speed_model", lambda day: (model, MV))
    monkeypatch.setattr(race_time_day, "load_day_races", lambda day: store_races())
    from horsetrack.cli import app

    outputs = []
    for extra in ([], ["--replay"]):
        if extra:
            clock.at = datetime(2025, 2, 18, 0, 0, tzinfo=UTC)
        result = CliRunner().invoke(app, ["japan", "race-time", "--date", DAY.isoformat(), *extra])
        out = ANSI.sub("", result.output)
        assert result.exit_code == 0, out
        outputs.append(out)

    stored = read_log()
    assert len(stored) == 2 * len(DAY_RACES)
    first, second = stored[: len(DAY_RACES)], stored[len(DAY_RACES) :]
    assert {r["mode"] for r in first if r["status"] == "ok"} == {"live"}
    assert {r["mode"] for r in second if r["status"] == "ok"} == {"replay"}
    for out in outputs:
        for venue in VENUES:
            lines = [line for line in out.splitlines() if venue in line]
            assert any(re.search(rf"\b{SCHEDULED[venue]}\b", line) for line in lines), (venue, out)


# --------------------------------------------------------------------------- round 2: the card

JST = timezone(timedelta(hours=9))
# the upcoming card's vocabulary: the card has no rail, and a missing course_config is "unknown"
CT16 = ("TOK", "turf", 1600, "outer", "unknown", "C1")
CTD14 = ("TOK", "dirt", 1400, "unknown", "unknown", "MAIDEN")
CK18 = ("KYO", "dirt", 1800, "unknown", "unknown", "C2")
CK16 = ("KYO", "turf", 1600, "outer", "unknown", "C1")
CARD_PAR = {CT16: 95.1, CTD14: 85.9, CK18: 113.4, CK16: 95.8}
# (venue, no, cell, the card's post)
UPCOMING = (
    ("TOK", 1, CT16, at(1, 0)),
    ("TOK", 2, CTD14, at(1, 35)),
    ("TOK", 3, CT16, at(2, 10)),
    ("TOK", 4, CTD14, at(2, 45)),
    ("KYO", 1, CK18, at(1, 10)),
    ("KYO", 2, CK16, at(1, 45)),
    ("KYO", 3, CK18, at(2, 20)),
)
UPCOMING_CELL = {rid(v, no): cell for v, no, cell, _ in UPCOMING}


@pytest.fixture(scope="module")
def card_model() -> SpeedModel:
    return SpeedModel(SPEC).fit(fit_frame(CARD_PAR))


def write_upcoming(day: date, venues: tuple[str, ...] = VENUES) -> None:
    """The upcoming card as it lies in processed/jra/upcoming/jra_<date>_<VENUE>/: races.parquet
    (no venue, race_date or rail_setting; post_time_local ISO +09:00) and its runners file."""
    for venue in venues:
        rows = [
            {
                "jurisdiction": "jra",
                "race_id": rid(v, no, day),
                "meeting_id": f"jra:{day.isoformat()}:{v}",
                "race_no": no,
                "race_name": f"synthetic {v} {no}",
                "race_class": cell[5],
                "distance_m": cell[2],
                "surface": cell[1],
                "course_config": None if cell[3] == "unknown" else cell[3],
                "post_time_local": post.astimezone(JST).isoformat(),
                "field_size": 10,
                "status": "declared",
                "weight_type": "age",
                "kai": 1,
                "nichi": 3,
            }
            for v, no, cell, post in UPCOMING
            if v == venue
        ]
        folder = paths.processed_dir("jra") / "upcoming" / f"jra_{day.isoformat()}_{venue}"
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), folder / "races.parquet")
        entries = [
            {"race_id": r["race_id"], "horse_no": h, "horse_id": f"h{h:04d}"}
            for r in rows
            for h in (1, 2, 3)
        ]
        pq.write_table(pa.Table.from_pylist(entries), folder / "racecard_runners.parquet")


def write_store(races: list[dict], runners: list[dict]) -> None:
    """The store's JRA races and runners tables, every declared column (null where not given)."""
    for name, rows in (("races", races), ("runners", runners)):
        columns = schemas.TABLES[name].columns
        schema = pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in columns.items()])
        data = {c: [row.get(c) for row in rows] for c in columns}
        path = tables.table_path("jra", name)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(data, schema=schema), path)


def resulted(venue: str, no: int, cell: tuple, post: datetime) -> tuple[dict, list[dict]]:
    """A store race (built from its result page) and its runners, placed 1, 2, 3, 3 (a dead heat
    for third) and 4, the times off the cell's par by the day and the race number."""
    day = post.astimezone(JST).date()
    race_id = rid(venue, no, day)
    race = {
        "jurisdiction": "jra",
        "race_id": race_id,
        "meeting_id": f"jra:{day.isoformat()}:{venue}",
        "race_date": day,
        "venue": venue,
        "race_no": no,
        "surface": cell[1],
        "distance_m": cell[2],
        "course_config": None if cell[3] == "unknown" else cell[3],
        "rail_setting": None if cell[4] == "unknown" else cell[4],
        "race_class": cell[5],
        "post_time_local": post.astimezone(JST).strftime("%H:%M"),
        "status": "result",
    }
    base = CARD_PAR[cell] + 0.13 * (day.toordinal() % 5) + 0.07 * no
    places = ((1, 0.0), (2, 0.2), (3, 0.5), (3, 0.9), (4, 2.0))
    runners = [
        {
            "jurisdiction": "jra",
            "race_id": race_id,
            "race_date": day,
            "venue": venue,
            "race_no": no,
            "horse_id": f"h{k:04d}",
            "horse_no": k,
            "finish_pos": pos,
            "finish_time_s": base + dt,
        }
        for k, (pos, dt) in enumerate(places, 1)
    ]
    return race, runners


def top3_of(runners: list[dict]) -> float:
    """(1st + 2nd + the mean of the 3rds) / 3."""
    by_pos = {p: [r["finish_time_s"] for r in runners if r["finish_pos"] == p] for p in (1, 2, 3)}
    return (by_pos[1][0] + by_pos[2][0] + sum(by_pos[3]) / len(by_pos[3])) / 3


def a_week_before() -> tuple[dict, list[dict]]:
    return resulted("TOK", 1, CT16, at(1, 0) - timedelta(days=7))


@pytest.mark.parametrize(
    ("card_venues", "results_in"),
    [
        (VENUES, ()),
        (VENUES, (("TOK", 1), ("KYO", 1))),
        (VENUES, tuple((v, no) for v, no, _, _ in UPCOMING)),
        (("TOK",), (("KYO", 1), ("KYO", 2))),
    ],
    ids=["before the first result", "partial results", "every result", "a venue without a card"],
)
def test_load_day_races_schedules_the_upcoming_card_and_joins_the_results(card_venues, results_in):
    """Round-2 finding B: the day's races are the upcoming card's (keys and post from the card),
    with the store's results joined on: every card race is in the frame before the first result
    and with a partial results build, NaN until resulted; a store race of the day that no card
    lists is kept too, and another day's race never comes in."""
    write_upcoming(DAY, card_venues)
    races, runners, top3 = [], [], {}
    for v, no, cell, post in UPCOMING:
        if (v, no) in results_in:
            race, rows = resulted(v, no, cell, post)
            races.append(race)
            runners += rows
            top3[race["race_id"]] = top3_of(rows)
    other, other_rows = a_week_before()
    write_store([*races, other], runners + other_rows)

    got = race_time_day.load_day_races(DAY)
    on_card = {rid(v, no) for v, no, _, _ in UPCOMING if v in card_venues}
    want = on_card | {rid(v, no) for v, no in results_in}
    assert got["race_id"].is_unique
    assert set(got["race_id"]) == want
    rows = {r["race_id"]: r for r in got.to_dict("records")}
    for v, no, cell, post in UPCOMING:
        race_id = rid(v, no)
        if race_id not in want:
            continue
        row = rows[race_id]
        assert row["venue"] == v, race_id
        assert tuple(row[k] for k in PAR_KEYS) == cell, (race_id, row)
        if race_id in on_card:
            assert pd.Timestamp(row["post_utc"]) == pd.Timestamp(post), race_id
        if race_id in top3:
            assert row["top3_time_s"] == pytest.approx(top3[race_id], abs=1e-9), race_id
        else:
            assert np.isnan(row["top3_time_s"]), race_id


def test_the_command_runs_from_the_upcoming_card_before_any_result_or_sidecar(
    card_model, monkeypatch
):
    """Round-2 findings B and C: with the upcoming card in place, nothing of the day in the store
    and Kyoto's odds logger not yet started, `japan race-time` records every card race live at
    its cutoff: the sidecar's post where it has the race (Tokyo R2 moved later; Kyoto's sidecar,
    written at 00:50, moves Kyoto R1), else the card's (Tokyo R4, Kyoto R3)."""
    write_upcoming(DAY)
    other, other_rows = a_week_before()
    write_store([other], other_rows)
    tok = {1: at(1, 0), 2: at(1, 40), 3: at(2, 10)}
    kyo = {1: at(1, 13), 2: at(1, 45)}
    odds_log.write_post_times(sidecar("TOK", ".races.json"), tok, at(0, 0))

    def kyo_logger_starts() -> None:
        odds_log.write_post_times(sidecar("KYO", ".races.json"), kyo, at(0, 50))

    clock = FakeTime(at(0, 30), events=[(at(0, 50), kyo_logger_starts)])
    monkeypatch.setattr(race_time_day, "utc_now", clock.now)
    monkeypatch.setattr(race_time_day, "sleep", clock.sleep)
    monkeypatch.setattr(race_time_day, "load_speed_model", lambda day: (card_model, MV))
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "race-time", "--date", DAY.isoformat()])
    out = ANSI.sub("", result.output)
    assert result.exit_code == 0, out

    stored = read_log()
    assert sorted(r["race_id"] for r in stored) == sorted(UPCOMING_CELL)
    want_post = {rid(v, no): post for v, no, _, post in UPCOMING}  # the card's
    want_post |= {rid("TOK", n): p for n, p in tok.items()}
    want_post |= {rid("KYO", n): p for n, p in kyo.items()}
    for rec in stored:
        race_id = rec["race_id"]
        assert rec["status"] == "ok", rec
        assert utc(rec["post_utc_at_cutoff"]) == want_post[race_id], race_id
        assert utc(rec["latest_post_utc"]) == want_post[race_id], race_id
        assert rec["mode"] == "live", race_id
        cutoff = want_post[race_id] - PRIMARY
        assert cutoff - timedelta(minutes=5) <= utc(rec["completion_utc"]) <= cutoff, race_id
        par = card_model.par(**keys_of(UPCOMING_CELL[race_id]))
        assert rec["par_s"] == pytest.approx(par, abs=1e-9), race_id
        assert rec["same_day_s"] == pytest.approx(par, abs=1e-9), race_id  # nothing seen yet
        assert rec["results_used"] == [], race_id
    for venue, n in (("TOK", 4), ("KYO", 3)):
        lines = [line for line in out.splitlines() if venue in line]
        assert any(re.search(rf"\b{n}\b", line) for line in lines), (venue, out)


def test_each_card_version_is_saved_and_verifies_the_records_made_under_it(model):
    """Round-2 finding F: Kyoto R2's post moves at 01:05, after Tokyo R1's forecast; both card
    versions are saved, each with the posts it had, and every record verifies against the
    snapshot its card_version names (Tokyo R1 against the final card would not). A replay rerun
    keeps the earlier snapshots."""
    kyo_r2 = at(1, 49)

    def kyo_post_moves() -> None:
        posts = {1: POSTS[rid("KYO", 1)], 2: kyo_r2}
        odds_log.write_post_times(sidecar("KYO", ".races.json"), posts, at(1, 5))

    write_sidecars()
    clock = FakeTime(at(0, 30), events=[(at(1, 5), kyo_post_moves)])
    got = race_time_day.run_race_time_day(
        DAY, model=model, model_version=MV, races=store_races, now=clock.now, sleep=clock.sleep
    )
    ok = {r["race_id"]: r for r in got.records if r["status"] == "ok"}
    first, later = ok[rid("TOK", 1)]["card_version"], ok[rid("KYO", 1)]["card_version"]
    assert first != later
    assert {r["card_version"] for r in ok.values()} == {first, later}
    assert utc(ok[rid("KYO", 2)]["post_utc_at_cutoff"]) == kyo_r2

    snaps = race_time_day.load_card_snapshots(DAY)
    assert {first, later} <= set(snaps)
    for version, kyo_post in ((first, POSTS[rid("KYO", 2)]), (later, kyo_r2)):
        snap = snaps[version]
        assert sorted(snap["race_id"]) == sorted(POSTS)
        posts = dict(zip(snap["race_id"], pd.to_datetime(snap["post_utc"], utc=True), strict=True))
        assert posts[rid("KYO", 2)] == pd.Timestamp(kyo_post), version
        assert posts[rid("TOK", 1)] == pd.Timestamp(POSTS[rid("TOK", 1)]), version
        assert pd.isna(posts[rid("KYO", 3)]), version
        for row in snap.to_dict("records"):
            assert tuple(row[k] for k in PAR_KEYS) == CELL[row["race_id"]], (version, row)

    source = source_results()
    source.loc[source["race_id"] == rid("KYO", 2), "post_utc"] = pd.Timestamp(kyo_r2)
    log = read_log()
    for version in (first, later):
        mine = [r for r in log if r["card_version"] == version]
        assert mine, version
        kwargs = dict(results=source, model=model, model_version=MV, card_version=version)
        assert verify_replay(mine, card=snaps[version], **kwargs) == [], version
    stale = verify_replay(
        [ok[rid("TOK", 1)]],
        card=snaps[later],
        results=source,
        model=model,
        model_version=MV,
        card_version=later,
    )
    assert any(rid("TOK", 1) in p for p in stale)

    again = run(model, datetime(2025, 2, 18, 0, 0, tzinfo=UTC), replay=True)
    kept = race_time_day.load_card_snapshots(DAY)
    assert {first, later} | {r["card_version"] for r in again.records} <= set(kept)


def _training_store(days) -> tuple[list[dict], list[dict]]:
    races, runners = [], []
    for day in days:
        for v, no, cell, post in UPCOMING:
            when = post.replace(year=day.year, month=day.month, day=day.day)
            race, rows = resulted(v, no, cell, when)
            races.append(race)
            runners += rows
    return races, runners


def test_the_model_version_follows_the_training_rows_not_the_store_files():
    """Round-2 finding E: rows of the race day or later added to the store leave the fitted model
    and its version as they were; a changed or dropped training row changes the version."""
    train, train_runners = _training_store(TRAIN_DAYS)
    write_store(train, train_runners)
    fitted, version = race_time_day.load_speed_model(DAY)
    assert isinstance(version, str) and version
    pars = {cell: fitted.par(**keys_of(cell)) for cell in CARD_PAR}

    for later in ((DAY,), (DAY, DAY + timedelta(days=1)), (DAY + timedelta(days=7),)):
        more, more_runners = _training_store(later)
        write_store(train + more, train_runners + more_runners)
        again, label = race_time_day.load_speed_model(DAY)
        assert label == version, later
        assert {cell: again.par(**keys_of(cell)) for cell in CARD_PAR} == pytest.approx(pars)

    target = rid("TOK", 3, TRAIN_DAYS[2])

    def slower_winner(races, runners):
        for r in runners:
            if r["race_id"] == target and r["finish_pos"] == 1:
                r["finish_time_s"] += 0.5
        return races, runners

    def dropped_race(races, runners):
        return (
            [r for r in races if r["race_id"] != target],
            [r for r in runners if r["race_id"] != target],
        )

    day_races, day_runners = _training_store((DAY,))
    for change in (slower_winner, dropped_race):
        races, runners = change(copy.deepcopy(train), copy.deepcopy(train_runners))
        write_store(races + day_races, runners + day_runners)
        _, label = race_time_day.load_speed_model(DAY)
        assert label != version, change.__name__


# --------------------------------------------------------------------------- D14: the holdout

HOLDOUT_YAML = Path(__file__).parents[4] / "src/horsetrack/countries/japan/config/holdout.yaml"


def _first_days() -> tuple[date, date]:
    """(the first holdout day, the first development day) on or after the committed start, by
    the pre-registered rule (sha256 of "<seed>:<date>", even means holdout), not by the module."""
    import hashlib

    import yaml

    section = yaml.safe_load(HOLDOUT_YAML.read_text(encoding="utf-8"))["holdout"]
    start = section["start"]
    start = start if isinstance(start, date) else date.fromisoformat(str(start))
    held = dev = None
    for k in range(60):
        d = start + timedelta(days=k)
        digest = hashlib.sha256(f"{section['seed']}:{d.isoformat()}".encode()).hexdigest()
        if int(digest, 16) % 2 == 0:
            held = held or d
        else:
            dev = dev or d
    assert held is not None and dev is not None and DAY < start
    return held, dev


HOLD_DAY, DEV_AFTER_START = _first_days()


class Leak(AssertionError):
    """A read that a holdout day forbids (its text never says holdout)."""


def _says_holdout(text: str, day: date) -> bool:
    return day.isoformat() in text and "holdout" in text.lower()


def _tripwires(monkeypatch) -> list[str]:
    """Every sidecar and store read is recorded and raises Leak; the holdout day's results
    sidecars are directories, so any other way of opening them fails too."""
    calls: list[str] = []

    def tripwire(name):
        def read(*args, **kwargs):
            calls.append(name)
            raise Leak(f"{name} was read")

        return read

    for name in ("read_published", "read_unpublished", "read_post_times", "_read_results"):
        if hasattr(odds_log, name):
            monkeypatch.setattr(odds_log, name, tripwire(f"odds_log.{name}"))
    monkeypatch.setattr(tables, "read_table", tripwire("tables.read_table"))
    monkeypatch.setattr(race_time_day, "read_table", tripwire("read_table"), raising=False)
    # the upcoming card (round-2 finding B) is Parquet too: no Parquet read on a holdout day
    monkeypatch.setattr(pd, "read_parquet", tripwire("pd.read_parquet"))
    monkeypatch.setattr(pq, "read_table", tripwire("pq.read_table"))
    for venue in VENUES:
        folder = paths.odds_snapshot_path("jra", HOLD_DAY.isoformat(), venue).parent
        (folder / f"{venue}.results.json").mkdir(parents=True, exist_ok=True)
    return calls


def _one_race_card(day: date) -> pd.DataFrame:
    return pd.DataFrame(
        [{"race_id": rid("TOK", 1, day), "venue": "TOK", **keys_of(T16), "top3_time_s": np.nan}]
    )


@pytest.mark.parametrize("replay", [False, True], ids=["prospective", "replay"])
def test_run_race_time_day_refuses_a_holdout_day_before_any_read(model, monkeypatch, replay):
    """D14: on a holdout day the runner raises, naming the date and the holdout, before it reads
    the card, a sidecar or the store, and writes no log. A development day on or after the
    holdout start still runs (one race, no sidecar: recorded as skipped)."""
    from horsetrack.core.eval.holdout import is_holdout_day

    assert is_holdout_day("japan", HOLD_DAY) and not is_holdout_day("japan", DEV_AFTER_START)
    dev = race_time_day.run_race_time_day(
        DEV_AFTER_START,
        model=model,
        model_version=MV,
        races=lambda: _one_race_card(DEV_AFTER_START),
        now=FakeTime(at(0, 30, day=DEV_AFTER_START)).now,
        sleep=no_sleep,
        replay=replay,
    )
    assert [(r["race_id"], r["status"]) for r in dev.records] == [
        (rid("TOK", 1, DEV_AFTER_START), "skipped")
    ]

    leaks = _tripwires(monkeypatch)
    card_reads: list[int] = []

    def races():
        card_reads.append(1)
        raise Leak("races() was read")

    with pytest.raises(Exception) as info:  # noqa: PT011 - the type is the coder's; the text is pinned
        race_time_day.run_race_time_day(
            HOLD_DAY,
            model=model,
            model_version=MV,
            races=races,
            now=FakeTime(at(0, 30, day=HOLD_DAY)).now,
            sleep=no_sleep,
            replay=replay,
        )
    assert not isinstance(info.value, Leak), info.value
    assert _says_holdout(str(info.value), HOLD_DAY), str(info.value)
    assert card_reads == [] and leaks == []
    assert not race_time_day.race_time_log_path(HOLD_DAY).exists()


def _store(days: tuple[date, ...]) -> dict[str, pd.DataFrame]:
    """A synthetic JRA store: two races a day; runners placed 1, 2, 3, 3 (a dead heat for third),
    their times varying by day and race."""
    races, runners = [], []
    for d_i, day in enumerate(days):
        for no, (venue, surface, dist) in enumerate(
            (("TOK", "turf", 1600), ("KYO", "dirt", 1800)), 1
        ):
            race_id = rid(venue, no, day)
            races.append(
                {
                    "race_id": race_id,
                    "meeting_id": f"{day.isoformat()}:{venue}",
                    "race_date": day.isoformat(),
                    "venue": venue,
                    "surface": surface,
                    "distance_m": dist,
                    "course_config": "outer",
                    "rail_setting": "B",
                    "race_class": 3,
                    "post_time_local": f"1{no}:40",
                }
            )
            base = dist / 16.6 + d_i + 0.3 * no
            for pos, t in (
                (1, base),
                (2, base + 0.2),
                (3, base + 0.5),
                (3, base + 0.9),
                (4, base + 2),
            ):
                runners.append(
                    {"race_id": race_id, "finish_pos": pos, "finish_time_s": t, "race_date": day}
                )
    return {"races": pd.DataFrame(races), "runners": pd.DataFrame(runners)}


def test_load_day_races_refuses_a_holdout_day_before_any_store_read(monkeypatch):
    """D14: load_day_races on a holdout day raises, naming the date and the holdout, and reads no
    store table (no races, no runners). On a development day on or after the start it returns
    the day's races with their top-3 times, (1st + 2nd + mean of the 3rds) / 3."""
    store = _store((DEV_AFTER_START, HOLD_DAY))

    def fake_read_table(source, name, columns=None):
        frame = store[name]
        return frame if columns is None else frame.loc[:, list(columns)]

    monkeypatch.setattr(tables, "read_table", fake_read_table)
    monkeypatch.setattr(race_time_day, "read_table", fake_read_table, raising=False)
    got = race_time_day.load_day_races(DEV_AFTER_START)
    assert sorted(got["race_id"]) == sorted(
        [rid("TOK", 1, DEV_AFTER_START), rid("KYO", 2, DEV_AFTER_START)]
    )
    want = {}
    for race_id, group in store["runners"].groupby("race_id"):
        t = dict(zip(group["finish_pos"], group["finish_time_s"], strict=False))
        thirds = group.loc[group["finish_pos"] == 3, "finish_time_s"].mean()
        want[race_id] = (t[1] + t[2] + thirds) / 3
    for row in got.to_dict("records"):
        assert row["top3_time_s"] == pytest.approx(want[row["race_id"]], abs=1e-9), row
        assert row["track"] == row["venue"]

    leaks = _tripwires(monkeypatch)
    with pytest.raises(Exception) as info:  # noqa: PT011 - the type is the coder's; the text is pinned
        race_time_day.load_day_races(HOLD_DAY)
    assert not isinstance(info.value, Leak), info.value
    assert _says_holdout(str(info.value), HOLD_DAY), str(info.value)
    assert leaks == []


def test_the_race_time_command_refuses_a_holdout_day(model, monkeypatch):
    """D14: `japan race-time --date <holdout day>` (and with --replay) exits non-zero with a
    message naming the date and the holdout, reads no card, sidecar or store, and writes no
    log."""
    leaks = _tripwires(monkeypatch)
    card_reads: list[date] = []

    def load_day_races(day):
        card_reads.append(day)
        raise Leak("load_day_races was called")

    clock = FakeTime(at(0, 30, day=HOLD_DAY))
    monkeypatch.setattr(race_time_day, "utc_now", clock.now)
    monkeypatch.setattr(race_time_day, "sleep", no_sleep)
    monkeypatch.setattr(race_time_day, "load_speed_model", lambda day: (model, MV))
    monkeypatch.setattr(race_time_day, "load_day_races", load_day_races)
    from horsetrack.cli import app

    for extra in ([], ["--replay"]):
        result = CliRunner().invoke(
            app, ["japan", "race-time", "--date", HOLD_DAY.isoformat(), *extra]
        )
        text = ANSI.sub("", result.output)  # stdout and stderr together
        assert result.exit_code != 0, text
        assert not isinstance(result.exception, Leak), text
        # one line of the message names both the date and the holdout
        assert any(_says_holdout(line, HOLD_DAY) for line in text.splitlines()), text
    assert card_reads == [] and leaks == []
    assert not race_time_day.race_time_log_path(HOLD_DAY).exists()


# --------------------------------------------------------------------------- round 3

CT16B = ("TOK", "turf", 1600, "outer", "B", "C1")  # CT16 with its rail known
RAIL_PAR = {**CARD_PAR, CT16B: 93.7}


def write_table(name: str, rows: list[dict]) -> None:
    """One store table, every declared column (null where not given)."""
    columns = schemas.TABLES[name].columns
    schema = pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in columns.items()])
    data = {c: [row.get(c) for row in rows] for c in columns}
    path = tables.table_path("jra", name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(data, schema=schema), path)


def with_rail(venue: str, no: int, cell: tuple, post: datetime, rail: str):
    race, runners = resulted(venue, no, cell, post)
    race["rail_setting"] = rail
    return race, runners


def rails_of(frame: pd.DataFrame) -> dict[str, str]:
    return dict(zip(frame["race_id"], frame["rail"], strict=True))


def test_a_retained_card_takes_the_store_rail_of_a_turf_race(monkeypatch):
    """Round-3 finding 3: with the upcoming card kept on disk (it has no rail), Tokyo R1's store
    row (rail B, from its going page) gives the card row rail B, so a replay forecasts it from the
    rail-B par, not the no-rail one; Tokyo R3 (no store row) keeps "unknown", and a dirt race
    stays "unknown" whatever its store row says. The card snapshot keeps rail B."""
    rail_model = SpeedModel(SPEC).fit(fit_frame(RAIL_PAR))
    b_par, no_rail_par = (rail_model.par(**keys_of(c)) for c in (CT16B, CT16))
    assert abs(b_par - no_rail_par) > 0.5
    write_upcoming(DAY)
    tok1, tok1_runners = with_rail("TOK", 1, CT16, at(1, 0), "B")
    tok2, tok2_runners = with_rail("TOK", 2, CTD14, at(1, 35), "A")
    other, other_rows = a_week_before()
    write_store([tok1, tok2, other], tok1_runners + tok2_runners + other_rows)

    got = rails_of(race_time_day.load_day_races(DAY))
    assert got[rid("TOK", 1)] == "B"
    for no_rail in (rid("TOK", 2), rid("TOK", 3), rid("TOK", 4), rid("KYO", 1), rid("KYO", 2)):
        assert got[no_rail] == "unknown", no_rail

    clock = FakeTime(datetime(2025, 2, 18, 0, 0, tzinfo=UTC))
    replayed = race_time_day.run_race_time_day(
        DAY,
        model=rail_model,
        model_version=MV,
        races=lambda: race_time_day.load_day_races(DAY),
        now=clock.now,
        sleep=no_sleep,
        replay=True,
    )
    recs = {r["race_id"]: r for r in replayed.records}
    assert recs[rid("TOK", 1)]["status"] == "ok"
    assert recs[rid("TOK", 1)]["par_s"] == pytest.approx(b_par, abs=1e-9)
    assert recs[rid("TOK", 3)]["par_s"] == pytest.approx(no_rail_par, abs=1e-9)
    dirt_par = rail_model.par(**keys_of(CTD14))
    assert recs[rid("TOK", 2)]["par_s"] == pytest.approx(dirt_par, abs=1e-9)
    snap = race_time_day.load_card_snapshots(DAY)[recs[rid("TOK", 1)]["card_version"]]
    assert rails_of(snap)[rid("TOK", 1)] == "B"


def test_a_turf_rail_is_the_track_condition_published_by_the_race_cutoff():
    """Round-3 finding 3: a turf race takes the latest turf rail of its venue and day published by
    its own cutoff (post - primary), ahead of its store row: rail C (00:30) for Tokyo R1 (cutoff
    00:58), rail D (published at Tokyo R3's cutoff, 02:08) for Tokyo R3, not rail E (02:30); a
    dirt publication, another day's rail and the other venue's races change nothing."""
    write_upcoming(DAY)
    tok1, tok1_runners = with_rail("TOK", 1, CT16, at(1, 0), "B")
    other, other_rows = a_week_before()
    write_store([tok1, other], tok1_runners + other_rows)
    week_before = DAY - timedelta(days=7)
    publications = [
        ("TOK", DAY, "turf", "C", at(0, 30)),
        ("TOK", DAY, "dirt", "Z", at(0, 40)),
        ("TOK", DAY, "turf", "D", at(2, 10) - PRIMARY),
        ("TOK", DAY, "turf", "E", at(2, 30)),
        ("KYO", week_before, "turf", "A", at(0, 30, day=week_before)),
    ]
    write_table(
        "track_condition",
        [
            {
                "jurisdiction": "jra",
                "venue": venue,
                "race_date": day,
                "surface": surface,
                "going": "good",
                "rail": rail,
                "published_utc": when,
                "available_utc": when,
                "ingested_utc": when + timedelta(minutes=1),
            }
            for venue, day, surface, rail, when in publications
        ],
    )
    assert at(0, 30) <= at(1, 0) - PRIMARY < at(2, 10) - PRIMARY < at(2, 30)

    got = rails_of(race_time_day.load_day_races(DAY))
    assert got[rid("TOK", 1)] == "C"
    assert got[rid("TOK", 3)] == "D"
    for race_id in (rid("TOK", 2), rid("TOK", 4), rid("KYO", 1), rid("KYO", 2), rid("KYO", 3)):
        assert got[race_id] == "unknown", race_id


class MovingCardModel(SpeedModel):
    """The #305 model that runs `during()` (a class attribute, so the recorder's copy runs it too)
    while it computes the forecast: the card refreshed mid-computation."""

    during = None

    def expected_top3_time(self, race, *, at_utc):
        if type(self).during is not None:
            type(self).during()
        return super().expected_top3_time(race, at_utc=at_utc)


@pytest.mark.parametrize(
    ("moved_to", "mode", "ready_by_cutoff"),
    [(None, "live", 1), (at(0, 58), "late", 0), (at(0, 57), "replay", 0)],
    ids=["card unchanged", "post moved two minutes earlier", "post moved to the completion"],
)
def test_a_card_post_moved_during_the_computation_is_judged_at_completion(
    model, moved_to, mode, ready_by_cutoff
):
    """Round-3 finding 2: no sidecar has Tokyo R1, so its post is the card's (01:00, cutoff
    00:58). The card is refreshed while the 00:57 forecast computes; the completion check rereads
    it: moved to 00:58 the record is late (revised cutoff 00:56), moved to 00:57 it is a replay,
    unchanged it is live. The post the inputs were taken for stays 01:00."""
    race_id = rid("TOK", 1)
    card = {"post": at(1, 0)}

    def races() -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "race_id": race_id,
                    "venue": "TOK",
                    **keys_of(T16),
                    "top3_time_s": np.nan,
                    "post_utc": card["post"],
                }
            ]
        )
        frame["post_utc"] = pd.to_datetime(frame["post_utc"], utc=True)
        return frame

    def refresh() -> None:
        if moved_to is not None:
            card["post"] = moved_to

    moving = copy.deepcopy(model)
    moving.__class__ = MovingCardModel
    clock = FakeTime(at(0, 56, 50))
    MovingCardModel.during = refresh
    try:
        got = race_time_day.run_race_time_day(
            DAY, model=moving, model_version=MV, races=races, now=clock.now, sleep=clock.sleep
        )
    finally:
        MovingCardModel.during = None
    (rec,) = got.records
    assert rec["status"] == "ok"
    assert utc(rec["post_utc_at_cutoff"]) == at(1, 0)
    assert utc(rec["input_cutoff_utc"]) == at(1, 0) - PRIMARY
    assert utc(rec["completion_utc"]) == at(0, 57)
    assert utc(rec["latest_post_utc"]) == (moved_to or at(1, 0))
    assert rec["mode"] == mode
    assert got.coverage["TOK"]["ready_by_cutoff"] == ready_by_cutoff
    assert read_log() == json.loads(json.dumps(got.records))


@pytest.mark.parametrize(
    ("stamped_after_read_s", "counts"),
    [(20, False), (-10, True)],
    ids=["sighting logged after the read", "sighting logged before the read"],
)
def test_a_sighting_logged_after_the_runner_read_its_clock_is_not_an_input(
    model, stamped_after_read_s, counts
):
    """Round-3 finding 1: the runner reads the clock (02:42:00) before the card; the card read
    takes 20 s, during which the odds logger records Tokyo R1's result. A sighting stamped after
    02:42:00 is not one Tokyo R4's forecast (cutoff 02:43) saw: input_snapshot_utc is 02:42:00,
    not the 02:42:20 of the recorder's own clock, and Tokyo R1 is neither used nor pending. A
    sighting stamped before the read is used. Both records verify against the source."""
    early, target = rid("TOK", 1), rid("TOK", 4)
    posts = {1: POSTS[early], 4: POSTS[target]}
    odds_log.write_post_times(sidecar("TOK", ".races.json"), posts, at(0, 0))
    read_at = at(2, 42)
    seen = read_at + timedelta(seconds=stamped_after_read_s)
    clock = FakeTime(read_at)
    calls = []

    def races() -> pd.DataFrame:
        if not calls:  # the first read is slow, and the logger writes meanwhile
            clock.at += timedelta(seconds=20)
            odds_log.record_published(sidecar("TOK", ".results.json"), [1], seen)
        calls.append(clock.at)
        frame = store_races()
        return frame[frame["race_id"].isin([early, target])].reset_index(drop=True)

    got = race_time_day.run_race_time_day(
        DAY, model=model, model_version=MV, races=races, now=clock.now, sleep=clock.sleep
    )
    recs = {r["race_id"]: r for r in got.records}
    rec = recs[target]
    assert rec["status"] == "ok" and rec["mode"] == "live"
    assert utc(rec["input_snapshot_utc"]) == read_at
    assert utc(rec["completion_utc"]) == read_at + timedelta(seconds=20)
    assert ({u["race_id"] for u in rec["results_used"]} == {early}) is counts
    assert early not in {u["race_id"] for u in rec["results_pending"]}

    src = store_races()
    src = src[src["race_id"].isin([early, target])].reset_index(drop=True)
    src["post_utc"] = pd.to_datetime(src["race_id"].map(POSTS), utc=True)
    src["first_seen_utc"] = pd.to_datetime(
        [seen if r == early else None for r in src["race_id"]], utc=True
    )
    ref = copy.deepcopy(model)
    if counts:
        ref.observe(
            src[src["race_id"] == early].rename(
                columns={"post_utc": "race_utc", "first_seen_utc": "result_available_utc"}
            )
        )
    target_row = {"race_id": target, "race_utc": POSTS[target], **keys_of(T16)}
    want = ref.expected_top3_time(target_row, at_utc=read_at + timedelta(days=30))
    assert rec["same_day_s"] == pytest.approx(want, abs=1e-9)
    if counts:
        assert abs(want - model.par(**keys_of(T16))) > 1e-6

    snap = race_time_day.load_card_snapshots(DAY)[rec["card_version"]]
    problems = verify_replay(
        got.records,
        card=snap,
        results=src,
        model=model,
        model_version=MV,
        card_version=rec["card_version"],
    )
    assert problems == []
