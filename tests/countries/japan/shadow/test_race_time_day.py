"""The race-time recorder's day entry point (ticket #414, review finding 1).

Contract pinned here (`horsetrack.countries.japan.shadow.race_time_day`; the coder follows it):

- `run_race_time_day(race_date, *, model, model_version, races, now, sleep, replay=False,
  day_dir=None) -> RaceTimeRun`: records every race of the day once through
  `race_time.record_race` and appends each record, as it is made, to the day's log.
  - `model`: a fitted #305 `SpeedModel` holding no observations; `model_version` its label.
  - `races()`: re-readable; a DataFrame of the day's card from the store, one row per race:
    race_id (`jra:<YYYY-MM-DD>:<VENUE>:<NN>`), venue, the model's par keys and top3_time_s (NaN
    until the race has a result). It carries no post time and no sighting.
  - Post times come from each venue's `<VENUE>.races.json` sidecar, first sightings (#225) from
    its `<VENUE>.results.json` sidecar, both in `board.snapshot.jra_day_dir(race_date, day_dir)`
    (the odds logger's layout); a race with no post there is recorded as skipped (no post).
  - Prospective (replay=False): each race with a post is recorded at most 5 minutes before its
    input cutoff (post - primary) and not after it when the run starts in time, waiting with
    `sleep(seconds)` between races; a race whose cutoff has already passed is recorded at once
    (late or replay by the clock). `latest_post` re-reads the races sidecar after the
    computation. The run returns once every race of the card has its record.
  - replay=True: every race is recorded at once with replay=True; it never waits (no `sleep`).
  - card_version: chosen by the runner from the card inputs (a non-empty text, the same for an
    unchanged card).
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
import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
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


def fit_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4141)
    rows = []
    for day in TRAIN_DAYS:
        shift = {g: rng.normal(0.0, 0.6) for g in V_DAY}
        for n, cell in enumerate(PAR):
            for rep in range(2):
                post = at(1 + n, 30 * rep, day=day)
                rows.append(
                    {
                        "race_id": f"jra:{day.isoformat()}:{cell[0]}:{10 * rep + n + 1:02d}",
                        "race_utc": post,
                        **keys_of(cell),
                        "top3_time_s": PAR[cell] + shift[cell[:2]] + rng.normal(0.0, 0.25),
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
    """A fake UTC clock; `sleep` moves it and nothing really waits."""

    def __init__(self, start: datetime) -> None:
        self.at = start

    def now(self) -> datetime:
        return self.at

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0, seconds
        self.at += timedelta(seconds=seconds)
        assert self.at < at(12, 0), "the run waits past the day's end"


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
