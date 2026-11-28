"""The day-before window end to end (ticket #358): capture_meeting with bracket quinella, the probe
of the オッズ menu, and `log-day` the evening before a race date, on the JRADB double.

Contract pinned here (the coder follows it), on top of test_day_before_schedule.py's for run_day
and test_log_day_cli.py's / test_unattended.py's for log-day:

capture (horsetrack.countries.japan.jra.capture)
- `capture_meeting` takes BRACKET_QUINELLA with the other tier-1 pools (and `log-day --pools`
  accepts it): one request more per race, the race list's 枠連 page, its rows in the shape tier 2
  wrote them (exotic_pages board rows: the page's prices and its "__pool__" row) with the
  capture's snapshot_utc. logger.yaml's `requests` say what a capture of its pools costs, for
  tier 1 and tier 2 alike.
- `sales_open(fetcher, race_date) -> bool`: one request, the オッズ menu; True when it lists a
  meeting under the race date's heading (the day-before sales are open), else False.

log-day
- The JRA site's `sales_open()` is capture.sales_open on its race date: run_day's probe.
- `--dry-run` for a race date (run the evening before it, or any time) prints, besides the plan
  of before: a line for the day-before window naming the probe start on the day before (its
  date, ISO, and probe_from), every day_before.every_s seconds, until each race's T-15 (the text
  "T-15"); probe_every_s and probe_until appear too; and a line saying each race is captured
  from T-15 (hot from "T-20") at the fine cadence (the last window's every_s). The pools line
  lists BRACKET_QUINELLA with the other tier-1 pools; the tier-2 line does not. A source without
  `day_before` (NAR) prints no such line.
- The race date when `--date` is not given (attended and `--unattended` alike), per source: today
  in Tokyo, unless that source has `day_before`, today is over for it (past today's `stop`, or
  today's block, no-meeting or day-done marker) and the time is at or after tomorrow's probe
  start (day_before.probe_from today): then tomorrow. Before the probe start a start past the
  stop sends nothing, as before. `--date D` is D for every source: an attended `--date D` run on
  the evening before runs D's day-before window. With `--sources jra,nar` each source logs its
  own race date in the one process: its own log paths, health lines (dated with its date) and,
  unattended, markers (`<source>-day-done.txt` with its date).
- A block on a probe (403/429) appends `<host> <race date> ...` to jra-block.txt for the race
  date, so every later start that evening and on the race date sends nothing.
- Unchanged board (decision 4 of #358): a round whose pages show the same 発表 clock as the
  previous capture still fetches every page and writes its rows, with its own snapshot_utc, in
  the log format of any capture: a snapshot like any other, no failure, no alert.

The double's pages are static (the same prices and the same 発表 clock, 14時00分, every time), so
every night round is an unchanged board; its オッズ menu lists the race date only from `opens_at`
(jradb_double.ClockedSite). Post times are moved 20 minutes later (late_posts) so every race's
T-20 falls in the 20 s window. Nothing goes to the network and nothing really sleeps.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.jra import capture
from horsetrack.countries.japan.jra.capture import capture_meeting, capture_tier2

# isort: split
import jradb_double as dbl
import nar_double as nd
from jradb_double import POOL_KIND, Meeting, World
from nar_double import NarMeeting, NarWorld
from thread_clock import ThreadClock

UTC = timezone.utc
TOKYO = ZoneInfo("Asia/Tokyo")
FRI, SAT, SUN, MON = (date(2026, 11, d) for d in range(27, 31))
MODULE = "horsetrack.countries.japan.logger.log_day"
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
BQ = "BRACKET_QUINELLA"
TIER1 = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", BQ]
EXOTIC_KEYS = {"snapshot_utc", "race_id", "source_update_utc", "fetched_utc", "page_state", "pool",
               "combination", "odds", "status_raw"}  # fmt: skip
T0 = datetime(2026, 11, 28, 9, 31, 0, tzinfo=UTC)
EPS = 1e-6
GRAIN_S = 300
LINE = re.compile(
    r"^(?:ALERT )?(?P<date>\d{4}-\d{2}-\d{2}) (?P<venue>\S+) .*?source=(?P<source>[a-z]+)\b"
)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    """The data directory under tmp_path; no real `.env` can supply a setting; no real network."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    monkeypatch.setenv("HORSETRACK_USER_AGENT", "horsetrack-tests/1.0")
    (tmp_path / "data").mkdir()

    def no_network(self, request):
        raise AssertionError(f"a real HTTP request was attempted: {request.method} {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    return tmp_path / "data"


@pytest.fixture
def late_posts(monkeypatch):
    """Every race of the double posts 20 minutes later than make_race says (first race 09:50+)."""
    original = dbl.make_race

    def make_race(day, venue, number):
        race = original(day, venue, number)
        return dataclasses.replace(race, post_minutes=race.post_minutes + 20)

    monkeypatch.setattr(dbl, "make_race", make_race)


class Clock:
    """now() is the fake time; sleep(s) moves it (never back) and really waits for nothing."""

    def __init__(self, start: datetime):
        self.t = start
        self.slept = 0.0

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        self.slept += seconds
        self.t += timedelta(seconds=seconds)


# ---------------------------------------------------------------- config and time


def shipped() -> dict:
    return copy.deepcopy(load_country_yaml("japan", "logger"))


def day_before() -> dict:
    block = shipped().get("day_before")
    assert isinstance(block, dict), "logger.yaml has no day_before block"
    return block


def tier2_pools() -> list[str]:
    return list(shipped()["tier2"]["pools"])


def at(day: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    """The UTC instant of a Tokyo wall time on `day`."""
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=TOKYO).astimezone(UTC)


def wall(day: date, hhmm: str) -> datetime:
    hh, mm = (int(x) for x in str(hhmm).split(":"))
    return at(day, hh, mm)


def first_window(day: date) -> datetime:
    return min(wall(day, w["from"]) for w in shipped()["windows"])


def hhmm(t: datetime) -> str:
    return t.astimezone(TOKYO).strftime("%m-%d %H:%M:%S")


def first_probe_at_or_after(t0: datetime, opens: datetime) -> datetime:
    """The probe grid from t0 (every probe_every_s): its first instant at or after `opens`."""
    every = float(day_before()["probe_every_s"])
    k = max(0, math.ceil((opens - t0).total_seconds() / every - EPS))
    return t0 + timedelta(seconds=k * every)


def number(value) -> str:
    """A regex for the number `value` standing alone in text (not inside 17:00, 1.5 or T-20)."""
    v = float(value)
    text = str(int(v)) if v == int(v) else re.escape(str(value))
    return rf"(?<![\d.:-]){text}(\.0+)?(?![\d.:])"


# ---------------------------------------------------------------- the double and the command


def evening_world(sat: list[Meeting] | None = None, sun: list[Meeting] | None = None):
    """Saturday racing at Tokyo, Sunday at Kyoto, and the オッズ menu of Saturday evening before
    Sunday's sales open (Saturday's meetings only)."""
    sat = sat if sat is not None else [Meeting("TOK", 1)]
    sun = sun if sun is not None else [Meeting("KYO", 1, nichi=8)]
    world = World({SAT: sat, SUN: sun}, tier2=tier2_pools())
    before = World({SAT: sat}, tier2=tier2_pools()).pages[dbl.ODDS_MENU]
    return world, before


def clocked(world: World, before: bytes, clock, opens: datetime, **kw) -> dbl.ClockedSite:
    return dbl.ClockedSite(dict(world.pages), clock=clock, opens_at=opens, menu_before=before, **kw)


def log_day(monkeypatch, clock: Clock, site: dbl.JradbSite, *args: str):
    """`horsetrack japan log-day *args` with the seams on the fake clock and `site`. Returns
    (result, seconds slept)."""
    mod = importlib.import_module(MODULE)
    monkeypatch.setattr(mod, "utc_now", clock.now)
    monkeypatch.setattr(mod, "sleep", clock.sleep)
    monkeypatch.setattr(mod, "make_fetcher", lambda *a, **k: dbl.make_fetcher(site))
    from horsetrack.cli import app

    before = clock.slept
    result = CliRunner().invoke(app, ["japan", "log-day", *args])
    out = plain(result.output)
    assert "No such option" not in out and "Usage:" not in out, f"{list(args)} refused: {out}"
    return result, clock.slept - before


def plain(text: str) -> str:
    return ANSI.sub("", text)


def lines_of(result) -> list[str]:
    return [line for line in plain(result.output).splitlines() if line.strip()]


def logs() -> Path:
    return paths.logs_dir("japan")


def health_lines() -> list[str]:
    path = logs() / "health.log"
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def marker_days(name: str) -> list[str]:
    path = logs() / name
    if not path.is_file():
        return []
    return [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line]


def block_lines() -> list[list[str]]:
    path = logs() / "jra-block.txt"
    if not path.is_file():
        return []
    return [line.split()[:2] for line in path.read_text(encoding="utf-8").splitlines() if line]


def tier1_members(day: date, venue: str) -> list[list[dict]]:
    """The venue's log members holding tier-1 pools (one per tier-1 capture)."""
    path = paths.odds_snapshot_path("jra", day.isoformat(), venue)
    return [m for m in odds_log.iter_members(path) if {r["pool"] for r in m} <= set(TIER1)]


def instant(value) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def snapshot_times(day: date, venue: str, race_id: str) -> list[datetime]:
    return sorted(
        {instant(r["snapshot_utc"]) for m in tier1_members(day, venue) for r in m
         if r["race_id"] == race_id}
    )  # fmt: skip


def price_board(member: list[dict]) -> list[tuple]:
    """A capture's prices and totals, every pool: what an unchanged board repeats."""
    return sorted(
        ((r["pool"], r["combination"], r.get("odds"), r.get("odds_hi"), r.get("investment"))
         for r in member),
        key=repr,
    )  # fmt: skip


def bracket(rows) -> tuple[dict[str, dict[str, float]], dict[str, list[float]]]:
    """race_id -> {combination: odds} of the BRACKET_QUINELLA price rows; race_id -> its pool
    totals (investment)."""
    boards: dict[str, dict[str, float]] = {}
    totals: dict[str, list[float]] = {}
    for r in rows:
        if r["pool"] != BQ:
            continue
        if r["combination"] == "__pool__":
            totals.setdefault(r["race_id"], []).append(r["investment"])
        else:
            boards.setdefault(r["race_id"], {})[r["combination"]] = r["odds"]
    return boards, totals


def assert_grain_until_t15(times: list[datetime], opened: datetime, post: datetime, where: str):
    """From the open: the first capture within day_before.every_s; no gap over it before the race
    date's first window, none over 5 minutes from then to T-15."""
    every = float(day_before()["every_s"])
    w = first_window(post.astimezone(TOKYO).date())
    t15 = post - timedelta(minutes=15)
    inside = [t for t in times if opened <= t <= t15]
    assert inside, f"{where}: not captured between the open {hhmm(opened)} and T-15"
    assert (inside[0] - opened).total_seconds() <= every + EPS, f"{where}: first {hhmm(inside[0])}"
    for a, b in zip(inside, inside[1:], strict=False):
        limit = every if b < w else GRAIN_S
        gap = (b - a).total_seconds()
        assert gap <= limit + EPS, f"{where}: {gap:.0f} s from {hhmm(a)} to {hhmm(b)}"
    assert (t15 - inside[-1]).total_seconds() <= GRAIN_S + EPS, f"{where}: last {hhmm(inside[-1])}"


# ---------------------------------------------------------------- capture


@pytest.mark.parametrize(
    ("races", "pools"),
    [
        (1, TIER1),
        (3, TIER1),
        (12, TIER1),
        (4, [BQ]),
        (4, ["WIN", BQ]),
        (5, [BQ, "QUINELLA_PLACE", "PLACE"]),
    ],
    ids=["1-race", "3-races", "12-races", "bq-only", "win-and-bq", "bq-wide-place"],
)
def test_capture_meeting_takes_bracket_quinella_with_the_tier1_pools(races, pools):
    world = World({SAT: [Meeting("TOK", races), Meeting("KYO", 2, nichi=3)]})
    site = dbl.JradbSite(dict(world.pages))
    fetcher = dbl.make_fetcher(site)
    with fetcher:
        cap = capture_meeting(fetcher, SAT, "TOK", pools, now=lambda: T0)

    kinds = {POOL_KIND[p] for p in pools if p != BQ}
    pages = len(kinds) + (BQ in pools)
    assert cap.requests == 2 + pages * races == len(site.seen), (cap.requests, len(site.seen))
    assert all(s.served for s in site.seen), "a request for a page the race lists do not link"
    asked = set(site.links()[2:])
    want = set()
    for n in range(1, races + 1):
        want |= {world.odds_link(SAT, "TOK", n, k) for k in kinds}
        if BQ in pools:
            want.add(world.odds_link(SAT, "TOK", n, BQ))
    assert asked == want
    assert {r["snapshot_utc"] for r in cap.rows} == {T0}

    classic = [p for p in pools if p != BQ]
    got = dbl.price_rows([r for r in cap.rows if r["pool"] != BQ])
    assert got == world.expected_rows(SAT, "TOK", classic)
    boards, totals = bracket(cap.rows)
    if BQ not in pools:
        assert boards == {} and totals == {}
        return
    for n in range(1, races + 1):
        race = world.race(SAT, "TOK", n)
        assert boards.get(race.race_id) == dbl.exotic_expected(race, BQ), race.race_id
        assert totals.get(race.race_id) == [dbl.exotic_tickets(race, BQ) * dbl.UNIT_STAKE]
    assert set(boards) == {world.race(SAT, "TOK", n).race_id for n in range(1, races + 1)}
    for r in cap.rows:
        if r["pool"] == BQ:
            keys = EXOTIC_KEYS | ({"investment"} if r["combination"] == "__pool__" else set())
            assert set(r) == keys, f"a bracket quinella row not in tier 2's shape: {sorted(r)}"


def test_logger_yaml_requests_are_what_a_capture_of_its_pools_costs():
    config = shipped()
    races = 5
    world = World({SAT: [Meeting("TOK", races)]}, tier2=config["tier2"]["pools"])
    site = dbl.JradbSite(dict(world.pages))
    fetcher = dbl.make_fetcher(site)
    nav, per = config["requests"]["navigation"], config["requests"]["per_race"]
    assert per == len(dbl.KINDS) + 1, "tier 1: the three pages of before and the 枠連 page"
    with fetcher:
        tier1 = capture_meeting(fetcher, SAT, "TOK", config["pools"], now=lambda: T0)
        tier2 = capture_tier2(fetcher, SAT, "TOK", config["tier2"]["pools"], races=[1, 3, 4],
                              now=lambda: T0)  # fmt: skip
    assert tier1.requests == nav + per * races, tier1.requests
    assert {r["pool"] for r in tier1.rows} == set(TIER1)
    t2 = config["tier2"]["requests"]
    assert tier2.requests == t2["navigation"] + t2["per_race"] * 3, tier2.requests
    assert {r["pool"] for r in tier2.rows} == set(config["tier2"]["pools"])
    assert all(s.served for s in site.seen)


@pytest.mark.parametrize(
    ("days", "race_date", "expected"),
    [
        ({SAT: [Meeting("TOK", 2)]}, SUN, False),  # Saturday evening, before Sunday's sales
        ({SAT: [Meeting("TOK", 2)]}, SAT, True),
        ({SAT: [Meeting("TOK", 2)], SUN: [Meeting("KYO", 2, nichi=8)]}, SUN, True),
        ({SAT: [Meeting("TOK", 2)], SUN: [Meeting("KYO", 2), Meeting("NAK", 1)]}, MON, False),
        ({SUN: [Meeting("NAK", 1, 4, 2)]}, SUN, True),
    ],
    ids=["sat-menu-for-sun", "sat-menu-for-sat", "both-for-sun", "both-for-mon", "sun-only"],
)
def test_sales_open_reads_the_odds_menu_for_the_race_date(days, race_date, expected):
    sales_open = getattr(capture, "sales_open", None)
    assert callable(sales_open), "capture.sales_open(fetcher, race_date) is missing"
    for pages in (World(days).pages, dbl.scramble(World(days).pages)[0]):
        site = dbl.JradbSite(dict(pages))
        fetcher = dbl.make_fetcher(site)
        with fetcher:
            got = sales_open(fetcher, race_date)
        assert got is expected, f"{race_date} on a menu of {sorted(days)}: {got}"
        assert site.links() == [dbl.ODDS_MENU], site.links()
        assert [t.page_type for t in fetcher.tasks] == ["official_menu"]


# ---------------------------------------------------------------- the dry run


@pytest.mark.parametrize(
    ("args", "race_date", "pools"),
    [
        (["--date", SUN.isoformat()], SUN, TIER1),
        ([], SUN, TIER1),  # 19:00 Saturday, past its stop: the default date is tomorrow
        (["--date", MON.isoformat(), "--venues", "KYO", "--pools", BQ], MON, [BQ]),
        (["--date", SAT.isoformat(), "--sources", "nar"], SAT, None),
    ],
    ids=["date-given", "default-date", "bracket-quinella-only", "nar"],
)
def test_dry_run_lists_the_day_before_window_and_the_t15_grain(
    monkeypatch, data_dir, args, race_date, pools
):
    world, before = evening_world()
    clock = Clock(at(SAT, 19, 0))
    site = clocked(world, before, clock.now, at(SAT, 18, 30))
    result, waited = log_day(monkeypatch, clock, site, "--dry-run", *args)

    out = plain(result.output)
    assert result.exit_code == 0, out
    assert site.seen == [] and waited == 0.0, "the dry run sent a request or waited"
    lines = lines_of(result)
    assert race_date.isoformat() in lines[0], f"the dry run is not for {race_date}: {lines[0]}"
    db, eve = day_before(), (race_date - timedelta(days=1)).isoformat()
    probe_lines = [x for x in lines if str(db["probe_from"]) in x and eve in x]
    if pools is None:  # NAR: no day-before window
        assert probe_lines == [] and "T-15" not in out, out
        return
    assert probe_lines, f"no line for the day-before window from {eve} {db['probe_from']}: {out}"
    line = probe_lines[0]
    assert re.search(number(db["every_s"]), line), f"the window's every_s not on: {line}"
    assert "T-15" in line, f"the window's end (each race's T-15) not on: {line}"
    assert re.search(number(db["probe_every_s"]), out), "probe_every_s not printed"
    assert str(db["probe_until"]) in out, "probe_until not printed"
    fine = sorted(shipped()["windows"], key=lambda w: str(w["from"]))[-1]["every_s"]
    grain = [x for x in lines if "T-15" in x and "T-20" in x and re.search(number(fine), x)]
    assert grain, f"no line for the fine grain from T-15 (hot from T-20) at {fine} s: {out}"
    named = [x for x in lines if all(p in x for p in pools)]
    assert named, f"no line lists the pools {pools}: {out}"
    assert not [x for x in lines if BQ in x and "FORECAST" in x], "bracket quinella in tier 2"


def test_log_day_pools_accepts_bracket_quinella(monkeypatch, data_dir):
    world, before = evening_world()
    clock = Clock(at(SAT, 8, 0))
    site = clocked(world, before, clock.now, at(FRI, 18, 30))
    args = ["--date", SAT.isoformat(), "--venues", "TOK", "--pools", f"WIN,{BQ}", "--dry-run"]
    result, _ = log_day(monkeypatch, clock, site, *args)
    assert result.exit_code == 0, plain(result.output)
    assert any("WIN" in x and BQ in x for x in lines_of(result)), plain(result.output)


# ---------------------------------------------------------------- a run the evening before


def test_log_day_the_evening_before_logs_the_night_from_the_sales_open(
    monkeypatch, data_dir, late_posts
):
    world, before = evening_world()
    opens = at(SAT, 18, 30)
    start = at(SAT, 18, 0)
    t0 = max(start, wall(SAT, day_before()["probe_from"]))
    clock = Clock(start)
    site = clocked(world, before, clock.now, opens)

    result, _ = log_day(monkeypatch, clock, site, "--date", SUN.isoformat())

    out = plain(result.output)
    assert result.exit_code == 0, out
    assert "ALERT" not in out
    assert site.seen and all(s.served for s in site.seen), "a request the double does not serve"

    # The probes: the オッズ menu every probe_every_s from t0 until the one that lists Sunday
    # (at that instant the first capture reads the menu too).
    opened = first_probe_at_or_after(t0, opens)
    every = float(day_before()["probe_every_s"])
    menu_times = site.times_of(dbl.ODDS_MENU)
    probes = [t for t in menu_times if t < opened - timedelta(seconds=EPS)]
    k = round((opened - t0).total_seconds() / every)
    want = [t0 + timedelta(seconds=j * every) for j in range(k)]
    assert len(probes) == len(want) and all(
        abs((a - b).total_seconds()) <= EPS for a, b in zip(probes, want, strict=True)
    ), f"probes {[hhmm(t) for t in probes[:4]]}..., expected every {every} s from {hhmm(t0)}"
    assert min(t for t in menu_times if t not in probes) == opened, "no probe at the open"
    # Before the open: the probes only; no discovery, card or odds page of Sunday.
    early = [s.link for s, t in zip(site.seen, site.at, strict=True) if t < opened]
    assert set(early) <= {dbl.ODDS_MENU}, f"before the open: {set(early) - {dbl.ODDS_MENU}}"

    # Every snapshot of the night, from the open, at the day-before grain until T-15.
    race = world.race(SUN, "KYO", 1)
    times = snapshot_times(SUN, "KYO", race.race_id)
    assert_grain_until_t15(times, opened, race.post_utc, "KYO race 1")
    assert any(at(SUN, 0, 0) <= t <= at(SUN, 5, 0) for t in times), "nothing logged overnight"
    assert not paths.odds_snapshot_path("jra", SAT.isoformat(), "TOK").exists()

    # Bracket quinella every round: each tier-1 snapshot holds it, as the page prints it.
    members = tier1_members(SUN, "KYO")
    assert len(members) == len(times)
    for member in members:
        assert {r["pool"] for r in member} == set(TIER1), {r["pool"] for r in member}
        boards, totals = bracket(member)
        assert boards == {race.race_id: dbl.exotic_expected(race, BQ)}
        assert totals == {race.race_id: [dbl.exotic_tickets(race, BQ) * dbl.UNIT_STAKE]}

    # Unchanged boards (the same 発表 clock and prices every round): every round fetched every
    # page and wrote all its rows, with its own snapshot_utc, in the format of any capture.
    night = [m for m in members if instant(m[0]["snapshot_utc"]) < first_window(SUN)]
    assert len(night) >= 2
    first = price_board(night[0])
    keys = {r["pool"]: set(r) for r in night[0]}
    for member in night[1:]:
        assert price_board(member) == first, "an unchanged board was not logged in full"
        assert {r["pool"]: set(r) for r in member} == keys, "the row format changed"
        assert len({r["snapshot_utc"] for r in member}) == 1
    stamps = [instant(m[0]["snapshot_utc"]) for m in members]
    assert stamps == sorted(set(stamps)), "two captures share a snapshot_utc"
    for kind in (*dbl.KINDS, BQ):
        fetched = len(site.times_of(world.odds_link(SUN, "KYO", 1, kind)))
        assert fetched >= len(members), f"{kind}: {fetched} fetches for {len(members)} snapshots"
    [line] = [x for x in health_lines() if x.startswith(f"{SUN.isoformat()} KYO ")]
    fields = dict(f.split("=", 1) for f in line.split()[2:])
    assert int(fields["snapshots"]) == len(members), line
    assert (fields["failures"], fields["wrong_meeting"], fields["stopped"]) == ("0", "0", "ended")
    assert not [x for x in health_lines() if x.startswith("ALERT")], health_lines()


# ---------------------------------------------------------------- unattended starts


def between_stop_and_probe() -> datetime:
    stop, probe = wall(SAT, shipped()["stop"]), wall(SAT, day_before()["probe_from"])
    assert stop < probe, "no instant between today's stop and tomorrow's probe start"
    return stop + (probe - stop) / 2


@pytest.mark.parametrize(
    ("when", "target"),
    [
        ("between-stop-and-probe", None),
        ("probe-start", SUN),
        ("evening", SUN),
        ("racing", SAT),
    ],
)
def test_an_unattended_start_past_todays_stop_logs_tomorrow(
    monkeypatch, data_dir, late_posts, when, target
):
    world, before = evening_world(sat=[Meeting("TOK", 3)])
    opens = at(SAT, 18, 30)
    start = {
        "between-stop-and-probe": between_stop_and_probe(),
        "probe-start": wall(SAT, day_before()["probe_from"]),
        "evening": at(SAT, 19, 0),
        "racing": at(SAT, 9, 0),
    }[when]
    clock = Clock(start)
    site = clocked(world, before, clock.now, opens)

    result, waited = log_day(monkeypatch, clock, site, "--unattended")

    out = plain(result.output)
    if target is None:
        assert (result.exit_code, len(site.seen), waited) == (0, 0, 0.0), out
        assert health_lines() == [] and marker_days("jra-day-done.txt") == []
        return
    assert result.exit_code == 0, out
    other = SAT if target == SUN else SUN
    venue, away = ("KYO", "TOK") if target == SUN else ("TOK", "KYO")
    dated = {x.split()[1] if x.startswith("ALERT") else x.split()[0] for x in health_lines()}
    assert dated == {target.isoformat()}, f"health lines for {dated}, not {target}: {out}"
    assert marker_days("jra-day-done.txt") == [target.isoformat()]
    assert paths.odds_snapshot_path("jra", target.isoformat(), venue).exists(), out
    assert not paths.odds_snapshot_path("jra", other.isoformat(), away).exists()
    asked = set(site.links())
    assert world.odds_list(other, away) not in asked and world.card_list(other, away) not in asked
    if target == SUN:
        opened = first_probe_at_or_after(start, opens)
        race = world.race(SUN, "KYO", 1)
        times = snapshot_times(SUN, "KYO", race.race_id)
        assert_grain_until_t15(times, opened, race.post_utc, f"KYO 1 from a {hhmm(start)} start")


def test_a_probe_blocked_the_evening_before_blocks_the_race_date(monkeypatch, data_dir):
    world, before = evening_world()
    start = at(SAT, 19, 0)
    clock = Clock(start)
    site = clocked(world, before, clock.now, at(SAT, 18, 30), block_from=start)

    result, _ = log_day(monkeypatch, clock, site, "--unattended")
    assert result.exit_code in (0, 2), plain(result.output)
    # #387: the probe refused (403), then once more after each of the on_block back-off's two
    # pauses (1 req/s, then the floor only); the third block ends the day
    assert len(site.seen) == 3, f"{len(site.seen)} requests, not the three refused (403)"
    lines = block_lines()
    assert lines and all(x == [dbl.HOST, SUN.isoformat()] for x in lines), lines
    assert SUN.isoformat() not in marker_days("jra-no-meeting.txt") + marker_days(
        "jra-day-done.txt"
    ), "a block is not a day without racing, nor a finished day"

    for later in (at(SAT, 19, 30), at(SAT, 23, 30), at(SUN, 9, 0)):
        clock = Clock(later)
        site = clocked(world, before, clock.now, at(SAT, 18, 30))
        result, waited = log_day(monkeypatch, clock, site, "--unattended")
        got = (result.exit_code, len(site.seen), waited)
        assert got == (0, 0, 0.0), f"a start at {hhmm(later)} after the block: {got}"


def test_jra_and_nar_each_log_their_own_race_date(monkeypatch, data_dir, late_posts):
    """Saturday 19:00 JST: NAR's Saturday is still racing (a Sonoda nighter), JRA's is over and
    Sunday's sales open at 18:30: one start logs NAR's Saturday and JRA's Sunday."""
    world, before = evening_world()
    nar_world = NarWorld({SAT: [NarMeeting("SON", ("19:40", "20:10"), nighter=True)]})
    clock = ThreadClock(at(SAT, 19, 0), hours=30)
    jra_site = clocked(world, before, clock.now, at(SAT, 18, 30))
    nar_site = nd.NarSite(nar_world, clock=clock.now)

    def make_fetcher(*a, **k):
        source = k.get("source", a[0] if a else "jra")
        if source == "jra":
            return dbl.make_fetcher(jra_site)
        return nd.make_fetcher(nar_site, requests_per_second=k.get("requests_per_second"),
                               clock=clock.now, sleep=clock.sleep)  # fmt: skip

    mod = importlib.import_module(MODULE)
    monkeypatch.setattr(mod, "utc_now", clock.now)
    monkeypatch.setattr(mod, "sleep", clock.sleep)
    monkeypatch.setattr(mod, "make_fetcher", make_fetcher)
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "log-day", "--sources", "jra,nar", "--unattended"])
    out = plain(result.output)
    assert clock.errors == [], clock.errors
    assert result.exit_code in (0, 2), out

    seen = set()
    for line in health_lines():
        m = LINE.match(line)
        assert m, f"a health line without its date and source: {line!r}"
        seen.add((m["date"], m["venue"], m["source"]))
    assert seen == {(SAT.isoformat(), "SON", "nar"), (SUN.isoformat(), "KYO", "jra")}, seen
    son = data_dir / "odds_snapshots" / "nar" / SAT.isoformat() / "SON.jsonl.gz"
    assert son.exists() and list(odds_log.read_rows(son)), "NAR's Saturday not logged"
    race = world.race(SUN, "KYO", 1)
    times = snapshot_times(SUN, "KYO", race.race_id)
    assert times and min(times) < at(SAT, 19, 5), "JRA's Sunday not logged from the evening"
    assert not paths.odds_snapshot_path("jra", SAT.isoformat(), "TOK").exists()
    assert marker_days("jra-day-done.txt") == [SUN.isoformat()]
    assert marker_days("nar-day-done.txt") == [SAT.isoformat()]
