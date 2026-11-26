"""`horsetrack japan log-day` (ticket #25): the all-venue logger from one command, with a daily
health line and an alert when a venue logs nothing.

Contract pinned here (the coder follows it):

The command lives in `horsetrack.countries.japan.logger.log_day` and registers on the japan group as
`log-day`. Three module attributes are its seams, looked up at call time so tests can replace them:
- `utc_now() -> datetime`: the only clock (aware UTC). It gives the default date (today in
  Asia/Tokyo), and it is the `now` of run_day and of every capture (so snapshot_utc is its time).
- `sleep(seconds)`: the only wait, passed to run_day.
- `make_fetcher(...) -> Fetcher`: the ingest Fetcher every JRADB request goes through (any
  arguments; the tests' replacement ignores them and serves the JRADB double).

Since #358 the default date is tomorrow once today is over for the source (past its stop) and
the time is at or after tomorrow's day-before probe start (logger.yaml `day_before.probe_from`,
today); tier 1 holds bracket quinella (test_day_before_cli.py has the rest of #358).

`log-day [--date YYYY-MM-DD] [--venues KYO,TOK] [--pools WIN,...] [--dry-run]`:
- wires the JRA site (discover_venues, read_post_times, capture_meeting with the pools asked, the
  default being logger.yaml's), the odds log (odds_snapshot_path('jra', date, venue)) and run_day
  with logger.yaml as its config;
- log records go to logs_dir('japan')/log-day-<date>.log; at most about 10 lines on the terminal;
- `--dry-run` prints the time zone, the windows, the stop rule, the pools, the rate and the venues
  given, exits 0 and sends no request;
- at the end it appends to logs_dir('japan')/health.log one line per venue
  `<date> <venue> snapshots=N rows=N failures=N wrong_meeting=N max_gap_t20_s=N stopped=<reason>`
  and `ALERT <date> <venue> <reason>` (also on stderr) for a venue that logged 0 rows, stopped
  early, or whose max_gap_t20_s > 120;
- exit 0 clean, 2 with any ALERT.

Tier 2 (ticket #130, D3 as amended 2026-11-24: all JRA pools are in scope), with logger.yaml's
`tier2` block on:
- log-day also captures the tier-2 pools of logger.yaml (`--pools` picks tier 1's only) into the
  same odds log, one member per capture, a member holding tier-1 pools only or tier-2 (and WIN5)
  pools only; tier-2 rows carry the page's prices exactly, plus each pool's "__pool__" row.
- snapshots, rows and failures on the health line count tier 1 only (the #25 meaning); the
  line ends ` tier2_skipped=N`, N being the venue's grid points (race x grid_min value) with no
  tier-2 capture, each capture falling in one grid interval [post - g, post - next smaller g).
- on a WIN5 day (a WIN5 link on the オッズ menu) one venue's log holds the WIN5 "__pool__" row.
- only pages of tier 1 and of the configured tier-2 pools are asked for (the double serves those
  alone).

NAR (ticket #38), `log-day --sources jra,nar` (default: logger.yaml's `sources`):
- each source runs with its own site, Fetcher and schedule, all at once in one process (see
  test_schedule_sources.py). `make_fetcher` is called once per source run, with the source key
  ("jra" or "nar") as its first argument or as `source=`; with none it is JRA's.
- `--venues` may name venues of either source (jra.yaml or nar.yaml codes); a source none of whose
  venues is named sends nothing. Without `--venues` each source discovers its own.
- JRA's log paths are unchanged; NAR's snapshots go to
  `odds_snapshots/nar/<date>/<VENUE>.jsonl.gz` (odds_log format, `nar:` race ids).
- Every health line names its source: a venue line is the line above with ` source=<key>` as its
  last field (after tier2_skipped when present); an ALERT line is
  `ALERT <date> <venue> source=<key> <reasons>`. Nothing else in JRA's line changes.
- A block on the NAR host (403/429) stops NAR only: its venues stop as circuit_open (with an
  ALERT), a block line `<host> <date> ...` names nar.netkeiba.com, and JRA logs its day and
  writes no block line.
- NAR's block policy (fix round, review of PR #255; owner decision in
  docs/research/nar-data-access.md): the first block pauses NAR alone for logger.yaml's
  `nar.on_block.cooldown_min` (15), then NAR resumes once on a fresh Fetcher,
  `make_fetcher("nar", requests_per_second=<nar.on_block.resume_requests_per_second>)` (0.05:
  20 s between requests to nar.netkeiba.com); a second block stops NAR for the day as above, and
  a restart that day sends NAR nothing. JRA never pauses and its make_fetcher is never given a
  rate. Ruling: `make_fetcher(source="jra", *, requests_per_second=None)`; a rate replaces the
  rate of every host of that source's http block, jitter and User-Agent unchanged.
- An error escaping one source's day (a 500 on NAR's race list at discovery: a FetchError) stops
  that source only. The other source logs its day; its health lines and, unattended, its
  day-done marker are written; the failed source gets no day-done, no-meeting or block line. An
  `ERROR` line naming the source (`nar`) and the error goes to the terminal.
  Ruling: log-day then exits 1 (EXIT_ERROR), after writing everything else; 1 wins over 2.

The JRADB double (jradb_double.py) serves a synthetic week through the real Fetcher; its post times
are moved 20 minutes later here so every race's T-20 window falls in logger.yaml's 20 s window
(a race off before 09:50 has 5-minute rounds in its window, a true gap alert). Nothing goes to the
network, and nothing really sleeps: the clock is fake and sleep moves it.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.countries.japan import odds_log

# isort: split
import jradb_double as dbl
import nar_double as nd
from jradb_double import Meeting, World
from nar_double import NarMeeting, NarWorld
from thread_clock import ThreadClock

UTC = timezone.utc
SAT = date(2026, 11, 28)
# tier 1, every round: bracket quinella joined the four in #358
ALL_POOLS = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "BRACKET_QUINELLA"]
MODULE = "horsetrack.countries.japan.logger.log_day"
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
HEALTH = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<venue>\S+) snapshots=(?P<snapshots>\d+) rows=(?P<rows>\d+)"
    r" failures=(?P<failures>\d+) wrong_meeting=(?P<wrong_meeting>\d+)"
    r" max_gap_t20_s=(?P<gap>\S+) stopped=(?P<stopped>\S+)"
    r"(?: tier2_skipped=(?P<tier2_skipped>\d+))?"  # with logger.yaml's tier2 on (#130)
    r"(?: source=(?P<source>[a-z]+))?$"  # the line's source, last (#38)
)
WIN5_SAT = dbl.Win5Day(
    legs=(("TOK", 1), ("KYO", 1), ("NAK", 11), ("HSN", 10), ("TOK", 11)),
    tickets=4_321_987,
    carryover_yen=312_450_000,
)
ALERT = re.compile(r"^ALERT (?P<date>\d{4}-\d{2}-\d{2}) (?P<venue>\S+) (?P<reason>\S.*)$")
MAX_TERMINAL_LINES = 12  # "at most about 10"


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


def run(monkeypatch, clock: Clock, world: World, *args: str):
    """Invoke `horsetrack japan log-day *args` with the seams on the fake clock and the double.
    Returns (result, site)."""
    mod = importlib.import_module(MODULE)  # the module under test; missing = the red state
    site = dbl.JradbSite(dict(world.pages))
    monkeypatch.setattr(mod, "utc_now", clock.now)
    monkeypatch.setattr(mod, "sleep", clock.sleep)
    monkeypatch.setattr(mod, "make_fetcher", lambda *a, **k: dbl.make_fetcher(site))
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "log-day", *args])
    return result, site


def tier2_config() -> dict:
    """logger.yaml's shipped tier 2 (#130): pools, grid_min, requests."""
    tier2 = load_country_yaml("japan", "logger").get("tier2")
    assert tier2, "logger.yaml ships no tier2 block"
    return tier2


def plain(text: str) -> str:
    return ANSI.sub("", text)


def terminal_lines(result) -> list[str]:
    return [line for line in plain(result.output).splitlines() if line.strip()]


def health_file() -> Path:
    return paths.logs_dir("japan") / "health.log"


def read_health(day: str) -> tuple[dict[str, list[dict]], list[dict]]:
    """health.log's lines for `day`: ({venue: [fields, ...]}, [ALERT fields, ...])."""
    health: dict[str, list[dict]] = {}
    alerts = []
    for line in health_file().read_text(encoding="utf-8").splitlines():
        if line.startswith("ALERT"):
            m = ALERT.match(line)
            assert m, f"malformed ALERT line: {line!r}"
            if m["date"] == day:
                alerts.append(m.groupdict())
            continue
        m = HEALTH.match(line)
        assert m, f"malformed health line: {line!r}"
        if m["date"] == day:
            health.setdefault(m["venue"], []).append(m.groupdict())
    return health, alerts


def snapshots_of(
    race_date: date, venue: str, pools: frozenset[str] | set[str] = frozenset(ALL_POOLS)
) -> dict[str, list[datetime]]:
    """race_id -> the snapshot instants the venue's log holds for it in `pools` (tier 1's)."""
    path = paths.odds_snapshot_path("jra", race_date.isoformat(), venue)
    out: dict[str, set[datetime]] = {}
    for row in odds_log.read_rows(path):
        if row["pool"] in pools:
            out.setdefault(row["race_id"], set()).add(datetime.fromisoformat(row["snapshot_utc"]))
    return {rid: sorted(times) for rid, times in out.items()}


def grid_points_captured(times: list[datetime], post: datetime, grid: list[int]) -> int:
    """The grid points of a race (post) that `times` (its tier-2 captures) cover, from first
    principles: g's interval is [post - g, post - next smaller g), the last one up to the post.
    Asserts every capture falls in one interval and no interval holds two."""
    values = sorted(grid, reverse=True)
    hits = dict.fromkeys(values, 0)
    for t in times:
        inside = []
        for i, g in enumerate(values):
            nxt = values[i + 1] if i + 1 < len(values) else 0
            if post - timedelta(minutes=g) <= t < post - timedelta(minutes=nxt):
                inside.append(g)
        assert inside, f"a tier-2 capture at {t.isoformat()} is off the grid (post {post})"
        hits[inside[0]] += 1
    assert all(n <= 1 for n in hits.values()), f"a grid point captured twice: {hits}"
    return sum(hits.values())


def longest_gap(times: list[datetime], post: datetime, within_min: int = 20) -> float:
    """The #34 criterion from first principles: the longest gap between captures inside
    [post - within_min, post], the first counted from the window start."""
    start = post - timedelta(minutes=within_min)
    inside = [t for t in times if start <= t <= post]
    assert inside, "no capture inside T-20"
    edges = [start, *inside]
    return max((b - a).total_seconds() for a, b in zip(edges, edges[1:], strict=False))


# ---------------------------------------------------------------- the dry run


@pytest.mark.parametrize(
    "args, venues, pools",
    [
        (["--date", "2026-11-28", "--venues", "KYO,TOK"], ["KYO", "TOK"], ALL_POOLS),
        (
            ["--date", "2026-11-29", "--venues", "NAK", "--pools", "WIN,QUINELLA"],
            ["NAK"],
            ["WIN", "QUINELLA"],
        ),
    ],
    ids=["kyo-tok-all-pools", "nak-two-pools"],
)
def test_dry_run_prints_the_plan_and_sends_nothing(monkeypatch, data_dir, args, venues, pools):
    world = World({SAT: [Meeting("KYO", 2), Meeting("TOK", 2)]})
    clock = Clock(datetime(2026, 11, 27, 23, 0, tzinfo=UTC))
    result, site = run(monkeypatch, clock, world, "--dry-run", *args)

    assert result.exit_code == 0, plain(result.output)
    assert site.seen == [], "the dry run sent a request"
    assert not (data_dir / "odds_snapshots").exists()

    out = plain(result.output)
    lines = terminal_lines(result)
    assert 0 < len(lines) <= MAX_TERMINAL_LINES, out
    config = load_country_yaml("japan", "logger")
    assert "Asia/Tokyo" in out
    for window in config["windows"]:
        assert str(window["from"]) in out
        assert re.search(rf"(?<![\d.]){window['every_s']}(?![\d.])", out), window
    assert str(config["stop"]) in out
    assert re.search(rf"(?<![\d.]){config['after_last_race_min']}(?![\d.])", out)
    for venue in venues:
        assert venue in out
    for venue in {"KYO", "TOK", "NAK"} - set(venues):
        assert venue not in out
    for pool in pools:
        assert pool in out
    if "PLACE" not in pools:
        assert "PLACE" not in out  # neither PLACE nor QUINELLA_PLACE was asked for
    rps = load_country_yaml("japan", "jra")["http"]["hosts"][dbl.HOST]["requests_per_second"]
    number = (
        rf"(?<![\d.:-]){int(rps)}(\.0+)?(?![\d.:-])" if rps == int(rps) else re.escape(str(rps))
    )
    assert any(re.search(r"rate|req", line, re.I) and re.search(number, line) for line in lines), (
        f"no line gives the rate ({rps} requests a second): {out}"
    )


# ---------------------------------------------------------------- a run


@pytest.mark.parametrize(
    "venues, meetings, pools, win5",
    [
        ("TOK,KYO", [Meeting("TOK", 1), Meeting("KYO", 1, nichi=3)], None, WIN5_SAT),
        ("FKS", [Meeting("FKS", 1, 3, 1), Meeting("TOK", 1)], "WIN,QUINELLA", None),
    ],
    ids=["tok-kyo-all-pools", "fks-two-pools"],
)
def test_log_day_writes_one_health_line_per_venue(
    monkeypatch, data_dir, late_posts, venues, meetings, pools, win5
):
    tier2 = tier2_config()
    tier2_pools, grid = set(tier2["pools"]), list(tier2["grid_min"])
    # The double serves tier 1's pages and the configured tier-2 pools' pages, nothing else.
    world = World({SAT: meetings}, tier2=tier2["pools"], win5={SAT: win5} if win5 else None)
    health_file().parent.mkdir(parents=True, exist_ok=True)
    earlier = (
        "2026-11-21 TOK snapshots=3 rows=30 failures=0 wrong_meeting=0"
        " max_gap_t20_s=20 stopped=ended"
    )
    health_file().write_text(earlier + "\n", encoding="utf-8")
    clock = Clock(datetime(2026, 11, 27, 21, 0, tzinfo=UTC))  # 06:00 in Tokyo on the race day
    args = ["--date", SAT.isoformat(), "--venues", venues]
    if pools:
        args += ["--pools", pools]

    result, site = run(monkeypatch, clock, world, *args)

    out = plain(result.output)
    assert result.exit_code == 0, out
    assert len(terminal_lines(result)) <= MAX_TERMINAL_LINES, out
    assert "ALERT" not in out
    # Every request was for a page the double serves: tier 1's and the configured tier-2 pools'.
    assert site.seen and all(s.served for s in site.seen)
    assert (paths.logs_dir("japan") / f"log-day-{SAT.isoformat()}.log").is_file()
    assert health_file().read_text(encoding="utf-8").splitlines()[0] == earlier  # appended

    health, alerts = read_health(SAT.isoformat())
    asked = venues.split(",")
    assert alerts == []
    assert sorted(health) == sorted(asked)
    wanted_pools = set(pools.split(",")) if pools else set(ALL_POOLS)
    win5_rows = []
    for venue in asked:
        assert len(health[venue]) == 1, venue
        line = health[venue][0]
        log_path = paths.odds_snapshot_path("jra", SAT.isoformat(), venue)
        members = list(odds_log.iter_members(log_path))
        # One capture, one member, of one tier: tier 1's pools, or tier 2's (and WIN5).
        tier1_members, tier2_members = [], []
        for member in members:
            in_member = {row["pool"] for row in member}
            if in_member <= set(ALL_POOLS):
                tier1_members.append(member)
            else:
                assert in_member <= tier2_pools | {"WIN5"}, (venue, in_member)
                tier2_members.append(member)
        rows = [row for member in tier1_members for row in member]
        assert rows, venue
        assert {row["pool"] for row in rows} == wanted_pools, venue
        assert int(line["snapshots"]) == len(tier1_members), venue
        assert int(line["rows"]) == len(rows), venue
        assert (int(line["failures"]), int(line["wrong_meeting"])) == (0, 0), venue
        # The gap, from the log's own tier-1 snapshot times and the double's post times.
        snaps = snapshots_of(SAT, venue)
        races = [world.race(SAT, venue, n) for n in range(1, world.meeting(SAT, venue).races + 1)]
        gaps = [longest_gap(snaps[race.race_id], race.post_utc) for race in races]
        assert float(line["gap"]) == pytest.approx(max(gaps), abs=1.0), venue
        assert float(line["gap"]) <= 120, venue

        # Tier 2: every configured pool logged (whatever --pools said), each board as printed.
        rows2 = [row for member in tier2_members for row in member]
        win5_rows += [row for row in rows2 if row["pool"] == "WIN5"]
        rows2 = [row for row in rows2 if row["pool"] != "WIN5"]
        assert {row["pool"] for row in rows2} == tier2_pools, venue
        boards: dict[tuple, dict[str, float]] = {}
        totals: dict[tuple, list[float]] = {}
        for row in rows2:
            key = (row["race_id"], row["snapshot_utc"], row["pool"])
            if row["combination"] == "__pool__":
                totals.setdefault(key, []).append(row["investment"])
            else:
                boards.setdefault(key, {})[row["combination"]] = row["odds"]
        by_race = {race.race_id: race for race in races}
        for (race_id, _, pool), board in boards.items():
            race = by_race[race_id]
            assert board == dbl.exotic_expected(race, pool), (venue, race_id, pool)
        for key, invested in totals.items():
            race = by_race[key[0]]
            assert invested == [dbl.exotic_tickets(race, key[2]) * dbl.UNIT_STAKE], key
        assert set(totals) == set(boards), venue  # each board with its pool total
        captures = {(rid, at) for rid, at, _ in boards}
        for rid, at in captures:  # a tier-2 capture of a race takes all its pools at once
            assert {(rid, at, p) for p in tier2_pools} <= set(boards), (rid, at)
        # tier2_skipped, from first principles: the grid points the log has no capture for.
        assert line["tier2_skipped"] is not None, f"{venue}: no tier2_skipped with tier 2 on"
        captured = 0
        for race in races:
            times = sorted(
                datetime.fromisoformat(at) for rid, at in captures if rid == race.race_id
            )
            covered = grid_points_captured(times, race.post_utc, grid)
            assert covered > 0, f"{venue} race {race.number}: tier 2 never captured"
            captured += covered
        assert int(line["tier2_skipped"]) == len(races) * len(grid) - captured, venue
    for venue in {m.venue for m in meetings} - set(asked):
        assert not paths.odds_snapshot_path("jra", SAT.isoformat(), venue).exists()

    # WIN5: on a WIN5 day its page is read and its pool row logged; on any other day nothing.
    if win5 is None:
        assert win5_rows == []
    else:
        assert site.links().count(world.win5_links[SAT]) >= 1
        assert win5_rows, "a WIN5 day logged no WIN5 row"
        for row in win5_rows:
            assert row["race_id"] == f"jra:{SAT.isoformat()}:WIN5", row
            assert row["combination"] == "__pool__", row
            assert row["legs"] == win5.race_ids(SAT), row
            assert row["investment"] == win5.tickets * dbl.UNIT_STAKE, row
            assert row["carryover"] == win5.carryover_yen, row


@pytest.mark.parametrize(
    "venues, empty",
    [("TOK,NAK", {"NAK"}), ("HSN,TOK,SAP", {"HSN", "SAP"})],
    ids=["one-empty", "two-empty"],
)
def test_a_venue_that_logged_nothing_raises_an_alert_and_exit_2(
    monkeypatch, data_dir, late_posts, venues, empty
):
    meetings = [Meeting("TOK", 1), Meeting("KYO", 1, nichi=3)]  # only TOK of the venues asked
    world = World({SAT: meetings}, tier2=tier2_config()["pools"])
    clock = Clock(datetime(2026, 11, 27, 21, 0, tzinfo=UTC))

    result, site = run(monkeypatch, clock, world, "--date", SAT.isoformat(), "--venues", venues)

    assert result.exit_code == 2, plain(result.output)
    assert len(terminal_lines(result)) <= MAX_TERMINAL_LINES, plain(result.output)
    health, alerts = read_health(SAT.isoformat())
    asked = venues.split(",")
    assert sorted(health) == sorted(asked)
    for venue in asked:
        assert len(health[venue]) == 1, venue
        logged = int(health[venue][0]["rows"])
        assert (logged == 0) is (venue in empty), venue
    assert health["TOK"][0]["tier2_skipped"] is not None, "no tier2_skipped with tier 2 on"
    assert site.seen and all(s.served for s in site.seen)
    assert {a["venue"] for a in alerts} == empty
    stderr_alerts = [line for line in plain(result.stderr).splitlines() if "ALERT" in line]
    assert {word for line in stderr_alerts for word in line.split()} >= empty
    assert not any("TOK" in line.split() for line in stderr_alerts)
    # The details went to the day's log, not the terminal.
    day_log = paths.logs_dir("japan") / f"log-day-{SAT.isoformat()}.log"
    text = day_log.read_text(encoding="utf-8")
    assert all(venue in text for venue in empty)


@pytest.mark.parametrize(
    "now, today",
    [
        (datetime(2026, 11, 27, 16, 0, tzinfo=UTC), date(2026, 11, 28)),  # 01:00 in Tokyo
        # #358: 23:59:59 is past Friday's stop and after Saturday's day-before probe start
        # (17:00 Friday): the race date is tomorrow, Saturday
        (datetime(2026, 11, 27, 14, 59, 59, tzinfo=UTC), date(2026, 11, 28)),  # 23:59:59
        (datetime(2026, 11, 27, 7, 50, tzinfo=UTC), date(2026, 11, 27)),  # 16:50: before 17:00
        (datetime(2026, 11, 28, 15, 0, tzinfo=UTC), date(2026, 11, 29)),  # midnight in Tokyo
    ],
    ids=["01h00-next-day", "23h59-past-the-stop-tomorrow", "16h50-before-the-probe", "midnight"],
)
def test_the_default_date_is_today_in_tokyo(monkeypatch, data_dir, late_posts, now, today):
    # TOK races on the UTC date, the Tokyo date and the day after alike: only the date in the
    # health log tells which day was logged.
    around = sorted({now.date(), today, today + timedelta(days=1)})
    world = World({d: [Meeting("TOK", 1, nichi=i)] for i, d in enumerate(around, start=5)})

    result, _ = run(monkeypatch, Clock(now), world, "--venues", "TOK")

    assert result.exit_code in (0, 2), plain(result.output)
    lines = health_file().read_text(encoding="utf-8").splitlines()
    assert lines
    dates = {line.split()[1] if line.startswith("ALERT") else line.split()[0] for line in lines}
    assert dates == {today.isoformat()}
    assert (paths.logs_dir("japan") / f"log-day-{today.isoformat()}.log").is_file()


# ---------------------------------------------------------------- two sources (#38)

SOURCE_LINE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<venue>\S+) snapshots=(?P<snapshots>\d+) rows=(?P<rows>\d+)"
    r" failures=\d+ wrong_meeting=\d+ max_gap_t20_s=\S+ stopped=(?P<stopped>\S+)"
    r"(?: tier2_skipped=\d+)? source=(?P<source>[a-z]+)$"
)
SOURCE_ALERT = re.compile(
    r"^ALERT (?P<date>\d{4}-\d{2}-\d{2}) (?P<venue>\S+) source=(?P<source>[a-z]+) (?P<reason>\S.*)$"
)


def nar_saturday() -> NarWorld:
    """Urawa by day, and a Sonoda nighter racing on past JRA's 16:45 stop."""
    return NarWorld(
        {
            SAT: [
                NarMeeting("URA", ("13:00", "13:30")),
                NarMeeting("SON", ("16:50", "17:20"), nighter=True),
            ]
        }
    )


def run_with_nar(
    monkeypatch,
    world: World,
    nar_world: NarWorld,
    *args: str,
    refuse_from=None,
    status_for=None,
    made: list | None = None,
):
    """`horsetrack japan log-day *args` with JRA on the JRADB double and NAR on the NAR double, on
    a clock with one virtual time per thread (a source on its own thread lives its own day). The
    NAR site records each request's time (`Seen.at`); a NAR fetcher asked for a rate
    (`requests_per_second=`) spaces its requests at that rate on that clock. Every make_fetcher
    call is appended to `made` as (source, its keyword arguments). Returns (result, jra site,
    nar site)."""
    mod = importlib.import_module(MODULE)
    clock = ThreadClock(datetime(2026, 11, 27, 21, 0, tzinfo=UTC))  # 06:00 in Tokyo
    jra_site = dbl.JradbSite(dict(world.pages))
    nar_site = nd.NarSite(nar_world, refuse_from=refuse_from, status_for=status_for,
                          clock=clock.now)  # fmt: skip

    def make_fetcher(*a, **k):
        source = k.get("source", a[0] if a else "jra")
        assert source in ("jra", "nar"), f"make_fetcher for {source!r}"
        if made is not None:
            made.append((source, {key: v for key, v in k.items() if key != "source"}))
        if source == "jra":
            return dbl.make_fetcher(jra_site)
        rate = k.get("requests_per_second")
        return nd.make_fetcher(nar_site, requests_per_second=rate, clock=clock.now,
                               sleep=clock.sleep)  # fmt: skip

    monkeypatch.setattr(mod, "utc_now", clock.now)
    monkeypatch.setattr(mod, "sleep", clock.sleep)
    monkeypatch.setattr(mod, "make_fetcher", make_fetcher)
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "log-day", *args])
    assert clock.errors == [], clock.errors
    # a usage error exits 2 as an alert does: the health log tells the two apart
    assert health_file().is_file(), f"log-day wrote no health line: {plain(result.output)}"
    return result, jra_site, nar_site


def source_health(day: str) -> tuple[dict[str, list[dict]], list[dict]]:
    """health.log's lines for `day`, each required to name its source."""
    lines, alerts = {}, []
    for line in health_file().read_text(encoding="utf-8").splitlines():
        m = (SOURCE_ALERT if line.startswith("ALERT") else SOURCE_LINE).match(line)
        assert m, f"a health line that does not name its source as pinned: {line!r}"
        if m["date"] == day:
            if line.startswith("ALERT"):
                alerts.append(m.groupdict())
            else:
                lines.setdefault(m["venue"], []).append(m.groupdict())
    return lines, alerts


def nar_log(data_dir: Path, day: date, venue: str) -> Path:
    return data_dir / "odds_snapshots" / "nar" / day.isoformat() / f"{venue}.jsonl.gz"


@pytest.mark.parametrize(
    "args, jra_venues, nar_venues, empty",
    [
        (["--sources", "jra,nar"], {"TOK"}, {"URA", "SON"}, set()),
        (["--sources", "jra,nar", "--venues", "TOK,URA,FUN"], {"TOK"}, {"URA", "FUN"}, {"FUN"}),
        (["--sources", "nar"], set(), {"URA", "SON"}, set()),
    ],
    ids=["both-discovered", "both-venues-given", "nar-only"],
)
def test_health_lines_name_the_source(
    monkeypatch, data_dir, late_posts, args, jra_venues, nar_venues, empty
):
    world = World({SAT: [Meeting("TOK", 1)]}, tier2=tier2_config()["pools"])
    nar_world = nar_saturday()

    result, jra_site, nar_site = run_with_nar(
        monkeypatch, world, nar_world, "--date", SAT.isoformat(), *args
    )

    out = plain(result.output)
    # 2 with the venue that logged nothing; otherwise NAR's own gap alerts are not pinned here
    assert result.exit_code == 2 if empty else result.exit_code in (0, 2), out
    lines, alerts = source_health(SAT.isoformat())
    assert sorted(lines) == sorted(jra_venues | nar_venues), out
    for venue, got in lines.items():
        assert len(got) == 1, (venue, got)
        assert got[0]["source"] == ("jra" if venue in jra_venues else "nar"), got[0]
    for alert in alerts:
        assert alert["source"] == ("jra" if alert["venue"] in jra_venues else "nar"), alert
    assert {a["venue"] for a in alerts} >= empty
    assert not {a["venue"] for a in alerts} & jra_venues, alerts

    # JRA: its log where it always was, its line counting what that log holds.
    tok = paths.odds_snapshot_path("jra", SAT.isoformat(), "TOK")
    if jra_venues:
        assert jra_site.seen and all(s.served for s in jra_site.seen)
        rows = [r for r in odds_log.read_rows(tok) if r["pool"] in ALL_POOLS]
        assert rows and {r["race_id"] for r in rows} == {"jra:2026-11-28:TOK:01"}
        assert int(lines["TOK"][0]["rows"]) == len(rows)
    else:
        assert jra_site.seen == [], "a NAR-only log-day sent a request to JRA"
        assert not tok.exists()

    # NAR: odds_snapshots/nar/<date>/<VENUE>.jsonl.gz, each board as the double printed it.
    assert nar_site.seen and all(s.served for s in nar_site.seen), nar_site.seen[:3]
    assert {(s.method, s.host) for s in nar_site.seen} == {("GET", nd.HOST)}
    for venue in nar_venues - empty:
        path = nar_log(data_dir, SAT, venue)
        members = list(odds_log.iter_members(path))
        rows = [row for member in members for row in member]
        assert rows, f"{venue}: nothing logged at {path}"
        assert int(lines[venue][0]["snapshots"]) == len(members), venue
        assert int(lines[venue][0]["rows"]) == len(rows), venue
        races = {r.race_id: r for r in nar_world.races(SAT, venue)}
        boards: dict[tuple, set] = {}
        for row in rows:
            assert row["race_id"] in races, row["race_id"]
            key = (row["race_id"], row["snapshot_utc"], row["pool"])
            boards.setdefault(key, set()).add(
                (row["combination"], row.get("odds"), row.get("odds_hi"), row.get("investment"))
            )
        for (race_id, _, pool), board in boards.items():
            assert board == races[race_id].board(pool), (venue, race_id, pool)
        assert not paths.odds_snapshot_path("jra", SAT.isoformat(), venue).exists(), venue
    for venue in empty:
        assert int(lines[venue][0]["rows"]) == 0, venue
        assert not nar_log(data_dir, SAT, venue).exists(), venue


def nar_block_policy() -> dict:
    """logger.yaml's NAR on_block (pinned to 15 min and 0.05 in test_schedule_sources.py)."""
    return load_country_yaml("japan", "logger")["nar"]["on_block"]


def marked_lines() -> list[list[str]]:
    """The first two fields of every line of every marker file (health.log aside)."""
    return [
        line.split()[:2]
        for path in paths.logs_dir("japan").iterdir()
        if path.is_file() and path.name != "health.log" and not path.name.endswith(".log")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]


def nar_fetchers(made: list) -> list[dict]:
    """The keyword arguments of each make_fetcher call for NAR, in order; JRA's never had a rate."""
    for source, kwargs in made:
        if source == "jra":
            assert kwargs.get("requests_per_second") is None, f"JRA's fetcher given {kwargs}"
    assert [s for s, _ in made].count("jra") <= 1, made
    return [kwargs for source, kwargs in made if source == "nar"]


def test_a_block_on_the_nar_host_stops_nar_only(monkeypatch, data_dir, late_posts):
    """nar.netkeiba.com answers 403 from its 4th request on: the first block pauses NAR, the
    resume on a fresh Fetcher is refused too, and that second block stops NAR for the day."""
    world = World({SAT: [Meeting("TOK", 1)]}, tier2=tier2_config()["pools"])
    args = ["--date", SAT.isoformat(), "--sources", "jra,nar", "--venues", "TOK,URA,SON"]
    policy = nar_block_policy()
    made: list = []

    result, jra_site, nar_site = run_with_nar(
        monkeypatch, world, nar_saturday(), *args, refuse_from=4, made=made
    )

    out = plain(result.output)
    assert result.exit_code == 2, out
    lines, alerts = source_health(SAT.isoformat())
    assert sorted(lines) == ["SON", "TOK", "URA"], out
    tok = lines["TOK"][0]
    assert (tok["stopped"], tok["source"]) == ("ended", "jra"), tok
    assert int(tok["rows"]) > 0
    assert jra_site.seen and all(s.served for s in jra_site.seen)
    for venue in ("URA", "SON"):
        assert (lines[venue][0]["stopped"], lines[venue][0]["source"]) == ("circuit_open", "nar")
    assert {(a["venue"], a["source"]) for a in alerts} == {("URA", "nar"), ("SON", "nar")}

    # Two blocks, each a request that was sent: the first Fetcher's, then the fresh one's after
    # the pause (an open breaker refuses without sending, so a reused Fetcher sends nothing).
    statuses = [s.status for s in nar_site.seen]
    first = statuses.index(403)
    assert statuses[first:] == [403, 403], statuses
    pause = nar_site.seen[first + 1].at - nar_site.seen[first].at
    assert pause >= timedelta(minutes=policy["cooldown_min"]), pause
    fetchers = nar_fetchers(made)
    assert len(fetchers) == 2, f"NAR's fetchers: {fetchers}"
    assert fetchers[0].get("requests_per_second") in (None, 0.1), fetchers[0]
    assert fetchers[1].get("requests_per_second") == pytest.approx(
        policy["resume_requests_per_second"]
    ), fetchers[1]

    # The block is NAR's alone: a block line for its host and today, none for JRA's.
    day = SAT.isoformat()
    marked = marked_lines()
    assert [nd.HOST, day] in marked, "no block line for nar.netkeiba.com today"
    assert [dbl.HOST, day] not in marked, "a block on the NAR host was recorded for JRA's host"

    # A restart that day sends NAR nothing.
    again: list = []
    result, _, nar_site = run_with_nar(
        monkeypatch, world, nar_saturday(), "--date", day, "--sources", "nar",
        "--venues", "URA,SON", made=again,
    )  # fmt: skip
    assert nar_site.seen == [], f"NAR after its second block: {nar_site.seen[:3]}"
    assert len(nar_fetchers(again)) <= 1, again


def test_a_first_nar_block_pauses_nar_then_resumes_on_a_slower_fresh_fetcher(
    monkeypatch, data_dir, late_posts
):
    """nar.netkeiba.com answers 403 to its 4th request only: NAR pauses, resumes on a fresh
    Fetcher at the resume rate (20 s between requests) and logs the rest of its day; JRA is
    never affected."""
    world = World({SAT: [Meeting("TOK", 1)]}, tier2=tier2_config()["pools"])
    args = ["--date", SAT.isoformat(), "--sources", "jra,nar", "--venues", "TOK,URA,SON"]
    policy = nar_block_policy()
    made: list = []

    result, jra_site, nar_site = run_with_nar(
        monkeypatch, world, nar_saturday(), *args,
        status_for=lambda n, path: 403 if n == 4 else None, made=made,
    )  # fmt: skip

    out = plain(result.output)
    assert result.exit_code in (0, 2), out
    lines, alerts = source_health(SAT.isoformat())
    assert sorted(lines) == ["SON", "TOK", "URA"], out
    for venue in ("URA", "SON"):
        got = lines[venue][0]
        assert (got["stopped"], got["source"]) == ("ended", "nar"), got
        assert int(got["rows"]) > 0, got

    seen = nar_site.seen
    assert [s.status for s in seen].count(403) == 1 and seen[3].status == 403, seen[:5]
    after = seen[4:]
    assert after and all(s.served for s in after), after[:3]
    assert after[0].at - seen[3].at >= timedelta(minutes=policy["cooldown_min"]), after[0].at
    gap = 1.0 / policy["resume_requests_per_second"]
    gaps = [(b.at - a.at).total_seconds() for a, b in zip(after, after[1:], strict=False)]
    assert gaps and min(gaps) >= gap - 1e-3, f"{min(gaps)} s between NAR requests after resume"
    fetchers = nar_fetchers(made)
    assert len(fetchers) == 2, f"NAR's fetchers: {fetchers}"
    assert fetchers[1].get("requests_per_second") == pytest.approx(
        policy["resume_requests_per_second"]
    ), fetchers[1]

    # JRA: its day as ever, no alert, no block line.
    tok = lines["TOK"][0]
    assert (tok["stopped"], tok["source"]) == ("ended", "jra"), tok
    assert int(tok["rows"]) > 0
    assert all(s.served for s in jra_site.seen)
    assert not [a for a in alerts if a["source"] == "jra"], alerts
    assert [dbl.HOST, SAT.isoformat()] not in marked_lines()


def test_an_error_in_one_source_leaves_the_other_its_day_health_and_markers(
    monkeypatch, data_dir, late_posts
):
    """Unattended, both sources discovered: NAR's race list answers 500 at discovery (a
    FetchError run_day does not catch). JRA logs its day and writes its health line and day-done
    marker; NAR is reported as an ERROR naming it; log-day exits 1."""
    world = World({SAT: [Meeting("TOK", 1)]}, tier2=tier2_config()["pools"])
    args = ["--date", SAT.isoformat(), "--sources", "jra,nar", "--unattended"]

    result, jra_site, nar_site = run_with_nar(
        monkeypatch, world, nar_saturday(), *args,
        status_for=lambda n, path: 500 if path == nd.RACE_LIST_PATH and n == 1 else None,
    )  # fmt: skip

    out = plain(result.output)
    assert result.exit_code == 1, out
    errors = [line for line in out.splitlines() if line.startswith("ERROR")]
    assert any(re.search(r"\bnar\b", line) and "500" in line for line in errors), errors
    assert [s.status for s in nar_site.seen] == [500], "NAR sent requests after its error"

    lines, _ = source_health(SAT.isoformat())
    assert "TOK" in lines, out
    tok = lines["TOK"][0]
    assert (tok["stopped"], tok["source"]) == ("ended", "jra"), tok
    assert int(tok["rows"]) > 0
    assert jra_site.seen and all(s.served for s in jra_site.seen)

    day = SAT.isoformat()
    logs = paths.logs_dir("japan")

    def days_in(name: str) -> list[str]:
        path = logs / name
        if not path.is_file():
            return []
        return [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line]

    assert days_in("jra-day-done.txt") == [day], "JRA's finished day was not marked"
    assert day not in days_in("nar-day-done.txt"), "a failed NAR day was marked done"
    assert day not in days_in("nar-no-meeting.txt"), "a failed discovery is not 'no meeting'"
    marked = marked_lines()
    assert [nd.HOST, day] not in marked and [dbl.HOST, day] not in marked, "an error is no block"


@pytest.mark.parametrize(
    ("source", "kwargs", "host"),
    [("nar", {}, nd.HOST), ("nar", "resume", nd.HOST), ("jra", {}, dbl.HOST)],
    ids=["nar", "nar-resume", "jra"],
)
def test_make_fetcher_spaces_each_host_at_its_own_rate(monkeypatch, source, kwargs, host):
    """The real log_day.make_fetcher, its Fetcher on a fake monotonic clock and a recording
    transport: nar.netkeiba.com 10 s apart (nar.yaml's 0.1, jitter 0), 20 s on the resume
    Fetcher (logger.yaml's nar.on_block rate), JRA's host at jra.yaml's own rate; one User-Agent."""
    from horsetrack.core.ingest import fetcher as ingest
    from horsetrack.core.ingest.types import FetchTask

    if kwargs == "resume":
        kwargs = {"requests_per_second": nar_block_policy()["resume_requests_per_second"]}
    t = [1_000.0]
    sent: list[tuple[float, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((t[0], request.url.host, request.headers.get("user-agent", "")))
        return httpx.Response(200, content=b"<html><body>ok</body></html>")

    def advance(seconds: float) -> None:
        t[0] += max(0.0, float(seconds))

    original = ingest.Fetcher.__init__

    def init(self, src, *, http, **kw):
        kw.update(transport=httpx.MockTransport(handler), sleep=advance, clock=lambda: t[0])
        original(self, src, http=http, **kw)

    monkeypatch.setattr(ingest.Fetcher, "__init__", init)
    mod = importlib.import_module(MODULE)
    fetcher = mod.make_fetcher(source, **kwargs)
    with fetcher:
        assert fetcher.source == source
        for i in range(6):
            url = f"https://{host}/odds/?race_id=2026421128{i + 1:02d}"
            fetcher.request(FetchTask(source, "odds_b1", f"k{i}", url, 2026, mutable=True))
    assert [h for _, h, _ in sent] == [host] * 6
    assert {ua for *_, ua in sent} == {"horsetrack-tests/1.0"}, "the User-Agent changed"
    gaps = [b - a for (a, *_), (b, *_) in zip(sent, sent[1:], strict=False)]

    rates = load_country_yaml("japan", source)["http"]
    entry = rates["hosts"].get(host) or rates.get("default") or rates["hosts"]["default"]
    rps = float(kwargs.get("requests_per_second", entry["requests_per_second"]))
    if source == "nar":
        assert kwargs or rps == pytest.approx(0.1), entry  # the DoD's 10 s
        assert gaps == pytest.approx([1.0 / rps] * 5, abs=1e-9), gaps  # jitter 0: exactly
    else:
        jitter = float(entry.get("jitter", 0.0))
        lo, hi = (1.0 - jitter) / rps, (1.0 + jitter) / rps
        assert all(lo - 1e-9 <= g <= hi + 1e-9 for g in gaps), (gaps, lo, hi)
        assert max(gaps) < 10.0, "JRA's host spaced at NAR's rate"
