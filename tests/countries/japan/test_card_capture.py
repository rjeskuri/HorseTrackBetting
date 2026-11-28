"""Capturing each race's 出馬表 (JRADB pw01dde card page) on race day, with its fetch time
(ticket #426): the capture, the card log that keeps every capture, and log-day's wiring.

Contract pinned here (the coder follows it):

`horsetrack.countries.japan.jra.capture`:
- `CardCapture`, a dataclass with `races` (the `official_card.parse` "races" rows, one per race
  read), `runners` (its "racecard_runners" rows), `snapshot_utc` (aware UTC: `now()`, called
  once before the first request) and `requests` (int: the HTTP requests this capture sent).
- `capture_cards(fetcher, race_date, venue, *, races=None, now=utc_now) -> CardCapture`: the
  出馬表 menu (CARD_MENU), the meeting's card race list, then the card page each asked race's row
  links, in race-list order (`races` None: every race): 2 + races requests, through
  `fetcher.fetch`. Every request is a FetchTask like the other JRADB ones (source "jra", POST,
  form body cname=<code>, mutable); a card page's page type is "official_card". Each page is
  parsed with `official_card.parse`; every race and runner row also carries `fetched_utc` (that
  page's RawPage.fetched_utc) and `snapshot_utc` (the capture's). A meeting the menu does not
  list raises NotListed; a card page whose date line names another meeting, or whose race number
  is not the race asked for, raises WrongMeeting.

`horsetrack.countries.japan.logger.log_day.OddsLogStore.write_cards(venue, capture) -> int`:
appends the capture as one gzip member (odds_log.append_capture) to the card log
`<VENUE>.cards.jsonl.gz` beside the odds log (odds_snapshot_path's directory), one row per race
and per runner, each row the capture's row plus `table` ("races" or "racecard_runners"); returns
the rows written. A later capture never rewrites an earlier one: the file only grows.

log-day with logger.yaml's `cards` block (see test_card_schedule.py for the cadence): every
race's card is captured once in each stretch of `cards.grid_min` through the JRA site's
`capture_cards` and written with `write_cards`; the odds log and the health lines are what they
are without the block.

Pages are synthetic (jradb_double serves one card page per race) and go through the real Fetcher;
expected values come from the double's invented cards, never from the code under test. Fetch times
are scripted by a Fetcher whose pages carry the times this test gives them.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.countries.japan import odds_log

# isort: split
# The names under test get a block of their own: while they are missing, they are the import error
# a red run reports, and the order lints the same before and after they exist.
from horsetrack.countries.japan.jra.capture import (
    CardCapture,
    NotListed,
    WrongMeeting,
    capture_cards,
)
from horsetrack.countries.japan.logger.log_day import OddsLogStore

# isort: split
import jradb_double as dbl
from jradb_double import CARD_MENU, HOST, Meeting, World

UTC = timezone.utc
SAT, SUN = date(2026, 11, 28), date(2026, 11, 29)
T_SNAP = datetime(2026, 11, 27, 23, 41, 7, tzinfo=UTC)
T_FETCH = datetime(2026, 11, 27, 23, 52, 0, tzinfo=UTC)
MODULE = "horsetrack.countries.japan.logger.log_day"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch) -> Path:
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


def week() -> World:
    """Sat: Tokyo (3 races), Kyoto (7), Fukushima (1); Sun: Tokyo, the meeting's next day."""
    return World(
        {
            SAT: [Meeting("TOK", 3), Meeting("KYO", 7, nichi=3), Meeting("FKS", 1, 3, 1)],
            SUN: [Meeting("TOK", 4, nichi=8)],
        }
    )


class StampingFetcher(dbl.RecordingFetcher):
    """The double's Fetcher, its k-th page stamped T_FETCH + 7k s + (k^2 mod 5) s: a fetch time
    no clock of the code under test gives. `stamps` keeps (page type, cname, stamp) per fetch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stamps: list[tuple[str, str, datetime]] = []

    def fetch(self, task, **kwargs):
        page = super().fetch(task, **kwargs)
        k = len(self.stamps)
        at = T_FETCH + timedelta(seconds=7 * k + (k * k) % 5)
        cname = parse_qs(task.body or "")["cname"][0]
        self.stamps.append((task.page_type, cname, at))
        return dataclasses.replace(page, fetched_utc=at)

    def stamp_of(self, cname: str) -> list[datetime]:
        return [at for _, c, at in self.stamps if c == cname]


def serve(world_pages) -> tuple[dbl.JradbSite, StampingFetcher]:
    site = dbl.JradbSite(dict(world_pages))
    fetcher = StampingFetcher(
        dbl.SOURCE, http=dbl.HTTP, transport=site.transport, sleep=lambda _s: None
    )
    return site, fetcher


class Once:
    """now(): T_SNAP, noting how many requests had been sent at each call."""

    def __init__(self, site: dbl.JradbSite, at: datetime = T_SNAP):
        self.site, self.at, self.calls = site, at, []

    def __call__(self) -> datetime:
        self.calls.append(len(self.site.seen))
        return self.at


def race_id(day: date, venue: str, n: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{n:02d}"


def expected_field(
    world: World, day: date, venue: str, n: int, scratched: dict[int, str] | None = None
) -> list[tuple]:
    """(race_id, horse_no, horse_id, is_scratched) of the double's card, from its spec."""
    spec = world.card_spec(day, venue, n, scratched)
    return [
        (race_id(day, venue, n), r.number, r.horse_id, r.scratched is not None)
        for r in spec.runners
    ]


def field_of(runners) -> list[tuple]:
    return sorted((r["race_id"], r["horse_no"], r["horse_id"], r["is_scratched"]) for r in runners)


def instant(value) -> datetime:
    """A row's timestamp (an aware datetime, or the log's ISO text) as an aware datetime."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def cards_log(day: date, venue: str) -> Path:
    return paths.odds_snapshot_path("jra", day.isoformat(), venue).with_name(
        f"{venue}.cards.jsonl.gz"
    )


# ---------------------------------------------------------------- the capture


@pytest.mark.parametrize("venue, n_races", [("TOK", 3), ("KYO", 7), ("FKS", 1)])
def test_capture_cards_reads_the_menu_the_race_list_then_each_card(venue, n_races):
    world = week()
    scrambled, mapping = dbl.scramble(world.pages)  # codes only readable off the pages

    def new(link):
        return (link[0], mapping.get(link[1], link[1]))

    site, fetcher = serve(scrambled)
    with fetcher:
        cap = capture_cards(fetcher, SAT, venue, now=Once(site))
    assert isinstance(cap, CardCapture)
    cost = 2 + n_races
    assert cap.requests == cost == fetcher.requests_sent == len(site.seen)
    cards = [new(world.card_link(SAT, venue, n)) for n in range(1, n_races + 1)]
    assert site.links() == [CARD_MENU, new(world.card_list(SAT, venue)), *cards]
    assert all(s.served for s in site.seen), "a request for a page no page links"
    assert {(s.method, s.host) for s in site.seen} == {("POST", HOST)}
    card_tasks = fetcher.tasks[2:]
    assert [t.page_type for t in card_tasks] == ["official_card"] * n_races
    assert all(t.mutable and t.source == "jra" for t in fetcher.tasks)

    races = sorted((r["race_id"], r["field_size"], r["status"]) for r in cap.races)
    expected_races = sorted(
        (race_id(SAT, venue, n), len(world.card_spec(SAT, venue, n).runners), "declared")
        for n in range(1, n_races + 1)
    )
    assert races == expected_races
    expected = [row for n in range(1, n_races + 1) for row in expected_field(world, SAT, venue, n)]
    assert field_of(cap.runners) == sorted(expected)


@pytest.mark.parametrize("asked", [[2, 5], [7], [1, 3, 4], [6, 2]])
def test_capture_cards_reads_only_the_races_asked_in_race_list_order(asked):
    world = week()
    site, fetcher = serve(world.pages)
    with fetcher:
        cap = capture_cards(fetcher, SAT, "KYO", races=asked, now=Once(site))
    in_order = sorted(asked)
    assert cap.requests == 2 + len(asked) == len(site.seen)
    assert site.links()[2:] == [world.card_link(SAT, "KYO", n) for n in in_order]
    assert sorted(r["race_id"] for r in cap.races) == [race_id(SAT, "KYO", n) for n in in_order]
    expected = [row for n in in_order for row in expected_field(world, SAT, "KYO", n)]
    assert field_of(cap.runners) == sorted(expected)


def test_every_row_carries_its_own_card_page_fetch_time():
    world = week()
    site, fetcher = serve(world.pages)
    now = Once(site)
    with fetcher:
        cap = capture_cards(fetcher, SAT, "KYO", now=now)
    assert now.calls == [0], "now() is called once, before the first request"
    assert cap.snapshot_utc == T_SNAP
    stamps = {}
    for n in range(1, 8):
        (stamp,) = fetcher.stamp_of(world.card_link(SAT, "KYO", n)[1])
        stamps[race_id(SAT, "KYO", n)] = stamp
    assert len(set(stamps.values())) == 7  # every page its own fetch time
    assert len(cap.races) == 7
    for row in cap.races:
        assert instant(row["fetched_utc"]) == stamps[row["race_id"]], row["race_id"]
        assert instant(row["snapshot_utc"]) == T_SNAP
    assert len(cap.runners) == sum(len(world.card_spec(SAT, "KYO", n).runners) for n in range(1, 8))
    for row in cap.runners:
        assert instant(row["fetched_utc"]) == stamps[row["race_id"]], row
        assert instant(row["card_scraped_utc"]) == stamps[row["race_id"]], row
        assert instant(row["snapshot_utc"]) == T_SNAP


def test_a_meeting_the_card_menu_does_not_list_is_not_listed():
    world = week()
    site, fetcher = serve(world.pages)
    with fetcher, pytest.raises(NotListed):
        capture_cards(fetcher, SAT, "NAK", now=Once(site))
    assert site.links() == [CARD_MENU]


@pytest.mark.parametrize("wrong", ["another meeting", "another race"])
def test_a_card_page_for_another_meeting_or_race_is_refused(wrong):
    world = week()
    served = dict(world.pages)
    link = world.card_link(SAT, "TOK", 2)
    if wrong == "another meeting":
        served[link] = dbl.relabel(
            served[link], SAT, world.meeting(SAT, "TOK").name, SUN, world.meeting(SUN, "TOK").name
        )
    else:
        served[link] = world.card_page(SAT, "TOK", 3)
    site, fetcher = serve(served)
    with fetcher, pytest.raises(WrongMeeting):
        capture_cards(fetcher, SAT, "TOK", now=Once(site))


# ---------------------------------------------------------------- captures over time, kept


@pytest.mark.parametrize(
    "venue, number, states",
    [
        ("TOK", 1, [{}, {3: "取消"}]),
        ("KYO", 4, [{}, {2: "除外"}, {2: "除外", 5: "取消"}]),
        ("KYO", 7, [{}, {}, {1: "取消"}]),
        ("FKS", 1, [{4: "取消"}, {4: "取消", 2: "取消"}]),
    ],
    ids=["tok1-one-scratch", "kyo4-two-scratches", "kyo7-late-scratch", "fks1-already-out"],
)
def test_each_capture_of_a_race_is_kept_with_its_own_fetch_time_and_scratches(
    venue, number, states
):
    """The ticket's test: captures of one race, runners scratched between them, give one set of
    rows per capture, each with its page's fetch time and its own is_scratched, and a later
    capture never rewrites an earlier one."""
    world = week()
    site, fetcher = serve(world.pages)
    link = world.card_link(SAT, venue, number)
    store = OddsLogStore(SAT, lambda: T_SNAP)
    path = cards_log(SAT, venue)
    caps, kept, written = [], [], []
    with fetcher:
        for i, scratched in enumerate(states):
            site.pages[link] = world.card_page(SAT, venue, number, scratched)
            cap = capture_cards(
                fetcher, SAT, venue, races=[number], now=Once(site, T_SNAP + timedelta(minutes=i))
            )
            caps.append((cap, copy.deepcopy(cap.races), copy.deepcopy(cap.runners)))
            n_rows = store.write_cards(venue, cap)
            assert n_rows == len(cap.races) + len(cap.runners)
            written.append(n_rows)
            body = path.read_bytes()
            for earlier in kept:  # the file only grows: every earlier capture byte for byte
                assert body.startswith(earlier)
            kept.append(body)
    stamps = fetcher.stamp_of(link[1])
    assert len(stamps) == len(states) and len(set(stamps)) == len(states)

    # In memory: each capture its own rows, unchanged by the captures after it.
    for (cap, races, runners), scratched, stamp in zip(caps, states, stamps, strict=True):
        assert cap.races == races and cap.runners == runners
        assert field_of(cap.runners) == expected_field(world, SAT, venue, number, scratched)
        (race,) = cap.races
        field = len(world.card_spec(SAT, venue, number).runners)
        assert race["field_size"] == field - len(scratched)
        assert {instant(r["fetched_utc"]) for r in [race, *cap.runners]} == {stamp}

    # On disk: one member per capture, in order, each as it was captured.
    members = list(odds_log.iter_members(path))
    assert [len(m) for m in members] == written
    for member, scratched, stamp in zip(members, states, stamps, strict=True):
        assert Counter(r["table"] for r in member) == {
            "races": 1,
            "racecard_runners": len(world.card_spec(SAT, venue, number).runners),
        }
        runners = [r for r in member if r["table"] == "racecard_runners"]
        assert field_of(runners) == expected_field(world, SAT, venue, number, scratched)
        (race,) = [r for r in member if r["table"] == "races"]
        assert race["race_id"] == race_id(SAT, venue, number)
        assert race["field_size"] == len(runners) - len(scratched)
        assert {instant(r["fetched_utc"]) for r in member} == {stamp}
    # The odds log is not where cards go.
    assert not paths.odds_snapshot_path("jra", SAT.isoformat(), venue).exists()


def test_the_card_log_is_per_venue_and_keeps_both_venues_captures():
    world = week()
    site, fetcher = serve(world.pages)
    store = OddsLogStore(SAT, lambda: T_SNAP)
    with fetcher:
        for venue in ("TOK", "KYO", "TOK"):
            store.write_cards(venue, capture_cards(fetcher, SAT, venue, now=Once(site)))
    tok = list(odds_log.iter_members(cards_log(SAT, "TOK")))
    kyo = list(odds_log.iter_members(cards_log(SAT, "KYO")))
    assert len(tok) == 2 and len(kyo) == 1
    for venue, members, n_races in (("TOK", tok, 3), ("KYO", kyo, 7)):
        for member in members:
            runners = [r for r in member if r["table"] == "racecard_runners"]
            numbers = range(1, n_races + 1)
            expected = [r for n in numbers for r in expected_field(world, SAT, venue, n)]
            assert field_of(runners) == sorted(expected), venue
    tok_stamps = [{instant(r["fetched_utc"]) for r in m} for m in tok]
    assert not tok_stamps[0] & tok_stamps[1], "the second capture's rows carry its own fetch times"


# ---------------------------------------------------------------- log-day


class Clock:
    """now() is the fake time; sleep(s) moves it (never back) and really waits for nothing."""

    def __init__(self, start: datetime):
        self.t = start

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += timedelta(seconds=max(0.0, float(seconds)))


def stretch(post: datetime, t: datetime, grid: list[int]) -> int | None:
    """The grid value g whose stretch [post - g, post - next smaller g) holds `t` (the last to
    the post), or None."""
    values = sorted(set(grid), reverse=True)
    for i, g in enumerate(values):
        nxt = values[i + 1] if i + 1 < len(values) else 0
        if post - timedelta(minutes=g) <= t < post - timedelta(minutes=nxt):
            return g
    return None


def test_logger_yaml_ships_a_card_cadence_that_catches_a_late_scratch():
    cards = load_country_yaml("japan", "logger").get("cards")
    assert cards, "logger.yaml ships no cards block: log-day would capture no card"
    grid = cards["grid_min"]
    assert grid and all(isinstance(g, int) and g > 0 for g in grid) and len(set(grid)) == len(grid)
    assert min(grid) <= 10, "no capture close enough to the off to see a late scratch"
    assert max(grid) >= 60, "no capture well before the off"
    # what capture_cards costs: the 出馬表 menu and the race list, then a page per race
    assert cards["requests"] == {"navigation": 2, "per_race": 1}


def _log_day(monkeypatch, tmp_path: Path, name: str, world: World, cards: bool):
    """`horsetrack japan log-day` for SAT's TOK on the fake clock and the double, in a data
    directory of its own; logger.yaml as shipped, or without its cards block."""
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / name))
    (tmp_path / name).mkdir()
    mod = importlib.import_module(MODULE)
    site = dbl.JradbSite(dict(world.pages))
    clock = Clock(datetime(2026, 11, 27, 21, 0, tzinfo=UTC))  # 06:00 in Tokyo on the race day
    real = load_country_yaml

    def config(country, file, *args, **kwargs):
        loaded = real(country, file, *args, **kwargs)
        if file == "logger" and not cards:
            loaded = {k: v for k, v in copy.deepcopy(loaded).items() if k != "cards"}
        return loaded

    monkeypatch.setattr(mod, "load_country_yaml", config)
    monkeypatch.setattr(mod, "utc_now", clock.now)
    monkeypatch.setattr(mod, "sleep", clock.sleep)
    monkeypatch.setattr(mod, "make_fetcher", lambda *a, **k: dbl.make_fetcher(site))
    from horsetrack.cli import app

    args = ["japan", "log-day", "--date", SAT.isoformat(), "--venues", "TOK", "--sources", "jra"]
    result = CliRunner().invoke(app, args)
    log = paths.odds_snapshot_path("jra", SAT.isoformat(), "TOK")
    odds = Counter(
        (r["race_id"], r["pool"], r["combination"], r.get("odds"), r.get("snapshot_utc"))
        for r in odds_log.read_rows(log)
    )
    health = (paths.logs_dir("japan") / "health.log").read_text(encoding="utf-8")
    members = list(odds_log.iter_members(cards_log(SAT, "TOK")))
    return result, site, odds, health, members


def test_log_day_captures_each_card_on_its_grid_and_leaves_the_odds_as_they_were(
    monkeypatch, tmp_path
):
    original = dbl.make_race

    def make_race(day, venue, number):  # posts 20 minutes later: every T-20 in the 20 s window
        race = original(day, venue, number)
        return dataclasses.replace(race, post_minutes=race.post_minutes + 20)

    monkeypatch.setattr(dbl, "make_race", make_race)
    shipped = load_country_yaml("japan", "logger")
    world = World({SAT: [Meeting("TOK", 2)]}, tier2=shipped["tier2"]["pools"])
    grid = list(shipped["cards"]["grid_min"])

    off, off_site, off_odds, off_health, off_members = _log_day(
        monkeypatch, tmp_path, "without", world, cards=False
    )
    on, on_site, on_odds, on_health, on_members = _log_day(
        monkeypatch, tmp_path, "with", world, cards=True
    )
    assert off.exit_code == 0, off.output
    assert on.exit_code == 0, on.output
    assert off_members == [], "without the cards block no card is captured"

    # The odds log and the health line are the same with the card captures as without.
    assert off_odds and on_odds == off_odds
    assert on_health == off_health

    # Every request went to a page the double serves; the cards were read through it.
    assert on_site.seen and all(s.served for s in on_site.seen)
    asked = on_site.links()
    for n in (1, 2):
        assert world.card_link(SAT, "TOK", n) in asked, n

    # Each race's card once in each grid stretch reached, the one nearest the off included; a
    # capture's time is its snapshot (log-day's utc_now, the fake clock).
    assert on_members
    seen: Counter = Counter()
    for member in on_members:
        assert {r["table"] for r in member} <= {"races", "racecard_runners"}
        assert all(r.get("fetched_utc") for r in member), "a row without its fetch time"
        for race in [r for r in member if r["table"] == "races"]:
            n = int(race["race_id"].rsplit(":", 1)[1])
            post = world.race(SAT, "TOK", n).post_utc
            g = stretch(post, instant(race["snapshot_utc"]), grid)
            assert g is not None, f"race {n}'s card captured outside every grid stretch"
            seen[(n, g)] += 1
            rid = race["race_id"]
            runners = [
                r for r in member if r["table"] == "racecard_runners" and r["race_id"] == rid
            ]
            assert field_of(runners) == expected_field(world, SAT, "TOK", n)
    assert all(count == 1 for count in seen.values()), seen
    for n in (1, 2):
        assert (n, min(grid)) in seen, f"race {n}: no card captured in the last stretch"
