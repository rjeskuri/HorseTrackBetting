"""JRA's result-publication time, logged per race (ticket #225): the first poll at which a race's
official result is on the site, in UTC, so same-day features stop relying on post + 30 min.

Contract pinned here (the coder follows it):

Which page counts (Ruling: a race's official result is published once the meeting's レース結果
race list links that race's result page: its row carries a link to /JRADB/accessS.html, the
results path. A row without one, or no row, is not published; the page's navigation link to the
レース結果 menu is in no race row and marks nothing. The real page's markup is unverified: a
`local` test against a real list captured during racing is a follow-up.)

- `horsetrack.countries.japan.jra.capture.published_results(fetcher, race_date, venue) -> set[int]`:
  the race numbers whose result the meeting's レース結果 race list links. Two requests: the
  レース結果 menu (POST /JRADB/accessS.html, cname pw01sli00/AF: the site's fixed entry code, like
  the 出馬表 and オッズ menus), then the meeting's race list, its link read off the menu under the
  race date's heading. NotListed when the menu does not list the meeting, WrongMeeting when the
  list's date line names another meeting (as for the other race lists).
- `horsetrack.countries.japan.odds_log.record_published(path, races, seen_utc)
  -> dict[int, datetime]` merges first-seen instants into the results sidecar at `path`: a race
  already there keeps its instant whatever a later call says; a new race gets `seen_utc`.
  Returns the sidecar's mapping after the write. Written atomically (no temporary file
  survives).
  `odds_log.read_published(path) -> dict[int, datetime]`: race number -> aware UTC instant;
  {} when there is no sidecar. The sidecar is `<VENUE>.results.json` beside the venue's odds log
  (`paths.odds_snapshot_path("jra", date, venue).with_name(f"{venue}.results.json")`).
- schedule.run_day with a `results` block in its config (`{every_s, until_min}`; logger.yaml ships
  one): the Site gains `published_results(venue) -> set[int]` and the Store
  `write_published(venue, races, seen_utc)`. For each venue with post times, run_day polls
  `published_results` from its first post (never before) to its last post + until_min (never
  after), at most once every `every_s`, and also after the venue's odds logging has ended
  (results come out after the last post + after_last_race_min); it stops polling a venue once
  every race with a post time has a logged instant. Each successful poll's races are written with
  `seen_utc` = now() at that poll. A poll that raises NotListed, WrongMeeting, FetchError or
  ValueError is retried at the next poll and never counts in VenueSummary.failures (the odds
  health line's); CircuitOpen ends the day as from any site call. Without a `results` block,
  run_day never calls published_results (every earlier caller's behaviour).
- log-day wires both: the JRA site's published_results through the same Fetcher, the odds log's
  results sidecar as the store.
- Ruling: logger.yaml's `results` has 0 < every_s <= 60 (minute resolution for the ledger's
  offset distribution) and until_min > 30 (it must see lags past the 30-minute proxy).
- `JRAAdapter().result_published_utc() -> dict[str, datetime]`: race_id
  (`jra:<YYYY-MM-DD>:<VENUE>:<NN>`) -> logged instant, from every results sidecar under the data
  dir's odds_snapshots/jra; the store build fills `races.result_published_utc` from it
  (tests/core/test_store_published.py).

The lower bound (review of PR #261: a run's first poll stamps every race already shown with that
poll's time, so a restart, a late start or a wake from sleep must not read as a measured lag):
- Each logged race also gets a lower bound, `unpublished_utc`: the instant of the last successful
  poll in the same run that did not show the race (the instant that poll's `seen_utc` would have
  been). A race shown on a run's first successful poll has none (None): a late sighting. A failed
  poll shows nothing and is never a bound.
- `record_published(path, races, seen_utc, unpublished_utc=None)`: every new race of the call
  gets `seen_utc` and `unpublished_utc` (None: late). Both are first-write: a race already in the
  sidecar keeps both whatever a later call (a later run) says. Returns the published mapping, as
  before. The sidecar gains `"unpublished_utc": {"<race>": ISO UTC}` holding the bounded races
  only.
- `odds_log.read_unpublished(path) -> dict[int, datetime | None]`: one entry per race of
  read_published, its bound as an aware UTC instant or None. A sidecar written before the bound
  (no `unpublished_utc` key) reads every race as None (late); {} when there is no sidecar.
- schedule's Store: `write_published(venue, races, seen_utc, unpublished_utc)`, the bound passed
  as the fourth argument or the keyword `unpublished_utc`.
- `JRAAdapter().result_unpublished_utc() -> dict[str, datetime]`: race_id -> bound, the bounded
  races of every results sidecar only; the store build fills `races.result_unpublished_utc`.

Everything is synthetic: the three tests/fixtures/synthetic/jra_results_*.html pages, the JRADB
double (jradb_double.py) and fakes. No HTTP, no real sleep.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

import httpx
import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import CircuitOpen, FetchError
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.jra import capture
from horsetrack.countries.japan.jra.capture import MeetingCapture, NotListed, WrongMeeting
from horsetrack.countries.japan.logger.schedule import run_day

# isort: split
import jradb_double as dbl
import jradb_pages as pages
from jradb_double import RESULTS_MENU, Meeting, World

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
SAT, SUN = date(2026, 11, 28), date(2026, 11, 29)
HOST = "www.jra.go.jp"
EPS = 1e-6
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic"
FIXTURE_LIST = (pages.RESULTS_PATH, "pw01srl10052026050720261128/5C")  # TOK, the menu's SAT link


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


def jst(h: int, m: int = 0, s: int = 0, day: date = SAT) -> datetime:
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=JST).astimezone(UTC)


def fixture_page(name: str) -> bytes:
    """A synthetic fixture (stored UTF-8) as the site sends it, cp932."""
    text = (FIXTURES / name).read_text(encoding="utf-8")
    assert text.startswith("<!-- synthetic:"), name
    return text.encode("cp932")


def serve(site_pages) -> tuple[dbl.JradbSite, dbl.RecordingFetcher]:
    site = dbl.JradbSite(dict(site_pages))
    return site, dbl.make_fetcher(site)


def sidecar(day: date, venue: str) -> Path:
    log = paths.odds_snapshot_path("jra", day.isoformat(), venue)
    return log.with_name(f"{venue}.results.json")


def first_seen(polls: list[tuple[datetime, frozenset[int]]]) -> dict[int, datetime]:
    """From first principles: race -> the earliest poll instant whose page showed it."""
    out: dict[int, datetime] = {}
    for t, shown in sorted(polls, key=lambda p: p[0]):
        for n in shown:
            out.setdefault(n, t)
    return out


def last_unshown(polls: list[tuple[datetime, frozenset[int]]]) -> dict[int, datetime | None]:
    """From first principles: race -> the latest of these (one run's successful) polls before its
    first sighting, a poll that did not show it; None when the first poll already showed it."""
    out: dict[int, datetime | None] = {}
    previous: datetime | None = None
    for t, shown in sorted(polls, key=lambda p: p[0]):
        for n in shown:
            if n not in out:
                out[n] = previous
        previous = t
    return out


# ---------------------------------------------------------------- the page: fixtures


@pytest.mark.parametrize(
    "list_fixture, expected",
    [("jra_results_race_list_early.html", {1, 2}), ("jra_results_race_list_later.html", {1, 2, 3})],
    ids=["early", "later"],
)
def test_published_results_reads_the_fixture_race_list(list_fixture, expected):
    menu = fixture_page("jra_results_menu.html")
    site, fetcher = serve({RESULTS_MENU: menu, FIXTURE_LIST: fixture_page(list_fixture)})
    with fetcher:
        got = capture.published_results(fetcher, SAT, "TOK")
    assert got == expected
    assert site.links() == [RESULTS_MENU, FIXTURE_LIST], "the menu, then the meeting's list"
    assert all(s.served for s in site.seen)
    assert all(s.method == "POST" for s in site.seen)


def test_polls_of_the_fixture_pages_log_each_race_once_at_its_first_appearance():
    """The logger's loop on the fixtures: poll, record at the poll's instant; three polls."""
    menu = fixture_page("jra_results_menu.html")
    polls = [
        (jst(10, 21, 40), "jra_results_race_list_early.html"),
        (jst(10, 52, 20), "jra_results_race_list_later.html"),
        (jst(11, 3, 0), "jra_results_race_list_later.html"),
    ]
    path = sidecar(SAT, "TOK")
    for t, name in polls:
        site, fetcher = serve({RESULTS_MENU: menu, FIXTURE_LIST: fixture_page(name)})
        with fetcher:
            shown = capture.published_results(fetcher, SAT, "TOK")
        odds_log.record_published(path, shown, t)
    assert odds_log.read_published(path) == {1: polls[0][0], 2: polls[0][0], 3: polls[1][0]}


# ---------------------------------------------------------------- the page: the double


@pytest.mark.parametrize(
    "tok, kyo",
    [(set(), {2}), ({1}, set()), ({1, 2, 5}, {1, 4}), (set(range(1, 7)), {1, 2, 3, 4})],
    ids=["none", "first", "gaps", "all"],
)
def test_published_results_is_the_linked_races_of_the_asked_meeting(tok, kyo):
    world = World(
        {SAT: [Meeting("TOK", 6), Meeting("KYO", 4, nichi=3)], SUN: [Meeting("TOK", 6, nichi=8)]},
        published={(SAT, "TOK"): tok, (SAT, "KYO"): kyo, (SUN, "TOK"): {1, 2, 3, 4, 5, 6}},
    )
    for venue, expected in (("TOK", tok), ("KYO", kyo)):
        site, fetcher = serve(world.pages)
        with fetcher:
            got = capture.published_results(fetcher, SAT, venue)
        assert got == expected, venue
        assert site.links() == [RESULTS_MENU, world.results_list(SAT, venue)]
        assert all(s.served for s in site.seen)


def test_published_results_reads_the_codes_off_the_pages():
    world = World({SAT: [Meeting("TOK", 5)]}, published={(SAT, "TOK"): {1, 3}})
    scrambled, _ = dbl.scramble(world.pages)
    site, fetcher = serve(scrambled)
    with fetcher:
        assert capture.published_results(fetcher, SAT, "TOK") == {1, 3}
    assert all(s.served for s in site.seen), "a request went to a code no page linked"


def test_a_meeting_the_results_menu_does_not_list_is_not_listed():
    world = World({SAT: [Meeting("TOK", 4)]})
    site, fetcher = serve(world.pages)
    with fetcher, pytest.raises(NotListed):
        capture.published_results(fetcher, SAT, "KYO")
    assert site.links() == [RESULTS_MENU]


def test_a_results_list_for_another_meeting_is_refused():
    world = World({SAT: [Meeting("TOK", 4)]}, published={(SAT, "TOK"): {1, 2}})
    link = world.results_list(SAT, "TOK")
    served = dict(world.pages)
    served[link] = dbl.relabel(served[link], SAT, dbl.meeting_name("TOK"), SUN, "5回東京8日")
    _, fetcher = serve(served)
    with fetcher, pytest.raises(WrongMeeting):
        capture.published_results(fetcher, SAT, "TOK")


# ---------------------------------------------------------------- the sidecar


@pytest.mark.parametrize(
    "polls",
    [
        [(jst(10, 12), [1]), (jst(10, 13), [1, 2]), (jst(10, 45), [1, 2, 3])],
        [(jst(11, 0), []), (jst(11, 1), [4, 2]), (jst(11, 2), [2, 4, 7]), (jst(11, 3), [7])],
        # a later call naming an earlier instant changes nothing already logged
        [(jst(12, 0), [5]), (jst(11, 0), [5, 6])],
    ],
    ids=["growing", "unordered", "earlier-later"],
)
def test_record_published_keeps_the_first_instant_per_race(polls):
    path = sidecar(SAT, "KYO")
    expected: dict[int, datetime] = {}
    for t, races in polls:
        for n in races:
            expected.setdefault(n, t)
        returned = odds_log.record_published(path, races, t)
        assert returned == expected
    got = odds_log.read_published(path)
    assert got == expected
    assert all(t.utcoffset() == timedelta(0) for t in got.values()), "read back as UTC"
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name], "no temporary file"


def test_record_published_stores_the_instant_whatever_its_zone():
    path = sidecar(SAT, "TOK")
    in_tokyo = datetime(2026, 11, 28, 10, 41, 7, tzinfo=JST)
    odds_log.record_published(path, {3}, in_tokyo)
    (got,) = odds_log.read_published(path).values()
    assert got == in_tokyo and got.utcoffset() == timedelta(0)
    assert got == datetime(2026, 11, 28, 1, 41, 7, tzinfo=UTC)


def test_read_published_without_a_sidecar_is_empty():
    assert odds_log.read_published(sidecar(SAT, "HSN")) == {}


@pytest.mark.parametrize(
    "calls",
    [
        # (seen, races, unpublished): a run's first poll shows 1 (late), later polls bound 2 and 3
        [(jst(10, 12), [1], None), (jst(10, 13), [1, 2], jst(10, 12)),
         (jst(10, 45), [1, 2, 3], jst(10, 44, 20))],
        # a later run's bound changes neither instant of a late race nor of a bounded one
        [(jst(11, 0), [4], None), (jst(11, 1), [4, 5], jst(11, 0)),
         (jst(11, 30), [4, 5, 7], jst(11, 29, 20)), (jst(11, 50), [4, 5, 7], jst(11, 49))],
        # a later call that calls a bounded race late changes nothing either
        [(jst(12, 0), [5], jst(11, 59)), (jst(12, 30), [5, 6], None)],
    ],
    ids=["one-run", "later-run-with-bounds", "later-run-late"],
)  # fmt: skip
def test_record_published_keeps_each_races_first_bound(calls):
    path = sidecar(SAT, "KYO")
    seen: dict[int, datetime] = {}
    bound: dict[int, datetime | None] = {}
    for t, races, lower in calls:
        for n in races:
            if n not in seen:
                seen[n], bound[n] = t, lower
        assert odds_log.record_published(path, races, t, unpublished_utc=lower) == seen
    assert odds_log.read_published(path) == seen
    got = odds_log.read_unpublished(path)
    assert got == bound
    assert all(t is None or t.utcoffset() == timedelta(0) for t in got.values()), "UTC"
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name], "no temporary file"


def test_record_published_without_a_bound_is_a_late_sighting():
    path = sidecar(SAT, "TOK")
    odds_log.record_published(path, [2, 3], jst(10, 41, 7))
    assert odds_log.read_unpublished(path) == {2: None, 3: None}


def test_a_sidecar_written_before_the_bound_reads_as_late_sightings():
    path = sidecar(SAT, "TOK")
    path.parent.mkdir(parents=True)
    old = {1: jst(10, 9, 20), 4: jst(11, 41)}
    document = {"published_utc": {str(n): t.isoformat() for n, t in old.items()}}
    path.write_text(json.dumps(document), encoding="utf-8")
    assert odds_log.read_published(path) == old
    assert odds_log.read_unpublished(path) == {1: None, 4: None}
    odds_log.record_published(path, [1, 4, 6], jst(12, 0), unpublished_utc=jst(11, 59))
    assert odds_log.read_published(path) == {**old, 6: jst(12, 0)}
    assert odds_log.read_unpublished(path) == {1: None, 4: None, 6: jst(11, 59)}


def test_read_unpublished_without_a_sidecar_is_empty():
    assert odds_log.read_unpublished(sidecar(SAT, "HSN")) == {}


# ---------------------------------------------------------------- run_day with fakes

CONFIG = {
    "pools": ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"],
    "min_poll_interval_s": 20,
    "windows": [{"from": "07:00", "every_s": 300}, {"from": "09:30", "every_s": 20}],
    "stop": "16:45",
    "after_last_race_min": 10,
    "hot_races": {"within_min": 20, "after_post_min": 5, "all_every_s": 300},
    "requests": {"navigation": 2, "per_race": 3},
    "max_wrong_meeting": 3,
    "on_block": {"cooldown_min": None},
    "results": {"every_s": 60, "until_min": 40},
}
ROUND_S = 20  # every_s of the window in force while results come out

POSTS = {
    "TOK": {1: jst(10, 0), 2: jst(10, 30), 3: jst(11, 0), 4: jst(11, 30)},
    "KYO": {1: jst(10, 10), 2: jst(10, 40), 3: jst(11, 10)},
}
# When each result comes out (None: never, a stewards' inquiry past the day). The last race of each
# venue comes out after its odds logging ends (last post + after_last_race_min).
APPEAR = {
    "TOK": {1: jst(10, 9), 2: jst(10, 44, 30), 3: None, 4: jst(11, 47, 10)},
    "KYO": {1: jst(10, 21, 5), 2: jst(10, 46), 3: jst(11, 35)},
}


class Runaway(BaseException):
    """run_day kept going where a day cannot."""


class Clock:
    def __init__(self, start: datetime):
        self.t = start
        self.limit = start + timedelta(hours=20)

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += timedelta(seconds=max(float(seconds), 0.0))
        if self.t > self.limit:
            raise Runaway(f"run_day still sleeping at {self.t.isoformat()}")


@dataclass
class Poll:
    t: datetime
    venue: str
    shown: frozenset[int] | None  # None: the poll raised


Fail = Callable[[datetime, str, int], BaseException | None]  # (t, venue, nth poll) -> error


class FakeSite:
    host = HOST
    requests_per_second = 1.0

    def __init__(self, clock: Clock, *, fail: Fail | None = None):
        self.clock, self.fail = clock, fail
        self.polls: list[Poll] = []
        self.calls: list[tuple[datetime, str]] = []

    def discover(self) -> list[str]:
        self.calls.append((self.clock.t, "discover"))
        return list(POSTS)

    def post_times(self, venue: str) -> dict[int, datetime]:
        self.calls.append((self.clock.t, "post_times"))
        return dict(POSTS[venue])

    def capture(self, venue: str, races: list[int] | None) -> MeetingCapture:
        t = self.clock.t
        self.calls.append((t, "capture"))
        numbers = races or sorted(POSTS[venue])
        rows = [
            {"race_id": f"jra:{SAT.isoformat()}:{venue}:{n:02d}", "pool": "WIN",
             "combination": "1", "odds": 2.5, "snapshot_utc": t}
            for n in numbers
        ]  # fmt: skip
        return MeetingCapture(rows, t, 2 + 3 * len(numbers))

    def published_results(self, venue: str) -> set[int]:
        t = self.clock.t
        self.calls.append((t, "published_results"))
        nth = 1 + sum(p.venue == venue for p in self.polls)
        error = self.fail(t, venue, nth) if self.fail else None
        if error is not None:
            self.polls.append(Poll(t, venue, None))
            raise error
        shown = frozenset(n for n, at in APPEAR[venue].items() if at is not None and at <= t)
        self.polls.append(Poll(t, venue, shown))
        return set(shown)

    def ok_polls(self, venue: str) -> list[tuple[datetime, frozenset[int]]]:
        return [(p.t, p.shown) for p in self.polls if p.venue == venue and p.shown is not None]


class FakeStore:
    def __init__(self):
        self.published: dict[str, dict[int, datetime]] = defaultdict(dict)
        self.unpublished: dict[str, dict[int, datetime | None]] = defaultdict(dict)

    def write(self, venue: str, cap: MeetingCapture) -> int:
        return len(cap.rows)

    def write_post_times(self, venue: str, times: dict[int, datetime]) -> None:
        pass

    def write_published(
        self, venue: str, races, seen_utc: datetime, unpublished_utc: datetime | None = None
    ) -> None:
        for n in races:  # first seen wins, its bound with it, as the sidecar does
            if int(n) not in self.published[venue]:
                self.published[venue][int(n)] = seen_utc
                self.unpublished[venue][int(n)] = unpublished_utc


class SidecarStore(FakeStore):
    """The fake store with the real results sidecar behind write_published."""

    def write_published(
        self, venue: str, races, seen_utc: datetime, unpublished_utc: datetime | None = None
    ) -> None:
        odds_log.record_published(sidecar(SAT, venue), races, seen_utc,
                                  unpublished_utc=unpublished_utc)  # fmt: skip


def config(**results) -> dict:
    cfg = copy.deepcopy(CONFIG)
    cfg["results"].update(results)
    return cfg


def day(cfg: dict, tmp_path: Path, fail: Fail | None = None):
    clock = Clock(jst(6, 0))
    site, store = FakeSite(clock, fail=fail), FakeStore()
    summary = run_day(SAT, site, store, config=cfg, now=clock.now, sleep=clock.sleep,
                      block_marker=tmp_path / "jra-block.txt")  # fmt: skip
    return summary, site, store


@pytest.mark.parametrize("every_s", [60, 120])
def test_run_day_logs_each_race_at_the_first_poll_that_shows_its_result(tmp_path, every_s):
    cfg = config(every_s=every_s)
    summary, site, store = day(cfg, tmp_path)
    for venue in POSTS:
        polls = site.ok_polls(venue)
        expected = first_seen(polls)
        appeared = {n for n, at in APPEAR[venue].items() if at is not None}
        assert set(expected) == appeared, f"{venue}: a result that came out was never polled"
        assert dict(store.published[venue]) == expected, venue
        for n, seen in expected.items():  # within one poll interval (plus a round) of coming out
            lag = (seen - APPEAR[venue][n]).total_seconds()
            assert 0 <= lag <= every_s + ROUND_S + EPS, (venue, n, lag)
        times = [t for t, _ in polls]
        first, last = min(POSTS[venue].values()), max(POSTS[venue].values())
        until = last + timedelta(minutes=cfg["results"]["until_min"])
        assert all(first <= t <= until for t in times), f"{venue}: a poll outside first post..until"
        gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:], strict=False)]
        assert all(g >= every_s - EPS for g in gaps), f"{venue}: polls {min(gaps)} s apart"
        assert summary.venues[venue].stopped == "ended"
        assert summary.venues[venue].failures == 0
    # TOK 3 never comes out: TOK is polled past its odds end, KYO not after its last result
    kyo_done = max(store.published["KYO"].values())
    assert [t for t, _ in site.ok_polls("KYO") if t > kyo_done] == []
    tok_end = max(POSTS["TOK"].values()) + timedelta(minutes=CONFIG["after_last_race_min"])
    assert any(t > tok_end for t, _ in site.ok_polls("TOK"))
    assert 3 not in store.published["TOK"]


def test_failed_polls_are_retried_and_never_count_as_failures(tmp_path):
    def fail(t: datetime, venue: str, nth: int) -> BaseException | None:
        if venue == "KYO" and t < APPEAR["KYO"][1]:
            return NotListed("no KYO results on the menu yet")
        if venue == "TOK" and nth in (2, 3):
            return FetchError("timed out") if nth == 2 else ValueError("an error page")
        return None

    summary, site, store = day(config(), tmp_path, fail=fail)
    assert any(p.shown is None for p in site.polls if p.venue == "KYO")
    assert sum(p.shown is None for p in site.polls if p.venue == "TOK") == 2
    for venue in POSTS:
        assert dict(store.published[venue]) == first_seen(site.ok_polls(venue)), venue
        appeared = {n for n, at in APPEAR[venue].items() if at is not None}
        assert set(store.published[venue]) == appeared, venue
        assert summary.venues[venue].failures == 0, venue
        assert summary.venues[venue].snapshots > 0, venue


def tok_fails_around_race_2(t: datetime, venue: str, nth: int) -> BaseException | None:
    """TOK's polls fail from 10:43 to 10:46: TOK 2 (out 10:44:30) is first seen after them."""
    if venue == "TOK" and jst(10, 43) <= t < jst(10, 46):
        return FetchError("timed out")
    return None


@pytest.mark.parametrize(
    "every_s, fail",
    [(60, None), (120, None), (60, tok_fails_around_race_2)],
    ids=["60s", "120s", "60s-failed-polls"],
)
def test_run_day_bounds_each_race_by_the_last_poll_that_did_not_show_it(tmp_path, every_s, fail):
    summary, site, store = day(config(every_s=every_s), tmp_path, fail=fail)
    for venue in POSTS:
        polls = site.ok_polls(venue)
        expected = last_unshown(polls)
        assert set(expected) == set(store.published[venue]), venue
        assert dict(store.unpublished[venue]) == expected, venue
        # the day starts at 06:00: the first poll (at first post) shows nothing, so none is late
        assert all(b is not None for b in expected.values()), venue
        for n, bound in expected.items():
            assert bound < APPEAR[venue][n] <= store.published[venue][n], (venue, n)
    if fail is not None:
        failed = [p.t for p in site.polls if p.venue == "TOK" and p.shown is None]
        assert failed, "no TOK poll failed"
        bound = store.unpublished["TOK"][2]
        assert bound < min(failed), "a failed poll was taken for a poll that did not show TOK 2"
        assert (store.published["TOK"][2] - bound).total_seconds() > every_s + ROUND_S


class Died(BaseException):
    """The run died mid-day (a crash, a kill, the lid closed)."""


def test_a_restart_records_results_out_during_the_gap_as_late_sightings(tmp_path):
    """launchd restarts a dead run: what came out while nothing ran is seen late, not measured."""
    dies_at, restart_at = jst(10, 30), jst(11, 5)
    gap = {(v, n) for v, races in APPEAR.items() for n, at in races.items()
           if at is not None and dies_at <= at <= restart_at}  # fmt: skip
    assert gap == {("TOK", 2), ("KYO", 2)}, "the fixture's gap"
    store = SidecarStore()

    first = Clock(jst(6, 0))

    def dying_sleep(seconds: float) -> None:
        if first.t >= dies_at:
            raise Died
        first.sleep(seconds)

    site1 = FakeSite(first)
    with pytest.raises(Died):
        run_day(SAT, site1, store, config=config(), now=first.now, sleep=dying_sleep,
                block_marker=tmp_path / "jra-block.txt")  # fmt: skip
    second = Clock(restart_at)
    site2 = FakeSite(second)
    run_day(SAT, site2, store, config=config(), now=second.now, sleep=second.sleep,
            block_marker=tmp_path / "jra-block.txt")  # fmt: skip

    for venue in POSTS:
        run1, run2 = site1.ok_polls(venue), site2.ok_polls(venue)
        assert run1 and run2, venue
        seen1, seen2 = first_seen(run1), first_seen(run2)
        bound1, bound2 = last_unshown(run1), last_unshown(run2)
        seen = {**seen2, **seen1}  # the first run's instants stand
        bound = {n: bound1[n] if n in seen1 else bound2[n] for n in seen}
        assert odds_log.read_published(sidecar(SAT, venue)) == seen, venue
        assert odds_log.read_unpublished(sidecar(SAT, venue)) == bound, venue
        restart_poll = min(t for t, _ in run2)
        for n, at in APPEAR[venue].items():
            if (venue, n) in gap:
                assert seen[n] == restart_poll, (venue, n)
                assert bound[n] is None, f"{venue} {n} came out during the gap: a late sighting"
            elif at is not None:
                assert bound[n] is not None and bound[n] < at <= seen[n], (venue, n)


def test_a_block_on_a_results_poll_ends_the_day(tmp_path):
    def fail(t: datetime, venue: str, nth: int) -> BaseException | None:
        return CircuitOpen("HTTP 403", http_status=403, host=HOST)

    summary, site, _ = day(config(), tmp_path, fail=fail)
    assert len(site.polls) == 1, f"{len(site.polls)} results polls; the first one is blocked"
    blocked = site.polls[0]
    assert [c for c in site.calls if c[0] > blocked.t] == [], "a site call after the block"
    assert summary.stopped == "circuit_open"
    assert {v.stopped for v in summary.venues.values()} == {"circuit_open"}
    text = (tmp_path / "jra-block.txt").read_text(encoding="utf-8")
    assert HOST in text and SAT.isoformat() in text


def test_logger_yaml_ships_a_results_block():
    results = load_country_yaml("japan", "logger").get("results")
    assert results, "logger.yaml has no results block"
    assert 0 < float(results["every_s"]) <= 60
    assert float(results["until_min"]) > 30


# ---------------------------------------------------------------- log-day end to end

LOG_DAY = "horsetrack.countries.japan.logger.log_day"
# minutes after post each result comes out (None: never)
LAGS = {
    ("TOK", 1): 8.0, ("TOK", 2): None, ("TOK", 3): 12.75, ("TOK", 4): 15.0,
    ("KYO", 1): 6.5, ("KYO", 2): 19.0, ("KYO", 3): 13.25,
}  # fmt: skip


@dataclass
class ResultsSite(dbl.JradbSite):
    """The double, its results race lists answering as of the fake clock."""

    clock: Clock | None = None
    lists: dict[tuple[str, str], str] = field(default_factory=dict)  # list link -> venue
    world: World | None = None
    appear: dict[tuple[str, int], datetime | None] = field(default_factory=dict)
    polls: list[tuple[datetime, str, frozenset[int]]] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = request.read().decode("ascii", errors="replace")
        cname = (parse_qs(body).get("cname") or [None])[0]
        venue = self.lists.get((request.url.path, cname or ""))
        if venue is not None:
            t = self.clock.t
            shown = frozenset(
                n for (v, n), at in self.appear.items() if v == venue and at is not None and at <= t
            )
            self.pages[(request.url.path, cname)] = self.world.results_page(SAT, venue, shown)
            self.polls.append((t, venue, shown))
        return super().handler(request)


@pytest.fixture
def late_posts(monkeypatch):
    """Every race of the double posts 20 minutes later (first race 09:50+): no T-20 gap alert."""
    original = dbl.make_race

    def make_race(day, venue, number):
        race = original(day, venue, number)
        return dataclasses.replace(race, post_minutes=race.post_minutes + 20)

    monkeypatch.setattr(dbl, "make_race", make_race)


def test_log_day_writes_each_venues_first_seen_publication_times(monkeypatch, late_posts):
    tier2 = load_country_yaml("japan", "logger").get("tier2") or {}
    world = World({SAT: [Meeting("TOK", 4), Meeting("KYO", 3, nichi=3)]},
                  tier2=tier2.get("pools", ()))  # fmt: skip
    clock = Clock(datetime(2026, 11, 27, 21, 0, tzinfo=UTC))  # 06:00 in Tokyo
    appear = {}
    for (venue, n), lag in LAGS.items():
        post = world.expected_post_times(SAT, venue)[n]
        appear[(venue, n)] = None if lag is None else post + timedelta(minutes=lag)
    site = ResultsSite(
        dict(world.pages), clock=clock, world=world, appear=appear,
        lists={world.results_list(SAT, v): v for v in ("TOK", "KYO")},
    )  # fmt: skip
    mod = importlib.import_module(LOG_DAY)
    monkeypatch.setattr(mod, "utc_now", clock.now)
    monkeypatch.setattr(mod, "sleep", clock.sleep)
    monkeypatch.setattr(mod, "make_fetcher", lambda *a, **k: dbl.make_fetcher(site))
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "log-day", "--date", SAT.isoformat(),
                                      "--venues", "TOK,KYO"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert all(s.served for s in site.seen), "a request for a page the double does not serve"
    every_s = float(load_country_yaml("japan", "logger")["results"]["every_s"])
    for venue in ("TOK", "KYO"):
        polls = [(t, shown) for t, v, shown in site.polls if v == venue]
        expected = first_seen(polls)
        appeared = {n for (v, n), at in appear.items() if v == venue and at is not None}
        assert set(expected) == appeared, f"{venue}: a result that came out was never polled"
        got = odds_log.read_published(sidecar(SAT, venue))
        assert got == expected, venue
        for n, seen in got.items():
            lag = (seen - appear[(venue, n)]).total_seconds()
            assert 0 <= lag <= every_s + ROUND_S + EPS, (venue, n, lag)
        bounds = odds_log.read_unpublished(sidecar(SAT, venue))
        assert bounds == last_unshown(polls), f"{venue}: log-day wrote no lower bound"
        assert all(b is not None and b < appear[(venue, n)] for n, b in bounds.items()), venue
    assert json.loads(sidecar(SAT, "TOK").read_text(encoding="utf-8")), "an empty sidecar"


# ---------------------------------------------------------------- the store's source


def test_jra_adapter_reads_every_logged_publication_time():
    from horsetrack.countries.japan.jra.adapter import JRAAdapter

    logged = {
        (SAT, "TOK"): {1: jst(10, 9, 20), 3: jst(11, 12, 0), 12: jst(16, 41, 40)},
        (SAT, "KYO"): {2: jst(10, 51, 0)},
        (SUN, "TOK"): {11: jst(15, 58, 20, day=SUN)},
    }
    for (d, venue), races in logged.items():
        path = sidecar(d, venue)
        for n, t in races.items():
            odds_log.record_published(path, [n], t)
        # the post-time sidecar beside it is not a publication time
        odds_log.write_post_times(path.with_name(f"{venue}.races.json"), {1: jst(10, 0, day=d)},
                                  jst(9, 0, day=d))  # fmt: skip
    expected = {
        f"jra:{d.isoformat()}:{venue}:{n:02d}": t
        for (d, venue), races in logged.items()
        for n, t in races.items()
    }
    assert JRAAdapter().result_published_utc() == expected


def test_jra_adapter_without_logged_days_has_no_publication_times():
    from horsetrack.countries.japan.jra.adapter import JRAAdapter

    assert dict(JRAAdapter().result_published_utc()) == {}
    assert dict(JRAAdapter().result_unpublished_utc()) == {}


def test_jra_adapter_reads_every_logged_lower_bound():
    from horsetrack.countries.japan.jra.adapter import JRAAdapter

    # (race, first seen, the last poll that did not show it or None: late)
    logged = {
        (SAT, "TOK"): [(1, jst(10, 9, 20), jst(10, 8, 20)), (3, jst(11, 12, 0), None),
                       (12, jst(16, 41, 40), jst(16, 40, 0))],
        (SAT, "KYO"): [(2, jst(10, 51, 0), None)],
        (SUN, "TOK"): [(11, jst(15, 58, 20, day=SUN), jst(15, 57, 20, day=SUN))],
    }  # fmt: skip
    for (d, venue), races in logged.items():
        for n, seen, bound in races:
            odds_log.record_published(sidecar(d, venue), [n], seen, unpublished_utc=bound)
    old = sidecar(SUN, "KYO")  # written before the bound: every race late
    old.write_text(json.dumps({"published_utc": {"4": jst(11, 2, day=SUN).isoformat()}}),
                   encoding="utf-8")  # fmt: skip
    expected = {
        f"jra:{d.isoformat()}:{venue}:{n:02d}": bound
        for (d, venue), races in logged.items()
        for n, _, bound in races
        if bound is not None
    }
    assert JRAAdapter().result_unpublished_utc() == expected
    assert f"jra:{SUN.isoformat()}:KYO:04" in JRAAdapter().result_published_utc()
