"""Capturing a NAR meeting from nar.netkeiba.com through the ingest fetcher (ticket #38), with the
JRA capture's signatures and refusals (`jra/capture.py`, tests in test_capture.py).

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.nar.capture` has `WrongMeeting` and `NotListed`, the JRA capture's
  classes or subclasses of them (so `schedule.run_day` treats a NAR refusal as a JRA one), and
  returns `jra.capture.MeetingCapture`-shaped results: `rows` (`nar.pages.odds_rows` dicts),
  `snapshot_utc` (aware UTC) and `requests` (the HTTP requests this capture sent, not the
  fetcher's running total).
- `discover_venues(fetcher, race_date) -> list[str]`: the nar.yaml codes of every meeting of the
  day, once each, from the day's race list alone (1 request); [] on a day without racing.
- `read_post_times(fetcher, race_date, venue) -> {race number: aware UTC}`: the race list
  (1 request); NotListed when the list has no meeting of that venue.
- `capture_meeting(fetcher, race_date, venue, pools, *, races=None, now=...) -> MeetingCapture`:
  the day's race list, then each asked race in race-list order, one request per distinct odds
  page its asked pools are on (`b1` WIN and PLACE, `b4` QUINELLA, `b5` QUINELLA_PLACE): 1 +
  pages x races requests. `now()` is read before the first request and is every row's
  snapshot_utc. NotListed when the race list has no meeting of `venue` (after 1 request). An odds
  page whose own race id is not the race asked for raises WrongMeeting. A page's own race is the
  one its race navigation marks active (`li.Active` in `div.RaceNumWrap` / `div.RaceKaisaiWrap`)
  and its `li.race_navi_*` menu links name, never "every `race_id=` on the page": a real odds
  page links to every race of its meeting and to races at the day's other venues (fix round,
  review of PR #255; the shape is `nar_pages.RaceNav`). A correct page with those links does not
  raise. A pool outside those four is a ValueError before any request.
- Requests: GET, source "nar", mutable, to https://nar.netkeiba.com:
  `/top/race_list_sub.html?kaisai_date=YYYYMMDD` and `/odds/?race_id=<12 digits>&type=<t>`, the
  race id read off the race list. Other query parameters are not pinned.

Ruling: a race whose odds page shows netkeiba's 予想オッズ (`nar.pages.NotOnSale`, before sales
open) gives no rows; the capture goes on with the other races and does not raise. A NAR card is
hours long and sales open race by race, so a refusal there would fail every morning capture.

Pages are synthetic (nar_pages.py) and served by nar_double.py through the real Fetcher. Expected
values come from the double's invented boards, never from the code under test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from bs4 import BeautifulSoup

from horsetrack.countries.japan.jra import capture as jra_capture

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a
# red run reports, and the order lints the same before and after it exists.
from horsetrack.countries.japan.nar.capture import (
    NotListed,
    WrongMeeting,
    capture_meeting,
    discover_venues,
    read_post_times,
)

# isort: split
import nar_double as nd
from nar_double import ALL_POOLS, HOST, NarMeeting, NarWorld

UTC = timezone.utc
WED, THU, FRI = date(2026, 11, 25), date(2026, 11, 26), date(2026, 11, 27)
T0 = datetime(2026, 11, 25, 2, 0, 15, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch) -> Path:
    """Raw pages under tmp_path; no real `.env` or environment variable can supply a setting."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("HORSETRACK_USER_AGENT", raising=False)
    return root


def week(before_sales=()) -> NarWorld:
    """Wed: Urawa (4 races, day) and Sonoda (3, nighter); Thu: Nagoya and Oi; Fri: no racing."""
    return NarWorld(
        {
            WED: [
                NarMeeting("URA", ("10:40", "11:10", "11:40", "12:10")),
                NarMeeting("SON", ("15:00", "15:30", "16:00"), nighter=True),
            ],
            THU: [
                NarMeeting("NGY", ("11:00", "11:30")),
                NarMeeting("OOI", ("15:10", "15:50", "20:10"), nighter=True),
            ],
        },
        before_sales=before_sales,
    )


def serve(world: NarWorld):
    site = nd.NarSite(world)
    return site, nd.make_fetcher(site)


class Ticks:
    """now(): T0, T0 + 1 s, ... one step per call, noting how many requests had been sent."""

    def __init__(self, site: nd.NarSite, start: datetime = T0):
        self.site, self.start, self.calls = site, start, []

    def __call__(self) -> datetime:
        self.calls.append(len(self.site.seen))
        return self.start + timedelta(seconds=len(self.calls) - 1)


def race_list_key(day: date) -> tuple:
    return ("race_list", f"{day:%Y%m%d}")


def odds_blocks(keys: list[tuple], start: int, size: int) -> list[set]:
    return [set(keys[i : i + size]) for i in range(start, len(keys), size)]


def linked(body: bytes) -> set[str]:
    """Every `race_id=` a page links to (the fixture's own check, not the code under test's)."""
    out: set[str] = set()
    for a in BeautifulSoup(body, "html.parser").find_all("a", href=True):
        out.update(parse_qs(urlsplit(str(a["href"])).query).get("race_id", []))
    return out


def assert_page_links_like_a_real_one(world: NarWorld, race: nd.NarRace, body: bytes) -> None:
    """The served page links, as real pages do, to every race of its meeting and to another
    venue's race, besides its own: a capture that passes over it reads the page's own race."""
    ids = linked(body)
    siblings = {r.netkeiba_id for r in world.races(race.day, race.venue)}
    assert race.netkeiba_id in ids and siblings <= ids, sorted(ids)
    other_venues = {i for i in ids if i[4:6] != race.netkeiba_id[4:6]}
    assert other_venues, f"no link to another venue's race: {sorted(ids)}"


# ---------------------------------------------------------------- navigation and rows


@pytest.mark.parametrize(
    ("venue", "pools", "races"),
    [
        ("URA", list(ALL_POOLS), None),
        ("SON", ["WIN"], None),
        ("URA", ["QUINELLA_PLACE", "PLACE"], [4, 2]),
        ("SON", ["QUINELLA"], [3, 1]),
        ("URA", ["PLACE", "WIN", "QUINELLA"], [3]),
    ],
    ids=["ura-all-pools", "son-win", "ura-wide-place-two-races", "son-quinella", "ura-one-race"],
)
def test_capture_meeting_navigates_and_returns_rows(venue, pools, races):
    world = week()
    site, fetcher = serve(world)
    every = world.races(WED, venue)
    asked = [r for r in every if races is None or r.number in races]  # race-list order
    types = nd.page_types(pools)
    for race in asked:  # the pages served link to sibling races and other venues
        for t in types:
            assert_page_links_like_a_real_one(world, race, world.page(race, t))
    with fetcher:
        ticks = Ticks(site)
        cap = capture_meeting(fetcher, WED, venue, pools, races=races, now=ticks)
        cost = 1 + len(types) * len(asked)
        assert cap.requests == cost
        assert fetcher.requests_sent == cost
        keys = site.keys()
        assert len(keys) == cost
        assert keys[0] == race_list_key(WED)
        expected_blocks = [{("odds", r.netkeiba_id, t) for t in types} for r in asked]
        assert odds_blocks(keys, 1, len(types)) == expected_blocks
        assert all(s.served for s in site.seen), "a request went to a page the site does not have"

        # The rows are the boards as printed, every pool total included; one snapshot for all.
        assert nd.rows_of(cap.rows) == sorted(
            (row for r in asked for row in r.expected(pools)), key=repr
        )
        assert {row["race_id"] for row in cap.rows} == {
            f"nar:2026-11-25:{venue}:{r.number:02d}" for r in asked
        }
        assert ticks.calls[0] == 0, "now() is read before the first request"
        assert cap.snapshot_utc == T0
        assert {row["snapshot_utc"] for row in cap.rows} == {T0}

        # A second capture through the same fetcher counts its own requests only.
        other = "SON" if venue == "URA" else "URA"
        later = T0 + timedelta(minutes=7)
        second = capture_meeting(fetcher, WED, other, ["WIN"], now=Ticks(site, later))
        n_other = len(world.races(WED, other))
        assert second.requests == 1 + n_other
        assert fetcher.requests_sent == cost + 1 + n_other
        assert {row["snapshot_utc"] for row in second.rows} == {later}
        assert nd.rows_of(second.rows) == sorted(
            (row for r in world.races(WED, other) for row in r.expected(["WIN"])), key=repr
        )

    for task, seen in zip(fetcher.tasks, site.seen, strict=True):
        assert (task.source, task.method, task.mutable) == ("nar", "GET", True)
        assert (seen.method, seen.host) == ("GET", HOST)
        assert task.url.startswith(f"https://{HOST}/")


def test_discover_venues_and_read_post_times_read_the_race_list():
    world = week()
    for day, venues in ((WED, ["URA", "SON"]), (THU, ["NGY", "OOI"]), (FRI, [])):
        site, fetcher = serve(world)
        with fetcher:
            got = discover_venues(fetcher, day)
        assert sorted(got) == sorted(venues), day
        assert len(got) == len(set(got)), day
        assert site.keys() == [race_list_key(day)], day

    for day, venue in ((WED, "SON"), (WED, "URA"), (THU, "OOI")):
        site, fetcher = serve(world)
        with fetcher:
            got = read_post_times(fetcher, day, venue)
        assert got == world.expected_post_times(day, venue), (day, venue)
        assert all(t.utcoffset() == timedelta(0) for t in got.values())
        assert site.keys() == [race_list_key(day)], (day, venue)


# ---------------------------------------------------------------- refusals


def _claims(world: NarWorld, day: date, venue: str, number: int) -> str:
    """The 12-digit id of another race a page claims to be (it need not race that day)."""
    if day in world.days and any(m.venue == venue for m in world.days[day]):
        return world.race(day, venue, number).netkeiba_id
    return nd.make_race(day, venue, number, "12:00").netkeiba_id


@pytest.mark.parametrize(
    ("venue", "race", "page_type", "claims", "links_asked"),
    [
        # another race of the meeting: its race nav links to the race asked for, not active
        ("URA", 2, "b1", (WED, "URA", 3), True),
        ("URA", 4, "b5", (WED, "URA", 1), True),
        # another venue's race, same day: its venue tabs link to the race asked for, not active
        ("URA", 1, "b5", (WED, "SON", 1), True),
        ("SON", 3, "b1", (WED, "URA", 3), True),
        ("SON", 2, "b4", (THU, "SON", 2), False),  # the same race number another day
        ("URA", 4, "b1", (date(2026, 11, 24), "URA", 4), False),  # the day before
    ],
    ids=[
        "other-race",
        "other-race-wide",
        "other-venue",
        "other-venue-tab",
        "next-day",
        "day-before",
    ],  # fmt: skip
)
def test_wrong_meeting_raises(venue, race, page_type, claims, links_asked):
    world = week()
    site, fetcher = serve(world)
    asked = world.race(WED, venue, race)
    other = _claims(world, *claims)
    assert other != asked.netkeiba_id
    # the page of `other`, as the site prints it: its own race active, its siblings and venues
    page = world.page(asked, page_type, as_id=other)
    assert other in linked(page)
    assert (asked.netkeiba_id in linked(page)) == links_asked, sorted(linked(page))
    site.replace[(asked.netkeiba_id, page_type)] = page
    with fetcher, pytest.raises(jra_capture.WrongMeeting) as caught:
        capture_meeting(fetcher, WED, venue, list(ALL_POOLS), now=Ticks(site))
    assert isinstance(caught.value, WrongMeeting)
    assert not isinstance(caught.value, jra_capture.NotListed)
    assert ("odds", asked.netkeiba_id, page_type) in site.keys()


@pytest.mark.parametrize(
    ("day", "venue"),
    [(WED, "FUN"), (WED, "NGY"), (THU, "URA"), (FRI, "SON")],
    ids=["not-racing", "races-thursday", "raced-wednesday", "no-racing-that-day"],
)
def test_a_meeting_the_race_list_does_not_list_raises_not_listed(day, venue):
    for call in (
        lambda f, t: capture_meeting(f, day, venue, list(ALL_POOLS), now=t),
        lambda f, t: read_post_times(f, day, venue),
    ):
        site, fetcher = serve(week())
        with fetcher, pytest.raises(jra_capture.NotListed) as caught:
            call(fetcher, Ticks(site))
        assert isinstance(caught.value, NotListed)
        assert not isinstance(caught.value, jra_capture.WrongMeeting)
        assert site.keys() == [race_list_key(day)]


@pytest.mark.parametrize(
    ("venue", "not_on_sale"),
    [("URA", {4}), ("URA", {1, 2}), ("SON", {1, 2, 3})],
    ids=["last-race", "first-two", "whole-card"],
)
def test_a_race_not_on_sale_yet_gives_no_rows(venue, not_on_sale):
    world = week(before_sales=[(WED, venue, n) for n in not_on_sale])
    site, fetcher = serve(world)
    with fetcher:
        cap = capture_meeting(fetcher, WED, venue, list(ALL_POOLS), now=Ticks(site))
    on_sale = [r for r in world.races(WED, venue) if r.number not in not_on_sale]
    assert nd.rows_of(cap.rows) == sorted(
        (row for r in on_sale for row in r.expected(ALL_POOLS)), key=repr
    )
    assert cap.requests == len(site.seen)


@pytest.mark.parametrize("pools", [["NOT_A_POOL"], ["WIN", "TIERCE"], ["PLACE", ""]])
def test_an_unknown_pool_is_a_value_error_before_any_request(pools):
    site, fetcher = serve(week())
    with fetcher, pytest.raises(ValueError):
        capture_meeting(fetcher, WED, "URA", pools, now=Ticks(site))
    assert site.seen == []
    assert fetcher.tasks == []
