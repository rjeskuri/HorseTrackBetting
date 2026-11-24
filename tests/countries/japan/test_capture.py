"""Capturing a JRA meeting through the ingest fetcher (ticket #23), fixing HKTrack bug (d): a page
of another meeting is no longer swallowed into an empty capture.

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.capture` has `WrongMeeting` and `NotListed` (exceptions, neither a
  subclass of the other) and `MeetingCapture` with attributes `rows` (the `official_pages.odds_rows`
  dicts), `snapshot_utc` (aware UTC datetime) and `requests` (int: the HTTP requests this capture
  sent, not the fetcher's running total).
- `discover_venues(fetcher, race_date) -> list[str]`: venue codes (TOK, KYO...) of every meeting
  under the day's heading of the 出馬表 menu, once each, from that menu alone (1 request).
- `read_post_times(fetcher, race_date, venue) -> {race number: aware UTC datetime}`: the 出馬表
  menu, then the meeting's card race list (2 requests).
- `capture_meeting(fetcher, race_date, venue, pools, *, races=None, now=...) -> MeetingCapture`:
  the オッズ menu, the meeting's race list, then each asked race in race-list order, one request
  per distinct odds page its asked pools are on. `now()` is called before the first request and
  its value is the snapshot of every row. A page dated for another meeting raises WrongMeeting;
  a meeting the menu does not list raises NotListed (after the menu, 1 request); a pool outside
  WIN, PLACE, QUINELLA, QUINELLA_PLACE is a ValueError before any request.
- Every JRADB request is a FetchTask of source "jra" to https://www.jra.go.jp<path>: POST, form
  body cname=<code>, mutable, page type official_menu, official_racelist, official_odds or
  official_card_racelist. Only the two entry codes are known in advance; every other code is
  read off a fetched page (the double's checksums are not derivable, and get scrambled).
- race_id is `jra:<YYYY-MM-DD>:<VENUE>:<NN>` (docs/specs/odds-log.md).

Pages are synthetic (jradb_pages.py) and served by jradb_double.py through the real Fetcher.
Expected values come from the double's invented boards, never from the code under test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs

import pytest

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a
# red run reports, and the order lints the same before and after it exists.
from horsetrack.countries.japan.jra.capture import (
    NotListed,
    WrongMeeting,
    capture_meeting,
    discover_venues,
    read_post_times,
)

# isort: split
import jradb_double as dbl
from jradb_double import CARD_MENU, HOST, KIND_POOLS, KINDS, ODDS_MENU, POOL_KIND, Meeting, World

UTC = timezone.utc
SAT, SUN, MON = date(2026, 11, 28), date(2026, 11, 29), date(2026, 11, 30)
ALL_POOLS = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"]
T0 = datetime(2026, 11, 28, 5, 0, 15, tzinfo=UTC)
FORM = "application/x-www-form-urlencoded"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch) -> Path:
    """Raw pages under tmp_path; no real `.env` or environment variable can supply a setting."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("HORSETRACK_USER_AGENT", raising=False)
    return root


def week(tok_races: int = 4) -> World:
    """Sat: Tokyo, Kyoto, Fukushima; Sun: Tokyo (next day of the meeting), Nakayama, Hanshin."""
    return World(
        {
            SAT: [Meeting("TOK", tok_races), Meeting("KYO", 7, nichi=3), Meeting("FKS", 3, 3, 1)],
            SUN: [Meeting("TOK", 5, nichi=8), Meeting("NAK", 6, 4, 2), Meeting("HSN", 2, 4, 6)],
        }
    )


def serve(world_pages) -> tuple[dbl.JradbSite, dbl.RecordingFetcher]:
    site = dbl.JradbSite(dict(world_pages))
    return site, dbl.make_fetcher(site)


class Ticks:
    """now(): T0, T0 + 1 s, ... one step per call, noting how many requests had been sent."""

    def __init__(self, site: dbl.JradbSite, start: datetime = T0):
        self.site, self.start, self.calls = site, start, []

    def __call__(self) -> datetime:
        self.calls.append(len(self.site.seen))
        return self.start + timedelta(seconds=len(self.calls) - 1)


def odds_blocks(links: list, start: int, size: int) -> list[set]:
    return [set(links[i : i + size]) for i in range(start, len(links), size)]


# ---------------------------------------------------------------- navigation and cost


@pytest.mark.parametrize("races", [1, 3, 12])
def test_capture_navigates_menu_list_then_odds_pages(races):
    world = week(tok_races=races)
    site, fetcher = serve(world.pages)
    with fetcher:
        cap = capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, now=Ticks(site))
        cost = 2 + 3 * races
        assert cap.requests == cost
        assert fetcher.requests_sent == cost
        links = site.links()
        assert len(links) == cost
        assert links[0] == ODDS_MENU
        assert links[1] == world.odds_list(SAT, "TOK")
        expected_blocks = [
            set(world.odds_links(SAT, "TOK", n).values()) for n in range(1, races + 1)
        ]
        assert odds_blocks(links, 2, 3) == expected_blocks
        assert all(s.served for s in site.seen), "a request went to a code no page linked"
        assert dbl.price_rows(cap.rows) == world.expected_rows(SAT, "TOK", ALL_POOLS)

        # A second capture through the same fetcher counts its own requests only.
        kyoto = capture_meeting(fetcher, SAT, "KYO", ALL_POOLS, now=Ticks(site))
        assert kyoto.requests == 2 + 3 * 7
        assert fetcher.requests_sent == cost + 2 + 3 * 7
        assert site.links()[cost : cost + 2] == [ODDS_MENU, world.odds_list(SAT, "KYO")]
        assert dbl.price_rows(kyoto.rows) == world.expected_rows(SAT, "KYO", ALL_POOLS)


def test_every_request_is_a_mutable_form_post_task():
    world = week(tok_races=2)
    site, fetcher = serve(world.pages)
    with fetcher:
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, now=Ticks(site))
        read_post_times(fetcher, SUN, "NAK")
    types = [t.page_type for t in fetcher.tasks]
    assert types == ["official_menu", "official_racelist"] + ["official_odds"] * 6 + [
        "official_menu",
        "official_card_racelist",
    ]
    assert len(fetcher.tasks) == len(site.seen)
    for task, seen in zip(fetcher.tasks, site.seen, strict=True):
        assert (task.source, task.method, task.mutable, task.year) == ("jra", "POST", True, 2026)
        assert task.url == f"https://{HOST}{seen.path}"
        assert parse_qs(task.body or "") == {"cname": [seen.cname]}
        assert (seen.method, seen.host) == ("POST", HOST)
        assert (seen.content_type or "").startswith(FORM)
    assert site.links()[-2:] == [CARD_MENU, world.card_list(SUN, "NAK")]


def test_rows_of_one_capture_share_one_snapshot_utc():
    world = week(tok_races=3)
    site, fetcher = serve(world.pages)
    with fetcher:
        ticks = Ticks(site)
        cap = capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, now=ticks)
        later = datetime(2026, 11, 28, 5, 7, 42, tzinfo=UTC)
        second = capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, now=Ticks(site, later))
    assert ticks.calls[0] == 0, "now() is read before the first request"
    assert dbl.price_rows(cap.rows) == world.expected_rows(SAT, "TOK", ALL_POOLS)
    assert cap.snapshot_utc == T0
    assert {r["snapshot_utc"] for r in cap.rows} == {T0}
    assert second.snapshot_utc == later
    assert {r["snapshot_utc"] for r in second.rows} == {later}


@pytest.mark.parametrize("races", [[1], [3, 6], [6, 2, 4]])
def test_only_the_asked_races_are_requested(races):
    world = week(tok_races=6)
    site, fetcher = serve(world.pages)
    with fetcher:
        cap = capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, races=races, now=Ticks(site))
    assert cap.requests == 2 + 3 * len(races)
    links = site.links()
    assert links[:2] == [ODDS_MENU, world.odds_list(SAT, "TOK")]
    asked = sorted(races)  # race-list order
    expected_blocks = [set(world.odds_links(SAT, "TOK", n).values()) for n in asked]
    assert odds_blocks(links, 2, 3) == expected_blocks
    assert dbl.price_rows(cap.rows) == world.expected_rows(SAT, "TOK", ALL_POOLS, races)
    assert {r["race_id"] for r in cap.rows} == {f"jra:2026-11-28:TOK:{n:02d}" for n in races}


@pytest.mark.parametrize(
    "pools",
    [
        ["WIN"],
        ["PLACE"],
        ["QUINELLA"],
        ["QUINELLA_PLACE"],
        ["WIN", "PLACE"],
        ["QUINELLA_PLACE", "WIN"],
        ["QUINELLA", "QUINELLA_PLACE", "PLACE"],
    ],
)
def test_only_the_asked_pools_are_fetched(pools):
    races = 4
    world = week(tok_races=races)
    site, fetcher = serve(world.pages)
    with fetcher:
        cap = capture_meeting(fetcher, SAT, "TOK", pools, now=Ticks(site))
    kinds = [k for k in KINDS if any(POOL_KIND[p] == k for p in pools)]
    assert all(set(KIND_POOLS[k]) & set(pools) for k in kinds)
    assert cap.requests == 2 + len(kinds) * races
    links = site.links()
    expected_blocks = [
        {world.odds_link(SAT, "TOK", n, k) for k in kinds} for n in range(1, races + 1)
    ]
    assert odds_blocks(links, 2, len(kinds)) == expected_blocks
    assert dbl.price_rows(cap.rows) == world.expected_rows(SAT, "TOK", pools)
    assert {r["pool"] for r in cap.rows} == set(pools)


@pytest.mark.parametrize("pools", [["NOT_A_POOL"], ["WIN", "NOT_A_POOL"], ["PLACE", ""]])
def test_an_unknown_pool_is_a_value_error_before_any_request(pools):
    site, fetcher = serve(week().pages)
    with fetcher, pytest.raises(ValueError):
        capture_meeting(fetcher, SAT, "TOK", pools, now=Ticks(site))
    assert site.seen == []
    assert fetcher.tasks == []


# ---------------------------------------------------------------- refusals (bug d)

# (day, meeting name) a page claims instead of Saturday's 5回東京7日
OTHER_MEETINGS = {
    "other_day": (SUN, "5回東京8日"),
    "other_venue": (SAT, "5回京都3日"),
    "other_day_and_venue": (SUN, "4回中山2日"),
}


@pytest.mark.parametrize("claims", list(OTHER_MEETINGS))
def test_a_race_list_of_another_meeting_raises_wrong_meeting(claims):
    world = week(tok_races=3)
    link = world.odds_list(SAT, "TOK")
    world.pages[link] = dbl.relabel(world.pages[link], SAT, "5回東京7日", *OTHER_MEETINGS[claims])
    site, fetcher = serve(world.pages)
    with fetcher, pytest.raises(WrongMeeting) as caught:
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, now=Ticks(site))
    assert not isinstance(caught.value, NotListed)
    assert site.links() == [ODDS_MENU, link], "no odds page of a refused meeting is fetched"


@pytest.mark.parametrize(
    ("race", "kind", "claims"),
    [
        (1, "winplace", "other_day"),
        (2, "quinella", "other_venue"),
        (3, "wide", "other_day_and_venue"),
        (3, "winplace", "other_venue"),
    ],
)
def test_an_odds_page_of_another_meeting_raises_wrong_meeting(race, kind, claims):
    world = week(tok_races=3)
    link = world.odds_link(SAT, "TOK", race, kind)
    world.pages[link] = dbl.relabel(world.pages[link], SAT, "5回東京7日", *OTHER_MEETINGS[claims])
    site, fetcher = serve(world.pages)
    with fetcher, pytest.raises(WrongMeeting) as caught:
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, now=Ticks(site))
    assert not isinstance(caught.value, NotListed)
    assert link in site.links()


@pytest.mark.parametrize(
    ("day", "venue"),
    [
        (SAT, "HSN"),  # races Sunday only
        (SUN, "FKS"),  # races Saturday only
        (MON, "TOK"),  # a day the menu has no heading for
        (SAT, "KOK"),  # not racing this week
    ],
)
def test_a_meeting_the_menu_does_not_list_raises_not_listed(day, venue):
    site, fetcher = serve(week().pages)
    with fetcher, pytest.raises(NotListed) as caught:
        capture_meeting(fetcher, day, venue, ALL_POOLS, now=Ticks(site))
    assert not isinstance(caught.value, WrongMeeting)
    assert site.links() == [ODDS_MENU]


# ---------------------------------------------------------------- links are read, not computed


def test_menu_codes_are_followed_not_computed():
    world = week(tok_races=4)
    scrambled, mapping = dbl.scramble(world.pages)
    assert len(mapping) > 20
    assert all(new != old for old, new in mapping.items() if old not in dbl.ENTRY_CNAMES)
    site, fetcher = serve(scrambled)
    with fetcher:
        cap = capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, now=Ticks(site))
        times = read_post_times(fetcher, SUN, "TOK")
    assert all(s.served for s in site.seen), "a request went to a code no page linked"
    assert cap.requests == 2 + 3 * 4
    original = {cname for _, cname in world.pages} - dbl.ENTRY_CNAMES
    assert not original & {s.cname for s in site.seen}, "an unscrambled code was requested"
    assert dbl.price_rows(cap.rows) == world.expected_rows(SAT, "TOK", ALL_POOLS)
    assert times == world.expected_post_times(SUN, "TOK")


# ---------------------------------------------------------------- the 出馬表 side


@pytest.mark.parametrize(
    ("day", "venues"),
    [
        (SAT, ["TOK", "KYO", "FKS"]),
        (SUN, ["TOK", "NAK", "HSN"]),
        (MON, ["KOK"]),
        (date(2026, 12, 5), ["SAP", "NII"]),
    ],
)
def test_discover_venues_returns_every_meeting_of_the_day(day, venues):
    days = {
        SAT: [Meeting("TOK"), Meeting("KYO", nichi=3), Meeting("FKS", 3, 3, 1)],
        SUN: [Meeting("TOK", nichi=8), Meeting("NAK", 6, 4, 2), Meeting("HSN", 2, 4, 6)],
        MON: [Meeting("KOK", 4, 2, 1)],
        date(2026, 12, 5): [Meeting("SAP", 3, 1, 1), Meeting("NII", 3, 1, 1)],
    }
    site, fetcher = serve(World(days).pages)
    with fetcher:
        got = discover_venues(fetcher, day)
    assert sorted(got) == sorted(venues)
    assert len(got) == len(set(got))
    assert site.links() == [CARD_MENU]
    assert [t.page_type for t in fetcher.tasks] == ["official_menu"]


@pytest.mark.parametrize(("day", "venue"), [(SAT, "TOK"), (SAT, "KYO"), (SUN, "TOK"), (SUN, "HSN")])
def test_read_post_times_costs_two_requests(day, venue):
    world = week(tok_races=12)
    site, fetcher = serve(world.pages)
    with fetcher:
        got = read_post_times(fetcher, day, venue)
    expected = world.expected_post_times(day, venue)
    assert got == expected
    assert all(t.utcoffset() == timedelta(0) for t in got.values())
    assert site.links() == [CARD_MENU, world.card_list(day, venue)]
    assert [t.page_type for t in fetcher.tasks] == ["official_menu", "official_card_racelist"]
