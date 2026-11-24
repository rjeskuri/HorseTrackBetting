"""Ticket #66: the HKJC source adapter that drives the HK backfill (ported from HKTrack's
`sources/hkjc/adapter.py` without its live methods), with HKTrack's `test_tasks.py` and
`test_core_review.py` ported onto synthetic pages.

Chain: fixture months -> race 1 of every local race day up to today -> (from the parsed race 1)
the other races of the card -> (from every parsed race) sectionals, comments on running and horse
profiles. Pages are the synthetic builders of hkjc_*_pages.py; expected keys, URLs, priorities
and years are written out here or derived from the builders' data.

Contract pinned here (the coder follows it):
- `horsetrack.countries.hongkong.hkjc.adapter.ADAPTER` satisfies
  `core.ingest.adapters.SourceAdapter`, `source == "hkjc"`, and `get_adapter("hkjc")` returns
  it. No live methods (upcoming_tasks, live_odds). `type(ADAPTER)(today=<callable returning a
  date>)` builds an adapter on an injected clock; without it, "today" is the date in
  Asia/Hong_Kong (hkjc.yaml timezone).
- The module keeps HKTrack's constants: FIXTURE_PRIORITY 10_000_000, HORSE_REFRESH_DAYS 7,
  CORUNNING_REFRESH_DAYS 14, MAX_RACES_PER_CARD 12.
- Tasks: `FetchTask.source == "hkjc"`; page types `fixture` (key YYYY-MM, priority
  FIXTURE_PRIORITY, mutable for the current and later months), `results` / `sectional` /
  `corunning` (key `<YYYY-MM-DD>_<VENUE>_R<NN>`), `horse` (key = the horse id, year = the
  registration year in the id, else the race year). Dated priority = race date ordinal * 10 +
  rank, rank results 5 > horse 4 > sectional 3 > corunning 2. Sectionals from 2008-06-01,
  comments on running from 2010-01-01. Results and sectionals are mutable on the race day only;
  horse profiles for 7 days after the race, comments on running for 14.
- `classify(page_type, body)` decides by content only (fetched / empty / abandoned / blocked);
  `parse(page)` dispatches to `horsetrack.countries.hongkong.hkjc.parse_<page_type>`.
"""

from __future__ import annotations

import re
import sys
import time
import types
from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import hkjc_corunning_pages as corunning_pages
import hkjc_entries_pages as entries_pages
import hkjc_horse_pages as horse_pages
import hkjc_pages as fixture_pages
import hkjc_racecard_pages as racecard_pages
import hkjc_results_pages as rp
import hkjc_sectional_pages as sectional_pages
import pytest

from horsetrack.core.ingest.adapters import SourceAdapter, get_adapter
from horsetrack.core.ingest.queue import FetchQueue
from horsetrack.core.ingest.types import FetchTask, RawPage
from horsetrack.core.store import build
from horsetrack.countries.hongkong.hkjc import (
    parse_common,
    parse_corunning,
    parse_entries,
    parse_fixture,
    parse_horse,
    parse_racecard,
    parse_results,
    parse_sectional,
    urls,
)

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.hongkong.hkjc.adapter import (
    ADAPTER,
    CORUNNING_REFRESH_DAYS,
    FIXTURE_PRIORITY,
    HORSE_REFRESH_DAYS,
    MAX_RACES_PER_CARD,
)

INFO = "https://racing.hkjc.com/en-us/local/information"
RACE_DAY = rp.LIVE.race_date  # 2026-10-14, Happy Valley
TODAY = RACE_DAY + timedelta(days=5)  # 2026-10-19

LIVE_R1 = replace(rp.LIVE, race_no=1)  # 2026-10-14 HV, a card of 9 races, 8 runners

M = fixture_pages.meeting
OCT_2026 = [
    M(4, "ST", "D", "Turf"),
    M(7, "HV", "N", "Turf"),
    M(11, "ST", "D", "Mixed"),
    M(14, "HV", "N", "Turf"),
    M(18, "ST", "D", "Turf", races=0),  # cancelled on the day: the results page decides
    M(21, "HV", "N", "Turf"),
    M(23, "CH", "D", "Turf"),  # Conghua: not a local meeting
    M(25, "ST", "D", "Turf"),
    M(28, "HV", "N", "Turf"),
]
OCT_2026_LOCAL = [(4, "ST"), (7, "HV"), (11, "ST"), (14, "HV"), (18, "ST"), (21, "HV"), (25, "ST"),
                  (28, "HV")]  # fmt: skip
JAN_2011 = [M(d, v, "D", "Turf") for d, v in [(1, "ST"), (5, "HV"), (9, "ST"), (12, "HV"),
            (16, "ST"), (19, "HV"), (23, "ST"), (26, "HV"), (30, "ST")]]  # fmt: skip
DEC_2022 = [M(d, v, "N", "Turf") for d, v in [(4, "ST"), (7, "HV"), (11, "ST"), (14, "HV"),
            (18, "ST"), (21, "HV"), (26, "ST"), (28, "HV")]]  # fmt: skip


def adapter_on(today: date):
    return type(ADAPTER)(today=lambda: today)


@pytest.fixture
def adapter():
    return adapter_on(TODAY)


def key(day: date, venue: str, race_no: int) -> str:
    return f"{day.isoformat()}_{venue}_R{race_no:02d}"


def rank(day: date, r: int) -> int:
    return day.toordinal() * 10 + r


def fixture_body(year: int, month: int, days) -> bytes:
    """The calendar inside the generic content container every server-rendered page has."""
    body = fixture_pages.calendar(year, month, days)
    return body.replace(b"<body>", b'<body><div class="commContent">').replace(
        b"</body>", b"</div></body>"
    )


def fixture_raw(year: int, month: int, days) -> RawPage:
    return RawPage(
        "hkjc", "fixture", f"{year:04d}-{month:02d}", fixture_pages.fixture_url(year, month),
        fixture_body(year, month, days),
    )  # fmt: skip


def results_raw(race: rp.Race, body: bytes | None = None) -> RawPage:
    url = (rp.live_url if race.race_date >= date(2026, 9, 1) else rp.archive_url)(
        race.race_date, race.venue, race.race_no
    )
    return RawPage("hkjc", "results", race.key, url, rp.render(race) if body is None else body)


def group(tasks: list[FetchTask]) -> dict[str, list[FetchTask]]:
    grouped: dict[str, list[FetchTask]] = {}
    for task in tasks:
        grouped.setdefault(task.page_type, []).append(task)
    return grouped


def followups(adapter, race: rp.Race, body: bytes | None = None) -> dict[str, list[FetchTask]]:
    """What the runner does with a results page: parse it, then ask for follow-ups."""
    return group(adapter.followup_tasks(adapter.parse(results_raw(race, body))))


def followups_on(today: date, race: rp.Race = LIVE_R1) -> dict[str, list[FetchTask]]:
    return followups(adapter_on(today), race)


def horse_ids(race: rp.Race) -> list[str]:
    return [r.horse_id for r in race.runners if r.horse_id is not None]


def with_simulcast_rows(race: rp.Race) -> bytes:
    """The results page with the tab rows a simulcast meeting adds: one row of links naming
    another racecourse, one row of links naming none under a non-local label."""
    d = race.race_date.strftime("%Y/%m/%d")
    img = '<img src="/racing/info/StaticFile/Images/Racing/racecard_rt_{}.gif"/>'
    named = "".join(
        f'<td><a href="/en-us/local/information/localresults?racedate={d}&amp;Racecourse=S1'
        f'&amp;RaceNo={n}">{img.format(n)}</a></td>'
        for n in range(1, 13)
    )
    unnamed = "".join(
        f'<td><a href="/en-us/local/information/simulcastresults?racedate={d}&amp;RaceNo={n}">'
        f"{img.format(n)}</a></td>"
        for n in range(1, 13)
    )
    rows = (
        f'<tr><td class="font_w7 f_tar"><span>Simulcast:</span> </td>{named}</tr>'
        f'<tr><td class="font_w7 f_tar"><span>Overseas:</span> </td>{unnamed}</tr>'
    )
    html = rp.render(race).decode()
    assert html.count("</tr></table></div>") >= 1
    return html.replace("</tr></table></div>", f"</tr>{rows}</table></div>", 1).encode()


# ---------------------------------------------------------------- wiring


def test_the_adapter_conforms_to_the_protocol():
    assert isinstance(ADAPTER, SourceAdapter)
    assert ADAPTER.source == "hkjc"
    assert get_adapter("hkjc") is ADAPTER
    built = adapter_on(date(2020, 1, 1))
    assert isinstance(built, SourceAdapter) and built.source == "hkjc"
    # the backfill adapter only: the live methods stay out of this port
    for live in ("upcoming_tasks", "live_odds"):
        assert not hasattr(ADAPTER, live), live
    # HKTrack's constants, kept
    assert (FIXTURE_PRIORITY, HORSE_REFRESH_DAYS, CORUNNING_REFRESH_DAYS, MAX_RACES_PER_CARD) == (
        10_000_000, 7, 14, 12,
    )  # fmt: skip


def test_today_defaults_to_the_date_in_hong_kong():
    hk_today = datetime.now(ZoneInfo("Asia/Hong_Kong")).date()
    tomorrow = hk_today + timedelta(days=1)
    pages = [
        fixture_raw(d.year, d.month, [M(d.day, "HV", "N", "Turf")]) for d in (hk_today, tomorrow)
    ]
    for default in (ADAPTER, type(ADAPTER)()):
        assert [t.key for t in default.meeting_tasks(pages)] == [key(hk_today, "HV", 1)]


# ---------------------------------------------------------------- the chain


def test_the_backfill_chain_from_fixture_month_to_follow_ups(adapter):
    # 1. the month's fixture task
    (october,) = adapter.fixture_tasks(date(2026, 10, 1), date(2026, 10, 31))
    assert october.url == f"{INFO}/fixture?calyear=2026&calmonth=10"
    fixture = RawPage("hkjc", "fixture", october.key, october.url, fixture_body(2026, 10, OCT_2026))
    assert adapter.classify("fixture", fixture.body) == "fetched"
    assert adapter.followup_tasks(adapter.parse(fixture)) == []

    # 2. race 1 of each local meeting up to today
    meetings = {t.key: t for t in adapter.meeting_tasks([fixture])}
    first = meetings[key(RACE_DAY, "HV", 1)]
    assert first.page_type == "results"

    # 3. race 1 fans out to the rest of the card and to its own follow-ups
    page = RawPage("hkjc", first.page_type, first.key, first.url, rp.render(LIVE_R1))
    assert adapter.classify("results", page.body) == "fetched"
    grouped = group(adapter.followup_tasks(adapter.parse(page)))
    assert set(grouped) == {"results", "sectional", "corunning", "horse"}
    assert sorted(t.key for t in grouped["results"]) == [
        key(RACE_DAY, "HV", n) for n in range(2, 10)
    ]

    # 4. a later race of the card: its own follow-ups only
    race3 = next(t for t in grouped["results"] if t.key == key(RACE_DAY, "HV", 3))
    page3 = RawPage("hkjc", "results", race3.key, race3.url, rp.render(rp.LIVE))
    later = group(adapter.followup_tasks(adapter.parse(page3)))
    assert set(later) == {"sectional", "corunning", "horse"}
    assert [t.key for t in later["sectional"]] == [key(RACE_DAY, "HV", 3)]
    assert [t.key for t in later["corunning"]] == [key(RACE_DAY, "HV", 3)]
    assert sorted(t.key for t in later["horse"]) == sorted(horse_ids(rp.LIVE))

    # 5. the follow-up pages themselves lead nowhere
    (sectional,) = later["sectional"]
    section = sectional_pages.Section("1", "", "13.50")
    sec_body = sectional_pages.sectional_page(RACE_DAY, "Happy Valley", [
        sectional_pages.Race(3, [sectional_pages.Runner("1", 6, "AMBER LANTERN", "HK_2023_X101",
                                                        [section], "1:38.42")]),
    ])  # fmt: skip
    (corunning,) = later["corunning"]
    co_body = corunning_pages.corunning_page(RACE_DAY, "HV", 3, CORUNNING_RUNNERS)
    horse = next(t for t in later["horse"] if t.key == "HK_2023_X101")
    horse_body = horse_pages.horse_page(horse_pages.Profile("HK_2023_X101", "AMBER LANTERN (X101)"))
    for task, body in ((sectional, sec_body), (corunning, co_body), (horse, horse_body)):
        assert adapter.classify(task.page_type, body) == "fetched", task.page_type
        parsed = adapter.parse(RawPage("hkjc", task.page_type, task.key, task.url, body))
        assert adapter.followup_tasks(parsed) == [], task.page_type


# ---------------------------------------------------------------- fixture tasks


def months_between(start: date, end: date) -> list[tuple[int, int]]:
    out, (y, m) = [], (start.year, start.month)
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def test_fixture_tasks_cover_every_month_from_2010_to_today(adapter):
    tasks = adapter.fixture_tasks(date(2010, 1, 1), TODAY)
    assert len(tasks) == 202  # 16 full years * 12 + 10 months of 2026
    assert sorted(t.key for t in tasks) == [f"{y:04d}-{m:02d}" for y, m in months_between(
        date(2010, 1, 1), TODAY)]  # fmt: skip
    assert len({t.key for t in tasks}) == len(tasks)
    first = min(tasks, key=lambda t: t.key)
    assert first == FetchTask(
        "hkjc", "fixture", "2010-01", f"{INFO}/fixture?calyear=2010&calmonth=01", 2010, 10_000_000
    )
    assert {t.priority for t in tasks} == {10_000_000}
    assert {t.page_type for t in tasks} == {"fixture"}
    assert {t.source for t in tasks} == {"hkjc"}


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2026, 8, 15), date(2027, 1, 2)),  # around today: the current and later months change
        (date(2019, 11, 30), date(2020, 2, 1)),  # across a year end, mid-month bounds
        (date(2024, 2, 29), date(2024, 2, 29)),  # one day: its month
        (date(2015, 3, 1), date(2015, 12, 31)),
    ],
)
def test_fixture_tasks_month_by_month(adapter, start, end):
    tasks = sorted(adapter.fixture_tasks(start, end), key=lambda t: t.key)
    expected = months_between(start, end)
    assert [t.key for t in tasks] == [f"{y:04d}-{m:02d}" for y, m in expected]
    for task, (year, month) in zip(tasks, expected, strict=True):
        assert task.url == f"{INFO}/fixture?calyear={year:04d}&calmonth={month:02d}"
        assert task.year == year
        assert task.mutable == ((year, month) >= (TODAY.year, TODAY.month)), task.key


def test_fixture_tasks_of_an_empty_range(adapter):
    assert adapter.fixture_tasks(date(2020, 2, 1), date(2020, 1, 1)) == []
    assert [t.year for t in adapter.fixture_tasks(date(2019, 12, 1), date(2020, 1, 1))] == [
        2019,
        2020,
    ]


# ---------------------------------------------------------------- meeting tasks


def test_meeting_tasks_emit_race_one_for_local_days_up_to_today(adapter):
    tasks = sorted(adapter.meeting_tasks([fixture_raw(2026, 10, OCT_2026)]), key=lambda t: t.key)
    assert [t.key for t in tasks] == [
        key(date(2026, 10, d), v, 1) for d, v in OCT_2026_LOCAL if date(2026, 10, d) <= TODAY
    ]
    hv = next(t for t in tasks if t.key == "2026-10-14_HV_R01")
    assert hv.page_type == "results"
    assert hv.url == urls.results(RACE_DAY, "HV", 1)
    assert hv.year == 2026
    assert hv.priority == rank(RACE_DAY, 5)
    assert not hv.mutable
    assert {t.source for t in tasks} == {"hkjc"}


def test_meeting_tasks_use_archive_urls_for_old_months_and_accept_many_pages(adapter):
    pages = [
        fixture_raw(2011, 1, JAN_2011),
        fixture_raw(2022, 12, DEC_2022),
        fixture_raw(2011, 1, JAN_2011),  # the same month fetched twice
    ]
    tasks = adapter.meeting_tasks(pages)
    assert len(tasks) == 9 + 8
    assert len({t.key for t in tasks}) == len(tasks)
    by_key = {t.key: t for t in tasks}
    assert by_key["2011-01-01_ST_R01"].url == (
        f"{INFO}/archive/localresults?racedate=2011/01/01&racecourse=ST&RaceNo=1"
    )
    assert by_key["2022-12-07_HV_R01"].url == (
        f"{INFO}/archive/localresults?racedate=2022/12/07&racecourse=HV&RaceNo=1"
    )
    for task in tasks:
        day = date.fromisoformat(task.key[:10])
        assert task.priority == rank(day, 5) < FIXTURE_PRIORITY
        assert task.year == day.year
        assert not task.mutable
    assert adapter.meeting_tasks([fixture_raw(2011, 2, [])]) == []


def test_newer_meetings_outrank_older_ones(adapter):
    tasks = sorted(adapter.meeting_tasks([fixture_raw(2026, 10, OCT_2026)]), key=lambda t: t.key)
    priorities = [t.priority for t in tasks]
    assert priorities == sorted(set(priorities))  # strictly up with the date


@pytest.mark.parametrize(
    ("today", "days"),
    [
        (date(2026, 10, 14), [4, 7, 11, 14]),  # race day: its own race 1 is queued, and mutable
        (date(2026, 10, 13), [4, 7, 11]),
        (date(2026, 10, 3), []),
        (date(2026, 10, 31), [4, 7, 11, 14, 18, 21, 25, 28]),
        (date(2026, 11, 2), [4, 7, 11, 14, 18, 21, 25, 28]),
    ],
)
def test_meeting_tasks_stop_at_today_and_same_day_results_stay_refetchable(today, days):
    tasks = adapter_on(today).meeting_tasks([fixture_raw(2026, 10, OCT_2026)])
    assert sorted(t.key[:10] for t in tasks) == [date(2026, 10, d).isoformat() for d in days]
    for task in tasks:
        assert task.mutable == (date.fromisoformat(task.key[:10]) == today), task.key


# ---------------------------------------------------------------- follow-up tasks


@pytest.mark.parametrize(
    ("race", "last"),
    [
        (LIVE_R1, 9),
        (rp.ARCHIVE_2010, 10),
        (rp.ARCHIVE_2004, 8),
        (replace(LIVE_R1, n_races=12), 12),
        (replace(rp.AWT, race_no=1, n_races=11), 11),
    ],
    ids=["live", "archive-2011", "archive-2005", "twelve", "awt"],
)
def test_race_one_fans_out_to_the_other_local_races(adapter, race, last):
    results = followups(adapter, race)["results"]
    assert sorted(t.key for t in results) == [
        key(race.race_date, race.venue, n) for n in range(2, last + 1)
    ]
    for task in results:
        n = int(task.key[-2:])
        assert task.url == urls.results(race.race_date, race.venue, n)
        assert task.priority == rank(race.race_date, 5)
        assert task.year == race.race_date.year


def test_archive_results_urls_are_written_out(adapter):
    results = {t.key: t for t in followups(adapter, rp.ARCHIVE_2010)["results"]}
    assert results["2011-01-30_ST_R04"].url == (
        f"{INFO}/archive/localresults?racedate=2011/01/30&racecourse=ST&RaceNo=4"
    )


def test_every_results_page_yields_sectional_corunning_and_horse_tasks(adapter):
    grouped = followups(adapter, LIVE_R1)
    (sectional,) = grouped["sectional"]
    (corunning,) = grouped["corunning"]
    race_key = key(RACE_DAY, "HV", 1)
    assert sectional == FetchTask(
        "hkjc", "sectional", race_key, urls.sectional(RACE_DAY, 1), 2026, rank(RACE_DAY, 3)
    )
    # 5 days after the meeting: comments on running may still be unpublished and the horse pages
    # may predate the race, so both stay refetchable; the sectional page is final
    assert corunning == FetchTask(
        "hkjc", "corunning", race_key, urls.corunning(RACE_DAY, 1), 2026, rank(RACE_DAY, 2),
        mutable=True,
    )  # fmt: skip
    horses = {t.key: t for t in grouped["horse"]}
    assert sorted(horses) == sorted(horse_ids(LIVE_R1))  # the withdrawn runner too
    assert len(grouped["horse"]) == len(horses) == 8
    for horse_id, task in horses.items():
        assert task == FetchTask(
            "hkjc", "horse", horse_id, f"{INFO}/horse?horseid={horse_id}&Option=1",
            int(horse_id[3:7]), rank(RACE_DAY, 4), mutable=True,
        )  # fmt: skip


def test_priorities_complete_a_date_before_moving_to_an_older_one(adapter):
    grouped = followups(adapter, LIVE_R1)
    newer = {name: {t.priority for t in tasks} for name, tasks in grouped.items()}
    assert all(len(p) == 1 for p in newer.values())
    top = {name: p.pop() for name, p in newer.items()}
    assert top["results"] > top["horse"] > top["sectional"] > top["corunning"]
    older = followups(adapter, rp.DEAD_HEAT)
    assert top["corunning"] > max(t.priority for tasks in older.values() for t in tasks)
    assert max(top.values()) < FIXTURE_PRIORITY


def test_later_races_do_not_fan_out_again(adapter):
    grouped = followups(adapter, rp.DEAD_HEAT)  # 2021-12-08 HV race 2
    assert "results" not in grouped
    assert [t.key for t in grouped["sectional"]] == ["2021-12-08_HV_R02"]
    assert grouped["sectional"][0].url == (
        f"{INFO}/archive/displaysectionaltime?racedate=08/12/2021&RaceNo=2"
    )
    assert grouped["corunning"][0].url == f"{INFO}/archive/corunning?date=20211208&raceno=2"
    assert sorted(t.key for t in grouped["horse"]) == sorted(horse_ids(rp.DEAD_HEAT))
    assert next(t for t in grouped["horse"] if t.key == "HK_2018_S390").year == 2018


def test_simulcast_tabs_never_become_tasks(adapter):
    race = replace(rp.LIVE, race_date=date(2026, 10, 11), venue="ST", race_no=1, n_races=10)
    results = followups(adapter, race, with_simulcast_rows(race))["results"]
    assert sorted(t.key for t in results) == [key(race.race_date, "ST", n) for n in range(2, 11)]
    assert {t.url for t in results} == {urls.results(race.race_date, "ST", n) for n in range(2, 11)}


@pytest.mark.parametrize(
    ("race_date", "sectional", "corunning"),
    [
        (date(2005, 3, 6), False, False),
        (date(2008, 5, 31), False, False),
        (date(2008, 6, 1), True, False),  # per-horse sectionals start in mid-2008
        (date(2009, 12, 31), True, False),
        (date(2010, 1, 1), True, True),  # comments on running are served back to 2010-01-01
        (date(2011, 1, 30), True, True),
    ],
)
def test_no_sectional_or_corunning_tasks_before_the_data_exists(adapter, race_date, sectional,
                                                                corunning):  # fmt: skip
    race = replace(rp.ARCHIVE_2010, race_date=race_date)
    grouped = followups(adapter, race)
    assert ("sectional" in grouped) is sectional
    assert ("corunning" in grouped) is corunning
    assert sorted(t.key for t in grouped["results"]) == [
        key(race_date, "ST", n) for n in range(2, 11)
    ]
    assert sorted(t.key for t in grouped["horse"]) == sorted(horse_ids(race))
    if sectional:
        assert [t.key for t in grouped["sectional"]] == [key(race_date, "ST", 1)]


def test_the_2005_layout_without_running_positions_still_yields_horses(adapter):
    grouped = followups(adapter, rp.ARCHIVE_2004)
    assert set(grouped) == {"results", "horse"}
    assert sorted(t.key for t in grouped["horse"]) == sorted(horse_ids(rp.ARCHIVE_2004))
    assert {t.year for t in grouped["horse"]} == {int(h[3:7]) for h in horse_ids(rp.ARCHIVE_2004)}


def test_abandoned_race_one_still_probes_the_rest_of_the_card_but_nothing_else(adapter):
    grouped = followups(adapter, rp.ABANDONED)  # 2022-12-07 HV race 1, nine races
    assert set(grouped) == {"results"}
    assert sorted(t.key for t in grouped["results"]) == [
        key(date(2022, 12, 7), "HV", n) for n in range(2, 10)
    ]


def test_races_called_off_mid_meeting_have_no_followups(adapter):
    # header-less layout (race 7) and a void race (race 8): nothing was run, and race 1 already
    # fanned out
    assert followups(adapter, rp.DECLARED_ABANDONED) == {}
    assert followups(adapter, rp.VOID) == {}


def test_a_header_less_abandoned_race_one_still_probes_the_rest_of_the_card(adapter):
    # the 2024+ layout served as race 1: the tab strip is all there is, and it is enough
    race = replace(rp.DECLARED_ABANDONED, race_no=1)
    tasks = adapter.followup_tasks(adapter.parse(results_raw(race)))
    assert sorted((t.page_type, t.key) for t in tasks) == [
        ("results", key(date(2025, 11, 12), "HV", n)) for n in range(2, 10)
    ]


def test_other_page_types_have_no_followups(adapter):
    fixture = adapter.parse(fixture_raw(2026, 10, OCT_2026))
    assert fixture["meetings"]
    assert adapter.followup_tasks(fixture) == []
    assert adapter.followup_tasks({}) == []
    assert adapter.followup_tasks({"runners": [{"horse_id": "HK_2023_X101"}]}) == []
    no_info_url = rp.archive_url(date(2000, 1, 1), "ST", 1)
    no_info = RawPage("hkjc", "results", "2000-01-01_ST_R01", no_info_url, rp.no_information())
    assert adapter.followup_tasks(adapter.parse(no_info)) == []


def test_unsafe_horse_ids_never_become_cache_keys(adapter):
    parsed = adapter.parse(results_raw(LIVE_R1))
    runners = parsed["runners"]
    runners[0]["horse_id"] = "../../etc/passwd"
    runners[1]["horse_id"] = None
    runners[2]["horse_id"] = "HK_2023_X478&Option=2"
    runners[4]["horse_id"] = runners[3]["horse_id"]  # the same horse twice: one task
    runners[5]["horse_id"] = "X390"  # no registration year in the id: the race's year
    horses = {t.key: t for t in adapter.followup_tasks(parsed) if t.page_type == "horse"}
    assert sorted(horses) == sorted(["HK_2023_X478", "X390", "HK_2024_Y210", "HK_2023_X300"])
    assert horses["X390"].year == 2026
    assert horses["HK_2024_Y210"].year == 2024
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", k) for k in horses)


# ---------------------------------------------------------------- race tabs are scraped input


def test_implausible_race_numbers_in_the_tabs_never_become_tasks():
    race = replace(LIVE_R1, n_races=8)
    html = rp.render(race).decode()
    assert all(html.count(f"RaceNo={n}") == 1 for n in (6, 7, 8))
    html = (
        html.replace("RaceNo=8", "RaceNo=99")
        .replace("RaceNo=7", "RaceNo=0")
        .replace("RaceNo=6", "RaceNo=-4")
    )
    adapter = adapter_on(TODAY)
    parsed = adapter.parse(results_raw(race, html.encode()))
    assert all(n > 0 for n in parsed["_meta"][0]["race_nos"])
    results = [t for t in adapter.followup_tasks(parsed) if t.page_type == "results"]
    assert sorted(t.key for t in results) == [key(RACE_DAY, "HV", n) for n in (2, 3, 4, 5)]


def test_race_numbers_handed_in_by_a_parser_are_bounded_too():
    adapter = adapter_on(TODAY)
    parsed = adapter.parse(results_raw(LIVE_R1))
    parsed["_meta"][0]["race_nos"] = [1, 2, 12, 13, 0, -1, 10**6, "3", None, 2.0]
    results = [t for t in adapter.followup_tasks(parsed) if t.page_type == "results"]
    assert sorted(t.key for t in results) == [key(RACE_DAY, "HV", 2), key(RACE_DAY, "HV", 12)]


# ---------------------------------------------------------------- classify

_GEAR = ("B", None, "TT", "V", None, "CP")
_HOW = ("wide", "keenly", "handily", "greenly", "evenly", "late")
CORUNNING_RUNNERS = tuple(
    corunning_pages.Runner(
        str(i), i, f"SYNTH RUNNER {i}", f"HK_2023_C{i:03d}", f"C{i:03d}", f"J Jockey{i}",
        _GEAR[i - 1], f"Raced {_HOW[i - 1]}.",
    )
    for i in range(1, 7)
)  # fmt: skip

_CARD = racecard_pages.Card(
    race_date=date(2026, 10, 21), venue="HV", race_no=4, race_name="SYNTH LANTERN HANDICAP",
    post="19:45", course='Turf, "C" Course, 1650M',
    prize="Prize Money: $1,170,000, Rating: 60-40, Class 4",
    starters=tuple(racecard_pages.starter(n, n) for n in range(1, 7)),
)  # fmt: skip
_SECTION = entries_pages.Section(
    prov="P10", section_no=1, race_name="SYNTH BAY HANDICAP", race_class="Class 4",
    distance_m=1200, band="60-40", entries=tuple(entries_pages.entry(n) for n in range(1, 6)),
)  # fmt: skip
_SECTIONAL_RACE = sectional_pages.Race(1, [
    sectional_pages.Runner("1", 6, "AMBER LANTERN", "HK_2023_X101",
                           [sectional_pages.Section("1", "", "13.50")], "1:38.42"),
])  # fmt: skip


def _page(kind: str) -> bytes:
    return {
        "results": lambda: rp.render(rp.LIVE),
        "results_2011": lambda: rp.render(rp.ARCHIVE_2010),
        "results_2005": lambda: rp.render(rp.ARCHIVE_2004),
        "results_dead_heat": lambda: rp.render(rp.DEAD_HEAT),
        "results_protest": lambda: rp.render(rp.PROTEST),
        "abandoned": lambda: rp.render(rp.ABANDONED),
        "declared_abandoned": lambda: rp.render(rp.DECLARED_ABANDONED),
        "void": lambda: rp.render(rp.VOID),
        "no_info": lambda: rp.no_information(css=False),
        # the live route's sentinel keeps the inline CSS, whose selectors name the results
        # container; a script and a comment naming it are not content either
        "no_info_live": rp.no_information,
        "no_info_script": lambda: rp.no_information(script=True),
        "busy": rp.busy_page,
        "racecard": lambda: racecard_pages.card(_CARD),
        "racecard_no_info": racecard_pages.no_information,
        "racecard_not_available": racecard_pages.not_available,
        "entries": lambda: entries_pages.entries_page(date(2026, 10, 21), "HV", [_SECTION]),
        "sectional": lambda: sectional_pages.sectional_page(RACE_DAY, "Happy Valley",
                                                            [_SECTIONAL_RACE]),
        "sectional_header_only": lambda: sectional_pages.sectional_page(
            date(2008, 1, 1), "Sha Tin", []),
        "corunning": lambda: corunning_pages.corunning_page(RACE_DAY, "HV", 1, CORUNNING_RUNNERS),
        "corunning_unpublished": lambda: corunning_pages.corunning_page(
            RACE_DAY, "HV", 1, [replace(r, comment=None) for r in CORUNNING_RUNNERS]),
        "horse": lambda: horse_pages.horse_page(
            horse_pages.Profile("HK_2023_X101", "AMBER LANTERN (X101)")),
        "horse_retired": lambda: horse_pages.horse_page(
            horse_pages.Profile("HK_2008_K277", "OLD SPARROW (K277) (Retired)", retired=True)),
        "fixture": lambda: fixture_body(2026, 10, OCT_2026),
        "fixture_2011": lambda: fixture_body(2011, 1, JAN_2011),
    }[kind]()  # fmt: skip


@pytest.mark.parametrize(
    ("page_type", "kind", "expected"),
    [
        ("results", "results", "fetched"),
        ("results", "results_2011", "fetched"),
        ("results", "results_2005", "fetched"),
        ("results", "results_dead_heat", "fetched"),
        ("results", "results_protest", "fetched"),
        ("results", "abandoned", "abandoned"),
        ("results", "declared_abandoned", "abandoned"),  # second banner wording, no race header
        ("results", "void", "abandoned"),  # race annulled after the start: no result either
        ("results", "no_info", "empty"),
        ("results", "no_info_live", "empty"),
        ("results", "no_info_script", "empty"),
        ("results", "busy", "blocked"),
        ("sectional", "no_info_live", "empty"),
        ("sectional", "sectional", "fetched"),
        ("sectional", "sectional_header_only", "empty"),  # a meeting before the data starts
        ("corunning", "corunning", "fetched"),
        ("corunning", "corunning_unpublished", "empty"),
        ("corunning", "no_info_live", "empty"),
        # "The Race Card is not available!" (before declarations) has no generic sentinel text
        ("racecard", "racecard_not_available", "empty"),
        ("racecard", "racecard_no_info", "empty"),
        ("racecard", "racecard", "fetched"),
        ("racecard", "no_info", "empty"),
        ("entries", "entries", "fetched"),
        ("horse", "horse", "fetched"),
        ("horse", "horse_retired", "fetched"),
        ("horse", "no_info", "empty"),
        ("fixture", "fixture", "fetched"),
        ("fixture", "fixture_2011", "fetched"),
        ("fixture", "results_dead_heat", "blocked"),  # right site, wrong container
        ("results", "fixture", "blocked"),
        ("sectional", "results_dead_heat", "blocked"),
        ("corunning", "sectional", "blocked"),
    ],
)
def test_classify_built_pages(adapter, page_type, kind, expected):
    assert adapter.classify(page_type, _page(kind)) == expected


@pytest.mark.parametrize(
    ("page_type", "html", "expected"),
    [
        ("results", "<html><body><h1>Access Denied</h1></body></html>", "blocked"),
        ("results", "", "blocked"),
        ("horse", "<html><body><h1>Access Denied</h1></body></html>", "blocked"),
        ("horse", '<div class="commContent"><table class="horseProfile"></table></div>', "fetched"),
        (
            "racecard",
            '<div class="commContent raceCard"><table id="racecardlist"></table></div>',
            "fetched",
        ),
        (
            "entries",
            '<div class="commContent">Race Meeting: 23/09/2026 Happy Valley</div>',
            "fetched",
        ),
        (
            "corunning",
            '<div class="corunning commContent"><td>Raced wide without cover.</td></div>',
            "fetched",
        ),
        (
            "corunning",
            '<div class="corunning commContent"><td>No Comments on Running information for '
            "this horse.</td></div>",
            "empty",
        ),
        ("corunning", '<div class="commContent">something else</div>', "blocked"),
        (
            "sectional",
            '<div class="dispalySectionalTime commContent"><table class="race_table">'
            "</table></div>",
            "fetched",
        ),
        (
            "sectional",
            '<div class="dispalySectionalTime commContent">Meeting Date:01/01/2008, Sha Tin</div>',
            "empty",
        ),
        # a stray phrase inside a real results page must not throw the page away
        (
            "results",
            '<div class="localResults commContent"><td>No information.</td></div>',
            "fetched",
        ),
        # class names inside <style>, <script> or comments are not content
        (
            "results",
            "<style>.localResults p {padding: 0}</style><script>var c = 'localResults';</script>"
            '<div class="commContent"><div id="errorContainer">No information.</div></div>',
            "empty",
        ),
        (
            "results",
            '<style>.localResults p {padding: 0}</style><div class="commContent">Service '
            "unavailable</div>",
            "blocked",
        ),
        (
            "sectional",
            "<style>.dispalySectionalTime .race_table td {padding: 0}</style>"
            '<div class="dispalySectionalTime commContent">Meeting Date:01/01/2008, Sha Tin</div>',
            "empty",
        ),
        (
            "results",
            '<div class="localResults commContent"><!-- This race is declared abandoned. -->'
            "<td>1</td></div>",
            "fetched",
        ),
        (
            "results",
            '<div class="localResults commContent"><div>This race is declared void. Under the '
            "Horse Race Betting Rules, Single Race Pools for this race, if any, will be refunded."
            "</div></div>",
            "abandoned",
        ),
        # the stewards' report may use the same words about ANOTHER race; only the banner counts
        (
            "results",
            '<div class="localResults commContent"><td>After Race 3 was declared void the '
            "field ...</td></div>",
            "fetched",
        ),
    ],
)
def test_classify_synthetic_snippets(adapter, page_type, html, expected):
    assert adapter.classify(page_type, html.encode()) == expected


@pytest.mark.parametrize(
    "opener", ["<!-- x ", "<style> x ", "<script> x ", "<!-- a <STYLE> b <script> c "]
)
def test_classify_stays_fast_when_openers_are_never_closed(opener):
    # One lazy regex over the page was quadratic: 20,000 unclosed openers (140 kB) took 10-15 s,
    # and classify() runs inside the unattended loop on whatever the site (or a proxy) sends.
    body = (opener * 20_000 + '<div class="localResults commContent"><td>1</td></div>').encode()
    started = time.perf_counter()
    status = adapter_on(RACE_DAY).classify("results", body)
    assert time.perf_counter() - started < 2.0
    assert status == "fetched"  # an opener without a closer hides nothing


def test_classify_still_ignores_closed_blocks_next_to_unclosed_ones():
    adapter = adapter_on(RACE_DAY)
    sentinel = (
        "<!-- never closed <style>.localResults p {padding: 0}</style><script>var c = "
        "'localResults';</script>"
        '<div class="commContent"><div id="errorContainer">No information.</div></div>'
    )
    assert adapter.classify("results", sentinel.encode()) == "empty"
    upper = '<STYLE>.localResults {}</STYLE >\n<div class="commContent">Service unavailable</div>'
    assert adapter.classify("results", upper.encode()) == "blocked"
    banner = (
        '<div class="localResults commContent"><!-- a --><!-- This race is declared void. -->'
        "<td>1</td></div>"
    )
    assert adapter.classify("results", banner.encode()) == "fetched"


@pytest.mark.parametrize(("keep", "expected"), [(6, "fetched"), (5, "fetched"), (1, "fetched"),
                                                (0, "empty")])  # fmt: skip
def test_corunning_is_empty_only_when_no_runner_has_a_comment(keep, expected):
    # Pages classified "empty" are never parsed. The sentinel is printed per horse, and the parser
    # keeps a race in which only some runners lack a comment, so classify() must not throw that
    # race away.
    runners = [r if i < keep else replace(r, comment=None) for i, r in enumerate(CORUNNING_RUNNERS)]
    body = corunning_pages.corunning_page(RACE_DAY, "HV", 1, runners)
    adapter = adapter_on(RACE_DAY)
    assert adapter.classify("corunning", body) == expected
    url = corunning_pages.corunning_url(RACE_DAY, 1, archive=False)
    rows = adapter.parse(RawPage("hkjc", "corunning", key(RACE_DAY, "HV", 1), url, body))
    # classify and the parser agree on what "no data" means
    assert bool(rows["running_comments"]) is (expected == "fetched")


# ---------------------------------------------------------------- parse dispatch


def test_parse_dispatches_on_page_type(adapter):
    sectional_body = _page("sectional")
    pages = [
        (parse_fixture, fixture_raw(2026, 10, OCT_2026)),
        (parse_results, results_raw(rp.LIVE)),
        (parse_results, results_raw(rp.ARCHIVE_2010)),
        (parse_sectional, RawPage("hkjc", "sectional", key(RACE_DAY, "HV", 1),
                                  sectional_pages.sectional_url(RACE_DAY, 1), sectional_body)),
        (parse_corunning, RawPage("hkjc", "corunning", key(RACE_DAY, "HV", 1),
                                  corunning_pages.corunning_url(RACE_DAY, 1, archive=False),
                                  _page("corunning"))),
        (parse_horse, RawPage("hkjc", "horse", "HK_2023_X101",
                              horse_pages.horse_url("HK_2023_X101"), _page("horse"))),
        (parse_racecard, RawPage("hkjc", "racecard", "2026-10-21_HV_R04",
                                 urls.racecard(date(2026, 10, 21), "HV", 4), _page("racecard"))),
        (parse_entries, RawPage("hkjc", "entries", "2026-10-21_HV",
                                entries_pages.entries_url(date(2026, 10, 21), "HV"),
                                _page("entries"))),
    ]  # fmt: skip
    for module, page in pages:
        assert adapter.parse(page) == module.parse(page), page.page_type
    assert len(adapter.parse(pages[0][1])["meetings"]) == 8
    assert len(adapter.parse(pages[1][1])["runners"]) == len(rp.LIVE.runners)
    assert len(adapter.parse(pages[2][1])["runners"]) == len(rp.ARCHIVE_2010.runners)


def test_parse_plugs_in_parser_modules_it_has_never_heard_of(adapter, monkeypatch):
    name = "horsetrack.countries.hongkong.hkjc.parse_madeup"
    module = types.ModuleType(name)
    module.parse = lambda page: {"madeup": [{"key": page.key}]}
    monkeypatch.setitem(sys.modules, name, module)
    assert adapter.parse(RawPage("hkjc", "madeup", "k1", "u", b"")) == {"madeup": [{"key": "k1"}]}
    assert adapter.parse(RawPage("hkjc", "madeup", "k2", "u", b"")) == {"madeup": [{"key": "k2"}]}


@pytest.mark.parametrize(
    "page_type", ["nosuchpagetype", "../results", "results.evil", "", "Results"]
)
def test_parse_of_an_unknown_page_type_is_a_clear_error(adapter, page_type):
    with pytest.raises(NotImplementedError, match="no HKJC parser"):
        adapter.parse(RawPage("hkjc", page_type, "k", "u", b""))


# ---------------------------------------------------------------- provisional pages


@pytest.mark.parametrize(
    ("age", "horse", "corunning"),
    [
        (0, True, True),
        (1, True, True),
        (7, True, True),  # HORSE_REFRESH_DAYS
        (8, False, True),
        (14, False, True),  # CORUNNING_REFRESH_DAYS
        (15, False, False),
        (400, False, False),
    ],
)
def test_pages_that_are_provisional_after_a_meeting_stay_mutable_for_their_window(
    age, horse, corunning
):
    grouped = followups_on(RACE_DAY + timedelta(days=age))
    assert {t.mutable for t in grouped["horse"]} == {horse}
    assert {t.mutable for t in grouped["corunning"]} == {corunning}
    # results and sectionals are final once the meeting is over: fetched exactly once
    assert {t.mutable for t in grouped["results"] + grouped["sectional"]} == {age == 0}


def test_the_backfill_of_old_meetings_is_fetch_once_for_every_page_type():
    for race in (rp.DEAD_HEAT, rp.ARCHIVE_2010, rp.ARCHIVE_2004, rp.ABANDONED):
        grouped = followups_on(TODAY, race)
        assert grouped
        assert not any(t.mutable for tasks in grouped.values() for t in tasks), race.key


# the old meeting the returning horse (LIVE's winner) ran in before
_HORSE = "HK_2023_X101"
OLD_RUN = replace(rp.DEAD_HEAT, runners=(replace(rp.DEAD_HEAT.runners[0], horse_id=_HORSE),
                                         *rp.DEAD_HEAT.runners[1:]))  # fmt: skip


def _mark_all(queue: FetchQueue, tasks, status: str = "fetched", digest: str = "a") -> None:
    for task in tasks:
        queue.mark(task, status, http_status=200, sha256=digest * 64, n_bytes=1)


def test_a_new_meeting_refreshes_a_horse_profile_that_was_cached_before_it(tmp_path):
    # The profile form table is the only source of rating-at-run and gear. A profile cached during
    # the backfill and queued fetch-once would never show any later run of the horse.
    with FetchQueue(tmp_path / "manifest.sqlite") as queue:
        # the backfill (long after the old meeting) cached the profile; the row is fetch-once
        old = followups_on(TODAY, OLD_RUN)["horse"]
        assert _HORSE in {t.key for t in old} and not any(t.mutable for t in old)
        assert queue.enqueue(old) == len(old)
        _mark_all(queue, queue.next_pending(50))
        assert queue.next_pending(50) == []

        # results of the horse's next race, processed the morning after: the profile is owed a
        # refetch
        queue.enqueue(followups_on(RACE_DAY + timedelta(days=1))["horse"])
        pending = {t.key: t for t in queue.next_pending(50)}
        assert _HORSE in pending and pending[_HORSE].mutable  # mutable = bypass the cache
        assert pending[_HORSE].priority == rank(RACE_DAY, 4)
        _mark_all(queue, pending.values(), digest="b")
        assert queue.next_pending(50) == []

        # the same results page walked again long after: one final fetch, then fetch-once again
        much_later = followups_on(RACE_DAY + timedelta(days=90))["horse"]
        queue.enqueue(much_later)
        final = {t.key: t for t in queue.next_pending(50)}
        assert _HORSE in final
        _mark_all(queue, final.values(), digest="c")
        queue.enqueue(much_later)
        queue.enqueue(followups_on(RACE_DAY + timedelta(days=91))["horse"])
        assert queue.next_pending(50) == []
        assert queue.get("horse", _HORSE)["mutable"] == 0


def test_an_old_meeting_never_resets_a_cached_horse_profile(tmp_path):
    # politeness: the backfill walks tens of thousands of old results pages; none of them may
    # trigger a second profile fetch
    with FetchQueue(tmp_path / "manifest.sqlite") as queue:
        horses = followups_on(TODAY, rp.DEAD_HEAT)["horse"]
        assert queue.enqueue(horses) == len(horses) == len(horse_ids(rp.DEAD_HEAT))
        _mark_all(queue, queue.next_pending(50))
        assert queue.enqueue(followups_on(TODAY + timedelta(days=1), rp.DEAD_HEAT)["horse"]) == 0
        assert queue.next_pending(50) == []


def test_unpublished_comments_on_running_can_be_fetched_again_inside_the_window(tmp_path):
    # A few days after a meeting the page still had no comments; classified "empty" and queued
    # fetch-once, the comments (and the per-race gear) of every recent meeting would be lost.
    with FetchQueue(tmp_path / "manifest.sqlite") as queue:
        (first,) = followups_on(RACE_DAY + timedelta(days=1))["corunning"]
        queue.enqueue([first])
        _mark_all(queue, [first], status="empty")
        assert queue.next_pending(5) == []
        # the results page is walked again a week later: the page is requested again
        (again,) = followups_on(RACE_DAY + timedelta(days=8))["corunning"]
        queue.enqueue([again])
        assert [(t.key, t.mutable) for t in queue.next_pending(5)] == [(first.key, True)]


# The store build labels meetings, races and runners with a season through the adapter's optional
# `season_start_month()` hook (core/store/build.py), falling back to calendar years. HKJC seasons
# start in September (hkjc.yaml `season.start_month: 9`); without the hook the rebuilt tables
# disagreed with the copied ones on 925 meetings, 8765 races and 110495 runners.
def test_the_store_build_reads_the_hkjc_season_start_month():
    assert build.season_start_month("hkjc") == parse_common.season_start_month() == 9


@pytest.mark.parametrize(
    "day, label",
    [
        (date(2010, 1, 1), "2009/10"),  # the copied-vs-rebuilt mismatch from the rebuild check
        (date(2026, 9, 23), "2026/27"),
        (date(2026, 9, 1), "2026/27"),  # first day of a season
        (date(2026, 8, 31), "2025/26"),  # last day of the previous one
        (date(2026, 12, 31), "2026/27"),
        (date(1999, 12, 31), "1999/00"),  # century wrap
    ],
)
def test_the_store_build_labels_hkjc_seasons_from_september(day, label):
    assert build._season_of(day, build.season_start_month("hkjc")) == label
