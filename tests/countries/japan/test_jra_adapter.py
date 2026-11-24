"""Ticket #54: the JRA source adapter that drives the history backfill (ported from HKTrack's
`sources/jra/adapter.py` without its live methods).

Chain: calendar months (race.netkeiba.com/top/calendar.html) -> one race-list task per race day up
to today (db.netkeiba.com/race/list/YYYYMMDD/) -> one results task per JRA race on that list
(db.netkeiba.com/race/<id>/). Pages are the synthetic builders of netkeiba_pages.py and
netkeiba_results_pages.py; every expected key, URL and id is written out here.

Contract pinned here (the coder follows it):
- `horsetrack.countries.japan.jra.adapter.ADAPTER` satisfies `core.ingest.adapters.SourceAdapter`,
  `source == "jra"`, and `get_adapter("jra")` returns it. No live methods (upcoming_tasks,
  live_odds). Importing the module pulls in neither pandas nor the odds logger.
- `type(ADAPTER)(today=<callable returning a date>)` builds an adapter on an injected clock;
  without it, "today" is the date in Asia/Tokyo (jra.yaml timezone).
- Tasks: `FetchTask.source == "jra"`; page types `fixture` (key YYYY-MM), `racelist` (key
  YYYY-MM-DD), `results` (key = the 12-digit race id), `pedigree_card` (key = the race id, built by
  `pedigree_task(race_key, race_date)`); `year` = the month's or the race date's year;
  `mutable` = the month or day is today's or later.
- Priorities (higher runs first): every calendar month above every dated task, a newer month
  above an older one; a newer race date's tasks above an older date's, the day's race list above
  its results; pedigree cards below every dated task, newest first.
- `classify(page_type, body)` decides by content only (fetched / empty / blocked).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import netkeiba_pages as pages
import netkeiba_results_pages as results_pages
import pytest

from horsetrack.core.ingest.adapters import SourceAdapter, get_adapter
from horsetrack.core.ingest.types import FetchTask, RawPage

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.japan.jra.adapter import ADAPTER

DB = "https://db.netkeiba.com"
RACE = "https://race.netkeiba.com"
TODAY = date(2026, 11, 18)

NOV_2025_RACE_DAYS = [1, 2, 8, 9, 15, 16, 22, 23, 24, 29, 30]
FEB_2024_RACE_DAYS = [3, 4, 10, 11, 17, 18, 24, 25, 29]  # 29: a leap day

# JRA venue number (race-id characters 5-6) -> jra.yaml code
VENUES = {"01": "SAP", "03": "FKS", "04": "NII", "05": "TOK", "08": "KYO", "09": "HSN"}


def adapter_on(today: date):
    return type(ADAPTER)(today=lambda: today)


@pytest.fixture
def adapter():
    return adapter_on(TODAY)


def fixture_page(year: int, month: int, days, **kw) -> RawPage:
    url = f"{RACE}/top/calendar.html?year={year}&month={month}"
    return RawPage("jra", "fixture", f"{year:04d}-{month:02d}", url,
                   pages.calendar(year, month, days, **kw))  # fmt: skip


def racelist_page(day: date, venues) -> RawPage:
    url = f"{DB}/race/list/{day:%Y%m%d}/"
    return RawPage("jra", "racelist", day.isoformat(), url, pages.race_list(day, venues))


def results_raw(page: results_pages.ResultPage) -> RawPage:
    url = f"{DB}/race/{page.key}/"
    return RawPage("jra", "results", page.key, url, results_pages.render(page))


def race_tasks(adapter, page: RawPage) -> list[FetchTask]:
    """What the runner does with a non-fixture page: parse, then ask for follow-ups."""
    return adapter.followup_tasks(adapter.parse(page))


def months_between(start: date, end: date) -> list[tuple[int, int]]:
    out, (y, m) = [], (start.year, start.month)
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def ids(year: int, venue: str, kai: int, day: int, races=range(1, 13)) -> list[str]:
    return [f"{year}{venue}{kai:02d}{day:02d}{r:02d}" for r in races]


# ---------------------------------------------------------------- wiring


def test_the_adapter_conforms_to_the_protocol():
    assert isinstance(ADAPTER, SourceAdapter)
    assert ADAPTER.source == "jra"
    assert get_adapter("jra") is ADAPTER
    built = adapter_on(date(2020, 1, 1))
    assert isinstance(built, SourceAdapter) and built.source == "jra"
    # the backfill adapter only: the live methods stay out of this port
    for live in ("upcoming_tasks", "live_odds"):
        assert not hasattr(ADAPTER, live), live

    # importing the adapter pulls in neither pandas nor the odds logger
    code = (
        "import sys\n"
        "import horsetrack.countries.japan.jra.adapter as m\n"
        "assert m.ADAPTER.source == 'jra'\n"
        "bad = sorted(n for n in sys.modules if n == 'pandas' or n.startswith('pandas.')\n"
        "             or n.startswith('horsetrack.countries.japan.logger'))\n"
        "print(','.join(bad))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=60)  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ""


# ---------------------------------------------------------------- the chain


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2010, 1, 1), TODAY),
        (date(2019, 11, 15), date(2020, 2, 3)),  # across a year end, mid-month bounds
        (date(2024, 2, 29), date(2024, 2, 29)),  # one day: its month
        (date(2026, 9, 30), date(2027, 1, 2)),  # around today: the current and later months change
    ],
)
def test_month_fixture_to_racelist_to_results_chain(adapter, start, end):
    # 1. one calendar task per month of [start, end], each with its own month's URL
    months = sorted(adapter.fixture_tasks(start, end), key=lambda t: t.key)
    expected_months = months_between(start, end)
    assert [t.key for t in months] == [f"{y:04d}-{m:02d}" for y, m in expected_months]
    for task, (year, month) in zip(months, expected_months, strict=True):
        assert (task.source, task.page_type, task.year) == ("jra", "fixture", year)
        parts = urlsplit(task.url)
        query = parse_qs(parts.query)
        assert (parts.scheme, parts.netloc, parts.path) == (
            "https", "race.netkeiba.com", "/top/calendar.html",
        )  # fmt: skip
        assert (int(query["year"][0]), int(query["month"][0])) == (year, month)
        assert task.mutable == ((year, month) >= (TODAY.year, TODAY.month)), task.key
    assert adapter.fixture_tasks(end + timedelta(days=40), end) == []

    # 2. calendar pages -> one race-list task per race day, deduplicated across pages
    calendars = [
        fixture_page(2025, 11, NOV_2025_RACE_DAYS),
        fixture_page(2024, 2, FEB_2024_RACE_DAYS),
        fixture_page(2025, 11, NOV_2025_RACE_DAYS),  # the same month fetched twice
    ]
    lists = adapter.meeting_tasks(calendars)
    days = sorted(
        [date(2025, 11, d) for d in NOV_2025_RACE_DAYS]
        + [date(2024, 2, d) for d in FEB_2024_RACE_DAYS]
    )
    assert sorted(t.key for t in lists) == [d.isoformat() for d in days]
    for task in lists:
        day = date.fromisoformat(task.key)
        assert (task.source, task.page_type, task.year) == ("jra", "racelist", day.year)
        assert task.url == f"{DB}/race/list/{day:%Y%m%d}/"
        assert not task.mutable  # a past day's list is final
    assert adapter.meeting_tasks([fixture_page(2025, 11, [])]) == []

    # 3. a race list -> one results task per JRA race on it; local (NAR) and overseas races dropped
    for day, jra_ids in [
        (date(2025, 11, 23), ids(2025, "05", 5, 5) + ids(2025, "08", 4, 6)),
        (date(2010, 1, 5), ids(2010, "06", 1, 1, range(1, 9))),
        (date(2024, 2, 29), ids(2024, "09", 1, 4, [3, 11])),
    ]:
        venues = [
            pages.jra("東京", jra_ids[: len(jra_ids) // 2]),
            pages.nar("大井", [f"{day.year}44{day:%m%d}{r:02d}" for r in (1, 2)]),
            pages.jra("京都", jra_ids[len(jra_ids) // 2 :]),
            pages.overseas("凱旋門", [f"{day.year}FPa00f0{r}" for r in (1, 2)]),
        ]
        races = race_tasks(adapter, racelist_page(day, venues))
        assert sorted(t.key for t in races) == sorted(jra_ids), day
        for task in races:
            assert (task.source, task.page_type, task.year) == ("jra", "results", day.year)
            assert task.url == f"{DB}/race/{task.key}/"
            assert not task.mutable
    # a list with only local and overseas races yields nothing
    day = date(2025, 11, 25)
    others = [pages.nar("大井", ["202544112501"]), pages.overseas("香港", ["2025H1a01e01"])]
    assert race_tasks(adapter, racelist_page(day, others)) == []

    # 4. a results page leads nowhere (horse pages are a later ticket)
    assert race_tasks(adapter, results_raw(results_pages.PLAIN)) == []


def test_meeting_tasks_stop_at_today():
    """Race days after today have no results yet: not queued (they come back when the month is
    refetched). Today itself is queued, and its list and results stay mutable."""
    november = fixture_page(2025, 11, NOV_2025_RACE_DAYS)
    for today, expected in [
        (date(2025, 11, 16), [1, 2, 8, 9, 15, 16]),
        (date(2025, 11, 21), [1, 2, 8, 9, 15, 16]),
        (date(2025, 11, 23), [1, 2, 8, 9, 15, 16, 22, 23]),
        (date(2025, 11, 30), NOV_2025_RACE_DAYS),
        (date(2025, 12, 1), NOV_2025_RACE_DAYS),
        (date(2025, 11, 1), [1]),
        (date(2025, 10, 31), []),
    ]:
        tasks = adapter_on(today).meeting_tasks([november])
        assert sorted(t.key for t in tasks) == [date(2025, 11, d).isoformat() for d in expected], (
            today
        )
        for task in tasks:
            assert task.mutable == (date.fromisoformat(task.key) >= today), (today, task.key)

    # the results of a day still being raced are mutable too, those of the day before are not
    on_23 = adapter_on(date(2025, 11, 23))
    for day, mutable in [(date(2025, 11, 23), True), (date(2025, 11, 22), False)]:
        page = racelist_page(day, [pages.jra("京都", ids(day.year, "08", 4, day.day - 17))])
        assert {t.mutable for t in race_tasks(on_23, page)} == {mutable}, day

    # without an injected clock, "today" is the date in Japan
    tokyo_today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    tomorrow = tokyo_today + timedelta(days=1)
    calendars = [fixture_page(d.year, d.month, [d.day]) for d in (tokyo_today, tomorrow)]
    for default in (ADAPTER, type(ADAPTER)()):
        keys = [t.key for t in default.meeting_tasks(calendars)]
        assert keys == [tokyo_today.isoformat()]


def test_priorities_newest_first(adapter):
    """Newest first and complete by date: enumeration of months first (newest month first), then
    every task of a race date above every task of an older date, the day's list before its
    results; pedigree cards below all of it."""
    months = sorted(adapter.fixture_tasks(date(2010, 1, 1), TODAY), key=lambda t: t.key)
    lists = adapter.meeting_tasks([fixture_page(2025, 11, NOV_2025_RACE_DAYS)])
    by_key = {t.key: t for t in lists}
    results = {}
    for day, kai in [(date(2025, 11, 22), 5), (date(2025, 11, 23), 6), (date(2025, 11, 24), 7)]:
        page = racelist_page(day, [pages.jra("京都", ids(2025, "08", 4, kai))])
        results[day] = race_tasks(adapter, page)
        assert len(results[day]) == 12
    old_day = date(2010, 1, 5)
    page = racelist_page(old_day, [pages.jra("中山", ids(2010, "06", 1, 1))])
    results[old_day] = race_tasks(adapter, page)
    today_list = adapter.meeting_tasks([fixture_page(TODAY.year, TODAY.month, [TODAY.day])])
    dated = lists + today_list + [t for ts in results.values() for t in ts]

    # every month above every dated task, a newer month above an older one
    assert min(t.priority for t in months) > max(t.priority for t in dated)
    assert [t.priority for t in months] == sorted({t.priority for t in months})  # strictly up
    later = adapter.fixture_tasks(date(2026, 12, 1), date(2027, 2, 28))
    assert min(t.priority for t in later) > max(t.priority for t in months)

    # race lists: a newer day above an older one
    ordered = sorted(lists, key=lambda t: t.key)
    assert [t.priority for t in ordered] == sorted({t.priority for t in ordered})
    assert today_list[0].priority > max(t.priority for t in lists)

    # complete by date: a day's results below its own list and above the previous day's list,
    # and below the next day's list
    for day, prev, nxt in [
        (date(2025, 11, 23), "2025-11-22", "2025-11-24"),
        (date(2025, 11, 22), "2025-11-16", "2025-11-23"),
        (date(2025, 11, 24), "2025-11-23", "2025-11-29"),
    ]:
        own = by_key[day.isoformat()].priority
        top = max(t.priority for t in results[day])
        bottom = min(t.priority for t in results[day])
        assert by_key[prev].priority < bottom <= top < own < by_key[nxt].priority, day
    # results of a newer day above the results of an older day
    assert min(t.priority for t in results[date(2025, 11, 23)]) > max(
        t.priority for t in results[date(2025, 11, 22)]
    )
    assert min(t.priority for t in results[date(2025, 11, 22)]) > max(
        t.priority for t in results[old_day]
    )

    # pedigree cards: below every dated task of any date, newest race first among them
    cards = [
        adapter.pedigree_task("202605040205", date(2026, 10, 11)),
        adapter.pedigree_task("202508040611", date(2025, 11, 23)),
        adapter.pedigree_task("201006010101", date(2010, 1, 5)),
    ]
    assert max(t.priority for t in cards) < min(t.priority for t in dated)
    assert [t.priority for t in cards] == sorted({t.priority for t in cards}, reverse=True)


# ---------------------------------------------------------------- classification

_RUNNERS = [
    pages.Runner("2022101101", "シンセサイア", "シンセダム", pages.bracketed("シンセ")),
    pages.Runner("2022101102", "ミドリサイア", "ミドリダム", pages.bracketed("ミドリ")),
]
_LIST_DAY = date(2025, 11, 23)


def _page(kind: str) -> bytes:
    return {
        "calendar": lambda: pages.calendar(2025, 11, NOV_2025_RACE_DAYS),
        "calendar_no_days": lambda: pages.calendar(2025, 11, []),
        "racelist": lambda: pages.race_list(_LIST_DAY, [pages.jra("京都", ids(2025, "08", 4, 6))]),
        "racelist_no_races": lambda: pages.race_list(_LIST_DAY, []),
        "results": lambda: results_pages.render(results_pages.PLAIN),
        "results_jump": lambda: results_pages.render(results_pages.JUMP),
        "results_cut_off": lambda: results_pages.cut_off(results_pages.PLAIN),
        "sentinel": results_pages.sentinel,
        "error": results_pages.error_page,
        "card": lambda: pages.pedigree_card(_RUNNERS, race_date=_LIST_DAY),
        "card_no_runners": lambda: pages.pedigree_card([], race_date=_LIST_DAY),
    }[kind]()


@pytest.mark.parametrize(
    ("page_type", "kind", "expected"),
    [
        # results: the result table decides; the HTTP-200 sentinel for an unknown id is empty
        ("results", "results", "fetched"),
        ("results", "results_jump", "fetched"),
        ("results", "sentinel", "empty"),
        ("results", "results_cut_off", "blocked"),
        ("results", "error", "blocked"),
        ("results", "calendar", "blocked"),
        ("results", "racelist", "blocked"),
        # race list: a list with race links is content, a list without races is empty
        ("racelist", "racelist", "fetched"),
        ("racelist", "racelist_no_races", "empty"),
        ("racelist", "results", "blocked"),  # has race links, but is not a race list
        ("racelist", "sentinel", "blocked"),
        ("racelist", "calendar", "blocked"),
        ("racelist", "error", "blocked"),
        # calendar: race-day links are content, a calendar without any is empty
        ("fixture", "calendar", "fetched"),
        ("fixture", "calendar_no_days", "empty"),
        ("fixture", "racelist", "blocked"),
        ("fixture", "results", "blocked"),
        ("fixture", "error", "blocked"),
        # pedigree card: runners are content; a card without any is the site failing
        ("pedigree_card", "card", "fetched"),
        ("pedigree_card", "card_no_runners", "blocked"),
        ("pedigree_card", "results", "blocked"),
        ("pedigree_card", "error", "blocked"),
        # a page type the adapter does not know is never content
        ("horse", "results", "blocked"),
    ],
)
def test_classify_by_content(page_type, kind, expected):
    assert ADAPTER.classify(page_type, _page(kind)) == expected


# ---------------------------------------------------------------- results and pedigree agree


@pytest.mark.parametrize(
    "result",
    [results_pages.PLAIN, results_pages.SCRATCHED, results_pages.DEMOTED, results_pages.JUMP],
    ids=["plain", "scratched", "demoted", "jump"],
)
def test_horse_ids_match_between_results_and_pedigree(adapter, result):
    """The results page and the five-run card of the same race, both parsed through the adapter,
    key the same race and the same horses: the pedigree joins history by horse_id."""
    printed = results_pages.runners(result)
    expected_ids = {f"{result.horse_stem}{r.horse_no:02d}" for r in printed}
    store_id = f"jra:{result.race_date.isoformat()}:{VENUES[result.key[4:6]]}:{result.race_no:02d}"

    parsed_results = adapter.parse(results_raw(result))
    history = {row["horse_id"] for row in parsed_results["runners"]}
    assert history == expected_ids
    assert {row["race_id"] for row in parsed_results["runners"]} == {store_id}

    # the card task of that race, built from the key the results page carries
    (race,) = parsed_results["races"]
    task = adapter.pedigree_task(race["source_race_key"], result.race_date)
    assert (task.source, task.page_type, task.key) == ("jra", "pedigree_card", result.key)
    assert task.url == f"{RACE}/race/shutuba_past.html?race_id={result.key}"
    assert task.year == result.race_date.year

    card = pages.pedigree_card(
        [
            pages.Runner(
                f"{result.horse_stem}{r.horse_no:02d}",
                f"サイア{r.horse_no}",
                f"ダム{r.horse_no}",
                pages.bracketed(f"ハハチチ{r.horse_no}"),
                scratched=r.placing in ("取", "除"),
            )  # fmt: skip
            for r in printed
        ],
        race_date=result.race_date,
        race_no=result.race_no,
    )
    page = RawPage("jra", task.page_type, task.key, task.url, card)
    assert adapter.classify(task.page_type, card) == "fetched"
    pedigree = adapter.parse(page)["horse_pedigree"]
    assert {row["horse_id"] for row in pedigree} == history
    assert {row["source_race_id"] for row in pedigree} == {store_id}

    with pytest.raises(ValueError):
        adapter.pedigree_task("202546112301", result.race_date)  # a local (NAR) race
