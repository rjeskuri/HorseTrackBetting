"""HKJC source adapter for the history backfill (ported from HKTrack's `sources/hkjc/adapter.py`
without its live methods): task generation, content classification and parser dispatch.

Job expansion is data-driven: fixture months -> race 1 of every local race day -> (from the parsed
race 1) the other races of the card -> (from every parsed race) sectionals, comments on running and
horse profiles.

Priorities give a newest-first, complete-by-date backfill: within one race date the order is
results, horse, sectional, corunning, and every task of a date outranks every task of an older
date. A horse is queued with the date of the race it was seen in; the queue ignores later
duplicates, so with a newest-first crawl each horse keeps its most recent date.

Fetch-once is only right for a page that is final. Two follow-ups are provisional for days after a
meeting: comments on running are published late (every cell still read "No Comments ..." five days
after a meeting), and a horse profile cached before a race lacks that race's form line (rating at
run and gear exist nowhere else). Tasks for such pages are emitted `mutable` inside a refresh
window; the queue resets an existing row when a mutable task arrives and owes it one final fetch
once the task arrives immutable again. Registered as "hkjc" in `core.ingest.adapters.ADAPTERS`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from importlib import import_module
from typing import Any
from zoneinfo import ZoneInfo

from horsetrack.core.config import load_source
from horsetrack.core.ingest.types import (
    STATUS_ABANDONED,
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FETCHED,
    FetchTask,
    ParsedPage,
    RawPage,
)
from horsetrack.countries.hongkong.hkjc import parse_fixture, urls
from horsetrack.countries.hongkong.hkjc.parse_common import (
    COUNTRY,
    NO_INFO_MARKER,
    SOURCE,
    season_start_month,
    strip_not_content,
)
from horsetrack.countries.hongkong.hkjc.parse_results import ABANDONED_MARKERS, safe_id

# Enumeration first: date priorities are toordinal() * 10 + rank, about 7.4 million today.
FIXTURE_PRIORITY = 10_000_000
# Rank within one race date.
_RANK = {"results": 5, "horse": 4, "sectional": 3, "corunning": 2}

SECTIONAL_FROM = date(2008, 6, 1)  # per-horse sectionals do not exist before mid-2008
CORUNNING_FROM = date(2010, 1, 1)  # comments on running are served back to 2010-01-01
MAX_RACES_PER_CARD = 12
# Days after a race during which a follow-up page is still provisional (0 = only on the race day).
HORSE_REFRESH_DAYS = 7  # a new meeting's results refresh the profiles of the horses that ran
CORUNNING_REFRESH_DAYS = 14  # comments on running appear some days after the meeting
_PROVISIONAL_DAYS = {"horse": HORSE_REFRESH_DAYS, "corunning": CORUNNING_REFRESH_DAYS}

NO_COMMENTS_MARKER = "No Comments on Running information"
# Real pages that say "nothing here" in their own words (the site-wide sentinel is NO_INFO_MARKER).
_EMPTY_MARKERS: dict[str, tuple[str, ...]] = {
    "corunning": (NO_COMMENTS_MARKER,),
    "racecard": ("is not available!",),  # "The Race Card is not available!", before declarations
}
_GENERIC_CONTENT = ("commContent",)  # every server-rendered racing.hkjc.com page has this container
# Substrings that must ALL be present for a page type to count as real content.
_CONTENT_MARKERS: dict[str, tuple[str, ...]] = {
    "fixture": ("commContent", 'class="fixture'),
    "results": ("commContent", "localResults"),
    "sectional": ("commContent", "dispalySectionalTime"),  # the typo is HKJC's
    "corunning": ("commContent", 'class="corunning'),
}
_SECTIONAL_TABLE = "race_table"  # absent on header-only sectional pages (before the data starts)
_HORSE_LINK = re.compile(r"horseid=", re.I)

_PAGE_TYPE = re.compile(r"[a-z][a-z0-9_]*")
_HORSE_YEAR = re.compile(r"[A-Za-z]+_(\d{4})_")


class HKJCAdapter:
    source = SOURCE

    def __init__(self, today: Callable[[], date] | None = None) -> None:
        self._today = today or _local_today

    def season_start_month(self) -> int:
        """The store build's season hook: HKJC seasons start in September (hkjc.yaml)."""
        return season_start_month()

    # ------------------------------------------------------------ task generation

    def fixture_tasks(self, start: date, end: date) -> list[FetchTask]:
        """One task per calendar month. Past months never change; the current and later ones do."""
        today = self._today()
        tasks = []
        year, month = start.year, start.month
        while (year, month) <= (end.year, end.month):
            tasks.append(
                FetchTask(
                    SOURCE,
                    "fixture",
                    f"{year:04d}-{month:02d}",
                    urls.fixture(year, month),
                    year,
                    priority=FIXTURE_PRIORITY,
                    mutable=(year, month) >= (today.year, today.month),
                )
            )
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return tasks

    def meeting_tasks(self, fixture_pages: Iterable[RawPage]) -> list[FetchTask]:
        """Race 1 of every local race day up to today. The card's length is unknown until race 1
        is parsed."""
        today = self._today()
        tasks: dict[str, FetchTask] = {}
        for page in fixture_pages:
            for meeting in parse_fixture.parse(page)["meetings"]:
                if meeting["race_date"] <= today:
                    task = self._race_task(
                        "results", meeting["race_date"], meeting["venue"], 1, today
                    )
                    tasks.setdefault(task.key, task)
        return sorted(tasks.values(), key=lambda t: t.key)

    def followup_tasks(self, parsed: ParsedPage) -> list[FetchTask]:
        meta = (parsed.get("_meta") or [{}])[0]
        races = parsed.get("races") or []
        if meta.get("page_type") != "results" or not races:
            return []
        race_date, venue, race_no = _split_race_id(races[0]["race_id"])
        today = self._today()
        tasks = []
        if race_no == 1:
            tasks += [
                self._race_task("results", race_date, venue, n, today)
                for n in meta.get("race_nos") or []
                # tab hrefs are scraped input: only a plausible race number becomes a request
                # and a cache key
                if n != race_no and isinstance(n, int) and 1 <= n <= MAX_RACES_PER_CARD
            ]
        if races[0].get("status") == "abandoned":
            return tasks  # nothing was run: no sectionals, no comments, no form line on horses
        if race_date >= SECTIONAL_FROM:
            tasks.append(self._race_task("sectional", race_date, venue, race_no, today))
        if race_date >= CORUNNING_FROM:
            tasks.append(self._race_task("corunning", race_date, venue, race_no, today))
        return tasks + self._horse_tasks(parsed.get("runners") or [], race_date, today)

    def _race_task(
        self, page_type: str, race_date: date, venue: str, race_no: int, today: date
    ) -> FetchTask:
        if page_type == "results":
            url = urls.results(race_date, venue, race_no)
        elif page_type == "sectional":
            url = urls.sectional(race_date, race_no)
        else:
            url = urls.corunning(race_date, race_no)
        # A meeting that is still being run changes under us; everything final is fetched once.
        return FetchTask(
            SOURCE,
            page_type,
            urls.race_key(race_date, venue, race_no),
            url,
            race_date.year,
            priority=_priority(race_date, page_type),
            mutable=_is_provisional(page_type, race_date, today),
        )

    def _horse_tasks(
        self, runners: list[dict[str, Any]], race_date: date, today: date
    ) -> list[FetchTask]:
        tasks: dict[str, FetchTask] = {}
        for runner in runners:
            # the id becomes a URL parameter and a cache file name
            horse_id = safe_id(runner.get("horse_id"))
            if horse_id is None or horse_id in tasks:
                continue
            registered = _HORSE_YEAR.match(horse_id)
            tasks[horse_id] = FetchTask(
                SOURCE,
                "horse",
                horse_id,
                urls.horse(horse_id),
                int(registered.group(1)) if registered else race_date.year,
                priority=_priority(race_date, "horse"),
                # A profile cached before this race has no form line for it: a recent race
                # refreshes the page.
                mutable=_is_provisional("horse", race_date, today),
            )
        return list(tasks.values())

    # ------------------------------------------------------------ classification

    def classify(self, page_type: str, body: bytes) -> str:
        """Decide by content, never by HTTP status: missing pages and abandoned races are served
        as 200."""
        text = strip_not_content(body.decode("utf-8", errors="replace"))
        specific = _CONTENT_MARKERS.get(page_type)
        has_content = all(marker in text for marker in specific or _GENERIC_CONTENT)
        # The sentinel page also carries the generic container, so for page types without a
        # specific marker the sentinel decides; for the others a stray "No information." inside
        # real content is ignored.
        if NO_INFO_MARKER in text and not (specific and has_content):
            return STATUS_EMPTY
        if not has_content:
            return STATUS_BLOCKED
        if page_type == "results" and any(marker in text for marker in ABANDONED_MARKERS):
            return STATUS_ABANDONED
        if page_type == "corunning" and _some_comments_published(text):
            return STATUS_FETCHED
        if any(marker in text for marker in _EMPTY_MARKERS.get(page_type, ())):
            return STATUS_EMPTY
        if page_type == "sectional" and _SECTIONAL_TABLE not in text:
            return STATUS_EMPTY
        return STATUS_FETCHED

    # ------------------------------------------------------------ parsing

    def parse(self, page: RawPage) -> ParsedPage:
        """Dispatch to hkjc.parse_<page_type>, so new parser modules plug in untouched."""
        if not _PAGE_TYPE.fullmatch(page.page_type or ""):
            raise NotImplementedError(f"no HKJC parser for page_type {page.page_type!r}")
        module_name = f"{__package__}.parse_{page.page_type}"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise  # the parser exists but one of ITS imports is missing: do not hide that
            raise NotImplementedError(
                f"no HKJC parser for page_type {page.page_type!r} (expected module {module_name})"
            ) from exc
        return module.parse(page)


def _split_race_id(rid: str) -> tuple[date, str, int]:
    """hkjc:<YYYY-MM-DD>:<VENUE>:<NN> (parse_common.race_id) -> (date, venue, race number)."""
    _, day, venue, race_no = rid.split(":")
    return date.fromisoformat(day), venue, int(race_no)


def _is_provisional(page_type: str, race_date: date, today: date) -> bool:
    return race_date >= today - timedelta(days=_PROVISIONAL_DAYS.get(page_type, 0))


def _some_comments_published(text: str) -> bool:
    """The "No Comments ..." sentinel is printed PER HORSE. The page is empty only when every
    runner has it (the parser's rule too); one runner without a comment must not throw away the
    comments and gear of the others, because pages classified empty are never parsed."""
    missing = text.count(NO_COMMENTS_MARKER)
    return 0 < missing < len(_HORSE_LINK.findall(text))


def _priority(race_date: date, page_type: str) -> int:
    return race_date.toordinal() * 10 + _RANK[page_type]


def _local_today() -> date:
    """Race dates are local dates, so "today" is the date in the source's time zone."""
    return datetime.now(ZoneInfo(load_source(COUNTRY, SOURCE).timezone)).date()


ADAPTER = HKJCAdapter()
