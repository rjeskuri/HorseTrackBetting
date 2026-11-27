"""JRA source adapter for the history backfill (ported from HKTrack's `sources/jra/adapter.py`
without its live methods): task generation, classification and parser dispatch over netkeiba.

Job expansion: calendar months -> the race list of every race day up to today -> one results page
per JRA race on that list. Results pages lead nowhere yet (horse pages are a later ticket).

Pedigree: `pedigree_card` tasks (the five-run card of a race, which prints every runner's sire, dam
and dam sire) are planned, not discovered: the adapter only builds, classifies and parses them.

Priorities give a newest-first, complete-by-date backfill: the calendar months come first (newest
month first), then every task of a race date outranks every task of an older date, the day's race
list before its results. Registered as "jra" in `core.ingest.adapters.ADAPTERS`.

Store hooks: `result_published_utc()` and `result_unpublished_utc()` give the store build the odds
logger's first-seen result publication times and their lower bounds (ticket #225);
`going_page_type`, `parse_going` and `attach_rail_setting` let it fill `races.rail_setting` from
the going pages (馬場情報) in the raw cache (ticket #253; fetched by `jra.going_fetch`).

Win sales: `win_sales` pages (JRADB 単勝・複勝 pages, fetched by `jra.win_sales_fetch`) are
classified and parsed by `jra.win_sales` into the store's `win_sales` table (ticket #278).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, datetime, timezone
from importlib import import_module
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from horsetrack.core import paths
from horsetrack.core.config import load_source
from horsetrack.core.ingest.types import (
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FETCHED,
    FetchTask,
    ParsedPage,
    RawPage,
)
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.jra import going_fetch, win_sales
from horsetrack.countries.japan.jra.netkeiba import (
    common,
    parse_fixture,
    parse_pedigree_card,
    parse_racelist,
    parse_results,
    urls,
)

SOURCE = common.SOURCE
PARSERS = "horsetrack.countries.japan.jra.netkeiba"

# Enumeration first: date priorities are toordinal() * 10 + rank, about 7.4 million today. Months
# add year * 12 + month on top, so a newer month outranks an older one.
FIXTURE_PRIORITY = 10_000_000
# Rank within one race date: the list first, it is what discovers the results.
_RANK = {"racelist": 6, "results": 5}
PEDIGREE_PAGE_TYPE = "pedigree_card"
RESULTS_SIDECAR = ".results.json"  # <VENUE>.results.json beside a day's odds log (#225)


class JRAAdapter:
    source = SOURCE
    pedigree_page_type = PEDIGREE_PAGE_TYPE

    def __init__(self, today: Callable[[], date] | None = None) -> None:
        self._today = today or _local_today

    # ------------------------------------------------------------ task generation

    def fixture_tasks(self, start: date, end: date) -> list[FetchTask]:
        """One calendar task per month. Past months never change; the current and later ones do."""
        today = self._today()
        tasks = []
        year, month = start.year, start.month
        while (year, month) <= (end.year, end.month):
            tasks.append(
                FetchTask(
                    SOURCE,
                    "fixture",
                    f"{year:04d}-{month:02d}",
                    urls.calendar(year, month),
                    year,
                    priority=FIXTURE_PRIORITY + year * 12 + month,
                    mutable=(year, month) >= (today.year, today.month),
                )
            )
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return tasks

    def meeting_tasks(self, fixture_pages: Iterable[RawPage]) -> list[FetchTask]:
        """The race list of every race day up to today; later days come back when their month is
        refetched."""
        today = self._today()
        tasks: dict[str, FetchTask] = {}
        for page in fixture_pages:
            for day in parse_fixture.parse(page)["race_days"]:
                race_date = day["race_date"]
                if race_date <= today:
                    key = race_date.isoformat()
                    if key not in tasks:
                        tasks[key] = _task(
                            "racelist", key, urls.racelist(race_date), race_date, today
                        )
        return sorted(tasks.values(), key=lambda t: t.key)

    def followup_tasks(self, parsed: ParsedPage) -> list[FetchTask]:
        meta = (parsed.get("_meta") or [{}])[0]
        if meta.get("page_type") == going_fetch.MENU_PAGE_TYPE:
            return going_fetch.followups(parsed)
        if meta.get("page_type") != "racelist":
            return []
        today = self._today()
        tasks: dict[str, FetchTask] = {}
        for row in parsed.get("race_links") or []:
            # checked again here: the id becomes a URL and a cache file name
            race_id = urls.jra_race_id(row.get("race_id"))
            if race_id is not None and race_id not in tasks:
                tasks[race_id] = _task(
                    "results", race_id, urls.results(race_id), row["race_date"], today
                )
        return list(tasks.values())

    def pedigree_task(self, race_key: str, race_date: date) -> FetchTask:
        """The pedigree card of one race (`race_key`: a 12-digit JRA race id; ValueError for
        anything else). Priority = the date's ordinal, without the x 10 of the dated tasks: below
        every results, race-list and calendar task of any date, so pedigree never delays the
        history backfill; newest date first among them."""
        url = urls.pedigree_card(race_key)
        return FetchTask(SOURCE, PEDIGREE_PAGE_TYPE, race_key, url, int(race_key[:4]),
                         priority=race_date.toordinal())  # fmt: skip

    # ------------------------------------------------------------ classification

    def classify(self, page_type: str, body: bytes) -> str:
        """Decide by content, never by HTTP status: netkeiba serves missing races as 200."""
        if page_type == "results":
            return parse_results.classify(body)
        if page_type in going_fetch.PAGE_TYPES:
            return going_fetch.classify(body)
        if page_type == win_sales.PAGE_TYPE:
            return _classify_win_sales(body)
        soup = BeautifulSoup(body, "lxml")
        if page_type == PEDIGREE_PAGE_TYPE:
            # every planned race comes from our own history or is asked for by id: a card without
            # runners is the site failing (blocked), not a race without data
            return STATUS_FETCHED if parse_pedigree_card.runner_rows(soup) else STATUS_BLOCKED
        if page_type == "racelist":
            if soup.select_one(parse_racelist.LIST_SELECTOR) is None:
                return STATUS_BLOCKED
            return STATUS_FETCHED if parse_racelist.race_hrefs(soup) else STATUS_EMPTY
        if page_type == "fixture":
            if soup.select_one(parse_fixture.CALENDAR_SELECTOR) is None:
                return STATUS_BLOCKED
            return STATUS_FETCHED if parse_fixture.race_dates(soup) else STATUS_EMPTY
        return STATUS_BLOCKED

    # ------------------------------------------------------------ parsing

    def parse(self, page: RawPage) -> ParsedPage:
        """Dispatch to netkeiba.parse_<page_type>, so new parser modules plug in untouched. Going
        pages (#253) are not tables: the tab page gives its follow-ups, the going page nothing."""
        if page.page_type == going_fetch.MENU_PAGE_TYPE:
            return going_fetch.parse_menu(page)
        if page.page_type == going_fetch.GOING_PAGE_TYPE:
            return {}
        if page.page_type == win_sales.PAGE_TYPE:
            return _parse_win_sales(page)
        module = _parser_module(page.page_type)
        if module is not None:
            return module.parse(page)
        raise NotImplementedError(
            f"no JRA parser for page_type {page.page_type!r} (expected {PARSERS}.parse_"
            f"{page.page_type})"
        )

    def parser_version(self, page_type: str) -> str | None:
        """PARSER_VERSION of netkeiba.parse_<page_type> ("0" when it declares none); None when the
        page type has no parser, which the store build skips."""
        if page_type == win_sales.PAGE_TYPE:
            return win_sales.PARSER_VERSION
        module = _parser_module(page_type)
        return None if module is None else str(getattr(module, "PARSER_VERSION", "0"))

    # ------------------------------------------------------------ store hooks

    going_page_type = going_fetch.GOING_PAGE_TYPE

    # pandas is imported here, not at the top: parse workers import this module and need none
    def parse_going(self, page: RawPage):
        """A cached going page as `going.parse_going_page` rows (ValueError for an error page);
        the manifest's fetch time, else the epoch, is the `fetched_utc`."""
        from horsetrack.countries.japan.jra import going

        fetched = page.fetched_utc or datetime.fromtimestamp(0, timezone.utc)
        return going.parse_going_page(page.body, fetched_utc=fetched)

    def attach_rail_setting(self, races, going_rows):
        """`races` with `rail_setting` from the concatenated `parse_going` rows."""
        from horsetrack.countries.japan.jra import going

        return going.attach_rail_setting(races, going_rows)

    def result_published_utc(self) -> dict[str, datetime]:
        """race_id -> when its official result was first seen published (aware UTC): every
        results sidecar the odds logger wrote under odds_snapshots/jra (ticket #225). The store
        build fills `races.result_published_utc` from it."""
        return _logged_results(odds_log.read_published)

    def result_unpublished_utc(self) -> dict[str, datetime]:
        """race_id -> its lower bound (aware UTC), the last logger poll that did not show its
        result, for the bounded races of every results sidecar only (a late sighting has none).
        The store build fills `races.result_unpublished_utc` from it."""
        return _logged_results(odds_log.read_unpublished)


def _classify_win_sales(body: bytes) -> str:
    """A 単勝・複勝 page names its race; a block or error page does not (#278)."""
    try:
        win_sales.page_race(win_sales.soup(body))
    except ValueError:
        return STATUS_BLOCKED
    return STATUS_FETCHED


def _parse_win_sales(page: RawPage) -> ParsedPage:
    """A cached 単勝・複勝 page as one `win_sales` row, none without a Win figure (#278). The
    manifest's fetch time dates a live page's clock; a page without one is an error, never dated
    by a guess."""
    if page.fetched_utc is None:
        raise ValueError(f"win_sales/{page.key}: the manifest has no fetch time")
    record = win_sales.parse_win_sales(page.body, fetched_utc=page.fetched_utc)
    return {} if record is None else {win_sales.PAGE_TYPE: [record]}


def _logged_results(read) -> dict[str, datetime]:
    """race_id -> the non-null instant `read(sidecar)` gives per race, over every results sidecar
    the odds logger wrote under odds_snapshots/jra."""
    root = paths.data_dir() / "odds_snapshots" / SOURCE
    out: dict[str, datetime] = {}
    for sidecar in sorted(root.glob(f"*/*{RESULTS_SIDECAR}")):
        venue = sidecar.name.removesuffix(RESULTS_SIDECAR)
        try:
            race_day = date.fromisoformat(sidecar.parent.name)
        except ValueError:
            continue  # not a race-date folder
        for number, at in read(sidecar).items():
            if at is not None:
                out[common.store_race_id(race_day, venue, number)] = at
    return out


def _parser_module(page_type: str):
    """The netkeiba `parse_<page_type>` module, None when there is none."""
    module_name = f"{PARSERS}.parse_{page_type}"
    if not page_type.isidentifier():
        return None
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise  # the parser exists but one of ITS imports is missing: do not hide that
        return None


def _task(page_type: str, key: str, url: str, race_date: date, today: date) -> FetchTask:
    # A day still being raced (or not yet) changes under us; everything final is fetched once.
    return FetchTask(
        SOURCE,
        page_type,
        key,
        url,
        race_date.year,
        priority=race_date.toordinal() * 10 + _RANK[page_type],
        mutable=race_date >= today,
    )


def _local_today() -> date:
    """Race dates are local dates, so "today" is the date in the source's time zone."""
    tz = load_source(common.COUNTRY, SOURCE).timezone
    return datetime.now(ZoneInfo(tz)).date()


ADAPTER = JRAAdapter()
