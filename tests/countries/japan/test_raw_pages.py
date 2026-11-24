"""Keeping each capture's raw odds pages (ticket #26), fixing HKTrack bug (h): odds pages were not
cached, card pages were, so a parser bug found later could not be re-run over a logged day.

Contract pinned here (the coder follows it):
- `capture_meeting(...)` in `horsetrack.countries.japan.jra.capture` takes a new keyword
  `raw_pages: str | None = None`. "all" keeps every odds page the capture fetches, "none" keeps
  nothing. None (the default) reads `raw_pages` from `countries/japan/config/logger.yaml` through
  `horsetrack.core.config.load_country_yaml("japan", "logger")`; the shipped file says `raw_pages:
  all`. Nothing else about the capture changes (rows, request counts, exceptions).
- Each odds page body (the `RawPage.body` of the `official_odds` fetch, byte for byte: JRADB serves
  cp932, never decoded or re-encoded) is written gzip-compressed to
  `raw_odds_path(race_date, venue, race_no, code, fetched_utc)`. Menu and race-list pages are not
  kept by this feature.
- `raw_odds_path(race_date, venue, race_no, code, fetched_utc) -> Path` is in the capture module:
  `core.paths.raw_dir("jra") / "official_odds" / <YYYY-MM-DD of race_date> / <VENUE> /
  <HHMMSS>Z-<RR>-<code>.html.gz`, where HHMMSS is the page's fetch time converted to UTC, RR the
  race number zero-padded to 2 digits, and `code` the page code (the cname before the '/', e.g.
  `pw15oli00` for `pw15oli00/6D`). The date directory is the race date, not the fetch date.
- The write is atomic (a temp file beside the target, then a rename: no `.tmp` file is left) and
  never fails the capture: on any OSError the capture still returns its rows and the module's
  logger (`logging.getLogger("horsetrack.countries.japan.jra.capture")`, i.e. `__name__`) logs
  exactly one WARNING for that capture, however many pages could not be written.

Pages are synthetic (jradb_pages.py) and served by jradb_double.py through the real Fetcher; the
expected bytes are what the double's site served. Fetch times are stamped by a Fetcher subclass
here, so file names are known without a real clock.
"""

from __future__ import annotations

import dataclasses
import gzip
import logging
import re
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs

import pytest

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a
# red run reports, and the order lints the same before and after it exists.
from horsetrack.countries.japan.jra.capture import capture_meeting, raw_odds_path

# isort: split
import jradb_double as dbl
from jradb_double import KINDS, POOL_KIND, Meeting, World

from horsetrack.core import config as core_config
from horsetrack.core import paths

UTC = timezone.utc
SAT, SUN = date(2026, 11, 28), date(2026, 11, 29)
ALL_POOLS = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"]
T0 = datetime(2026, 11, 28, 5, 0, 15, tzinfo=UTC)
LOGGER = "horsetrack.countries.japan.jra.capture"
DATE_PART = re.compile(r"\d{4}-\d{2}-\d{2}")


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
    return World(
        {
            SAT: [Meeting("TOK", tok_races), Meeting("KYO", 3, nichi=3)],
            SUN: [Meeting("NAK", 2, 4, 2)],
        }
    )


class StampedFetcher(dbl.RecordingFetcher):
    """The real Fetcher whose pages carry invented fetch times: `start`, + 1 s, ... one step per
    fetch. `stamps` is [(cname, time)] in fetch order."""

    def __init__(self, site: dbl.JradbSite):
        super().__init__(dbl.SOURCE, http=dbl.HTTP, transport=site.transport, sleep=lambda _s: None)
        self.start, self.fetches = T0, 0
        self.stamps: list[tuple[str, datetime]] = []

    def fetch(self, task, *, force=False):
        page = super().fetch(task, force=force)
        stamp = self.start + timedelta(seconds=self.fetches)
        self.fetches += 1
        self.stamps.append((parse_qs(task.body or "")["cname"][0], stamp))
        return dataclasses.replace(page, fetched_utc=stamp)


def serve(world: World) -> tuple[dbl.JradbSite, StampedFetcher]:
    site = dbl.JradbSite(dict(world.pages))
    return site, StampedFetcher(site)


def raw_root() -> Path:
    return paths.raw_dir("jra")


def kept_files() -> list[Path]:
    """Every file under raw/jra with a date directory in its path (the fetcher's own cache is
    keyed by year and page type, never by date)."""
    root = raw_root()
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and any(DATE_PART.fullmatch(part) for part in p.relative_to(root).parts)
    )


def expected_kept(
    world: World,
    fetcher: StampedFetcher,
    day: date,
    venue: str,
    pools: list[str],
    races: list[int] | None = None,
    since: int = 0,
) -> list[Path]:
    """The path of every odds page fetched from stamp `since` on, written out by hand from the
    double's lookups and the stamped fetch times: the layout of the ticket, not the code's."""
    kinds = [k for k in KINDS if any(POOL_KIND[p] == k for p in pools)]
    numbers = races if races is not None else range(1, world.meeting(day, venue).races + 1)
    number_of = {world.odds_link(day, venue, n, k)[1]: n for n in numbers for k in kinds}
    out = []
    for cname, stamp in fetcher.stamps[since:]:
        if cname in number_of:
            name = f"{stamp:%H%M%S}Z-{number_of[cname]:02d}-{cname.split('/')[0]}.html.gz"
            out.append(raw_root() / "official_odds" / day.isoformat() / venue / name)
    return sorted(out)


def gunzipped(path: Path) -> bytes:
    with gzip.open(path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------- the path function


@pytest.mark.parametrize(
    ("race_date", "venue", "race_no", "code", "fetched", "relative"),
    [
        (SAT, "TOK", 3, "pw15oli00", T0, "2026-11-28/TOK/050015Z-03-pw15oli00.html.gz"),
        (
            SUN,
            "NAK",
            12,
            "pw153ou1006020220261129Z",
            datetime(2026, 11, 29, 23, 59, 59, tzinfo=UTC),
            "2026-11-29/NAK/235959Z-12-pw153ou1006020220261129Z.html.gz",
        ),
        (
            SAT,
            "KYO",
            1,
            "pw151ou1008",
            datetime(2026, 11, 28, 0, 0, 0, tzinfo=UTC),
            "2026-11-28/KYO/000000Z-01-pw151ou1008.html.gz",
        ),
        # The directory is the race date, whatever day the fetch fell on.
        (
            SAT,
            "HSN",
            9,
            "pw154ou1009",
            datetime(2026, 11, 29, 0, 10, 0, tzinfo=UTC),
            "2026-11-28/HSN/001000Z-09-pw154ou1009.html.gz",
        ),
        # An aware time in another zone is written in UTC.
        (
            SAT,
            "TOK",
            7,
            "pw155ou1005",
            datetime(2026, 11, 28, 14, 5, 9, tzinfo=timezone(timedelta(hours=9))),
            "2026-11-28/TOK/050509Z-07-pw155ou1005.html.gz",
        ),
    ],
)
def test_raw_odds_path_layout(race_date, venue, race_no, code, fetched, relative):
    got = raw_odds_path(race_date, venue, race_no, code, fetched)
    assert got == paths.raw_dir("jra") / "official_odds" / relative
    assert not got.is_relative_to(Path(__file__).resolve().parents[3]), "never inside the repo"


# ---------------------------------------------------------------- the capture keeps its pages


@pytest.mark.parametrize(
    ("day", "venue", "pools", "races"),
    [
        (SAT, "TOK", ALL_POOLS, None),
        (SAT, "KYO", ALL_POOLS, None),
        (SUN, "NAK", ALL_POOLS, None),
        (SAT, "TOK", ALL_POOLS, [2, 4]),
        (SAT, "TOK", ["WIN", "PLACE"], None),  # one page per race: WIN and PLACE share it
        (SAT, "TOK", ["QUINELLA", "QUINELLA_PLACE"], [1, 3]),  # two pages per race
        (SAT, "TOK", ["QUINELLA_PLACE"], None),
    ],
)
def test_each_odds_page_of_a_capture_is_kept(day, venue, pools, races):
    world = week(tok_races=4)
    site, fetcher = serve(world)
    with fetcher:
        cap = capture_meeting(fetcher, day, venue, pools, races=races, raw_pages="all")
    # One odds page per kind of the asked pools per race: WIN and PLACE share a page.
    count = len({POOL_KIND[p] for p in pools}) * len(
        races or range(1, world.meeting(day, venue).races + 1)
    )
    assert cap.requests == 2 + count  # the menu, the race list, then the odds pages
    expected = expected_kept(world, fetcher, day, venue, pools, races)
    assert len(expected) == count, "every odds page has a file name of its own"
    got = kept_files()
    assert got == expected, "exactly the odds pages, nothing else (no menu, no race list)"
    if pools == ALL_POOLS and races is None:  # R races, 3 pages each
        assert len(got) == 3 * world.meeting(day, venue).races
    for path in got:
        assert not list(path.parent.glob("*.tmp")), "an atomic write leaves no temp file"


def test_menu_and_race_list_pages_are_not_kept():
    world = week(tok_races=2)
    site, fetcher = serve(world)
    with fetcher:
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, raw_pages="all")
    assert len(site.seen) == 2 + 6
    served_codes = {s.cname.split("/")[0] for s in site.seen}
    menu_and_list = {s.cname.split("/")[0] for s in site.seen[:2]}
    kept = kept_files()
    kept_codes = {re.fullmatch(r"\d{6}Z-\d{2}-(.+)\.html\.gz", p.name).group(1) for p in kept}
    assert len(kept) == len(kept_codes) == 6
    assert kept_codes == served_codes - menu_and_list
    assert {p.parts[-4] for p in kept} == {"official_odds"}


def test_two_captures_keep_each_their_own_pages():
    """A page fetched again later is a new file (its fetch time differs), never an overwrite."""
    world = week(tok_races=2)
    site, fetcher = serve(world)
    with fetcher:
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, raw_pages="all")
        mark = len(fetcher.stamps)
        fetcher.start = T0 + timedelta(minutes=5)
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, raw_pages="all")
    first = expected_kept(world, fetcher, SAT, "TOK", ALL_POOLS, since=0)
    first = [p for p in first if p.name < "050500Z"]
    second = expected_kept(world, fetcher, SAT, "TOK", ALL_POOLS, since=mark)
    assert len(first) == len(second) == 6
    assert kept_files() == sorted(first + second)


# ---------------------------------------------------------------- what is written


def test_the_kept_body_is_the_served_bytes():
    world = week(tok_races=3)
    site, fetcher = serve(world)
    with fetcher:
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, raw_pages="all")
        # The site's board moves: the same page codes now serve other bytes, five minutes later.
        mark = len(fetcher.stamps)
        fetcher.start = T0 + timedelta(minutes=5)
        moved: dict[tuple[int, str], bytes] = {}
        for n in (1, 2, 3):
            race = world.race(SAT, "TOK", n)
            race = dataclasses.replace(
                race,
                win={h: p + 1.0 for h, p in race.win.items()},
                place={h: (lo + 1.0, hi + 1.0) for h, (lo, hi) in race.place.items()},
                quinella={k: p + 1.0 for k, p in race.quinella.items()},
                wide={k: (lo + 1.0, hi + 1.0) for k, (lo, hi) in race.wide.items()},
            )
            for kind in KINDS:
                body = world.odds_page(race, kind, SAT, world.meeting(SAT, "TOK").name)
                site.pages[world.odds_link(SAT, "TOK", n, kind)] = body
                moved[(n, kind)] = body
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, raw_pages="all")

    by_cname = {world.odds_link(SAT, "TOK", n, k)[1]: (n, k) for n in (1, 2, 3) for k in KINDS}
    checked, first_gen, second_gen = 0, set(), set()
    for index, (cname, stamp) in enumerate(fetcher.stamps):
        if cname not in by_cname:
            continue
        n, kind = by_cname[cname]
        name = f"{stamp:%H%M%S}Z-{n:02d}-{cname.split('/')[0]}.html.gz"
        path = raw_root() / "official_odds" / "2026-11-28" / "TOK" / name
        assert path.is_file(), name
        assert path.read_bytes()[:2] == b"\x1f\x8b", "gzip-compressed"
        if index < mark:
            expected = world.pages[world.odds_link(SAT, "TOK", n, kind)]  # as first served
            first_gen.add(expected)
        else:
            expected = moved[(n, kind)]
            second_gen.add(expected)
        assert gunzipped(path) == expected, f"{name}: not the bytes the site served"
        checked += 1
    assert checked == 18 == len(kept_files())
    assert len(first_gen) == len(second_gen) == 9 and not first_gen & second_gen
    assert all(not b.isascii() for b in first_gen), "cp932 pages: kept undecoded"


# ---------------------------------------------------------------- raw_pages: none, and the default


def test_raw_pages_none_keeps_nothing():
    world = week(tok_races=3)
    site, fetcher = serve(world)
    with fetcher:
        cap = capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, raw_pages="none")
    assert kept_files() == []
    assert not (raw_root() / "official_odds" / "2026-11-28").exists()
    # The capture itself is unchanged: same rows, same requests.
    assert cap.requests == 2 + 9
    assert dbl.price_rows(cap.rows) == world.expected_rows(SAT, "TOK", ALL_POOLS)


def test_the_shipped_logger_yaml_keeps_everything_by_default():
    core_config.load_country_yaml.cache_clear()
    try:
        assert core_config.load_country_yaml("japan", "logger")["raw_pages"] == "all"
        world = week(tok_races=2)
        site, fetcher = serve(world)
        with fetcher:
            capture_meeting(fetcher, SAT, "TOK", ALL_POOLS)  # no keyword: logger.yaml decides
        assert len(kept_files()) == 6
    finally:
        core_config.load_country_yaml.cache_clear()


@pytest.mark.parametrize(("setting", "files"), [("all", 6), ("none", 0)])
def test_the_default_follows_logger_yaml(setting, files, tmp_path, monkeypatch):
    copy = tmp_path / "countries"
    shutil.copytree(core_config._COUNTRIES / "japan" / "config", copy / "japan" / "config")
    (copy / "japan" / "config" / "logger.yaml").write_text(f"raw_pages: {setting}\n")
    monkeypatch.setattr(core_config, "_COUNTRIES", copy)
    core_config.load_country_yaml.cache_clear()
    core_config.load_source.cache_clear()
    try:
        world = week(tok_races=2)
        site, fetcher = serve(world)
        with fetcher:
            cap = capture_meeting(fetcher, SAT, "TOK", ALL_POOLS)
        assert len(kept_files()) == files
        assert dbl.price_rows(cap.rows) == world.expected_rows(SAT, "TOK", ALL_POOLS)
    finally:
        monkeypatch.undo()
        core_config.load_country_yaml.cache_clear()
        core_config.load_source.cache_clear()


# ---------------------------------------------------------------- a failed write


def test_a_write_failure_does_not_fail_the_capture(caplog):
    world = week(tok_races=3)
    site, fetcher = serve(world)
    # A plain file where the race date's directory must go: every write of the capture fails.
    blocker = raw_root() / "official_odds" / "2026-11-28"
    blocker.parent.mkdir(parents=True)
    blocker.write_text("not a directory")
    caplog.set_level(logging.DEBUG)
    with fetcher:
        cap = capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, raw_pages="all")
    assert cap.requests == 2 + 9
    assert dbl.price_rows(cap.rows) == world.expected_rows(SAT, "TOK", ALL_POOLS)
    assert blocker.read_text() == "not a directory"
    ours = [r for r in caplog.records if r.name == LOGGER]
    warnings = [r for r in ours if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert not [r for r in ours if r.levelno > logging.WARNING]
    assert not list(raw_root().rglob("*.tmp"))


def test_a_later_capture_still_keeps_its_pages_after_a_failed_one():
    """The failure is of that capture only: once the obstacle is gone, pages are kept again."""
    world = week(tok_races=2)
    site, fetcher = serve(world)
    blocker = raw_root() / "official_odds" / "2026-11-28"
    blocker.parent.mkdir(parents=True)
    blocker.write_text("x")
    with fetcher:
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, raw_pages="all")
        assert kept_files() == [blocker]
        blocker.unlink()
        fetcher.start = T0 + timedelta(minutes=1)
        mark = len(fetcher.stamps)
        capture_meeting(fetcher, SAT, "TOK", ALL_POOLS, raw_pages="all")
    expected = expected_kept(world, fetcher, SAT, "TOK", ALL_POOLS, since=mark)
    assert len(expected) == 6
    assert kept_files() == expected
