"""Per-race Win sales in the store (ticket #278): the `win_sales` table, its build from the cached
odds pages, its as-of read, and the fetch through `core.ingest`.

Contract pinned here (the coder follows it; the parser's is in test_win_sales.py):

- `core.store.schemas.TABLES["win_sales"]`: one row per publication of a race's Win sales, columns
  exactly COLUMNS (kinds as listed), sorted by (race_id, published_utc). Documented in
  docs/specs/store.md.
- The store build reads every manifest page of the JRA source with page_type `win_sales` (any key;
  the page itself says which race it is), parses it with `jra.win_sales.parse_win_sales`
  (`fetched_utc` = the manifest's) and writes `processed/jra/win_sales.parquet`: one row per
  (race_id, published_utc), the latest fetch winning when one publication was fetched twice. A page
  without a Win figure adds no row; a page that cannot be parsed does not stop the build or touch
  the other rows.
- As of a decision time t, `core.store.asof.read_asof(con, "win_sales", t, rules=japan's)` shows
  exactly the rows with published_utc <= t (the shipped availability rule: `column` on
  `published_utc`; tests/countries/japan/test_availability.py pins the rule itself). A final
  figure's published_utc is its post + 30 min, so before then only the live figures show.
- `jra.win_sales_fetch.fetch_win_sales(race_date, *, http, fetcher=None, venues=None)` fetches, for
  every meeting of `race_date` on the JRADB オッズ menu (or only `venues`), each race's 単勝・複勝
  page, every request a FetchTask through the given `core.ingest` fetcher (no other HTTP), and
  records each such page in the JRA manifest as page_type `win_sales`, so the build above reads
  it. It requests no Quinella or Wide page and nothing of another day. Running it twice for the
  same day leaves the table unchanged.

Pages are synthetic (`win_sales_pages`, `jradb_double`); expected rows are written out here.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import going_pages as gp
import jradb_double as jd
import jradb_pages as jp
import pandas as pd
import pytest
import win_sales_pages as wsp

from horsetrack.core import paths
from horsetrack.core.store import build, schemas, tables
from horsetrack.core.store.asof import load_availability, read_asof

SOURCE = "jra"
PAGE_TYPE = "win_sales"
TABLE = "win_sales"
VAR = "HORSETRACK_DATA_DIR"
UTC = timezone.utc
US = timedelta(microseconds=1)
UNIT_STAKE = 100  # yen per ticket (jra.yaml)
FINAL = "最終オッズ"
POST_TO_FINAL = timedelta(minutes=30)  # a final figure is published at its post + 30 min (#278)
COLUMNS = {
    "jurisdiction": schemas.STR,
    "race_id": schemas.STR,
    "race_date": schemas.DATE,
    "venue": schemas.STR,
    "race_no": schemas.INT,
    "win_tickets": schemas.INT,
    "win_sales_yen": schemas.INT,
    "is_final": schemas.BOOL,
    "published_utc": schemas.UTC,
    "fetched_utc": schemas.UTC,
}


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


@pytest.fixture
def data(tmp_path, monkeypatch):
    d = (tmp_path / "data").resolve()
    monkeypatch.setenv(VAR, str(d))
    return d


def add_page(key: str, day: date, meeting: str, race_no: int, state: str, post: str | None,
             tickets: int, fetched: datetime) -> None:  # fmt: skip
    votes = [("単勝", f"{tickets:,}"), ("複勝", f"{tickets * 2:,}")]
    body = wsp.sales_page(day, meeting, race_no, state=state, post=post, votes=votes)
    gp.add_raw(SOURCE, PAGE_TYPE, key, day.year, body, fetched.isoformat())


def stored() -> pd.DataFrame:
    return tables.read_table(SOURCE, TABLE)


def rows_of(frame: pd.DataFrame) -> list[tuple]:
    out = []
    for r in frame.itertuples(index=False):
        out.append((r.race_id, r.race_date, r.venue, int(r.race_no), int(r.win_tickets),
                    int(r.win_sales_yen), bool(r.is_final), r.published_utc.to_pydatetime(),
                    r.fetched_utc.to_pydatetime()))  # fmt: skip
    return out


# ---------------------------------------------------------------- the table


def test_the_store_declares_a_win_sales_table_with_its_publication_time():
    spec = schemas.TABLES[TABLE]
    assert dict(spec.columns) == COLUMNS
    assert tuple(spec.sort_by) == ("race_id", "published_utc")


# ---------------------------------------------------------------- the build

# Tokyo 2026-11-28 R05, post 14:25 JST (05:25 UTC): two live pages and the final one, added out of
# order. Kyoto 2026-11-29 R11, post 15:40 JST: the final page fetched twice. Nakayama 2015-01-05
# R01 (the price era), post 10:05 JST: an archive page fetched in 2026. A final figure is
# published at its post + 30 min (Tokyo 05:55 UTC, Kyoto 07:10 UTC, Nakayama 01:35 UTC).
TOK, KYO, NAK = "jra:2026-11-28:TOK:05", "jra:2026-11-29:KYO:11", "jra:2015-01-05:NAK:01"
SAT, SUN, OLD = date(2026, 11, 28), date(2026, 11, 29), date(2015, 1, 5)
EXPECTED = [  # sorted by (race_id, published_utc)
    (NAK, OLD, "NAK", 1, 8_808, 880_800, True, utc(2015, 1, 5, 1, 35), utc(2026, 11, 20, 10, 0)),
    (TOK, SAT, "TOK", 5, 12_345, 1_234_500, False, utc(2026, 11, 28, 3, 10),
     utc(2026, 11, 28, 3, 12, 40)),
    (TOK, SAT, "TOK", 5, 54_321, 5_432_100, False, utc(2026, 11, 28, 5, 20),
     utc(2026, 11, 28, 5, 20, 41)),
    (TOK, SAT, "TOK", 5, 60_002, 6_000_200, True, utc(2026, 11, 28, 5, 55),
     utc(2026, 11, 28, 7, 0)),
    (KYO, SUN, "KYO", 11, 1_234_567, 123_456_700, True, utc(2026, 11, 29, 7, 10),
     utc(2026, 11, 30, 1, 0)),
]  # fmt: skip


def add_the_pages() -> None:
    add_page("t3", SAT, "5回東京7日", 5, FINAL, "14時25分", 60_002, utc(2026, 11, 28, 7, 0))
    add_page("t1", SAT, "5回東京7日", 5, "12時10分現在", "14時25分", 12_345,
             utc(2026, 11, 28, 3, 12, 40))  # fmt: skip
    add_page("t2", SAT, "5回東京7日", 5, "14時20分現在", "14時25分", 54_321,
             utc(2026, 11, 28, 5, 20, 41))  # fmt: skip
    add_page("k2", SUN, "5回京都9日", 11, FINAL, "15時40分", 1_234_567, utc(2026, 11, 30, 1, 0))
    add_page("k1", SUN, "5回京都9日", 11, FINAL, "15時40分", 1_234_567, utc(2026, 11, 29, 8, 0))
    add_page("n1", OLD, "1回中山1日", 1, FINAL, "10時05分", 8_808, utc(2026, 11, 20, 10, 0))
    # no Win figure (a 馬連 page cached under the type): no row
    quinella = jp.pairs(SAT, "5回東京7日", 6, {(1, 2): 32.1}, state="12時00分現在",
                        votes=[("馬連", 87_724)])  # fmt: skip
    gp.add_raw(SOURCE, PAGE_TYPE, "q1", 2026, quinella, "2026-11-28T03:01:00+00:00")


def test_the_build_writes_one_row_per_publication_of_each_race(data):
    add_the_pages()

    summary = build.build_tables(SOURCE, workers=1)

    assert PAGE_TYPE not in summary.skipped_page_types
    assert rows_of(stored()) == EXPECTED
    assert set(stored()["jurisdiction"]) == {SOURCE}


def test_a_page_added_after_a_build_shows_on_the_next_build(data):
    add_page("n1", OLD, "1回中山1日", 1, FINAL, "10時05分", 8_808, utc(2026, 11, 20, 10, 0))
    build.build_tables(SOURCE, workers=1)
    assert [r[0] for r in rows_of(stored())] == [NAK]

    add_page("k1", SUN, "5回京都9日", 11, FINAL, "15時40分", 1_234_567, utc(2026, 11, 30, 1, 0))
    build.build_tables(SOURCE, workers=1)

    assert rows_of(stored()) == [EXPECTED[0], EXPECTED[-1]]


def test_an_unparseable_page_does_not_stop_the_build(data):
    add_page("k1", SUN, "5回京都9日", 11, FINAL, "15時40分", 1_234_567, utc(2026, 11, 30, 1, 0))
    error_page = "<html><body>エラー</body></html>".encode("cp932")
    gp.add_raw(SOURCE, PAGE_TYPE, "broken", 2026, error_page, "2026-11-29T08:00:00+00:00")

    build.build_tables(SOURCE, workers=1)

    assert rows_of(stored()) == [EXPECTED[-1]]


# ---------------------------------------------------------------- as of a decision time


def test_as_of_t_only_the_sales_published_by_t_are_visible(data):
    add_the_pages()
    build.build_tables(SOURCE, workers=1)
    rules = load_availability("japan")
    published = sorted({row[7] for row in EXPECTED})
    probes = [utc(2010, 1, 1), utc(2030, 1, 1), *published, *(t - US for t in published)]

    con = tables.connect(SOURCE)
    try:
        for t in probes:
            view = read_asof(con, TABLE, t, rules=rules)
            got = sorted(zip(view["race_id"], view["win_tickets"].astype(int), strict=True))
            want = sorted((row[0], row[4]) for row in EXPECTED if row[7] <= t)
            assert got == want, f"as of {t.isoformat()}"
    finally:
        con.close()


def test_until_post_plus_30_only_the_live_figures_are_visible_and_the_final_one_from_then(data):
    add_the_pages()
    build.build_tables(SOURCE, workers=1)
    rules = load_availability("japan")
    post = utc(2026, 11, 28, 5, 25)  # Tokyo R05 at 14:25 JST
    final_from = post + timedelta(minutes=30)

    con = tables.connect(SOURCE)
    try:
        probes = (post, final_from - US, final_from)
        views = {t: read_asof(con, TABLE, t, rules=rules) for t in probes}
    finally:
        con.close()

    for t in (post, final_from - US):  # at the post and up to post + 30 min: live figures only
        tok = views[t][views[t]["race_id"] == TOK]
        assert sorted(tok["win_tickets"].astype(int)) == [12_345, 54_321], t.isoformat()
        assert not tok["is_final"].astype(bool).any(), t.isoformat()
    at = views[final_from]
    assert sorted(at[at["race_id"] == TOK]["win_tickets"].astype(int)) == [12_345, 54_321, 60_002]


# ---------------------------------------------------------------- the fetch through core.ingest


def tickets(race: jd.Race) -> int:
    """Invented 単勝 tickets for a race of the synthetic week: varies by day, venue and race."""
    return (
        3_001
        + race.number * 7_919
        + list(jd.VENUES).index(race.venue) * 104_729
        + (race.day.toordinal() % 13) * 31
    )


class SalesWorld(jd.World):
    """The synthetic JRADB week, its 単勝・複勝 pages final, printing the post time and 発売票数."""

    @staticmethod
    def odds_page(race: jd.Race, kind: str, day: date, meeting: str) -> bytes:
        if kind != "winplace":
            return jd.World.odds_page(race, kind, day, meeting)
        n = tickets(race)
        return wsp.sales_page(day, meeting, race.number, state=FINAL, post=race.post_text,
                              votes=[("単勝", f"{n:,}"), ("複勝", f"{2 * n + 1:,}")])  # fmt: skip


FETCH_SAT, FETCH_SUN = date(2026, 11, 28), date(2026, 11, 29)


def sales_world() -> SalesWorld:
    return SalesWorld(
        {
            FETCH_SAT: [jd.Meeting("TOK", races=3), jd.Meeting("KYO", races=2, kai=5, nichi=7)],
            FETCH_SUN: [jd.Meeting("TOK", races=2, nichi=8)],
        }
    )


def expected_fetched(world: SalesWorld, day: date) -> list[tuple]:
    out = []
    for meeting in world.days[day]:
        for n in range(1, meeting.races + 1):
            race = world.race(day, meeting.venue, n)
            k = tickets(race)
            # final pages: published at the post + 30 min
            out.append((race.race_id, k, k * UNIT_STAKE, True, race.post_utc + POST_TO_FINAL))
    return sorted(out)


def fetched_rows() -> list[tuple]:
    frame = stored()
    return sorted(
        (r.race_id, int(r.win_tickets), int(r.win_sales_yen), bool(r.is_final),
         r.published_utc.to_pydatetime())
        for r in frame.itertuples(index=False)
    )  # fmt: skip


def test_fetching_a_day_puts_each_race_win_sales_in_the_store_through_core_ingest(data):
    from horsetrack.countries.japan.jra import win_sales_fetch

    world = sales_world()
    site = jd.JradbSite(dict(world.pages))
    fetcher = jd.make_fetcher(site)

    win_sales_fetch.fetch_win_sales(FETCH_SAT, http=jd.HTTP, fetcher=fetcher)
    build.build_tables(SOURCE, workers=1)

    expected = expected_fetched(world, FETCH_SAT)
    assert len(expected) == 5 and len({row[1] for row in expected}) == 5
    assert fetched_rows() == expected
    # every request went through the core.ingest fetcher, as a JRA task, and was a real page
    assert fetcher.tasks and all(task.source == SOURCE for task in fetcher.tasks)
    assert len(site.seen) == len(fetcher.tasks)
    assert all(seen.served for seen in site.seen)
    # the 単勝・複勝 page of every race of the day, and no Quinella, Wide or other day's page
    asked = set(site.links())
    winplace = {world.odds_link(FETCH_SAT, m.venue, n, "winplace")
                for m in world.days[FETCH_SAT] for n in range(1, m.races + 1)}  # fmt: skip
    others = {world.odds_link(day, m.venue, n, kind)
              for day, meetings in world.days.items() for m in meetings
              for n in range(1, m.races + 1) for kind in jd.KINDS} - winplace  # fmt: skip
    assert winplace <= asked
    assert not asked & others
    # the manifest holds the pages under the type the build reads
    rows = build.read_manifest(SOURCE)
    assert sum(r["page_type"] == PAGE_TYPE for r in rows) == len(expected)
    assert paths.manifest_path(SOURCE).exists()


def test_fetching_only_some_venues_and_fetching_twice_leaves_one_row_per_publication(data):
    from horsetrack.countries.japan.jra import win_sales_fetch

    world = sales_world()
    site = jd.JradbSite(dict(world.pages))

    win_sales_fetch.fetch_win_sales(FETCH_SAT, http=jd.HTTP, fetcher=jd.make_fetcher(site),
                                    venues=["KYO"])  # fmt: skip
    build.build_tables(SOURCE, workers=1)
    kyoto = [row for row in expected_fetched(world, FETCH_SAT) if ":KYO:" in row[0]]
    assert fetched_rows() == kyoto

    for _ in range(2):
        win_sales_fetch.fetch_win_sales(FETCH_SAT, http=jd.HTTP, fetcher=jd.make_fetcher(site))
    build.build_tables(SOURCE, workers=1)

    assert fetched_rows() == expected_fetched(world, FETCH_SAT)
