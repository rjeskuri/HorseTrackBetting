"""The tier-2 JRADB odds pages (ticket #130): 枠連 (BRACKET_QUINELLA), 馬単 (FORECAST), 3連複
(TRIO), 3連単 (TIERCE), one pool per page, and the WIN5 page. Pure: no network; fetching goes
through `horsetrack.core.ingest` (see `capture.capture_tier2`). Pages are cp932 and untrusted data,
only parsed.

An odds page is laid out as the 馬連 page: one table per leading combination in `#odds_list`, its
caption the leading legs joined by '-' ("3" or "3-12"), one row per last leg (`th`) with the price
(`td`). Rows have the keys of `official_pages.odds_rows` rows (no `odds_hi`: one figure per
ticket). Combinations are numbers joined by ",": FORECAST and TIERCE in finishing order as printed,
TRIO and BRACKET_QUINELLA sorted ascending (a bracket pair may name one bracket twice, "1,1"). A
combination naming a scratched runner (取消 / 除外) gets no row; a cell that is not a number gets a
row with odds None and its text as status_raw. 発売票数 of the page's own pool becomes the
"__pool__" row.

A 3連単 page of an 18-runner field holds 4,896 prices, so odds pages are read with lxml's pull
parser: each table is dropped once read, and only (combination, odds, status) tuples are kept.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from lxml import etree

from horsetrack.countries.japan.jra import official_pages as op

SOURCE = "jra"
WIN5 = "WIN5"
_FEED_CHARS = 1 << 16
_LEG = re.compile(r"([0-9])レース目")
_LEG_RACE = re.compile(r"(\D+?)([0-9]{1,2})レース")
_DATE = re.compile(r"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日")


@dataclass
class Board:
    """What one tier-2 odds page holds: its date line (as `official_pages.page_meeting` reads it),
    race number, refresh-line text, prices in page order and the tickets sold in its pool."""

    meeting: tuple[date, str | None] | None = None
    race_no: int | None = None
    state: str = ""
    prices: list[tuple[str, float | None, str | None]] = field(default_factory=list)
    tickets: int | None = None


def _shape(pool: str) -> tuple[int, bool]:
    """(legs, ordered) of a tier-2 pool, from jra.yaml; ValueError for any other pool."""
    spec = op._config().pools.get(pool)
    if spec is None or spec.legs not in (2, 3) or pool in op.PAIR_POOLS:
        raise ValueError(f"{pool!r} is not a tier-2 pool (枠連, 馬単, 3連複, 3連単)")
    return spec.legs, spec.kind.startswith("ordered")


def _classes(el: Any) -> list[str]:
    return str(el.get("class") or "").split()


def _text(el: Any) -> str:
    return op.clean("".join(el.itertext()))


def _meeting(text: str) -> tuple[date, str | None] | None:
    m = op._PAGE_MEETING.search(text)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), op.venue_code(m.group(4))
    except ValueError:
        return None


def _events(body: bytes) -> Iterator[tuple[str, Any]]:
    """(event, element) of the page, fed to the pull parser a slice at a time."""
    parser = etree.HTMLPullParser(events=("start", "end"))
    text = body.decode("cp932", errors="replace")
    for i in range(0, len(text), _FEED_CHARS):
        parser.feed(text[i : i + _FEED_CHARS])
        yield from parser.read_events()
    parser.close()
    yield from parser.read_events()


def read_board(body: bytes, pool: str) -> Board:
    """The board of one tier-2 odds page of `pool` (ValueError for another pool)."""
    legs, ordered = _shape(pool)
    label_pool = op._config().pool_by_label
    board = Board()
    seen: set[str] = set()
    where: list[str] = []  # the open sections: odds_list, votes, refresh_line, date_line
    lead: list[int] | None = None
    for event, el in _events(body):
        marks = [m for m in ("odds_list", "votes") if el.get("id") == m]
        marks += [m for m in ("refresh_line", "date_line", "race_number") if m in _classes(el)]
        if event == "start":
            where += marks
            continue
        for m in marks:
            where.remove(m)
        tag = el.tag
        if "odds_list" in where and tag == "caption":
            parts = _text(el).split("-")
            lead = [int(p) for p in parts] if all(p.isdigit() for p in parts) else None
            lead = lead if lead is not None and len(lead) == legs - 1 else None
        elif "odds_list" in where and tag == "tr" and lead is not None:
            head, cell = el.find("th"), el.find("td")
            last = _text(head) if head is not None else ""
            if last.isdigit() and cell is not None:
                numbers = [*lead, int(last)]
                combination = ",".join(map(str, numbers if ordered else sorted(numbers)))
                price = _text(cell)
                if combination not in seen and not any(w in price for w in op.SCRATCH_WORDS):
                    seen.add(combination)
                    odds = op._number(price)
                    board.prices.append((combination, odds, None if odds is not None else price))
            el.clear()
        elif "odds_list" in where and tag == "table":
            lead = None
            el.clear()
            while el.getprevious() is not None:  # drop the tables already read
                del el.getparent()[0]
        elif "votes" in where and tag == "tr":
            head, cell = el.find("th"), el.find("td")
            named = label_pool(_text(head)) if head is not None else None
            if named is not None and named.code == pool and cell is not None:
                board.tickets = op._int(_text(cell).replace(",", ""))
        elif "refresh_line" in where and "time" in _classes(el):
            board.state = _text(el)
        elif "date_line" in where and "date" in _classes(el) and board.meeting is None:
            board.meeting = _meeting(_text(el))
        elif "race_number" in where and tag == "img" and board.race_no is None:
            m = op._RACE_NO.fullmatch(op.clean(str(el.get("alt"))))
            board.race_no = int(m.group(1)) if m else None
    return board


def _base(state: str, race_id: str, snapshot_utc: datetime, fetched_utc: datetime) -> dict:
    return {
        "snapshot_utc": snapshot_utc,
        "race_id": race_id,
        "source_update_utc": op.source_update_utc(state, fetched_utc),
        "fetched_utc": fetched_utc,
        "page_state": state or None,
    }


def board_rows(
    board: Board, race_id: str, pool: str, *, snapshot_utc: datetime, fetched_utc: datetime
) -> Iterator[dict[str, Any]]:
    """The odds-log rows of a board: one per price, then the pool total when the page prints it."""
    base = _base(board.state, race_id, snapshot_utc, fetched_utc)
    for combination, odds, status in board.prices:
        yield {**base, "pool": pool, "combination": combination, "odds": odds, "status_raw": status}
    if board.tickets is not None:
        investment = float(board.tickets * op._config().unit_stake)
        yield {**base, "pool": pool, "combination": op.POOL_TOTAL, "odds": None,
               "status_raw": None, "investment": investment}  # fmt: skip


def odds_rows(
    body: bytes, race_id: str, pool: str, *, snapshot_utc: datetime, fetched_utc: datetime
) -> Iterator[dict[str, Any]]:
    """The odds-log rows of one tier-2 odds page of `pool`."""
    board = read_board(body, pool)
    return board_rows(board, race_id, pool, snapshot_utc=snapshot_utc, fetched_utc=fetched_utc)


# ---------------------------------------------------------------- WIN5


def win5_rows(body: bytes, *, snapshot_utc: datetime, fetched_utc: datetime) -> list[dict]:
    """The WIN5 page as one "__pool__" row: race_id jra:<date>:WIN5, `investment` (tickets x unit
    stake, None when not printed), `carryover` (yen, None when not printed) and `legs`, the five
    designated races' race_ids in leg order. ValueError when the page has no date or not five
    legs of known venues."""
    doc = op.soup(body)
    line = doc.select_one(".race_header .date_line .date")
    m = _DATE.search(op.clean(line.get_text(" "))) if line is not None else None
    if not m:
        raise ValueError("WIN5 page without a date line (an error page?)")
    day = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    legs: dict[int, str] = {}
    for tr in doc.select("#win5_list tr"):
        head, cell = tr.find("th"), tr.select_one("td.race")
        leg = _LEG.search(op.clean(head.get_text())) if head is not None else None
        race = _LEG_RACE.search(op.clean(cell.get_text())) if cell is not None else None
        venue = op.venue_code(race.group(1)) if race else None
        if leg and race and venue:
            legs[int(leg.group(1))] = f"{SOURCE}:{day.isoformat()}:{venue}:{int(race.group(2)):02d}"
    if sorted(legs) != [1, 2, 3, 4, 5]:
        raise ValueError(f"WIN5 page for {day}: legs {sorted(legs)}, not 1 to 5")
    config = op._config()
    tickets = None
    for tr in doc.select("#votes tr"):
        named = config.pool_by_label(op.clean(tr.th.get_text())) if tr.th is not None else None
        if named is not None and named.code == WIN5 and tr.td is not None:
            tickets = op._int(tr.td.get_text().replace(",", ""))
    carry = doc.select_one(".carryover td")
    carry_yen = op._int(re.sub(r"[,円\s]", "", carry.get_text())) if carry is not None else None
    state = doc.select_one(".refresh_line .time")
    base = _base(op.clean(state.get_text(" ")) if state is not None else "",
                 f"{SOURCE}:{day.isoformat()}:{WIN5}", snapshot_utc, fetched_utc)  # fmt: skip
    return [
        {
            **base,
            "pool": WIN5,
            "combination": op.POOL_TOTAL,
            "odds": None,
            "status_raw": None,
            "investment": None if tickets is None else float(tickets * config.unit_stake),
            "carryover": None if carry_yen is None else float(carry_yen),
            "legs": [legs[n] for n in range(1, 6)],
        }
    ]
