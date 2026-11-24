"""Builders for minimal synthetic nar.netkeiba.com pages (ticket #37): the day's race list
(`top/race_list_sub.html`), the 単勝・複勝 odds page (`odds/?type=b1`) on sale and before sales, and
a pair-pool odds page (`b5` ワイド, `b4` 馬連).

Each page holds only the structure `countries/japan/nar/pages.py` reads, in the shape of the
"Logger spec for NAR" in docs/research/nar-data-access.md, plus a few decoys. Pages come back as
UTF-8 bytes, as the site sends them. Every name, id and number is invented; no real page, name or
scraped value is copied here.

Verified markup (seen in the captured pages, per the research doc):
- race list: `ul.RaceList_ProvinceSelect li a` venue tabs (href has `kaisai_id`), one
  `dl.RaceList_DataList` per meeting, `p.RaceList_DataTitle` with `small.icon-nighter` on a
  nighter, races `li.RaceList_DataItem > a[href*="shutuba.html?race_id="]`, `.Race_Num`, post
  time in the first `span` of `.RaceData`, runners as `N頭`.
- `b1` on sale (seen on a finished race only): `#odds_tan_block` (h2 単勝) and
  `#odds_fuku_block` (h2 複勝), each a `table.RaceOdds_HorseList_Table` with headers 枠, 馬番, 印,
  馬名, オッズ; 馬番 a `td` with no class; price `td.Odds span.Odds`; 複勝 a range `lo - hi`.
- `b1` before sales: one `table.RaceOdds_HorseList_Table.Ninki` with headers 人気, 馬番, 印,
  馬名, 予想オッズ and a notice that real odds replace the forecast when sales open.

GUESSES (unverified: no page of a race on sale was captured; revise when one is seen):
- Scratched runner: the row gets class `Cancel` and its price cell prints the word (取消 or
  除外) in place of a number, in `td.Odds span.Odds`, on both 単勝 and 複勝.
- Update time: `#odds_update_time`, text `HH:MM現在` (JST), on a page on sale. Absent on the
  finished-race page that was captured, so a page may print none.
- Pool totals: `#odds_vote_block table.RaceOdds_Vote_Table`, one `tr` per pool, `th` the pool
  label and `td` the tickets sold as `12,345票` (one ticket = one 100-yen unit, as JRA's 発売票数),
  plus a `合計` row that is not a pool. The captured finished page showed no totals at all.
- Pair pools (`b5` ワイド, `b4` 馬連): one block `#odds_pair_block` with an h2 naming the pool, then
  one `table.Odds_Table` per first runner: a header `th` with the first runner's number, then
  per second runner a `tr` of `td.Num` (its number) and `td.Odds span.Odds` (one figure, or
  `lo - hi` on ワイド). A pair naming a scratched runner prints the scratch word.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

BASE = "https://nar.netkeiba.com"
SCRATCH_WORDS = ("取消", "除外")
TAN_COLUMNS = ("枠", "馬番", "印", "馬名", "オッズ")

# A price spec: a number (one figure), a (lo, hi) pair (a range), or text printed as is
# (取消, 除外, ---.-).
Price = float | tuple[float, float] | str


def page(body: str, *, imitates: str, title: str = "synthetic") -> bytes:
    """A whole UTF-8 page, starting with the synthetic-fixture marker."""
    return (
        f"<!-- synthetic: {imitates} -->\n"
        '<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8">'
        f"<title>{html.escape(title)}</title></head><body>{body}</body></html>"
    ).encode()


def netkeiba_id(day: date, venue_number: str, race_no: int) -> str:
    """The 12-digit race id: YYYY VV MMDD RR."""
    return f"{day:%Y}{venue_number}{day:%m%d}{race_no:02d}"


def kaisai_id(day: date, venue_number: str) -> str:
    """The 10-digit meeting id: the race id without its race number."""
    return netkeiba_id(day, venue_number, 0)[:10]


# ---------------------------------------------------------------- race list


@dataclass(frozen=True)
class Race:
    number: int
    post: str  # HH:MM, JST, as printed
    runners: int = 10


@dataclass(frozen=True)
class Meeting:
    venue_number: str  # characters 5-6 of the race id
    name: str  # the tab text and the title's venue name
    races: Sequence[Race]
    nighter: bool = False
    round_no: int = 9
    day_no: int = 2


def _race_item(day: date, meeting: Meeting, race: Race) -> str:
    rid = netkeiba_id(day, meeting.venue_number, race.number)
    return (
        '<li class="RaceList_DataItem">'
        f'<a href="../race/shutuba.html?race_id={rid}&amp;rf=race_list">'
        f'<div class="Race_Num"><span>{race.number}R</span></div>'
        '<div class="RaceList_ItemContent">'
        f'<div class="ItemTitle">ギジ特別{race.number}</div>'
        f'<div class="RaceData"><span>{race.post}</span>'
        f'<span class="Dart">ダ1400m</span><span>{race.runners}頭</span></div>'
        "</div></a></li>"
    )


def race_list(day: date, meetings: Sequence[Meeting], *, decoys: bool = True) -> bytes:
    """The race_list_sub page of `day`: venue tabs, then one `dl` per meeting with its races.
    With `decoys`: a link to the next day's list, and per meeting a 払戻一覧 item that is not a
    race (no shutuba link)."""
    tabs = "".join(
        f'<li><a href="?kaisai_date={day:%Y%m%d}&amp;kaisai_id={kaisai_id(day, m.venue_number)}">'
        f"{html.escape(m.name)}</a></li>"
        for m in meetings
    )
    parts = [f'<ul class="RaceList_ProvinceSelect">{tabs}</ul>']
    if decoys:
        nxt = date.fromordinal(day.toordinal() + 1)
        parts.append(
            '<div class="RaceList_Nav">'
            f'<a href="race_list.html?kaisai_date={nxt:%Y%m%d}">翌日</a></div>'
        )
    for m in meetings:
        nighter = ' <small class="icon-nighter">ナイター</small>' if m.nighter else ""
        title = f"第{m.round_no}回 {html.escape(m.name)} {m.day_no}日目{nighter}"
        items = "".join(_race_item(day, m, r) for r in m.races)
        if decoys:
            items += (
                '<li class="RaceList_DataItem"><a href="../top/payback_list.html?'
                f'kaisai_date={day:%Y%m%d}&amp;kaisai_id={kaisai_id(day, m.venue_number)}">'
                "払戻一覧</a></li>"
            )
        parts.append(
            '<dl class="RaceList_DataList">'
            f'<dt class="RaceList_DataHeader"><p class="RaceList_DataTitle">{title}</p></dt>'
            f'<dd class="RaceList_Data"><ul>{items}</ul></dd></dl>'
        )
    return page("".join(parts), imitates="nar.netkeiba.com race_list_sub, one day")


# ---------------------------------------------------------------- odds pages


def _price(spec: Price) -> str:
    if isinstance(spec, tuple):
        lo, hi = spec
        return f"{lo:.1f} - {hi:.1f}"
    if isinstance(spec, str):
        return html.escape(spec)
    return f"{spec:.1f}"


def _odds_cell(spec: Price) -> str:
    return f'<td class="Odds Txt_R"><span class="Odds">{_price(spec)}</span></td>'


def _is_scratch(spec: Price) -> bool:
    return isinstance(spec, str) and spec in SCRATCH_WORDS


def bracket_of(number: int, field_size: int) -> int:
    """The standard bracket (枠) of a runner: one per bracket up to 8 runners, then the last
    brackets take two (or more), filled from bracket 8 down."""
    if field_size <= 8:
        return number
    singles = 16 - field_size  # brackets holding one runner (fields of 9-16)
    if number <= singles:
        return number
    return singles + 1 + (number - singles - 1) // 2


def _update(updated: str | None) -> str:
    """GUESS: the as-of clock, `HH:MM現在`; nothing when `updated` is None."""
    if updated is None:
        return ""
    return f'<div class="Odds_Update"><span id="odds_update_time">{updated}現在</span></div>'


def _votes(votes: Sequence[tuple[str, int]]) -> str:
    """GUESS: tickets sold per pool label, `12,345票`; nothing when `votes` is empty."""
    if not votes:
        return ""
    rows = "".join(f"<tr><th>{label}</th><td>{tickets:,}票</td></tr>" for label, tickets in votes)
    return f'<div id="odds_vote_block"><table class="RaceOdds_Vote_Table">{rows}</table></div>'


def _nav(race_id: str) -> str:
    """The odds-type tabs every odds page carries (their text names pools: a decoy)."""
    tabs = (("b1", "単勝・複勝"), ("b3", "枠連"), ("b4", "馬連"), ("b5", "ワイド"), ("b6", "馬単"))
    links = "".join(
        f'<li><a href="/odds/?race_id={race_id}&amp;type={t}">{label}</a></li>' for t, label in tabs
    )
    return f'<ul class="RaceSubMenu">{links}</ul>'


def _horse_table(
    runners: Mapping[int, Price], which: int, columns: Sequence[str], field_size: int
) -> str:
    head = "".join(f"<th>{c}</th>" for c in columns)
    rows = []
    for number, spec in runners.items():
        bracket = bracket_of(number, field_size)
        cells = {
            "枠": f'<td class="Waku{bracket} Txt_C"><span>{bracket}</span></td>',
            "馬番": f"<td>{number}</td>",
            "印": '<td class="Txt_C"><span class="Mark"></span></td>',
            "馬名": f'<td class="Horse_Name"><a href="#">ギジノウマ{number}</a></td>',
            "オッズ": _odds_cell(spec),
        }
        cls = "HorseList Cancel" if _is_scratch(spec) else "HorseList"
        rows.append(
            f'<tr class="{cls}" id="tr_{which}_{number}">{"".join(cells[c] for c in columns)}</tr>'
        )
    return (
        '<table class="RaceOdds_HorseList_Table">'
        f'<thead><tr class="Header">{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def odds_b1(
    race_id: str,
    runners: Mapping[int, tuple[Price, Price]],
    *,
    updated: str | None = "14:27",
    votes: Sequence[tuple[str, int]] = (),
    columns: Sequence[str] = TAN_COLUMNS,
) -> bytes:
    """The 単勝・複勝 page of a race on sale. `runners`: number -> (win price, place price);
    a place price is a (lo, hi) range, a scratched runner prints 取消 or 除外 in both.
    `columns`: the header order of both tables (the parser keys by label)."""
    size = len(runners)
    win = {n: w for n, (w, _) in runners.items()}
    place = {n: p for n, (_, p) in runners.items()}
    body = (
        _nav(race_id) + _update(updated) + '<div id="odds_tan_block" class="RaceOdds_HorseList">'
        '<div class="RaceOdds_HorseList_Title"><h2>単勝</h2></div>'
        + _horse_table(win, 1, columns, size)
        + "</div>"
        + '<div id="odds_fuku_block" class="RaceOdds_HorseList">'
        '<div class="RaceOdds_HorseList_Title"><h2>複勝</h2></div>'
        + _horse_table(place, 2, columns, size)
        + "</div>"
        + _votes(votes)
    )
    return page(body, imitates="nar.netkeiba.com odds b1, race on sale", title=race_id)


def odds_b1_before_sales(
    race_id: str, forecast: Mapping[int, float], *, in_tan_block: bool = False
) -> bytes:
    """The 単勝・複勝 page before sales open: netkeiba's 予想オッズ table (`.Ninki`), ordered by
    forecast favouritism, with the same `td.Odds span.Odds` cells as real prices (a trap for a
    loose parser). `in_tan_block` wraps it in `#odds_tan_block` with an h2 単勝 (a GUESS that the
    container is shared)."""
    order = sorted(forecast, key=lambda n: forecast[n])
    head = "".join(f"<th>{c}</th>" for c in ("人気", "馬番", "印", "馬名", "予想オッズ"))
    rows = "".join(
        f'<tr class="HorseList"><td class="Ninki Txt_C">{rank}</td><td>{n}</td>'
        f'<td class="Txt_C"><span class="Mark"></span></td>'
        f'<td class="Horse_Name"><a href="#">ギジノウマ{n}</a></td>{_odds_cell(forecast[n])}</tr>'
        for rank, n in enumerate(order, start=1)
    )
    table = (
        '<table class="RaceOdds_HorseList_Table Ninki">'
        f'<thead><tr class="Header">{head}</tr></thead><tbody>{rows}</tbody></table>'
    )
    notice = '<p class="Caption">※発売開始後、予想オッズは実際のオッズに切り替わります。</p>'
    inner = notice + table
    if in_tan_block:
        inner = (
            '<div id="odds_tan_block" class="RaceOdds_HorseList">'
            f'<div class="RaceOdds_HorseList_Title"><h2>単勝</h2></div>{inner}</div>'
        )
    return page(_nav(race_id) + inner, imitates="nar.netkeiba.com odds b1, before sales")


def odds_pairs(
    race_id: str,
    pairs: Mapping[tuple[int, int], Price],
    *,
    label: str = "ワイド",
    updated: str | None = "14:27",
    votes: Sequence[tuple[str, int]] = (),
) -> bytes:
    """GUESS of a pair-pool page (`b5` ワイド, `b4` 馬連/馬複): one table per first runner as the
    pair is given (so (10, 9) prints under 10), each row the second runner and its price."""
    by_first: dict[int, list[tuple[int, Price]]] = {}
    for (a, b), spec in pairs.items():
        by_first.setdefault(a, []).append((b, spec))
    tables = "".join(
        f'<table class="Odds_Table"><tr><th class="Num" colspan="2">{a}</th></tr>'
        + "".join(f'<tr><td class="Num">{b}</td>{_odds_cell(spec)}</tr>' for b, spec in seconds)
        + "</table>"
        for a, seconds in by_first.items()
    )
    body = (
        _nav(race_id)
        + _update(updated)
        + f'<div id="odds_pair_block" class="GraphOdds"><h2>{label}</h2>{tables}</div>'
        + _votes(votes)
    )
    return page(body, imitates=f"nar.netkeiba.com odds page, {label} (markup guessed)")
