"""Builders for minimal synthetic nar.netkeiba.com pages (ticket #37): the day's race list
(`top/race_list_sub.html`), the 単勝・複勝 odds page (`odds/?type=b1`) on sale and before sales, and
a pair-pool odds page (`b5` ワイド, `b4` 馬連).

Each page holds only the structure `countries/japan/nar/pages.py` reads, in the shape of the
"Logger spec for NAR" in docs/research/nar-data-access.md, plus a few decoys. Pages come back as
UTF-8 bytes, as the site sends them. Every name, id and number is invented; no real page, name or
scraped value is copied here.

Verified markup (seen in the captured pages, per the research doc). Header cells are printed as
the site prints them: a two-line label is split by `<br>` (`馬<br>番`), so a parser must not compare
the raw cell text with the one-line label.
- race list: `ul.RaceList_ProvinceSelect li a` venue tabs (href has `kaisai_id`, the first `li`
  `.Active`), one `dl.RaceList_DataList` per meeting, `p.RaceList_DataTitle` as
  `<small>第N回</small>venue<small>N日目</small>` plus an empty `small.icon-nighter` on a nighter,
  races `li.RaceList_DataItem > a[href*="shutuba.html?race_id="]`, `.Race_Num` holding a hidden
  `span.MyRace_List_Item` before `NR`, post time in the first `span` of `.RaceData`, runners as a
  bare `N頭` text node.
- `b1` on sale (seen on a finished race only): `#odds_tan_block` (h2 単勝) and
  `#odds_fuku_block` (h2 複勝), each a `table.RaceOdds_HorseList_Table` (no thead or tbody) with
  headers `th.Waku` 枠, `th.W31` `馬<br>番`, `th.Mark` 印, 馬名, オッズ; 馬番 a `td` with no class;
  price `td.Odds span.Odds`; 複勝 a range `lo - hi`.
- `b1` before sales: an h2 `予想オッズ（単勝）` and one `table.RaceOdds_HorseList_Table.Ninki`, its
  header row `tr.col_label`: `th.Ninki` `人<br>気`, `th.Waku` `馬<br>番`, `th.Mark` 印,
  `th.Horse_Name` 馬名, `th.Odds` `予想<br>オッズ`; the number in `td.Waku<bracket>`, the forecast
  bare in `td.Odds`; and a notice that real odds replace the forecast when sales open.

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
# Each column's header cell as the b1 page on sale prints it (verified).
TAN_HEADERS = {
    "枠": '<th class="Waku">枠</th>',
    "馬番": '<th class="W31">馬<br>番</th>',
    "印": '<th class="Mark">印</th>',
    "馬名": "<th>馬名</th>",
    "オッズ": "<th>オッズ</th>",
}
# The header cells of the before-sales 予想オッズ table (verified).
FORECAST_HEADERS = (
    '<th class="Ninki">人<br>気</th>'
    '<th class="Waku">馬<br>番</th>'
    '<th class="Mark">印</th>'
    '<th class="Horse_Name">馬名</th>'
    '<th class="Odds">予想<br>オッズ</th>'
)

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
        '<div class="Race_Num">\n<span>\n'
        f'<span class="MyRace_List_Item" id="myrace_{rid}" style="display: none;"></span>\n'
        f"{race.number}R\n</span>\n</div>"
        '<div class="RaceList_ItemContent">'
        f'<div class="RaceList_ItemTitle">ギジ特別{race.number}</div>'
        f'<div class="RaceData">\n<span>{race.post}</span>\n'
        f'<span class="Dart">ダ1400m</span>\n{race.runners}頭\n</div>'
        "</div></a></li>"
    )


def race_list(day: date, meetings: Sequence[Meeting], *, decoys: bool = True) -> bytes:
    """The race_list_sub page of `day`: venue tabs, then one `dl` per meeting with its races.
    With `decoys`: a link to the next day's list, and per meeting a 払戻一覧 item that is not a
    race (no shutuba link)."""
    tabs = "".join(
        ('<li class="Active">' if i == 0 else "<li>")
        + f'<a href="?kaisai_id={kaisai_id(day, m.venue_number)}&amp;kaisai_date={day:%Y%m%d}'
        f'&amp;rf=race_list">{html.escape(m.name)}</a></li>'
        for i, m in enumerate(meetings)
    )
    parts = [f'<ul class="RaceList_ProvinceSelect">{tabs}</ul>']
    if decoys:
        nxt = date.fromordinal(day.toordinal() + 1)
        parts.append(
            '<div class="RaceList_Nav">'
            f'<a href="race_list.html?kaisai_date={nxt:%Y%m%d}">翌日</a></div>'
        )
    for m in meetings:
        nighter = '<small class="icon-nighter"></small>' if m.nighter else ""
        title = (
            f"\n<small>第{m.round_no}回</small>{html.escape(m.name)}"
            f"<small>{m.day_no}日目</small>{nighter}\n"
        )
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


@dataclass(frozen=True)
class RaceNav:
    """The race navigation a real odds page carries (ticket #38; shape seen on captured b1 pages,
    no content copied). A page links to far more races than its own:
    - venue tabs `div.RaceKaisaiWrap ul.Col li a[href="?race_id=..."]`: one race per venue racing
      that day (other venues included), the page's own as `li.Active`;
    - race numbers `div.RaceNumWrap ul.fc li a[href="?race_id=..."][title="NR"]`: every race of
      the page's meeting, the page's own as `li.Active`;
    - the page menu `ul.RaceMainMenu li.race_navi_*`, whose links name the page's own race.
    So a page's own race is its active nav entry (and its race_navi links), not "every race_id= on
    the page"."""

    own: str  # the page's own 12-digit race id
    meeting: Sequence[str]  # the race ids of the page's meeting, race 1 first
    venues: Sequence[str]  # one race id per venue tab, the page's own among them


def _race_nav(nav: RaceNav | None) -> str:
    if nav is None:
        return ""
    own = nav.own
    tabs = "".join(
        ('<li class="Active">' if rid == own else "<li>")
        + f'<a href="?race_id={rid}">ギジ{rid[4:6]}</a></li>'
        for rid in nav.venues
    )
    numbers = "".join(
        ('<li class="Active">' if rid == own else "<li>")
        + f'<a href="?race_id={rid}" title="{int(rid[10:])}R">{int(rid[10:])}R</a></li>'
        for rid in nav.meeting
    )
    menu = (
        '<ul class="RaceMainMenu">'
        f'<li id="a-navi_shutuba" class="race_navi_shutuba"><a href="/race/shutuba.html?'
        f'race_id={own}">出馬表</a></li>'
        f'<li id="a-navi_odds" class="race_navi_odds"><a href="/odds/?race_id={own}&amp;type=b1">'
        "オッズ</a></li>"
        f'<li id="a-navi_result"><a href="/race/result.html?race_id={own}">結果</a></li></ul>'
    )
    return (
        '<div class="RaceMainColumn">'
        f'<div class="RaceKaisaiWrap"><ul class="Col Col">{tabs}</ul></div>'
        f'<div class="RaceNumWrap"><ul class="fc">{numbers}</ul></div></div>' + menu
    )


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
    head = "".join(TAN_HEADERS[c] for c in columns)
    rows = []
    for number, spec in runners.items():
        bracket = bracket_of(number, field_size)
        cells = {
            "枠": f'<td class="Waku{bracket}">{bracket}</td>',
            "馬番": f"<td>{number}</td>",
            "印": f'<td class="Mark_User"><span class="MarkIcon"><div id="mymarkb{which}_{number}">'
            "</div></span></td>",
            "馬名": f'<td class="Horse_Name"><a href="#">ギジノウマ{number}</a></td>',
            "オッズ": _odds_cell(spec),
        }
        tr = '<tr class="Cancel">' if _is_scratch(spec) else "<tr>"  # the class: a GUESS
        rows.append(f"{tr}{''.join(cells[c] for c in columns)}</tr>")
    # as the site sends it: no thead or tbody, the header row a plain tr of th
    return f'<table class="RaceOdds_HorseList_Table"><tr>{head}</tr>{"".join(rows)}</table>'


def odds_b1(
    race_id: str,
    runners: Mapping[int, tuple[Price, Price]],
    *,
    updated: str | None = "14:27",
    votes: Sequence[tuple[str, int]] = (),
    columns: Sequence[str] = TAN_COLUMNS,
    race_nav: RaceNav | None = None,
) -> bytes:
    """The 単勝・複勝 page of a race on sale. `runners`: number -> (win price, place price);
    a place price is a (lo, hi) range, a scratched runner prints 取消 or 除外 in both.
    `columns`: the header order of both tables (the parser keys by label). `race_nav`: the page's
    race navigation (none by default, as #37's pages)."""
    size = len(runners)
    win = {n: w for n, (w, _) in runners.items()}
    place = {n: p for n, (_, p) in runners.items()}
    body = (
        _race_nav(race_nav)
        + _nav(race_id)
        + _update(updated)
        + '<div id="odds_tan_block" class="RaceOdds_HorseList">'
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
    race_id: str,
    forecast: Mapping[int, float],
    *,
    in_tan_block: bool = False,
    ninki_class: bool = True,
    race_nav: RaceNav | None = None,
) -> bytes:
    """The 単勝・複勝 page before sales open: netkeiba's 予想オッズ table (`.Ninki`), ordered by
    forecast favouritism, its forecast in a `td.Odds` cell as a real price is (a trap for a loose
    parser). `ninki_class=False` drops the table's `.Ninki` class, so only its `予想<br>オッズ`
    header tells the page apart. `in_tan_block` wraps it in `#odds_tan_block` with an h2 単勝 (a
    GUESS that the container is shared)."""
    order = sorted(forecast, key=lambda n: forecast[n])
    size = len(forecast)
    rows = "".join(
        f'<tr><td class="Ninki">{rank}</td><td class="Waku{bracket_of(n, size)}">{n}</td>'
        f'<td class="Mark_User"><span class="MarkIcon"><div id="mymarkb1_{n}"></div></span></td>'
        f'<td class="Horse_Name"><a href="#">ギジノウマ{n}</a></td>'
        f'<td class="Odds">{forecast[n]:.1f}</td></tr>'
        for rank, n in enumerate(order, start=1)
    )
    cls = "RaceOdds_HorseList_Table Ninki" if ninki_class else "RaceOdds_HorseList_Table"
    table = f'<table class="{cls}"><tr class="col_label">{FORECAST_HEADERS}</tr>{rows}</table>'
    notice = '<p class="Caption">※発売開始後、予想オッズは実際のオッズに切り替わります。</p>'
    inner = "<h2>予想オッズ（単勝）</h2>" + notice + table
    if in_tan_block:
        inner = (
            '<div id="odds_tan_block" class="RaceOdds_HorseList">'
            f'<div class="RaceOdds_HorseList_Title"><h2>単勝</h2></div>{inner}</div>'
        )
    body = _race_nav(race_nav) + _nav(race_id) + inner
    return page(body, imitates="nar.netkeiba.com odds b1, before sales")


def odds_pairs(
    race_id: str,
    pairs: Mapping[tuple[int, int], Price],
    *,
    label: str = "ワイド",
    updated: str | None = "14:27",
    votes: Sequence[tuple[str, int]] = (),
    race_nav: RaceNav | None = None,
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
        _race_nav(race_nav)
        + _nav(race_id)
        + _update(updated)
        + f'<div id="odds_pair_block" class="GraphOdds"><h2>{label}</h2>{tables}</div>'
        + _votes(votes)
    )
    return page(body, imitates=f"nar.netkeiba.com odds page, {label} (markup guessed)")
