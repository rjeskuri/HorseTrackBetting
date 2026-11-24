"""Builders for minimal synthetic JRADB pages (ticket #21): menus, race lists and odds pages.

Each page holds only the structure `official_pages` reads (the selectors of HKTrack's
`sources/jra/official.py`) and the labels it matches (pool names, 取消 / 除外, 発走済, the clock
and date forms); every value is invented. Pages come back as bytes encoded cp932, as the site
sends them. No real page, name or scraped value is copied here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

ODDS_PATH = "/JRADB/accessO.html"
CARD_PATH = "/JRADB/accessD.html"
OTHER_PATH = "/JRADB/access5.html"
SCRATCH_WORDS = ("取消", "除外")
_WEEKDAY = "月火水木金土日"

# A cell spec is a number (one figure), a (min, max) pair (a range), or text printed as is.
Cell = float | tuple[float, float] | str


def page(body: str) -> bytes:
    head = '<head><meta charset="Shift_JIS"><title>synthetic</title></head>'
    return f'<!doctype html><html lang="ja">{head}<body>{body}</body></html>'.encode("cp932")


def on_click(path: str, cname: str) -> str:
    return f"return doAction('{path}', '{cname}');"


def action_link(path: str, cname: str, inner: str) -> str:
    return f'<a href="#" onclick="{on_click(path, cname)}">{inner}</a>'


def href_link(path: str, cname: str, inner: str) -> str:
    return f'<a href="{path}?CNAME={cname}">{inner}</a>'


def day_heading(day: date) -> str:
    """11月28日（土曜）"""
    return f"{day.month}月{day.day}日（{_WEEKDAY[day.weekday()]}曜）"


def date_line(day: date, meeting: str) -> str:
    """2026年11月28日（土曜） 5回東京7日"""
    return f"{day.year}年{day_heading(day)} {meeting}"


def race_header(day: date | None, meeting: str, race_no: int | None = None) -> str:
    line = ""
    if day is not None:
        line = (
            '<div class="date_line"><div class="inner">'
            f'<div class="cell date">{date_line(day, meeting)}</div>'
            "</div></div>"
        )
    number = ""
    if race_no is not None:
        number = f'<div class="race_number"><img alt="{race_no}レース" src="n.png"></div>'
    return f'<div class="race_header">{line}{number}</div>'


# ---------------------------------------------------------------- menus


def menu(
    days: Mapping[date, Sequence[tuple[str, str]]],
    path: str = ODDS_PATH,
    *,
    decoys: bool = True,
) -> bytes:
    """An 出馬表 or オッズ menu: per day, a panel with its heading (11月28日（土曜）) and one link
    per meeting `(name like 5回東京7日, cname)` to `path`. With `decoys`, each day's panel also
    has a meeting-named link to another path, and a second panel under the same heading lists
    single races (東京11R) the way the real menus feature them."""
    panels = []
    for day, meetings in days.items():
        links = [f"<li>{action_link(path, cname, name)}</li>" for name, cname in meetings]
        if decoys and meetings:
            name, _ = meetings[0]
            links.append(f"<li>{action_link(OTHER_PATH, 'pw17x0/00', name)}</li>")
        panels.append(
            f'<div class="panel"><h3>{day_heading(day)}</h3><ul>{"".join(links)}</ul></div>'
        )
    if decoys:
        for day, meetings in days.items():
            races = []
            for i, (name, _) in enumerate(meetings):
                venue = name.split("回", 1)[1].rstrip("0123456789日")
                races.append(f"<li>{action_link(path, f'pw99x{i}/AA', f'{venue}11R')}</li>")
            panels.append(
                f'<div class="panel"><h3>{day_heading(day)}</h3><ul>{"".join(races)}</ul></div>'
            )
    return page("".join(panels))


# ---------------------------------------------------------------- race lists


def odds_race_list(
    day: date,
    meeting: str,
    races: Mapping[int, Mapping[str, str]],
    times: Mapping[int, str] | None = None,
) -> bytes:
    """オッズ race list: one row per race, `races[n]` = {link text: cname} of its odds buttons
    ("単勝 複勝" prints as two spans in one button). The race-number cell links the first one."""
    rows = []
    for race_no, buttons in races.items():
        first = next(iter(buttons.values()))
        number = action_link(ODDS_PATH, first, f'<img alt="{race_no}レース" src="b.png">')
        cells = []
        for text, cname in buttons.items():
            inner = "".join(f"<span>{word}</span>" for word in text.split())
            cells.append(f"<div>{action_link(ODDS_PATH, cname, f'<div>{inner}</div>')}</div>")
        time = (times or {}).get(race_no, "")
        rows.append(
            f'<tr><th class="race_num" scope="row">{number}</th><td class="time">{time}</td>'
            f'<td class="odds"><div class="btn_list">{"".join(cells)}</div></td></tr>'
        )
    return page(race_header(day, meeting) + _table(rows, columns=2))


def card_race_list(day: date, meeting: str, times: Mapping[int, str]) -> bytes:
    """出馬表 race list: one row per race, its race-number image, its time cell as given
    (a clock like 9時55分, or text such as 発走済) and a link to its card."""
    rows = []
    for race_no, time in times.items():
        cname = f"pw01dde{race_no:02d}{day:%Y%m%d}/{race_no:02X}"
        number = href_link(CARD_PATH, cname, f'<img alt="{race_no}レース" src="s.png">')
        rows.append(
            f'<tr><th class="race_num" scope="row">{number}</th><td class="time">{time}</td>'
            f'<td class="syutsuba">{href_link(CARD_PATH, cname, "<span>x</span>")}</td></tr>'
        )
    return page(race_header(day, meeting) + _table(rows, columns=1))


def _table(rows: Sequence[str], *, columns: int) -> str:
    """A race-list table: a header row of empty `th` cells (no race number), then `rows`."""
    head = "".join('<th scope="col"></th>' for _ in range(columns))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


# ---------------------------------------------------------------- odds pages


def _range(low: float, high: float) -> str:
    return (
        f'<span class="inner"><span class="min">{low:,.1f}</span><span class="cap">-</span>'
        f'<span class="max">{high:,.1f}</span></span>'
    )


def _cell(cls: str, spec: Cell) -> str:
    if isinstance(spec, str):
        extra = " cancel" if spec in SCRATCH_WORDS else ""
        return f'<td class="{cls}{extra}"><span class="inner"><span>{spec}</span></span></td>'
    if isinstance(spec, tuple):
        return f'<td class="{cls}">{_range(*spec)}</td>'
    return f'<td class="{cls}"><strong>{spec:,.1f}</strong></td>'


def _refresh(state: str | None) -> str:
    if state is None:
        return ""
    return f'<div class="refresh_line"><div class="cell time"><strong>{state}</strong></div></div>'


def _votes(votes: Sequence[tuple[str, int]]) -> str:
    rows = "".join(f'<tr><th scope="row">{label}</th><td>{n:,}</td></tr>' for label, n in votes)
    return f'<div id="votes"><table><tbody>{rows}</tbody></table></div>'


def win_place(
    day: date,
    meeting: str,
    race_no: int,
    runners: Mapping[int, tuple[Cell, Cell]],
    *,
    state: str | None,
    votes: Sequence[tuple[str, int]] = (),
) -> bytes:
    """単勝・複勝 page: `runners[n] = (win cell, place cell)`; a place cell is (min, max), a win
    cell a number; either may be text (取消, 除外, or anything not a number)."""
    rows = ['<tr><th scope="col"></th><th scope="col"></th><th scope="col"></th></tr>']
    for number, (win, place) in runners.items():
        rows.append(
            f'<tr><td class="waku"><img alt="w" src="w.png"></td><td class="num">{number}</td>'
            f'<td class="horse"><a href="#">h{number}</a></td>'
            f"{_cell('odds_tan', win)}{_cell('odds_fuku', place)}</tr>"
        )
    body = f'<div id="odds_list"><table><tbody>{"".join(rows)}</tbody></table></div>'
    return page(race_header(day, meeting, race_no) + _refresh(state) + body + _votes(votes))


def pairs(
    day: date,
    meeting: str,
    race_no: int,
    odds: Mapping[tuple[int, int], Cell],
    *,
    state: str | None,
    votes: Sequence[tuple[str, int]] = (),
) -> bytes:
    """馬連 or ワイド page: one table per first horse (its `caption`), one row per second horse
    (`th`) with the price `td`. `odds[(first, second)]` is a number (馬連), a (min, max) range
    (ワイド) or text (取消). Keys are laid out as given, so (12, 3) prints under caption 12."""
    by_first: dict[int, list[tuple[int, Cell]]] = defaultdict(list)
    for (first, second), spec in odds.items():
        by_first[first].append((second, spec))
    tables = []
    for first, seconds in by_first.items():
        rows = "".join(
            f'<tr><th scope="row">{second}</th>{_cell("odds", spec)}</tr>'
            for second, spec in seconds
        )
        tables.append(f"<table><caption>{first}</caption><tbody>{rows}</tbody></table>")
    body = f'<div id="odds_list">{"".join(tables)}</div>'
    return page(race_header(day, meeting, race_no) + _refresh(state) + body + _votes(votes))


def no_header_page() -> bytes:
    """A page with no date line and no race number (what an error page looks like)."""
    return page("<div><p>x</p></div>")
