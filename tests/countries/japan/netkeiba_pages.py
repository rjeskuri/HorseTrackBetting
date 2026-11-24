"""Builders for minimal synthetic netkeiba pages (ticket #51): the monthly calendar, the race list
of a day and the five-run card that prints every runner's pedigree.

Each page holds only the structure the parsers read (the selectors of HKTrack's
`sources/jra/parse_{fixture,racelist,pedigree_card}.py`) plus the decoys the real pages carry:
links outside the table the parser reads, rows that look like runners but are not, past runs
that link another horse. Every name, id and value is invented; no real page is copied here.
Pages come back as bytes in the charset they declare (UTF-8, or EUC-JP for the race list as
db.netkeiba.com sends it).
"""

from __future__ import annotations

import calendar as _calendar
import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

DB = "https://db.netkeiba.com"
RACE = "https://race.netkeiba.com"
ABSENT = object()  # a pedigree field whose div is not printed at all


def page(body: str, *, title: str = "synthetic", encoding: str = "utf-8") -> bytes:
    charset = {"utf-8": "UTF-8", "euc-jp": "EUC-JP"}[encoding]
    head = (
        f'<head><meta http-equiv="Content-Type" content="text/html; charset={charset}">'
        f"<title>{html.escape(title)}</title></head>"
    )
    return f'<!doctype html><html lang="ja">{head}<body>{body}</body></html>'.encode(encoding)


def jp_date(day: date) -> str:
    """2025年11月23日"""
    return f"{day.year}年{day.month}月{day.day}日"


# ---------------------------------------------------------------- calendar


def kaisai_href(day: date) -> str:
    return f"../top/race_list.html?kaisai_date={day:%Y%m%d}"


def calendar(
    year: int,
    month: int,
    race_days: Iterable[int],
    *,
    extra_hrefs: Sequence[str] = (),
    decoys: bool = True,
) -> bytes:
    """A month in `table.Calendar_Table`: one cell per day, the race days linking
    `race_list.html?kaisai_date=YYYYMMDD` once per venue (two links for the same day).
    `extra_hrefs` go inside the table as they are. With `decoys`, a box outside the table links
    another day's race list, and the table carries links that are not race days."""
    days = set(race_days)
    cells = []
    for day_no in range(1, _calendar.monthrange(year, month)[1] + 1):
        day = date(year, month, day_no)
        inner = f'<span class="Day">{day_no}</span>'
        if day_no in days:
            inner += "".join(
                f'<a href="{kaisai_href(day)}"><span class="JyoName">{venue}</span></a>'
                for venue in ("東京", "京都")
            )
        cells.append(f'<td class="RaceCellBox">{inner}</td>')
    cells += [f'<td class="RaceCellBox"><a href="{html.escape(h)}">x</a></td>' for h in extra_hrefs]
    if decoys:
        cells.append(
            f'<td class="Other"><a href="{RACE}/top/calendar.html?year={year}">年</a></td>'
        )
        cells.append('<td class="Other"><a href="#">top</a></td>')
    rows = "".join(f"<tr>{''.join(cells[i : i + 7])}</tr>" for i in range(0, len(cells), 7))
    table = f'<table class="Calendar_Table"><tbody>{rows}</tbody></table>'
    nav = ""
    if decoys:
        other = date(year + 1, 1, 5)
        nav = f'<div class="Header_Nav"><a href="{kaisai_href(other)}">本日のレース</a></div>'
    return page(nav + f'<div class="RaceCalendar">{table}</div>', title=f"{year}年{month}月")


# ---------------------------------------------------------------- race list


def race_href(race_id: str) -> str:
    return f"{DB}/race/{race_id}/"


@dataclass(frozen=True)
class Venue:
    group: str  # 中央, 地方, 海外
    name: str
    hrefs: Sequence[str]


def jra(name: str, race_ids: Iterable[str]) -> Venue:
    return Venue("中央", name, [race_href(r) for r in race_ids])


def nar(name: str, race_ids: Iterable[str]) -> Venue:
    return Venue("地方", name, [race_href(r) for r in race_ids])


def overseas(name: str, race_ids: Iterable[str]) -> Venue:
    return Venue("海外", name, [race_href(r) for r in race_ids])


def race_list(
    day: date,
    venues: Sequence[Venue],
    *,
    heading: bool = True,
    decoys: bool = True,
    encoding: str = "euc-jp",
) -> bytes:
    """The race list of a day: inside `div.race_kaisai_info`, per venue a `race_top_hold_list`
    and per race a `dl.race_top_data_info` whose link is the race (hrefs printed as given).
    With `heading`, the h2 and the title name the day. With `decoys`, the venue heading links a
    race outside any `dl.race_top_data_info`, and a box outside the list links another race."""
    blocks = []
    for venue in venues:
        items = "".join(
            f'<li><dl class="race_top_data_info fc"><dt>{n}R</dt><dd>'
            f'<a href="{html.escape(href)}" title="race">レース{n}</a><br>'
            f"<div>芝1600m</div></dd></dl></li>"
            for n, href in enumerate(venue.hrefs, start=1)
        )
        hold = f"<p>{venue.name}</p>"
        if decoys:
            hold += f'<p><a href="{race_href(f"{day.year}08990199")}">{venue.name}</a></p>'
        blocks.append(
            f"<h3>{venue.group}</h3>"
            f'<div class="race_list fc"><dl class="race_top_hold_list">'
            f'<dt class="race_top_hold_data">{hold}</dt><dd><ul>{items}</ul></dd></dl></div>'
        )
    body = f'<div class="race_kaisai_info">{"".join(blocks)}</div>'
    if heading:
        body = f"<h2>{jp_date(day)}のレース一覧</h2>" + body
    if decoys:
        pickup = race_href(f"{day.year}05990101")
        body += (
            '<div class="pickup"><dl class="race_top_data_info">'
            f'<a href="{pickup}">注目</a></dl></div>'
        )
    title = f"{jp_date(day)}のレース一覧" if heading else "レース一覧"
    return page(body, title=title, encoding=encoding)


# ---------------------------------------------------------------- five-run card


@dataclass(frozen=True)
class Runner:
    """One runner row. `sire`, `dam` and `dam_sire` are the text of their divs exactly as printed
    (the dam sire with its brackets, see `bracketed`); None prints an empty div, ABSENT no div."""

    horse_id: str
    sire: object
    dam: object
    dam_sire: object
    name: str = "シンセティック"
    scratched: bool = False
    past_winners: Sequence[str] = ("2015199901", "2016199902")


def bracketed(name: str, brackets: str = "()") -> str:
    return f"{brackets[0]}{name}{brackets[1]}"


def _div(cls: str, text: object) -> str:
    if text is ABSENT:
        return ""
    return f'<div class="{cls}">{html.escape(str(text)) if text is not None else ""}</div>'


def _runner_row(no: int, runner: Runner, cls: str = "HorseList") -> str:
    if runner.scratched:
        cls += " Cancel"
    info = (
        _div("Horse01 fc", runner.sire)
        + f'<div class="Horse02"><a href="{DB}/horse/{runner.horse_id}" target="_blank">'
        f"{html.escape(runner.name)}</a></div>"
        + _div("Horse03", runner.dam)
        + _div("Horse04", runner.dam_sire)
    )
    past = "".join(
        f'<td class="Past"><div class="Data_Item"><div class="Data07">1着</div>'
        f'<div class="Data12"><a href="{DB}/horse/{winner}/">勝ち馬</a></div></div></td>'
        for winner in runner.past_winners
    )
    return (
        f'<tr class="{cls}" id="tr_{no}"><td class="Waku">{(no + 1) // 2}</td>'
        f'<td class="Umaban">{no}</td><td class="Horse_Info">{info}</td>{past}</tr>'
    )


def pedigree_card(
    runners: Sequence[Runner],
    *,
    race_date: date | None = None,
    venue_name: str = "京都",
    race_no: int = 11,
    decoys: bool = True,
) -> bytes:
    """The five-run card: the runner rows in `div.Shutuba_HorseList table.Shutuba_Table`,
    one `tr.HorseList` per runner (`tr.HorseList.Cancel` when scratched). The title prints
    `race_date` when given. With `decoys`, the runner table also holds a `tr.HorseList`
    without a horse link, and a legend table and a training table outside it carry
    `tr.HorseList` rows that link horses in `.Horse02`."""
    rows = [_runner_row(no, r) for no, r in enumerate(runners, start=1)]
    header = '<tr class="Header"><th>枠</th><th>馬番</th><th>馬名</th><th>前走</th></tr>'
    before = after = ""
    if decoys:
        rows.insert(len(rows) // 2, '<tr class="HorseList"><td class="Horse_Info">-</td></tr>')
        legend = Runner("2017199903", "レジェンドサイア", "レジェンドダム", bracketed("レジェ"))
        before = (
            '<div class="Legend"><table class="Shutuba_Table">'
            f"{_runner_row(0, legend)}</table></div>"
        )
        drill = Runner("2017199904", "トレーニング", "ドリル", bracketed("チョウキョウ"))
        after = (
            '<div class="Training"><table class="Shutuba_Table OikiriTable">'
            f"{_runner_row(99, drill)}</table></div>"
        )
    table = (
        '<div class="Shutuba_HorseList"><table class="Shutuba_Table Shutuba_Past5_Table">'
        f"<tbody>{header}{''.join(rows)}</tbody></table></div>"
    )
    when = f" | {jp_date(race_date)} {venue_name}{race_no}R" if race_date is not None else ""
    title = f"シンセティック賞 出馬表(5走表示){when} レース情報"
    return page(before + table + after, title=title)
