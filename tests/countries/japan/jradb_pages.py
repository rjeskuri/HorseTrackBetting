"""Builders for minimal synthetic JRADB pages (ticket #21): menus, race lists and odds pages,
(ticket #53) the race card (出馬表, `card`) and (ticket #225) the レース結果 race list.

Each page holds only the structure `official_pages` reads (the selectors of HKTrack's
`sources/jra/official.py`) and the labels it matches (pool names, 取消 / 除外, 発走済, the clock
and date forms); every value is invented. Pages come back as bytes encoded cp932, as the site
sends them. No real page, name or scraped value is copied here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

ODDS_PATH = "/JRADB/accessO.html"
CARD_PATH = "/JRADB/accessD.html"
OTHER_PATH = "/JRADB/access5.html"
RESULTS_PATH = "/JRADB/accessS.html"  # レース結果: the results menu, race lists and result pages
RESULTS_MENU_CNAME = "pw01sli00/AF"  # the レース結果 menu, the site's fixed entry (ticket #225)
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


def list_header(day: date, meeting: str) -> str:
    """The date line of a race list: the real race lists print it in an h2 (and the page's h1),
    not in a `.race_header` as the race pages do (ticket #31)."""
    return f"<h2>{date_line(day, meeting)}</h2>"


# ---------------------------------------------------------------- menus


def menu(
    days: Mapping[date, Sequence[tuple[str, str]]],
    path: str = ODDS_PATH,
    *,
    decoys: bool = True,
    extra: Mapping[date, Sequence[tuple[str, str]]] | None = None,
) -> bytes:
    """An 出馬表 or オッズ menu: per day, a panel with its heading (11月28日（土曜）) and one link
    per meeting `(name like 5回東京7日, cname)` to `path`. With `decoys`, each day's panel also
    has a meeting-named link to another path, and a second panel under the same heading lists
    single races (東京11R) the way the real menus feature them. `extra[day]` adds links
    `(text, cname)` to `path` in that day's panel after its meetings (the WIN5 link, #130)."""
    panels = []
    for day, meetings in days.items():
        links = [f"<li>{action_link(path, cname, name)}</li>" for name, cname in meetings]
        links += [
            f"<li>{action_link(path, c, text)}</li>" for text, c in (extra or {}).get(day, ())
        ]
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
    return page(list_header(day, meeting) + _table(rows, columns=2))


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
    return page(list_header(day, meeting) + _table(rows, columns=1))


def results_race_list(day: date, meeting: str, results: Mapping[int, str | None]) -> bytes:
    """レース結果 race list (ticket #225): the date line, the site's navigation (its レース結果
    menu link, on the results path but in no race row), then one row per race of `results`.
    `results[n]` is the cname of race n's result page: its race-number image links that page on
    RESULTS_PATH. None is a race whose result is not out yet: its image links nothing and its row
    links only its card (出馬表, on CARD_PATH), so only a row's RESULTS_PATH link marks a result."""
    menu_link = action_link(RESULTS_PATH, RESULTS_MENU_CNAME, "レース結果")
    nav = f'<ul class="nav"><li>{menu_link}</li></ul>'
    rows = []
    for race_no, cname in results.items():
        img = f'<img alt="{race_no}レース" src="r.png">'
        card = f"pw01dde{race_no:02d}{day:%Y%m%d}/{race_no:02X}"
        if cname is None:
            number = img
            cell = f'<td class="result">{href_link(CARD_PATH, card, "出馬表")}</td>'
        else:
            number = action_link(RESULTS_PATH, cname, img)
            cell = f'<td class="result">{action_link(RESULTS_PATH, cname, "結果")}</td>'
        rows.append(f'<tr><th class="race_num" scope="row">{number}</th>{cell}</tr>')
    return page(nav + list_header(day, meeting) + _table(rows, columns=1))


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


def legs(
    day: date,
    meeting: str,
    race_no: int,
    odds: Mapping[tuple[int, ...], Cell],
    *,
    state: str | None,
    votes: Sequence[tuple[str, int]] = (),
) -> bytes:
    """枠連, 馬単, 3連複 or 3連単 page (ticket #130), laid out like the 馬連 page: one table per
    leading combination, its `caption` the leading legs joined by '-' ("3" for a two-leg pool,
    "3-12" for a three-leg one), one row per last leg (`th`) with the price `td`.
    `odds[(a, b)]` or `odds[(a, b, c)]` is a number or text (取消, ---.-). Keys are laid out as
    given: (12, 3, 7) prints row 7 under caption 12-3, so an unordered pool's page may list a
    combination in any order. Table and row order follow the order of `odds`."""
    by_lead: dict[tuple[int, ...], list[tuple[int, Cell]]] = defaultdict(list)
    for combination, spec in odds.items():
        by_lead[tuple(combination[:-1])].append((combination[-1], spec))
    tables = []
    for lead, lasts in by_lead.items():
        rows = "".join(
            f'<tr><th scope="row">{last}</th>{_cell("odds", spec)}</tr>' for last, spec in lasts
        )
        caption = "-".join(str(n) for n in lead)
        tables.append(f"<table><caption>{caption}</caption><tbody>{rows}</tbody></table>")
    body = f'<div id="odds_list">{"".join(tables)}</div>'
    return page(race_header(day, meeting, race_no) + _refresh(state) + body + _votes(votes))


def win5(
    day: date,
    races: Sequence[tuple[str, int]],
    *,
    state: str | None,
    tickets: int | None,
    carryover_yen: int | None,
) -> bytes:
    """WIN5 page (ticket #130): the date line (no meeting: WIN5 spans venues), one row per leg
    (`th` 1レース目 ... 5レース目, `td` the venue name as printed and the race, e.g. 東京10レース),
    the tickets sold in the votes table under the label WIN5 (one ticket = one unit stake), and
    the carryover in yen (キャリーオーバー 312,450,000円). None leaves that figure out."""
    header = (
        '<div class="race_header"><div class="date_line"><div class="inner">'
        f'<div class="cell date">{day.year}年{day_heading(day)}</div>'
        "</div></div></div>"
    )
    rows = "".join(
        f'<tr><th scope="row">{i}レース目</th><td class="race">{venue}{race_no}レース</td></tr>'
        for i, (venue, race_no) in enumerate(races, start=1)
    )
    body = f'<div id="win5_list"><table><tbody>{rows}</tbody></table></div>'
    votes = _votes([("WIN5", tickets)]) if tickets is not None else ""
    carry = ""
    if carryover_yen is not None:
        carry = (
            '<div class="carryover"><table><tbody><tr><th scope="row">キャリーオーバー</th>'
            f"<td>{carryover_yen:,}円</td></tr></tbody></table></div>"
        )
    return page(header + _refresh(state) + body + votes + carry)


PAGE_KINDS = (
    "card_menu",
    "odds_menu",
    "card_race_list",
    "odds_race_list",
    "win_place",
    "quinella",
    "wide",
)


def sample_pages(day: date = date(2026, 11, 15), meeting: str = "5回京都4日") -> dict[str, bytes]:
    """One page per kind of `PAGE_KINDS` (the kinds of `official_pages.SELECTORS`), each with every
    structure the parser reads: a race number, a clock, a range, a figure, a scratched cell and
    the votes table (ticket #31)."""
    final = "最終オッズ"
    return {
        "card_menu": menu({day: [(meeting, "pw01drl10082026050420261115/82")]}, CARD_PATH),
        "odds_menu": menu({day: [(meeting, "pw15orl10082026050420261115/FC")]}, ODDS_PATH),
        "card_race_list": card_race_list(day, meeting, {9: "14時20分", 10: "15時01分"}),
        "odds_race_list": odds_race_list(
            day,
            meeting,
            {10: {"単勝 複勝": "pw151x/AA", "馬連": "pw154x/BB", "ワイド": "pw155x/CC"}},
            {10: "15時01分"},
        ),
        "win_place": win_place(
            day,
            meeting,
            10,
            {1: (37.8, (3.0, 7.3)), 6: ("取消", "取消"), 9: (1.7, (1.1, 1.1))},
            state=final,
            votes=[("単勝", 716_227), ("複勝", 648_500)],
        ),
        "quinella": pairs(
            day,
            meeting,
            10,
            {(1, 2): 326.8, (1, 9): 25.3, (2, 9): 63.6},
            state=final,
            votes=[("馬連", 877_724)],
        ),
        "wide": pairs(
            day,
            meeting,
            10,
            {(1, 2): (63.5, 65.9), (1, 9): (6.5, 8.5), (2, 9): (14.3, 18.4)},
            state=final,
            votes=[("ワイド", 764_551)],
        ),
    }


def no_header_page() -> bytes:
    """A page with no date line and no race number (what an error page looks like)."""
    return page("<div><p>x</p></div>")


# ---------------------------------------------------------------- race card (ticket #53)

BRACKET_COLOURS = "白黒赤青黄緑橙桃"  # 枠1白 ... 枠8桃
BLINKERS_ICON = (
    '<span class="horse_icon blinker">'
    '<img src="/JRADB/img/kigo/icon_blinker.png" alt="ブリンカー着用" /></span>'
)


@dataclass(frozen=True)
class CardRunner:
    """One row of the card's field table, laid out as HKTrack's parser reads the real one.

    `horse_id` None prints the name without a link. `body_kg` None prints no body-weight cell
    (before the weigh-in, or a scratched runner); `body_change` follows it in brackets, an int
    printed with its sign (+4, -2) or a string printed as is ('±0'); None prints no bracket.
    `claim_mark` (☆ △ ◇ ▲ ★) goes before the rider's name, with `claim_title` ('1kg減量') as
    its title when given. `scratched` (取消 / 除外) is printed in place of the number."""

    number: int
    bracket: int
    horse_id: str | None
    name: str
    sex: str  # 牡 / 牝 / セ
    age: int
    carried_kg: float
    jockey_code: str  # 5 digits
    jockey: str
    trainer_code: str  # 5 digits
    trainer: str
    centre: str = "栗東"
    body_kg: int | None = None
    body_change: int | str | None = None
    claim_mark: str | None = None
    claim_title: str | None = None
    blinkers: bool = False
    scratched: str | None = None
    owner: str = "試験ホールディングス"
    sire: str = "ギジサイアー"
    dam: str = "ギジダム"
    dam_sire: str = "Synthetic Sire"


@dataclass(frozen=True)
class Card:
    """A race card page: the race header (date line, post time, race number, name, class,
    weight rule, course) and the field, laid out in the order of `runners`."""

    day: date
    meeting: str  # 5回東京7日
    race_no: int
    race_name: str
    category: str  # 3歳以上
    race_class: str  # 2勝クラス
    weight_rule: str  # 定量 / 別定 / ハンデ / 馬齢
    distance_m: int  # printed with a thousands comma: 1,800
    detail: str  # （ダート・右）, （芝・右 外）
    post: str  # 15時25分
    runners: tuple[CardRunner, ...] = ()


def _card_header(spec: Card) -> str:
    date_cell = f'<div class="cell date">{date_line(spec.day, spec.meeting)}</div>'
    time_cell = f'<div class="cell time">発走時刻：<strong>{spec.post}</strong></div>'
    win5 = action_link(OTHER_PATH, "pw17hde10x/A3", '<img src="w5.png" alt="WIN5 2レース目" />')
    number = (
        '<div class="num"><div class="num_wrap"><div class="race_number">'
        f'<img src="n.png" alt="{spec.race_no}レース"></div></div>'
        f'<div class="win5">{win5}</div></div>'
    )
    kind = (
        '<div class="type">'
        f'<div class="cell category">{spec.category}</div>'
        f'<div class="cell class">{spec.race_class}</div>'
        '<div class="cell rule">（混合）（特指）</div>'
        f'<div class="cell weight">{spec.weight_rule}</div>'
        '<div class="cell course"><span class="cap">距離：</span>'
        f'{spec.distance_m:,}<span class="unit">メートル</span>'
        f'<span class="detail">{spec.detail}</span></div>'
        "</div>"
    )
    title = (
        f'<div class="race_title"><div class="inner">{number}'
        '<div class="txt"><h2><span class="main">'
        f'<span class="race_name">{spec.race_name}</span></span></h2>{kind}</div>'
        "</div></div>"
    )
    prize = (
        '<ul class="prize"><li><div class="prize_unit"><div class="cell cap">本賞金</div>'
        '<div class="cell data"><ol><li><span class="cap">1着</span><span class="num">1,000</span>'
        "</li></ol></div></div></li></ul>"
    )
    return (
        '<div class="race_header"><div class="left">'
        f'<div class="date_line"><div class="inner">{date_cell}{time_cell}</div></div>'
        f"{title}{prize}</div></div>"
    )


def _body_weight(r: CardRunner) -> str:
    if r.body_kg is None:
        return ""
    change = r.body_change
    printed = f"{change:+d}" if isinstance(change, int) else change
    transition = f'<span class="transition">({printed})</span>' if change is not None else ""
    return f'<div class="cell weight">{r.body_kg}kg{transition}</div>'


def _card_row(r: CardRunner) -> str:
    alt = f"枠{r.bracket}{BRACKET_COLOURS[r.bracket - 1]}"
    waku = f'<td class="waku"><img src="/JRADB/img/waku/{r.bracket}.png" alt="{alt}" /></td>'
    if r.scratched:
        num = f'<td class="num"><span class="cap">{r.scratched}</span></td>'
    else:
        num = f'<td class="num">{r.number}{BLINKERS_ICON if r.blinkers else ""}</td>'
    if r.horse_id is not None:
        name = href_link("/JRADB/accessU.html", f"pw01dud00{r.horse_id}/A3", r.name)
    else:
        name = r.name
    price = r.scratched or f"{10.0 + r.number:.1f}"
    odds = (
        '<div class="odds"><div class="odds_line">'
        f'<span class="num"><strong>{price}</strong></span></div></div>'
    )
    trainer = action_link("/JRADB/accessC.html", f"pw05cmk0{r.trainer_code}/9F", r.trainer)
    horse = (
        '<td class="horse">'
        f'<div class="name_line"><div class="name">{name}</div>{odds}</div>'
        f'<div class="result_line">{_body_weight(r)}</div>'
        f'<p class="owner">{r.owner}</p><p class="breeder">試験牧場</p>'
        f'<p class="trainer">{trainer}<span class="division">({r.centre})</span></p>'
        '<ul class="family_line">'
        f'<li class="sire"><span>父：</span>{r.sire}</li>'
        f'<li class="mare"><span>母：</span>{r.dam}'
        f'<span class="bloodmare">(母の父：{r.dam_sire})</span></li>'
        "</ul></td>"
    )
    mark = ""
    if r.claim_mark is not None:
        title = f' title="{r.claim_title}"' if r.claim_title is not None else ""
        mark = f'<span{title} class="mark jockey">{r.claim_mark}</span>'
    rider = action_link("/JRADB/accessK.html", f"pw04kmk0{r.jockey_code}/90", r.jockey)
    jockey = (
        '<td class="jockey"><p class="cloth"><img src="c.png" alt="勝負服" /></p>'
        f'<p class="age">{r.sex}{r.age}/鹿</p>'
        f'<p class="weight">\n{r.carried_kg:.1f}<span>kg</span>\n</p>'
        f'<p class="jockey">{mark}{rider}</p></td>'
    )
    return f"<tr>{waku}{num}{horse}{jockey}{_past_cell()}</tr>"


def _past_cell() -> str:
    """A past-run cell: another race's date line, link, rider (with an apprentice mark), weight
    carried and body weight. None of it is this race's."""
    race = href_link("/JRADB/accessS.html", "pw01sde1004202601061020260809/2D", "過去特別")
    return (
        '<td class="past p1" data-place="" data-same-race="">'
        '<div class="date_line"><div class="date">2026年8月9日</div>'
        '<div class="rc">新潟</div></div>'
        f'<div class="race_line"><div class="name">{race}</div></div>'
        '<div class="info_line1"><div class="jockey"><span title="4kg減量" class="mark jockey">'
        '★</span>過去 騎手</div><div class="weight">51.0<span>kg</span></div></div>'
        '<div class="info_line2"><p class="h_weight">400<span>kg</span></p></div>'
        "</td>"
    )


def _past_winners() -> str:
    """The 過去5年の成績 table: an earlier running of the race, with horse and jockey links in
    td.horse / td.jockey cells. Not this race's field."""
    horse = href_link("/JRADB/accessU.html", "pw01dud002019109999/2C", "カコノショウシャ")
    rider = action_link("/JRADB/accessK.html", "pw04kmk009999/95", "過去 騎手")
    return (
        '<div class="backnumber_block mt40"><table class="basic narrow-xy">'
        '<caption class="h4"><div class="inner"><div class="main">過去5年の成績</div></div>'
        "</caption><thead><tr>"
        '<th class="date" scope="col">年月日</th><th class="horse" scope="col">馬名</th>'
        '<th class="jockey" scope="col">騎手</th></tr></thead><tbody><tr>'
        f'<td class="date">2023年</td><td class="horse">{horse}</td><td class="jockey">{rider}</td>'
        "</tr></tbody></table></div>"
    )


def card(spec: Card) -> bytes:
    """出馬表 page of one race: the race header in the field table's caption (as on the real
    page), a header row, one row per runner of `spec.runners` (each with a past-run cell), then
    the past-winners table, all under #syutsuba."""
    head = (
        "<thead><tr>"
        '<th scope="col" class="waku">枠</th><th scope="col" class="num">馬番</th>'
        '<th scope="col" class="horse">馬名</th><th scope="col" class="jockey">騎手</th>'
        '<th scope="col" class="past p1">前走</th>'
        "</tr></thead>"
    )
    rows = "".join(_card_row(r) for r in spec.runners)
    table = (
        f'<table class="basic narrow-xy mt20"><caption>{_card_header(spec)}</caption>'
        f"{head}<tbody>{rows}</tbody></table>"
    )
    return page(f'<div id="syutsuba">{table}{_past_winners()}</div>')
