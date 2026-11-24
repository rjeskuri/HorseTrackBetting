"""Builders for minimal synthetic HKJC race-card pages (ticket #64).

A card holds only the structure HKTrack's `sources/hkjc/parse_racecard.py` reads, laid out the way
the live card lays it out: the race header as one `div.f_fs13` block of `<br>`-separated lines
(title, date line, course line, prize line); the starter table `table.starter`, nested in the
`table#racecardlist` wrapper, with all 27 columns in the HTML and some hidden by display:none; and
the stand-by starters in their own 10-column `table#standbylist`, whose No. and Horse cells carry
the TdScratch class on every row. The apprentice claim is printed inside the jockey link
("B Birch (-2)"). Around them sit the decoys a real card carries: the meeting banner, the links to
the meeting's other races (racedate/Racecourse/RaceNo), the "setup my starter list" table whose
checkbox labels repeat the column names, and the gear legend table.

Every name, id, code and number is invented; no real page is copied here.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

HOST = "https://racing.hkjc.com"
INFO = "/en-us/local/information"
# config/hkjc.yaml venues: the date line prints the name, the store keeps the code
VENUE_NAMES = {"HV": "Happy Valley", "ST": "Sha Tin"}
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
)  # fmt: skip

# The starter table's 27 columns in the live card's order; the HIDDEN ones carry display: none.
COLUMNS: tuple[str, ...] = (
    "Horse No.", "Last 6 Runs", "Colour", "Horse", "Brand No.", "Wt.", "Jockey", "Over Wt.",
    "Draw", "Trainer", "Int'l Rtg.", "Rtg.", "Rtg.+/-", "Horse Wt. (Declaration)",
    "Wt.+/- (vs Declaration)", "Best Time", "Age", "WFA", "Sex", "Season Stakes", "Priority",
    "Days since Last Run", "Gear", "Owner", "Sire", "Dam", "Import Cat.",
)  # fmt: skip
HIDDEN = frozenset(
    {
        "Brand No.", "Over Wt.", "Int'l Rtg.", "Wt.+/- (vs Declaration)", "Best Time", "Age",
        "WFA", "Sex", "Season Stakes", "Days since Last Run", "Owner", "Sire", "Dam",
        "Import Cat.",
    }
)  # fmt: skip
# The stand-by table's 10 columns, in the live card's order.
RESERVE_COLUMNS: tuple[str, ...] = (
    "No.", "Horse", "Horse Wt. (Declaration)", "Wt.", "Rtg.", "Age", "Last 6 Runs", "Trainer",
    "Priority", "Gear",
)  # fmt: skip

WORDS = (
    "AMBER", "BRONZE", "COBALT", "DENIM", "EMBER", "FLINT", "GARNET", "HAZE", "INDIGO", "JADE",
    "KHAKI", "LILAC", "MAUVE", "NAVY",
)  # fmt: skip
JOCKEYS = (
    "A Alder", "B Birch", "C Cedar", "D Dogwood", "E Elm", "F Fir", "G Gorse", "H Hazel",
    "I Ilex", "J Juniper", "K Kauri", "L Larch", "M Maple", "N Nutmeg",
)  # fmt: skip
TRAINERS = ("P Poplar", "Q Quince", "R Rowan", "S Spruce")
# '+' trump card, '*' priority to run, the digit the trainer's order of preference
PRIORITIES = ("+ 1", "* 2", "+ * 3", None, "1", "* 1")


@dataclass(frozen=True)
class Starter:
    """One declared starter, as typed values; `card` prints them the way the live card does."""

    number: int
    draw: int | None
    horse_id: str | None  # the horseid of the Horse link, printed as given (None: no link)
    name: str
    brand: str
    wt: int | None  # the card "Wt.", before the apprentice claim
    jockey: str | None
    jockey_code: str | None
    trainer: str
    trainer_code: str | None
    claim: int | None = None  # printed "(-N)" after the jockey's name, inside the link
    overweight: int | None = None  # "Over Wt."
    intl_rating: int | None = None
    rating: int | None = None
    rating_change: int | None = None
    declared_wt: int | None = None  # the body weight, empty until about 17:00 on declaration day
    wt_change: int | None = None
    best_time_s: float | None = None
    age: int | None = None
    wfa: int | None = None
    sex: str | None = None
    season_stakes: int | None = None
    priority: str | None = None
    days_since: int | None = None
    gear: str | None = None
    owner: str | None = None
    sire: str | None = None
    dam: str | None = None
    import_cat: str | None = None
    last6: str | None = None
    row_class: str = ""  # an extra class on the <tr>
    cell_class: str = ""  # a class on every <td> of the row
    horse_note: str = ""  # text after the horse link
    jockey_note: str = ""  # text after the jockey link


def starter(n: int, draw: int | None, **kw) -> Starter:
    """Starter number `n` of an invented field; every default differs with `n`."""
    values = dict(
        horse_id=f"HK_2030_Z{100 + n}",
        name=f"SYNTH {WORDS[n - 1]}",
        brand=f"Z{100 + n}",
        wt=136 - 2 * n,
        jockey=JOCKEYS[n - 1],
        jockey_code=f"JK{n:02d}",
        trainer=TRAINERS[n % len(TRAINERS)],
        trainer_code=f"TR{n % len(TRAINERS):02d}",
        rating=84 - 3 * n,
        rating_change=(-1, 2, 0, None, 5, -3)[n % 6],
        best_time_s=round(69.0 + 1.37 * n, 2),
        age=3 + n % 5,
        sex=("g", "h", "g", "r", "c")[n % 5],
        season_stakes=123_450 * (n - 1),
        priority=PRIORITIES[n % len(PRIORITIES)],
        days_since=7 * n + 3,
        gear=("B", "TT", None, "CP-/B1", "V-/PC1", "H/TT")[n % 6],
        owner=f"Synthetic Syndicate {n}",
        sire=f"Fake Sire {n}",
        dam=f"Fake Dam {n}",
        import_cat=("PP", "PPG", "ISG", "VIS")[n % 4],
        last6="/".join(str((n * k) % 14 + 1) for k in range(1, 7)),
    )
    values.update(kw)
    return Starter(n, draw, **values)


@dataclass(frozen=True)
class Reserve:
    """One stand-by starter: the 10 columns of table#standbylist."""

    number: int  # the stand-by order ("No.")
    horse_id: str | None
    name: str
    wt: int | None
    rating: int | None
    age: int | None
    last6: str | None
    trainer: str
    trainer_code: str | None
    priority: str | None = None
    gear: str | None = None
    declared_wt: int | None = None


def reserve(n: int, **kw) -> Reserve:
    """Stand-by starter number `n`; ids and names never clash with `starter`'s."""
    values = dict(
        horse_id=f"HK_2030_Y{200 + n}",
        name=f"SPARE {WORDS[-n]}",
        wt=131 - 3 * n,
        rating=52 - 2 * n,
        age=4 + n % 3,
        last6="/".join(str((n * k) % 12 + 2) for k in range(1, 7)),
        trainer=TRAINERS[(n + 1) % len(TRAINERS)],
        trainer_code=f"TR{(n + 1) % len(TRAINERS):02d}",
        priority=("+ 1", "1", None)[n % 3],
        gear=("TT", "B/BL1/TT", None)[n % 3],
    )
    values.update(kw)
    return Reserve(n, **values)


@dataclass(frozen=True)
class Card:
    race_date: date
    venue: str  # code; the date line prints its name
    race_no: int
    race_name: str
    post: str  # "19:15", local time
    course: str  # the course line as printed, e.g. 'Turf, "C" Course, 1650M'
    prize: str  # the prize line as printed, e.g. 'Prize Money: $1,170,000, Rating: 60-40, Class 4'
    starters: tuple[Starter, ...]
    reserves: tuple[Reserve, ...] = ()
    meeting_races: int = 11
    # "" prints the card's own date line ("Wednesday, October 15, 2031, Happy Valley, 19:15");
    # None leaves the line out; any other text is printed as given
    date_line: str | None = ""


def header_lines(c: Card) -> list[str]:
    date_line = c.date_line
    if date_line == "":
        d = c.race_date
        date_line = (
            f"{WEEKDAYS[d.weekday()]}, {MONTHS[d.month - 1]} {d.day}, {d.year}, "
            f"{VENUE_NAMES[c.venue]}, {c.post}"
        )
    lines = [f"Race {c.race_no} - {c.race_name}"]
    if date_line is not None:
        lines.append(date_line)
    return [*lines, c.course, c.prize]


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _num(value: int | None, blank: str = "") -> str:
    return blank if value is None else str(value)


def _signed(value: int | None) -> str:
    """'+5', '-3', '0'; '-' when there is nothing to report."""
    if value is None:
        return "-"
    return f"+{value}" if value > 0 else str(value)


def _best_time(seconds: float | None) -> str:
    """70.37 -> '1.10.37' (minutes.seconds.hundredths)."""
    if seconds is None:
        return ""
    minutes, rest = divmod(round(seconds * 100), 6000)
    return f"{minutes}.{rest // 100:02d}.{rest % 100:02d}"


def _link(page: str, param: str, value: str | None, text: str) -> str:
    """A profile link as the card prints it; the bare text when there is no id."""
    if value is None:
        return _esc(text)
    href = f"{INFO}/{page}?{param}={value}"
    return (
        f'<a href="{_esc(href)}" onclick="window.open(this.href, \'_blank\'); return false;">'
        f"{_esc(text)}</a>"
    )


def _td(content: str, *, hidden: bool = False, css: str = "") -> str:
    style = "width: 40px; display: none;" if hidden else "width: 40px;"
    klass = f' class="{css}"' if css else ""
    return f'<td{klass} style="{style}">{content}</td>'


def _starter_cells(s: Starter) -> dict[str, str]:
    jockey = ""
    if s.jockey is not None:
        shown = s.jockey + (f" (-{s.claim})" if s.claim else "")
        jockey = _link("jockeyprofile", "jockeyid", s.jockey_code, shown)
    return {
        "Horse No.": str(s.number),
        "Last 6 Runs": _esc(s.last6 or ""),
        "Colour": f'<img src="/racing/content/Images/RaceColor/{_esc(s.brand)}.gif" alt="" />',
        "Horse": _link("horse", "horseid", s.horse_id, s.name) + _esc(s.horse_note),
        "Brand No.": _esc(s.brand),
        "Wt.": _num(s.wt),
        "Jockey": jockey + _esc(s.jockey_note),
        "Over Wt.": _num(s.overweight),
        "Draw": _num(s.draw),
        "Trainer": _link("trainerprofile", "trainerid", s.trainer_code, s.trainer),
        "Int'l Rtg.": _num(s.intl_rating, "-"),
        "Rtg.": _num(s.rating, "-"),
        "Rtg.+/-": _signed(s.rating_change),
        "Horse Wt. (Declaration)": _num(s.declared_wt),
        "Wt.+/- (vs Declaration)": _signed(s.wt_change),
        "Best Time": _best_time(s.best_time_s),
        "Age": _num(s.age),
        "WFA": _num(s.wfa, "-"),
        "Sex": _esc(s.sex or ""),
        "Season Stakes": "" if s.season_stakes is None else f"{s.season_stakes:,}",
        "Priority": _esc(s.priority or ""),
        "Days since Last Run": _num(s.days_since, "-"),
        "Gear": _esc(s.gear or ""),
        "Owner": _esc(s.owner or ""),
        "Sire": _esc(s.sire or ""),
        "Dam": _esc(s.dam or ""),
        "Import Cat.": _esc(s.import_cat or ""),
    }


def _reserve_cells(r: Reserve) -> dict[str, str]:
    return {
        "No.": str(r.number),
        "Horse": _link("horse", "horseid", r.horse_id, r.name),
        "Horse Wt. (Declaration)": _num(r.declared_wt),
        "Wt.": _num(r.wt),
        "Rtg.": _num(r.rating, "-"),
        "Age": _num(r.age),
        "Last 6 Runs": _esc(r.last6 or ""),
        "Trainer": _link("trainerprofile", "trainerid", r.trainer_code, r.trainer),
        "Priority": _esc(r.priority or ""),
        "Gear": _esc(r.gear or ""),
    }


def _starter_row(s: Starter, columns: Sequence[str]) -> str:
    cells = _starter_cells(s)
    tds = "".join(_td(cells[c], hidden=c in HIDDEN, css=s.cell_class) for c in columns)
    row_class = f"f_tac f_fs13 {s.row_class}".strip()
    return f'<tr class="{row_class}">{tds}</tr>'


def _reserve_row(r: Reserve, columns: Sequence[str]) -> str:
    cells = _reserve_cells(r)
    # the live stand-by table styles its No. and Horse cells with the scratch class on every row
    tds = "".join(_td(cells[c], css="TdScratch" if c in ("No.", "Horse") else "") for c in columns)
    return f'<tr class="bg_white f_tac f_fs13">{tds}</tr>'


_SETUP = (
    '<div class="starter_custom_title f_clear"><span class="font_wb f_fs14">SETUP MY STARTER LIST'
    ' -</span></div><div id="ColSelectBody" style="display: none;"><form>'
    '<table class="setup tableBorderBlue trBgBlue"><tbody>'
    '<tr class="f_fs11"><td class="bg_e6caae f_fs13 font_wb">HORSE</td>'
    '<td><input value="Horse No." type="checkbox" checked="checked">Horse No.</td>'
    '<td><input value="Horse" type="checkbox" checked="checked">Horse</td>'
    '<td><input value="Draw" type="checkbox" checked="checked">Draw</td>'
    '<td><input value="Wt." type="checkbox" checked="checked">Handicap Weight</td></tr>'
    '<tr class="f_fs11"><td class="bg_e6caae f_fs13 font_wb">JOCKEY</td>'
    '<td><input value="Jockey" type="checkbox" checked="checked">Jockey</td>'
    '<td><input value="Over Wt." type="checkbox">Probable Over Weight</td></tr>'
    '<tr class="f_fs11"><td class="bg_e6caae f_fs13 font_wb">TRAINER</td>'
    '<td><input value="Trainer" type="checkbox" checked="checked">Trainer</td></tr>'
    "</tbody></table></form></div>"
)

_GEAR_LEGEND = (
    '<div class="Gear"><table class="borderSpacing"><tr>'
    '<td class="gear2">B :</td><td class="gear1">Blinkers</td>'
    '<td class="gear2">H :</td><td class="gear1">Hood</td>'
    '<td class="gear2">TT :</td><td class="gear1">Tongue Tie</td>'
    "</tr></table></div>"
)


def _standby(reserves: Sequence[Reserve], columns: Sequence[str]) -> str:
    if not reserves:
        return ""
    head = "".join(f"<td><b>{_esc(c)}</b></td>" for c in columns)
    rows = "".join(_reserve_row(r, columns) for r in reserves)
    return (
        '<div style="padding: 5px 0;"><table border="0" class="tableBorderBlue" id="standbylist">'
        f'<thead><tr><th class="f_tal" colspan="{len(columns)}">Stand-by Starter</th></tr></thead>'
        f'<tbody><tr class="f_fs13 f_tac bg_e6caae">{head}</tr>{rows}</tbody></table></div>'
    )


def card(
    c: Card,
    *,
    columns: Sequence[str] = COLUMNS,
    reserve_columns: Sequence[str] = RESERVE_COLUMNS,
) -> bytes:
    """The race card page. `columns` / `reserve_columns` set which columns the two tables print
    and in what order (the live order by default)."""
    d, venue_name = c.race_date, VENUE_NAMES[c.venue]
    links = "".join(
        f'<td><a href="?racedate={d:%Y/%m/%d}&amp;Racecourse={c.venue}&amp;RaceNo={n}">'
        f'<img src="/racing/info/StaticFile/Images/Racing/racecard_rt_{n}.gif" alt="{n}" /></a>'
        "</td>"
        for n in range(1, c.meeting_races + 1)
    )
    nav = (
        f'<div class="date_title"><a class="blueBtn1">{d.day} {MONTHS[d.month - 1][:3]} - '
        f"{venue_name}</a></div>"
        '<div class="racingNum top_races js_racecard_rt_num"><table class="f_fs12 js_racecard">'
        f'<tr><td class="font_w7 f_tar"><span>{venue_name}: </span></td>{links}</tr></table></div>'
    )
    title, *rest = header_lines(c)
    lines = [f'<span class="font_wb">{_esc(title)}</span>', *(_esc(line) for line in rest)]
    header = (
        '<div class="margin_top10" style="width: 760px;">'
        f'<div class="f_fs13" style="line-height: 20px;">{"<br>".join(lines)}</div></div>'
    )
    head = "".join(_td(_esc(label), hidden=label in HIDDEN) for label in columns)
    rows = "".join(_starter_row(s, columns) for s in c.starters)
    starters = (
        '<div style="padding: 10px 0;"><table class="tableBorderBlue tdAlignC" id="racecardlist">'
        '<thead><tr><th class="f_fs14 font_wb f_tal">MY Race Card LIST</th></tr></thead>'
        "<tbody><tr><td>"
        '<table cellpadding="1" cellspacing="1" class="starter f_tac f_fs13 draggable hiddenable">'
        f'<thead><tr class="trBg01 tdAlignC boldFont13">{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table>"
        "</td></tr></tbody></table></div>"
    )
    return page(
        nav + header + _SETUP + starters + _standby(c.reserves, reserve_columns) + _GEAR_LEGEND
    )


def no_information() -> bytes:
    """The HTTP 200 body served for a race with no card (no such race, not yet declared, past)."""
    return page(
        '<div class="errorout"><div class="errorleft"><span>You may visit any <br />of the pages '
        'below.<br /><br />&gt;&gt;<a href="//www.example.invalid">Home Page</a></span></div>'
        '<div class="errorright"><div id="errorContainer" style="text-align: center;">'
        "No information.</div></div></div>"
    )


def not_available() -> bytes:
    """The parameterless race-card address before the declarations are out."""
    return page(
        '<div class="errorout"><div class="errorright"><div class="contentR1"><span>The '
        '<span style="color:green;font-weight:bold;">Race Card</span> is not available!</span>'
        '</div><div class="contentR1_2">The Race Card will be available around noon two days '
        "before the meeting.</div></div></div>"
    )


def page(body: str) -> bytes:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Race Card</title>'
        f'</head><body><div class="commContent raceCard">{body}</div></body></html>'
    ).encode()
