"""Builders for minimal synthetic HKJC entries pages, View=All (ticket #62).

A page holds only the structure HKTrack's `sources/hkjc/parse_entries.py` reads, laid out the way
the live page lays it out: a sequence of sections, each one a header block (`div#ulrow` whose
direct children are `ul.ulDiv` lists of label/value `li` pairs, the first `li` the meeting label
"15/10/2031 Happy Valley"), an entries table (`table.section_table`) and, after a "Reserve :"
caption, a reserves table (`table.section_table`) with an extra leading "No." column, the
stand-by number. Sections are separated by an `hr`. The provisional race code (P10, P20, ...)
appears only in the "Other Section(s)/Division(s)" links at the foot of the page
(`ajaxNav('...entries?...&SectionNo=N&ProvRaceNo=P10')`, inside `ul.ulDiv1` lists).

Around them sit the decoys a real page carries: the date banner, the remarks legend and the
"Entries Overview" link. Every name, id, code and number is invented; no real page is copied here.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

HOST = "https://racing.hkjc.com"
INFO = "/en-us/local/information"
VENUE_NAMES = {"HV": "Happy Valley", "ST": "Sha Tin"}

# The entries table's columns in the live page's order; the reserves table puts "No." first.
COLUMNS: tuple[str, ...] = (
    "Horse (Brand No)", "Trainer", "Wt.", "WFA.", "Rtg.", "Rtg.+/-", "P", "Remarks",
)  # fmt: skip
RESERVE_COLUMNS: tuple[str, ...] = ("No.", *COLUMNS)
_TITLES = {
    "Wt.": "Handicap Weight",
    "WFA.": "Weight-For-Age Allowance",
    "Rtg.": "Rating",
    "Rtg.+/-": "Rating Different",
    "P": "Order of Trainer Preference",
}

WORDS = (
    "ALDER", "BIRCH", "CEDAR", "DOGWOOD", "ELM", "FIR", "GORSE", "HAZEL", "ILEX", "JUNIPER",
    "KAURI", "LARCH", "MAPLE", "NUTMEG", "OAK", "PINE", "QUINCE", "ROWAN", "SPRUCE", "TEAK",
    "UMBER", "VIOLET", "WILLOW", "XENIA", "YEW", "ZELKOVA",
)  # fmt: skip
SUFFIXES = ("DREAM", "COMET", "RIVER")
TRAINERS = (("P Poplar", "PPO"), ("Q Quince-Lee", "QQL"), ("R O'Rowan", "ROR"), ("S Spruce", "SSP"))
# as printed: '+' trump card, '*' priority to run, '#' late entry
REMARKS = ("+", "*", "+ *", "", "# *", "*", "#", "")


@dataclass(frozen=True)
class Entry:
    """One horse in a section's entries (standby None) or reserves table (standby: its "No.")."""

    horse_id: str | None  # the horseid of the Horse link (None: no link)
    name: str
    brand: str
    trainer: str
    trainer_code: str | None  # the trainerid of the Trainer link (None: no link)
    wt: int | None
    wfa: int | None
    rating: int | None
    rating_change: int | None  # printed as the plain number ("6", "-1", "0"), None as "-"
    preference: int | None
    remarks: str  # as printed; "" prints an empty cell
    standby: int | None = None
    href: str | None = None  # replaces the horse link's address (untrusted-link tests)
    trainer_href: str | None = None  # replaces the trainer link's address


def entry(n: int, **kw) -> Entry:
    """Entry number `n` (1..77) of a meeting: varied values, unique id, brand and name."""
    trainer, code = TRAINERS[n % len(TRAINERS)]
    values = dict(
        horse_id=f"HK_2029_Z{n:03d}",
        name=f"{WORDS[n % len(WORDS)]} {SUFFIXES[n // len(WORDS)]}",
        brand=f"Z{n:03d}",
        trainer=trainer,
        trainer_code=code,
        wt=135 - (n * 3) % 23,
        wfa=3 if n % 7 == 0 else None,
        rating=40 + (n * 7) % 45,
        rating_change=None if n % 9 == 4 else (n * 5) % 13 - 6,
        preference=1 + n % 3,
        remarks=REMARKS[n % len(REMARKS)],
    )
    values.update(kw)
    return Entry(**values)


@dataclass(frozen=True)
class Section:
    prov: str  # the provisional race code the foot links name, e.g. "P10"
    section_no: int
    race_name: str
    race_class: str  # "Class 3"
    distance_m: int
    band: str  # the Rating Range as printed, "80-60"
    track: str = "Turf"  # "Turf" or "All Weather Track"
    course: str = '"C" Course'  # as printed; the quoted letters are the course configuration
    entries: Sequence[Entry] = field(default_factory=tuple)
    reserves: Sequence[Entry] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        """The foot link's text: 'Class 3 1200m Section 1 NAME (Ratings 80-60)'."""
        return (
            f"{self.race_class} {self.distance_m}m Section {self.section_no} {self.race_name} "
            f"(Ratings {self.band})"
        )


def _meeting_label(race_date: date, venue: str) -> str:
    return f'<span class="f_fs13">{race_date:%d/%m/%Y}</span> {html.escape(VENUE_NAMES[venue])}'


def _pairs(*items: str) -> str:
    return '<ul class="ulDiv">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _header(s: Section, label: str) -> str:
    return (
        '<div class="f_clear" id="ulrow">'
        + _pairs(f"\n{label}\n", "Track :", html.escape(s.track))
        + _pairs("Meeting No :", "5&nbsp;", "Course :", html.escape(s.course))
        + _pairs(
            "Race Name :", f"{html.escape(s.race_name)}&nbsp;", "Distance :", f"{s.distance_m}m"
        )
        + _pairs(
            "Race Class :",
            f"{html.escape(s.race_class)}&nbsp;",
            "Sec./Div. No :",
            f'<span class="f_fs13">{s.section_no}&nbsp;</span>',
        )
        + _pairs(
            "Rating Range :",
            f'<span class="f_fs13">{html.escape(s.band)}</span>',
            "&nbsp;",
            "&nbsp;",
        )
        + "</div>"
    )


def _num(value: int | None) -> str:
    return "-" if value is None else str(value)


def _cell(e: Entry, column: str) -> str:
    if column == "No.":
        return _num(e.standby)
    if column == "Horse (Brand No)":
        href = e.href
        if href is None and e.horse_id is not None:
            href = f"{INFO}/horse?horseid={e.horse_id}"
        text = f" {html.escape(e.name)} ({html.escape(e.brand)})"
        return f'<a href="{html.escape(href)}">{text}</a>' if href else text
    if column == "Trainer":
        href = e.trainer_href
        if href is None and e.trainer_code is not None:
            href = f"{INFO}/trainerprofile?trainerid={e.trainer_code}"
        name = html.escape(e.trainer)
        return f'<a href="{html.escape(href)}">{name}</a>' if href else name
    if column == "Wt.":
        return _num(e.wt)
    if column == "WFA.":
        return _num(e.wfa)
    if column == "Rtg.":
        return _num(e.rating)
    if column == "Rtg.+/-":
        return _num(e.rating_change)
    if column == "P":
        return "" if e.preference is None else str(e.preference)
    if column == "Remarks":
        return html.escape(e.remarks)
    raise ValueError(column)


def _table(rows: Sequence[Entry], columns: Sequence[str], css: str) -> str:
    head = "".join(
        f'<td><span title="{_TITLES[c]}">{html.escape(c)}</span></td>' if c in _TITLES
        else f"<td>{html.escape(c)}</td>"
        for c in columns
    )  # fmt: skip
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell(e, c)}</td>" for c in columns) + "</tr>\n" for e in rows
    )
    return (
        f'<table style="width:760px;" class="{css}"><thead>'
        f'<tr class="bg_blue color_w">{head}</tr></thead><tbody>\n{body}</tbody></table>'
    )


def _nav(race_date: date, venue: str, sections: Sequence[Section]) -> str:
    links = "".join(
        '<div><ul class="ulDiv1"><li><span style="display:inline-block; width:20px;">'
        '<img src="/racing/Info/StaticFile/Images/circle.gif" alt="&#8226;"></span>'
        '<span style="display:inline-block; font-size:13px;">'
        f"<a href=\"javascript:ajaxNav('{INFO}/entries?racedate={race_date:%Y/%m/%d}"
        f"&Racecourse={venue}&View=Section&SectionNo={s.section_no}&ProvRaceNo={s.prov}')\">"
        f"{html.escape(s.label)}</a></span></li></ul></div>"
        for s in sections
    )
    return (
        '<div class="margin_top15" style="font-size:14px;">'
        f"<a href=\"javascript:ajaxNav('{INFO}/entries?racedate={race_date:%Y/%m/%d}"
        f"&Racecourse={venue}&View=Overview')\">Entries Overview</a></div>"
        '<div class="margin_top10" style="font-size:13px;">Other Section(s)/Division(s)</div>'
        f'<div class="margin_top5">{links}</div>'
    )


def entries_page(
    race_date: date,
    venue: str,
    sections: Sequence[Section],
    *,
    label: bool = True,
    nav: bool = True,
    nav_reversed: bool = False,
    columns: Sequence[str] = COLUMNS,
    reserve_columns: Sequence[str] = RESERVE_COLUMNS,
) -> bytes:
    """A View=All entries page. `label` False prints an empty meeting-label `li`; `nav` False
    leaves out the foot links (and so every provisional race code); `nav_reversed` prints them
    in the reverse of page order."""
    meeting = _meeting_label(race_date, venue) if label else ""
    blocks = []
    for s in sections:
        reserves = ""
        if s.reserves:
            reserves = (
                '<div class="margin_top15 font13">Reserve :</div><div class="margin_top5">'
                + _table(
                    s.reserves, reserve_columns, "table_bd f_tac f_fs13 section_table loc_reserved"
                )
                + "</div>"
            )
        blocks.append(
            "<div>"
            + _header(s, meeting)
            + '<div class="rowDiv10">'
            + _table(s.entries, columns, "f_tac table_bd margin_top10 section_table loc")
            + "</div>"
            + reserves
            + "</div>"
        )
    legend = (
        '<div class="margin_top15"><span>Remarks: + = Trump Card; * = Priority to run; '
        "# = Late entry; R1 = the 1st standby horse, R3 = the 3rd standby horse</span><br /></div>"
    )
    foot = ""
    if nav:
        foot = _nav(race_date, venue, list(reversed(sections)) if nav_reversed else sections)
    banner = (
        '<div class="date_title"><div class="date_title f_tar"><a class="blueBtn1">'
        f"{race_date.day} {race_date:%b} - {html.escape(VENUE_NAMES[venue])}</a></div></div>"
    )
    sep = '<div class="margin_top15 sectionline"><hr></div>'
    body = (
        '<div class="commContent" id="entries"><div id="entriescontent">'
        f"{banner}{sep.join(blocks)}{legend}{foot}</div></div>"
    )
    return page(body)


def page(body: str) -> bytes:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>Entries</title></head><body>{body}</body></html>"
    ).encode()


def entries_url(race_date: date, venue: str) -> str:
    return f"{HOST}{INFO}/entries?racedate={race_date:%Y/%m/%d}&Racecourse={venue}&View=All"


def meeting_key(race_date: date, venue: str) -> str:
    return f"{race_date.isoformat()}_{venue}"
