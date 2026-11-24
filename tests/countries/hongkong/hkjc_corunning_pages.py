"""Builders for minimal synthetic HKJC Comments on Running pages (ticket #62).

A page holds only the structure HKTrack's `sources/hkjc/parse_corunning.py` reads, laid out the way
the live page lays it out, inside `div.corunning`: a date picker listing other meetings, the race
tabs (`corunning?date=YYYYMMDD&raceno=N` links to the meeting's other races), the race header
(`div.f_fs16` with "Race 1 (275)", then a line naming the race, rating, distance, venue and
going), the page's own Results link (`localresults?racedate=YYYY/MM/DD&racecourse=VV&raceno=N`),
and the comment table (`table.table_bd`): Placing, Horse<br/>No, Horse Name (the profile link,
then the brand in brackets, which is not the id), Jockey, Gear ("--" for none) and Comment, one
row per runner. A comment not published yet reads "No Comments on Running information for this
horse."

Every name, id, code and comment is invented; no real page is copied here.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

HOST = "https://racing.hkjc.com"
INFO = "/en-us/local/information"
VENUE_NAMES = {"HV": "Happy Valley", "ST": "Sha Tin"}
NOT_PUBLISHED = "No Comments on Running information for this horse."

COLUMNS: tuple[str, ...] = ("Placing", "Horse No", "Horse Name", "Jockey", "Gear", "Comment")


@dataclass(frozen=True)
class Runner:
    placing: str
    horse_no: int | None  # None prints an empty cell (a malformed row)
    name: str
    horse_id: str | None  # the horseid of the profile link (None: no link)
    brand: str  # printed in brackets after the name, e.g. "CZ277"
    jockey: str
    gear: str | None  # as printed; None prints the "--" placeholder
    comment: str | None  # printed as given, whitespace and all; None prints NOT_PUBLISHED
    href: str | None = None  # replaces the profile link's address (untrusted-link tests)


def _cell(r: Runner, column: str) -> str:
    if column == "Placing":
        return f'<td style="text-align: center;">{html.escape(r.placing)}</td>'
    if column == "Horse No":
        no = "" if r.horse_no is None else str(r.horse_no)
        return f'<td style="text-align: center;">{no}</td>'
    if column == "Horse Name":
        href = r.href
        if href is None and r.horse_id is not None:
            href = f"{INFO}/horse?horseid={r.horse_id}"
        name = html.escape(r.name)
        link = f'<a href="{html.escape(href)}">{name}</a>' if href else name
        return (
            '<td style="width:110px; text-align: center;">\n '
            f"{link}\n ({html.escape(r.brand)})\n </td>"
        )
    if column == "Jockey":
        return f'<td style="text-align: center;">{html.escape(r.jockey)}</td>'
    if column == "Gear":
        gear = "--" if r.gear is None else html.escape(r.gear)
        return f'<td style="width:36px; text-align: center;">{gear}</td>'
    if column == "Comment":
        text = NOT_PUBLISHED if r.comment is None else r.comment
        return f'<td class="f_fs16 f_tal">\n {html.escape(text)}\n </td>'
    raise ValueError(column)


def _head(column: str) -> str:
    label = "Horse<br />No" if column == "Horse No" else html.escape(column)
    css = ' class="f_tal"' if column in ("Gear", "Comment") else ""
    return f"<td{css}>{label}</td>"


def results_href(race_date: date, venue: str, race_no: int) -> str:
    return (
        f"{INFO}/localresults?racedate={race_date:%Y/%m/%d}&amp;racecourse={venue}"
        f"&amp;raceno={race_no}"
    )


def corunning_page(
    race_date: date,
    venue: str,
    race_no: int,
    runners: Sequence[Runner],
    *,
    race_name: str = "SYNTHETIC LANTERN HANDICAP",
    columns: Sequence[str] = COLUMNS,
    results_link: bool = True,
    site_nav: str = "",
    extra_rows: str = "",
    other_dates: Sequence[date] = (date(2031, 9, 6), date(2026, 7, 15), date(2025, 9, 7)),
) -> bytes:
    """One race's Comments on Running page. `site_nav` is raw HTML put above the page's own
    content (a full, untrimmed page has site navigation there); `extra_rows` raw `<tr>`s appended
    to the comment table; `results_link` False leaves out the page's own Results link."""
    options = "".join(f'<option value="{d:%Y%m%d}">{d:%d/%m/%Y}</option>' for d in other_dates)
    tabs = "".join(
        f'<td class="cur">Race {n}</td>' if n == race_no
        else f'<td><a href="{INFO}/corunning?date={race_date:%Y%m%d}&amp;raceno={n}">'
        f"\n Race {n}\n </a></td>"
        for n in range(1, max(race_no, 9) + 1)
    )  # fmt: skip
    header = (
        '<div class="f_fs16">'
        f"<p>Race {race_no} ({200 + race_no * 7})</p>"
        f"<p>{html.escape(race_name)} &nbsp; Rating:60-40 &nbsp;  1200m  &nbsp; "
        f"{html.escape(VENUE_NAMES[venue])} &nbsp; Good </p>"
        "<p>Section 1</p></div>"
    )
    results = ""
    if results_link:
        results = (
            '<div class="corunning_tab performance"><p class="tit f_fs16">'
            f'<a href="{results_href(race_date, venue, race_no)}">\n Results\n </a></p></div>'
        )
    head = "".join(_head(c) for c in columns)
    rows = "".join("<tr >" + "".join(_cell(r, c) for c in columns) + "</tr>\n" for r in runners)
    table = (
        '<table class="table_bd" style="word-break: normal;">'
        f'<thead class="bg_blue color_w font_wb"><tr>{head}</tr></thead>'
        f'<tbody class="f_fs13">\n{rows}{extra_rows}</tbody></table>'
    )
    body = (
        f"{site_nav}"
        '<div class="corunning commContent">'
        '<div class="commTitlePic"><img src="/racing/content/Images/StaticFile/English/Title/'
        'title_report_comments.gif" /></div>'
        f'<div><div class="select f_fs12 f_tar"><span><select id="raceDate">{options}</select>'
        "</span></div>"
        f'<div class="nav"><table class="f_fs16 f_tac" style="width:100%;"><tr>{tabs}</tr></table>'
        f"</div>{header}{results}{table}</div></div>"
    )
    return page(body)


def page(body: str) -> bytes:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>Comments on Running</title></head><body>{body}</body></html>"
    ).encode()


def corunning_url(race_date: date, race_no: int, *, archive: bool) -> str:
    """The archive endpoint (date / raceno) or the live one (Date / RaceNo), both YYYYMMDD."""
    if archive:
        return f"{HOST}{INFO}/archive/corunning?date={race_date:%Y%m%d}&raceno={race_no}"
    return f"{HOST}{INFO}/corunning?Date={race_date:%Y%m%d}&RaceNo={race_no}"
