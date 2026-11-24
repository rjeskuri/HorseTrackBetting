"""Builders for minimal synthetic HKJC monthly fixture (calendar) pages (ticket #60).

A page holds only the structure HKTrack's `sources/hkjc/parse_fixture.py` reads: the month title
cell (`div.fixture_tab thead td[colspan=7]`, printed "9/2026"), one `td.calendar` cell per day with
the day number in `span.f_fl` and the venue, session and surface icons in `span.f_fr`, then one
`<p>` per race line. Around it sit the decoys a real page carries: blank padding cells before the
1st, weekday headings, month navigation links and an icon legend outside the calendar. Every value
is invented; no real page is copied here.
"""

from __future__ import annotations

import calendar as _calendar
import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

HOST = "https://racing.hkjc.com"


@dataclass(frozen=True)
class Icon:
    """One `<img>` in a cell header: its alt text (None: no alt attribute) and its file stem."""

    alt: str | None
    stem: str

    def render(self) -> str:
        alt = "" if self.alt is None else f' alt="{html.escape(self.alt)}"'
        src = f"/racing/content/Images/StaticFile/English/fixture/{self.stem}.gif"
        return f'<img src="{src}"{alt} />'


def icon(alt: str) -> Icon:
    """An icon whose alt text names it and whose file stem is that text in lower case."""
    return Icon(alt, alt.lower())


@dataclass(frozen=True)
class Day:
    """A calendar cell: the day number, its icons, and how many race lines it prints."""

    day: int
    icons: Sequence[Icon] = field(default_factory=tuple)
    races: int = 0


def meeting(day: int, *alts: str, races: int = 8) -> Day:
    """A meeting cell with icons named by alt text, e.g. meeting(23, "HV", "N", "Turf")."""
    return Day(day, tuple(icon(a) for a in alts), races)


def _cell(spec: Day) -> str:
    icons = "".join(i.render() for i in spec.icons)
    lines = "".join(
        f"<p><span>{1000 + 200 * (n % 4)}({n + 1})</span><span>{80 - 5 * n}-{60 - 5 * n}</span></p>"
        for n in range(spec.races)
    )
    return (
        f'<td class="calendar"><p class="f_clear"><span class="f_fl f_fs14">{spec.day}</span>'
        f'<span class="f_fr">{icons}</span></p>{lines}</td>'
    )


def calendar(
    year: int,
    month: int,
    meetings: Iterable[Day],
    *,
    title: str | None = "",
    legend: bool = True,
) -> bytes:
    """A month's fixture page. Days not in `meetings` are plain cells with only the day number.

    `title` is the month cell's text: "" prints the real month ("9/2026"), None prints no title.
    With `legend`, an icon legend outside the calendar shows every icon (Conghua included)."""
    by_day = {m.day: m for m in meetings}
    first_weekday = (_calendar.weekday(year, month, 1) + 1) % 7  # Sunday-first columns
    cells = ['<td class="calendar"></td>'] * first_weekday
    days_in_month = _calendar.monthrange(year, month)[1]
    for day in range(1, days_in_month + 1):
        cells.append(_cell(by_day.get(day, Day(day))))
    # cells for days the month does not have (a spec asking for them is a test of the parser)
    cells += [_cell(m) for d, m in sorted(by_day.items()) if d < 1 or d > days_in_month]
    while len(cells) % 7:
        cells.append('<td class="calendar"></td>')
    rows = "".join(f"<tr>{''.join(cells[i : i + 7])}</tr>" for i in range(0, len(cells), 7))

    heading = ""
    if title is not None:
        text = title or f"{month}/{year}"
        heading = f'<tr><td colspan="7" class="f_tac">{html.escape(text)}</td></tr>'
    weekdays = "".join(f"<td>{d}</td>" for d in ("SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"))
    table = (
        f'<div class="fixture_tab"><table><thead>{heading}<tr>{weekdays}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )
    prev_y, prev_m = (year, month - 1) if month > 1 else (year - 1, 12)
    next_y, next_m = (year, month + 1) if month < 12 else (year + 1, 1)
    nav = (
        '<div class="fixture_nav">'
        f'<a href="fixture?calyear={prev_y}&amp;calmonth={prev_m:02d}">{prev_m}/{prev_y}</a>'
        f'<a href="fixture?calyear={next_y}&amp;calmonth={next_m:02d}">{next_m}/{next_y}</a>'
        "</div>"
    )
    key = ""
    if legend:
        items = "".join(
            f"<li>{icon(a).render()} {html.escape(a)}</li>"
            for a in ("ST", "HV", "CH", "S1", "D", "N", "T", "Turf", "AWT", "Mixed")
        )
        key = f'<div class="fixture_legend"><ul>{items}</ul></div>'
    return page(nav + table + key)


def page(body: str) -> bytes:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>Fixture</title></head><body>{body}</body></html>"
    ).encode()


def fixture_url(year: int, month: int) -> str:
    return f"{HOST}/en-us/local/information/fixture?calyear={year:04d}&calmonth={month:02d}"
