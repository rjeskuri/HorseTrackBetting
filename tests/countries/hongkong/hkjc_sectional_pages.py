"""Builders for minimal synthetic HKJC sectional-time pages (ticket #61).

A page holds only the structure HKTrack's `sources/hkjc/parse_sectional.py` reads: the meeting
line (`div.search span.f_fl`, "Meeting Date:16/09/2026, Happy Valley"), one `div#Race<n>` block
per race, and in it a `table.race_table` with three header rows (Finishing Order, Horse No.,
Horse, a colspan over the section columns, Time) and one row per runner. A section cell holds the
running position (`span.f_fl`), the margin (`i`) and, in a second `p`, the section time; 2026-era
cells add the 200 m sub-splits inside that `p` (`span.color_blue2 span`). Unused section columns
hold a blank image.

Around it sit the decoys a real page carries: race navigation links naming other races, a date
picker listing other meetings, and the leader's own summary table (cumulative and sectional times
with sub-splits) outside the race table. Every value is invented; no real page is copied here.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

HOST = "https://racing.hkjc.com"
_BLANK = '<td><img src="/racing/content/images/StaticFile/blank_img.gif" /></td>'
_ORDINALS = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th")


@dataclass(frozen=True)
class Section:
    """One section cell. `pos` and `time` both None: an unused section (blank image).

    `margin` None prints no `<i>` at all; `time` "" prints an empty time `<p>`; `subs` are the
    sub-split texts, printed inside the time `<p>` (none: the pre-2026 `<p>{time}</p>`)."""

    pos: str | None = None
    margin: str | None = None
    time: str | None = None
    subs: Sequence[str] = ()

    def render(self) -> str:
        if self.pos is None and self.time is None:
            return _BLANK
        margin = "" if self.margin is None else f"<i>{html.escape(self.margin)}</i>"
        head = f'<p class="f_clear"><span class="f_fl">{self.pos or ""}</span>{margin}</p>'
        time = html.escape(self.time or "")
        if self.subs:
            spans = "&nbsp;&nbsp;".join(
                f'<span style="width: 28px;">{html.escape(s)}</span>' for s in self.subs
            )
            body = f'<p class="sectional_200">{time}\n<span class="color_blue2">{spans}</span></p>'
        else:
            body = f"<p>{time}</p>"
        return f"<td>\n{head}\n{body}\n</td>"


BLANK = Section()


@dataclass(frozen=True)
class Runner:
    """One runner row. `horse_id` None prints the name without a profile link; `href` replaces
    the profile link's address (for untrusted-link tests)."""

    order: str
    horse_no: int
    name: str
    horse_id: str | None
    sections: Sequence[Section] = field(default_factory=tuple)
    finish: str = "---"
    href: str | None = None

    def render(self, shown: int) -> str:
        label = f"{html.escape(self.name)}&nbsp;({self.horse_no:03d})"
        href = self.href
        if href is None and self.horse_id is not None:
            href = f"/en-us/local/information/horse?horseid={self.horse_id}"
        horse = f'<a target="_blank" href="{html.escape(href)}">{label}</a>' if href else label
        cells = [s.render() for s in self.sections]
        cells += [_BLANK] * (shown - len(self.sections))
        return (
            f"<tr><td>{html.escape(self.order)} </td><td>{self.horse_no}</td>"
            f'<td style="white-space: nowrap;">{horse}</td>{"".join(cells)}'
            f"<td>{html.escape(self.finish)}</td></tr>"
        )


@dataclass(frozen=True)
class Race:
    race_no: int
    runners: Sequence[Runner]
    shown: int = 6  # section columns the table prints, used or not

    def render(self) -> str:
        return (
            f'<div id="Race{self.race_no}"><div class="Race f_clear">'
            f'<p class="bg_blue color_w font_wb">&nbsp;Race {self.race_no}</p>'
            f'<p class="f_fl"><span class="f_fl">Class 4 - Turf - "A" Course - Good</span></p>'
            f"{self._leader_summary()}</div>"
            f'<table class="table_bd f_tac race_table"><thead>'
            '<tr><td rowspan="3">Finishing Order</td><td rowspan="3">Horse No.</td>'
            f'<td rowspan="3">Horse</td><td colspan="{self.shown}">Running Position &amp; '
            'Margin Behind Leader</td><td rowspan="3">Time</td></tr>'
            f'<tr><td colspan="{self.shown}">Sectional Time</td></tr>'
            f"<tr>{''.join(f'<td>{o} Sec.</td>' for o in _ORDINALS[: self.shown])}</tr>"
            f"</thead><tbody>{''.join(r.render(self.shown) for r in self.runners)}</tbody>"
            "</table></div>"
        )

    def _leader_summary(self) -> str:
        """The race's own summary of the leader's times: numbers outside the race table."""
        used = [s for s in (self.runners[0].sections if self.runners else ()) if s.time]
        cumulative, total = [], 0.0
        for s in used:
            try:
                total += float(s.time or "")
            except ValueError:
                break
            cumulative.append(f"<td>({total:.2f})</td>")
        sectional = "".join(
            f'<td>{s.time}<div class="f_fs12 color_blue2">{"&nbsp;&nbsp;".join(s.subs)}</div></td>'
            for s in used
        )
        return (
            '<table class="f_tac f_fl f_fs13"><tr><td class="f_tal">Race Time:</td>'
            f'{"".join(cumulative)}</tr><tr><td class="f_tal">Sectional Time:</td>{sectional}'
            "</tr></table>"
        )


def sectional_page(
    meeting_date: date | None,
    venue_name: str | None,
    races: Sequence[Race],
    *,
    other_dates: Sequence[date] = (date(2026, 9, 13), date(2025, 12, 3)),
) -> bytes:
    """A sectional-times page: the meeting line, then one block per race (an `All=True` page
    stacks several). `meeting_date` None prints no meeting line."""
    day = meeting_date or (other_dates[0] if other_dates else date(2000, 1, 1))
    nav_races = sorted({r.race_no for r in races} | {1, 2, 3, 7})
    nav = "".join(
        f'<td><a href="/en-us/local/information/displaysectionaltime?racedate='
        f'{day:%d/%m/%Y}&amp;RaceNo={n}"><img src="/racing/racecard_rt_{n}.gif" /></a></td>'
        for n in nav_races
    )
    options = "".join(f'<option value="{d:%d/%m/%Y}">{d:%d/%m/%Y}</option>' for d in other_dates)
    meeting = ""
    if meeting_date is not None:
        meeting = (
            f'<div class="search"><p class="f_clear"><span class="f_fl">Meeting Date:'
            f"{meeting_date:%d/%m/%Y}, {html.escape(venue_name or '')} </span>"
            f'<span class="f_fr"><select id="selectId">{options}</select></span></p></div>'
        )
    body = (
        f'<div class="dispalySectionalTime commContent"><table class="js_racecard">'
        f"<tr>{nav}</tr></table>{meeting}{''.join(r.render() for r in races)}</div>"
        '<div id="footer"><p>Footer text</p></div>'
    )
    return page(body)


def page(body: str) -> bytes:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>Sectional Time</title></head><body>{body}</body></html>"
    ).encode()


def sectional_url(race_date: date, race_no: int) -> str:
    return (
        f"{HOST}/en-us/local/information/displaysectionaltime"
        f"?racedate={race_date:%d/%m/%Y}&RaceNo={race_no}"
    )
