"""Builders for minimal synthetic HKJC horse profile pages (ticket #63).

A page holds only the structure HKTrack's `sources/hkjc/parse_horse.py` reads: the profile block
(`table.horseProfile`) with the title (`span.title_text`, "NAME (BRAND)", "(Retired)" appended on
retired pages), a column of links to the horse's other pages, and two `table.table_eng_text`
tables of `label | : | value` rows; then the form table (`table.bigborder`), one header row of
`td.hsubheader` cells, season separator rows ("26/27 Season", one colspan cell) and one row per
run whose Video Replay cell nests its own table.

The two layouts differ the way the real ones do. Active (`/horse`) pages print "Country of Origin
/ Age", the current trainer, Current and Start of Season Rating, and DD/MM/YY run dates. Retired
(`/otherhorse`) pages print "Country of Origin" alone, Last Rating and no trainer, DD/MM/YYYY run
dates, `htable_text` cells and cell text padded with whitespace.

The profile block also prints the as-of-scrape values (ratings, stakes, record, trainer, owner,
age, location, import date) the parser must ignore. Every name, brand and number is invented; no
real page is copied here.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

HOST = "https://racing.hkjc.com"
INFO = "/en-us/local/information"

HEADER = (
    "Race<br/>Index",
    "Pla.",
    "Date",
    '<span title="Racecourse">RC</span>/Track/<br/>Course',
    "Dist.",
    "G",
    "Race<br/>Class",
    "Dr.",
    "Rtg.",
    "Trainer",
    "Jockey",
    "LBW",
    "Win Odds",
    "Act.<br/>Wt.",
    "Running<br/>Position",
    "Finish Time",
    "Declar.<br/>Horse Wt.",
    "Gear",
    "Video<br/>Replay",
)


def _e(text: str) -> str:
    return html.escape(text, quote=True)


@dataclass(frozen=True)
class Profile:
    """The profile block. `retired` switches the layout; the title is printed as given."""

    horse_id: str
    title: str
    retired: bool = False
    # static attributes
    origin: str = "AUS"
    colour_sex: str = "Brown / Gelding"
    import_type: str = "PPG"
    sire: str = "Harbour Light"
    dam: str = "Paper Lantern"
    dam_sire: str = "Copper Bell"
    # as-of-scrape values (never history)
    age: str = "6"
    season_stakes: str = "$615,000"
    total_stakes: str = "$3,147,250"
    record: str = "3-2-4-27"
    recent_starts: str = "1"
    location: str = "Hong Kong"
    arrival: str = "(11/08/2024)"
    import_date: str = "17/08/2023"
    trainer: str = "T Example"
    trainer_code: str = "XTE"
    owner: str = "Sample Owner Syndicate"
    current_rating: str = "47"
    season_start_rating: str = "41"
    last_rating: str = "58"
    same_sire: Sequence[tuple[str, str]] = (("HK_2024_X555", "OTHER SAMPLE"),)

    def render(self) -> str:
        route = "otherhorse" if self.retired else "horse"
        hid = _e(self.horse_id)
        pages = (
            [(route, "Form Records")]
            if self.retired
            else [
                (route, "Form Records"),
                ("ratingresultweight", "Rating/Wt/Placing"),
                ("performance", "Performance by Distance"),
                ("trackworkresult", "Trackwork Records"),
                ("movementrecords", "Movement Records"),
            ]
        )
        links = "".join(
            f'<li><a class="table_eng_text" href="{INFO}/{p}?horseid={hid}">{label}</a></li>'
            for p, label in pages
        )
        links += f'<li><a class="table_eng_text" href="{INFO}/selecthorse">Other Horses</a></li>'
        left = (
            '<td style="width: 280px;" valign="top"><table style="width: 280px; border:0;"><tbody>'
            f'<tr><td class="subsubheader" colspan="2"><span class="title_text">{_e(self.title)}'
            "</span></td></tr>"
            '<tr><td valign="top"><table id="Table3"><tr><td><img alt="" src="/img/s.jpg"/>'
            "</td></tr></table></td>"
            f'<td class="table_eng_text"><ul>{links}</ul></td></tr></tbody></table></td>'
        )
        if self.retired:
            middle_rows = [
                ("Country of Origin", _e(self.origin)),
                ("Colour / Sex", _e(self.colour_sex)),
                ("Import Type", _e(self.import_type)),
                ("Total Stakes*", _e(self.total_stakes)),
                ("No. of 1-2-3-Starts*", _e(self.record)),
            ]
            right_rows = [
                ("Owner", self._owner()),
                ("Last Rating", _e(self.last_rating)),
                ("Sire", _e(self.sire)),
                ("Dam", _e(self.dam)),
                ("Dam's Sire", _e(self.dam_sire)),
                ("Same Sire", "\n Nil      "),
            ]
        else:
            middle_rows = [
                ("Country of Origin / Age", f"{_e(self.origin)} / {_e(self.age)}"),
                ("Colour / Sex", _e(self.colour_sex)),
                ("Import Type", _e(self.import_type)),
                ("Season Stakes*", _e(self.season_stakes)),
                ("Total Stakes*", _e(self.total_stakes)),
                ("No. of 1-2-3-Starts*", _e(self.record)),
                ("No. of starts in past 10<br/>race meetings", _e(self.recent_starts)),
                (
                    "Current Location<br/>(Arrival Date)",
                    f"{_e(self.location)}<br/>{_e(self.arrival)}",
                ),
                ("Import Date", _e(self.import_date)),
            ]
            options = "".join(
                f'<option value="{_e(v)}">{_e(n)}</option>' for v, n in self.same_sire
            )
            right_rows = [
                (
                    "Trainer",
                    f'<a href="{INFO}/trainerwinstat?trainerid={_e(self.trainer_code)}">'
                    f"{_e(self.trainer)}</a>",
                ),
                ("Owner", self._owner()),
                ("Current Rating", _e(self.current_rating)),
                ("Start of<br/>Season Rating", _e(self.season_start_rating)),
                (
                    "Sire",
                    f'\n<a href="{INFO}/samesire?horsesire={_e(self.sire)}">'
                    f"\n {_e(self.sire)}\n </a>\n",
                ),
                ("Dam", _e(self.dam)),
                ("Dam's Sire", _e(self.dam_sire)),
                ("Same Sire", f'<select id="SameSire">{options}</select>'),
            ]
        return (
            f'<table class="horseProfile"><tbody><tr>{left}'
            f'<td style="width: 260px;" valign="top">{_fields(middle_rows)}</td>'
            f'<td style="width: 280px;" valign="top">{_fields(right_rows)}</td>'
            '</tr></tbody><tfoot class="table_eng_text"><tr><td colspan="3">*Includes local and '
            "overseas records and earnings</td></tr></tfoot></table>"
        )

    def _owner(self) -> str:
        return f'<a href="{INFO}/ownersearch?horseowner={_e(self.owner)}">{_e(self.owner)}</a>'


def _fields(rows: Sequence[tuple[str, str]]) -> str:
    body = "".join(f"<tr><td>{label}</td><td>:</td><td>{value}</td></tr>" for label, value in rows)
    return (
        '<table class="table_eng_text" style="border:0;">'
        f'<tbody style="vertical-align: top; white-space: nowrap;">{body}</tbody></table>'
    )


def results_href(racedate: str, venue: str, race_no: int) -> str:
    """Race Index link, e.g. results_href("2026/09/16", "HV", 1)."""
    return f"{INFO}/localresults?racedate={racedate}&Racecourse={venue}&RaceNo={race_no}"


@dataclass(frozen=True)
class Season:
    """A season separator row, e.g. Season("26/27")."""

    label: str


@dataclass(frozen=True)
class Run:
    """One form-table row, every cell as printed. `href` None prints the Race Index as plain
    text; otherwise it is the Race Index link's address."""

    date: str
    course: str
    index: str = "101"
    href: str | None = None
    placing: str = "05"
    distance: str = "1200"
    going: str = "G"
    race_class: str = "4"
    draw: str = "6"
    rating: str = "52"
    trainer: str = "T Example"
    jockey: str = "J Rider"
    lbw: str = "2-1/4"
    win_odds: str = "12"
    act_wt: str = "124"
    running: str = "6 5 5"
    finish: str = "1.10.12"
    horse_wt: str = "1102"
    gear: str = "B"

    def cells(self, horse_id: str, retired: bool) -> list[str]:
        """Inner HTML of the 19 cells, in HEADER order."""
        pad = "\n      " if retired else ""
        index = (
            f'\n<a class="htable_eng_text" href="{_e(self.href)}">{_e(self.index)}</a>\n'
            if self.href is not None
            else _e(self.index)
        )
        trainer = (
            f"\n{_e(self.trainer)}{pad}"
            if retired or self.trainer.strip("-") == ""
            else f'\n<a href="{INFO}/trainerwinstat?trainerid=XTE">{_e(self.trainer)}</a>\n'
        )
        jockey = (
            _e(self.jockey)
            if self.jockey.strip("-") == ""
            else f'\n<a href="{INFO}/jockeyprofile?jockeyid=XJR">{_e(self.jockey)}</a>\n'
        )
        route = "otherhorse" if retired else "horse"
        video = (
            '\n<table border="0" cellpadding="0" cellspacing="0"><tr><td>'
            '<a class="htable_text" href="javascript:OpenPopWin(\'/video.html?type=replay'
            f"&amp;rf={HOST}{INFO}/{route}?horseid={_e(horse_id)}',640,390)\">"
            '<img alt="Race Replay" src="/replay.png"/></a></td></tr></table>\n'
        )
        return [
            index,
            f'\n<span class="htable_eng_text2">{_e(self.placing)}</span>\n',
            _e(self.date),
            f"\n{_e(self.course)}                     ",
            _e(self.distance),
            _e(self.going),
            _e(self.race_class),
            f"\n{_e(self.draw)}{pad}",
            _e(self.rating),
            trainer,
            jockey,
            f'<span class="htable_eng_text2">{_e(self.lbw)}</span>',
            _e(self.win_odds),
            _e(self.act_wt),
            f'<span class="htable_rpnarrow_text">{_e(self.running)}</span>',
            _e(self.finish),
            _e(self.horse_wt),
            _e(self.gear),
            video,
        ]


def form(rows: Sequence[Run | Season], horse_id: str, *, retired: bool = False) -> str:
    """The form table in the HEADER layout."""
    css = "htable_text" if retired else "htable_eng_text"
    head = "".join(f'<td class="hsubheader">{h}</td>' for h in HEADER)
    out = [f'<table class="bigborder" border="0" width="100%"><tbody><tr>{head}</tr>']
    for row in rows:
        if isinstance(row, Season):
            out.append(
                '<tr height="18"><td align="left" class="htable_bold_text" '
                f'colspan="{len(HEADER)}"> <span class="htable_eng_bold_text">{_e(row.label)}'
                "</span> Season</td></tr>"
            )
            continue
        cells = row.cells(horse_id, retired)
        out.append(
            '<tr bgcolor="#F8F4EF">'
            + "".join(f'<td align="center" class="{css}">{c}</td>' for c in cells)
            + "</tr>"
        )
    return "".join(out) + "</tbody></table>"


def raw_form(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A form table from raw cell HTML: for layouts the Run builder does not print (another
    column order, merged cells, odd colspans)."""
    head = "".join(f'<td class="hsubheader">{h}</td>' for h in header)
    body = "".join(
        "<tr>" + "".join(f'<td class="htable_eng_text">{c}</td>' for c in r) + "</tr>" for r in rows
    )
    return f'<table class="bigborder"><tr>{head}</tr>{body}</table>'


def horse_page(
    profile: Profile | None, rows: Sequence[Run | Season] | None = (), *, extra: str = ""
) -> bytes:
    """A whole profile page: a site header with one link to another horse (a decoy for id
    detection), the profile block, then the form table (`rows` None: no form table at all)."""
    hid = profile.horse_id if profile is not None else "HK_2000_X000"
    retired = profile.retired if profile is not None else False
    decoy = (
        f'<div class="nav"><a href="{INFO}/horse?horseid=HK_2025_X777">Recently viewed</a></div>'
    )
    body = decoy + (profile.render() if profile is not None else "")
    if rows is not None:
        body += form(rows, hid, retired=retired)
    return page(body + extra)


def page(body: str) -> bytes:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<title>Horse</title></head><body><div class="commContent">{body}</div>'
        '<div id="footer"><p>Footer text</p></div></body></html>'
    ).encode()


def horse_url(horse_id: str, *, retired: bool = False) -> str:
    """The profile URL as the fetcher records it: /horse with Option=1, or the /otherhorse
    redirect target."""
    if retired:
        return f"{HOST}{INFO}/otherhorse?horseid={horse_id}"
    return f"{HOST}{INFO}/horse?horseid={horse_id}&Option=1"
