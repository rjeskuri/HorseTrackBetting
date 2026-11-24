"""Builders for minimal synthetic HKJC results pages (ticket #65): `localresults`, live and
archive routes.

A page holds only the structure HKTrack's `sources/hkjc/parse_results.py` reads, laid out as on
the site, inside `div.localResults.commContent`: the race tab strip (`div.top_races`, the race on
show is the one unlinked `racecard_rt_<n>_o.gif`), the meeting label (`div.raceMeeting_select`,
"Race Meeting: 14/10/2026 Happy Valley", next to a date picker listing other meetings), an
optional banner for a race without a result, the race header (`div.race_tab`, rows found by
their "Going :", "Course :" and "Time :" labels, with a spacer row and a "Sectional Time :" row
the parser must skip), the runner table (`div.performance`, columns read by header text), the
dividends (`div.dividend_tab`; a pool's label cell spans its rows) and the stewards' report
(a per-horse table with an Incident column, or one narrative paragraph). Every route also
ships an inline <style> naming those containers, as the site does.

Every horse, person, id, time, price and sentence is invented; no real page is copied here.
Pages come back as UTF-8 bytes. Build variants with `dataclasses.replace`.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, replace
from datetime import date, timedelta

HOST = "https://racing.hkjc.com"
VENUE_NAMES = {"HV": "Happy Valley", "ST": "Sha Tin"}

LIVE_COLUMNS = (
    "Pla.",
    "Horse No.",
    "Horse",
    "Jockey",
    "Trainer",
    "Act. Wt.",
    "Declar. Horse Wt.",
    "Dr.",
    "LBW",
    "Running Position",
    "Finish Time",
    "Win Odds",
)
# the 2004-era archive layout: no running positions
NO_RUNNING_COLUMNS = tuple(c for c in LIVE_COLUMNS if c != "Running Position")
# a race that was never run: no running positions and no win odds
NOT_RUN_COLUMNS = tuple(c for c in NO_RUNNING_COLUMNS if c != "Win Odds")

# Inline CSS as every route ships it, even on the "No information." page: its selectors name the
# very containers a classifier looks for, so it is not content.
_ROUTE_CSS = (
    ".commContent { min-height: 320px; }\n"
    ".localResults .top_races td { padding: 2px; }\n"
    ".localResults .race_tab tbody { margin-top: 4px; }\n"
    ".localResults .performance table { border: 1px solid #224488; }\n"
    ".localResults .dividend_tab table { margin-right: 16px; }\n"
    ".localResults .race_incident_report .info_p { padding: 12px; }\n"
)


@dataclass(frozen=True)
class Runner:
    """One runner row as printed ("" = an empty cell)."""

    placing: str
    horse_no: str
    horse: str
    horse_id: str | None  # the profile link's horseid; None prints the name without a link
    brand: str  # printed after the name in brackets; "" prints none
    jockey: str
    jockey_code: str | None  # None: printed without a link (no licence any more)
    trainer: str
    trainer_code: str | None
    act_wt: str
    decl_wt: str
    draw: str
    lbw: str
    running: str  # positions split by spaces, one inner <div> each; "" an empty cell
    finish_time: str
    win_odds: str
    photo: bool = False  # a finish-photo icon link before the placing
    short: int | None = None  # print only the first `short` cells (a malformed row)


@dataclass(frozen=True)
class Pool:
    """A pool's rows: (winning combination, dividend) each; a row of another length prints its
    cells as given (a malformed row)."""

    label: str
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Incident:
    """One row of the per-horse stewards' table."""

    placing: str
    horse_no: str
    horse: str
    horse_id: str | None
    brand: str
    text: str


@dataclass(frozen=True)
class Race:
    race_date: date
    venue: str  # HV | ST
    race_no: int
    race_index: int | None = None  # printed "RACE 3 (141)"; None prints "RACE 3"
    class_line: str = ""
    going: str = ""
    race_name: str = ""
    course: str = ""
    prize: str = ""
    times: tuple[str, ...] = ()  # the "Time :" row, each printed in brackets
    runners: tuple[Runner, ...] = ()
    columns: tuple[str, ...] = LIVE_COLUMNS
    pools: tuple[Pool, ...] = ()
    pool_classes: bool = True  # False: label cells without class or rowspan (no-result pages)
    incidents: tuple[Incident, ...] = ()  # per-horse table
    narrative: tuple[str, ...] = ()  # narrative report, one <br/>-separated line each
    banner: str | None = None
    n_races: int = 9
    tabs: bool = True
    meeting_label: bool = True
    header: bool = True
    runner_table: bool = True

    @property
    def key(self) -> str:
        """The cache key the fetcher stores the page under."""
        return f"{self.race_date.isoformat()}_{self.venue}_R{self.race_no:02d}"


def live_url(race_date: date, venue: str, race_no: int) -> str:
    d = race_date.strftime("%Y/%m/%d")
    return (
        f"{HOST}/en-us/local/information/localresults"
        f"?racedate={d}&Racecourse={venue}&RaceNo={race_no}"
    )


def archive_url(race_date: date, venue: str, race_no: int) -> str:
    d = race_date.strftime("%Y/%m/%d")
    return (
        f"{HOST}/en-us/local/information/archive/localresults"
        f"?racedate={d}&racecourse={venue}&RaceNo={race_no}"
    )


def _e(text: str) -> str:
    return html.escape(text)


def _tabs(r: Race) -> str:
    d = r.race_date.strftime("%Y/%m/%d")
    cells = [f'<td class="font_w7 f_tar"><span>{_e(VENUE_NAMES[r.venue])}:</span> </td>']
    for n in range(1, r.n_races + 1):
        if n == r.race_no:
            cells.append(
                f'<td><img src="/racing/info/StaticFile/Images/Racing/racecard_rt_{n}_o.gif"/></td>'
            )
            continue
        href = (
            f"/en-us/local/information/localresults"
            f"?racedate={d}&amp;Racecourse={r.venue}&amp;RaceNo={n}"
        )
        img = f'<img src="/racing/info/StaticFile/Images/Racing/racecard_rt_{n}.gif"/>'
        cells.append(f'<td><a href="{href}">{img}</a></td>')
    cells.append(
        f'<td><a class="local" href="/en-us/local/information/resultsall?racedate={d}&amp;'
        f'Racecourse={r.venue}">'
        '<img src="/racing/info/StaticFile/Images/Racing/racecard_rt_all.gif"/>'
        "</a></td>"
    )
    return (
        '<div class="f_clear top_races"><table class="f_fs12 js_racecard">'
        f"<tr>{''.join(cells)}</tr></table></div>"
    )


def _meeting(r: Race) -> str:
    # the date picker lists other meetings: only the label says which one is on show
    picks = [r.race_date + timedelta(days=4), r.race_date, r.race_date - timedelta(days=3)]
    options = "".join(
        f'<option value="{_e(json.dumps({"date": p.strftime("%d/%m/%Y"), "venue": ""}))}"'
        f"{' selected' if p == r.race_date else ''}>{p.strftime('%d/%m/%Y')} </option>"
        for p in picks
    )
    label = f"Race Meeting:  {r.race_date.strftime('%d/%m/%Y')}    {VENUE_NAMES[r.venue]}"
    return (
        '<div class="raceMeeting_select"><p class="f_clear">'
        f'<span class="f_fl f_fs13">{_e(label)}</span>'
        f'<span class="f_fr"><select class="f_fs11" id="selectId">{options}</select></span>'
        "</p></div>"
    )


def _header(r: Race) -> str:
    title = f"RACE {r.race_no}" + (f" ({r.race_index})" if r.race_index is not None else "")
    times = "".join(f'<td class="f_tac">({_e(t)})</td>' for t in r.times)
    # sectional times: other numbers, which the parser must not read as the race's times
    sectionals = "".join(
        f'<td class="f_tac">{11 + i}.{i}7<div class="f_fs12 color_blue2">5.{i}1  6.{i}6</div></td>'
        for i in range(len(r.times))
    )
    return (
        '<div class="race_tab"><table style="width: 760px;">'
        '<thead><tr class="bg_blue color_w font_wb">'
        f'<td colspan="16">{title}</td><td></td><td></td></tr></thead>'
        '<tbody class="f_fs13"><tr style="height:10px;"><td></td><td></td><td></td></tr>'
        f'<tr><td>{_e(r.class_line)}</td><td>Going :</td><td colspan="14">{_e(r.going)}</td></tr>'
        f'<tr><td>{_e(r.race_name)}</td><td>Course :</td><td colspan="14">{_e(r.course)}</td></tr>'
        f"<tr><td>{_e(r.prize)}</td><td>Time :</td>{times}</tr>"
        f"<tr><td></td><td>Sectional Time :</td>{sectionals}</tr>"
        "</tbody></table></div>"
    )


def _horse_cell(name: str, horse_id: str | None, brand: str) -> str:
    shown = _e(name)
    if horse_id is not None:
        href = f"/en-us/local/information/horse?horseid={_e(horse_id)}"
        shown = f'<a class="local" href="{href}">{shown}</a>'
    suffix = f" ({_e(brand)})" if brand else ""
    return f'<td class="f_fs13 f_tal" style="white-space: nowrap;">{shown}{suffix}</td>'


def _person_cell(name: str, code: str | None, page: str, param: str) -> str:
    if code is None:
        return f'<td class="f_fs13 f_tal">\n{_e(name)}                                </td>'
    href = f"/en-us/local/information/{page}?{param}={_e(code)}&amp;Season=Current"
    return f'<td class="f_fs13 f_tal">\n<a class="local" href="{href}">\n {_e(name)}\n </a>\n</td>'


def _running_cell(running: str) -> str:
    if not running or running.strip("-") == "":
        return f"<td>{_e(running)}</td>"
    inner = "".join(
        f'<div style="display:inline-block; width:16px;">\n{_e(p)}                </div>'
        for p in running.split()
    )
    return f'<td style="text-align: center;"><div style="font-size: 0;">{inner}</div></td>'


def _runner_cells(r: Runner, columns: tuple[str, ...]) -> list[str]:
    cells = []
    for column in columns:
        if column == "Pla.":
            photo = (
                '<a href="//racing.hkjc.com/en-us/local/info/finishphoto?path=p.jpg">'
                '<img src="/racing/content/images/StaticFile/icon-camera.png"/></a> '
                if r.photo
                else ""
            )
            cells.append(f'<td style="white-space: nowrap;">{photo}{_e(r.placing)}</td>')
        elif column == "Horse No.":
            cells.append(f"<td>\n{_e(r.horse_no)}                                </td>")
        elif column == "Horse":
            cells.append(_horse_cell(r.horse, r.horse_id, r.brand))
        elif column == "Jockey":
            cells.append(_person_cell(r.jockey, r.jockey_code, "jockeyprofile", "jockeyid"))
        elif column == "Trainer":
            cells.append(_person_cell(r.trainer, r.trainer_code, "trainerprofile", "trainerid"))
        elif column == "Running Position":
            cells.append(_running_cell(r.running))
        else:
            value = {
                "Act. Wt.": r.act_wt,
                "Declar. Horse Wt.": r.decl_wt,
                "Dr.": r.draw,
                "LBW": r.lbw,
                "Finish Time": r.finish_time,
                "Win Odds": r.win_odds,
            }[column]
            cells.append(f"<td>{_e(value)}</td>")
    return cells if r.short is None else cells[: r.short]


def _runner_table(r: Race) -> str:
    head = "".join(
        "<td>Running<br/>Position</td>" if c == "Running Position" else f"<td>{_e(c)}</td>"
        for c in r.columns
    )
    rows = "".join(f"<tr>{''.join(_runner_cells(x, r.columns))}</tr>" for x in r.runners)
    return (
        '<div class="performance"><table class="f_tac table_bd draggable" style="width: 760px;">'
        f'<thead><tr class="bg_blue color_w">{head}</tr></thead>'
        f'<tbody class="f_fs12">{rows}</tbody></table></div>'
    )


def _dividends(r: Race) -> str:
    body = []
    for i, pool in enumerate(r.pools):
        shade = ' class="bg_dc"' if i % 2 == 0 else ' class=""'
        for j, cells in enumerate(pool.rows):
            tds = "".join(
                f'<td class="f_fs14{" f_tar" if k == 1 else ""}">{_e(c)}</td>'
                for k, c in enumerate(cells)
            )
            if j == 0:
                attrs = f' class="fontXi" rowspan="{len(pool.rows)}"' if r.pool_classes else ""
                tds = f"<td{attrs}>{_e(pool.label)}</td>{tds}"
            body.append(f"<tr{shade}>{tds}</tr>")
    return (
        '<div class="dividend_tab f_clear"><div class="position_left">'
        '<table class="table_bd f_tac f_fs13 f_fl"><thead>'
        '<tr class="bg_blue color_w font_wb f_fs14">'
        '<td class="f_tal" colspan="3">Dividend</td></tr>'
        '<tr class="bg_e6caae">'
        "<td>Pool</td><td>Winning Combination</td><td>Dividend (HK$)</td></tr>"
        f"</thead><tbody>{''.join(body)}</tbody></table>"
        '<div class="dividend_remark f_fs13 f_fl">Note: "F" in a combination stands for any runner.'
        "</div></div></div>"
    )


def _incident_table(r: Race) -> str:
    rows = "".join(
        f'<tr><td style="white-space: nowrap;">{_e(i.placing)}</td>'
        f"<td>\n{_e(i.horse_no)}                                </td>"
        f"{_horse_cell(i.horse, i.horse_id, i.brand)}"
        f'<td class="f_fs13 f_tal">{_e(i.text)}</td></tr>'
        for i in r.incidents
    )
    return (
        '<div class="race_incident_report performance">'
        '<p class="bg_blue font_wb color_w f_fs14">Racing Incident Report</p>'
        '<table class="f_tac table_bd"><thead><tr class="bg_blue color_w">'
        "<td>Pla.</td><td>Horse No.</td><td>Horse</td><td>Incident</td></tr></thead>"
        f'<tbody class="f_fs12 fontFam">{rows}</tbody></table></div>'
    )


def _narrative(r: Race) -> str:
    lines = "<br/>\n".join(_e(line) for line in r.narrative)
    return (
        '<div class="race_incident_report">'
        '<p class="bg_blue font_wb color_w f_fs14">Racing Incident Report</p>'
        f'<p class="info_p f_fs12">{lines}</p></div>'
    )


def render(r: Race) -> bytes:
    parts = [
        '<div class="commTitlePic">'
        '<img src="/racing/info/StaticFile/Images/title_results.gif"/></div>'
    ]
    if r.tabs:
        parts.append(_tabs(r))
    if r.meeting_label:
        parts.append(_meeting(r))
    if r.banner is not None:
        parts.append(f'<div style="color: #c00;">{_e(r.banner)}</div>')
    if r.header:
        parts.append(_header(r))
    if r.runner_table:
        parts.append(_runner_table(r))
    if r.pools:
        parts.append(_dividends(r))
    if r.incidents:
        parts.append(_incident_table(r))
    elif r.narrative:
        parts.append(_narrative(r))
    parts.append('<div class="remark_botm_div f_fs12"><p>Remarks: an invented footnote.</p></div>')
    return page(f'<div class="localResults commContent">{"".join(parts)}</div>')


def page(main: str, *, css: bool = True, script: bool = False) -> bytes:
    """A whole document around `main`. `script` adds a script and an HTML comment that name the
    results containers (neither is content)."""
    style = f"<style>\n{_ROUTE_CSS}</style>" if css else ""
    tail = (
        '<script>var panel = "div.localResults.commContent";</script>'
        '<!-- <div class="localResults commContent"> -->'
        if script
        else ""
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Local Results</title>'
        f'{style}</head><body><div class="site_header"><a href="/">Racing</a></div>{main}{tail}'
        "</body></html>"
    ).encode()


def no_information(*, css: bool = True, script: bool = False) -> bytes:
    """The page the site serves with HTTP 200 when there is no such race or meeting."""
    return page(
        '<div class="commContent"><div class="errorout"><div class="errorleft"><span>'
        'Try one of the pages below.<br/><a href="/">Racing home</a></span></div>'
        '<div class="errorright"><div id="errorContainer" style="text-align: center;">'
        "No information.</div></div></div></div>",
        css=css,
        script=script,
    )


def busy_page() -> bytes:
    """A page from the site that is neither a result nor the "No information." page."""
    return page(
        '<div class="commContent"><p>The service is busy. Please try again later.</p></div>'
    )


# R(placing, no, horse, horse_id, brand, jockey, jockey_code, trainer, trainer_code,
#   act_wt, decl_wt, draw, lbw, running, finish_time, win_odds, photo=..., short=...)
R = Runner
REFUND = ("-", "REFUND")


# ---------------------------------------------------------------- a live-route race (2026/27)

# fmt: off
LIVE = Race(
    race_date=date(2026, 10, 14), venue="HV", race_no=3, race_index=141,
    class_line="Class 4 - 1650M - (60-40)", going="GOOD TO FIRM",
    race_name="LANTERN BAY HANDICAP", course='TURF - "C+3" Course', prize="HK$ 1,250,000",
    times=("25.31", "48.90", "1:12.66", "1:38.42"),
    runners=(
        R("1", "6", "AMBER LANTERN", "HK_2023_X101", "X101", "A Quillfeather (-2)", "QFA",
          "M Q Larkspur", "LMQ", "126", "1105", "4", "---", "3 2 1 1", "1:38.42", "3.7"),
        R("2", "2", "COPPER TIDE", "HK_2022_W215", "W215", "T Brambleton", "BMT",
          "K Y Saltmarsh", "SKY", "133", "1180", "7", "N", "6 6 4 2", "1:38.46", "12"),
        R("3", "9", "SILENT ORCHARD", "HK_2024_Y033", "Y033", "R Oakhollow", "OHR",
          "P Ashgrove", "APG", "120", "1042", "1", "3/4", "1 1 2 3", "1:38.54", "25",
          photo=True),
        R("4", "11", "FOXGLOVE DASH", "HK_2023_X478", "X478", "L Fernwhistle", "FWL",
          "K Y Saltmarsh", "SKY", "118", "998", "10", "1-1/4", "7 7 7 4", "1:38.62", "8.9"),
        R("5", "4", "RIVER OF MINT", "HK_2021_V560", "V560", "C Marlowick", "MKC",
          "D Linnetby", "LND", "129", "1213", "3", "2-1/2", "2 3 3 5", "1:38.81", "45"),
        R("6", "1", "HARBOUR MILE", "HK_2022_W390", "W390", "E Thistledown", "TDE",
          "G Pellaworth", "PWG", "135", "1260", "8", "4-1/2", "5 5 6 6", "1:39.14", "99"),
        R("7", "12", "QUIET METEOR", "HK_2024_Y210", "Y210", "O Kestrelby", "KBO",
          "M Q Larkspur", "LMQ", "115", "1020", "2", "10", "4 4 5 7", "1:40.04", "61"),
        R("WV-A", "7", "NORTHERN SAPPHIRE", "HK_2023_X300", "X300", "H Wrenfield", "WFH",
          "D Linnetby", "LND", "118", "---", "---", "---", "---", "---", "---"),
    ),
    pools=(
        Pool("WIN", (("6", "37.00"),)),
        Pool("PLACE", (("6", "15.50"), ("2", "32.00"), ("9", "61.50"))),
        Pool("QUINELLA", (("2,6", "221.00"),)),
        Pool("QUINELLA PLACE", (("2,6", "80.50"), ("6,9", "142.00"), ("2,9", "301.00"))),
        Pool("FORECAST", (("6,2", "402.00"),)),
        Pool("TIERCE", (("6,2,9", "3,918.00"),)),
        Pool("TRIO", (("2,6,9", "1,066.00"),)),
        Pool("FIRST 4", (("2,6,9,11", "6,251.00"),)),
        Pool("QUARTET", (("6,2,9,11", "57,340.00"),)),
        Pool("3 PICK 1 (COMPOSITE WIN)",
             (("A1", "10.10"), ("A2", "NOT WIN"), ("A3", "14.50"), ("", "Detail"))),
        Pool("2ND DOUBLE", (("4/6", "96.50"), ("4/2", "412.00"))),
    ),
    incidents=(
        Incident("1", "6", "AMBER LANTERN", "HK_2023_X101", "X101",
                 "Bumped at the start.  Sent for sampling."),
        Incident("2", "2", "COPPER TIDE", "HK_2022_W215", "W215",
                 "Raced keenly in the early stages."),
        Incident("3", "9", "SILENT ORCHARD", "HK_2024_Y033", "Y033", "No information."),
        Incident("4", "11", "FOXGLOVE DASH", "HK_2023_X478", "X478", ""),
        Incident("5", "4", "RIVER OF MINT", None, "V560", "Lost its off-fore plate."),
        Incident("6", "1", "HARBOUR MILE", "HK_2022_W390", "W390",
                 "Slow to begin.  The Stewards noted that Race 3 at the previous meeting was "
                 "declared void."),
        Incident("7", "12", "QUIET METEOR", "HK_2024_Y210", "Y210", "Sent for sampling."),
        Incident("WV-A", "7", "NORTHERN SAPPHIRE", "HK_2023_X300", "X300",
                 "Withdrawn by the veterinary officer after becoming fractious in the stalls."),
    ),
)

# header variants of the live page (same runners and dividends)
AWT = replace(
    LIVE, race_date=date(2023, 12, 16), venue="ST", race_no=6, race_index=377,
    class_line="Class 4 (Special Condition) - 1200M - (60-35)", going="GOOD",
    race_name="MOSSY BRIDGE HANDICAP", course="ALL WEATHER TRACK", prize="HK$ 1,170,000",
    times=("23.90", "46.80", "1:09.70"), n_races=10,
)
GROUP = replace(
    LIVE, race_date=date(2025, 12, 14), venue="ST", race_no=8, race_index=255,
    class_line="Group One - 2400M", going="YIELDING", race_name="WINTER PEARL CUP",
    course='Turf - "A+2" course', prize="HK$ 30,000,000",
    times=("25.40", "50.10", "1:14.32", "1:38.75", "2:02.61", "2:27.35"), n_races=10,
)
OPEN_BAND = replace(
    LIVE, race_date=date(2026, 11, 1), venue="ST", race_no=9, race_index=180,
    class_line="Class 1 - 1400M - (95+)", going="GOOD TO YIELDING",
    race_name="AUTUMN REED HANDICAP", course="TURF - “B” Course", prize="HK$ 3,450,000",
    times=("13.30", "35.02", "58.61", "1:22.09"), n_races=10,
)


# ---------------------------------------------------------------- archive layouts

# 2010-era archive: narrative stewards' report, brand codes that differ from the horse id,
# retired people printed without a link
ARCHIVE_2010 = Race(
    race_date=date(2011, 1, 30), venue="ST", race_no=1, race_index=312,
    class_line="Class 3 - 1200M - (80-60)", going="GOOD", race_name="PEBBLE COVE HANDICAP",
    course='TURF - "B+2" Course', prize="HK$ 900,000", times=("23.85", "46.71", "1:09.94"),
    n_races=10,
    runners=(
        R("1", "3", "SPARROW KING", "HK_2009_Q101", "BQ101", "N Gallowmere", None,
          "P O'Ashgrove", None, "127", "1079", "9", "---", "4 2 1", "1:09.94", "2.4"),
        R("2", "8", "MISTY HARBOUR", "HK_2009_Q215", "BQ215", "S Ravensby", "RVS",
          "W Y Gantry", "GWY", "121", "1032", "2", "1/2", "1 1 2", "1:10.02", "6.1"),
        R("3", "1", "TIGER'S DAWN", "HK_2008_P330", "AP330", "J Kittering", "KTJ",
          "D Linnetby", "LND", "133", "1150", "5", "1-1/4", "6 5 3", "1:10.14", "15"),
        R("4", "6", "SPARROW", "HK_2009_Q044", "BQ044", "M Brackwood", "BWM",
          "W Y Gantry", "GWY", "118", "1001", "12", "1-1/2", "3 4 4", "1:10.18", "9.9"),
        R("5", "2", "COPPER BELL", "HK_2008_P512", "AP512", "F Underhay", "UHF",
          "P O'Ashgrove", None, "126", "1110", "7", "2-3/4", "7 7 5", "1:10.38", "33"),
        R("6", "7", "RIVER OTTER", "HK_2009_Q620", "BQ620", "T Wickerly", "WKT",
          "D Linnetby", "LND", "115", "987", "1", "5", "2 3 6", "1:10.74", "41"),
        R("7", "4", "QUIET FERN", "HK_2007_N077", "AN077", "N Gallowmere", None,
          "W Y Gantry", "GWY", "113", "1066", "3", "8-1/4", "5 6 7", "1:11.26", "72"),
    ),
    pools=(
        Pool("WIN", (("3", "24.50"),)),
        Pool("PLACE", (("3", "12.00"), ("8", "19.50"), ("1", "27.00"))),
        Pool("QUINELLA", ((" 3 , 8 ", "88.00"),)),
        Pool("QUINELLA PLACE", (("3,8", "35.50"), ("01,03", "48.00"), ("1,8", "77.50"))),
        Pool("TIERCE", (("3,8,1", "1,290.00"),)),
        Pool("TRIO", (("1,3,8", "302.00"),)),
        Pool("QUARTET", (("3,8,1,6", "226,778.10/$1.0"),)),
        Pool("1ST DOUBLE", (("2/3", "61.00"),)),
        Pool("TRIPLE TRIO(Consolation)", (("2,5,7/1,4,9/Any Combination", "5,193.00"),)),
    ),
    narrative=(
        "SPARROW KING was slow to begin.  On jumping, TIGER’S DAWN shifted in and bumped "
        "SPARROW.  SPARROW then raced wide without cover.He said the gelding did not handle "
        "the bend.",
        "The start was delayed by two minutes.",
        "A veterinary inspection of COPPER BELL found it to be lame\nin the left foreleg.  "
        "MISTY HARBOUR and RIVER OTTER were sent for sampling.  COPPER BELL will be examined "
        "again before its next start.",
    ),
)

# 2004-era archive: no Running Position column, no stewards' report
ARCHIVE_2004 = Race(
    race_date=date(2005, 3, 6), venue="ST", race_no=1, race_index=458,
    class_line="Griffin Race - 1200M", going="GOOD TO FIRM", race_name="FERN GULLY PLATE",
    course='TURF - "C" Course', prize="HK$ 400,000", times=("24.10", "47.02", "1:10.48"),
    columns=NO_RUNNING_COLUMNS, n_races=8,
    runners=(
        R("1", "3", "PEBBLE PRINCE", "HK_2002_R086", "CR086", "B Hollowell", None,
          "W Y Gantry", "GWY", "122", "1088", "2", "---", "", "1:10.48", "4.8"),
        R("2", "5", "GENTLE THUNDER", "HK_2003_S117", "CS117", "K Ormsby", "ORK",
          "A Fieldhurst", "FHA", "126", "1121", "6", "3/4", "", "1:10.60", "7.2"),
        R("3", "1", "WILLOW CHARM", "HK_2002_R240", "CR240", "E Pennick", None,
          "A Fieldhurst", "FHA", "130", "1170", "4", "1-3/4", "", "1:10.76", "11"),
        R("4", "4", "MARBLE COAST", "HK_2003_S005", "CS005", "L Daventree", "DVL",
          "W Y Gantry", "GWY", "119", "1012", "1", "2", "", "1:10.80", "19"),
        R("5", "2", "DUSTY BEACON", "HK_2001_Q901", "BQ901", "G Asherby", "ASG",
          "R Tolliver", None, "117", "1095", "3", "6", "", "1:11.44", "35"),
    ),
    pools=(
        Pool("WIN", (("3", "48.50"),)),
        Pool("PLACE", (("3", "18.00"), ("5", "22.50"), ("1", "30.00"))),
        Pool("QUINELLA", (("3,5", "96.00"),)),
    ),
)


# ---------------------------------------------------------------- dead heats

DEAD_HEAT = Race(  # for first
    race_date=date(2021, 12, 8), venue="HV", race_no=2, race_index=190,
    class_line="Class 4 - 1200M - (60-40)", going="GOOD", race_name="WILLOW LANE HANDICAP",
    course='TURF - "A" Course', prize="HK$ 1,050,000", times=("23.95", "46.66", "1:10.62"),
    runners=(
        R("1 DH", "5", "SPEEDY ACORN", "HK_2019_T253", "T253", "Q Hartfield", "HFQ",
          "C Brookby", "BKC", "131", "1219", "3", "DH", "3 3 1", "1:10.62", "4.1"),
        R("1 DH", "12", "GREAT MEADOW", "HK_2018_S390", "S390", "V Dunmore", "DMV",
          "K Lowther", "LWK", "119", "1131", "12", "DH", "6 6 1", "1:10.62", "13"),
        R("3", "8", "VITAL STREAM", "HK_2017_R057", "R057", "Y Corrigan", "CRY",
          "K W Lunsford", "LKW", "128", "1140", "10", "NOSE", "2 2 3", "1:10.63", "34"),
        R("4", "11", "NOBLE FROST", "HK_2019_T614", "T614", "S Ravensby", "RVS",
          "C Brookby", "BKC", "122", "1066", "6", "SH", "1 1 4", "1:10.64", "7.5"),
        R("5", "3", "LUCKY CLOVER", "HK_2018_S222", "S222", "E Pennick", "PNE",
          "K Lowther", "LWK", "126", "1175", "1", "HD", "4 4 5", "1:10.65", "19"),
        R("6", "1", "JADE RUNNER", "HK_2020_U009", "U009", "B Avonley", "AVB",
          "F Rookwood", "RWF", "133", "1202", "8", "2-1/2", "5 5 6", "1:11.02", "60"),
    ),
    pools=(
        Pool("WIN", (("5", "20.50"), ("12", "65.50"))),
        Pool("PLACE", (("5", "15.50"), ("12", "35.00"), ("8", "68.50"))),
        Pool("QUINELLA", (("5,12", "274.00"),)),
        Pool("QUINELLA PLACE", (("5,12", "97.50"), ("5,8", "182.00"), ("8,12", "515.00"))),
        Pool("FORECAST", (("5,12", "251.00"), ("12,5", "433.00"))),
        Pool("TIERCE", (("5,12,8", "5,044.00"), ("12,5,8", "8,120.00"))),
        Pool("TRIO", (("5,8,12", "2,710.00"),)),
        Pool("QUARTET", (("5,12,8,11", "46,980.00"), ("12,5,8,11", "77,105.00"))),
    ),
)

DEAD_HEAT_THIRD = Race(
    race_date=date(2022, 2, 16), venue="HV", race_no=5, race_index=327,
    class_line="Class 3 - 1650M - (80-60)", going="GOOD", race_name="MAPLE HILL HANDICAP",
    course='TURF - "B" Course', prize="HK$ 1,450,000",
    times=("25.10", "48.70", "1:13.20", "1:39.80"),
    runners=(
        R("1", "2", "MAPLE COMET", "HK_2020_U201", "U201", "S Ravensby", "RVS",
          "C Brookby", "BKC", "128", "1120", "5", "---", "2 2 1", "1:39.80", "3.2"),
        R("2", "7", "HONEY RIDGE", "HK_2019_T733", "T733", "Y Corrigan", "CRY",
          "K Lowther", "LWK", "124", "1075", "9", "1/2", "1 1 2", "1:39.88", "6.6"),
        R("3 DH", "4", "IRON THISTLE", "HK_2020_U480", "U480", "V Dunmore", "DMV",
          "C Brookby", "BKC", "131", "1190", "2", "1-1/4", "5 4 3", "1:40.00", "14"),
        R("3 DH", "10", "PAPER KITE", "HK_2021_V091", "V091", "Q Hartfield", "HFQ",
          "K W Lunsford", "LKW", "119", "1018", "11", "1-1/4", "4 5 3", "1:40.00", "27"),
        R("5", "1", "OCEAN BRAMBLE", "HK_2019_T150", "T150", "E Pennick", "PNE",
          "K Lowther", "LWK", "133", "1233", "1", "2-3/4", "3 3 5", "1:40.24", "8.0"),
    ),
    pools=(
        Pool("WIN", (("2", "32.50"),)),
        Pool("PLACE", (("2", "14.00"), ("7", "20.50"), ("4", "22.00"), ("10", "39.50"))),
        Pool("QUINELLA", (("2,7", "66.00"),)),
        Pool("TIERCE", (("2,7,4", "412.00"), ("2,7,10", "655.00"))),
        Pool("TRIO", (("2,4,7", "133.00"), ("2,7,10", "241.00"))),
    ),
)


# ---------------------------------------------------------------- races without a result

# A meeting abandoned on the night (2019-era banner): header and declared runners, nothing run.
ABANDONED = Race(
    race_date=date(2022, 12, 7), venue="HV", race_no=1, race_index=0,
    class_line="Class 5 - 1200M - (40-0)", going="", race_name="BAMBOO PATH HANDICAP",
    course='TURF - "C" Course', prize="HK$ 760,000", times=(),
    banner="This race has been abandoned or not offered.  Bets on its single race pools are "
           "refunded.",
    columns=NOT_RUN_COLUMNS, pool_classes=False,
    runners=(
        R("", "1", "MORNING CEDAR", "HK_2020_U301", "U301", "H Wrenfield", "WFH",
          "C Brookby", "BKC", "---", "---", "12", "---", "", "---", ""),
        R("", "2", "LAKESIDE GLOW", "HK_2019_T402", "T402", "K Ellery", "ELK",
          "K Lowther", "LWK", "---", "---", "7", "---", "", "---", ""),
        R("", "3", "RUBY TRAIL", "HK_2021_V503", "V503", "S Ravensby", "RVS",
          "F Rookwood", "RWF", "---", "---", "9", "---", "", "---", ""),
        R("", "4", "WINTER QUILL", "HK_2020_U604", "U604", "B Avonley", "AVB",
          "D Linnetby", "LND", "---", "---", "3", "---", "", "---", ""),
    ),
    pools=(
        Pool("WIN", (REFUND,)),
        Pool("PLACE", (REFUND,)),
        Pool("QUINELLA", (REFUND,)),
        Pool("QUINELLA PLACE", (REFUND,)),
        Pool("3 PICK 1 (COMPOSITE WIN)", (("A1", "REFUND"), ("A2", "REFUND"), ("A3", "REFUND"))),
    ),
)

# 2024+ wording: neither a race header nor a runner table, only the refunds; multi-race pools
# that ended on this leg still pay ("F" = any runner in that leg).
DECLARED_ABANDONED = Race(
    race_date=date(2025, 11, 12), venue="HV", race_no=7,
    banner="This race is declared abandoned.  Bets on its single race pools are refunded.",
    header=False, runner_table=False, pool_classes=False,
    pools=(
        *(Pool(label, (REFUND,)) for label in (
            "WIN", "PLACE", "QUINELLA", "QUINELLA PLACE", "FORECAST", "TIERCE", "TRIO",
            "FIRST 4", "QUARTET",
        )),
        Pool("6TH DOUBLE", (("7/F", "31.50"),)),
        Pool("3RD DOUBLE TRIO", (("7,10,11/F", "2,904.00"),)),
    ),
)

# Annulled after the start: header, runners all placed VOID, one withdrawn without a number,
# and the stewards' per-horse table still published.
VOID = Race(
    race_date=date(2025, 10, 18), venue="ST", race_no=8, race_index=214,
    class_line="Class 4 - 1200M - (60-40)", going="", race_name="OPAL TERRACE HANDICAP",
    course='TURF - "A+3" Course', prize="HK$ 1,150,000", times=(), n_races=10,
    banner="This race is declared void.  Bets on its single race pools are refunded.",
    columns=NOT_RUN_COLUMNS, pool_classes=False,
    runners=(
        R("VOID", "1", "ENDLESS SUMMIT", "HK_2021_V033", "V033", "H Wrenfield", "WFH",
          "J Ritchfield", "RFJ", "---", "---", "12", "---", "", "---", ""),
        R("VOID", "2", "EMBER COVE", "HK_2022_W122", "W122", "K Ellery", "ELK",
          "F Rookwood", "RWF", "---", "---", "9", "---", "", "---", ""),
        R("VOID", "3", "AURORA MILL", "HK_2021_V056", "V056", "J Ormeroyd", "ORJ",
          "K Lowther", "LWK", "---", "---", "4", "---", "", "---", ""),
        R("VOID", "4", "DREAMY HOLLOW", "HK_2022_W046", "W046", "M Brackwood", "BWM",
          "F Rookwood", "RWF", "---", "---", "1", "---", "", "---", ""),
        R("WV", "", "HUSHED VALE", "HK_2023_X547", "X547", "Y Corrigan", "CRY",
          "C Brookby", "BKC", "121", "---", "---", "---", "", "---", ""),
    ),
    pools=tuple(Pool(label, (REFUND,)) for label in (
        "WIN", "PLACE", "QUINELLA", "QUINELLA PLACE", "FORECAST", "TIERCE",
    )),
    incidents=(
        Incident("VOID", "1", "ENDLESS SUMMIT", "HK_2021_V033", "V033",
                 "Hampered by DREAMY HOLLOW at the start."),
        Incident("VOID", "2", "EMBER COVE", "HK_2022_W122", "W122", ""),
        Incident("VOID", "3", "AURORA MILL", "HK_2021_V056", "V056", "Raced without cover."),
        Incident("VOID", "4", "DREAMY HOLLOW", "HK_2022_W046", "W046",
                 "Dislodged its rider soon after the start and galloped riderless."),
    ),
)


# ---------------------------------------------------------------- placings amended after a protest

# The first horse home (3) is relegated to second: the promoted winner (4) prints a negative
# margin "-HD", the relegated horse "---"; the race time is the relegated horse's.
PROTEST = Race(
    race_date=date(2026, 3, 15), venue="ST", race_no=5, race_index=466,
    class_line="Class 4 - 1600M - (60-40)", going="GOOD", race_name="BRONZE GATE HANDICAP",
    course='TURF - "C" Course', prize="HK$ 1,170,000",
    times=("24.31", "47.25", "1:10.90", "1:34.70"), n_races=11,
    runners=(
        R("1", "4", "THE BRONZE KNIGHT", "HK_2022_W341", "W341", "J Ormeroyd", "ORJ",
          "K W Lunsford", "LKW", "130", "1158", "4", "-HD", "8 7 6 2", "1:34.72", "2.9"),
        R("2", "3", "MASTER OF TIDES", "HK_2021_V442", "V442", "K Ellery", "ELK",
          "F Rookwood", "RWF", "135", "1091", "1", "---", "4 4 3 1", "1:34.70", "9.5"),
        R("3", "9", "MEGA LANTERN", "HK_2024_Y044", "Y044", "B Avonley", "AVB",
          "J Ritchfield", "RFJ", "123", "1212", "12", "1/2", "2 2 2 3", "1:34.79", "15"),
        R("4", "1", "SILVER DAWN", "HK_2023_X612", "X612", "H Wrenfield", "WFH",
          "F Rookwood", "RWF", "128", "1101", "6", "1-1/4", "5 5 5 4", "1:34.92", "21"),
        R("PU", "6", "RUSTY ANCHOR", "HK_2020_U118", "U118", "M Brackwood", "BWM",
          "K W Lunsford", "LKW", "121", "1043", "9", "---", "6 6", "---", "48"),
        R("WX-A", "2", "GOLDEN SPUR", "HK_2022_W777", "W777", "D Fairholm", "FHD",
          "J Ritchfield", "RFJ", "126", "---", "---", "---", "---", "---", "---"),
    ),
    pools=(
        Pool("WIN", (("4", "29.50"),)),
        Pool("PLACE", (("4", "13.00"), ("3", "26.50"), ("9", "41.00"))),
        Pool("QUINELLA", (("3,4", "158.00"),)),
        Pool("QUINELLA PLACE", (("3,4", "60.50"), ("4,9", "88.00"), ("3,9", "175.50"))),
    ),
    incidents=(
        Incident("1", "4", "THE BRONZE KNIGHT", "HK_2022_W341", "W341",
                 "Bumped near the 300 Metres.  Ran on strongly."),
        Incident("2", "3", "MASTER OF TIDES", "HK_2021_V442", "V442",
                 "Shifted in near the 300 Metres and bumped THE BRONZE KNIGHT.  After an "
                 "inquiry the Stewards placed it second."),
        Incident("3", "9", "MEGA LANTERN", "HK_2024_Y044", "Y044", ""),
        Incident("4", "1", "SILVER DAWN", None, "X612", "Sent for sampling."),
        Incident("PU", "6", "RUSTY ANCHOR", "HK_2020_U118", "U118",
                 "Pulled up near the 600 Metres."),
        Incident("WX-A", "2", "GOLDEN SPUR", "HK_2022_W777", "W777",
                 "Withdrawn by order of the Stewards."),
    ),
)


# ---------------------------------------------------------------- malformed cells

MALFORMED = Race(
    race_date=date(2021, 5, 23), venue="ST", race_no=3, race_index=88,
    class_line="Group One - 2400M", going="YIELDING", race_name="GRAND MEADOW CUP",
    course='TURF - "A+3" Course', prize="HK$ 24,000,000", times=("25.10", "2:27.35"),
    runners=(
        R("1", "4", "STORM PETREL", "HK_2020_Z100", "Z100", "P Quayside (-2)", "QSP",
          "J Merrow", None, "126", "1100", "2", "-", "3 2 1", "2:27.35", "1.9"),
        R("2 DH", "7", "BRAVE OTTER", "HK_2020_Z200", "Z200", "A Bellweather", "../x",
          "C Dunlin", None, "126", "abc", "x", "WEIRD", "1 1 x", "2.27.50", "n/a"),
        R("2 DH", "8", "CALM HERON", "HK_2020_Z300", "Z300", "G Fennimore", "FNG",
          "C Dunlin", None, "126", "1000", "5", "1", "2 3 2", "2:27.50", "15"),
        R("PU", "9", "DUSKY WREN", "HK_2020_Z400", "Z400", "H Tamworth", "TMH",
          "C Dunlin", None, "126", "1000", "6", "---", "", "---", "99"),
        R("WV", "10", "EMBER FINCH", "HK_2020_Z500", "Z500", "K Ashby", "ASK",
          "C Dunlin", None, "---", "---", "---", "---", "---", "---", "---"),
        R("5", "11", "FROSTY LARK", "../../etc/passwd", "Z600", "", None,
          "", None, "", "", "", "", "", "", "", short=3),
        R("stray note row", "", "", None, "", "", None,
          "", None, "", "", "", "", "", "", "", short=1),
    ),
    pools=(
        Pool("WIN", (("4", "19.00"),)),
        Pool("PLACE", ((" 4 ", "1,234.50"), ("07 , 8", "garbled"), ("orphan continuation",))),
    ),
)
# fmt: on
