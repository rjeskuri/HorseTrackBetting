"""Builders for minimal synthetic netkeiba result pages (ticket #52): db.netkeiba.com/race/<id>/.

Only the result page lives here; the calendar, race-list and pedigree-card builders of #51 are in
netkeiba_pages.py.

Each page holds only the structure HKTrack's `sources/jra/parse_results.py` reads, laid out as on
the site: the race-number strip (`div.race_num`, the current race's link `a.active`), the race
header `dl.racedata` (the race name in `h1`, the course / 天候 / going / 発走 line in `dd p span`)
and `p.smalltxt` (date, meeting, class, conditions), the runner table `table.race_table_01` read
by its header text (with a few columns the parser skips, some inside `<diary_snap_cut>` as on the
site), the two `table.pay_table_01`, and the `table.result_table_02` laps (the ラップ row, next to
a ペース row the parser must not read). Every horse, jockey, trainer, owner, id, time, price and
race name is invented; no real page is copied. Pages come back as UTF-8 bytes.

The five pages: PLAIN (a flat turf race), DEAD_HEAT (a dead heat for 3rd), SCRATCHED (a 取消
runner and a dead heat for 2nd), DEMOTED (a 降着, a 失格 and a 除外 runner), JUMP (a jump race,
two runners pulled up 中止, no lap times). Build variants with `dataclasses.replace` and `restate`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

# A runner table is one line per runner, in page order, of these printed cells ("-" = empty).
RUNNER_COLUMNS = (
    "placing",
    "bracket",
    "horse_no",
    "sex_age",
    "carried_kg",
    "jockey_code",
    "time",
    "margin",
    "passing",
    "last_3f",
    "win_odds",
    "fav_rank",
    "body_wt",
    "trainer_code",
)
EMPTY = "-"
SENTINEL_TITLE = "1970年1月01日"

_NUMERALS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二")
JOCKEYS = {f"059{i:02d}": f"架空騎手{n}" for i, n in enumerate(_NUMERALS, 1)}
TRAINERS = {f"012{i:02d}": f"架空調教師{n}" for i, n in enumerate(_NUMERALS, 1)}
# horse name = the page's name stem + the word of its horse number
_WORDS = (
    "アカ", "アオ", "キイロ", "ミドリ", "シロ", "クロ", "モモ", "ハイ", "ギン",
    "キン", "ソラ", "ウミ", "ヤマ", "カワ", "ホシ", "ツキ", "ユキ", "カゼ",
)  # fmt: skip
_PAY_CLASSES = {
    "単勝": "tan", "複勝": "fuku", "枠連": "waku", "馬連": "uren",
    "ワイド": "wide", "馬単": "utan", "三連複": "sanfuku", "三連単": "santan",
}  # fmt: skip
_FIRST_PAY_TABLE = ("単勝", "複勝", "枠連", "馬連")
_PRIZES = ("8,000.0", "3,200.0", "2,000.0", "1,200.0", "800.0")


@dataclass(frozen=True)
class Runner:
    """One runner row as printed ("" = an empty cell)."""

    placing: str
    bracket: int
    horse_no: int
    horse_id: str
    horse_name: str
    sex_age: str
    carried_kg: str
    jockey_code: str
    jockey_name: str
    time: str
    margin: str
    passing: str
    last_3f: str
    win_odds: str
    fav_rank: str
    body_wt: str
    trainer_code: str
    trainer_name: str
    trainer_region: str


@dataclass(frozen=True)
class ResultPage:
    key: str  # the 12-digit race id: year, venue number, meeting (kai), day, race number
    race_date: date
    meeting: str  # 5回京都6日目
    race_name: str  # the h1: the race's name, or its class text when it has none
    course: str  # 芝右 外1600m
    weather: str
    going_label: str  # 芝 | ダート
    going: str  # 良 | 稍重 | 重 | 不良
    post_time: str
    class_text: str  # 3歳以上オープン
    conditions: str  # (国際)(指)(定量)
    table: str  # see RUNNER_COLUMNS
    pays: str  # one line per pool: label, then "combination amount" pairs split by " / "
    laps: str | None  # the ラップ cell; None prints the lap table without rows (jump races)
    horse_stem: str  # 8 digits: horse id = stem + the 2-digit horse number
    name_stem: str
    old_links: bool = False  # /jockey/<code>/ links instead of /jockey/result/recent/<code>/

    @property
    def race_no(self) -> int:
        return int(self.key[10:12])


def runners(page: ResultPage) -> list[Runner]:
    """The runner rows of `page` as printed, in page order."""
    out = []
    for cells in _table_rows(page):
        printed = {column: ("" if value == EMPTY else value) for column, value in cells.items()}
        no = int(printed["horse_no"])
        jockey, trainer = printed["jockey_code"], printed["trainer_code"]
        out.append(
            Runner(
                placing=printed["placing"],
                bracket=int(printed["bracket"]),
                horse_no=no,
                horse_id=f"{page.horse_stem}{no:02d}",
                horse_name=page.name_stem + _WORDS[no - 1],
                sex_age=printed["sex_age"],
                carried_kg=printed["carried_kg"],
                jockey_code=jockey,
                jockey_name=JOCKEYS[jockey],
                time=printed["time"],
                margin=printed["margin"],
                passing=printed["passing"],
                last_3f=printed["last_3f"],
                win_odds=printed["win_odds"],
                fav_rank=printed["fav_rank"],
                body_wt=printed["body_wt"],
                trainer_code=trainer,
                trainer_name=TRAINERS[trainer],
                trainer_region="東" if int(trainer) % 2 else "西",
            )
        )
    return out


def restate(page: ResultPage, horse_no: int, **cells: str) -> ResultPage:
    """`page` with some printed cells of one runner changed, e.g. restate(p, 7, placing="除")."""
    unknown = set(cells) - set(RUNNER_COLUMNS)
    if unknown:
        raise KeyError(f"not runner columns: {sorted(unknown)}")
    lines, found = [], False
    for row in _table_rows(page):
        if int(row["horse_no"]) == horse_no:
            row.update(cells)
            found = True
        lines.append(" ".join(row[column] for column in RUNNER_COLUMNS))
    if not found:
        raise KeyError(f"no runner {horse_no}")
    return replace(page, table="\n".join(lines))


def pay_rows(page: ResultPage) -> list[tuple[str, list[tuple[str, str]]]]:
    """(label, [(combination, amount), ...]) per pool, in page order."""
    rows = []
    for line in page.pays.strip().splitlines():
        label, rest = line.split(maxsplit=1)
        rows.append((label, [tuple(part.split()) for part in rest.split(" / ")]))
    return rows


# ---------------------------------------------------------------- pages


def render(page: ResultPage) -> bytes:
    d = page.race_date
    title = f"{page.race_name}｜{d.year}年{d.month}月{d.day:02d}日 | synthetic"
    body = (
        _race_strip(page)
        + _race_header(page)
        + _runner_table(page)
        + _pay_tables(page)
        + _lap_tables(page)
    )
    return _document(title, body)


def cut_off(page: ResultPage) -> bytes:
    """`page` cut off before its result table (a page the site did not finish sending)."""
    body = render(page)
    return body[: body.index(b'<table class="race_table_01"')]


def sentinel() -> bytes:
    """What the site answers for an unknown race id: HTTP 200 and an epoch-dated race page with
    empty fields and no result table."""
    header = (
        '<div class="data_intro">\n<dl class="racedata fc">\n<dt>\n\n</dt>\n<dd>\n<h1>\n\n</h1>\n'
        "<p>\n<span>\n&nbsp;/&nbsp;\n天候 : &nbsp;/&nbsp;\n&nbsp;&nbsp;/&nbsp;\n</span><br />\n"
        "</p>\n</dd>\n</dl>\n"
        '<p class="smalltxt">1970年01月01日 &nbsp;&nbsp;&nbsp;</p>\n</div>\n'
    )
    return _document(f"｜{SENTINEL_TITLE} | synthetic", header)


def error_page() -> bytes:
    return _document("503 Service Unavailable", "<h1>Service Unavailable</h1><p>Try later.</p>")


def _document(title: str, body: str) -> bytes:
    head = f'<head>\n<meta charset="UTF-8">\n<title>{title}</title>\n</head>'
    return f'<!DOCTYPE html>\n<html lang="ja">\n{head}\n<body>\n{body}</body>\n</html>\n'.encode()


def _table_rows(page: ResultPage) -> list[dict[str, str]]:
    return [
        dict(zip(RUNNER_COLUMNS, line.split(), strict=True))
        for line in page.table.strip().splitlines()
    ]


def _race_strip(page: ResultPage) -> str:
    stem = page.key[:10]
    items = []
    for n in range(1, 13):
        active = "active" if n == page.race_no else ""
        items.append(
            f'<li><a href="/race/{stem}{n:02d}/" title="{n}R" class="{active}">{n}R</a></li>'
        )
    return f'<div class="race_num fc">\n<ul class="fc">\n{"".join(items)}\n</ul>\n</div>\n'


def _race_header(page: ResultPage) -> str:
    d = page.race_date
    line = (
        f"{page.course}&nbsp;/&nbsp;\n天候 : {page.weather}&nbsp;/&nbsp;\n"
        f"{page.going_label} : {page.going}&nbsp;&nbsp;/&nbsp;\n発走 : {page.post_time}"
    )
    return (
        '<div class="data_intro">\n<dl class="racedata fc">\n'
        f"<dt>\n{page.race_no}R\n</dt>\n<dd>\n<h1>\n{page.race_name}\n</h1>\n"
        f"<diary_snap_cut>\n<p>\n<span>\n{line}\n</span><br />\n</p>\n</diary_snap_cut>\n"
        "</dd>\n</dl>\n"
        f'<p class="smalltxt">{d.year}年{d.month:02d}月{d.day:02d}日 {page.meeting}'
        f"&nbsp;{page.class_text}&nbsp;&nbsp;{page.conditions}</p>\n</div>\n"
    )


_RUNNER_HEADER = (
    '<tr class="txt_c">\n'
    "<th>着<br />順</th>\n<th>枠<br />番</th>\n<th>馬<br />番</th>\n<th>馬名</th>\n"
    "<th>性齢</th>\n<th>斤量</th>\n<th>騎手</th>\n<th>タイム</th>\n<th>着差</th>\n"
    "<diary_snap_cut>\n<th>タイム<br />指数</th>\n<th>通過</th>\n<th>上り</th>\n</diary_snap_cut>\n"
    "<th>単勝</th>\n<th>人<br />気</th>\n<th>馬体重</th>\n"
    "<diary_snap_cut>\n<th>備考</th>\n</diary_snap_cut>\n<th>調教師</th>\n"
    "<diary_snap_cut>\n<th>馬主</th>\n<th>賞金<br />(万円)</th>\n</diary_snap_cut>\n"
    "</tr>\n"
)


def _runner_table(page: ResultPage) -> str:
    rows = "".join(_runner_row(page, r) for r in runners(page))
    return f'<table class="race_table_01">\n{_RUNNER_HEADER}{rows}</table>\n'


def _runner_row(page: ResultPage, r: Runner) -> str:
    people = "" if page.old_links else "result/recent/"
    jockey = f"/jockey/{people}{r.jockey_code}/"
    trainer = f"/trainer/{people}{r.trainer_code}/"
    owner_id = f"9{page.horse_stem[-3:]}{r.horse_no:02d}"
    owner = f"架空馬主{_NUMERALS[(r.horse_no - 1) % len(_NUMERALS)]}"
    place = int(r.placing) if r.placing.isdigit() else None
    prize = _PRIZES[place - 1] if place is not None and place <= len(_PRIZES) else ""
    return (
        "<tr>\n"
        f'<td class="txt_r">{r.placing}</td>\n'
        f'<td class="w{r.bracket}ml"><span>{r.bracket}</span></td>\n'
        f'<td class="txt_r">{r.horse_no}</td>\n'
        f'<td class="txt_l">\n'
        f'<a href="/horse/{r.horse_id}/" title="{r.horse_name}">{r.horse_name}</a>\n</td>\n'
        f'<td class="txt_c">{r.sex_age}</td>\n'
        f'<td class="txt_c">{r.carried_kg}</td>\n'
        f'<td class="txt_l">\n'
        f'<a href="{jockey}" title="{r.jockey_name}">{r.jockey_name}</a>\n</td>\n'
        f'<td class="txt_r">{r.time}</td>\n'
        f'<td class="txt_c">{r.margin}</td>\n'
        "<diary_snap_cut>\n"
        '<td class="speed_index"><div class="txt_c"><a href="#">**</a></div></td>\n'
        f"<td>{r.passing}</td>\n"
        f'<td class="txt_c"><span>{r.last_3f}</span></td>\n'
        "</diary_snap_cut>\n"
        f'<td class="txt_r">{r.win_odds}</td>\n'
        f'<td class="txt_r"><span>{r.fav_rank}</span></td>\n'
        f"<td>{r.body_wt}</td>\n"
        "<diary_snap_cut>\n<td></td>\n</diary_snap_cut>\n"
        f'<td class="txt_l">\n[{r.trainer_region}]\n'
        f'<a href="{trainer}" title="{r.trainer_name}">{r.trainer_name}</a>\n</td>\n'
        "<diary_snap_cut>\n"
        f'<td class="txt_l"><a href="/owner/result/recent/{owner_id}/">{owner}</a></td>\n'
        f'<td class="txt_r">{prize}</td>\n'
        "</diary_snap_cut>\n"
        "</tr>\n"
    )


def _pay_tables(page: ResultPage) -> str:
    first, second = [], []
    for label, pairs in pay_rows(page):
        combinations = "<br />".join(c for c, _ in pairs)
        amounts = "<br />".join(a for _, a in pairs)
        popularity = "<br />".join(str(i) for i in range(1, len(pairs) + 1))
        row = (
            f'<tr>\n<th class="{_PAY_CLASSES[label]}">{label}</th>\n<td>{combinations}</td>\n'
            f'<td class="txt_r">{amounts}</td>\n<td class="txt_r">{popularity}</td>\n</tr>\n'
        )
        (first if label in _FIRST_PAY_TABLE else second).append(row)
    tables = "".join(
        f'<table class="pay_table_01">\n{"".join(rows)}</table>\n' for rows in (first, second)
    )
    return f'<dl class="pay_block">\n<dt>払い戻し</dt>\n<dd class="fc">\n{tables}</dd>\n</dl>\n'


def _lap_tables(page: ResultPage) -> str:
    order = "-".join(str(r.horse_no) for r in runners(page) if r.placing[:1].isdigit())
    corners = (
        '<table class="result_table_02">\n<caption>コーナー通過順位</caption>\n'
        f"<tr>\n<th>4コーナー</th>\n<td>{order}</td>\n</tr>\n</table>\n"
    )
    rows = ""
    if page.laps is not None:
        total, pace = 0.0, []
        for lap in page.laps.split("-"):
            total = round(total + float(lap), 1)
            pace.append(f"{total:.1f}")
        rows = (
            f'<tr>\n<th>ラップ</th>\n<td class="race_lap_cell">{page.laps}</td>\n</tr>\n'
            f'<tr>\n<th>ペース</th>\n<td class="race_lap_cell">{" - ".join(pace)}'
            "&nbsp;(35.0-35.0)</td>\n</tr>\n"
        )
    laps = f'<table class="result_table_02">\n<caption>ラップタイム</caption>\n{rows}</table>\n'
    return corners + laps


# ---------------------------------------------------------------- the five pages

# A flat turf race: 10 runners, every margin form (n, a/b, n.a/b, ハナ, クビ, アタマ, 大差).
PLAIN = ResultPage(
    key="202608050611",
    race_date=date(2026, 11, 22),
    meeting="5回京都6日目",
    race_name="第9回シンセティックマイルS(GI)",
    course="芝右 外1600m",
    weather="晴",
    going_label="芝",
    going="良",
    post_time="15:40",
    class_text="3歳以上オープン",
    conditions="(国際)(指)(定量)",
    table="""
        1  7 8  牡4 58   05901 1:33.0 -     2-2   33.9 3.4   1  498(+4)  01201
        2  3 3  牡5 58   05902 1:33.2 1.1/4 5-5   33.8 5.1   2  476(-2)  01202
        3  8 10 牝4 56   05903 1:33.2 ハナ  8-7   33.5 12.6  5  452(0)   01203
        4  1 1  牡3 57   05904 1:33.3 クビ  1-1   34.6 7.8   3  510(+6)  01204
        5  6 6  セ6 58   05905 1:33.3 アタマ 3-3  34.3 21.4  7  488(-8)  01205
        6  8 9  牡4 58   05906 1:33.4 3/4   10-10 33.4 9.9   4  502(+2)  01206
        7  2 2  牝5 56   05907 1:33.7 2     6-6   34.4 45.0  9  464(-4)  01207
        8  5 5  牡7 58   05908 1:33.8 1/2   4-4   34.7 15.2  6  520(+10) 01208
        9  4 4  牡3 57.5 05909 1:34.1 1.3/4 9-9   34.5 33.3  8  480(-6)  01209
        10 7 7  牡6 58   05910 1:36.0 大差  7-8   36.9 102.7 10 494(0)   01210
    """,
    pays="""
        単勝 8 340
        複勝 8 150 / 3 180 / 10 310
        枠連 3-7 820
        馬連 3-8 910
        ワイド 3-8 380 / 8-10 720 / 3-10 1,040
        馬単 8-3 1,560
        三連複 3-8-10 4,280
        三連単 8-3-10 19,870
    """,
    laps="12.3 - 10.9 - 11.4 - 11.8 - 11.9 - 11.5 - 11.2 - 12.0",
    horse_stem="20221051",
    name_stem="ミヤコ",
)

# A dead heat for 3rd on the straight 1000 m (no corners: no passing order): four places paid,
# five quinella-place pairs, two trios and two tierces.
DEAD_HEAT = ResultPage(
    key="202604020810",
    race_date=date(2026, 8, 30),
    meeting="2回新潟8日目",
    race_name="シンセティック直線S(L)",
    course="芝直線1000m",
    weather="雨",
    going_label="芝",
    going="不良",
    post_time="15:01",
    class_text="3歳以上オープン",
    conditions="(国際)(特指)(ハンデ)",
    table="""
        1 5 5 牝4 54   05901 0:55.8 -     - 32.1 4.2  2 460(+2)  01202
        2 2 2 牡5 56   05903 0:55.9 1/2   - 32.0 3.1  1 488(0)   01204
        3 7 7 牡6 57   05905 0:56.1 1     - 32.2 11.8 5 502(-4)  01206
        3 8 9 牝5 53   05907 0:56.1 同着  - 32.3 25.6 7 444(+6)  01208
        5 1 1 セ7 55   05909 0:56.1 ハナ  - 32.4 8.7  3 476(-2)  01210
        6 4 4 牡4 56   05902 0:56.2 クビ  - 32.4 9.5  4 490(+8)  01201
        7 8 8 牝3 51   05904 0:56.4 1.1/2 - 32.7 18.3 6 432(0)   01203
        8 3 3 牡8 57.5 05906 0:56.9 3     - 33.1 51.2 8 510(+12) 01205
        9 6 6 牡5 55   05908 0:58.5 大    - 34.6 88.9 9 484(-10) 01207
    """,
    pays="""
        単勝 5 420
        複勝 5 170 / 2 130 / 7 290 / 9 480
        枠連 2-5 690
        馬連 2-5 720
        ワイド 2-5 300 / 5-7 840 / 5-9 1,350 / 2-7 610 / 2-9 1,020
        馬単 5-2 1,480
        三連複 2-5-7 2,350 / 2-5-9 3,980
        三連単 5-2-7 9,870 / 5-2-9 16,540
    """,
    laps="11.9 - 10.2 - 10.8 - 11.1 - 11.8",
    horse_stem="20211042",
    name_stem="ナギサ",
)

# A debut race: one runner withdrawn (取), a dead heat for 2nd (two brackets, quinellas,
# forecasts and tierces), two body weights printed without a change; older person links.
SCRATCHED = ResultPage(
    key="202605040205",
    race_date=date(2026, 10, 11),
    meeting="4回東京2日目",
    race_name="2歳新馬",
    course="芝左1800m",
    weather="曇",
    going_label="芝",
    going="稍重",
    post_time="12:10",
    class_text="2歳新馬",
    conditions="(混)[指](馬齢)",
    table="""
        1  4 4  牡2 55 05902 1:49.6 -      5-5-4-3   34.1 6.2   3  470      01201
        2  8 10 牝2 55 05904 1:49.8 1.1/4  2-2-2-2   34.5 2.8   1  448(-2)  01203
        2  1 1  牡2 55 05906 1:49.8 同着   8-8-7-6   33.9 4.5   2  482(+4)  01205
        4  6 7  牡2 55 05908 1:49.9 3/4    3-3-3-4   34.4 9.6   4  466(+2)  01207
        5  8 11 牝2 55 05910 1:49.9 クビ   10-10-9-8 34.0 14.7  5  430(-6)  01209
        6  2 2  牡2 55 05901 1:50.3 2.1/2  1-1-1-1   35.2 21.0  6  494(0)   01202
        7  7 9  牝2 55 05903 1:50.3 アタマ 6-6-6-7   34.7 37.5  7  452      01204
        8  3 3  牡2 55 05905 1:50.4 1/2    9-9-10-10 34.3 58.1  9  476(-4)  01206
        9  7 8  牡2 55 05907 1:51.2 5      4-4-5-5   35.6 44.9  8  500(+2)  01208
        10 5 5  牝2 55 05909 1:53.0 大     7-7-8-9   37.0 123.4 10 438(0)   01210
        取 6 6  牡2 55 05911 -      -      -         -    ---   -  計不     01211
    """,
    pays="""
        単勝 4 620
        複勝 4 190 / 10 120 / 1 140
        枠連 4-8 530 / 1-4 760
        馬連 4-10 640 / 1-4 980
        ワイド 4-10 280 / 1-4 350 / 1-10 210
        馬単 4-10 1,450 / 4-1 2,130
        三連複 1-4-10 1,170
        三連単 4-10-1 4,560 / 4-1-10 6,230
    """,
    laps="12.8 - 11.5 - 12.0 - 12.4 - 12.6 - 12.3 - 11.6 - 11.3 - 13.1",
    horse_stem="20241063",
    name_stem="コガネ",
    old_links=True,
)

# Heavy-ish dirt: 12 declared, a runner demoted to 4th (4(降)), a disqualified runner (失, printed
# time and margin to the horse it followed) and an excluded runner (除, weighed, no odds).
DEMOTED = ResultPage(
    key="202609030207",
    race_date=date(2026, 6, 14),
    meeting="3回阪神2日目",
    race_name="3歳以上1勝クラス",
    course="ダ右1800m",
    weather="晴",
    going_label="ダート",
    going="重",
    post_time="13:20",
    class_text="3歳以上1勝クラス",
    conditions="(混)(定量)",
    table="""
        1     5 6  牡4 57 05901 1:52.3 -      2-2-2-2     38.1 2.6  1  486(+2) 01201
        2     1 1  牝3 54 05902 1:52.6 2      4-4-3-3     38.0 7.4  4  452(-4) 01202
        3     8 11 牡5 57 05903 1:52.6 クビ   1-1-1-1     38.6 5.9  3  510(0)  01203
        4(降) 7 9  牡3 56 05904 1:52.7 3/4    6-6-5-4     38.2 4.1  2  478(+6) 01204
        5     4 4  セ5 57 05905 1:53.0 1.1/2  3-3-4-5     38.5 12.3 5  498(-2) 01205
        6     6 8  牝4 55 05906 1:53.0 アタマ 8-8-7-6     38.1 18.6 6  440(+4) 01206
        7     2 2  牡4 57 05907 1:53.2 1      5-5-6-7     38.7 25.2 7  502(-6) 01207
        8     7 10 牡6 57 05908 1:53.2 ハナ   10-10-9-8   38.0 33.7 8  490(0)  01208
        9     5 5  牝3 54 05909 1:53.6 2.1/2  7-7-8-9     39.0 47.1 9  446(+2) 01209
        10    6 7  牡5 57 05910 1:55.4 大     9-9-10-10   40.3 66.8 10 516(+8) 01210
        失    3 3  牡4 57 05911 1:56.8 大     11-11-11-11 40.1 88.2 11 480(-2) 01211
        除    8 12 牝3 54 05912 -      -      -           -    ---  -  496(+4) 01212
    """,
    pays="""
        単勝 6 260
        複勝 6 130 / 1 210 / 11 190
        枠連 1-5 1,030
        馬連 1-6 1,120
        ワイド 1-6 430 / 6-11 390 / 1-11 870
        馬単 6-1 1,690
        三連複 1-6-11 2,740
        三連単 6-1-11 9,310
    """,
    laps="12.5 - 11.4 - 13.0 - 12.8 - 12.6 - 12.4 - 12.3 - 12.2 - 13.1",
    horse_stem="20231074",
    name_stem="サザナミ",
)

# A jump race over both courses (外-内): two runners pulled up (中, passing order and odds
# printed, no time); 上り is printed but means something else in a jump; no lap times.
JUMP = ResultPage(
    key="202603020204",
    race_date=date(2026, 7, 5),
    meeting="2回福島2日目",
    race_name="障害3歳以上未勝利",
    course="障芝 外-内2750m",
    weather="晴",
    going_label="芝",
    going="良",
    post_time="11:05",
    class_text="障害3歳以上未勝利",
    conditions="(混)(別定)",
    table="""
        1  7 7 牡5 60 05901 3:02.5 -     2-2-1-1 13.4 2.2  1 478(-2)  01201
        2  2 2 牡4 60 05902 3:03.2 4     4-4-3-2 13.5 6.8  3 492(+4)  01202
        3  5 5 セ6 60 05903 3:03.6 2.1/2 1-1-2-3 13.6 4.0  2 510(0)   01203
        4  1 1 牡4 60 05904 3:03.9 1.3/4 6-5-5-4 13.6 12.5 5 466(-6)  01204
        5  8 8 牡7 60 05905 3:04.9 6     5-6-6-5 13.8 18.9 6 500(+2)  01205
        6  3 3 牝4 58 05906 3:04.9 クビ  8-7-7-6 13.8 31.4 7 448(+10) 01206
        7  6 6 牡5 60 05907 3:06.6 大差  7-8-8-7 14.1 59.6 9 484(-4)  01207
        中 4 4 牡6 60 05908 -      -     3-3-4   -    9.8  4 470(+2)  01208
        中 8 9 牡4 60 05909 -      -     9-9     -    45.1 8 488(0)   01209
    """,
    pays="""
        単勝 7 220
        複勝 7 110 / 2 200 / 5 140
        枠連 2-7 780
        馬連 2-7 820
        ワイド 2-7 330 / 5-7 190 / 2-5 450
        馬単 7-2 1,230
        三連複 2-5-7 760
        三連単 7-2-5 3,890
    """,
    laps=None,
    horse_stem="20221085",
    name_stem="ヤマビコ",
)
