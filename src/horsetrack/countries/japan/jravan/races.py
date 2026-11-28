"""JV-Data RA (race detail) records -> the jravan store's `races` and `meetings` (#364).

`races_from_records` maps `records.decode`d RA arrays to `races` rows in the vocabulary the jra
store writes (`countries/japan/jra/netkeiba/parse_results.py`); `meetings_from_races` derives the
meetings. `build_races` reads `raw/jravan/stored/RACE/RA/<YYYY>.jvd` one year at a time, keeps one
version per race (`records.latest_versions` on the layout's key items) and the JRA venues (01-10),
and writes `processed/jravan/races.parquet` and `meetings.parquet` in the store's column order,
types and sort keys. The NAR and overseas races (#377) go to `processed/jravan/form_races.parquet`
in the `races` schema, under their own `jurisdiction` and form ids (`keys.form_race_id`, `venue`
the JV-Data code), with no meetings: horse form only, read by `horse_runs` and by no betting
reader. A venue without a jurisdiction is written nowhere. The code values each mapping uses are
the constants below (the code-table text stays in the licensed workbook); a code a mapping lacks
becomes null and is counted per code in the report (JRA races only). A blank or all-zero code is
JV-Data's "unset": a null value (its `*_raw` column too), counted as unmapped only on a race that
was run. See docs/specs/jravan.md.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyarrow as pa
import typer

from horsetrack.core import paths
from horsetrack.core.store.build import _season_of, arrow_schema
from horsetrack.core.store.schemas import TABLES
from horsetrack.countries.japan.jravan import keys, records, spec

SOURCE = records.SOURCE
JURISDICTION = "jra"
FORM_RACES = "form_races"  # the NAR and overseas races, horse form only (#377)
SEASON_START_MONTH = 1  # the jra build's calendar-year seasons

# RA items (the `records.decode` keys)
KUBUN = "データ区分"
CREATED = "データ作成年月日"
YEAR, MMDD, VENUE, KAI, NICHI, RACE_NO = (
    "開催年", "開催月日", "競馬場コード", "開催回[第N回]", "開催日目[N日目]", "レース番号",
)  # fmt: skip
RACE_NAME = "競走名本題"
GRADE = "グレードコード"
WEIGHT_TYPE = "重量種別コード"
CONDITION = "競走条件コード最若年条件"  # the condition of the youngest age allowed in
DISTANCE = "距離"
TRACK = "トラックコード"
COURSE = "コース区分"
PRIZE = "本賞金"  # in units of 100 yen; the first occurrence is the winner's
PRIZE_UNIT_YEN = 100
POST_TIME = "発走時刻"
FIELD_SIZE = "出走頭数"
TURF_GOING = "芝馬場状態コード"
DIRT_GOING = "ダート馬場状態コード"
LAPS = "ラップタイム"  # tenths of a second per lap
ITEMS = (
    KUBUN, CREATED, YEAR, MMDD, VENUE, KAI, NICHI, RACE_NO, RACE_NAME, GRADE, WEIGHT_TYPE,
    CONDITION, DISTANCE, TRACK, COURSE, PRIZE, POST_TIME, FIELD_SIZE, TURF_GOING, DIRT_GOING, LAPS,
)  # fmt: skip

KUBUN_DELETED = "0"
KUBUN_ABANDONED = "9"

CONDITION_CLASSES = {
    "701": "DEBUT", "702": "MAIDEN", "703": "MAIDEN", "005": "C1", "010": "C2", "016": "C3",
    "999": "OP",
}  # fmt: skip
OPEN_CLASS = "OP"
# graded and listed races (jump grades F-H as their flat code); blank, D and E keep the class
GRADES = {"A": "G1", "F": "G1", "B": "G2", "G": "G2", "C": "G3", "H": "G3", "L": "L"}
WEIGHT_TYPES = {"1": "handicap", "2": "allowance", "3": "age", "4": "set"}
GOINGS = {"1": "GOOD", "2": "GOOD TO YIELDING", "3": "SOFT", "4": "HEAVY"}
TURF, DIRT, JUMP = "turf", "dirt", "jump"
SURFACES = {
    **{str(code): TURF for code in range(10, 23)},
    **{str(code): DIRT for code in (23, 24, 25, 26, 29)},
    **{str(code): JUMP for code in range(51, 60)},
}
INNER_TRACKS = frozenset({"15", "21", "25", "58"})  # the inner course only
OUTER_TRACKS = frozenset({"12", "16", "18", "22", "26", "55", "59"})  # the outer course only
RAIL_SETTINGS = frozenset({"A", "B", "C", "D"})

# store columns whose unmapped codes the report counts
UNMAPPED_COLUMNS = ("race_class", "weight_type", "surface", "going")
YEAR_COUNTS = (
    "records", "versions_dropped", "deletions", "races_written", "non_jra_nar",
    "non_jra_overseas", "non_jra_unknown", "unmapped", "form_without_id",
)  # fmt: skip


def _unset(code: str | None) -> bool:
    return code is None or set(code) == {"0"}


def _laps(raw: np.ndarray) -> list[list[float]]:
    """Cumulative leader times per record: blank and zero laps dropped."""
    laps = records.scaled(raw, 1)
    if laps.ndim == 1:
        laps = laps[:, None]
    out = []
    for row in laps:
        total, splits = 0.0, []
        for lap in row:
            if np.isfinite(lap) and lap > 0:
                total = round(total + float(lap), 1)
                splits.append(total)
        out.append(splits)
    return out


def _integers(raw: np.ndarray) -> list[int | None]:
    return [None if np.isnan(v) else int(v) for v in records.integer(raw)]


def _map(
    cols: Mapping[str, np.ndarray], *, form: bool = False
) -> tuple[pd.DataFrame, dict[str, collections.Counter]]:
    """The `races` rows of `cols` and, per store column, the unmapped codes; with `form`, the NAR
    and overseas races under their form ids and own jurisdiction."""
    text = {name: records.text(cols[name]) for name in ITEMS if name not in (PRIZE, LAPS)}
    n = len(text[YEAR])
    unmapped = {column: collections.Counter() for column in UNMAPPED_COLUMNS}
    to_race_id, to_meeting_id = (
        (keys.form_race_id, keys.form_meeting_id) if form else (keys.race_id, keys.meeting_id)
    )
    race_ids = [
        to_race_id(y, md, v, r)
        for y, md, v, r in zip(text[YEAR], text[MMDD], text[VENUE], text[RACE_NO], strict=True)
    ]
    parts = [rid.split(":") if rid else [None] * 4 for rid in race_ids]
    race_dates = [date.fromisoformat(p[1]) if p[1] else None for p in parts]
    prize = records.integer(cols[PRIZE])
    if prize.ndim > 1:
        prize = prize[:, 0]
    rows: dict[str, list] = {column: [] for column in TABLES["races"].columns}
    for i in range(n):
        run = text[KUBUN][i] != KUBUN_ABANDONED

        def count(column: str, code: str | None, run: bool = run) -> None:
            if run or not _unset(code):
                unmapped[column][code or ""] += 1  # a blank code counts as ''

        condition = text[CONDITION][i]
        race_class = CONDITION_CLASSES.get(condition) if condition else None
        if race_class in (None, OPEN_CLASS):
            race_class = GRADES.get(text[GRADE][i] or "", race_class)
        if race_class is None:
            count("race_class", condition)
        weight_type = WEIGHT_TYPES.get(text[WEIGHT_TYPE][i] or "")
        if weight_type is None:
            count("weight_type", text[WEIGHT_TYPE][i])
        track = text[TRACK][i]
        surface = SURFACES.get(track or "")
        if surface is None:
            count("surface", track)
        going_code = None
        if surface is not None:
            going_code = text[DIRT_GOING if surface == DIRT else TURF_GOING][i]
        going_raw = None if _unset(going_code) else going_code
        going = GOINGS.get(going_raw or "")
        if going is None and surface is not None:  # no surface: no going to read
            count("going", going_code)
        course = text[COURSE][i]
        day = race_dates[i]
        rows["jurisdiction"].append(parts[i][0] if form else JURISDICTION)
        rows["race_id"].append(race_ids[i])
        rows["meeting_id"].append(to_meeting_id(text[YEAR][i], text[MMDD][i], text[VENUE][i]))
        rows["race_date"].append(day)
        rows["venue"].append(parts[i][2])
        rows["race_no"].append(int(parts[i][3]) if parts[i][3] else None)
        rows["season"].append(_season_of(day, SEASON_START_MONTH) if day else None)
        rows["race_index"].append(None)
        rows["race_name"].append(text[RACE_NAME][i])
        rows["race_class"].append(race_class)
        rows["rating_hi"].append(None)
        rows["rating_lo"].append(None)
        rows["surface"].append(surface)
        rows["course_config"].append(
            "inner" if track in INNER_TRACKS else "outer" if track in OUTER_TRACKS else None
        )
        rows["going"].append(going)
        rows["prize"].append(None if np.isnan(prize[i]) else int(prize[i]) * PRIZE_UNIT_YEN)
        rows["post_time_local"].append(keys.post_time_local(text[POST_TIME][i]))
        rows["finish_time_s"].append(None)
        rows["status"].append("run" if run else "abandoned")
        rows["weight_type"].append(weight_type)
        rows["race_class_raw"].append(None if _unset(condition) else condition)
        rows["going_raw"].append(going_raw)
        rows["source_race_key"].append(
            keys.source_race_key(
                text[YEAR][i], text[VENUE][i], text[KAI][i], text[NICHI][i], text[RACE_NO][i]
            )
        )
        rows["rail_setting"].append(course if surface == TURF and course in RAIL_SETTINGS else None)
        rows["result_published_utc"].append(None)
        rows["result_unpublished_utc"].append(None)
    rows["distance_m"] = _integers(cols[DISTANCE])
    rows["field_size"] = _integers(cols[FIELD_SIZE])
    rows["split_times_s"] = _laps(cols[LAPS])
    return pd.DataFrame(rows, columns=list(TABLES["races"].columns)), unmapped


def races_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One `races` row per RA record of `cols` (JRA venues, one version per race), with the
    columns of `schemas.TABLES['races']` in order."""
    return _map(cols)[0]


def meetings_from_races(races: pd.DataFrame) -> pd.DataFrame:
    """One `meetings` row per `meeting_id` of `races`: `run` when any of its races ran, else
    `abandoned`; `day_night` and `surface_mix` null."""
    known = races[races["meeting_id"].notna()]
    first = known.drop_duplicates("meeting_id")[["meeting_id", "race_date", "venue", "season"]]
    ran = known["status"].eq("run").groupby(known["meeting_id"]).any()
    meetings = first.assign(
        jurisdiction=JURISDICTION,
        day_night=None,
        surface_mix=None,
        status=first["meeting_id"].map(ran).map({True: "run", False: "abandoned"}),
    )
    return meetings[list(TABLES["meetings"].columns)].reset_index(drop=True)


def _sorted(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    order = list(TABLES[table].sort_by)
    return frame.sort_values(order, kind="stable").reset_index(drop=True)


def _write(frame: pd.DataFrame, table: str, out: Path, name: str | None = None) -> None:
    """`frame` in the schema of store table `table`, to `<out>/<name or table>.parquet`."""
    arrow = pa.Table.from_pandas(frame, schema=arrow_schema(table), preserve_index=False)
    records.write_parquet(arrow.replace_schema_metadata(None), out / f"{name or table}.parquet")


def _unique(races: pd.DataFrame) -> None:
    duplicated = races["race_id"].duplicated()
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} RA races share a race_id with another race")


def _read_year(path: Path, layout: spec.RecordSpec) -> dict[str, np.ndarray]:
    """The RA items the mapping uses, for every record of one year's file."""
    chunks = [{name: part[name] for name in ITEMS} for part in records.iter_file(path, layout)]
    if not chunks:
        return {}
    return {name: np.concatenate([c[name] for c in chunks]) for name in ITEMS}


def _year_files(folder: Path, years: list[int] | None) -> dict[int, Path]:
    found = {int(p.stem): p for p in folder.glob("*.jvd") if p.stem.isdigit()}
    if years is None:
        return dict(sorted(found.items()))
    missing = [y for y in years if y not in found]
    if missing:
        raise FileNotFoundError(f"no year file for {missing} under {folder}")
    return {y: found[y] for y in sorted(set(years))}


def build_races(
    *, years: list[int] | None = None, raw: Path | None = None, out: Path | None = None
) -> records.BuildReport:
    """Build `races.parquet` and `meetings.parquet` (JRA) and `form_races.parquet` (NAR and
    overseas, form only) from the RA files of `years` (default: every year file), one year at a
    time; the report holds counts and codes only, of the JRA races."""
    raw = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    out = Path(out) if out is not None else paths.processed_dir(SOURCE)
    layout = spec.record_spec("RA")
    key = [item.name for item in layout.items if item.is_key]
    by_year: dict[int, dict[str, int]] = {}
    unmapped = {column: collections.Counter() for column in UNMAPPED_COLUMNS}
    frames, form_frames = [], []
    for year, path in _year_files(raw / "stored" / "RACE" / "RA", years).items():
        cols = _read_year(path, layout)
        counts = dict.fromkeys(YEAR_COUNTS, 0)
        by_year[year] = counts
        if not cols:
            continue
        index = pd.DataFrame({name: records.text(cols[name]) for name in (*key, CREATED, KUBUN)})
        counts["records"] = len(index)
        races_kept = index.drop_duplicates(key).shape[0]
        kept = records.latest_versions(index, key)
        counts["versions_dropped"] = len(index) - races_kept
        counts["deletions"] = races_kept - len(kept)
        where = collections.Counter(keys.jurisdiction(v) for v in kept[VENUE])
        counts["non_jra_nar"] = where["nar"]
        counts["non_jra_overseas"] = where["overseas"]
        counts["non_jra_unknown"] = where[None]
        rows = np.flatnonzero(index.index.isin(kept.index))
        where_kept = [keys.jurisdiction(v) for v in index[VENUE].to_numpy()[rows]]
        jra = rows[[w == JURISDICTION for w in where_kept]]
        frame, codes = _map({name: array[jra] for name, array in cols.items()})
        frame = frame[frame["race_id"].notna()]
        counts["races_written"] = len(frame)
        counts["unmapped"] = sum(sum(c.values()) for c in codes.values())
        for column, counter in codes.items():
            unmapped[column].update(counter)
        frames.append(frame)
        form = rows[[w in keys.FORM_JURISDICTIONS for w in where_kept]]
        if len(form):
            form_frame = _map({name: array[form] for name, array in cols.items()}, form=True)[0]
            form_frame = form_frame[form_frame["race_id"].notna()]
            counts["form_without_id"] = len(form) - len(form_frame)  # bad date or race number
            form_frames.append(form_frame)
    empty = arrow_schema("races").empty_table().to_pandas()  # typed, so an empty file writes
    races = pd.concat(frames, ignore_index=True) if frames else empty
    form_races = pd.concat(form_frames, ignore_index=True) if form_frames else empty
    _unique(races)
    _unique(form_races)
    _write(_sorted(races, "races"), "races", out)
    _write(_sorted(meetings_from_races(races), "meetings"), "meetings", out)
    _write(_sorted(form_races, "races"), "races", out, FORM_RACES)
    totals = {k: sum(c[k] for c in by_year.values()) for k in YEAR_COUNTS}
    return records.BuildReport(
        name="races",
        counts={"years": len(by_year), **totals},
        by_year=by_year,
        unmapped={column: dict(sorted(c.items())) for column, c in unmapped.items()},
        failures=[],
    )


def races_command(
    year: Annotated[
        list[int] | None, typer.Option("--year", help="Only this year (repeatable).")
    ] = None,
) -> None:
    """Build the jravan store's races and meetings from the JV-Data RA records; the report
    (counts and codes only) goes to <reports dir>/jravan-races.json."""
    report = build_races(years=year or None)
    target = records.write_build_report(report)
    c = report.counts
    typer.echo(
        f"RA: {c['years']} years, {c['records']} records, {c['races_written']} races written, "
        f"{c['versions_dropped']} versions dropped, {c['deletions']} deletions, "
        f"{c['non_jra_nar'] + c['non_jra_overseas'] + c['non_jra_unknown']} non-JRA races, "
        f"{c['form_without_id']} NAR/overseas races without a form id"
    )
    for column, codes in report.unmapped.items():
        if codes:
            listed = ", ".join(f"{code} x{n}" for code, n in codes.items())
            typer.echo(f"unmapped {column}: {listed}")
    typer.echo(f"report: {target}")
