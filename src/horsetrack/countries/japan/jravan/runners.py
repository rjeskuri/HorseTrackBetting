"""JV-Data SE (runner per race) records -> the jravan store's `runners`, and each race's winner's
time on `races` (#365).

`runners_from_records` maps `records.decode`d SE arrays to `runners` rows in the vocabulary the jra
store writes (`countries/japan/jra/netkeiba/parse_results.py`: its placing codes, scratched codes,
margin names, beaten-lengths sum and pound conversion). `build_runners` reads
`raw/jravan/stored/RACE/SE/<YYYY>.jvd` one year at a time, keeps one version per runner
(`records.latest_versions` on the layout's key items) and the JRA venues (01-10), writes
`processed/jravan/runners.parquet` in the store's column order, types and sort key, then sets
`finish_time_s` on `races.parquet` (from `races.build_races`) to each race's winner's time; both
files keep one row group per race day (#502). The
code values each mapping uses are the constants below (the code-table text stays in the licensed
workbook); a code a mapping lacks becomes null and is counted per code in the report. A blank or
all-zero margin code is JV-Data's "unset": null and never counted. See docs/specs/jravan.md.
"""

from __future__ import annotations

import collections
import itertools
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.store.build import _season_of, arrow_schema
from horsetrack.core.store.schemas import TABLES
from horsetrack.countries.japan.jra.netkeiba.parse_results import (
    _PLACING_CODES,
    _SCRATCHED_CODES,
    _beaten_lengths,
    _lb,
)
from horsetrack.countries.japan.jravan import keys, races, records, spec

SOURCE = records.SOURCE
JURISDICTION = races.JURISDICTION
RECORD_TYPE = "SE"

# SE items (the `records.decode` keys)
KUBUN, CREATED = races.KUBUN, races.CREATED
YEAR, MMDD, VENUE, KAI, NICHI, RACE_NO = (
    races.YEAR, races.MMDD, races.VENUE, races.KAI, races.NICHI, races.RACE_NO,
)  # fmt: skip
HORSE_NO = "馬番"
HORSE_ID = "血統登録番号"  # the top-level item, not the one inside 1着馬(相手馬)情報
BRACKET = "枠番"
HORSE_NAME = "馬名"
SEX = "性別コード"
AGE = "馬齢"
JOCKEY, JOCKEY_NAME = "騎手コード", "騎手名略称"
TRAINER, TRAINER_NAME = "調教師コード", "調教師名略称"
CARRIED = "負担重量"  # tenths of a kg
BODY_WT = "馬体重"  # kg
CHANGE_SIGN, CHANGE = "増減符号", "増減差"  # kg
ABNORMAL = "異常区分コード"
PLACE = "確定着順"
DEAD_HEAT = "同着区分"
TIME = "走破タイム"  # M SS S: minutes, then seconds to a tenth
MARGIN = "着差コード"
CORNERS = ("1コーナーでの順位", "2コーナーでの順位", "3コーナーでの順位", "4コーナーでの順位")
WIN_ODDS = "単勝オッズ"  # tenths
FAV_RANK = "単勝人気順"
LAST_3F = "後3ハロンタイム"  # tenths of a second
ITEMS = (
    KUBUN, CREATED, YEAR, MMDD, VENUE, KAI, NICHI, RACE_NO, HORSE_NO, HORSE_ID, BRACKET,
    HORSE_NAME, SEX, AGE, JOCKEY, JOCKEY_NAME, TRAINER, TRAINER_NAME, CARRIED, BODY_WT,
    CHANGE_SIGN, CHANGE, ABNORMAL, PLACE, DEAD_HEAT, TIME, MARGIN, *CORNERS, WIN_ODDS, FAV_RANK,
    LAST_3F,
)  # fmt: skip

# 異常区分コード: a plain place, a remount (落馬再騎乗: it finished, so a plain place), a demotion
# (the official place stays, marked as netkeiba marks it), and the codes without a place, as
# netkeiba prints them (their placing codes are `parse_results._PLACING_CODES`)
ABNORMAL_PLACED = "0"
ABNORMAL_REMOUNTED = "6"
ABNORMAL_DEMOTED = "7"
ABNORMAL_MARKS = {"1": "取", "2": "除", "3": "除", "4": "中", "5": "失"}
DEMOTED_MARK = "(降)"
DEAD_HEAT_SET, DEAD_HEAT_UNSET = "1", "0"
DEAD_HEAT_CODE = "DH"
SEXES = {"1": "H", "2": "M", "3": "G"}
NEGATIVE = "-"
NOT_WEIGHED = 999  # 馬体重 and 増減差
NO_LAST_3F = 999  # 後3ハロンタイム with no time (99.9 s)
_FRACTIONS = {"14": "1/4", "12": "1/2", "34": "3/4"}
# 着差コード -> the netkeiba text of the same margin (`parse_results.margin_lengths` reads it)
MARGIN_TEXT = {
    "H": "ハナ", "A": "アタマ", "K": "クビ", "D": "同着", "T": "大差", "Z": "大差",
    **_FRACTIONS,
    **{str(n): str(n) for n in range(1, 10)},
    **{f"{n}{code}": f"{n}.{text}" for n in range(1, 10) for code, text in _FRACTIONS.items()},
}  # fmt: skip

# the fields whose unmapped codes the report counts (異常区分コード, 性別コード, 同着区分,
# 着差コード)
PLACING, SEX_FIELD, DEAD_HEAT_FIELD, MARGIN_FIELD = "placing", "sex", "dead_heat", "margin"
UNMAPPED_FIELDS = (PLACING, SEX_FIELD, DEAD_HEAT_FIELD, MARGIN_FIELD)
YEAR_COUNTS = (
    "records", "versions_dropped", "deletions", "runners_written", "non_jra_nar",
    "non_jra_overseas", "non_jra_unknown", "unmapped",
)  # fmt: skip


def _unset(code: str | None) -> bool:
    return code is None or set(code) == {"0"}


def _values(numbers: np.ndarray, null: tuple[float, ...] = ()) -> list:
    """Floats as Python ints, NaN and the `null` values as None."""
    return [None if np.isnan(v) or v in null else int(v) for v in numbers]


def _tenths(raw: np.ndarray, null: tuple[float, ...] = (0,)) -> list:
    """Tenths as floats; NaN and the `null` raw values as None."""
    numbers = records.integer(raw)
    return [None if np.isnan(v) or v in null else round(v / 10, 1) for v in numbers]


def _race_time(raw: np.ndarray) -> list:
    """走破タイム `MSSS` (1345 = 1:34.5) in seconds; 0000 and non-digits None."""
    out = []
    for v in records.integer(raw):
        if np.isnan(v) or v == 0:
            out.append(None)
        else:
            minutes, tenths = divmod(int(v), 1000)
            out.append(round(minutes * 60 + tenths / 10, 1))
    return out


def _lengths(race_ids: list, finish_pos: list, horse_no: list, margins: list) -> list[float | None]:
    """Beaten lengths per runner, summed down each race's placed order (finish_pos, then horse
    number) by `parse_results._beaten_lengths`; None for a runner without a place."""
    out: list[float | None] = [None] * len(race_ids)
    placed = [i for i, pos in enumerate(finish_pos) if pos is not None and race_ids[i]]
    placed.sort(key=lambda i: (race_ids[i], finish_pos[i], horse_no[i] is None, horse_no[i] or 0))
    for _, group in itertools.groupby(placed, key=race_ids.__getitem__):
        members = list(group)
        field = [{"finish_pos": finish_pos[i], "lbw_raw": MARGIN_TEXT.get(margins[i] or "")}
                 for i in members]  # fmt: skip
        _beaten_lengths(field)
        for i, runner in zip(members, field, strict=True):
            out[i] = runner["lbw_lengths"]
    return out


def _map(cols: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, collections.Counter]]:
    """The `runners` rows of `cols` and, per SE field, the unmapped codes."""
    text = {
        name: records.text(cols[name])
        for name in (KUBUN, YEAR, MMDD, VENUE, RACE_NO, HORSE_ID, HORSE_NAME, SEX, JOCKEY,
                     JOCKEY_NAME, TRAINER, TRAINER_NAME, CHANGE_SIGN, ABNORMAL, DEAD_HEAT, MARGIN)
    }  # fmt: skip
    n = len(text[YEAR])
    unmapped = {field: collections.Counter() for field in UNMAPPED_FIELDS}
    race_keys = list(zip(text[YEAR], text[MMDD], text[VENUE], text[RACE_NO], strict=True))
    by_key = {}
    for y, md, v, r in set(race_keys):
        rid = keys.race_id(y, md, v, r)
        day = date.fromisoformat(rid.split(":")[1]) if rid else None
        by_key[(y, md, v, r)] = (
            rid,
            keys.meeting_id(y, md, v),
            day,
            rid.split(":")[2] if rid else None,
            int(rid.split(":")[3]) if rid else None,
            _season_of(day, races.SEASON_START_MONTH) if day else None,
        )
    race = [by_key[k] for k in race_keys]
    race_ids = [r[0] for r in race]

    horse_no = _values(records.integer(cols[HORSE_NO]))
    places = _values(records.integer(cols[PLACE]), null=(0,))
    body = _values(records.integer(cols[BODY_WT]), null=(0, NOT_WEIGHED))
    change = _values(records.integer(cols[CHANGE]), null=(NOT_WEIGHED,))
    carried = _tenths(cols[CARRIED])
    corners = np.stack([records.integer(cols[c]) for c in CORNERS], axis=1)

    rows: dict[str, list] = {column: [] for column in TABLES["runners"].columns}
    finish_pos: list[int | None] = []
    for i in range(n):
        abnormal, place = text[ABNORMAL][i], places[i]
        mark = ABNORMAL_MARKS.get(abnormal or "")
        placing_code = _PLACING_CODES[mark] if mark else None
        if mark is not None:
            place = None
            placing_raw = mark
        else:
            if abnormal not in (ABNORMAL_PLACED, ABNORMAL_REMOUNTED, ABNORMAL_DEMOTED):
                unmapped[PLACING][abnormal or ""] += 1
            placing_raw = None if place is None else str(place)
            if placing_raw is not None and abnormal == ABNORMAL_DEMOTED:
                placing_raw += DEMOTED_MARK
        dh = text[DEAD_HEAT][i]
        if dh not in (DEAD_HEAT_SET, DEAD_HEAT_UNSET):
            unmapped[DEAD_HEAT_FIELD][dh or ""] += 1
        dead_heat = dh == DEAD_HEAT_SET and place is not None
        if dead_heat:
            placing_code = DEAD_HEAT_CODE
        sex = SEXES.get(text[SEX][i] or "")
        if sex is None:
            unmapped[SEX_FIELD][text[SEX][i] or ""] += 1
        margin = text[MARGIN][i]
        if not _unset(margin) and margin not in MARGIN_TEXT:
            unmapped[MARGIN_FIELD][margin] += 1
        wt_change = change[i]
        if wt_change is not None and text[CHANGE_SIGN][i] == NEGATIVE:
            wt_change = -wt_change
        jockey, trainer = text[JOCKEY][i], text[TRAINER][i]
        rid, meeting, day, venue, race_no, season = race[i]
        finish_pos.append(place)
        rows["jurisdiction"].append(JURISDICTION)
        rows["race_id"].append(rid)
        rows["meeting_id"].append(meeting)
        rows["race_date"].append(day)
        rows["venue"].append(venue)
        rows["race_no"].append(race_no)
        rows["season"].append(season)
        rows["horse_id"].append(text[HORSE_ID][i])
        rows["horse_no"].append(horse_no[i])
        rows["horse_name"].append(text[HORSE_NAME][i])
        rows["brand_raw"].append(None)
        rows["jockey_key"].append(f"jockey:{jockey}" if jockey else None)
        rows["jockey_code"].append(jockey)
        rows["jockey_name"].append(text[JOCKEY_NAME][i])
        rows["trainer_key"].append(f"trainer:{trainer}" if trainer else None)
        rows["trainer_code"].append(trainer)
        rows["trainer_name"].append(text[TRAINER_NAME][i])
        rows["actual_wt_lb"].append(_lb(carried[i]))
        rows["declared_horse_wt_lb"].append(_lb(body[i]))
        rows["draw"].append(horse_no[i])  # JRA gates are numbered by saddle number
        rows["placing_raw"].append(placing_raw)
        rows["finish_pos"].append(place)
        rows["placing_code"].append(placing_code)
        rows["dead_heat"].append(dead_heat)
        rows["lbw_raw"].append(None if _unset(margin) else margin)
        rows["running_positions"].append([int(p) for p in corners[i] if p > 0])
        rows["is_scratched"].append(placing_code in _SCRATCHED_CODES)
        rows["rating"].append(None)
        rows["gear_raw"].append(None)
        rows["actual_wt_kg"].append(carried[i])
        rows["horse_wt_kg"].append(body[i])
        rows["horse_wt_change_kg"].append(wt_change)
        rows["sex"].append(sex)
    rows["lbw_lengths"] = _lengths(race_ids, finish_pos, horse_no, text[MARGIN].tolist())
    rows["finish_time_s"] = _race_time(cols[TIME])
    rows["win_odds"] = _tenths(cols[WIN_ODDS])
    rows["bracket"] = _values(records.integer(cols[BRACKET]), null=(0,))
    rows["last_3f_s"] = _tenths(cols[LAST_3F], null=(0, NO_LAST_3F))
    rows["win_fav_rank"] = _values(records.integer(cols[FAV_RANK]), null=(0,))
    rows["age"] = _values(records.integer(cols[AGE]), null=(0,))
    return pd.DataFrame(rows, columns=list(TABLES["runners"].columns)), unmapped


def runners_from_records(cols: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """One `runners` row per SE record of `cols` (JRA venues, one version per runner), with the
    columns of `schemas.TABLES['runners']` in order; beaten lengths are summed per race over the
    records of `cols`."""
    return _map(cols)[0]


def _read_year(path: Path, layout: spec.RecordSpec) -> dict[str, np.ndarray]:
    """The SE items the mapping uses, for every record of one year's file."""
    chunks = [{name: part[name] for name in ITEMS} for part in records.iter_file(path, layout)]
    if not chunks:
        return {}
    return {name: np.concatenate([c[name] for c in chunks]) for name in ITEMS}


def fill_winners_time(runners: pd.DataFrame, path: Path) -> int:
    """Set `finish_time_s` of the races file at `path` to the time of each race's `finish_pos` 1
    runner (the lower horse number on a dead heat; null for a race without one) and rewrite it
    whole, every other column unchanged. Returns the races with a time."""
    winners = runners[runners["finish_pos"].eq(1).fillna(False).astype(bool)]
    winners = winners.sort_values(["race_id", "horse_no"], kind="stable")
    first = winners.drop_duplicates("race_id").set_index("race_id")["finish_time_s"]
    table = pq.read_table(path).cast(arrow_schema("races"))
    times = [first.get(rid) for rid in table["race_id"].to_pylist()]
    times = [None if t is None or pd.isna(t) else float(t) for t in times]
    at = table.schema.get_field_index("finish_time_s")
    table = table.set_column(at, table.schema.field(at), pa.array(times, type=pa.float64()))
    records.write_parquet(table.replace_schema_metadata(None), path, by_day=True)
    return sum(t is not None for t in times)


def build_runners(
    *, years: list[int] | None = None, raw: Path | None = None, out: Path | None = None
) -> records.BuildReport:
    """Build `runners.parquet` from the SE files of `years` (default: every year file), one year
    at a time, then fill the winner's time on `<out>/races.parquet`, which `races.build_races`
    must have written first. The report holds counts and codes only."""
    raw = Path(raw) if raw is not None else paths.raw_dir(SOURCE)
    out = Path(out) if out is not None else paths.processed_dir(SOURCE)
    races_path = out / "races.parquet"
    if not races_path.is_file():
        raise FileNotFoundError(
            f"no races file at {races_path}: build the races first (horsetrack japan jravan races)"
        )
    layout = spec.record_spec(RECORD_TYPE)
    key = [item.name for item in layout.items if item.is_key]
    by_year: dict[int, dict[str, int]] = {}
    unmapped = {field: collections.Counter() for field in UNMAPPED_FIELDS}
    schema = arrow_schema("runners")
    sort_by = TABLES["runners"].sort_by
    tables: list[pa.Table] = []
    for year, path in races._year_files(raw / "stored" / "RACE" / RECORD_TYPE, years).items():
        cols = _read_year(path, layout)
        counts = dict.fromkeys(YEAR_COUNTS, 0)
        by_year[year] = counts
        if not cols:
            continue
        index = pd.DataFrame({name: records.text(cols[name]) for name in (*key, CREATED, KUBUN)})
        counts["records"] = len(index)
        runners_kept = index.drop_duplicates(key).shape[0]
        kept = records.latest_versions(index, key)
        counts["versions_dropped"] = len(index) - runners_kept
        counts["deletions"] = runners_kept - len(kept)
        where = collections.Counter(keys.jurisdiction(v) for v in kept[VENUE])
        counts["non_jra_nar"] = where["nar"]
        counts["non_jra_overseas"] = where["overseas"]
        counts["non_jra_unknown"] = where[None]
        rows = np.flatnonzero(index.index.isin(kept.index))
        jra = rows[[keys.jurisdiction(v) == JURISDICTION for v in index[VENUE].to_numpy()[rows]]]
        del index, kept
        frame, codes = _map({name: array[jra] for name, array in cols.items()})
        del cols
        frame = frame[frame["race_id"].notna()]
        counts["runners_written"] = len(frame)
        counts["unmapped"] = sum(sum(c.values()) for c in codes.values())
        for field, counter in codes.items():
            unmapped[field].update(counter)
        arrow = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        tables.append(arrow.replace_schema_metadata(None))
        del frame, arrow
    runners = pa.concat_tables(tables) if tables else schema.empty_table()
    del tables
    runners = runners.take(pc.sort_indices(runners, [(c, "ascending") for c in sort_by]))
    records.write_parquet(runners, out / "runners.parquet", by_day=True)
    timed = fill_winners_time(
        runners.select(["race_id", "horse_no", "finish_pos", "finish_time_s"]).to_pandas(),
        races_path,
    )
    totals = {k: sum(c[k] for c in by_year.values()) for k in YEAR_COUNTS}
    return records.BuildReport(
        name="runners",
        counts={"years": len(by_year), **totals, "races_timed": timed},
        by_year=by_year,
        unmapped={field: dict(sorted(c.items())) for field, c in unmapped.items()},
        failures=[],
    )


def runners_command(
    year: Annotated[
        list[int] | None, typer.Option("--year", help="Only this year (repeatable).")
    ] = None,
) -> None:
    """Build the jravan store's runners from the JV-Data SE records and fill each race's winner's
    time; the report (counts and codes only) goes to <reports dir>/jravan-runners.json."""
    try:
        report = build_runners(years=year or None)
        target = records.write_build_report(report)
    except (paths.DataDirNotSet, ValueError, LookupError, OSError) as exc:
        _die(f"jravan runners failed: {type(exc).__name__}: {exc}")
    c = report.counts
    typer.echo(
        f"SE: {c['years']} years, {c['records']} records, {c['runners_written']} runners written, "
        f"{c['versions_dropped']} versions dropped, {c['deletions']} deletions, "
        f"{c['non_jra_nar'] + c['non_jra_overseas'] + c['non_jra_unknown']} non-JRA runners; "
        f"{c['races_timed']} races with the winner's time"
    )
    for field, codes in report.unmapped.items():
        if codes:
            listed = ", ".join(f"{code} x{n}" for code, n in codes.items())
            typer.echo(f"unmapped {field}: {listed}")
    typer.echo(f"report: {target}")


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
