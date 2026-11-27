"""JV-Data RA and SE versions by データ区分: which ones the export holds (#401).

JV-Data publishes a race (RA) and each of its runners (SE) in versions told apart by データ区分
(`KUBUN_NAMES`): the entries and the card before the race (`PRE_RACE`), flash results on race day,
then the final result; each version is stamped with データ作成年月日. The builders keep the latest
version of each key (`records.latest_versions`), so the card as it stood before the race survives
only if the export holds a pre-race version. `census` counts what it holds; nothing is built.

- `record_versions` maps `records.decode`d RA or SE arrays to one row per record, never
  deduplicated: `race_id` (`keys.race_id`, null outside JRA), `data_kubun`, `created`
  (データ作成年月日) and `race_date` (開催年 + 開催月日, every venue) as dates, NaT when invalid;
  SE adds `horse_no` and `horse_id` (the top-level 血統登録番号).
- `census(*, raw=None, as_of)` reads `<raw>/stored/<DATASPEC>/RA|SE/*.jvd` of every DATASPEC
  folder (the files directly in the folder; folders, then files, in sorted order), never
  `stored_shift/`, `_incoming/` or `probe/`, streamed by `records.iter_lines`. Per record it keeps
  only the layout's key items (as bytes), データ区分, データ作成年月日, the race date and 開催年.
  Each record's two-byte record type is checked first, whatever its width or day
  (`records.check_types`: a file of another type in the folder raises `RecordTypeError`). Its race
  date is read next, from the bytes of 開催年 and 開催月日 alone: a record on a holdout day
  (`is_holdout_day('japan', race_date)`, D14) is read no further, whatever its width, and counted
  only in `holdout_skipped`. The others are then checked and decoded
  (`records.decode_lines`, a bad width raising `RecordLengthError` with the file's record number;
  a record too short to hold its date is one). A record without a valid race date cannot be
  cleared of being on a holdout day, so it is counted only in `undated`.
- `final_only` is true when no version before the final result (データ区分 1-6) is held, and null
  (inconclusive) when the record type has no record to count: no file, only empty files, or every
  record skipped. The command fails on a record type with no file under any DATASPEC folder.
- A version is a key's (データ区分, データ作成年月日). A record repeating a version already read
  (the same record in `stored/RACE` and `stored/DIFN`, or twice in one file) is counted once, and
  the repeats in `duplicates` only. A key's versions are pooled over every folder and file of its
  record type, so the key-level counts do not depend on where a version is filed.
- A key's versions are ordered by データ作成年月日, a tie in read order, as `latest_versions`
  gives a tie to the later row: the combo ends with the version the builders keep.

The report (`CardVersions.to_json`, written by `records.write_build_report` to
`<reports dir>/jravan-card-versions.json`) holds counts, codes, years and the as-of date only
(D21). See docs/specs/jravan.md, "Card versions (#401)".
"""

from __future__ import annotations

import collections
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, NoReturn

import numpy as np
import pandas as pd
import typer

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.jravan import keys, races, records, runners, spec

COUNTRY = "japan"
NAME = "card-versions"
RECORD_TYPES = ("RA", "SE")

# RA and SE items (the `records.decode` keys)
KUBUN, CREATED = races.KUBUN, races.CREATED
YEAR, MMDD, VENUE, RACE_NO = races.YEAR, races.MMDD, races.VENUE, races.RACE_NO
HORSE_NO, HORSE_ID = runners.HORSE_NO, runners.HORSE_ID

# データ区分 of RA and SE, translated
KUBUN_NAMES = {
    "1": "entries (出走馬名表)",
    "2": "card (出馬表)",
    "3": "flash result, first three placed (速報成績)",
    "4": "flash result, first five placed (速報成績)",
    "5": "flash result, every place (速報成績)",
    "6": "flash result, every place and corner (速報成績)",
    "7": "final result (成績)",
    "9": "abandoned (レース中止)",
    "A": "NAR (地方競馬)",
    "B": "overseas (海外国際レース)",
    "0": "deletion (削除)",
}
PRE_RACE = frozenset({"1", "2"})
NOT_FINAL = frozenset("123456")  # the versions before the final result
COMBO_SEPARATOR = " > "

_NAT = np.datetime64("NaT", "D")


@dataclass(frozen=True)
class CardVersions:
    """The census: per record type, counts, codes and years only (D21), and the as-of date."""

    as_of: date
    record_types: dict[str, dict[str, Any]]
    name: str = NAME

    def to_json(self) -> str:
        body = {
            "name": self.name,
            "as_of": self.as_of.isoformat(),
            "record_types": self.record_types,
        }
        return json.dumps(body, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


# ------------------------------------------------------------------------------------------------
# Per record


def _join(*arrays: np.ndarray) -> np.ndarray:
    """Fixed-width byte arrays shaped (n,), side by side, as one `S<total>` array."""
    n = len(arrays[0])
    width = sum(a.dtype.itemsize for a in arrays)
    if n == 0:
        return np.empty(0, dtype=f"S{width}")
    parts = [np.ascontiguousarray(a).view(np.uint8).reshape(n, a.dtype.itemsize) for a in arrays]
    return np.ascontiguousarray(np.hstack(parts)).view(f"S{width}").reshape(n)


def _date(raw: bytes) -> np.datetime64:
    if len(raw) != 8 or not raw.isdigit():
        return _NAT
    try:
        return np.datetime64(date(int(raw[:4]), int(raw[4:6]), int(raw[6:])), "D")
    except ValueError:
        return _NAT


def _dates(raw: np.ndarray) -> np.ndarray:
    """`YYYYMMDD` bytes as datetime64[D], each distinct value parsed once; anything else (blank,
    00000000, a day that does not exist) is NaT."""
    values, inverse = np.unique(raw, return_inverse=True)
    parsed = np.array([_date(bytes(v)) for v in values], dtype="datetime64[D]")
    return parsed[inverse.reshape(-1)].reshape(raw.shape)


def _race_dates(cols: Mapping[str, np.ndarray]) -> np.ndarray:
    return _dates(_join(cols[YEAR], cols[MMDD]))


def _line_dates(lines: list[bytes], year: slice, mmdd: slice) -> np.ndarray:
    """Each undecoded record's race date from its 開催年 and 開催月日 bytes alone (`_dates`); NaT
    for a record too short to hold them."""
    end = max(year.stop, mmdd.stop)
    width = (year.stop - year.start) + (mmdd.stop - mmdd.start)
    raw = [line[year] + line[mmdd] if len(line) >= end else b"" for line in lines]
    return _dates(np.array(raw, dtype=f"S{width}"))


def record_versions(cols: Mapping[str, np.ndarray], record_type: str) -> pd.DataFrame:
    """One row per RA or SE record of `cols` (as `records.decode` returns them), never
    deduplicated, with `race_id`, `data_kubun`, `created` and `race_date`, and for SE
    `horse_no` and `horse_id` (see the module docstring)."""
    if record_type not in RECORD_TYPES:
        raise ValueError(f"record_type must be one of {RECORD_TYPES}, got {record_type!r}")
    text = {name: records.text(cols[name]) for name in (YEAR, MMDD, VENUE, RACE_NO)}
    race_keys = list(zip(text[YEAR], text[MMDD], text[VENUE], text[RACE_NO], strict=True))
    race_ids = {k: keys.race_id(*k) for k in set(race_keys)}
    frame = pd.DataFrame(
        {
            "race_id": pd.Series([race_ids[k] for k in race_keys], dtype=object),
            "data_kubun": records.text(cols[KUBUN]),
            "created": _dates(cols[CREATED]),
            "race_date": _race_dates(cols),
        }
    )
    if record_type == "SE":
        numbers = records.integer(cols[HORSE_NO])
        frame["horse_no"] = pd.array(
            [None if np.isnan(v) else int(v) for v in numbers], dtype="Int64"
        )
        frame["horse_id"] = records.text(cols[HORSE_ID])
    return frame


# ------------------------------------------------------------------------------------------------
# The census


def _files(raw: Path, record_type: str) -> list[Path]:
    stored = raw / "stored"
    if not stored.is_dir():
        raise FileNotFoundError(f"JRA-VAN export not found: {stored}")
    return [
        path
        for dataspec in sorted(stored.iterdir())
        if dataspec.is_dir()
        for path in sorted((dataspec / record_type).glob("*.jvd"))
        if path.is_file()
    ]


def _holdout_flags(days: np.ndarray, holdout: Callable[[date], bool]) -> np.ndarray:
    """Whether each (valid) race date is a holdout day, `holdout` asked once per distinct day."""
    values, inverse = np.unique(days, return_inverse=True)
    flags = np.array([holdout(v.item()) for v in values], dtype=bool)
    return flags[inverse.reshape(-1)]


def _read(files: list[Path], layout: spec.RecordSpec) -> tuple[dict[str, np.ndarray], int, int]:
    """Per record outside the holdout: its key bytes, データ区分, データ作成年月日, race date and
    開催年, in read order; and the records skipped as on a holdout day and as undated."""
    key_items = [item.name for item in layout.items if item.is_key]
    year, mmdd = records.leaf_bytes(layout, YEAR), records.leaf_bytes(layout, MMDD)
    cache: dict[date, bool] = {}

    def holdout(day: date) -> bool:
        if day not in cache:
            cache[day] = is_holdout_day(COUNTRY, day)
        return cache[day]

    parts: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    skipped = undated = 0
    for path in files:
        for first, lines in records.iter_lines(path, layout):
            records.check_types(lines, layout, first)  # before any date is trusted
            days = _line_dates(lines, year, mmdd)  # the only bytes read before the holdout check
            dated = ~np.isnat(days)
            held = np.zeros(len(days), dtype=bool)
            held[dated] = _holdout_flags(days[dated], holdout)
            skipped += int(held.sum())
            rest = np.flatnonzero(~held)
            if not rest.size:
                continue
            kept_lines = lines if rest.size == len(lines) else [lines[i] for i in rest]
            cols = records.decode_lines(kept_lines, layout, first + rest)
            days, use = days[rest], dated[rest]
            undated += int((~use).sum())
            parts["key"].append(_join(*(cols[name][use] for name in key_items)))
            parts["kubun"].append(cols[KUBUN][use])
            parts["created"].append(cols[CREATED][use])
            parts["day"].append(days[use])
            parts["year"].append(cols[YEAR][use])
            del cols, kept_lines
    kept = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    return kept, skipped, undated


def _codes(raw: np.ndarray) -> tuple[list[str], np.ndarray]:
    """The distinct texts of `raw` ('' for blank) and each value's index into them."""
    values, inverse = np.unique(raw, return_inverse=True)
    return [t or "" for t in records.text(values)], inverse.reshape(-1)


def _section(kept: dict[str, np.ndarray], as_of: date) -> dict[str, Any]:
    """The counts of one record type over its records outside the holdout (see the docstring);
    with no record to count, `final_only` is None (inconclusive)."""
    if not kept or len(kept["key"]) == 0:
        return {
            "records": 0, "duplicates": 0, "by_kubun": {}, "by_year": {},
            "multi_version_keys": 0, "combos": {}, "pre_race_on_run_races": 0, "held": {},
            "final_only": None,
        }  # fmt: skip
    # one row per version, in read order
    _, first = np.unique(_join(kept["key"], kept["kubun"], kept["created"]), return_index=True)
    first = np.sort(first)
    duplicates = len(kept["key"]) - len(first)
    key, created, day = kept["key"][first], kept["created"][first], kept["day"][first]
    kubun_text, kubun = _codes(kept["kubun"][first])
    year_text, year = _codes(kept["year"][first])

    by_kubun: collections.Counter[str] = collections.Counter()
    for i, n in enumerate(np.bincount(kubun, minlength=len(kubun_text))):
        by_kubun[kubun_text[i]] += int(n)
    by_year: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    pairs = year * len(kubun_text) + kubun
    for pair, n in zip(*np.unique(pairs, return_counts=True), strict=True):
        y, k = divmod(int(pair), len(kubun_text))
        by_year[year_text[y]][kubun_text[k]] += int(n)

    pre = np.isin(kubun, [i for i, k in enumerate(kubun_text) if k in PRE_RACE])
    pre_race_on_run = int((pre & (day < np.datetime64(as_of, "D"))).sum())

    _, key_index, key_counts = np.unique(key, return_inverse=True, return_counts=True)
    key_index = key_index.reshape(-1)
    multi = np.flatnonzero(key_counts[key_index] > 1)
    _, created_rank = np.unique(created[multi], return_inverse=True)
    rows = multi[np.lexsort((multi, created_rank.reshape(-1), key_index[multi]))]
    combos: collections.Counter[str] = collections.Counter()
    if len(rows):
        starts = np.flatnonzero(np.diff(key_index[rows])) + 1
        for versions in np.split(kubun[rows], starts):
            combos[COMBO_SEPARATOR.join(kubun_text[k] for k in versions)] += 1

    return {
        "records": len(first),
        "duplicates": int(duplicates),
        "by_kubun": dict(sorted(by_kubun.items())),
        "by_year": {y: dict(sorted(c.items())) for y, c in sorted(by_year.items())},
        "multi_version_keys": int((key_counts > 1).sum()),
        "combos": dict(sorted(combos.items())),
        "pre_race_on_run_races": pre_race_on_run,
        "held": {k: KUBUN_NAMES.get(k) for k in sorted(by_kubun)},
        "final_only": not set(by_kubun) & NOT_FINAL,
    }


def census(*, raw: Path | None = None, as_of: date) -> CardVersions:
    """Count the RA and SE records of the export per データ区分, race year and key (see the
    module docstring); `as_of` decides which races count as already run."""
    raw = Path(raw) if raw is not None else paths.raw_dir(records.SOURCE)
    sections = {}
    for record_type in RECORD_TYPES:
        files = _files(raw, record_type)
        layout = spec.record_spec(record_type)
        kept, skipped, undated = _read(files, layout) if files else ({}, 0, 0)
        sections[record_type] = {
            **_section(kept, as_of),
            "holdout_skipped": skipped,
            "undated": undated,
        }
        del kept
    return CardVersions(as_of=as_of, record_types=sections)


# ------------------------------------------------------------------------------------------------
# The command


def card_versions_command(
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="Races before this day YYYY-MM-DD are run (default today)."),
    ] = None,
) -> None:
    """Count the JV-Data RA and SE records of the export per データ区分 (which versions it
    holds); the report (counts and codes only) goes to <reports dir>/jravan-card-versions.json."""
    day = _parse_day(as_of) if as_of else datetime.now(keys.JST).date()
    errors = (paths.DataDirNotSet, LicensedOutputInRepo, ValueError, LookupError, OSError)
    try:
        raw = paths.raw_dir(records.SOURCE)
        missing = [t for t in RECORD_TYPES if not _files(raw, t)]
    except errors as exc:
        _die(f"jravan card-versions failed: {type(exc).__name__}: {exc}")
    if missing:  # an incomplete copy of the export, not evidence of anything
        where = "; ".join(f"no {t} file under {raw / 'stored'}/<DATASPEC>/{t}/" for t in missing)
        _die(f"jravan card-versions failed: {where}")
    try:
        report = census(raw=raw, as_of=day)
        target = records.write_build_report(report)
    except errors as exc:
        _die(f"jravan card-versions failed: {type(exc).__name__}: {exc}")
    for record_type, section in report.record_types.items():
        held = ", ".join(section["held"]) or "none"
        if section["final_only"] is None:  # nothing to count: no verdict either way
            final_only = pre_race = "inconclusive"
        else:
            final_only = "yes" if section["final_only"] else "no"
            pre_race = "yes" if section["pre_race_on_run_races"] else "none"
        typer.echo(
            f"{record_type}: データ区分 held {held}; final version only: {final_only}; "
            f"pre-race versions of races run before {day.isoformat()}: {pre_race}"
        )
    typer.echo(f"report: {target}")


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{value!r} is not a YYYY-MM-DD date", param_hint="--as-of"
        ) from exc


def _die(message: str) -> NoReturn:
    typer.echo(f"ERROR {message}", err=True)
    raise typer.Exit(1)
