"""JRDB fixed-width files -> one typed DataFrame row per record, driven by per-family field specs.

Ported from HKTrack's ``sources/jra/jrdb/parse.py``. JRDB's layouts are licensed, so no spec ships
with the package: each family's spec is ``<data dir>/jrdb/specs/<family in lower case>.csv``, read
on every call (a changed HORSETRACK_DATA_DIR takes effect at once), and any family with a spec there
is accepted. A spec is UTF-8, ``#`` lines are comments, and its csv header is
``name,start,length,kind,added,label``: one row per field in record order, ``start`` the 1-based
byte offset, ``length`` in bytes, ``added`` the date of the spec edition that added the field,
``label`` the spec document's item name. The 2-byte record terminator is the last row, a ``skip``.
Fields must cover the record from byte 1 without gaps or overlaps, with positive lengths, known
kinds and unique names; anything else is a ValueError, never a silent mis-slice.

Kinds: ``text`` (cp932, half- and full-width spaces trimmed at both ends only, blank -> None),
``num`` (float whatever the alignment: '56 ' -> 56, '- 4' -> -4; blank or a bare sign -> NaN),
``code`` (text whose leading zeros matter), ``date`` (YYYYMMDD -> ``datetime.date``, blank or
invalid -> None) and ``skip`` (reserve bytes, the parts of the race key, the terminator: no column).

Records end in CR LF, LF or LF LF; empty lines are skipped and the last record may lack its
terminator. Bytes are sliced before decoding. A record whose data is not ``record_length - 2``
bytes raises ``RecordLengthError``: a file of another family or one cut off mid-record never parses
silently.

A family whose first five fields are the race key's parts gets ``race_key`` (bytes 1-8: venue 2,
year 2, kai 1, hex nichi 1, race 2, as printed); one whose first four are gets ``meeting_key``
(bytes 1-6); any other family has no key column.
"""

from __future__ import annotations

import csv
import re
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from horsetrack.core import paths

ENCODING = "cp932"  # Shift_JIS plus the Windows extensions JRDB uses (㈱, ㈲)
TERMINATOR_BYTES = 2
KINDS = ("text", "num", "code", "date", "skip")
TRIM = " 　"  # half- and full-width spaces
SPEC_HEADER = ("name", "start", "length", "kind", "added", "label")

_RACE_KEY_PARTS = ("venue_code", "year", "kai", "nichi", "race_no")
_FAMILY = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Field:
    name: str
    start: int  # 1-based
    length: int
    kind: str
    added: date
    label: str


class RecordLengthError(ValueError):
    """A record whose data length is not the family's (a file of another family, or one cut off)."""

    def __init__(self, family: str, record: int, length: int, expected: int) -> None:
        super().__init__(f"{family} record {record}: {length} data bytes, the spec has {expected}")
        self.family = family
        self.record = record
        self.length = length
        self.expected = expected


def specs_dir() -> Path:
    """Where the families' spec tables live: ``<data dir>/jrdb/specs``."""
    return paths.data_dir() / "jrdb" / "specs"


def spec(family: str) -> tuple[Field, ...]:
    """The fields of ``family`` in record order, from its spec in ``specs_dir()``, checked."""
    family = _family_name(family)
    path = specs_dir() / f"{family.lower()}.csv"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise LookupError(f"no spec for JRDB family {family}: {path} does not exist") from None
    return _fields(path.name, text)


def _family_name(family: str) -> str:
    if not _FAMILY.fullmatch(family):
        raise ValueError(f"{family!r} is not a JRDB family name (letters only)")
    return family.upper()


def _fields(source: str, text: str) -> tuple[Field, ...]:
    rows = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
    missing = [column for column in SPEC_HEADER if column not in (rows.fieldnames or ())]
    if missing:
        raise ValueError(f"{source}: the header lacks {', '.join(missing)}")
    fields = []
    for row in rows:
        try:
            fields.append(
                Field(
                    row["name"],
                    int(row["start"]),
                    int(row["length"]),
                    row["kind"],
                    date.fromisoformat(row["added"]),
                    row["label"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: row {row!r} is not a field ({exc})") from None
    pos, previous = 1, None
    for f in fields:
        if f.length <= 0:
            raise ValueError(f"{source}: field {f.name} has length {f.length}")
        if f.kind not in KINDS:
            raise ValueError(f"{source}: field {f.name} has kind {f.kind!r}, not one of {KINDS}")
        if f.start != pos:
            if previous is None:
                where = "the record starts at byte 1"
            elif f.start < pos:
                where = f"it overlaps {previous.name}, bytes {previous.start}-{pos - 1}"
            else:
                where = f"bytes {pos}-{f.start - 1} after {previous.name} are not covered"
            raise ValueError(f"{source}: field {f.name} starts at byte {f.start}: {where}")
        pos += f.length
        previous = f
    names = [f.name for f in fields]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{source}: duplicate field names {', '.join(duplicates)}")
    if not fields or fields[-1].kind != "skip" or fields[-1].length != TERMINATOR_BYTES:
        raise ValueError(f"{source}: the last field must be the {TERMINATOR_BYTES}-byte terminator")
    return tuple(fields)


def record_length(family: str) -> int:
    """The spec's record length: data bytes plus the terminator."""
    return sum(f.length for f in spec(family))


def _key_column(fields: tuple[Field, ...]) -> tuple[str, int] | None:
    names = tuple(f.name for f in fields[: len(_RACE_KEY_PARTS)])
    if names == _RACE_KEY_PARTS:
        return "race_key", sum(f.length for f in fields[: len(_RACE_KEY_PARTS)])
    if names[:4] == _RACE_KEY_PARTS[:4]:
        return "meeting_key", sum(f.length for f in fields[:4])
    return None


def key_column(family: str) -> tuple[str, int] | None:
    """(name, width) of the key printed at the start of each record: race_key or meeting_key."""
    return _key_column(spec(family))


def _columns(fields: tuple[Field, ...]) -> list[str]:
    key = _key_column(fields)
    return ([key[0]] if key else []) + [f.name for f in fields if f.kind != "skip"]


def columns(family: str) -> list[str]:
    """The columns ``parse_bytes`` returns for ``family``, in order."""
    return _columns(spec(family))


def _records(family: str, data: bytes, width: int) -> list[bytes]:
    records = [line[:-1] if line.endswith(b"\r") else line for line in data.split(b"\n")]
    records = [r for r in records if r]
    for number, record in enumerate(records, start=1):
        if len(record) != width:
            raise RecordLengthError(family, number, len(record), width)
    return records


def _by_value(raw: np.ndarray, convert: Callable[[bytes], object]) -> pd.Series:
    """``convert`` applied once per distinct byte string (values repeat a lot): an object Series.
    The dtype is explicit because pandas 3 infers ``str`` from strings, turning None into NaN."""
    if raw.size == 0:
        return pd.Series([], dtype=object)
    values, inverse = np.unique(raw, return_inverse=True)
    converted = np.empty(len(values), dtype=object)
    converted[:] = [convert(bytes(v)) for v in values]
    return pd.Series(converted[inverse.reshape(-1)], dtype=object)


def _text(raw: bytes) -> str | None:
    return raw.decode(ENCODING, errors="replace").strip(TRIM) or None


def _number(raw: bytes) -> float:
    digits = raw.replace(b" ", b"")
    try:
        return float(digits) if digits else np.nan
    except ValueError:
        return np.nan


def _date(raw: bytes) -> date | None:
    text = raw.decode("ascii", errors="replace").strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError:
        return None


def parse_bytes(family: str, data: bytes) -> pd.DataFrame:
    """One row per record of a ``family`` file's bytes, in file order (see the module docstring)."""
    family = _family_name(family)
    fields = spec(family)
    width = sum(f.length for f in fields) - TERMINATOR_BYTES
    records = _records(family, data, width)
    matrix = np.frombuffer(b"".join(records), dtype=np.uint8).reshape(len(records), width)

    def column(start: int, length: int) -> np.ndarray:
        part = np.ascontiguousarray(matrix[:, start - 1 : start - 1 + length])
        return part.view(f"S{length}").reshape(-1)

    out: dict[str, object] = {}
    key = _key_column(fields)
    if key is not None:
        out[key[0]] = _by_value(column(1, key[1]), lambda b: b.decode("ascii", errors="replace"))
    for f in fields:
        if f.kind == "skip":
            continue
        raw = column(f.start, f.length)
        if f.kind == "num":
            out[f.name] = _by_value(raw, _number).astype("float64")
        elif f.kind == "date":
            out[f.name] = _by_value(raw, _date)
        else:
            out[f.name] = _by_value(raw, _text)
    return pd.DataFrame(out, columns=_columns(fields))


def family_of(name: str | Path) -> str:
    """The family a file is named after, its leading letters: 'abc150712.txt' -> 'ABC'."""
    m = _FAMILY.match(Path(name).name)
    if not m:
        raise ValueError(f"{name}: not named after a JRDB file family (no leading letters)")
    return m.group(0).upper()


def members(path: str | Path) -> Iterator[tuple[str, bytes]]:
    """(name, raw bytes) of each data file in ``path``: the file itself, or a .zip's .txt members
    in name order."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        yield path.name, path.read_bytes()
    elif suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = sorted(n for n in archive.namelist() if n.lower().endswith(".txt"))
            for name in names:
                yield Path(name).name, archive.read(name)
    else:
        raise ValueError(f"{path.name}: only .txt and .zip files are read")


def parse_file(path: str | Path) -> pd.DataFrame:
    """A ``<family>yymmdd.txt`` file, or a .zip of one or more such members (a year pack), as one
    frame; the family comes from the file name, and a member named after another family is left
    out."""
    family = family_of(path)
    frames = []
    for name, data in members(path):
        prefix = _FAMILY.match(name)
        if prefix and prefix.group(0).upper() != family:
            continue
        frames.append(parse_bytes(family, data))
    if not frames:
        return parse_bytes(family, b"")
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
