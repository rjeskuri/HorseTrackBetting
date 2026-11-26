"""JV-Data records -> named byte arrays, the converters on them, and the families' output helpers.

`decode` slices a record type's bytes with the layout `spec.record_spec` reads, as
`countries/japan/jrdb/parse.py` does: the records become one uint8 matrix and every leaf item is a
fixed-width view of its columns, an `S<length>` array shaped (n,), (n, repeat) or
(n, repeat, sub-repeat) (a repeat of 1 adds no axis). Keys are a top-level item's name or
`group/sub`; a name that occurs again at the same level is `name#2`, `name#3`. Items named `予備`
and the record separator (the layout's last item) are left out. Records split on LF (a CR before
it is dropped), empty lines are skipped and the last record may lack its separator. Each record
must be `spec.length - 2` bytes (else `RecordLengthError`) and start with `spec.record_type` (else
`RecordTypeError`); both are `ValueError`s carrying the 1-based `record` number.

`iter_file` streams a file in chunks of whole records, so a multi-GB file decodes in bounded
memory. `text`, `integer` and `scaled` convert each distinct value once and keep the input shape.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from horsetrack.core import paths
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.jravan.spec import SEPARATOR_BYTES, Item, RecordSpec

SOURCE = "jravan"
ENCODING = "cp932"
TRIM = " 　"  # half- and full-width spaces
RESERVE = "予備"
CHUNK_BYTES = 16 * 1024 * 1024  # a chunk's raw bytes at most (an H6 record alone is ~100 KB)

_INTEGER = re.compile(rb" *([+-]?) *([0-9]+) *")

Arrays = dict[str, np.ndarray]


class RecordLengthError(ValueError):
    """A record whose data width is not the layout's (a file of another type, or one cut off)."""

    def __init__(self, record_type: str, record: int, length: int, expected: int) -> None:
        super().__init__(
            f"{record_type} record {record}: {length} data bytes, the layout has {expected}"
        )
        self.record_type = record_type
        self.record = record
        self.length = length
        self.expected = expected


class RecordTypeError(ValueError):
    """A record of the layout's width that does not start with the layout's record type."""

    def __init__(self, record_type: str, record: int, found: str) -> None:
        super().__init__(
            f"{record_type} record {record}: the record type is {found!r}, not {record_type!r}"
        )
        self.record_type = record_type
        self.record = record
        self.found = found


# ------------------------------------------------------------------------------------------------
# Decoding


@dataclass(frozen=True)
class _Leaf:
    key: str
    base: int  # 0-based offset of the top-level item in the record
    repeat: int  # the top-level item's repeat
    stride: int  # the top-level item's width of one occurrence
    offset: int  # 0-based offset of the leaf in one occurrence (0 for a plain item)
    sub_repeat: int
    length: int

    def shape(self, n: int) -> tuple[int, ...]:
        return (
            (n,)
            + ((self.repeat,) if self.repeat > 1 else ())
            + ((self.sub_repeat,) if self.sub_repeat > 1 else ())
        )


def _named(items: tuple[Item, ...]) -> list[tuple[str, Item]]:
    """Each item but the reserves, under its name or `name#k` for its k-th occurrence."""
    seen: dict[str, int] = {}
    out = []
    for item in items:
        if item.name == RESERVE:
            continue
        seen[item.name] = seen.get(item.name, 0) + 1
        k = seen[item.name]
        out.append((item.name if k == 1 else f"{item.name}#{k}", item))
    return out


@lru_cache(maxsize=64)
def _leaves(spec: RecordSpec) -> tuple[_Leaf, ...]:
    leaves = []
    for name, item in _named(spec.items[:-1]):  # the last item is the separator
        base = item.start - 1
        if not item.items:
            leaves.append(_Leaf(name, base, item.repeat, item.length, 0, 1, item.length))
            continue
        for sub_name, sub in _named(item.items):
            leaves.append(
                _Leaf(
                    f"{name}/{sub_name}",
                    base,
                    item.repeat,
                    item.length,
                    sub.start - 1,
                    sub.repeat,
                    sub.length,
                )
            )
    return tuple(leaves)


def leaf_axes(spec: RecordSpec) -> dict[str, int]:
    """Each `decode` key of `spec`, in layout order, with its number of repeat axes (0-2)."""
    return {leaf.key: (leaf.repeat > 1) + (leaf.sub_repeat > 1) for leaf in _leaves(spec)}


def _lines(data: bytes) -> list[bytes]:
    lines = (line[:-1] if line.endswith(b"\r") else line for line in data.split(b"\n"))
    return [line for line in lines if line]


def _decode_lines(lines: list[bytes], spec: RecordSpec, first: int = 1) -> Arrays:
    """`lines` (records without separators), the first of them record number `first`."""
    width = spec.length - SEPARATOR_BYTES
    lengths = np.fromiter(map(len, lines), dtype=np.int64, count=len(lines))
    wrong = np.flatnonzero(lengths != width)
    if wrong.size:
        i = int(wrong[0])
        raise RecordLengthError(spec.record_type, first + i, int(lengths[i]), width)
    n = len(lines)
    matrix = np.frombuffer(b"".join(lines), dtype=np.uint8).reshape(n, width)
    prefix = np.frombuffer(spec.record_type.encode("ascii"), dtype=np.uint8)
    other = np.flatnonzero((matrix[:, : prefix.size] != prefix).any(axis=1))
    if other.size:
        i = int(other[0])
        found = bytes(matrix[i, : prefix.size]).decode("ascii", errors="replace")
        raise RecordTypeError(spec.record_type, first + i, found)
    out: Arrays = {}
    for leaf in _leaves(spec):
        block = matrix[:, leaf.base : leaf.base + leaf.repeat * leaf.stride]
        block = block.reshape(n, leaf.repeat, leaf.stride)
        part = block[:, :, leaf.offset : leaf.offset + leaf.sub_repeat * leaf.length]
        view = np.ascontiguousarray(part).view(f"S{leaf.length}")
        out[leaf.key] = view.reshape(leaf.shape(n))
    return out


def decode(data: bytes, spec: RecordSpec) -> Arrays:
    """One `S<length>` array per leaf item of every record in `data` (see the module docstring)."""
    return _decode_lines(_lines(data), spec)


def iter_file(path: Path, spec: RecordSpec, *, chunk_records: int = 20_000) -> Iterator[Arrays]:
    """`decode`-shaped dicts of at most `chunk_records` records each (fewer when that many would
    exceed `CHUNK_BYTES`), in file order, read lazily; record numbers in errors count the file."""
    if chunk_records < 1:
        raise ValueError(f"chunk_records must be at least 1, got {chunk_records}")
    per_chunk = max(1, min(chunk_records, CHUNK_BYTES // spec.length))
    read_bytes = per_chunk * spec.length
    first = 1
    pending: list[bytes] = []
    tail = b""
    with open(path, "rb") as handle:
        while block := handle.read(read_bytes):
            parts = (tail + block).split(b"\n")
            tail = parts.pop()
            parts = [p[:-1] if p.endswith(b"\r") else p for p in parts]
            pending.extend(p for p in parts if p)
            while len(pending) >= per_chunk:
                batch, pending = pending[:per_chunk], pending[per_chunk:]
                yield _decode_lines(batch, spec, first)
                first += len(batch)
    pending.extend(_lines(tail))
    if pending:
        yield _decode_lines(pending, spec, first)


# ------------------------------------------------------------------------------------------------
# Converters


def _by_value(raw: np.ndarray, convert: Callable[[bytes], object], dtype) -> np.ndarray:
    """`convert` applied once per distinct value of `raw`, shaped as `raw`."""
    raw = np.asarray(raw)
    if raw.size == 0:
        return np.empty(raw.shape, dtype=dtype)
    values, inverse = np.unique(raw.reshape(-1), return_inverse=True)
    converted = np.empty(len(values), dtype=dtype)
    converted[:] = [convert(bytes(v)) for v in values]
    return converted[inverse.reshape(-1)].reshape(raw.shape)


def _text(raw: bytes) -> str | None:
    return raw.decode(ENCODING, errors="replace").strip(TRIM) or None


def _integer(raw: bytes) -> float:
    match = _INTEGER.fullmatch(raw)
    if match is None:
        return np.nan
    value = float(match.group(2))
    return -value if match.group(1) == b"-" else value


def text(raw: np.ndarray) -> np.ndarray:
    """cp932 text with half- and full-width spaces trimmed at both ends; blank is None."""
    return _by_value(raw, _text, object)


def integer(raw: np.ndarray) -> np.ndarray:
    """Digits with an optional sign, space-padded, as float64; anything else is NaN."""
    return _by_value(raw, _integer, np.float64)


def scaled(raw: np.ndarray, places: int) -> np.ndarray:
    """`integer(raw)` with `places` implied decimal places: `b'0123'`, 1 -> 12.3."""
    return integer(raw) / 10**places


# ------------------------------------------------------------------------------------------------
# Versions


def latest_versions(
    frame: pd.DataFrame,
    key: list[str],
    *,
    created: str = "データ作成年月日",
    kubun: str = "データ区分",
) -> pd.DataFrame:
    """One row per `key`: the latest `created` (YYYYMMDD text), a tie going to the later row; a
    key whose kept row has `kubun` '0' (a deletion) is dropped. Kept rows stay in input order,
    with their index labels."""
    work = pd.DataFrame({f"k{i}": frame[column].to_numpy() for i, column in enumerate(key)})
    work["created"] = frame[created].to_numpy()
    by_created = work.sort_values("created", kind="stable", na_position="first")
    last = by_created.drop_duplicates([f"k{i}" for i in range(len(key))], keep="last")
    kept = np.sort(last.index.to_numpy())
    kept = kept[frame[kubun].to_numpy()[kept] != "0"]
    return frame.iloc[kept]


# ------------------------------------------------------------------------------------------------
# Output helpers


@dataclass
class BuildReport:
    """Counts a family's build reports: no record content, only counts, years and codes."""

    name: str
    counts: dict[str, int]
    by_year: dict[int, dict[str, int]]
    unmapped: dict[str, dict[str, int]]  # field -> code -> count
    failures: list[dict]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def write_parquet(table: pa.Table, path: Path) -> None:
    """`table` to `path` through a tmp file in the same folder and `os.replace`; the same table
    always gives the same bytes (no wall-clock metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_build_report(report: BuildReport) -> Path:
    """`report` as `<reports dir>/jravan-<name>.json`, through `write_report` (never the repo)."""
    target = paths.reports_dir() / f"jravan-{report.name}.json"
    return write_report(target, report.to_json(), sources=(SOURCE,))
