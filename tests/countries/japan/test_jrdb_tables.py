"""`horsetrack japan jrdb build` (ticket #57, ported from HKTrack's `sources/jra/jrdb/tables.py`).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jrdb.tables`:
- `run()` reads every archive under `paths.raw_dir("jrdb") / <FAM>/` for the pre-race families
  KYI, CYB, JOA, TYB and the post-race families SED, SKB, parses them with `parse` (the specs are
  `<data dir>/jrdb/specs/<fam>.csv`), and writes `jrdb_pre_race.parquet` and
  `jrdb_post_race.parquet` under `paths.processed_dir("jra")`. It returns an outcome with
  `.lines` and `.error` (None when the build completed). The CLI command
  `horsetrack japan jrdb build` runs it: exit 0 when it completes, exit 1 with one `ERROR ...`
  line when it stops.
- Each table: a table's families merged on (race_key, horse_no) (an outer merge: a runner in any one
  family is a row), joined to the store's races and runners (`join.join_store`), and only rows whose
  race AND horse matched are kept. `jrdb_horse_id` is the first family's (table order) that prints
  one. Columns: the key columns the store declares for the table (`schemas.TABLES[table].columns`,
  `jurisdiction` "jra" included), then every other parsed column of family F as `<f>_<column>`
  (lower-case family). A family that was not delivered still has its columns, all null. The file
  is readable by `core.store.tables.read_table` and stored in its sort order (race_id, horse_no).
- A delivery file (`<FAM>yymmdd`) is read once: from a day file (`KYI261128.zip` or `.txt`)
  before a year pack (`KYI_2026.zip`, `KYI2026.zip`) holding a member of the same name; the pack's
  other members are still read.
- A family whose folder holds only .lzh archives stops the build before anything is written: one
  ERROR line naming the family. A .lzh beside a readable .zip is skipped, not an error.
- `paths.reports_dir() / "jrdb-build.json"`: per table its `rows` (JRDB rows after the merge),
  `table_rows` (rows kept), `race_unmatched_rate` and `horse_unmatched_rate` (as `JoinReport`).
- A `date`-kind field is a date32 column, whether its family was delivered or not (ticket #186).
- Upcoming cards (ticket #186, HKTrack's `_store` + `with_meeting_keys`): the store the JRDB rows
  join to is the history races and runners plus every upcoming meeting folder
  `paths.processed_dir("jra") / "upcoming" / <meeting_id with ':' as '_'>/` that holds both
  `races.parquet` and `racecard_runners.parquet` (a folder with only one of them is not read). A
  card race carries `meeting_id` (`jra:<YYYY-MM-DD>:<venue>`), `race_no`, `kai` and `nichi` (回, 日)
  but no `venue` or `race_date` (the columns are absent or null): both come from its `meeting_id`.
  History wins a race_id: a stale card for a race already in history changes nothing.
- UKC (ticket #186, Ruling: skip it; #209 owns the UKC horse-master merge): a UKC archive under
  `raw/jrdb/UKC/` (.zip, .txt or only .lzh) does not stop the build and is not read;
  `processed/jra/horse_pedigree.parquet` is neither written nor touched (a netkeiba-built one stays
  byte for byte; none stays none). The skip is stated: `jrdb-build.json` has
  `"horse_pedigree": {"skipped": <reason>}`, the reason naming UKC and #209, and the build prints a
  line naming both.

The families' fields, offsets and lengths below are invented (only the family names and the
race-key part names come from the ported code); every race, horse and id is invented. Expected rows
are worked out from the rule above over the records the test writes, never by calling the code
under test.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.store import schemas
from horsetrack.core.store import tables as store_tables

# isort: split
# (its own block: ruff ranks a module that does not exist yet as third-party)
from horsetrack.countries.japan.jrdb import tables

PRE, POST = "jrdb_pre_race", "jrdb_post_race"
TABLE_FAMILIES = {PRE: ("KYI", "CYB", "JOA", "TYB"), POST: ("SED", "SKB")}
ALL_FAMILIES = (*TABLE_FAMILIES[PRE], *TABLE_FAMILIES[POST])
VENUE_NO = {"KYO": "08", "TOK": "05", "HSN": "09"}
UNPREFIXED = ("race_key", "horse_no", "jrdb_horse_id")

_KEY = [("venue_code", 2, "skip"), ("year", 2, "skip"), ("kai", 1, "skip")]
_KEY += [("nichi", 1, "skip"), ("race_no", 2, "skip")]
# (name, length, kind): invented layouts; starts are cumulative, the 2-byte terminator is added.
LAYOUTS: dict[str, list[tuple[str, int, str]]] = {
    "KYI": [("horse_no", 2, "num"), ("jrdb_horse_id", 8, "code"), ("idm", 5, "num")]
    + [("mark", 2, "code"), ("card_date", 8, "date")],
    "CYB": [("horse_no", 2, "num"), ("tune", 3, "num")],
    "JOA": [("horse_no", 2, "num"), ("jrdb_horse_id", 8, "code"), ("idm", 5, "num")],
    "TYB": [("horse_no", 2, "num"), ("paddock_index", 5, "num")],
    "SED": [("horse_no", 2, "num"), ("jrdb_horse_id", 8, "code"), ("finish", 2, "num")]
    + [("note", 8, "text"), ("run_date", 8, "date")],
    "SKB": [("horse_no", 2, "num"), ("jrdb_horse_id", 8, "code"), ("comment_code", 3, "code")],
}
DATE_COLUMNS = {PRE: "kyi_card_date", POST: "sed_run_date"}  # the date-kind fields, prefixed
DATA_COLUMNS = {
    fam: [name for name, _, kind in layout if kind != "skip" and name not in UNPREFIXED]
    for fam, layout in LAYOUTS.items()
}


def full_layout(family: str) -> list[tuple[str, int, int, str]]:
    out, pos = [], 1
    for name, length, kind in [*_KEY, *LAYOUTS[family], ("eol", 2, "skip")]:
        out.append((name, pos, length, kind))
        pos += length
    return out


# ------------------------------------------------------------------------------------------------
# The synthetic world: store races and runners, and what each family prints about them
# ------------------------------------------------------------------------------------------------

DAY1, DAY2 = date(2026, 11, 28), date(2026, 11, 29)


@dataclass(frozen=True)
class Race:
    venue: str
    day: date
    kai: int
    nichi: int
    no: int
    horses: tuple[str, ...]  # store horse_id by horse_no (1-based)
    in_store: bool  # in the history store
    card: bool = False  # on an upcoming card that has its racecard_runners

    @property
    def race_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{self.venue}:{self.no:02d}"

    @property
    def jrdb_key(self) -> str:
        return (
            f"{VENUE_NO[self.venue]}{self.day.year % 100:02d}{self.kai}{self.nichi:x}{self.no:02d}"
        )

    @property
    def source_key(self) -> str:
        return f"{self.day.year}{VENUE_NO[self.venue]}{self.kai:02d}{self.nichi:02d}{self.no:02d}"


def _ids(year: int, first: int, n: int) -> tuple[str, ...]:
    return tuple(f"{year}10{first + i:04d}" for i in range(n))


R1 = Race("KYO", DAY1, 5, 7, 1, _ids(2023, 100, 6), True)
R2 = Race("KYO", DAY1, 5, 7, 2, _ids(2022, 200, 5), True)
R3 = Race("KYO", DAY1, 5, 7, 3, _ids(2021, 300, 4), False)  # JRDB prints it, the store lacks it
R11 = Race("TOK", DAY2, 4, 10, 11, _ids(2020, 400, 4), True)  # nichi 10 is printed as 'a'
RACES = (R1, R2, R3, R11)

# Upcoming meetings: not in history, on a card written under processed/jra/upcoming/.
DAY3, DAY4 = date(2026, 12, 5), date(2026, 12, 6)
U1 = Race("KYO", DAY3, 5, 9, 1, _ids(2024, 500, 5), False, card=True)
U2 = Race("KYO", DAY3, 5, 9, 2, _ids(2023, 600, 4), False, card=True)
U11 = Race("TOK", DAY4, 5, 12, 11, _ids(2022, 700, 6), False, card=True)  # nichi 12 is 'c'
U_HALF = Race("HSN", DAY4, 3, 4, 5, _ids(2021, 800, 3), False)  # its folder has no runners file
CARD_RACES = (U1, U2, U11)
UPCOMING = (*CARD_RACES, U_HALF)

MISMATCH = {("KYI", R2, 3): "99999999", ("SED", R11, 2): "88888888"}  # ids naming another horse
MISMATCH[("KYI", U2, 3)] = "77777777"


def selection(family: str, race: Race) -> list[int]:
    """The horse numbers a family prints for a race."""
    n = len(race.horses)
    every = list(range(1, n + 1))
    if family == "KYI":  # R1's last horse only in JOA (its id from there); R11's 4 only by number
        # U11's 6 has its id from JOA; U1's 5 is in neither, so it is matched by number
        return [h for h in every if (race, h) not in ((R1, 6), (R11, 4), (U11, 6), (U1, 5))]
    if family == "JOA":
        return [h for h in every if (race, h) not in ((R11, 4), (U1, 5))]
    if family == "CYB":
        return [*every, n + 1]  # a horse number the store's race does not have
    if family == "TYB":
        return [h for h in every if (race, h) != (R2, 5)]
    if family == "SKB":
        return [1, 2]
    return every  # SED


def jrdb_id(family: str, race: Race, h: int) -> str | None:
    if "jrdb_horse_id" not in [name for name, _, _ in LAYOUTS[family]]:
        return None
    if (family, race, h) in MISMATCH:
        return MISMATCH[(family, race, h)]
    if h > len(race.horses):
        return f"{19 + race.no:02d}10{900 + h:04d}"
    return race.horses[h - 1][2:]


def value(family: str, column: str, race: Race, h: int, stale: bool = False) -> tuple[str, Any]:
    """(the text as it sits in the record, the parsed value) of one data field."""
    if column == "idm":  # KYI and JOA both print an idm: the prefix keeps them apart
        v = (50.0 if family == "KYI" else 70.0) + race.no + h / 2 + (20.0 if stale else 0.0)
        return f"{v:5.1f}", v
    if column == "mark":
        text = "99" if stale else f"{(race.no + h) % 10:02d}"
        return text, text  # a code keeps its leading zero
    if column == "tune":
        v = 100 + 10 * race.no + h
        return f"{v:3d}", float(v)
    if column == "paddock_index":
        v = h * 1.5 - race.no
        return f"{v:5.1f}", v
    if column == "finish":
        v = len(race.horses) + 1 - h if h <= len(race.horses) else 99
        return f"{v:2d}", float(v)
    if column == "note":
        text = f"差し{h}" if h % 2 else f"逃げ{race.no}"
        return text, text
    if column == "comment_code":
        text = f"{(race.no * 7 + h) % 1000:03d}"
        return text, text
    if column in ("card_date", "run_date"):  # a blank date is None
        if (column == "card_date" and h == 2) or (column == "run_date" and h % 4 == 0):
            return "", None
        back = (30 if stale else 1 + h % 3) if column == "card_date" else 0
        d = race.day - timedelta(days=back)
        return f"{d:%Y%m%d}", d
    raise AssertionError(column)


def record(family: str, race: Race, h: int, stale: bool = False) -> bytes:
    out = bytearray()
    for name, start, length, _kind in full_layout(family)[:-1]:
        assert len(out) == start - 1
        if name in ("venue_code", "year", "kai", "nichi", "race_no"):
            raw = race.jrdb_key[start - 1 : start - 1 + length].encode("ascii")
        elif name == "horse_no":
            raw = f"{h:2d}".encode("ascii")
        elif name == "jrdb_horse_id":
            raw = (jrdb_id(family, race, h) or "").encode("ascii")
        else:
            raw = value(family, name, race, h, stale)[0].encode("cp932")
        assert len(raw) <= length, (family, name, raw)
        out += raw.ljust(length, b" ")
    return bytes(out)


def day_records(
    family: str,
    day: date,
    stale: bool = False,
    extra: bool = False,
    races: tuple[Race, ...] = RACES,
) -> bytes:
    rows = []
    for race in races:
        if race.day != day:
            continue
        horses = selection(family, race)
        if extra and race is R1 and 6 not in horses:
            horses = [*horses, 6]
        rows += [record(family, race, h, stale) for h in horses]
    return b"".join(r + b"\r\n" for r in rows)


def delivery(family: str, day: date) -> str:
    return f"{family}{day:%y%m%d}"


# ------------------------------------------------------------------------------------------------
# Writing the data directory
# ------------------------------------------------------------------------------------------------


def write_specs(data: Path) -> None:
    specs = data / "jrdb" / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    for family in ALL_FAMILIES:
        rows = ["# synthetic: an invented layout for tests, not a JRDB spec"]
        rows.append("name,start,length,kind,added,label")
        rows += [
            f"{n},{s},{ln},{k},2020-01-01,synthetic {n}" for n, s, ln, k in full_layout(family)
        ]
        (specs / f"{family.lower()}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_store_table(name: str, rows: list[dict[str, Any]]) -> None:
    """A store table with every declared column (None where the test has no value), typed."""
    spec = schemas.TABLES[name]
    schema = pa.schema([(c, store_tables.ARROW_TYPES[k]) for c, k in spec.columns.items()])
    columns = {c: [r.get(c) for r in rows] for c in spec.columns}
    path = paths.processed_dir("jra") / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), path)


def write_store() -> None:
    races, runners = [], []
    for race in RACES:
        if not race.in_store:
            continue
        keys = {
            "jurisdiction": "jra",
            "race_id": race.race_id,
            "meeting_id": f"jra:{race.day.isoformat()}:{race.venue}",
            "race_date": race.day,
            "venue": race.venue,
            "race_no": race.no,
        }
        races.append({**keys, "status": "run", "source_race_key": race.source_key})
        runners += [
            {**keys, "horse_id": hid, "horse_no": i, "is_scratched": False}
            for i, hid in enumerate(race.horses, 1)
        ]
    write_store_table("races", races)
    write_store_table("runners", runners)


def zip_of(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def write_day_files(
    families: tuple[str, ...] | list[str],
    days: tuple[date, ...] = (DAY1, DAY2),
    races: tuple[Race, ...] = RACES,
) -> None:
    root = paths.raw_dir("jrdb")
    for family in families:
        for day in days:
            name = delivery(family, day)
            data = day_records(family, day, races=races)
            zip_of(root / family / f"{name}.zip", {f"{name}.txt": data})


def meeting_id(race: Race) -> str:
    return f"jra:{race.day.isoformat()}:{race.venue}"


def write_card(meeting: list[Race], keys: str = "absent", runners: bool = True) -> Path:
    """One upcoming meeting's card in the shape HKTrack's card writer leaves it: races.parquet
    without venue and race_date (``keys`` "absent": no such columns; "null": the columns, all
    null) but with the meeting's kai and nichi; and racecard_runners.parquet (unless ``runners``
    is False). Returns the meeting folder."""
    mid = meeting_id(meeting[0])
    folder = paths.processed_dir("jra") / "upcoming" / mid.replace(":", "_")
    folder.mkdir(parents=True, exist_ok=True)
    n = len(meeting)
    columns: dict[str, pa.Array] = {
        "jurisdiction": pa.array(["jra"] * n, pa.string()),
        "race_id": pa.array([r.race_id for r in meeting], pa.string()),
        "meeting_id": pa.array([mid] * n, pa.string()),
        "race_no": pa.array([r.no for r in meeting], pa.int64()),
        "race_name": pa.array([f"synthetic race {r.no}" for r in meeting], pa.string()),
        "post_time_local": pa.array(
            [f"{r.day.isoformat()}T{9 + r.no % 8:02d}:45:00+09:00" for r in meeting], pa.string()
        ),
        "status": pa.array(["declared"] * n, pa.string()),
        "kai": pa.array([r.kai for r in meeting], pa.int64()),
        "nichi": pa.array([r.nichi for r in meeting], pa.int64()),
    }
    if keys == "null":
        columns["venue"] = pa.nulls(n, pa.string())
        columns["race_date"] = pa.nulls(n, pa.date32())
    pq.write_table(pa.table(columns), folder / "races.parquet")
    if runners:
        rows = [(r, i, hid) for r in meeting for i, hid in enumerate(r.horses, 1)]
        card = {
            "jurisdiction": pa.array(["jra"] * len(rows), pa.string()),
            "race_id": pa.array([r.race_id for r, _, _ in rows], pa.string()),
            "meeting_id": pa.array([mid] * len(rows), pa.string()),
            "horse_no": pa.array([i for _, i, _ in rows], pa.int64()),
            "horse_id": pa.array([hid for _, _, hid in rows], pa.string()),
            "horse_name": pa.array([f"synthetic {hid}" for _, _, hid in rows], pa.string()),
            "is_scratched": pa.array([False] * len(rows), pa.bool_()),
        }
        pq.write_table(pa.table(card), folder / "racecard_runners.parquet")
    return folder


def world(base: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh data directory under ``base`` with the specs and the store (no archives yet)."""
    data = base / "data"
    data.mkdir(parents=True)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    write_specs(data)
    write_store()
    return data


@pytest.fixture(autouse=True)
def no_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No real `.env` can supply a setting."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)


# ------------------------------------------------------------------------------------------------
# Expected tables, from the rule in the module docstring
# ------------------------------------------------------------------------------------------------


def _horse_id(jid: str) -> str:
    return ("19" if int(jid[:2]) >= 70 else "20") + jid


def expected(
    table: str, delivered: set[str], races: tuple[Race, ...] = RACES
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """The table's rows in (race_id, horse_no) order, and its counts for jrdb-build.json. A race is
    found when it is in history or on a card with its runners."""
    families = [f for f in TABLE_FAMILIES[table] if f in delivered]
    keys: list[tuple[Race, int]] = []
    for race in races:
        for family in families:
            keys += [(race, h) for h in selection(family, race) if (race, h) not in keys]
    rows, race_matched = [], 0
    for race, h in keys:
        printed = [f for f in families if h in selection(f, race)]
        ids = [jrdb_id(f, race, h) for f in printed]
        jid = next((i for i in ids if i is not None), None)
        if not (race.in_store or race.card):
            continue
        race_matched += 1
        if h > len(race.horses):
            continue
        store_id = race.horses[h - 1]
        if jid is not None and _horse_id(jid) != store_id:
            continue
        row: dict[str, Any] = {
            "jurisdiction": "jra",
            "race_id": race.race_id,
            "race_date": race.day,
            "horse_no": h,
            "horse_id": store_id,
            "race_key": race.jrdb_key,
            "jrdb_horse_id": jid,
            "matched_by": "horse_no" if jid is None else "horse_id",
        }
        for family in TABLE_FAMILIES[table]:
            for column in DATA_COLUMNS[family]:
                there = family in printed
                row[f"{family.lower()}_{column}"] = (
                    value(family, column, race, h)[1] if there else None
                )
        rows.append(row)
    rows.sort(key=lambda r: (r["race_id"], r["horse_no"]))
    counts = {
        "rows": len(keys),
        "table_rows": len(rows),
        "race_unmatched_rate": 1 - race_matched / len(keys),
        "horse_unmatched_rate": 1 - len(rows) / race_matched,
    }
    return rows, counts


def table_columns(table: str) -> set[str]:
    out = set(schemas.TABLES[table].columns)
    for family in TABLE_FAMILIES[table]:
        out |= {f"{family.lower()}_{c}" for c in DATA_COLUMNS[family]}
    return out


def _plain(v: Any) -> Any:
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.date()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):  # numpy scalars
        return _plain(v.item())
    return v


def built(table: str) -> pd.DataFrame:
    path = paths.processed_dir("jra") / f"{table}.parquet"
    assert path.exists(), f"{path} was not written"
    return pd.read_parquet(path)


def assert_table(table: str, delivered: set[str], races: tuple[Race, ...] = RACES) -> None:
    """The written table has exactly the expected columns, rows, order and values, and the store
    reader reads its key columns."""
    want, _ = expected(table, delivered, races)
    frame = built(table)
    assert set(frame.columns) == table_columns(table)
    got = [{c: _plain(v) for c, v in row.items()} for row in frame.to_dict("records")]
    for row in got:
        row["horse_no"] = int(row["horse_no"])
    assert [(r["race_id"], r["horse_no"]) for r in got] == [
        (r["race_id"], r["horse_no"]) for r in want
    ]
    wrong = {
        f"{w['race_id']}#{w['horse_no']}.{c}": (g.get(c), w[c])
        for g, w in zip(got, want, strict=True)
        for c in w
        if g.get(c) != w[c]
    }
    assert wrong == {}, wrong

    keys = store_tables.read_table("jra", table)
    assert len(keys) == len(want)
    assert [int(n) for n in keys["horse_no"]] == [w["horse_no"] for w in want]
    assert list(keys["race_date"]) == [w["race_date"] for w in want]
    assert set(keys["jurisdiction"]) == {"jra"}


def invoke_build():
    return CliRunner().invoke(app, ["japan", "jrdb", "build"])


def build_ok() -> None:
    result = invoke_build()
    assert result.exit_code == 0, result.output
    assert "ERROR" not in result.output, result.output


# ------------------------------------------------------------------------------------------------
# The tests
# ------------------------------------------------------------------------------------------------


def test_the_synthetic_world_is_what_the_tests_claim() -> None:
    """Guards the fixtures: each table has kept rows, dropped rows of every kind and both
    matched_by values, so a table that kept everything or nothing could not pass."""
    for table in (PRE, POST):
        rows, counts = expected(table, set(ALL_FAMILIES))
        assert 0 < counts["race_unmatched_rate"] < 1 and 0 < counts["horse_unmatched_rate"] < 1
        assert rows and counts["table_rows"] < counts["rows"]
    pre, _ = expected(PRE, set(ALL_FAMILIES))
    assert {r["matched_by"] for r in pre} == {"horse_id", "horse_no"}
    r1_6 = next(r for r in pre if r["race_id"] == R1.race_id and r["horse_no"] == 6)
    assert r1_6["kyi_idm"] is None and r1_6["joa_idm"] is not None and r1_6["jrdb_horse_id"]


def test_synthetic_archives_build_both_tables(tmp_path: Path, monkeypatch) -> None:
    """Day-file archives of all six families -> both tables, keyed and named as the contract says,
    and jrdb-build.json with each table's row counts and join rates."""
    world(tmp_path, monkeypatch)
    write_day_files(ALL_FAMILIES)
    build_ok()
    for table in (PRE, POST):
        assert_table(table, set(ALL_FAMILIES))

    report = json.loads((paths.reports_dir() / "jrdb-build.json").read_text(encoding="utf-8"))
    for table in (PRE, POST):
        _, counts = expected(table, set(ALL_FAMILIES))
        got = report[table]
        assert (got["rows"], got["table_rows"]) == (counts["rows"], counts["table_rows"]), table
        for rate in ("race_unmatched_rate", "horse_unmatched_rate"):
            assert got[rate] == pytest.approx(counts[rate], abs=1e-6), (table, rate)

    # A day file held as a bare .txt reads the same as one inside a .zip.
    other = world(tmp_path / "txt", monkeypatch)
    for family in ALL_FAMILIES:
        for day in (DAY1, DAY2):
            path = other / "raw" / "jrdb" / family / f"{delivery(family, day)}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(day_records(family, day))
    build_ok()
    for table in (PRE, POST):
        assert_table(table, set(ALL_FAMILIES))


@pytest.mark.parametrize(
    "delivered",
    [("KYI", "SED"), ("TYB", "SKB"), ("CYB", "JOA", "SED"), ("KYI", "CYB", "JOA", "SED", "SKB")],
    ids="+".join,
)
def test_a_missing_family_still_has_its_columns(
    tmp_path: Path, monkeypatch, delivered: tuple[str, ...]
) -> None:
    """Families not delivered (no folder, or an empty one) keep their `<fam>_<column>` columns,
    all null; the rows are those of the families delivered."""
    data = world(tmp_path, monkeypatch)
    write_day_files(delivered)
    missing = [f for f in ALL_FAMILIES if f not in delivered]
    (data / "raw" / "jrdb" / missing[0]).mkdir(parents=True)  # an empty folder is no delivery
    build_ok()
    for table in (PRE, POST):
        assert_table(table, set(delivered))
        frame = built(table)
        for family in TABLE_FAMILIES[table]:
            if family in missing:
                for column in DATA_COLUMNS[family]:
                    assert frame[f"{family.lower()}_{column}"].isna().all(), (family, column)


@pytest.mark.parametrize(
    ("pack", "day_suffix", "day_file_older"),
    [
        ("KYI_2026.zip", ".zip", True),
        ("KYI2026.zip", ".zip", False),
        ("KYI2026.zip", ".txt", True),
        ("KYI_2026.zip", ".txt", False),
    ],
)
def test_a_day_file_wins_over_a_year_pack(
    tmp_path: Path, monkeypatch, pack: str, day_suffix: str, day_file_older: bool
) -> None:
    """KYI261128 is held twice: as a day file and in a year pack whose copy differs (other values,
    one more runner). The day file's copy is read, the pack's copy not at all; the pack's other
    member (KYI261129) is still read. Pack names sort before and after the day file, and the day
    file is older or newer than the pack, so neither name order nor file age decides it."""
    data = world(tmp_path, monkeypatch)
    write_day_files(["SED"])
    folder = data / "raw" / "jrdb" / "KYI"
    pack_path = folder / pack
    zip_of(
        pack_path,
        {
            f"{delivery('KYI', DAY1)}.txt": day_records("KYI", DAY1, stale=True, extra=True),
            f"{delivery('KYI', DAY2)}.txt": day_records("KYI", DAY2),
        },
    )
    day_path = folder / f"{delivery('KYI', DAY1)}{day_suffix}"
    if day_suffix == ".zip":
        zip_of(day_path, {f"{delivery('KYI', DAY1)}.txt": day_records("KYI", DAY1)})
    else:
        day_path.write_bytes(day_records("KYI", DAY1))
    stamp = 1_700_000_000
    os.utime(day_path, (stamp, stamp) if day_file_older else (stamp + 86400, stamp + 86400))
    os.utime(pack_path, (stamp + 86400, stamp + 86400) if day_file_older else (stamp, stamp))

    build_ok()
    assert_table(PRE, {"KYI", "SED"})
    assert_table(POST, {"KYI", "SED"})
    frame = built(PRE)
    assert not frame["kyi_mark"].eq("99").any()  # nothing from the pack's copy of KYI261128
    assert frame["race_id"].eq(R11.race_id).any()  # KYI261129, held only in the pack, is read


@pytest.mark.parametrize("lzh_only", [("SKB",), ("KYI",), ("TYB", "SED")], ids="+".join)
def test_an_lzh_only_family_is_one_error_line(
    tmp_path: Path, monkeypatch, lzh_only: tuple[str, ...]
) -> None:
    """A family held only as .lzh stops the build: exit 1, exactly one output line, an ERROR line
    naming each such family; no table or report is written or replaced. `run()` reports the same
    as its error. A .lzh beside a readable .zip of the same family is skipped, not an error."""
    data = world(tmp_path, monkeypatch)
    readable = [f for f in ALL_FAMILIES if f not in lzh_only]
    write_day_files(readable)
    for family in lzh_only:
        folder = data / "raw" / "jrdb" / family
        folder.mkdir(parents=True)
        (folder / f"{family}_2026.lzh").write_bytes(b"-lh5- not a zip " + family.encode())
        (folder / f"{delivery(family, DAY1)}.lzh").write_bytes(b"-lh5- day " + family.encode())
    outputs = [
        paths.processed_dir("jra") / f"{PRE}.parquet",
        paths.processed_dir("jra") / f"{POST}.parquet",
        paths.reports_dir() / "jrdb-build.json",
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"left from an earlier build " + path.name.encode())
    before = {path: path.read_bytes() for path in outputs}

    result = invoke_build()
    assert result.exit_code == 1, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1, result.output
    assert lines[0].startswith("ERROR "), result.output
    for family in lzh_only:
        assert family in lines[0], (family, lines[0])
    assert {path: path.read_bytes() for path in outputs} == before

    outcome = tables.run()
    assert outcome.error and all(family in outcome.error for family in lzh_only)
    assert {path: path.read_bytes() for path in outputs} == before

    # The same family with a readable .zip beside its .lzh builds, the .lzh skipped.
    for path in outputs:
        path.unlink()
    write_day_files(lzh_only)
    build_ok()
    for table in (PRE, POST):
        assert_table(table, set(ALL_FAMILIES))


# ------------------------------------------------------------------------------------------------
# Ticket #186: date columns, upcoming cards, UKC
# ------------------------------------------------------------------------------------------------


def report_json() -> dict[str, Any]:
    return json.loads((paths.reports_dir() / "jrdb-build.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "delivered",
    [tuple(ALL_FAMILIES), ("KYI", "SED"), ("TYB", "SKB"), ("KYI", "SKB"), ("CYB", "SED")],
    ids="+".join,
)
def test_a_date_column_is_date32_delivered_or_missing(
    tmp_path: Path, monkeypatch, delivered: tuple[str, ...]
) -> None:
    """KYI's card_date and SED's run_date are date-kind fields: in the written file each is a
    date32 column, both when its family is delivered (dates, and None where the field is blank)
    and when it is not (all null)."""
    world(tmp_path, monkeypatch)
    write_day_files(delivered)
    build_ok()
    for table in (PRE, POST):
        assert_table(table, set(delivered))
        column = DATE_COLUMNS[table]
        path = paths.processed_dir("jra") / f"{table}.parquet"
        assert pq.read_schema(path).field(column).type == pa.date32(), (table, column)
        values = [_plain(v) for v in built(table)[column]]
        family = column.split("_", 1)[0].upper()
        if family in delivered:
            assert any(isinstance(v, date) for v in values) and None in values, (column, values)
        else:
            assert values and all(v is None for v in values), (column, values)


def test_the_upcoming_world_is_what_the_tests_claim() -> None:
    """Guards the fixtures: the card races keep rows matched by id and by number and drop a
    mismatched id and an unknown number; the race whose folder has no runners keeps none."""
    rows, counts = expected(PRE, set(ALL_FAMILIES), UPCOMING)
    kept = {(r["race_id"], r["horse_no"]): r for r in rows}
    assert {r["matched_by"] for r in rows} == {"horse_id", "horse_no"}
    assert {r["race_id"] for r in rows} == {u.race_id for u in CARD_RACES}
    assert (U2.race_id, 3) not in kept and (U1.race_id, len(U1.horses) + 1) not in kept
    assert kept[(U1.race_id, 5)]["matched_by"] == "horse_no"
    assert kept[(U11.race_id, 6)]["kyi_idm"] is None and kept[(U11.race_id, 6)]["jrdb_horse_id"]
    assert all(r["race_date"] in (DAY3, DAY4) for r in rows)
    assert counts["table_rows"] < counts["rows"]


def test_upcoming_card_races_get_their_pre_race_rows(tmp_path: Path, monkeypatch) -> None:
    """Evening-before KYI/CYB/JOA/TYB for two upcoming meetings whose cards carry no venue or
    race_date (absent at one meeting, null at the other) reach jrdb_pre_race, matched to the
    card's runners, with race_id from the card and race_date from its meeting_id, beside the
    history rows. A meeting folder holding races.parquet but no racecard_runners.parquet adds
    nothing and does not stop the build. jrdb_post_race is the history's alone."""
    world(tmp_path, monkeypatch)
    write_day_files(ALL_FAMILIES)
    write_day_files(TABLE_FAMILIES[PRE], days=(DAY3, DAY4), races=UPCOMING)
    write_card([U1, U2], keys="absent")
    write_card([U11], keys="null")
    write_card([U_HALF], runners=False)
    build_ok()

    races = (*RACES, *UPCOMING)
    assert_table(PRE, set(ALL_FAMILIES), races)
    assert_table(POST, set(ALL_FAMILIES))
    frame = built(PRE)
    for race in CARD_RACES:
        assert frame["race_id"].eq(race.race_id).any(), race.race_id
    assert not frame["race_id"].eq(U_HALF.race_id).any()

    report = report_json()
    _, counts = expected(PRE, set(ALL_FAMILIES), races)
    got = report[PRE]
    assert (got["rows"], got["table_rows"]) == (counts["rows"], counts["table_rows"])


def test_history_wins_over_a_stale_card(tmp_path: Path, monkeypatch) -> None:
    """A card left on disk for a meeting already in history, naming other horses at R1's 1 and
    R2's 2, changes nothing: both tables are as from history alone."""
    world(tmp_path, monkeypatch)
    write_day_files(ALL_FAMILIES)
    stale_r1 = replace(R1, horses=("2025109001", *R1.horses[1:]))
    stale_r2 = replace(R2, horses=(R2.horses[0], "2025109002", *R2.horses[2:]))
    write_card([stale_r1, stale_r2])
    build_ok()
    for table in (PRE, POST):
        assert_table(table, set(ALL_FAMILIES))


def write_netkeiba_pedigree() -> Path:
    """A netkeiba-built horse_pedigree (invented horses and names)."""
    rows = [
        {
            "jurisdiction": "jra",
            "horse_id": hid,
            "sire_name": f"シンセティック{i}",
            "sire_name_norm": f"シンセティック{i}",
            "dam_name": f"ダミー{i}",
            "dam_name_norm": f"ダミー{i}",
            "source_race_id": R1.race_id,
        }
        for i, hid in enumerate(R1.horses[:3], 1)
    ]
    write_store_table("horse_pedigree", rows)
    return paths.processed_dir("jra") / "horse_pedigree.parquet"


@pytest.mark.parametrize(
    ("ukc", "netkeiba"),
    [("zip", True), ("txt", True), ("lzh", True), ("zip", False), ("lzh", False)],
)
def test_a_ukc_archive_is_skipped_with_its_reason(
    tmp_path: Path, monkeypatch, ukc: str, netkeiba: bool
) -> None:
    """Ruling: the build skips UKC (#209 owns the horse-master merge). With a UKC archive present
    (a .zip, a bare .txt, or only .lzh) the build completes, both JRDB tables are as without it,
    horse_pedigree.parquet is untouched (a netkeiba-built one byte for byte and mtime, none stays
    none), and the skip is stated in jrdb-build.json and in the output, naming UKC and #209."""
    data = world(tmp_path, monkeypatch)
    write_day_files(ALL_FAMILIES)
    folder = data / "raw" / "jrdb" / "UKC"
    folder.mkdir(parents=True)
    records = b"".join(f"{hid[2:]}synthetic horse master\r\n".encode() for hid in R1.horses)
    if ukc == "zip":
        zip_of(folder / "UKC261128.zip", {"UKC261128.txt": records})
    elif ukc == "txt":
        (folder / "UKC261128.txt").write_bytes(records)
    else:
        (folder / "UKC_2026.lzh").write_bytes(b"-lh5- not a zip UKC")
    pedigree_path = paths.processed_dir("jra") / "horse_pedigree.parquet"
    if netkeiba:
        write_netkeiba_pedigree()
        stamp = 1_700_000_000
        os.utime(pedigree_path, (stamp, stamp))
        before = (pedigree_path.read_bytes(), pedigree_path.stat().st_mtime_ns)

    result = invoke_build()
    assert result.exit_code == 0, result.output
    assert "ERROR" not in result.output, result.output
    for table in (PRE, POST):
        assert_table(table, set(ALL_FAMILIES))

    if netkeiba:
        assert (pedigree_path.read_bytes(), pedigree_path.stat().st_mtime_ns) == before
    else:
        assert not pedigree_path.exists()

    reason = report_json()["horse_pedigree"]["skipped"]
    assert isinstance(reason, str) and "UKC" in reason and "209" in reason, reason
    lines = result.output.splitlines()
    assert any("UKC" in line and "209" in line for line in lines), result.output
