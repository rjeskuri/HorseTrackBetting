"""JV-Data HR (payout) records into the jravan store's `dividends` table (ticket #366).

Contract pinned here (the coder follows it), module `horsetrack.countries.japan.jravan.dividends`:
- `dividends_from_records(cols: Mapping[str, numpy.ndarray]) -> pandas.DataFrame`: `cols` is
  `records.decode` of HR records; one row per paid combination, in the columns of
  `schemas.TABLES['dividends']`, in that order.
- `build_dividends(*, years=None, raw=None, out=None) -> records.BuildReport`: reads
  `<raw>/stored/RACE/HR/<YYYY>.jvd` (`raw` is the export root, default
  `paths.raw_dir('jravan')`; `years` None reads every year file), with the layout
  `spec.record_spec('HR')`; keeps one version per race and JRA venues only; writes
  `<out>/dividends.parquet` (`out` a folder, default `paths.processed_dir('jravan')`) through
  `records.write_parquet`, in the declared column order and `tables.ARROW_TYPES`, stable-sorted
  by `race_id`. The report's name is `dividends`.
- `horsetrack japan jravan dividends`: the same build, then `records.write_build_report`
  (`<reports dir>/jravan-dividends.json`).

The workbook items each column comes from (HR, the JV-Data 4.9.0.1 workbook on this Mac; names
as `spec.record_spec` returns them, keys as `records.decode` builds them):
- `jurisdiction`: `jra`. Only records whose `競馬場コード` is a JRA venue (01-10) are kept.
- `race_id`: `keys.race_id(開催年, 開催月日, 競馬場コード, レース番号)`.
- one version per race: `records.latest_versions` on `開催年`, `開催月日`, `競馬場コード`,
  `開催回[第N回]`, `開催日目[N日目]`, `レース番号` (`データ作成年月日`, `データ区分`; a race whose
  latest record is a deletion, `データ区分` 0, has no rows).
- `pool`, `combination`, `dividend`: one row per slot of each payout group, the groups in this
  order (the `pool` codes of `config/jra.yaml`), the slots in the record's order:
    `単勝払戻/馬番` WIN, `複勝払戻/馬番` PLACE, `枠連払戻/組番` BRACKET_QUINELLA,
    `馬連払戻/組番` QUINELLA, `ワイド払戻/組番` QUINELLA_PLACE, `馬単払戻/組番` FORECAST,
    `3連複払戻/組番` TRIO, `3連単払戻/組番` TIERCE;
  `dividend` is the slot's `<group>/払戻金` (yen per 100, as printed), `<group>/人気順` unused.
  A `組番` holds the legs (1, 2 or 3) as equal-width numbers (the item's width / legs; a bracket
  is one digit, a horse two). `combination` joins them with `,`, leading zeros dropped:
  FORECAST and TIERCE in the printed (finishing) order, the others ascending by number.
- `is_refund`: false on every row.
- A slot writes a row only when every number of its combination and its payout are positive
  integers: blank slots, zero combinations (an unsold pool prints zeros and a blank payout) and
  zero or blank payouts write nothing. An unsold pool has no rows, never zeros.
- The pool flags `不成立フラグ<P>` (not established), `特払フラグ<P>` (special payout) and
  `返還フラグ<P>` (refund), P one of 単勝 複勝 枠連 馬連 ワイド 馬単 3連複 3連単, are set when
  `1`. A pool flagged not established or special payout writes no row (its slots hold zeros and
  the amount paid back). A refund flag writes no row of its own: the pool's paid combinations
  are still rows. `返還馬番情報(馬番01～28)` (one `1` per refunded horse number, in horse order)
  writes no row.
- The report counts, per `開催年` (`by_year[year]`) and in total (`counts`), over the kept JRA
  records: `not_established`, `special_payout` and `refund` (races x pools flagged) and
  `refunded_horses` (horse numbers set); `counts['rows']` is the rows written.

Every layout here is made up in the workbook's shape (`jravan_workbook.write_workbook`): the item
names are the workbook's, the widths of the payouts and popularity, every repeat count and so every
position are invented, so nothing may be sliced by hand. Every record and value is invented too.
Expected rows are written out by hand.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.store import schemas, tables
from horsetrack.core.store.asof import load_availability, read_asof, read_label

# isort: split
# (its own block: the sibling helper module, then the module this ticket creates)
import jravan_workbook as wb

from horsetrack.countries.japan.jravan import dividends, records, spec

SOURCE = "jravan"
UTC = timezone.utc
US = timedelta(microseconds=1)
COLUMNS = list(schemas.TABLES["dividends"].columns)

CODES = (
    "WIN", "PLACE", "BRACKET_QUINELLA", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO", "TIERCE",
)  # fmt: skip
JV_NAMES = ("単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単")
JV_POOL = dict(zip(CODES, JV_NAMES, strict=True))
FLAGS = {"not_established": "不成立", "special_payout": "特払", "refund": "返還"}
REFUNDED_HORSES = "返還馬番情報(馬番01～28)"
VENUES = {  # number -> short code, as config/jra.yaml lists them
    "01": "SAP", "02": "HAK", "03": "FKS", "04": "NII", "05": "TOK",
    "06": "NAK", "07": "CKY", "08": "KYO", "09": "HSN", "10": "KOK",
}  # fmt: skip

# ------------------------------------------------------------------------------------------------
# A made-up HR layout: the workbook's item names, invented widths and repeats


PAY = 8  # bytes of 払戻金 (invented)
POP = 3  # bytes of 人気順 (invented)
HORSE_SLOTS = 18  # repeat of the refunded-horse item (invented)


@dataclass(frozen=True)
class Payout:
    group: str
    number: str
    width: int  # bytes of the number item: legs x digits
    repeat: int  # slots (invented)


PAYOUTS = {
    "WIN": Payout("単勝払戻", "馬番", 2, 3),
    "PLACE": Payout("複勝払戻", "馬番", 2, 4),
    "BRACKET_QUINELLA": Payout("枠連払戻", "組番", 2, 2),
    "QUINELLA": Payout("馬連払戻", "組番", 4, 3),
    "QUINELLA_PLACE": Payout("ワイド払戻", "組番", 4, 5),
    "FORECAST": Payout("馬単払戻", "組番", 4, 3),
    "TRIO": Payout("3連複払戻", "組番", 6, 2),
    "TIERCE": Payout("3連単払戻", "組番", 6, 4),
}
GROUP_POOL = {p.group: code for code, p in PAYOUTS.items()}

HEAD = (
    ("レコード種別ID", 2),
    ("データ区分", 1),
    ("データ作成年月日", 8),
    ("開催年", 4),
    ("開催月日", 4),
    ("競馬場コード", 2),
    ("開催回[第N回]", 2),
    ("開催日目[N日目]", 2),
    ("レース番号", 2),
    ("登録頭数", 2),
    ("出走頭数", 2),
)


@dataclass(frozen=True)
class Field:
    name: str
    length: int
    repeat: int = 1
    subs: tuple[tuple[str, int], ...] = ()

    @property
    def width(self) -> int:
        return sum(n for _, n in self.subs) if self.subs else self.length


def layout() -> tuple[Field, ...]:
    out = [Field(name, length) for name, length in HEAD]
    for kind in FLAGS.values():
        for code in CODES:
            out.append(Field(f"{kind}フラグ{JV_POOL[code]}", 1))
            if code == "QUINELLA_PLACE":
                out.append(Field("予備", 1))
    out += [
        Field(REFUNDED_HORSES, 1, HORSE_SLOTS),
        Field("返還枠番情報(枠番1～8)", 1, 8),
        Field("返還同枠情報(枠番1～8)", 1, 8),
    ]
    for code in CODES:
        p = PAYOUTS[code]
        out.append(
            Field(p.group, 0, p.repeat, ((p.number, p.width), ("払戻金", PAY), ("人気順", POP)))
        )
        if code == "QUINELLA_PLACE":
            out.append(Field("予備", 0, 2, (("予備", 4), ("予備", PAY), ("予備", POP))))
    out.append(Field("レコード区切", 2))
    return tuple(out)


def hr_section(number: int = 2) -> wb.Section:
    rows: list[wb.Row] = []
    pos = 1
    for no, f in enumerate(layout(), start=1):
        if f.subs:
            rows.append(wb.group(no, f"<{f.name}>", pos, f.width, f.repeat, desc="払戻（架空）"))
            rel = 1
            for i, (name, length) in enumerate(f.subs):
                rows.append(wb.sub(wb.letter_code(i), name, rel, length, desc="値（架空）"))
                rel += length
            pos += f.width * f.repeat
            continue
        if no == 1:
            rows.append(
                wb.item(1, f.name, 1, f.length, key="●", initial="HR", desc=wb.id_desc("HR"))
            )
        else:
            repeat = f.repeat if f.repeat > 1 else None
            rows.append(wb.item(no, f.name, pos, f.length, repeat=repeat, desc="（架空）"))
        pos += f.length * f.repeat
    return wb.Section(number, "払戻（架空）", pos - 1, tuple(rows))


def write_hr_workbook(path: Path) -> Path:
    decoy = wb.simple_section(1, "XQ", "架空その他", [("架空項目", 6)])
    return wb.write_workbook(path, [decoy, hr_section(2)], [])


def amount(yen: int) -> str:
    return f"{yen:0{PAY}d}"


def paid(number: str, yen: int) -> tuple[str, str]:
    """A paid slot: the number item as printed and the payout."""
    return number, amount(yen)


@dataclass
class HR:
    """One made-up HR record; unused slots are blank, flags and refund items `0`."""

    year: str = "2024"
    mmdd: str = "0601"
    venue: str = "05"
    race: str = "11"
    kai: str = "03"
    nichi: str = "02"
    kubun: str = "2"
    created: str = "20240603"
    pays: Mapping[str, Sequence[tuple[str, str]]] = field(default_factory=dict)
    flags: Mapping[str, Sequence[str]] = field(default_factory=dict)  # FLAGS key -> pool codes
    refunded: Sequence[int] = ()
    refunded_brackets: Sequence[int] = ()

    def record(self) -> bytes:
        flat = dict(
            zip(
                [name for name, _ in HEAD],
                ("HR", self.kubun, self.created, self.year, self.mmdd, self.venue, self.kai,
                 self.nichi, self.race, "16", "16"),
                strict=True,
            )
        )  # fmt: skip
        for key, kind in FLAGS.items():
            for code in CODES:
                flat[f"{kind}フラグ{JV_POOL[code]}"] = (
                    "1" if code in self.flags.get(key, ()) else "0"
                )
        out: list[str] = []
        for f in layout()[:-1]:
            if f.subs:
                slots = list(self.pays.get(GROUP_POOL[f.name], ())) if f.name != "予備" else []
                assert len(slots) <= f.repeat, (f.name, slots)
                for r in range(f.repeat):
                    number, pay = slots[r] if r < len(slots) else ("", "")
                    pop = f"{r + 1:0{POP}d}" if r < len(slots) else ""
                    for (_, length), value in zip(f.subs, (number, pay, pop), strict=True):
                        out.append(fit(value, length))
            elif f.name == REFUNDED_HORSES:
                out.append(
                    "".join("1" if h in self.refunded else "0" for h in range(1, HORSE_SLOTS + 1))
                )
            elif f.name.startswith("返還枠番"):
                out.append(
                    "".join("1" if b in self.refunded_brackets else "0" for b in range(1, 9))
                )
            elif f.name.startswith("返還同枠"):
                out.append("0" * 8)
            elif f.name == "予備":
                out.append(" " * f.length)
            else:
                out.append(fit(flat[f.name], f.length))
        body = "".join(out).encode("cp932")
        assert len(body) == hr_section().length - 2
        return body + b"\r\n"


def fit(value: str, length: int) -> str:
    """A field as printed: `value` at the field's exact width, or blank."""
    if value and len(value) != length:
        raise ValueError(f"{value!r} is not {length} wide")
    return value or " " * length


# ------------------------------------------------------------------------------------------------
# Helpers


@pytest.fixture(scope="module")
def hr_layout(tmp_path_factory: pytest.TempPathFactory) -> spec.RecordSpec:
    path = write_hr_workbook(tmp_path_factory.mktemp("workbook") / wb.WORKBOOK_NAME)
    return spec.record_spec("HR", path)


def from_records(layout_: spec.RecordSpec, races: Sequence[HR]) -> pd.DataFrame:
    cols = records.decode(b"".join(r.record() for r in races), layout_)
    return dividends.dividends_from_records(cols)


Row = tuple[str, str, str, float]


def rows_of(frame: pd.DataFrame) -> list[Row]:
    """(race_id, pool, combination, dividend) per row, after checking the fixed columns."""
    assert list(frame.columns) == COLUMNS
    assert frame["jurisdiction"].tolist() == ["jra"] * len(frame)
    assert [x is False or x is np.False_ for x in frame["is_refund"].tolist()] == [True] * len(
        frame
    ), frame["is_refund"].tolist()
    return [
        (r, p, c, float(d))
        for r, p, c, d in zip(
            frame["race_id"], frame["pool"], frame["combination"], frame["dividend"], strict=True
        )
    ]


def by_race(rows: Sequence[Row]) -> list[Row]:
    """Rows stable-sorted by race_id: the in-race order is kept, whatever the race order."""
    return sorted(rows, key=lambda row: row[0])


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data dir with the made-up workbook and an empty export."""
    root = tmp_path / "data"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(root))
    write_hr_workbook(root / "specs" / "jravan" / wb.WORKBOOK_NAME)
    (root / "raw" / SOURCE / "stored" / "RACE" / "HR").mkdir(parents=True)
    return root


def write_year(data: Path, year: int, races: Sequence[HR]) -> Path:
    path = data / "raw" / SOURCE / "stored" / "RACE" / "HR" / f"{year}.jvd"
    return wb.write_jvd(path, [r.record() for r in races])


def store_rows() -> list[Row]:
    return rows_of(tables.read_table(SOURCE, "dividends"))


# ------------------------------------------------------------------------------------------------
# dividends_from_records


def test_each_paid_combination_is_one_row_with_the_jra_pool_codes(
    hr_layout: spec.RecordSpec,
) -> None:
    a = HR(
        year="2024", mmdd="0601", venue="05", race="11",
        pays={
            "WIN": [paid("04", 350)],
            "PLACE": [paid("04", 140), paid("12", 310), paid("07", 160)],
            "BRACKET_QUINELLA": [paid("26", 1230)],
            "QUINELLA": [paid("0412", 2450)],
            "QUINELLA_PLACE": [paid("0412", 820), paid("0407", 450), paid("0712", 1310)],
            "FORECAST": [paid("0412", 4380)],
            "TRIO": [paid("040712", 6720)],
            "TIERCE": [paid("041207", 35890)],
        },
    )  # fmt: skip
    b = HR(
        year="2019", mmdd="1228", venue="06", race="09", kai="05", nichi="09", created="20191230",
        pays={
            "WIN": [paid("16", 12340)],
            "PLACE": [paid("16", 2870), paid("01", 190), paid("09", 650)],
            "BRACKET_QUINELLA": [paid("18", 5560)],
            "QUINELLA": [paid("0116", 18900)],
            "QUINELLA_PLACE": [paid("0116", 5230), paid("0916", 14320), paid("0109", 2110)],
            "FORECAST": [paid("1601", 61250)],
            "TRIO": [paid("010916", 152340)],
            "TIERCE": [paid("160109", 98765430)],
        },
    )  # fmt: skip
    ra, rb = "jra:2024-06-01:TOK:11", "jra:2019-12-28:NAK:09"
    expected = [
        (ra, "WIN", "4", 350.0),
        (ra, "PLACE", "4", 140.0),
        (ra, "PLACE", "12", 310.0),
        (ra, "PLACE", "7", 160.0),
        (ra, "BRACKET_QUINELLA", "2,6", 1230.0),
        (ra, "QUINELLA", "4,12", 2450.0),
        (ra, "QUINELLA_PLACE", "4,12", 820.0),
        (ra, "QUINELLA_PLACE", "4,7", 450.0),
        (ra, "QUINELLA_PLACE", "7,12", 1310.0),
        (ra, "FORECAST", "4,12", 4380.0),
        (ra, "TRIO", "4,7,12", 6720.0),
        (ra, "TIERCE", "4,12,7", 35890.0),
        (rb, "WIN", "16", 12340.0),
        (rb, "PLACE", "16", 2870.0),
        (rb, "PLACE", "1", 190.0),
        (rb, "PLACE", "9", 650.0),
        (rb, "BRACKET_QUINELLA", "1,8", 5560.0),
        (rb, "QUINELLA", "1,16", 18900.0),
        (rb, "QUINELLA_PLACE", "1,16", 5230.0),
        (rb, "QUINELLA_PLACE", "9,16", 14320.0),
        (rb, "QUINELLA_PLACE", "1,9", 2110.0),
        (rb, "FORECAST", "16,1", 61250.0),
        (rb, "TRIO", "1,9,16", 152340.0),
        (rb, "TIERCE", "16,1,9", 98765430.0),
    ]
    # Every JRA venue, each race paying another horse another amount.
    races = [a, b]
    for i, (number, code) in enumerate(VENUES.items()):
        races.append(
            HR(year="2010", mmdd="0815", venue=number, race=f"{i + 1:02d}", created="20100816",
               pays={"WIN": [paid(f"{i + 3:02d}", 100 + 70 * i)]})
        )  # fmt: skip
        expected.append((f"jra:2010-08-15:{code}:{i + 1:02d}", "WIN", str(i + 3), 100.0 + 70 * i))

    frame = from_records(hr_layout, races)
    got = rows_of(frame)
    assert by_race(got) == by_race(expected)
    assert {row[1] for row in got} == set(CODES)


ORDER_CASES = [  # (pool, number item as printed, combination)
    ("BRACKET_QUINELLA", "53", "3,5"),
    ("BRACKET_QUINELLA", "77", "7,7"),
    ("BRACKET_QUINELLA", "18", "1,8"),
    ("QUINELLA", "1203", "3,12"),
    ("QUINELLA", "0901", "1,9"),
    ("QUINELLA", "1018", "10,18"),
    ("QUINELLA_PLACE", "1602", "2,16"),
    ("QUINELLA_PLACE", "0510", "5,10"),
    ("FORECAST", "1203", "12,3"),
    ("FORECAST", "0102", "1,2"),
    ("FORECAST", "0910", "9,10"),
    ("FORECAST", "1009", "10,9"),
    ("TRIO", "120309", "3,9,12"),
    ("TRIO", "100201", "1,2,10"),
    ("TRIO", "011805", "1,5,18"),
    ("TIERCE", "120309", "12,3,9"),
    ("TIERCE", "100201", "10,2,1"),
    ("TIERCE", "011805", "1,18,5"),
    ("WIN", "09", "9"),
    ("PLACE", "10", "10"),
]


def test_ordered_pools_keep_finishing_order_and_the_others_sort(
    hr_layout: spec.RecordSpec,
) -> None:
    races, expected = [], []
    for i, (pool, printed, combination) in enumerate(ORDER_CASES):
        race = f"{i + 1:02d}"
        races.append(
            HR(mmdd="0915", venue="07", race=race, pays={pool: [paid(printed, 200 + 30 * i)]})
        )
        expected.append((f"jra:2024-09-15:CKY:{race}", pool, combination, 200.0 + 30 * i))
    assert by_race(rows_of(from_records(hr_layout, races))) == by_race(expected)


def test_dead_heat_winners_each_get_a_row(hr_layout: spec.RecordSpec) -> None:
    two = HR(
        mmdd="0602", venue="05", race="05",
        pays={
            "WIN": [paid("03", 230), paid("08", 410)],
            "PLACE": [paid("03", 110), paid("08", 150), paid("11", 320), paid("05", 290)],
            "QUINELLA": [paid("0308", 680)],
            "QUINELLA_PLACE": [
                paid("0308", 250), paid("0311", 900), paid("0811", 1120), paid("0305", 870),
                paid("0508", 1040),
            ],
            "FORECAST": [paid("0308", 1150), paid("0803", 1420)],
            "TRIO": [paid("030811", 2300), paid("030508", 2010)],
            "TIERCE": [
                paid("030811", 9870), paid("080311", 11230), paid("030805", 8640),
                paid("080305", 10110),
            ],
        },
    )  # fmt: skip
    three = HR(
        mmdd="0602", venue="05", race="06",
        pays={"WIN": [paid("02", 510), paid("14", 620), paid("07", 380)]},
    )  # fmt: skip
    r5, r6 = "jra:2024-06-02:TOK:05", "jra:2024-06-02:TOK:06"
    expected = [
        (r5, "WIN", "3", 230.0),
        (r5, "WIN", "8", 410.0),
        (r5, "PLACE", "3", 110.0),
        (r5, "PLACE", "8", 150.0),
        (r5, "PLACE", "11", 320.0),
        (r5, "PLACE", "5", 290.0),
        (r5, "QUINELLA", "3,8", 680.0),
        (r5, "QUINELLA_PLACE", "3,8", 250.0),
        (r5, "QUINELLA_PLACE", "3,11", 900.0),
        (r5, "QUINELLA_PLACE", "8,11", 1120.0),
        (r5, "QUINELLA_PLACE", "3,5", 870.0),
        (r5, "QUINELLA_PLACE", "5,8", 1040.0),
        (r5, "FORECAST", "3,8", 1150.0),
        (r5, "FORECAST", "8,3", 1420.0),
        (r5, "TRIO", "3,8,11", 2300.0),
        (r5, "TRIO", "3,5,8", 2010.0),
        (r5, "TIERCE", "3,8,11", 9870.0),
        (r5, "TIERCE", "8,3,11", 11230.0),
        (r5, "TIERCE", "3,8,5", 8640.0),
        (r5, "TIERCE", "8,3,5", 10110.0),
        (r6, "WIN", "2", 510.0),
        (r6, "WIN", "14", 620.0),
        (r6, "WIN", "7", 380.0),
    ]
    assert by_race(rows_of(from_records(hr_layout, [two, three]))) == by_race(expected)


def test_empty_slots_write_no_row(hr_layout: spec.RecordSpec) -> None:
    blank = ""
    # A year before Quinella, Exacta, Trio and Trifecta: those pools print zeros or nothing.
    unsold = HR(
        year="1990", mmdd="0506", venue="05", race="10", created="19900507",
        pays={
            "WIN": [paid("05", 420)],
            "PLACE": [paid("05", 160), paid("09", 220), paid("02", 180)],
            "BRACKET_QUINELLA": [paid("35", 990)],
            "QUINELLA": [("0000", blank)],
            "FORECAST": [("0000", blank)],
            "TRIO": [("000000", blank)],
        },
    )  # fmt: skip
    # A small field: Place pays two, its third slot prints zeros.
    small = HR(
        year="1990", mmdd="0506", venue="05", race="11", created="19900507",
        pays={"WIN": [paid("06", 250)], "PLACE": [paid("06", 130), paid("01", 170), ("00", blank)]},
    )  # fmt: skip
    # Each other empty form next to a paid slot.
    forms = HR(
        year="1990", mmdd="0506", venue="05", race="12", created="19900507",
        pays={
            "WIN": [paid("07", 560), ("00", amount(0)), ("08", blank)],
            "PLACE": [("03", amount(0)), paid("07", 190), ("  ", amount(150))],
            "QUINELLA": [("0700", amount(880))],
            "QUINELLA_PLACE": [paid("0307", 300), ("    ", amount(500)), ("0307", blank)],
            "FORECAST": [("0703", blank)],
            "TRIO": [("000000", amount(7000))],
            "TIERCE": [paid("070308", 12340)],
        },
    )  # fmt: skip
    # A race with no payout at all.
    nothing = HR(year="1990", mmdd="0506", venue="05", race="09", created="19900507")
    r10, r11, r12 = (f"jra:1990-05-06:TOK:{n}" for n in ("10", "11", "12"))
    expected = [
        (r10, "WIN", "5", 420.0),
        (r10, "PLACE", "5", 160.0),
        (r10, "PLACE", "9", 220.0),
        (r10, "PLACE", "2", 180.0),
        (r10, "BRACKET_QUINELLA", "3,5", 990.0),
        (r11, "WIN", "6", 250.0),
        (r11, "PLACE", "6", 130.0),
        (r11, "PLACE", "1", 170.0),
        (r12, "WIN", "7", 560.0),
        (r12, "PLACE", "7", 190.0),
        (r12, "QUINELLA_PLACE", "3,7", 300.0),
        (r12, "TIERCE", "7,3,8", 12340.0),
    ]
    got = rows_of(from_records(hr_layout, [unsold, small, forms, nothing]))
    assert by_race(got) == by_race(expected)

    # Only blank records: an empty frame with the declared columns.
    empty = from_records(hr_layout, [nothing])
    assert list(empty.columns) == COLUMNS
    assert len(empty) == 0


# ------------------------------------------------------------------------------------------------
# build_dividends


def build_races() -> dict[int, list[HR]]:
    """Two year files, the races out of race_id order, one NAR race."""
    return {
        2024: [
            HR(year="2024", mmdd="0602", venue="05", race="11", created="20240604",
               pays={"WIN": [paid("04", 350)], "PLACE": [paid("04", 140), paid("12", 310)],
                     "TIERCE": [paid("041207", 35890)]}),
            HR(year="2024", mmdd="0602", venue="08", race="02", created="20240604",
               pays={"WIN": [paid("12", 2210)], "QUINELLA_PLACE": [paid("0612", 990)]}),
            HR(year="2024", mmdd="0602", venue="30", race="05", created="20240604",
               pays={"WIN": [paid("03", 500)]}),
            HR(year="2024", mmdd="0601", venue="05", race="01", created="20240603",
               pays={"WIN": [paid("01", 180)], "TRIO": [paid("010203", 1500)]}),
        ],
        2023: [
            HR(year="2023", mmdd="1228", venue="06", race="11", created="20231229",
               pays={"WIN": [paid("05", 670)], "FORECAST": [paid("0905", 4530)]}),
        ],
    }  # fmt: skip


ROWS_2023 = [
    ("jra:2023-12-28:NAK:11", "WIN", "5", 670.0),
    ("jra:2023-12-28:NAK:11", "FORECAST", "9,5", 4530.0),
]
ROWS_2024 = [
    ("jra:2024-06-01:TOK:01", "WIN", "1", 180.0),
    ("jra:2024-06-01:TOK:01", "TRIO", "1,2,3", 1500.0),
    ("jra:2024-06-02:KYO:02", "WIN", "12", 2210.0),
    ("jra:2024-06-02:KYO:02", "QUINELLA_PLACE", "6,12", 990.0),
    ("jra:2024-06-02:TOK:11", "WIN", "4", 350.0),
    ("jra:2024-06-02:TOK:11", "PLACE", "4", 140.0),
    ("jra:2024-06-02:TOK:11", "PLACE", "12", 310.0),
    ("jra:2024-06-02:TOK:11", "TIERCE", "4,12,7", 35890.0),
]


def test_build_writes_dividends_readable_through_the_store(data: Path, tmp_path: Path) -> None:
    for year, races in build_races().items():
        write_year(data, year, races)

    report = dividends.build_dividends()
    assert isinstance(report, records.BuildReport)
    assert report.name == "dividends"
    path = tables.table_path(SOURCE, "dividends")
    assert path == data.resolve() / "processed" / SOURCE / "dividends.parquet"
    assert path.is_file()
    declared = schemas.TABLES["dividends"].columns
    assert pq.read_schema(path) == pa.schema(
        [(c, tables.ARROW_TYPES[k]) for c, k in declared.items()]
    )
    # Every JRA payout, sorted by race_id with each race's own order kept; no NAR row.
    assert store_rows() == ROWS_2023 + ROWS_2024
    assert report.counts["rows"] == len(ROWS_2023) + len(ROWS_2024)

    # One year only.
    dividends.build_dividends(years=[2023])
    assert store_rows() == ROWS_2023

    # Another export and another output folder; the store's file is left alone.
    other = tmp_path / "elsewhere" / "jravan"
    shutil.copytree(data / "raw" / SOURCE, other)
    out = tmp_path / "out"
    report = dividends.build_dividends(years=[2024], raw=other, out=out)
    assert report.counts["rows"] == len(ROWS_2024)
    assert rows_of(pq.read_table(out / "dividends.parquet").to_pandas()) == ROWS_2024
    assert store_rows() == ROWS_2023


def write_races(rows: Sequence[dict]) -> None:
    """A synthetic jravan races table: the declared columns, only the keys and post time set."""
    declared = schemas.TABLES["races"].columns
    columns = {c: [row.get(c) for row in rows] for c in declared}
    table = pa.table(
        {c: pa.array(v, type=tables.ARROW_TYPES[declared[c]]) for c, v in columns.items()}
    )
    path = tables.table_path(SOURCE, "races")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def race_row(race_id: str, post: str | None) -> dict:
    _, day, venue, race_no = race_id.split(":")
    return {
        "jurisdiction": "jra",
        "race_id": race_id,
        "meeting_id": race_id.rsplit(":", 1)[0],
        "race_date": date.fromisoformat(day),
        "venue": venue,
        "race_no": int(race_no),
        "post_time_local": post,
    }


def test_dividends_enter_the_views_at_post_plus_30_through_jravan_races(data: Path) -> None:
    a, b, c, d = (
        "jra:2024-06-01:TOK:11", "jra:2024-06-01:KYO:01", "jra:2024-06-02:TOK:05",
        "jra:2024-06-01:HSN:03",
    )  # fmt: skip
    write_year(
        data,
        2024,
        [
            HR(mmdd="0601", venue="05", race="11",
               pays={"WIN": [paid("04", 350)], "QUINELLA": [paid("0412", 2450)]}),
            HR(mmdd="0601", venue="08", race="01", pays={"WIN": [paid("09", 1720)]}),
            HR(mmdd="0602", venue="05", race="05", pays={"WIN": [paid("02", 260)]}),
            HR(mmdd="0601", venue="09", race="03", pays={"WIN": [paid("06", 440)]}),
        ],
    )  # fmt: skip
    dividends.build_dividends()
    # a posts 15:40 JST (06:40 UTC), b 9:50 (00:50 UTC); c has no post time, d no races row.
    write_races([race_row(a, "15:40"), race_row(b, "9:50"), race_row(c, None)])

    rules = load_availability("japan")
    con = tables.connect(SOURCE)
    rows_a = {(a, "WIN", "4", 350.0), (a, "QUINELLA", "4,12", 2450.0)}
    rows_b = {(b, "WIN", "9", 1720.0)}
    a_open = datetime(2024, 6, 1, 7, 10, tzinfo=UTC)
    b_open = datetime(2024, 6, 1, 1, 20, tzinfo=UTC)
    probes = {
        b_open - US: set(),
        b_open: rows_b,
        a_open - US: rows_b,
        a_open: rows_a | rows_b,
        datetime(2024, 6, 30, tzinfo=UTC): rows_a | rows_b,
    }
    for at, expected in probes.items():
        view = read_asof(con, "dividends", at, rules=rules)
        got = {
            (r, p, cb, float(dv))
            for r, p, cb, dv in zip(view["race_id"], view["pool"], view["combination"],
                                    view["dividend"], strict=True)
        }  # fmt: skip
        assert got == expected, at

    # read_label returns the table whole, every race, as on the jra store.
    label = read_label(con, "dividends", rules=rules)
    whole = store_rows()
    assert len(label) == len(whole) == 5
    assert sorted(rows_of(label[COLUMNS])) == sorted(whole)
    assert {row[0] for row in whole} == {a, b, c, d}


def test_flagged_pools_write_no_row_and_are_counted(data: Path) -> None:
    write_year(
        data,
        2023,
        [
            # Horse 5 scratched after sales: Win, Place and Quinella refund it, the winners pay.
            HR(year="2023", mmdd="0415", venue="09", race="11", created="20230417",
               flags={"refund": ("WIN", "PLACE", "QUINELLA")}, refunded=(5,),
               pays={"WIN": [paid("10", 680)],
                     "PLACE": [paid("10", 230), paid("03", 150), paid("14", 410)],
                     "QUINELLA": [paid("0310", 1960)]}),
            # Special payouts on Trio and Exacta, Trifecta not established.
            HR(year="2023", mmdd="0415", venue="09", race="12", created="20230417",
               flags={"special_payout": ("TRIO", "FORECAST"), "not_established": ("TIERCE",)},
               pays={"WIN": [paid("01", 150)], "TRIO": [("000000", amount(70))],
                     "FORECAST": [("0102", amount(70))], "TIERCE": [("000000", amount(100))]}),
        ],
    )  # fmt: skip
    write_year(
        data,
        2024,
        [
            HR(year="2024", mmdd="1124", venue="05", race="12", created="20241126",
               flags={"not_established": ("PLACE",), "refund": CODES}, refunded=(2, 14),
               refunded_brackets=(1,),
               pays={"WIN": [paid("07", 920)], "PLACE": [("00", amount(100))],
                     "BRACKET_QUINELLA": [paid("47", 3380)], "QUINELLA": [paid("0708", 4410)],
                     "QUINELLA_PLACE": [paid("0708", 1500)], "FORECAST": [paid("0708", 8800)],
                     "TRIO": [paid("070811", 15220)], "TIERCE": [paid("070811", 66660)]}),
            HR(year="2024", mmdd="1124", venue="05", race="01", created="20241126",
               pays={"WIN": [paid("02", 310)]}),
        ],
    )  # fmt: skip
    h11, h12 = "jra:2023-04-15:HSN:11", "jra:2023-04-15:HSN:12"
    t01, t12 = "jra:2024-11-24:TOK:01", "jra:2024-11-24:TOK:12"
    expected = [
        (h11, "WIN", "10", 680.0),
        (h11, "PLACE", "10", 230.0),
        (h11, "PLACE", "3", 150.0),
        (h11, "PLACE", "14", 410.0),
        (h11, "QUINELLA", "3,10", 1960.0),
        (h12, "WIN", "1", 150.0),
        (t01, "WIN", "2", 310.0),
        (t12, "WIN", "7", 920.0),
        (t12, "BRACKET_QUINELLA", "4,7", 3380.0),
        (t12, "QUINELLA", "7,8", 4410.0),
        (t12, "QUINELLA_PLACE", "7,8", 1500.0),
        (t12, "FORECAST", "7,8", 8800.0),
        (t12, "TRIO", "7,8,11", 15220.0),
        (t12, "TIERCE", "7,8,11", 66660.0),
    ]
    want_years = {
        2023: {"not_established": 1, "special_payout": 2, "refund": 3, "refunded_horses": 1},
        2024: {"not_established": 1, "special_payout": 0, "refund": 8, "refunded_horses": 2},
    }
    want_total = {k: sum(y[k] for y in want_years.values()) for k in want_years[2023]}

    report = dividends.build_dividends()
    assert store_rows() == expected
    for year, counts in want_years.items():
        got = {k: report.by_year.get(year, {}).get(k, 0) for k in counts}
        assert got == counts, year
    assert {k: report.counts.get(k, 0) for k in want_total} == want_total
    assert report.counts["rows"] == len(expected)

    # The command builds the same table and writes the report, counts only.
    result = CliRunner().invoke(app, ["japan", "jravan", "dividends"])
    assert result.exit_code == 0, result.output
    assert store_rows() == expected
    written = paths.reports_dir() / "jravan-dividends.json"
    assert written.is_file()
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed["name"] == "dividends"
    for year, counts in want_years.items():
        got = {k: parsed["by_year"].get(str(year), {}).get(k, 0) for k in counts}
        assert got == counts, year
    assert {k: parsed["counts"].get(k, 0) for k in want_total} == want_total
    assert parsed["counts"]["rows"] == len(expected)


def test_the_latest_version_of_a_payout_is_kept(data: Path) -> None:
    def kyo(race: str, kubun: str, created: str, pays: dict) -> HR:
        return HR(mmdd="0608", venue="08", race=race, kubun=kubun, created=created, pays=pays)

    write_year(
        data,
        2024,
        [
            kyo("10", "2", "20240610", {"WIN": [paid("06", 480)],
                                        "PLACE": [paid("06", 170), paid("11", 260),
                                                  paid("02", 200)]}),
            kyo("11", "1", "20240608", {"WIN": [paid("13", 990)]}),
            kyo("10", "1", "20240608", {"WIN": [paid("06", 470)],
                                        "PLACE": [paid("06", 170), paid("11", 250)],
                                        "QUINELLA": [paid("0611", 1990)]}),
            kyo("12", "2", "20240610", {"WIN": [paid("01", 220)]}),
            kyo("11", "0", "20240611", {}),
            # Same day and race number at another venue: another race.
            HR(mmdd="0608", venue="05", race="10", created="20240608",
               pays={"WIN": [paid("09", 330)]}),
        ],
    )  # fmt: skip
    dividends.build_dividends()
    assert store_rows() == [
        ("jra:2024-06-08:KYO:10", "WIN", "6", 480.0),
        ("jra:2024-06-08:KYO:10", "PLACE", "6", 170.0),
        ("jra:2024-06-08:KYO:10", "PLACE", "11", 260.0),
        ("jra:2024-06-08:KYO:10", "PLACE", "2", 200.0),
        ("jra:2024-06-08:KYO:12", "WIN", "1", 220.0),
        ("jra:2024-06-08:TOK:10", "WIN", "9", 330.0),
    ]


def test_two_builds_write_identical_bytes(data: Path) -> None:
    for year, races in build_races().items():
        write_year(data, year, races)
    write_year(data, 2010, [HR(year="2010", mmdd="0105", venue="06", race="01",
                               created="20100106", pays={"WIN": [paid("08", 1110)]})])  # fmt: skip
    first_report = dividends.build_dividends()
    path = tables.table_path(SOURCE, "dividends")
    first = path.read_bytes()

    # The same inputs rewritten (new mtimes): the same bytes and the same report.
    for year, races in build_races().items():
        write_year(data, year, races)
    second_report = dividends.build_dividends()
    assert path.read_bytes() == first
    assert second_report.to_json() == first_report.to_json()
    assert sorted(p.name for p in path.parent.iterdir()) == ["dividends.parquet"]


# ------------------------------------------------------------------------------------------------
# The real export (local: needs $HORSETRACK_DATA_DIR with the export and the workbook)

REAL_YEAR = 2024
HORSE = r"(?:[1-9]|1[0-8])"


@pytest.mark.local
def test_a_real_year_maps(tmp_path: Path) -> None:
    export, workbook = wb.real_export_or_skip()
    source = export / "stored" / "RACE" / "HR" / f"{REAL_YEAR}.jvd"
    if not source.is_file():
        pytest.skip(f"no HR file for {REAL_YEAR}")
    out = tmp_path / "out"
    report = dividends.build_dividends(years=[REAL_YEAR], out=out)
    got = pq.read_table(out / "dividends.parquet")
    assert got.schema.names == COLUMNS
    df = got.to_pandas()
    n = len(df)
    assert n > 0 and report.counts["rows"] == n
    assert set(df["pool"]) == set(CODES)
    assert (df["jurisdiction"] == "jra").all()
    assert not df["is_refund"].any()
    venues = "|".join(VENUES.values())
    race_re = re.compile(rf"jra:{REAL_YEAR}-\d\d-\d\d:(?:{venues}):\d\d")
    assert df["race_id"].map(lambda r: bool(race_re.fullmatch(r))).all()
    assert df["race_id"].is_monotonic_increasing
    assert not df.duplicated(["race_id", "pool", "combination"]).any()
    assert ((df["dividend"] >= 100) & (df["dividend"] % 10 == 0)).all()

    legs = {"WIN": 1, "PLACE": 1, "BRACKET_QUINELLA": 2, "QUINELLA": 2, "QUINELLA_PLACE": 2,
            "FORECAST": 2, "TRIO": 3, "TIERCE": 3}  # fmt: skip
    bad = 0
    for pool, combination in zip(df["pool"], df["combination"], strict=True):
        token = r"[1-8]" if pool == "BRACKET_QUINELLA" else HORSE
        parts = combination.split(",")
        if len(parts) != legs[pool] or not all(re.fullmatch(token, p) for p in parts):
            bad += 1
            continue
        numbers = [int(p) for p in parts]
        if pool == "BRACKET_QUINELLA":
            bad += numbers != sorted(numbers)
        elif pool in ("QUINELLA", "QUINELLA_PLACE", "TRIO"):
            bad += numbers != sorted(set(numbers))
        else:
            bad += len(set(numbers)) != len(numbers)
    assert bad == 0, f"{bad} malformed combinations"

    # Independently of the module: every JRA race with a positive Win payout has a Win row.
    layout_ = spec.record_spec("HR", workbook)
    cols = records.decode(source.read_bytes(), layout_)
    jra = np.isin(cols["競馬場コード"], [v.encode() for v in VENUES])
    win_paid = (records.integer(cols["単勝払戻/払戻金"]) > 0).any(axis=1)
    keys = np.char.add(np.char.add(cols["開催月日"], cols["競馬場コード"]), cols["レース番号"])[
        jra & win_paid
    ]
    assert df.loc[df["pool"] == "WIN", "race_id"].nunique() == len(np.unique(keys))
    assert df["race_id"].nunique() == len(np.unique(keys))

    # The flag counts, from the decoded flags of the same file.
    kept = jra
    for key, kind in FLAGS.items():
        flagged = sum(int((cols[f"{kind}フラグ{JV_POOL[c]}"][kept] == b"1").sum()) for c in CODES)
        assert report.by_year.get(REAL_YEAR, {}).get(key, 0) == flagged, key
    horses = int((cols[REFUNDED_HORSES][kept] == b"1").sum())
    assert report.by_year.get(REAL_YEAR, {}).get("refunded_horses", 0) == horses
