"""JRDB rows -> our JRA store: race keys to race_id, 血統登録番号 to horse_id, with a report
of what did not match.

Ported from HKTrack's ``sources/jra/jrdb/join.py``.

Horse: JRDB's 8-character 血統登録番号 is the 2-digit birth year + 6 digits; our (netkeiba)
horse_id is the 4-digit birth year + the same 6 digits. Two-digit years from
``CENTURY_PIVOT`` up are 19xx, the rest 20xx (JRDB's files start in 1999, whose oldest
runners were born in the 1980s).

Race: the JRDB レースキー 場2 年2 回1 日1 (hex) R2 becomes the 12-digit netkeiba race id
that the results parser keeps in ``races.source_race_key``: YYYY 場 回(2) 日(2, decimal) R.
A race without one (an upcoming card race) is found through its own venue number
(countries/japan/config/jra.yaml), year, ``kai`` (回), ``nichi`` (日) and race number, read
when the races frame carries those columns. JRDB's synthetic keys for non-JRA runs (回 and
日 as letters) match nothing.

A row that prints a JRDB id is matched when that id, converted, is the store runner's horse
at the same race and horse number; when it names another horse it is a mismatch and is
never rescued by its number. A row without an id (TYB, CYB) takes the store runner at
(race, horse number).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from horsetrack.core.config import load_source

COUNTRY, SOURCE = "japan", "jra"
CENTURY_PIVOT = 70  # two-digit years >= 70 are 19xx
_HORSE_ID = re.compile(r"[0-9]{8}")
_RACE_KEY = re.compile(r"[0-9]{4}[0-9][0-9a-f][0-9]{2}")
_RACE_CARD_COLUMNS = ("venue", "race_date", "kai", "nichi", "race_no")


def horse_id(jrdb_id: Any) -> str | None:
    """'19103588' -> '2019103588', '99101712' -> '1999101712'; None for anything but 8 digits."""
    if not isinstance(jrdb_id, str) or not _HORSE_ID.fullmatch(jrdb_id):
        return None
    return full_year(jrdb_id[:2]) + jrdb_id[2:]


def full_year(two_digits: str) -> str:
    """'15' -> '2015', '99' -> '1999' (``CENTURY_PIVOT``)."""
    return ("19" if int(two_digits) >= CENTURY_PIVOT else "20") + two_digits


def source_race_key(race_key: Any) -> str | None:
    """'08254611' -> '202508040611' (Kyoto 2025, 4回, 6日, R11); None for a malformed or
    non-JRA key."""
    if not isinstance(race_key, str) or not _RACE_KEY.fullmatch(race_key):
        return None
    venue, year, kai, nichi, race_no = (
        race_key[:2],
        race_key[2:4],
        race_key[4],
        race_key[5],
        race_key[6:],
    )
    return f"{full_year(year)}{venue}{int(kai):02d}{int(nichi, 16):02d}{race_no}"


@dataclass
class JoinReport:
    rows: int = 0
    race_matched: int = 0  # rows whose race was found
    horse_matched: int = 0  # of those, rows matched to a store runner (by id or by number)
    matched_by_horse_no: int = 0  # of those, rows matched by number only (no JRDB id in the row)
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def race_unmatched_rate(self) -> float:
        return 1.0 - self.race_matched / self.rows if self.rows else float("nan")

    @property
    def horse_unmatched_rate(self) -> float:
        return 1.0 - self.horse_matched / self.race_matched if self.race_matched else float("nan")

    def to_json(self, examples: int = 20) -> dict[str, Any]:
        def rate(value: float) -> float | None:
            return None if np.isnan(value) else round(value, 6)

        out = asdict(self)
        out["mismatches"] = len(self.mismatches)
        out["mismatch_examples"] = self.mismatches[:examples]
        out["race_unmatched_rate"] = rate(self.race_unmatched_rate)
        out["horse_unmatched_rate"] = rate(self.horse_unmatched_rate)
        return out


def _mapped(values: pd.Series, fn) -> pd.Series:
    """``fn`` applied once per distinct value (missing values as None)."""
    values = values.astype("object").where(values.notna(), None)
    lookup = {v: fn(v) for v in pd.unique(values)}
    return pd.Series([lookup[v] for v in values], index=values.index, dtype="object")


def race_source_keys(races: pd.DataFrame) -> pd.Series:
    """Each race row's 12-digit source key: ``source_race_key``, else built from venue, year,
    kai, nichi and race number."""
    key = (
        races["source_race_key"]
        if "source_race_key" in races.columns
        else pd.Series(None, index=races.index)
    )
    key = key.astype("object")
    key = key.where(key.notna() & key.ne(""), None)
    if not set(_RACE_CARD_COLUMNS) <= set(races.columns):
        return key
    numbers = {
        code: venue.number
        for code, venue in load_source(COUNTRY, SOURCE).venues.items()
        if venue.number
    }
    parts = pd.DataFrame(
        {
            "venue": races["venue"].map(numbers),
            "year": pd.to_datetime(races["race_date"], errors="coerce").dt.year,
            **{c: pd.to_numeric(races[c], errors="coerce") for c in ("kai", "nichi", "race_no")},
        },
        index=races.index,
    )
    usable = parts.notna().all(axis=1) & key.isna()
    derived = [
        f"{int(r.year)}{r.venue}{int(r.kai):02d}{int(r.nichi):02d}{int(r.race_no):02d}"
        for r in parts[usable].itertuples(index=False)
    ]
    key.loc[usable] = derived
    return key


def _as_dates(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values, errors="coerce")
    return pd.Series(
        [None if pd.isna(d) else d.date() for d in stamps], index=values.index, dtype="object"
    )


def match_races(race_keys: pd.Series, races: pd.DataFrame) -> pd.DataFrame:
    """``race_id`` and ``race_date`` (a date) of the store race each JRDB race key names;
    None where none does."""
    lookup = pd.DataFrame(
        {
            "_skey": race_source_keys(races),
            "race_id": races["race_id"],
            "race_date": races["race_date"],
        }
    )
    lookup = lookup.dropna(subset=["_skey", "race_id"]).drop_duplicates("_skey", keep="first")
    wanted = pd.DataFrame(
        {"_skey": _mapped(race_keys.astype("object"), source_race_key).to_numpy()}
    )
    found = wanted.merge(lookup, on="_skey", how="left", validate="many_to_one")
    found.index = race_keys.index
    race_id = found["race_id"].astype("object")
    return pd.DataFrame(
        {
            "race_id": race_id.where(race_id.notna(), None),
            "race_date": _as_dates(found["race_date"]),
        }
    )


def join_store(
    rows: pd.DataFrame, races: pd.DataFrame, runners: pd.DataFrame
) -> tuple[pd.DataFrame, JoinReport]:
    """Every row of ``rows`` (``race_key``, ``horse_no``, optionally ``jrdb_horse_id``), in
    order, plus ``race_id``, ``race_date``, ``horse_id`` and ``matched_by`` ("horse_id",
    "horse_no" or None); see the module docstring."""
    out = rows.reset_index(drop=True).copy()
    races_found = match_races(out["race_key"], races)
    race_id = races_found["race_id"]
    number = pd.to_numeric(out["horse_no"], errors="coerce").astype("float64")

    store = runners[["race_id", "horse_no", "horse_id"]].copy()
    store["horse_no"] = pd.to_numeric(store["horse_no"], errors="coerce").astype("float64")
    store = store.dropna(subset=["race_id", "horse_no"]).drop_duplicates(
        ["race_id", "horse_no"], keep="first"
    )
    store = store.rename(columns={"horse_id": "_store"})
    at_number = (
        pd.DataFrame({"race_id": race_id.to_numpy(), "horse_no": number.to_numpy()})
        .merge(store, on=["race_id", "horse_no"], how="left", validate="many_to_one")["_store"]
        .astype("object")
    )
    at_number.index = out.index
    at_number = at_number.where(at_number.notna() & race_id.notna(), None)

    jrdb_ids = (
        out["jrdb_horse_id"] if "jrdb_horse_id" in out.columns else pd.Series(None, index=out.index)
    )
    converted = _mapped(jrdb_ids.astype("object"), horse_id)
    has_id, found, in_store = converted.notna(), race_id.notna(), at_number.notna()
    by_id = found & has_id & in_store & converted.eq(at_number)
    by_number = found & ~has_id & in_store
    mismatch = found & has_id & in_store & ~converted.eq(at_number)

    out["race_id"] = race_id
    out["race_date"] = races_found["race_date"]
    out["horse_id"] = converted.where(has_id, at_number.where(by_number, None))
    out["matched_by"] = pd.Series(
        np.where(by_id, "horse_id", np.where(by_number, "horse_no", None)),
        index=out.index,
        dtype="object",
    )
    report = JoinReport(
        rows=len(out),
        race_matched=int(found.sum()),
        horse_matched=int((by_id | by_number).sum()),
        matched_by_horse_no=int(by_number.sum()),
        mismatches=[
            {
                "race_id": race_id[i],
                "race_key": out.at[i, "race_key"],
                "horse_no": int(number[i]),
                "horse_id": converted[i],
                "store_horse_id": at_number[i],
            }
            for i in np.flatnonzero(mismatch.to_numpy())
        ],
    )
    return out, report
