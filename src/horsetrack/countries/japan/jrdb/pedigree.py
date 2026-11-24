"""JRDB UKC (馬基本データ) -> the ``horse_pedigree`` table, keyed by our 10-digit horse_id.

Ported from HKTrack's ``sources/jra/jrdb/pedigree.py``.

UKC is JRDB's horse master: one record per 血統登録番号 with the sire, dam and dam sire
names, their birth years and the founding sire-line codes (系統コード, added in the
2002-08-20 edition). ``build_horse_pedigree`` maps each record's 8-digit id to our horse_id
(``join.horse_id``), normalises the three ancestor names with ``normalise_name`` (so a sire
printed in full-width katakana keys the same as one in half-width), keeps the record with
the most recent ``data_date`` per horse, and derives the paternal grand- and
great-grandsire by a self-join on the normalised names.

``merge_pedigree`` layers a netkeiba-built ``horse_pedigree`` under the UKC one: UKC is
primary, and a netkeiba row is kept only for a horse UKC does not cover (foreign horses
whose master JRDB does not carry). The netkeiba fallback names are re-normalised with
``normalise_name`` so both sources share one key space.

``normalise_name`` lives here rather than in core: only Japan uses it so far.

Scraped/parsed data is untrusted: this module only extracts and reshapes, it never executes
anything from it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

from horsetrack.countries.japan.jrdb import join

HORSE_PEDIGREE = "horse_pedigree"
UKC_FAMILY = "UKC"

_CLAIM_SUFFIX = re.compile(r"\(\s*[-+]?\d+\s*\)")
_NON_NAME = re.compile(r"[\W_]+")  # anything but letters and digits, of any script

_NAME_COLUMNS = ("sire", "dam", "dam_sire")
_BIRTH_YEARS = ("sire_birth_year", "dam_birth_year", "dam_sire_birth_year")
# Columns ``build_horse_pedigree`` reads off a parsed UKC frame (missing ones read as empty).
INPUT_COLUMNS = (
    "jrdb_horse_id", "horse_name", "data_date",
    "sire_name", "dam_name", "dam_sire_name",
    "sire_line_code", "dam_sire_line_code",
    *_BIRTH_YEARS,
)  # fmt: skip
# The table ``build_horse_pedigree`` returns, in order.
OUTPUT_COLUMNS = (
    "horse_id",
    "sire_name", "sire_name_norm",
    "dam_name", "dam_name_norm",
    "dam_sire_name", "dam_sire_name_norm",
    "sire_line_code", "dam_sire_line_code",
    *_BIRTH_YEARS,
    "paternal_grandsire_name_norm", "paternal_great_grandsire_name_norm",
)  # fmt: skip


def normalise_name(name: str) -> str:
    """NFKC (full- and half-width forms fold together), drop claim suffixes like '(-7)', casefold,
    keep letters and digits of any script; everything else becomes single spaces."""
    s = _CLAIM_SUFFIX.sub(" ", unicodedata.normalize("NFKC", name or "")).casefold()
    return _NON_NAME.sub(" ", s).strip()


def _norm(name: Any) -> str | None:
    """``normalise_name`` of a printed name, with the empty result as None."""
    if not isinstance(name, str):
        return None
    return normalise_name(name) or None


def _norm_series(values: pd.Series) -> pd.Series:
    return values.astype("object").map(_norm)


def _as_year(values: pd.Series) -> pd.Series:
    """A birth year as nullable Int64 (the parser leaves it a float; a blank is NaN)."""
    return pd.to_numeric(values, errors="coerce").round().astype("Int64")


def _paternal_sire_map(names: pd.Series, sires: pd.Series) -> dict[str, str]:
    """``horse_name_norm -> sire_name_norm`` for the self-join, dropping names whose records
    disagree on the sire (recycled brand names) so an ambiguous hop resolves to NULL rather
    than a wrong ancestor."""
    pairs = pd.DataFrame({"name": names, "sire": sires})
    pairs = pairs[pairs["name"].notna() & pairs["sire"].notna()]
    if pairs.empty:
        return {}
    by_name = pairs.groupby("name")["sire"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
    return {name: sire for name, sire in by_name.items() if sire is not None}


def ancestor_collisions(names: pd.Series, sires: pd.Series) -> int:
    """How many distinct normalised horse names map to more than one sire (excluded from the
    paternal map)."""
    pairs = pd.DataFrame({"name": names, "sire": sires})
    pairs = pairs[pairs["name"].notna() & pairs["sire"].notna()]
    if pairs.empty:
        return 0
    return int((pairs.groupby("name")["sire"].nunique() > 1).sum())


def build_horse_pedigree(ukc: pd.DataFrame) -> pd.DataFrame:
    """A ``horse_pedigree`` frame from parsed UKC records: one row per horse_id (most recent
    ``data_date`` wins), with the normalised ancestor names, the line codes, the birth years and the
    paternal grand/great-grandsire."""
    frame = ukc.reindex(columns=list(INPUT_COLUMNS)).copy()
    frame["horse_id"] = frame["jrdb_horse_id"].astype("object").map(join.horse_id)
    frame = frame[frame["horse_id"].notna()].copy()

    # Most recent record per horse: a missing data_date ranks first, so a dated record always wins.
    frame["_order"] = pd.to_datetime(frame["data_date"], errors="coerce")
    frame = frame.sort_values("_order", kind="stable", na_position="first")
    frame = frame.drop_duplicates("horse_id", keep="last").reset_index(drop=True)

    frame["horse_name_norm"] = _norm_series(frame["horse_name"])
    for col in _NAME_COLUMNS:
        frame[f"{col}_name_norm"] = _norm_series(frame[f"{col}_name"])
    for col in _BIRTH_YEARS:
        frame[col] = _as_year(frame[col])

    sire_map = _paternal_sire_map(frame["horse_name_norm"], frame["sire_name_norm"])
    grandsire = frame["sire_name_norm"].map(lambda s: sire_map.get(s) if s is not None else None)
    great = grandsire.map(lambda s: sire_map.get(s) if s is not None else None)
    frame["paternal_grandsire_name_norm"] = grandsire.astype("object").where(
        grandsire.notna(), None
    )
    frame["paternal_great_grandsire_name_norm"] = great.astype("object").where(great.notna(), None)

    out = frame.reindex(columns=list(OUTPUT_COLUMNS))
    return out.astype("object").where(out.notna(), None).astype({c: "Int64" for c in _BIRTH_YEARS})


def merge_pedigree(ukc_table: pd.DataFrame, netkeiba: pd.DataFrame | None) -> pd.DataFrame:
    """UKC rows plus netkeiba rows for horse_ids UKC lacks (foreign horses). The netkeiba
    ancestor names are re-normalised with ``normalise_name`` so both sources key sires the
    same way; columns UKC has and netkeiba does not (line codes, birth years, the paternal
    hops) are NULL on the fallback rows."""
    if netkeiba is None or "horse_id" not in getattr(netkeiba, "columns", []):
        return ukc_table.reset_index(drop=True)
    have = set(ukc_table["horse_id"].dropna())
    extra = netkeiba[~netkeiba["horse_id"].isin(have)].copy()
    if extra.empty:
        return ukc_table.reset_index(drop=True)
    for col in _NAME_COLUMNS:
        name_col = f"{col}_name"
        if name_col in extra.columns:
            extra[f"{col}_name_norm"] = _norm_series(extra[name_col])
    extra = extra.reindex(columns=list(OUTPUT_COLUMNS))
    extra = (
        extra.astype("object").where(extra.notna(), None).astype({c: "Int64" for c in _BIRTH_YEARS})
    )
    return pd.concat([ukc_table[list(OUTPUT_COLUMNS)], extra], ignore_index=True)


def coverage(table: pd.DataFrame) -> dict[str, Any]:
    """Row counts and the share of rows carrying a founding sire-line code, for the build report."""
    rows = len(table)
    with_code = (
        int(table["sire_line_code"].notna().sum()) if "sire_line_code" in table.columns else 0
    )
    with_grandsire = (
        int(table["paternal_grandsire_name_norm"].notna().sum())
        if "paternal_grandsire_name_norm" in table.columns
        else 0
    )
    return {
        "rows": rows,
        "distinct_horse_ids": int(table["horse_id"].nunique()) if rows else 0,
        "with_sire_line_code": with_code,
        "sire_line_code_rate": (with_code / rows) if rows else float("nan"),
        "with_paternal_grandsire": with_grandsire,
    }
