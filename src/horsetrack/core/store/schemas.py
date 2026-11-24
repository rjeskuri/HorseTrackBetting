"""Column contracts of the store's tables (docs/specs/store.md): names, kinds and sort order.

Pure data, no pandas or pyarrow imports, so every store module and test can share it cheaply. The
tables and columns are HKTrack's (its SPEC section 5), plus `jurisdiction` and the key columns on
the two JRDB tables, whose per-family columns are declared by the ticket that builds them.
"""

from __future__ import annotations

from dataclasses import dataclass

# column kinds; the Arrow type of each is in tables.ARROW_TYPES
INT = "int"  # int64, pandas Int64
FLOAT = "float"  # float64
STR = "str"  # string
BOOL = "bool"  # bool, pandas boolean
DATE = "date"  # date32 (the local race date)
UTC = "utc"  # timestamp[us, UTC]; only for *_utc values that came with the pages
INT_LIST = "int_list"  # list<int32>
FLOAT_LIST = "float_list"  # list<float64>

KINDS: tuple[str, ...] = (INT, FLOAT, STR, BOOL, DATE, UTC, INT_LIST, FLOAT_LIST)


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: dict[str, str]  # ordered: column -> kind
    sort_by: tuple[str, ...]


_RACE_KEYS = {"meeting_id": STR, "race_date": DATE, "venue": STR, "race_no": INT, "season": STR}
# a JRDB row is one runner matched to the store's races and runners
_JRDB_KEYS = {
    "jurisdiction": STR, "race_id": STR, "race_date": DATE, "horse_no": INT, "horse_id": STR,
    "race_key": STR, "jrdb_horse_id": STR, "matched_by": STR,
}  # fmt: skip

_SPECS = (
    TableSpec(
        "meetings",
        {
            "jurisdiction": STR, "meeting_id": STR, "race_date": DATE, "venue": STR, "season": STR,
            "day_night": STR, "surface_mix": STR, "status": STR,
        },
        ("race_date", "venue"),
    ),
    TableSpec(
        "races",
        {
            "jurisdiction": STR, "race_id": STR, **_RACE_KEYS, "race_index": INT, "race_name": STR,
            "race_class": STR, "rating_hi": INT, "rating_lo": INT, "distance_m": INT,
            "surface": STR, "course_config": STR, "going": STR, "prize": INT,
            "post_time_local": STR, "field_size": INT, "finish_time_s": FLOAT,
            "split_times_s": FLOAT_LIST, "status": STR,
            "weight_type": STR, "race_class_raw": STR, "going_raw": STR, "source_race_key": STR,
            "result_published_utc": UTC, "result_unpublished_utc": UTC,
        },
        ("race_id",),
    ),
    TableSpec(
        "runners",
        {
            "jurisdiction": STR, "race_id": STR, **_RACE_KEYS, "horse_id": STR, "horse_no": INT,
            "horse_name": STR, "brand_raw": STR, "jockey_key": STR, "jockey_code": STR,
            "jockey_name": STR, "trainer_key": STR, "trainer_code": STR, "trainer_name": STR,
            "actual_wt_lb": INT, "declared_horse_wt_lb": INT, "draw": INT, "placing_raw": STR,
            "finish_pos": INT, "placing_code": STR, "dead_heat": BOOL, "lbw_raw": STR,
            "lbw_lengths": FLOAT, "running_positions": INT_LIST, "finish_time_s": FLOAT,
            "win_odds": FLOAT, "is_scratched": BOOL, "rating": INT, "gear_raw": STR,
            "bracket": INT, "actual_wt_kg": FLOAT, "horse_wt_kg": INT, "horse_wt_change_kg": INT,
            "last_3f_s": FLOAT, "win_fav_rank": INT, "sex": STR, "age": INT,
        },
        ("race_id", "horse_no", "horse_id"),
    ),
    TableSpec(
        "dividends",
        {
            "jurisdiction": STR, "race_id": STR, "pool": STR, "combination": STR,
            "dividend": FLOAT, "is_refund": BOOL,
        },
        ("race_id",),  # page order within a race is kept (stable sort)
    ),
    TableSpec(
        "sectionals",
        {
            "jurisdiction": STR, "race_id": STR, "horse_id": STR, "horse_no": INT,
            "section_no": INT, "section_pos": INT, "section_margin_lengths": FLOAT,
            "section_time_s": FLOAT, "section_margin_raw": STR, "sub_split_times_s": FLOAT_LIST,
        },
        ("race_id", "horse_no", "section_no"),
    ),
    TableSpec(
        "incidents",
        {"jurisdiction": STR, "race_id": STR, "horse_id": STR, "horse_no": INT, "text": STR},
        ("race_id",),
    ),
    TableSpec(
        "running_comments",
        {
            "jurisdiction": STR, "race_id": STR, "horse_id": STR, "horse_no": INT,
            "gear_raw": STR, "comment": STR,
        },
        ("race_id", "horse_no"),
    ),
    TableSpec(
        "horses",
        {
            "jurisdiction": STR, "horse_id": STR, "name": STR, "origin": STR, "colour": STR,
            "sex": STR, "import_type": STR, "sire": STR, "dam": STR, "dam_sire": STR,
            "is_retired": BOOL, "profile_scraped_utc": UTC,
        },
        ("horse_id",),
    ),
    TableSpec(
        "horse_runs",
        {
            "jurisdiction": STR, "horse_id": STR, "race_date": DATE, "race_index": INT,
            "venue": STR, "surface": STR, "course_config": STR, "distance_m": INT, "going": STR,
            "race_class": STR, "draw": INT, "rating": INT, "trainer_name": STR,
            "jockey_name": STR, "lbw_raw": STR, "win_odds": FLOAT, "actual_wt_lb": INT,
            "running_positions": INT_LIST, "finish_time_s": FLOAT, "declared_horse_wt_lb": INT,
            "gear_raw": STR, "placing_raw": STR, "race_id": STR,
        },
        ("horse_id", "race_date", "race_index"),
    ),
    TableSpec(
        "horse_pedigree",
        {
            "jurisdiction": STR, "horse_id": STR, "sire_name": STR, "sire_name_norm": STR,
            "sire_id": STR, "dam_name": STR, "dam_name_norm": STR, "dam_id": STR,
            "dam_sire_name": STR, "dam_sire_name_norm": STR, "dam_sire_id": STR,
            "source_race_id": STR, "scraped_utc": UTC,
        },
        ("horse_id",),
    ),
    TableSpec("jrdb_pre_race", dict(_JRDB_KEYS), ("race_id", "horse_no")),
    TableSpec("jrdb_post_race", dict(_JRDB_KEYS), ("race_id", "horse_no")),
)  # fmt: skip

TABLES: dict[str, TableSpec] = {spec.name: spec for spec in _SPECS}
TABLE_NAMES: tuple[str, ...] = tuple(TABLES)

# The tables the store build writes and the page types whose parsed rows feed each (HKTrack's
# TableSpec.sources); a page's rows for any other table are ignored. The JRDB tables are built
# elsewhere, so no page type feeds them.
FED_BY: dict[str, tuple[str, ...]] = {
    "meetings": ("fixture", "results"),  # a calendar without venues: meetings from the results
    "races": ("results",),
    "runners": ("results",),
    "dividends": ("results",),
    "sectionals": ("sectional",),
    "incidents": ("results",),
    "running_comments": ("corunning",),
    "horses": ("horse",),
    "horse_runs": ("horse",),
    "horse_pedigree": ("pedigree_card",),
}
# Written only for a source whose manifest holds pages of their page types, so a source without
# them gets no empty file. One row per id: the first sort column.
SOURCE_TABLES: tuple[str, ...] = ("horse_pedigree",)
# Page types the build places. Upcoming-meeting pages (race cards, entries) are deliberately not
# here: they would leak future races into `races`.
PAGE_TYPES: tuple[str, ...] = tuple(sorted({t for types in FED_BY.values() for t in types}))
