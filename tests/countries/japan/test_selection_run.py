"""The development-fold run of the Family P selection study on the JRA-VAN boards at t (#462).

Module under test: `horsetrack.countries.japan.studies.selection_run` (new). Contract pinned here:

- `build_frame(*, date_from: date, date_to: date) -> pd.DataFrame`: the selection frame from the
  jravan store under `paths.processed_dir("jravan")` (HORSETRACK_DATA_DIR): the day files
  `odds_timeseries/<YYYY-MM-DD>.parquet` (`logger.compact.SCHEMA`), and `races`, `runners`,
  `final_odds` and `dividends` (`.parquet`). Development race days in `date_from..date_to` only: a
  holdout day (`core.eval.holdout.is_holdout_day("japan", d)`) is dropped by its file name and its
  file is never opened, nor is a day file outside the span; no row of a holdout day reaches the
  frame. A `date_from` before 2014-06-07 (D10) raises `models.v0.store.PreReformPriceError`
  before any file under the data dir is opened. One row per starter (a runner not scratched, with
  a final Win price), with at least:
    race_id, horse_no, race_date, meeting_id    the race, the runner, its day and meeting
    post_utc        the race's post, timezone-aware UTC (races.post_time_local, JST, on race_date)
    won             1 (or True) for the finish_pos 1 runner, else 0
    odds_final      the final Win odds (final_odds, pool WIN)
    win_dividend    for a winner, the WIN dividend per unit stake (dividends.dividend / 100)
    takeout         the Win pool's takeout (config/jra.yaml, 0.20 since 2014-06-07)
    weight_type     races.weight_type
    R12             "in" for race_no 12, else "out"
    odds_t          the runner's Win odds on the board at T-2: for each (race, WIN, horse) the
                    latest capture whose snapshot_utc (the 発表 minute) is at or before
                    post_utc - 2 min (exactly T-2 counts); NaN when no capture is that early.
                    Never the final board, a later capture, or another pool.
    rank_t          the min-rank of odds_t within the race (joint favourites both 1); NaN with
                    odds_t
    entropy_t       the normalised entropy of the race's Win board at t (its priced runners)
    entropy_t_bin   1..5: `searchsorted(cut_points, entropy_t, side="right") + 1`, cut at
                    config/calibration.yaml's `entropy_cut_points`
  and, for each column the committed family's rules read (odds_t, rank_t, entropy_t_bin,
  weight_type, R12), its declaration `<col>__source` (board_at_t for the board columns; one of
  selection.ADMITTED_SOURCES) and `<col>__available_utc` (tz-aware, at or before t), so the frame
  passes `selection.run_study`'s own input checks.
- `run_selection(*, date_from, date_to, out_dir, seed, registry=None, holdout=None)
  -> SelectionRun`: refuses a span before 2014-06-07 (PreReformPriceError) before any file is
  opened or any record appended; then, before any store file is opened, appends the committed
  yaml's `dropped` variants as failed records of `japan-family-p-selection-v1` and freezes
  `dataclasses.asdict(load_family("japan"))` in `registry` (default
  `default_registry("japan")`); builds the frame; calls `run_study` through its module global
  (`selection_run.run_study`, a test wraps it) with that spec, `registry`, `holdout` (never
  opened), `seed` and `sources=("jra", "jravan")`; writes the private report through its module
  global `write_report` (sources naming jravan) under `out_dir`. `SelectionRun.study` is the
  `StudyReport`, `SelectionRun.report_path` the written report. The run writes nothing inside the
  repository.

Every store row is synthetic and built here; HORSETRACK_DATA_DIR points under tmp. Expected values
are worked out from the generator's own boards, never from the module under test. The folds are a
short monthly scheme patched into `selection.load_country_scheme` so a few months of synthetic
days score.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import builtins
import dataclasses
import glob
import io
import math
import os
import pathlib
import re
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import yaml

from horsetrack.core import env
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.folds import FoldScheme
from horsetrack.core.eval.holdout import (
    DEFAULT_COUNTRIES_DIR,
    LOG_NAME,
    LockedHoldout,
    is_holdout_day,
)
from horsetrack.core.eval.registry import Registry
from horsetrack.core.store.build import arrow_schema
from horsetrack.countries.japan.jravan.odds import FINAL_ODDS_SCHEMA
from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.models.v0.store import PreReformPriceError
from horsetrack.countries.japan.studies import selection
from horsetrack.countries.japan.studies import selection_run
from horsetrack.countries.japan.studies.selection import ADMITTED_SOURCES, load_family, spec_hash
from horsetrack.countries.japan.studies.selection_run import build_frame, run_selection

UTC = timezone.utc
FAMILY_FILE = DEFAULT_COUNTRIES_DIR / "japan" / "config" / "selection_family.yaml"
FAMILY = yaml.safe_load(FAMILY_FILE.read_text(encoding="utf-8"))
FAMILY_ID = FAMILY["family_id"]
RULE_IDS = [rule["rule_id"] for rule in FAMILY["rules"]]
DROPPED = [entry["variant"] for entry in FAMILY["dropped"]]
MINUTES = int(re.fullmatch(r"T-(\d+)", FAMILY["decision_time"]).group(1))  # T-2
CUTS = [float(c) for c in load_country_yaml("japan", "calibration")["entropy_cut_points"]]
SPAN_START = date(2014, 6, 7)  # D10
EARLY_DAY = date(2014, 6, 1)  # before the price era
HOLDOUT_DAY = date(2026, 11, 29)
LATE_DEV_DAY = date(2026, 11, 28)  # on the holdout's start, but a development day
GARBAGE = b"not a parquet file: a holdout day's file is never opened\n"
POOL_ROW = "__pool__"
SEED = 462
BOARD_COLS = ("odds_t", "rank_t", "entropy_t_bin")
CARD_COLS = ("weight_type", "R12")
SHORT = FoldScheme(
    unit="month",
    test_length=1,
    step=1,
    min_train=2,
    window="expanding",
    embargo_days=7,
    anchor_month=1,
)


def takeout_on(day: date) -> float:
    return float(load_source("japan", "jra").pools["WIN"].takeout_on(day))


# ---------------------------------------------------------------- a synthetic race in every table


@dataclasses.dataclass
class Race:
    """One race: its Win captures (minutes before post -> {horse: odds}), other pools' captures,
    the final board, the finish and the Win dividend (yen per 100)."""

    day: date
    venue: str
    race_no: int
    weight_type: str
    horses: list[int]
    captures: list[tuple[float, dict[int, float]]]
    final: dict[int, float]
    winner: int
    dividend: float
    scratched: tuple[int, ...] = ()
    place: list[tuple[float, dict[int, float]]] = dataclasses.field(default_factory=list)

    @property
    def race_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{self.venue}:{self.race_no:02d}"

    @property
    def meeting_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{self.venue}"

    @property
    def post(self) -> datetime:
        d = self.day
        offset = 0 if self.venue == "TOK" else 15
        return datetime(d.year, d.month, d.day, 1, 0, tzinfo=UTC) + timedelta(
            minutes=30 * self.race_no + offset
        )

    @property
    def starters(self) -> list[int]:
        return [h for h in self.horses if h not in self.scratched]


def ts_row(race: Race, pool: str, combination: str, minutes: float, odds, investment=None):
    post = race.post
    return {
        "race_id": race.race_id,
        "venue": race.venue,
        "race_no": race.race_no,
        "pool": pool,
        "combination": combination,
        "snapshot_utc": post - timedelta(minutes=minutes),
        "fetched_utc": None,  # the JRA-VAN export has no fetch time
        "post_utc": post,
        "minutes_to_post": float(minutes),
        "odds": None if odds is None else float(odds),
        "odds_hi": None,
        "investment": investment,
        "status_raw": None,
        "is_final_capture": False,
        "dividend": None,
        "is_refund": None,
    }


def ts_rows(race: Race) -> list[dict]:
    rows = []
    for pool, captures in (("WIN", race.captures), ("PLACE", race.place)):
        for minutes, board in captures:
            rows += [ts_row(race, pool, str(h), minutes, o) for h, o in board.items()]
            rows.append(ts_row(race, pool, POOL_ROW, minutes, None, investment=1.0e7))
    return rows


def table_rows(race: Race) -> dict[str, list[dict]]:
    base = {"jurisdiction": "jra", "race_id": race.race_id, "race_date": race.day,
            "venue": race.venue, "race_no": race.race_no}  # fmt: skip
    jst = race.post + timedelta(hours=9)
    races = [{**base, "meeting_id": race.meeting_id, "season": str(race.day.year),
              "race_name": "synthetic", "race_class": "C1", "distance_m": 1600,
              "surface": "turf", "post_time_local": f"{jst:%H:%M}",
              "field_size": len(race.starters), "status": "run",
              "weight_type": race.weight_type}]  # fmt: skip
    order = [race.winner] + [h for h in race.starters if h != race.winner]
    finish = {h: i + 1 for i, h in enumerate(order)}
    runners = [{**base, "meeting_id": race.meeting_id, "season": str(race.day.year),
                "horse_id": f"{race.race_id}:{h}", "horse_no": h, "horse_name": f"horse {h}",
                "finish_pos": finish.get(h), "placing_raw": str(finish.get(h, "")),
                "win_odds": race.final.get(h), "is_scratched": h in race.scratched}
               for h in race.horses]  # fmt: skip
    final = [{**base, "pool": "WIN", "combination": str(h), "odds": float(o),
              "published_utc": race.post, "data_kubun": "4"}
             for h, o in race.final.items()]  # fmt: skip
    final += [{**base, "pool": "PLACE", "combination": str(h), "odds": 1.1 + h / 10,
               "published_utc": race.post, "data_kubun": "4"} for h in race.starters]  # fmt: skip
    final.append({**base, "pool": "WIN", "combination": POOL_ROW, "odds": None,
                  "investment": 1.0e8, "published_utc": race.post, "data_kubun": "4"})  # fmt: skip
    paid = [{"jurisdiction": "jra", "race_id": race.race_id, "pool": "WIN",
             "combination": str(race.winner), "dividend": float(race.dividend),
             "is_refund": False}]  # fmt: skip
    paid += [{"jurisdiction": "jra", "race_id": race.race_id, "pool": "PLACE",
              "combination": str(h), "dividend": 110.0 + 10 * i, "is_refund": False}
             for i, h in enumerate(order[:3])]  # fmt: skip
    return {"races": races, "runners": runners, "final_odds": final, "dividends": paid}


SHAPES = {
    "races": arrow_schema("races"),
    "runners": arrow_schema("runners"),
    "final_odds": FINAL_ODDS_SCHEMA,
    "dividends": arrow_schema("dividends"),
}


def write_store(data: Path, races: list[Race], *, garbage_days=()) -> SimpleNamespace:
    """processed/jravan under `data`: a day file per race day (a garbage file on `garbage_days`,
    which must never be opened) and the four tables."""
    processed = data / "processed" / "jravan"
    folder = processed / "odds_timeseries"
    folder.mkdir(parents=True)
    (folder / "_build.json").write_text("{}\n")
    by_day: dict[date, list[Race]] = {}
    for race in races:
        by_day.setdefault(race.day, []).append(race)
    files = {}
    for day, day_races in sorted(by_day.items()):
        path = folder / f"{day.isoformat()}.parquet"
        if day in garbage_days:
            path.write_bytes(GARBAGE)
        else:
            rows = [row for race in day_races for row in ts_rows(race)]
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
        files[day] = path
    tables: dict[str, list[dict]] = {name: [] for name in SHAPES}
    for race in races:
        for name, rows in table_rows(race).items():
            tables[name] += rows
    for name, rows in tables.items():
        pq.write_table(
            pa.Table.from_pylist(rows, schema=SHAPES[name]), processed / f"{name}.parquet"
        )
    return SimpleNamespace(data=data, processed=processed, files=files, races=races)


# ---------------------------------------------------------------- references (first principles)


def board_at_t(race: Race, minutes: float = MINUTES) -> dict[int, tuple[float, float]]:
    """{horse: (odds, minutes before post of the capture)}: each horse's latest Win capture at or
    before t (minutes_to_post >= minutes)."""
    board: dict[int, tuple[float, float]] = {}
    for mins, capture in sorted(race.captures, key=lambda c: -c[0]):  # earliest first
        if mins < minutes:
            continue
        for h, o in capture.items():
            board[h] = (float(o), mins)
    return board


def min_ranks(odds: dict[int, float]) -> dict[int, int]:
    return {h: 1 + sum(1 for v in odds.values() if v < o) for h, o in odds.items()}


def norm_entropy(odds: dict[int, float]) -> float:
    inv = np.array([1.0 / o for o in odds.values()])
    q = inv / inv.sum()
    return float(-(q * np.log(q)).sum() / math.log(len(q))) if len(q) >= 2 else math.nan


def entropy_bin(value: float) -> int:
    return 1 + sum(1 for c in CUTS if value >= c)


def expected_rows(race: Race) -> dict[int, dict]:
    board = {h: o for h, (o, _) in board_at_t(race).items()}
    ranks, entropy = min_ranks(board), norm_entropy(board)
    out = {}
    for h in race.starters:
        out[h] = {
            "race_date": race.day,
            "meeting_id": race.meeting_id,
            "post_utc": race.post,
            "won": int(h == race.winner),
            "odds_final": race.final[h],
            "win_dividend": race.dividend / 100 if h == race.winner else None,
            "takeout": takeout_on(race.day),
            "weight_type": race.weight_type,
            "R12": "in" if race.race_no == 12 else "out",
            "odds_t": board.get(h, math.nan),
            "rank_t": ranks.get(h, math.nan),
            "entropy_t": entropy,
            "entropy_t_bin": entropy_bin(entropy),
        }
    return out


# ---------------------------------------------------------------- file spies


class ForbiddenRead(BaseException):
    """A file the build must not open was opened. A BaseException, so `except Exception` cannot
    hide it."""


_PATH_KEYS = ("source", "path", "where", "file", "path_or_paths", "filename")


def _arg_paths(value) -> list[Path]:
    if isinstance(value, (list, tuple)):
        return [p for v in value for p in _arg_paths(v)]
    if not isinstance(value, (str, os.PathLike)):
        return []
    text = os.fsdecode(os.fspath(value))
    if any(ch in text for ch in "*?["):
        return [Path(x) for x in glob.glob(text, recursive=True)]
    path = Path(text)
    if path.is_dir():  # a directory read reads every file under it
        return [p for p in path.rglob("*") if p.is_file()]
    return [path]


def install_file_spy(mp: pytest.MonkeyPatch, on_open: Callable[[Path], None]) -> None:
    """Call `on_open(path)` for every path a parquet reader or file opener is handed (it may
    raise ForbiddenRead); every call then goes through."""

    def guard(args, kwargs) -> None:
        value = args[0] if args else next((kwargs[k] for k in _PATH_KEYS if k in kwargs), None)
        for path in _arg_paths(value):
            on_open(path)

    def wrap(func):
        def spy(*args, **kwargs):
            guard(args, kwargs)
            return func(*args, **kwargs)

        return spy

    readers = [
        (pq, "read_table"),
        (pq, "read_schema"),
        (pq, "read_metadata"),
        (pq, "read_pandas"),
        (ds, "dataset"),
        (pa, "memory_map"),
        (pa, "OSFile"),
        (pd, "read_parquet"),
        (builtins, "open"),
        (io, "open"),
        (os, "open"),
    ]
    try:
        import polars as pl

        readers += [(pl, n) for n in ("read_parquet", "scan_parquet", "read_parquet_schema")]
    except ImportError:  # pragma: no cover
        pass
    for module, name in readers:
        if hasattr(module, name):
            mp.setattr(module, name, wrap(getattr(module, name)))
    for name in ("ParquetFile", "ParquetDataset"):
        original = getattr(pq, name)

        def init(self, *args, _original=original, **kwargs):
            guard(args, kwargs)
            _original.__init__(self, *args, **kwargs)

        mp.setattr(pq, name, type(name, (original,), {"__init__": init}))
    accessor = getattr(pathlib, "_NormalAccessor", None)  # Python 3.10's Path.open
    if accessor is not None and hasattr(accessor, "open"):
        mp.setattr(accessor, "open", staticmethod(wrap(accessor.open)))


def real(path: Path) -> str:
    return os.path.realpath(path)


def under(root: Path) -> Callable[[Path], bool]:
    base = Path(real(root))
    return lambda p: Path(real(p)) == base or base in Path(real(p)).parents


def forbid(paths: list[Path]) -> Callable[[Path], None]:
    """An `on_open` raising ForbiddenRead for any of `paths`."""
    banned = frozenset(real(p) for p in paths)

    def on_open(path: Path) -> None:
        if real(path) in banned:
            raise ForbiddenRead(str(path))

    return on_open


# ---------------------------------------------------------------- the frame builder


def hand_races() -> list[Race]:
    """Hand-made races whose board at T-2 differs from every earlier and later capture, from
    another pool's board and from the final board, on two development days, a holdout day, a day
    before the price era and a development day after the span."""
    d1 = date(2025, 6, 1)
    after = date(2026, 12, 5)
    assert not is_holdout_day("japan", d1) and not is_holdout_day("japan", LATE_DEV_DAY)
    assert is_holdout_day("japan", HOLDOUT_DAY) and not is_holdout_day("japan", after)
    a = Race(  # the board at t (T-2.5) is concentrated on horse 3; later and final boards are not
        day=d1, venue="TOK", race_no=11, weight_type="handicap", horses=[1, 2, 3, 4, 5, 6],
        captures=[
            (10.0, {1: 3.0, 2: 4.0, 3: 6.0, 4: 8.0, 5: 12.0, 6: 20.0}),
            (2.5, {1: 9.0, 2: 15.0, 3: 1.3, 4: 40.0, 5: 60.0, 6: 90.0}),
            (1.0, {1: 1.6, 2: 5.0, 3: 9.0, 4: 12.0, 5: 15.0, 6: 20.0}),
            (0.0, {1: 1.8, 2: 5.5, 3: 8.0, 4: 11.0, 5: 14.0, 6: 22.0}),
        ],
        place=[(2.5, {1: 1.1, 2: 1.2, 3: 3.0, 4: 1.5, 5: 1.4, 6: 2.0})],
        final={1: 4.0, 2: 5.0, 3: 6.0, 4: 7.0, 5: 2.5, 6: 10.0},
        winner=2, dividend=530.0,
    )  # fmt: skip
    b = Race(  # a capture exactly at T-2 counts and ties two favourites; T-1 does not count
        day=d1, venue="TOK", race_no=12, weight_type="set_weights", horses=[1, 2, 3, 4, 5],
        captures=[
            (6.0, {1: 5.0, 2: 6.0, 3: 2.0, 4: 9.0, 5: 12.0}),
            (2.0, {1: 2.2, 2: 2.2, 3: 8.0, 4: 12.0, 5: 25.0}),
            (1.0, {1: 6.0, 2: 7.0, 3: 1.5, 4: 10.0, 5: 30.0}),
        ],
        final={1: 5.5, 2: 6.5, 3: 1.7, 4: 11.0, 5: 28.0},
        winner=4, dividend=1170.0,
    )  # fmt: skip
    c = Race(  # horse 6's latest early capture is T-8; horse 7 is quoted only after t; 8 scratched
        day=d1, venue="KYO", race_no=1, weight_type="set_weights",
        horses=[1, 2, 3, 4, 5, 6, 7, 8],
        captures=[
            (8.0, {1: 4.0, 2: 5.0, 3: 6.0, 4: 7.0, 5: 9.0, 6: 2.9}),
            (3.0, {1: 3.1, 2: 4.2, 3: 7.5, 4: 8.0, 5: 10.0}),
            (1.0, {1: 2.0, 2: 4.0, 3: 9.0, 4: 9.0, 5: 11.0, 6: 4.0, 7: 1.9}),
        ],
        final={1: 2.4, 2: 4.4, 3: 9.9, 4: 9.5, 5: 12.0, 6: 4.8, 7: 2.1},
        winner=1, dividend=260.0, scratched=(8,),
    )  # fmt: skip
    d = Race(
        day=LATE_DEV_DAY, venue="TOK", race_no=12, weight_type="handicap",
        horses=[1, 2, 3, 4, 5, 6],
        captures=[
            (5.0, {1: 3.0, 2: 3.5, 3: 6.0, 4: 8.0, 5: 10.0, 6: 14.0}),
            (3.0, {1: 3.4, 2: 3.2, 3: 5.0, 4: 9.0, 5: 11.0, 6: 15.0}),
            (1.0, {1: 2.0, 2: 6.0, 3: 4.0, 4: 9.0, 5: 12.0, 6: 16.0}),
        ],
        final={1: 2.2, 2: 5.8, 3: 4.1, 4: 9.2, 5: 12.5, 6: 16.5},
        winner=3, dividend=430.0,
    )  # fmt: skip
    plain = {1: 2.0, 2: 3.0, 3: 5.0, 4: 8.0}
    others = [
        Race(day=day, venue="TOK", race_no=5, weight_type="handicap", horses=[1, 2, 3, 4],
             captures=[(3.0, plain)], final=dict(plain), winner=1, dividend=210.0)
        for day in (HOLDOUT_DAY, EARLY_DAY, after)
    ]  # fmt: skip
    return [a, b, c, d, *others]


SPAN = (SPAN_START, LATE_DEV_DAY)
KEPT = 4  # the first four hand races are development races in the span


@pytest.fixture(scope="module")
def hand_store(tmp_path_factory) -> SimpleNamespace:
    races = hand_races()
    store = write_store(tmp_path_factory.mktemp("selection_hand") / "data", races,
                        garbage_days={HOLDOUT_DAY})  # fmt: skip
    store.kept = races[:KEPT]
    store.forbidden = [store.files[d] for d in (HOLDOUT_DAY, EARLY_DAY, date(2026, 12, 5))]
    return store


@pytest.fixture(scope="module")
def hand_frame(hand_store) -> pd.DataFrame:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HORSETRACK_DATA_DIR", str(hand_store.data))
        install_file_spy(mp, forbid(hand_store.forbidden))
        try:
            frame = build_frame(date_from=SPAN[0], date_to=SPAN[1])
        except ForbiddenRead as exc:
            pytest.fail(f"the frame builder opened a day file it must never open: {exc}")
    return frame


def by_key(frame: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
    return {(str(r["race_id"]), int(r["horse_no"])): r for _, r in frame.iterrows()}


def same(actual, expected) -> bool:
    if isinstance(expected, float) and math.isnan(expected):
        return actual is None or pd.isna(actual)
    if actual is None or (not isinstance(actual, str) and pd.isna(actual)):
        return False
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-12)
    return actual == expected


def test_the_premise_boards_at_t_differ_from_later_and_final_boards():
    """The fixture is meaningful: in each hand race the board at t has another favourite, odds or
    entropy bin than the final board and than the first capture after t."""
    for race in hand_races()[:3]:
        at_t = {h: o for h, (o, _) in board_at_t(race).items()}
        later = max((c for c in race.captures if c[0] < MINUTES), key=lambda c: c[0])[1]
        for other in (race.final, later):
            assert at_t != {h: other[h] for h in at_t if h in other}, race.race_id
            fav_t = {h for h, r in min_ranks(at_t).items() if r == 1}
            fav_o = {h for h, r in min_ranks(other).items() if r == 1}
            assert fav_t != fav_o, race.race_id
    a = hand_races()[0]
    assert entropy_bin(norm_entropy({h: o for h, (o, _) in board_at_t(a).items()})) == 1
    assert entropy_bin(norm_entropy(a.final)) != 1


def test_frame_has_one_row_per_starter_with_the_outline_columns(hand_store, hand_frame):
    expected = {(r.race_id, h): row for r in hand_store.kept for h, row in expected_rows(r).items()}
    got = by_key(hand_frame)
    assert set(got) == set(expected)
    assert len(hand_frame) == len(expected)  # one row per starter, no duplicates
    columns = ["race_date", "meeting_id", "post_utc", "won", "odds_final", "takeout",
               "weight_type", "R12", "odds_t", "rank_t", "entropy_t", "entropy_t_bin"]  # fmt: skip
    assert set(columns) <= set(hand_frame.columns)
    assert isinstance(hand_frame["post_utc"].dtype, pd.DatetimeTZDtype)
    wrong = {}
    for key, want in expected.items():
        row = got[key]
        for col in columns:
            value = row[col]
            if col == "race_date":
                value = pd.Timestamp(value).date()
            elif col == "post_utc":
                value = pd.Timestamp(value).to_pydatetime()
            elif col == "won":
                value = int(value)
            if not same(value, want[col]):
                wrong[(key, col)] = (value, want[col])
        if want["win_dividend"] is not None and not same(row["win_dividend"], want["win_dividend"]):
            wrong[(key, "win_dividend")] = (row["win_dividend"], want["win_dividend"])
    assert wrong == {}


def test_selection_columns_come_from_the_board_at_t_only(hand_store, hand_frame):
    """odds_t, rank_t and entropy_t_bin match the board at T-2 race by race, never the final
    board or a later capture: the favourite at t, a capture exactly at T-2, an older capture of a
    horse missing from the last one before t, and a horse quoted only after t (NaN)."""
    got = by_key(hand_frame)
    a, b, c, _ = hand_store.kept
    # Race a: horse 3 is the favourite at t (1.3), horse 5 on the final board, horse 1 at T-1.
    assert got[(a.race_id, 3)]["rank_t"] == 1 and got[(a.race_id, 3)]["odds_t"] == 1.3
    assert got[(a.race_id, 5)]["rank_t"] == 5 and got[(a.race_id, 1)]["rank_t"] == 2
    assert {int(got[(a.race_id, h)]["entropy_t_bin"]) for h in a.starters} == {1}
    # Race b: the T-2.0 capture counts; joint favourites 1 and 2 are both rank 1.
    assert [int(got[(b.race_id, h)]["rank_t"]) for h in b.starters] == [1, 1, 3, 4, 5]
    # Race c: horse 6 keeps its T-8 quote; horse 7, quoted only after t, has no odds_t or rank_t.
    assert got[(c.race_id, 6)]["odds_t"] == 2.9 and got[(c.race_id, 6)]["rank_t"] == 1
    assert pd.isna(got[(c.race_id, 7)]["odds_t"]) and pd.isna(got[(c.race_id, 7)]["rank_t"])
    assert (c.race_id, 8) not in got  # scratched


def test_frame_declares_every_rule_input_as_known_by_t(hand_store, hand_frame):
    t = hand_frame["post_utc"] - pd.Timedelta(minutes=MINUTES)
    for col in (*BOARD_COLS, *CARD_COLS):
        source, stamp = hand_frame[f"{col}__source"], hand_frame[f"{col}__available_utc"]
        present = hand_frame[col].notna()
        assert isinstance(stamp.dtype, pd.DatetimeTZDtype), col
        assert source[present].isin(ADMITTED_SOURCES).all(), col
        assert (stamp[present] <= t[present]).all(), col
        assert not (stamp.notna() & ~(stamp <= t)).any(), col
        if col in BOARD_COLS:
            assert (source[present] == "board_at_t").all(), col


def test_holdout_and_out_of_span_days_never_reach_the_frame(hand_store, hand_frame):
    days = set(pd.to_datetime(hand_frame["race_date"]).dt.date)
    assert days == {date(2025, 6, 1), LATE_DEV_DAY}
    assert not any(is_holdout_day("japan", d) for d in days)
    text = hand_frame.astype(str).to_csv()
    for day in (HOLDOUT_DAY, EARLY_DAY, date(2026, 12, 5)):
        assert f"jra:{day.isoformat()}" not in text


def test_a_span_before_the_price_era_is_refused_before_any_file_is_opened(hand_store):
    opened: list[Path] = []
    inside = under(hand_store.data)

    def on_open(path: Path) -> None:
        if inside(path):
            opened.append(path)
            raise ForbiddenRead(str(path))

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HORSETRACK_DATA_DIR", str(hand_store.data))
        install_file_spy(mp, on_open)
        with pytest.raises(PreReformPriceError):
            build_frame(date_from=EARLY_DAY, date_to=SPAN[1])
    assert opened == []


class ScoringStarted(BaseException):
    """Raised when the study fits its first benchmark: every input check came before it."""


def _no_scoring(*args, **kwargs):
    raise ScoringStarted("run_study reached its first benchmark fit")


class HoldoutOpened(BaseException):
    pass


class TrippedHoldout(LockedHoldout):
    """A LockedHoldout whose open records the call and raises."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def open(self, **kwargs):
        self.calls.append(kwargs)
        raise HoldoutOpened(f"the run opened the holdout: {kwargs}")

    def __repr__(self) -> str:
        return "TrippedHoldout()"


def new_registry(root: Path) -> Registry:
    log = root / "countries" / "japan" / LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    return Registry(root / "data" / "registry" / "japan.jsonl", holdout_log=log)


def test_the_frame_passes_run_study_input_checks(hand_store, hand_frame, tmp_path, monkeypatch):
    """The committed family, frozen, on the built frame: run_study's own checks (frozen spec,
    columns, declarations at t, development days) all pass and it reaches scoring."""
    spec = load_family("japan")
    registry = new_registry(tmp_path)
    registry.freeze_family(spec.family_id, dataclasses.asdict(spec))
    monkeypatch.setattr(selection, "fit_win_calibration", _no_scoring)
    monkeypatch.setattr(selection, "load_country_scheme", lambda country: SHORT)
    holdout = TrippedHoldout()
    with pytest.raises(ScoringStarted):
        selection.run_study(spec, hand_frame.copy(), registry=registry, holdout=holdout,
                            seed=SEED, sources=("jra", "jravan"))  # fmt: skip
    assert holdout.calls == []


# ---------------------------------------------------------------- the run


RUN_FROM, RUN_TO = SPAN_START, date(2027, 1, 10)
VENUES = ("TOK", "KYO")
RACE_NOS = (10, 11, 12)


def weekend_days(first: date, last: date) -> list[date]:
    out, d = [], first
    while d <= last:
        if d.weekday() >= 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def synth_race(rng, day: date, venue: str, race_no: int) -> Race:
    n = int(rng.integers(8, 11))
    horses = list(range(1, n + 1))
    if rng.random() < 0.35:  # a concentrated market: one dominant favourite
        rest = rng.gamma(2.0, size=n - 1)
        p = np.r_[0.65, 0.35 * rest / rest.sum()]
        p = p[rng.permutation(n)]
    else:
        p = rng.gamma(2.0, size=n)
        p /= p.sum()

    def board(noise: float) -> dict[int, float]:
        q = p * np.exp(rng.normal(0.0, noise, size=n))
        q /= q.sum()
        return {h: float(max(round(0.8 / v, 1), 1.1)) for h, v in zip(horses, q, strict=True)}

    t_minutes = 2.0 if rng.random() < 0.5 else 3.0
    final = board(0.15)
    winner = int(rng.choice(horses, p=p))
    return Race(
        day=day, venue=venue, race_no=race_no,
        weight_type="handicap" if rng.random() < 0.35 else "set_weights",
        horses=horses,
        captures=[(8.0, board(0.3)), (t_minutes, board(0.1)), (1.0, board(0.3))],
        final=final, winner=winner, dividend=round(final[winner] * 100 + 20),
    )  # fmt: skip


def run_races() -> tuple[list[Race], dict[str, date]]:
    rng = np.random.default_rng(4620)
    days = weekend_days(date(2026, 9, 5), date(2027, 1, 10))
    after = next(d for d in weekend_days(RUN_TO + timedelta(days=1), RUN_TO + timedelta(days=30))
                 if not is_holdout_day("japan", d))  # fmt: skip
    special = {"early": EARLY_DAY, "after": after}
    races = [synth_race(rng, d, v, no) for d in [*days, EARLY_DAY, after]
             for v in VENUES for no in RACE_NOS]  # fmt: skip
    return races, special


@pytest.fixture(scope="module")
def run_store(tmp_path_factory) -> SimpleNamespace:
    races, special = run_races()
    held = {r.day for r in races if is_holdout_day("japan", r.day)}
    assert HOLDOUT_DAY in held
    store = write_store(tmp_path_factory.mktemp("selection_run") / "data", races, garbage_days=held)
    store.held = held
    store.forbidden = [store.files[d] for d in (*held, special["early"], special["after"])]
    store.dev = [r for r in races if r.day not in held and RUN_FROM <= r.day <= RUN_TO]
    return store


def holdout_log_bytes() -> bytes | None:
    log = DEFAULT_COUNTRIES_DIR / "japan" / LOG_NAME
    return log.read_bytes() if log.exists() else None


def repo_state() -> dict[str, tuple[int, int]]:
    root = env.repo_root()
    state = {}
    for top in ("docs", "src"):
        for path in (root / top).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                stat = path.stat()
                state[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return state


@pytest.fixture(scope="module")
def main_run(run_store, tmp_path_factory) -> SimpleNamespace:
    """One run over the store with every forbidden day file raising on open, the registry's state
    noted at the first store read and at the first scoring fit, and the run_study and
    write_report calls recorded."""
    root = tmp_path_factory.mktemp("selection_main")
    registry = new_registry(root)
    out_dir = run_store.data / "reports" / "selection"
    holdout = TrippedHoldout()
    seen = SimpleNamespace(first_read=None, first_fit=None, study_calls=[], report_calls=[])
    forbidden = forbid(run_store.forbidden)
    in_store = under(run_store.processed)

    def on_open(path: Path) -> None:
        forbidden(path)
        if seen.first_read is None and in_store(path):
            seen.first_read = (registry.frozen_spec_hash(FAMILY_ID), registry.records(FAMILY_ID))

    real_fit = selection.fit_win_calibration

    def fit_spy(*args, **kwargs):
        if seen.first_fit is None:
            seen.first_fit = (registry.frozen_spec_hash(FAMILY_ID), registry.records(FAMILY_ID))
        return real_fit(*args, **kwargs)

    real_study, real_write = selection_run.run_study, selection_run.write_report

    def study_spy(spec, data, **kwargs):
        seen.study_calls.append((spec, data.copy(), kwargs))
        return real_study(spec, data, **kwargs)

    def write_spy(path, text, **kwargs):
        seen.report_calls.append((Path(path), kwargs))
        return real_write(path, text, **kwargs)

    log_before = holdout_log_bytes()
    repo_before = repo_state()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HORSETRACK_DATA_DIR", str(run_store.data))
        mp.setattr(selection, "load_country_scheme", lambda country: SHORT)
        if hasattr(selection_run, "load_country_scheme"):
            mp.setattr(selection_run, "load_country_scheme", lambda country: SHORT)
        mp.setattr(selection, "fit_win_calibration", fit_spy)
        mp.setattr(selection_run, "run_study", study_spy)
        mp.setattr(selection_run, "write_report", write_spy)

        def never_open(self, **kwargs):
            raise HoldoutOpened(f"LockedHoldout.open was called: {kwargs}")

        mp.setattr(LockedHoldout, "open", never_open)
        install_file_spy(mp, on_open)
        try:
            result = run_selection(date_from=RUN_FROM, date_to=RUN_TO, out_dir=out_dir,
                                   seed=SEED, registry=registry, holdout=holdout)  # fmt: skip
        except ForbiddenRead as exc:
            pytest.fail(f"the run opened a day file it must never open: {exc}")
    return SimpleNamespace(
        result=result,
        registry=registry,
        holdout=holdout,
        out_dir=out_dir,
        seen=seen,
        log_changed=holdout_log_bytes() != log_before,
        repo_changed=sorted(
            set(repo_before.items()) ^ set(repo_state().items()), key=lambda kv: kv[0]
        ),
    )


def test_the_family_is_frozen_with_its_dropped_variants_before_any_row_is_read(main_run):
    expected = spec_hash(load_family("japan"))
    frozen, records = main_run.seen.first_read
    assert frozen == expected, "the store was read before the family was frozen"
    frozen, records = main_run.seen.first_fit
    assert frozen == expected
    assert {r.variant for r in records if r.status == "failed"} >= set(DROPPED), (
        "the dropped variants were not recorded before the first scoring fit"
    )
    assert main_run.registry.frozen_spec_hash(FAMILY_ID) == expected


def test_the_family_size_counts_every_rule_and_dropped_variant(main_run):
    records = main_run.registry.records(FAMILY_ID)
    for variant in DROPPED:
        mine = [r for r in records if r.variant == variant]
        assert len(mine) == 1 and mine[0].status == "failed", variant
    assert {r.variant for r in records} == {*RULE_IDS, *DROPPED}
    size = len(RULE_IDS) + len(DROPPED)
    assert main_run.registry.family_size(FAMILY_ID) == size
    assert main_run.result.study.family_size == size
    for r in records:
        if r.variant in RULE_IDS:
            assert "jravan" in r.sources and "jra" in r.sources, r.variant


def test_the_run_reads_development_days_only_and_never_opens_the_holdout(run_store, main_run):
    assert main_run.holdout.calls == []
    assert not main_run.log_changed
    assert len(main_run.seen.study_calls) == 1
    spec, data, kwargs = main_run.seen.study_calls[0]
    assert spec == load_family("japan")
    assert kwargs["registry"] is main_run.registry and kwargs["holdout"] is main_run.holdout
    assert set(kwargs["sources"]) >= {"jra", "jravan"}
    keys = set(zip(data["race_id"].astype(str), data["horse_no"].astype(int), strict=True))
    want = {(r.race_id, h) for r in run_store.dev for h in r.starters}
    assert keys == want
    days = set(pd.to_datetime(data["race_date"]).dt.date)
    assert not any(is_holdout_day("japan", d) for d in days)
    assert min(days) >= RUN_FROM and max(days) <= RUN_TO


def test_the_private_report_states_size_p_and_rungs_and_stays_out_of_the_repo(main_run):
    study = main_run.result.study
    assert set(study.rungs) == set(RULE_IDS) and set(study.rungs.values()) <= {"S0", "S1"}
    assert set(study.p_beats_takeout) == set(RULE_IDS)
    assert any(np.isfinite(p) for p in study.p_beats_takeout.values()), "no rule reached omega"

    assert len(main_run.seen.report_calls) >= 1
    path = Path(main_run.result.report_path)
    called = {real(p): kw for p, kw in main_run.seen.report_calls}
    assert real(path) in called, "the report was not written through write_report"
    assert "jravan" in called[real(path)]["sources"]
    assert under(main_run.out_dir)(path) and path.is_file()
    assert not under(env.repo_root())(path)

    text = path.read_text(encoding="utf-8")
    assert re.search(rf"(?<![\d.]){study.family_size}(?![\d.])", text), "family size missing"
    assert not re.search(r"\bS[2-9]\b", text), "a rung above S1"
    lines = text.splitlines()
    for rid in RULE_IDS:
        mine = [line for line in lines if rid in line]
        assert mine, rid
        assert any(re.search(rf"\b{study.rungs[rid]}\b", line) for line in mine), rid
        p = study.p_beats_takeout[rid]
        if np.isfinite(p):
            numbers = [float(x) for line in mine for x in re.findall(r"\d+(?:\.\d+)?", line)]
            assert any(abs(x - p) <= 0.006 or abs(x - 100 * p) <= 0.6 for x in numbers), rid
    assert main_run.repo_changed == [], "the run wrote inside the repository"


def test_a_run_before_the_price_era_is_refused_before_anything(run_store, tmp_path, monkeypatch):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(run_store.data))
    registry = new_registry(tmp_path)
    opened: list[Path] = []
    inside = under(run_store.data)

    def on_open(path: Path) -> None:
        if inside(path):
            opened.append(path)
            raise ForbiddenRead(str(path))

    install_file_spy(monkeypatch, on_open)
    with pytest.raises(PreReformPriceError):
        run_selection(date_from=EARLY_DAY, date_to=RUN_TO, out_dir=tmp_path / "out", seed=SEED,
                      registry=registry, holdout=TrippedHoldout())  # fmt: skip
    monkeypatch.undo()
    assert opened == []
    assert registry.records(FAMILY_ID) == []
