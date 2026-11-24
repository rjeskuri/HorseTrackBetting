"""Past-performance (PP) lines as of decision time (ticket #128).

Contract pinned here (`horsetrack.countries.japan.handicap.pp`):

- `pp_lines(race_id, *, as_of, n=10) -> polars.DataFrame`. `race_id` is a store race id
  (`jra:<YYYY-MM-DD>:<VENUE>:<NN>`); `as_of` is a timezone-aware datetime in any zone (the instant
  counts).
- Shape: one row per prior start of each declared runner of the race, newest first, at most `n`
  per runner. Key columns `race_id` (the race asked for), `horse_no`, `horse_id` (the runner in
  it), `line_no` (1 = newest prior start) and `reason`; then the line columns in `LINE_COLUMNS`.
  Extra columns are allowed. A runner's rows appear in `line_no` order. Every call returns the
  same schema, whatever the race (so frames of different races concatenate).
- A runner with no prior start gets exactly one row: `line_no` null, every line column null,
  `reason` "debut". A runner with lines has `reason` null on each of them.
- A prior start is a `runners` row of the same `horse_id`, not scratched, whose `race_date` is
  before the race's own date (a same-day start, the race itself and later starts never count) and
  whose result was published before `as_of`.
- Published (Ruling): the store holds no publication time, so a start's result counts as published
  at the end of its local race day, (race_date + 1 day) 00:00 Asia/Tokyo; the start is included
  when that instant is strictly before `as_of`.
- Line columns come from the store as they are: `pp_race_id` (the start's race id), `pp_date` (its
  race_date, a date), from `races`: `venue, surface, distance_m, course_config, going,
  race_class, field_size`, `winner_time_s` (= races.finish_time_s); from `runners`: `draw,
  actual_wt_kg, horse_wt_kg, horse_wt_change_kg, jockey_key, jockey_name, trainer_key,
  trainer_name, finish_pos, lbw_lengths, finish_time_s, last_3f_s, running_positions` (the corner
  positions), `win_odds` (final), `win_fav_rank` (popularity). Derived: `last_3f_rank`, the
  start's rank of last_3f_s among that race's runners that have one (1 = fastest, ties share the
  lowest rank); `days_since_prev`, the days since the horse's previous prior start, even one beyond
  the `n` cap (null for its first).
- Reads go through #114's scoped reader (`read_store_table` on `races` and `runners`), so no holdout
  day is read. The tests mark a past day holdout by patching `is_holdout_day` as test_v0_store does;
  a direct Parquet read would not see the patch.
- `build_pp_table(date_from, date_to) -> dict[date, Path]`: for each race day in the inclusive range
  that has races in the store and is not a holdout day, one Parquet file under the data dir whose
  rows are the day's races' `pp_lines` (default n) as of the race day; under the ruling these equal
  `pp_lines(race_id, as_of=<any instant on that local race day>)`. Each returned path contains the
  day's ISO date; no other Parquet file appears under the data dir.

Data: synthetic only, written under tmp_path; HORSETRACK_DATA_DIR points there (monkeypatched env
var, as in tests/countries/japan/models/test_v0_store.py), with `races.parquet` and
`runners.parquet` in `processed/jra` holding every declared column of `core.store.schemas`.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import random
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

from horsetrack.core.eval import holdout
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.handicap import pp
from horsetrack.countries.japan.handicap.pp import build_pp_table, pp_lines
from horsetrack.countries.japan.models.v0 import store as v0_store

DATA_VAR = "HORSETRACK_DATA_DIR"
JST = ZoneInfo("Asia/Tokyo")
VENUES = ["TOK", "NAK", "KYO", "HSN", "SAP"]

LINE_COLUMNS = [
    "pp_race_id", "pp_date", "venue", "surface", "distance_m", "course_config", "going",
    "race_class", "field_size", "draw", "actual_wt_kg", "horse_wt_kg", "horse_wt_change_kg",
    "jockey_key", "jockey_name", "trainer_key", "trainer_name", "finish_pos", "lbw_lengths",
    "finish_time_s", "last_3f_s", "last_3f_rank", "running_positions", "win_odds",
    "win_fav_rank", "days_since_prev", "winner_time_s",
]  # fmt: skip
KEY_COLUMNS = ["race_id", "horse_no", "horse_id", "line_no", "reason"]


def rid(day: date, venue: str, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{race_no:02d}"


def at_jst(day: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm, ss), tzinfo=JST)


def published(day: date) -> datetime:
    """The ruling: a result counts as published at the end of its local race day."""
    return datetime.combine(day + timedelta(days=1), time(0), tzinfo=JST)


class SyntheticStore:
    """Store-shaped races and runners rows, plus the truth each tracked horse's lines come from."""

    def __init__(self):
        self.races: list[dict] = []
        self.runners: list[dict] = []
        self.starts: dict[str, list[dict]] = {}  # horse_id -> its entries, scratched ones included
        self.races_by_day: dict[date, list[str]] = {}

    def _race_row(self, race_id, day, venue, race_no, **fields):
        if any(r["race_id"] == race_id for r in self.races):
            raise AssertionError(f"fixture bug: duplicate race {race_id}")
        self.races_by_day.setdefault(day, []).append(race_id)
        self.races.append(
            {
                "jurisdiction": "jra", "race_id": race_id, "meeting_id": f"jra:{day}:{venue}",
                "race_date": day, "venue": venue, "race_no": race_no, "status": "run",
                "post_time_local": f"{9 + race_no // 2:02d}:{(race_no * 25) % 60:02d}", **fields,
            }
        )  # fmt: skip

    def _runner_row(self, race_id, day, venue, race_no, **fields):
        self.runners.append(
            {
                "jurisdiction": "jra", "race_id": race_id, "meeting_id": f"jra:{day}:{venue}",
                "race_date": day, "venue": venue, "race_no": race_no, "dead_heat": False,
                **fields,
            }
        )  # fmt: skip

    def past_race(self, horse_id, k, day, *, horse_wt, wt_change, scratched=False, tie=False):
        """A run race with `horse_id` in it plus a filler field; k varies every value."""
        venue = VENUES[k % 5]
        race_no = 1 + k % 12
        race_id = rid(day, venue, race_no)
        field = 8 + (k * 3) % 9
        distance = [1200, 1400, 1600, 1800, 2000, 2400][k % 6]
        winner_time = round(distance * 0.06 + 0.1 * (k % 7), 1)
        race = {
            "surface": "dirt" if k % 3 == 1 else "turf",
            "distance_m": distance,
            "course_config": ["inner", "outer", None][(k // 2) % 3],
            "going": ["GOOD", "GOOD TO YIELDING", "SOFT", "HEAVY"][k % 4],
            "race_class": ["maiden", "1win", "2win", "3win", "open", "G3"][(k + 1) % 6],
            "field_size": field,
            "finish_time_s": winner_time,
        }
        self._race_row(race_id, day, venue, race_no, **race)

        finish = 1 + (k * 7) % field
        horse_no = 1 + (k * 5) % field
        rank3f = 1 + (k * 3) % 5
        last3f = round(33.0 + 0.3 * (k % 9), 1)
        corners = 2 if distance <= 1400 else 4
        subject = {
            "horse_id": horse_id, "horse_no": horse_no, "horse_name": horse_id.upper(),
            "draw": horse_no, "bracket": 1 + (horse_no - 1) // 2,
            "actual_wt_kg": 54.0 + 0.5 * (k % 7), "horse_wt_kg": horse_wt,
            "horse_wt_change_kg": wt_change,
            "jockey_key": f"jockey:{1100 + k % 4:05d}", "jockey_name": f"Jockey {'ABCD'[k % 4]}",
            "trainer_key": f"trainer:{2200 + k % 3:05d}", "trainer_name": f"Trainer {'XYZ'[k % 3]}",
            "finish_pos": finish, "placing_raw": str(finish),
            "lbw_lengths": 0.0 if finish == 1 else 0.5 * (finish - 1) + 0.25 * (k % 2),
            "finish_time_s": round(winner_time + 0.2 * (finish - 1), 1),
            "last_3f_s": last3f,
            "running_positions": [1 + (k + 2 * c + finish) % field for c in range(corners)],
            "win_odds": round(1.8 + 1.3 * (k % 11), 1), "win_fav_rank": 1 + (k * 2) % field,
            "is_scratched": False,
        }  # fmt: skip
        if scratched:
            subject.update(
                finish_pos=None, placing_raw="取消", lbw_lengths=None, finish_time_s=None,
                last_3f_s=None, running_positions=[], win_odds=None, win_fav_rank=None,
                is_scratched=True,
            )  # fmt: skip
        self._runner_row(race_id, day, venue, race_no, **subject)

        # The filler field: the other horse numbers and finishing places; rank3f - 1 of them
        # finish faster over the last 3F than the subject, the rest slower (one tied when `tie`).
        numbers = [h for h in range(1, field + 1) if h != horse_no]
        places = [p for p in range(1, field + 1) if p != finish]
        if scratched:  # the subject is out: the fillers take every place of the field
            numbers = [h for h in range(1, field + 2) if h != horse_no]
            places = list(range(1, field + 1))
        for j, (no, place) in enumerate(zip(numbers, places, strict=True)):
            if j < rank3f - 1:
                f3 = round(last3f - 0.1 * (j + 1), 1)
            elif tie and j == rank3f - 1:
                f3 = last3f
            else:
                f3 = round(last3f + 0.1 * (j + 1), 1)
            self._runner_row(
                race_id, day, venue, race_no,
                horse_id=f"f:{race_id}:{no}", horse_no=no, draw=no, horse_name=f"FILLER {no}",
                finish_pos=place, placing_raw=str(place),
                finish_time_s=round(winner_time + 0.2 * (place - 1), 1),
                lbw_lengths=0.0 if place == 1 else 0.5 * (place - 1), last_3f_s=f3,
                win_odds=3.0 + no, win_fav_rank=no, running_positions=[place, place],
                jockey_key=f"jockey:{9000 + no:05d}", jockey_name=f"Jockey F{no}",
                trainer_key=f"trainer:{9000 + no:05d}", trainer_name=f"Trainer F{no}",
                actual_wt_kg=55.0, horse_wt_kg=460 + no, horse_wt_change_kg=0, is_scratched=False,
            )  # fmt: skip
        if k % 4 == 0 and not scratched:  # a scratched filler: no last 3F, not in the field
            self._runner_row(
                race_id, day, venue, race_no, horse_id=f"f:{race_id}:x", horse_no=field + 1,
                draw=field + 1, finish_pos=None, placing_raw="除外", last_3f_s=None,
                is_scratched=True,
            )  # fmt: skip

        truth = {
            "scratched": scratched, "pp_race_id": race_id, "pp_date": day, "venue": venue,
            **{c: race[c] for c in ("surface", "distance_m", "course_config", "going",
                                    "race_class", "field_size")},
            "winner_time_s": winner_time, "last_3f_rank": rank3f,
            **{c: subject[c] for c in ("draw", "actual_wt_kg", "horse_wt_kg",
                                       "horse_wt_change_kg", "jockey_key", "jockey_name",
                                       "trainer_key", "trainer_name", "finish_pos",
                                       "lbw_lengths", "finish_time_s", "last_3f_s",
                                       "running_positions", "win_odds", "win_fav_rank")},
        }  # fmt: skip
        self.starts.setdefault(horse_id, []).append(truth)
        return race_id

    def target_race(self, race_id, day, venue, race_no, horses: dict[str, int]):
        """A race whose own result is in the store (bait: it must never be a line)."""
        self._race_row(
            race_id, day, venue, race_no, surface="turf", distance_m=1600, field_size=len(horses),
            going="GOOD", race_class="open", finish_time_s=95.0,
        )  # fmt: skip
        for place, (horse_id, no) in enumerate(horses.items(), start=1):
            self._runner_row(
                race_id, day, venue, race_no, horse_id=horse_id, horse_no=no, draw=no,
                finish_pos=place, placing_raw=str(place), finish_time_s=95.0 + 0.2 * (place - 1),
                last_3f_s=34.0 + 0.1 * place, win_odds=2.0 * place, win_fav_rank=place,
                running_positions=[place, place, place, place], is_scratched=False,
                jockey_name="Jockey T", trainer_name="Trainer T", actual_wt_kg=56.0,
                horse_wt_kg=480, horse_wt_change_kg=0, lbw_lengths=0.5 * (place - 1),
            )  # fmt: skip
            self.starts.setdefault(horse_id, []).append(
                {"scratched": False, "pp_race_id": race_id, "pp_date": day, "bait": True}
            )

    def write(self, tmp_path, monkeypatch) -> Path:
        data = (tmp_path / "data").resolve()
        monkeypatch.setenv(DATA_VAR, str(data))
        jra = data / "processed" / "jra"
        jra.mkdir(parents=True)
        for name, rows in (("races", self.races), ("runners", self.runners)):
            spec = schemas.TABLES[name].columns
            schema = pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in spec.items()])
            shuffled = list(rows)
            random.Random(7).shuffle(shuffled)  # nothing may rely on the file's row order
            pq.write_table(pa.Table.from_pylist(shuffled, schema=schema), jra / f"{name}.parquet")
        return data


# ---------------------------------------------------------------------------------------------
# The main synthetic history: race day T, target race TOK 11 with four runners.
#   A (no 3): 12 starts 2024-03-02 .. 2025-05-31, a scratched entry, a start earlier on T (TOK 02)
#             and one the day after T (TOK 05); one start's last 3F is tied.
#   B (no 7): a debutant.
#   C (no 1): 3 starts, the newest a week before T.
#   D (no 5): 2 starts, the newest the day before T (the as-of cut).
# ---------------------------------------------------------------------------------------------

T = date(2025, 6, 15)
TARGET = rid(T, "TOK", 11)
TARGET_RUNNERS = {"h:alpha": 3, "h:bravo": 7, "h:charlie": 1, "h:delta": 5}
AS_OF = at_jst(T, 15, 0)  # race day, mid-afternoon: after TOK 02 has been run
A_GAPS = [21, 35, 14, 56, 28, 70, 21, 42, 28, 14, 126]
A_DATES = [date(2024, 3, 2) + timedelta(days=sum(A_GAPS[:i])) for i in range(12)]
A_WT_CHANGES = [None, 4, -2, 0, 6, -8, 2, 10, -4, 0, 2, -6]
C_DATES = [date(2025, 1, 12), date(2025, 3, 9), date(2025, 6, 8)]
D_DATES = [date(2025, 5, 18), date(2025, 6, 14)]


def build_main_store() -> SyntheticStore:
    s = SyntheticStore()
    wt = 470
    for i, day in enumerate(A_DATES):
        wt += A_WT_CHANGES[i] or 0
        s.past_race("h:alpha", i, day, horse_wt=wt, wt_change=A_WT_CHANGES[i], tie=(i == 4))
    s.past_race("h:alpha", 70, A_DATES[5] + timedelta(days=10), horse_wt=None, wt_change=None,
                scratched=True)  # fmt: skip
    s.past_race("h:alpha", 60, T, horse_wt=wt, wt_change=0)  # same day, earlier race
    s.past_race("h:alpha", 61, T + timedelta(days=1), horse_wt=wt, wt_change=0)  # after T
    for i, day in enumerate(C_DATES):
        s.past_race("h:charlie", 20 + i, day, horse_wt=500 + 2 * i, wt_change=None if i == 0 else 2)
    for i, day in enumerate(D_DATES):
        s.past_race("h:delta", 40 + i, day, horse_wt=440 - 4 * i, wt_change=None if i == 0 else -4)
    s.target_race(TARGET, T, "TOK", 11, TARGET_RUNNERS)
    return s


def expected_lines(s, horse_id, *, race_day, as_of, n=10, hidden=frozenset()):
    """A horse's PP lines derived from the fixture's truth alone."""
    starts = [
        st
        for st in s.starts.get(horse_id, [])
        if not st["scratched"]
        and st["pp_date"] < race_day
        and published(st["pp_date"]) < as_of
        and st["pp_date"] not in hidden
    ]
    assert not any(st.get("bait") for st in starts)
    starts.sort(key=lambda st: st["pp_date"], reverse=True)
    lines = []
    for j, st in enumerate(starts[:n]):
        prev = starts[j + 1]["pp_date"] if j + 1 < len(starts) else None
        line = {c: st[c] for c in LINE_COLUMNS if c != "days_since_prev"}
        line["days_since_prev"] = (st["pp_date"] - prev).days if prev is not None else None
        lines.append({"line_no": j + 1, "reason": None, **line})
    return lines


def lines_of(frame: pl.DataFrame, horse_id: str) -> list[dict]:
    missing = [c for c in KEY_COLUMNS + LINE_COLUMNS if c not in frame.columns]
    assert not missing, f"pp_lines lacks columns {missing}"
    rows = frame.filter(pl.col("horse_id") == horse_id)
    return rows.select(["line_no", "reason", *LINE_COLUMNS]).to_dicts()


def patch_holdout(monkeypatch, days):
    """Mark exactly `days` as holdout wherever the reader (or pp) looks is_holdout_day up."""
    marked = set(days)

    def fake(country, day):
        return day in marked

    monkeypatch.setattr(holdout, "is_holdout_day", fake)
    monkeypatch.setattr(v0_store, "is_holdout_day", fake, raising=False)
    monkeypatch.setattr(pp, "is_holdout_day", fake, raising=False)


@pytest.fixture
def main_store(tmp_path, monkeypatch):
    s = build_main_store()
    s.write(tmp_path, monkeypatch)
    return s


# ---------------------------------------------------------------------------------------------
# DoD: for a synthetic horse with known history, the PP lines match it exactly, newest first,
# capped at n
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [None, 1, 3, 10, 11, 12, 20])
def test_known_history_matches_exactly_newest_first_capped_at_n(main_store, n):
    """A has 12 prior starts (plus a scratched entry, which is not a start), C has 3; n=None is the
    default, 10. The oldest line kept under the cap still counts its days from the start beyond
    the cap."""
    kwargs = {} if n is None else {"n": n}
    frame = pp_lines(TARGET, as_of=AS_OF, **kwargs)
    assert isinstance(frame, pl.DataFrame)
    cap = 10 if n is None else n
    for horse_id in ("h:alpha", "h:charlie", "h:delta"):
        want = expected_lines(main_store, horse_id, race_day=T, as_of=AS_OF, n=cap)
        assert lines_of(frame, horse_id) == want, horse_id
    assert len(lines_of(frame, "h:alpha")) == min(cap, 12)
    assert len(lines_of(frame, "h:charlie")) == min(cap, 3)


def test_every_runner_of_the_race_is_covered(main_store):
    frame = pp_lines(TARGET, as_of=AS_OF)
    assert set(frame["race_id"].to_list()) == {TARGET}
    pairs = frame.select("horse_id", "horse_no").unique().iter_rows()
    assert set(pairs) == set(TARGET_RUNNERS.items())
    counts = dict(frame.group_by("horse_id").len().iter_rows())
    assert counts == {"h:alpha": 10, "h:bravo": 1, "h:charlie": 3, "h:delta": 2}
    lined = frame.filter(pl.col("horse_id") != "h:bravo")
    assert lined["reason"].null_count() == lined.height
    assert lined["line_no"].null_count() == 0


# ---------------------------------------------------------------------------------------------
# DoD: a debutant yields an empty PP with reason `debut`
# ---------------------------------------------------------------------------------------------


def debut_row():
    return {"line_no": None, "reason": "debut", **dict.fromkeys(LINE_COLUMNS)}


@pytest.mark.parametrize("n", [None, 1, 10])
def test_a_debutant_gets_one_row_with_reason_debut(main_store, n):
    kwargs = {} if n is None else {"n": n}
    frame = pp_lines(TARGET, as_of=AS_OF, **kwargs)
    assert lines_of(frame, "h:bravo") == [debut_row()]


def test_a_horse_is_a_debutant_in_its_own_first_race(main_store):
    """A's first race: A and every filler have no prior start, so every runner is a debutant."""
    first = main_store.starts["h:alpha"][0]["pp_race_id"]
    frame = pp_lines(first, as_of=at_jst(A_DATES[0], 9))
    declared = {
        r["horse_id"] for r in main_store.runners if r["race_id"] == first and not r["is_scratched"]
    }
    assert len(declared) >= 8
    for horse_id in declared:
        assert lines_of(frame, horse_id) == [debut_row()], horse_id
    assert set(frame["reason"].to_list()) == {"debut"}


# ---------------------------------------------------------------------------------------------
# DoD: a start whose result is published after `as_of` is excluded, and so is a same-day start
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_of",
    [
        at_jst(date(2025, 5, 19), 12),  # D's 05-18 published; A's 05-31 not yet run
        at_jst(date(2025, 6, 1), 0, 0, 1),  # A's 05-31 just published
        at_jst(date(2025, 6, 14), 12),  # D's 06-14 race not yet run
        at_jst(date(2025, 6, 14), 23, 59, 59),  # D's 06-14 run, not yet published (ruling)
        datetime(2025, 6, 14, 15, 0, 1, tzinfo=timezone.utc),  # 06-15 00:00:01 JST: published
        AS_OF,
    ],
    ids=["0519", "0601-boundary", "0614-noon", "0614-2359-jst", "0615-0000-utc", "race-day"],
)
def test_a_start_published_after_as_of_is_excluded(main_store, as_of):
    frame = pp_lines(TARGET, as_of=as_of)
    for horse_id in ("h:alpha", "h:charlie", "h:delta"):
        want = expected_lines(main_store, horse_id, race_day=T, as_of=as_of)
        assert lines_of(frame, horse_id) == want, horse_id


def test_the_as_of_cut_changes_the_lines(main_store):
    """The cut is live: D's 06-14 start is a line only once published."""
    before = lines_of(pp_lines(TARGET, as_of=at_jst(date(2025, 6, 14), 23, 59, 59)), "h:delta")
    after = lines_of(pp_lines(TARGET, as_of=at_jst(date(2025, 6, 15), 0, 0, 1)), "h:delta")
    assert [line["pp_date"] for line in before] == [date(2025, 5, 18)]
    assert [line["pp_date"] for line in after] == [date(2025, 6, 14), date(2025, 5, 18)]
    assert after[0]["days_since_prev"] == 27


@pytest.mark.parametrize(
    "as_of",
    [AS_OF, at_jst(T + timedelta(days=2), 12), at_jst(T + timedelta(days=60), 12)],
    ids=["race-day", "two-days-on", "two-months-on"],
)
def test_same_day_and_later_starts_are_never_lines(main_store, as_of):
    """A's earlier race on T, the target race's own result and A's start the day after T are all
    in the store and (with a late as_of) published; none is a prior start."""
    frame = pp_lines(TARGET, as_of=as_of)
    alpha = lines_of(frame, "h:alpha")
    assert alpha == expected_lines(main_store, "h:alpha", race_day=T, as_of=as_of)
    assert all(line["pp_date"] < T for line in alpha)
    late = {st["pp_race_id"] for st in main_store.starts["h:alpha"] if st["pp_date"] >= T}
    assert len(late) == 3 and TARGET in late
    assert set(frame["pp_race_id"].drop_nulls().to_list()).isdisjoint(late)


# ---------------------------------------------------------------------------------------------
# Holdout: no holdout day is read (#71's days; 2026-11-28 on)
# ---------------------------------------------------------------------------------------------


def test_a_holdout_day_is_never_read(main_store, monkeypatch):
    """A past day marked holdout (patched, as test_v0_store does) is dropped by the scoped reader;
    only a read through it can drop it."""
    hidden = A_DATES[-2]
    patch_holdout(monkeypatch, [hidden])
    frame = pp_lines(TARGET, as_of=AS_OF, n=20)
    got = [line["pp_race_id"] for line in lines_of(frame, "h:alpha")]
    truth = expected_lines(main_store, "h:alpha", race_day=T, as_of=AS_OF, n=20, hidden={hidden})
    want = [line["pp_race_id"] for line in truth]
    assert got == want
    assert len(got) == 11


def test_committed_holdout_days_from_2026_11_28_are_never_read(tmp_path, monkeypatch):
    """Unpatched: a start on a day #71's committed rule marks holdout (on or after 2026-11-28) is
    not a line, and build_pp_table writes no partition for that day."""
    hold = next(
        date(2026, 11, 28) + timedelta(days=k)
        for k in range(60)
        if holdout.is_holdout_day("japan", date(2026, 11, 28) + timedelta(days=k))
    )
    race_day = next(
        hold + timedelta(days=k)
        for k in range(1, 60)
        if not holdout.is_holdout_day("japan", hold + timedelta(days=k))
    )
    dev = date(2026, 11, 21)
    s = SyntheticStore()
    s.past_race("h:echo", 80, dev, horse_wt=470, wt_change=None)
    s.past_race("h:echo", 81, hold, horse_wt=474, wt_change=4)
    target = rid(race_day, "TOK", 11)
    s.target_race(target, race_day, "TOK", 11, {"h:echo": 2, "h:foxtrot": 4})
    s.write(tmp_path, monkeypatch)

    frame = pp_lines(target, as_of=at_jst(race_day, 9))
    assert [line["pp_date"] for line in lines_of(frame, "h:echo")] == [dev]
    assert lines_of(frame, "h:foxtrot") == [debut_row()]

    written = build_pp_table(dev, race_day)
    assert set(written) == {dev, race_day}
    for path in written.values():
        assert hold.isoformat() not in str(path)
        days = pl.read_parquet(path)["pp_date"].drop_nulls().to_list()
        assert hold not in days


# ---------------------------------------------------------------------------------------------
# build_pp_table: one Parquet partition per race day under the data dir, equal to pp_lines
# ---------------------------------------------------------------------------------------------

KEY_ORDER = ["race_id", "horse_no", "line_no"]


def day_expected(s, day) -> pl.DataFrame:
    frames = [pp_lines(race_id, as_of=at_jst(day, 9)) for race_id in sorted(s.races_by_day[day])]
    return pl.concat(frames).sort(KEY_ORDER, nulls_last=True)


def parquet_outside_store(data: Path) -> set[Path]:
    jra = data / "processed" / "jra"
    store_files = {(jra / "races.parquet").resolve(), (jra / "runners.parquet").resolve()}
    return {p.resolve() for p in data.rglob("*.parquet")} - store_files


@pytest.mark.parametrize(
    "date_from, date_to",
    [
        (date(2025, 6, 14), date(2025, 6, 16)),
        (date(2025, 6, 8), date(2025, 6, 14)),
        (T, T),
        (date(2025, 6, 9), date(2025, 6, 13)),  # no race day in range
    ],
)
def test_build_writes_one_partition_per_race_day_equal_to_pp_lines(
    tmp_path, monkeypatch, date_from, date_to
):
    s = build_main_store()
    data = s.write(tmp_path, monkeypatch)
    want_days = sorted(d for d in s.races_by_day if date_from <= d <= date_to)

    written = build_pp_table(date_from, date_to)

    assert sorted(written) == want_days
    assert parquet_outside_store(data) == {Path(p).resolve() for p in written.values()}
    repo = Path(__file__).resolve().parents[4]
    for day, path in written.items():
        path = Path(path).resolve()
        assert path.is_relative_to(data) and not path.is_relative_to(repo)
        assert day.isoformat() in str(path)
        got = pl.read_parquet(path).sort(KEY_ORDER, nulls_last=True)
        assert_frame_equal(got, day_expected(s, day))


def test_build_skips_a_holdout_day(tmp_path, monkeypatch):
    s = build_main_store()
    data = s.write(tmp_path, monkeypatch)
    patch_holdout(monkeypatch, [T + timedelta(days=1)])
    written = build_pp_table(date(2025, 6, 14), date(2025, 6, 16))
    assert sorted(written) == [date(2025, 6, 14), T]
    assert len(parquet_outside_store(data)) == 2
