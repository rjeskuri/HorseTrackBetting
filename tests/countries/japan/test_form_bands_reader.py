"""#218 review: the real-data reader of the form-bands family counts every race it leaves out.

Module under test: `horsetrack.countries.japan.studies.form_bands.read_rows(date_from, date_to,
scheme) -> (rows, folds, counts, builds)`, which reads, under `$HORSETRACK_DATA_DIR`:

- `V0_OOF` (`race_id`, `horse_no`, `fold`, `p_cal`): v0's out-of-fold starters. v0's generator
  leaves out first-place dead heats, races without a priced winner and starters without a final
  price, so a race or runner absent here is not thereby void or scratched;
- the JRA store, `processed/jra/{races,runners}.parquet` (races: `race_id`, `race_date`,
  `meeting_id`; runners: `race_id`, `race_date`, `horse_no`, `finish_pos`, `is_scratched`,
  `win_odds`), the outcomes and the scratch flags;
- `features/jra/handicap/<day>.parquet` (`race_id`, `horse_no`, `handicap_score`): h;
- `processed/jravan/odds_timeseries/<day>.parquet` (`logger.compact.SCHEMA`): the T-2 Win board.

Contract pinned here (the review's findings on form_bands.py:684 and :745):
- The scoring population is the store's races on the days read. A store race absent from v0 is
  left out of `rows` and counted by its reason: `counts["dropped_dead_heat"]` (two or more
  runners with finish_pos 1), `counts["dropped_no_priced_winner"]` (a winner, but none with a
  final price) and `counts["dropped_void"]` (no runner with finish_pos 1).
- A confirmed starter (store `is_scratched` False) with no v0 row takes its race out of `rows`,
  counted in `counts["dropped_starter_without_v0"]`, whether or not it is priced at T-2: it is
  never read as a scratch.
- A runner the store flags scratched and the T-2 board still prices (scratched after T-2) stays a
  row of its race with `scratched` True and its `win_odds_t2`, so it ranks on the T-2 board
  (`assign_groups`); it is not a starter and is not scored. Its race is kept.
- Every other race is in `rows` with all its starters, none scratched.

Data: a synthetic data directory under tmp_path (five race days in 2025, development days);
nothing real is read.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.studies import form_bands as fb

DAYS = [date(2025, 1, 1), date(2025, 1, 15), date(2025, 2, 5), date(2025, 2, 19), date(2025, 3, 5)]
SPECIAL_DAY = date(2025, 2, 5)
MONTHLY = FoldScheme(
    unit="month", test_length=1, step=1, min_train=0, window="expanding", embargo_days=0,
    anchor_month=1,
)  # fmt: skip
FIELD = 6
NORMAL_RACES = (1, 2, 3)
DEAD_HEAT, NO_PRICED_WINNER, VOID = 4, 5, 6
STARTER_PRICED_AT_T2, STARTER_OFF_T2, SCRATCHED_AFTER_T2 = 7, 8, 9
ODD_ONE = 6  # the special runner of races 7-9


def rid(day: date, race_no: int) -> str:
    return f"jra:{day.isoformat()}:TOK:{race_no:02d}"


def runner_rows() -> list[dict]:
    """One store row per runner: horse h of race r finishes (h + r) % FIELD + 1."""
    out = []
    for day in DAYS:
        races = NORMAL_RACES + (
            (DEAD_HEAT, NO_PRICED_WINNER, VOID, STARTER_PRICED_AT_T2, STARTER_OFF_T2,
             SCRATCHED_AFTER_T2) if day == SPECIAL_DAY else ()
        )  # fmt: skip
        for r in races:
            for h in range(1, FIELD + 1):
                finish = (h + r) % FIELD + 1
                odds = 2.0 + 3.0 * h
                scratched = False
                if r == DEAD_HEAT and h in (1, 2):
                    finish = 1
                elif r == DEAD_HEAT and finish == 1:
                    finish = 3
                if r == NO_PRICED_WINNER and finish == 1:
                    odds = np.nan
                if r == VOID:
                    finish = None
                if r in (STARTER_PRICED_AT_T2, STARTER_OFF_T2) and h == ODD_ONE:
                    odds = np.nan  # a starter without a final price: v0 leaves it out
                if r == SCRATCHED_AFTER_T2 and h == ODD_ONE:
                    finish, odds, scratched = None, np.nan, True
                out.append(
                    {"race_id": rid(day, r), "race_date": day, "race_no": r, "horse_no": h,
                     "finish_pos": finish, "is_scratched": scratched, "win_odds": odds}
                )  # fmt: skip
    return out


def in_v0(row: dict) -> bool:
    r, h = row["race_no"], row["horse_no"]
    if r in (DEAD_HEAT, NO_PRICED_WINNER, VOID):
        return False
    return not (r in (STARTER_PRICED_AT_T2, STARTER_OFF_T2, SCRATCHED_AFTER_T2) and h == ODD_ONE)


def on_t2_board(row: dict) -> float | None:
    """The runner's T-2 Win odds, None when it is off the board."""
    r, h = row["race_no"], row["horse_no"]
    if r == STARTER_OFF_T2 and h == ODD_ONE:
        return None
    if r == SCRATCHED_AFTER_T2 and h == ODD_ONE:
        return 1.8  # the T-2 favourite, scratched before the off
    return 2.5 + 3.0 * h


def ts_row(row: dict, odds, *, combination=None, investment=None) -> dict:
    post = datetime(row["race_date"].year, row["race_date"].month, row["race_date"].day, 6, 0,
                    tzinfo=timezone.utc) + timedelta(minutes=30 * row["race_no"])  # fmt: skip
    fetched = post - timedelta(minutes=2)
    return {
        "race_id": row["race_id"],
        "venue": "TOK",
        "race_no": row["race_no"],
        "pool": "WIN",
        "combination": combination or str(row["horse_no"]),
        "snapshot_utc": fetched - timedelta(seconds=4),
        "fetched_utc": fetched,
        "post_utc": post,
        "minutes_to_post": 2.0,
        "odds": odds,
        "odds_hi": None,
        "investment": investment,
        "status_raw": None if investment else "15時38分現在",
        "is_final_capture": False,
        "dividend": None,
        "is_refund": None,
    }


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    assert not any(is_holdout_day("japan", d) for d in DAYS)
    data = tmp_path / "data"
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    runners = runner_rows()
    store = data / "processed" / "jra"
    store.mkdir(parents=True)
    races = pd.DataFrame(
        [{"race_id": r["race_id"], "race_date": r["race_date"],
          "meeting_id": f"{r['race_date'].isoformat()}:TOK"} for r in runners]
    ).drop_duplicates("race_id")  # fmt: skip
    pq.write_table(
        pa.Table.from_pandas(races, preserve_index=False).cast(
            pa.schema([("race_id", pa.string()), ("race_date", pa.date32()),
                       ("meeting_id", pa.string())])
        ),
        store / "races.parquet",
    )  # fmt: skip
    frame = pd.DataFrame(runners).drop(columns="race_no")
    frame["finish_pos"] = frame["finish_pos"].astype("Int64")
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False).cast(
            pa.schema([("race_id", pa.string()), ("race_date", pa.date32()),
                       ("horse_no", pa.int64()), ("finish_pos", pa.int64()),
                       ("is_scratched", pa.bool_()), ("win_odds", pa.float64())])
        ),
        store / "runners.parquet",
    )  # fmt: skip

    folds = walk_forward(sorted(set(races["race_date"])), scheme=MONTHLY)
    fold_of = {}
    store_days = sorted(set(races["race_date"]))
    for k, fold in enumerate(folds):
        for i in fold.test_idx:
            fold_of[store_days[i]] = k
    v0 = [r for r in runners if in_v0(r)]
    oof = pd.DataFrame(
        {"race_id": [r["race_id"] for r in v0], "horse_no": [r["horse_no"] for r in v0],
         "fold": [fold_of[r["race_date"]] for r in v0],
         "p_cal": [1.0 / (2.0 + 3.0 * r["horse_no"]) for r in v0]}
    )  # fmt: skip
    oof["p_cal"] = oof["p_cal"] / oof.groupby("race_id")["p_cal"].transform("sum")
    v0_path = data / fb.V0_OOF
    v0_path.parent.mkdir(parents=True)
    oof.to_parquet(v0_path, index=False)

    handicap = data / "features" / "jra" / "handicap"
    odds_dir = data / "processed" / "jravan" / "odds_timeseries"
    handicap.mkdir(parents=True)
    odds_dir.mkdir(parents=True)
    for day in DAYS:
        mine = [r for r in runners if r["race_date"] == day]
        pd.DataFrame(
            {"race_id": [r["race_id"] for r in mine], "horse_no": [r["horse_no"] for r in mine],
             "handicap_score": [0.05 + 0.1 * r["horse_no"] for r in mine]}
        ).to_parquet(handicap / f"{day.isoformat()}.parquet", index=False)  # fmt: skip
        ts = []
        for r in mine:
            odds = on_t2_board(r)
            if odds is not None:
                ts.append(ts_row(r, odds))
        for first in {r["race_id"]: r for r in mine}.values():
            ts.append(ts_row(first, None, combination="__pool__", investment=1.0e7))
        pq.write_table(pa.Table.from_pylist(ts, schema=SCHEMA), odds_dir / f"{day}.parquet")
    return data


def test_races_left_out_of_v0_are_dropped_and_counted_by_reason(data_dir):
    rows, folds, counts, _ = fb.read_rows(DAYS[0], DAYS[-1], MONTHLY)
    present = set(rows["race_id"])
    for race_no, key in (
        (DEAD_HEAT, "dropped_dead_heat"),
        (NO_PRICED_WINNER, "dropped_no_priced_winner"),
        (VOID, "dropped_void"),
    ):
        assert rid(SPECIAL_DAY, race_no) not in present, key
        assert counts[key] == 1, key
    assert len(folds) == 3


@pytest.mark.parametrize("race_no", [STARTER_PRICED_AT_T2, STARTER_OFF_T2])
def test_a_confirmed_starter_without_a_v0_row_takes_its_race_out(data_dir, race_no):
    rows, _, counts, _ = fb.read_rows(DAYS[0], DAYS[-1], MONTHLY)
    assert rid(SPECIAL_DAY, race_no) not in set(rows["race_id"])
    assert counts["dropped_starter_without_v0"] == 2  # races 7 and 8


def test_a_runner_scratched_after_t2_ranks_but_is_not_a_starter(data_dir):
    rows, _, _, _ = fb.read_rows(DAYS[0], DAYS[-1], MONTHLY)
    race = rows[rows["race_id"] == rid(SPECIAL_DAY, SCRATCHED_AFTER_T2)].set_index("horse_no")
    assert sorted(race.index) == list(range(1, FIELD + 1))
    assert bool(race.loc[ODD_ONE, "scratched"]) is True
    assert race.loc[ODD_ONE, "win_odds_t2"] == pytest.approx(1.8)
    assert not race.drop(index=ODD_ONE)["scratched"].astype(bool).any()
    ranks = fb.assign_groups(race.reset_index()).set_index("horse_no")["fav_rank"].astype(int)
    assert ranks.loc[ODD_ONE] == 1
    assert sorted(ranks.drop(index=ODD_ONE)) == [2, 3, 4, 5, 6]


def test_every_other_race_is_read_whole(data_dir):
    rows, _, _, _ = fb.read_rows(DAYS[0], DAYS[-1], MONTHLY)
    want = {rid(d, r) for d in DAYS for r in NORMAL_RACES} | {rid(SPECIAL_DAY, SCRATCHED_AFTER_T2)}
    assert set(rows["race_id"]) == want
    normal = rows[rows["race_id"] != rid(SPECIAL_DAY, SCRATCHED_AFTER_T2)]
    assert normal.groupby("race_id").size().eq(FIELD).all()
    assert not normal["scratched"].astype(bool).any()
    assert normal["win_odds_t2"].notna().all() and normal["h"].notna().all()
