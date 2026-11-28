"""#218 reviews: the real-data reader of the form-bands family, and the v2 family it records under.

Module under test: `horsetrack.countries.japan.studies.form_bands`.

`read_rows(date_from, date_to, scheme) -> (rows, folds, counts, builds)` reads, under
`$HORSETRACK_DATA_DIR`:
- `V0_OOF` (`race_id`, `horse_no`, `fold`, `q`, `p_cal`): v0's out-of-fold file, as
  `models.v0.calibration.run_fit` writes it (`_development_frame`, `walk_forward` over its dates
  with the fold scheme, `oof_predict`). It leaves out first-place dead heats, races without a
  priced winner and starters without a final price;
- the JRA store, `processed/jra/{races,runners}.parquet` (races: `race_id`, `race_date`,
  `meeting_id`; runners: `calibration.STORE_COLUMNS`, `horse_no` nullable), read from
  `PRICE_ERA_START`: the outcomes, the scratch flags and v0's training rows;
- `features/jra/handicap/<day>.parquet` (`race_id`, `horse_no`, `handicap_score`): h;
- `processed/jravan/odds_timeseries/<day>.parquet` (`logger.compact.SCHEMA`): the T-2 Win board.

Contract pinned here (round-1 findings 684/745; round-2 findings A, B, C):
- A. v0's fold fits are refitted from the store exactly as `run_fit` makes them (`scheme`, the
  development frame without dead heats): `fit_win_calibration` on each fold's training rows. The
  refit must reproduce the v0 file's p_cal on the races the file holds and the reader keeps (else
  ValueError). A first-place dead heat with every starter priced is read: its p_cal is its fold's
  refit prediction (`predict_many` on q, the race's normalised 1/odds), each co-winner has `won`
  True, and `counts["dead_heat_races_refit"]` counts these races; no dead heat is dropped.
- The population is the store's races from `date_from` to `date_to` on development days. A store
  race v0 cannot score is left out of `rows` and counted by reason: `dropped_no_priced_winner`
  (a winner, none with a final price) and `dropped_void` (no finish_pos 1);
  `dropped_not_in_v0_other` stays 0 when no other reason applies.
- A confirmed starter (store `is_scratched` False) with no v0 row takes its race out, counted in
  `dropped_starter_without_v0`, whether or not it is priced at T-2.
- B. Every v0 row is checked against the store's starter fields: a v0 row for a runner the store
  now flags scratched (or does not hold) takes its race out, counted in
  `dropped_v0_disagrees_with_store`.
- C. The store's `horse_no` may be null. The reader does not abort on it: an unnumbered runner
  the store flags scratched leaves its race in `rows` (its numbered runners whole); an unnumbered
  confirmed starter takes its race out, counted in `dropped_unnumbered_starter`.
- A runner the store flags scratched, priced on the T-2 board and with no v0 row (scratched
  after T-2) stays a row with `scratched` True and its `win_odds_t2`: it ranks, it is not scored.
- Each race left out counts under one reason only. Every other race is in `rows` whole.

Round 3 (owner, #218, 2026-11-27, fix B): a fold's refit is checked against every row the v0
file holds for that fold, whatever the requested range (the rows outside the range enter the
check, they are not scored); a dead heat is dropped as `dropped_dead_heat_no_v0_reference` only
when the v0 file holds no row of its fold at all.

D. The family the study now records under is v3 (owner, #218, 2026-11-27, decision A):
- `FAMILY_ID == "japan-form-bands-t2-win-v3"`; `FAMILY_ID_V2 == "japan-form-bands-t2-win-v2"` and
  `FAMILY_ID_V1 == "japan-form-bands-t2-win"`, the families frozen and run before, left as they are.
- `build_spec(n_boot, data)` and `data_description(date_from, date_to, scheme)` stay the v1
  builders, byte-identical: the real v1 spec hashes to `V1_SPEC_HASH` below.
- `build_spec_v2(n_boot, data)` and `data_description_v2(date_from, date_to, scheme)` stay the v2
  builders, byte-identical: the real v2 spec hashes to `V2_SPEC_HASH` below.
- `build_spec_v3(n_boot, data)` and `data_description_v3(date_from, date_to, scheme)` are v3's.
  v3 is v2's design exactly: it differs from v2 in `family_id`, `data` and `supersedes` (which
  names v2), and optionally in `inputs`, `rulings` and `approved`. v3's data description names the
  round-3 drop reasons (`dropped_dead_heat_embargo`, `dropped_dead_heat_no_v0_reference`) and the
  void rule for a race with no runner rows.
- `run_study(out_dir, *, date_from, date_to, n_boot)` freezes `build_spec_v3(n_boot,
  data_description_v3(date_from, date_to, scheme))` under `FAMILY_ID` in `default_registry` before
  reading rows and records the 12 variants there; nothing goes to the v1 or v2 family. (The test
  swaps `form_bands.load_country_scheme` for a monthly scheme: the synthetic days are three months.)

Data: a synthetic data directory under tmp_path, nine race days in 2025 (development days); the
v0 file is made here by `models.v0.calibration` itself. Expected values come from that module,
from the stated rules and the planted races; nothing real is read.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import copy
import hashlib
import json
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import default_registry
from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.models.v0 import calibration
from horsetrack.countries.japan.studies import form_bands as fb

V1_SPEC_HASH = "3585a45d05f6ecfcf1de92d605d08982ec2b8c4def4d9932823c63c031655bad"
V2_SPEC_HASH = "4d1f93ba902c368184c395c58a64bdf2c0a8e79b94d8b8893c82d406e8421256"  # as frozen
DAYS = [
    date(2025, 1, 1), date(2025, 1, 8), date(2025, 1, 15), date(2025, 1, 22),  # v0 trains only
    date(2025, 2, 5), date(2025, 2, 12), date(2025, 2, 19),  # v0 fold 0
    date(2025, 3, 5), date(2025, 3, 12),  # v0 fold 1
]  # fmt: skip
READ_FROM = date(2025, 2, 1)  # v0's first out-of-fold month, as 2020-Q1 for the real file
SPECIAL_DAY, LAST_DAY = date(2025, 2, 12), date(2025, 3, 12)
MONTHLY = FoldScheme(
    unit="month", test_length=1, step=1, min_train=1, window="expanding", embargo_days=7,
    anchor_month=1,
)  # fmt: skip
FIELD, N_NORMAL = 8, 12
DEAD_HEAT, NO_PRICED_WINNER, VOID = 13, 14, 15
STARTER_PRICED_AT_T2, STARTER_OFF_T2, SCRATCHED_AFTER_T2 = 16, 17, 18
UNNUMBERED_SCRATCH, UNNUMBERED_STARTER = 19, 20
SPECIALS = (DEAD_HEAT, NO_PRICED_WINNER, VOID, STARTER_PRICED_AT_T2, STARTER_OFF_T2,
            SCRATCHED_AFTER_T2, UNNUMBERED_SCRATCH, UNNUMBERED_STARTER)  # fmt: skip
LATE_SCRATCH = 13  # on LAST_DAY: horse 3 ran when v0 was made, the store now scratches it
ODD_ONE = FIELD  # the special runner of races 16-18
RUNNER_SCHEMA = pa.schema(
    [("race_id", pa.string()), ("race_date", pa.date32()), ("horse_no", pa.int64()),
     ("win_odds", pa.float64()), ("finish_pos", pa.int64()), ("is_scratched", pa.bool_()),
     ("dead_heat", pa.bool_())]
)  # fmt: skip


def rid(day: date, race_no: int) -> str:
    return f"jra:{day.isoformat()}:TOK:{race_no:02d}"


def race(rng, day: date, race_no: int, *, not_winner: int | None = None) -> list[dict]:
    s = rng.normal(0.0, 1.2, FIELD)
    p = np.exp(s) / np.exp(s).sum()
    odds = np.maximum(np.round(0.8 / p, 1), 1.1)
    winner = int(rng.choice(FIELD, p=p)) + 1
    if winner == not_winner:
        winner = 1 if not_winner != 1 else 2
    others = [h for h in rng.permutation(np.arange(1, FIELD + 1)).tolist() if h != winner]
    finish = {winner: 1, **{h: k + 2 for k, h in enumerate(others)}}
    return [
        {"race_id": rid(day, race_no), "race_date": day, "race_no": race_no, "horse_no": h,
         "win_odds": float(odds[h - 1]), "finish_pos": finish[h], "is_scratched": False,
         "dead_heat": False}
        for h in range(1, FIELD + 1)
    ]  # fmt: skip


def special(rng, race_no: int) -> list[dict]:
    rows = race(rng, SPECIAL_DAY, race_no, not_winner=ODD_ONE)
    by = {r["horse_no"]: r for r in rows}
    winner = next(h for h, r in by.items() if r["finish_pos"] == 1)
    if race_no == DEAD_HEAT:
        second = next(h for h, r in by.items() if r["finish_pos"] == 2)
        by[second]["finish_pos"] = 1
        by[winner]["dead_heat"] = by[second]["dead_heat"] = True
    elif race_no == NO_PRICED_WINNER:
        by[winner]["win_odds"] = np.nan
    elif race_no == VOID:
        for r in rows:
            r["finish_pos"] = None
    elif race_no in (STARTER_PRICED_AT_T2, STARTER_OFF_T2):
        by[ODD_ONE]["win_odds"] = np.nan  # a starter without a final price: v0 leaves it out
    elif race_no == SCRATCHED_AFTER_T2:
        by[ODD_ONE].update(win_odds=np.nan, finish_pos=None, is_scratched=True)
    elif race_no in (UNNUMBERED_SCRATCH, UNNUMBERED_STARTER):
        scratched = race_no == UNNUMBERED_SCRATCH
        unnumbered = {"horse_no": None, "win_odds": np.nan, "is_scratched": scratched,
                      "finish_pos": None if scratched else FIELD + 1}  # fmt: skip
        rows.append({**rows[0], **unnumbered})
    return rows


def world(seed: int = 218, *, unnumbered: bool = True) -> tuple[list[dict], list[dict]]:
    """The store's runners as v0 saw them, and as the store holds them now."""
    rng = np.random.default_rng(seed)
    then = []
    for day in DAYS:
        for r in range(1, N_NORMAL + 1):
            then += race(rng, day, r)
        if day == SPECIAL_DAY:
            for r in SPECIALS if unnumbered else SPECIALS[:-2]:
                then += special(rng, r)
        if day == LAST_DAY:
            then += race(rng, day, LATE_SCRATCH, not_winner=3)
    now = copy.deepcopy(then)
    for r in now:
        if r["race_id"] == rid(LAST_DAY, LATE_SCRATCH) and r["horse_no"] == 3:
            r.update(win_odds=np.nan, finish_pos=None, is_scratched=True)
    return then, now


def runner_table(rows: list[dict]) -> pa.Table:
    frame = pd.DataFrame(rows).drop(columns="race_no")
    frame["horse_no"] = frame["horse_no"].astype("Int64")
    frame["finish_pos"] = frame["finish_pos"].astype("Int64")
    return pa.Table.from_pandas(frame[RUNNER_SCHEMA.names], preserve_index=False).cast(
        RUNNER_SCHEMA
    )


def v0_frame_and_folds(rows: list[dict]):
    frame, _, _ = calibration._development_frame(runner_table(rows))
    return frame, walk_forward(frame["race_date"], scheme=MONTHLY)


def t2_odds(r: dict) -> float | None:
    if r["horse_no"] is None:
        return None
    if (
        r["race_no"] == STARTER_OFF_T2
        and r["race_date"] == SPECIAL_DAY
        and r["horse_no"] == ODD_ONE
    ):
        return None
    if (r["race_no"] == SCRATCHED_AFTER_T2 and r["race_date"] == SPECIAL_DAY
            and r["horse_no"] == ODD_ONE):  # fmt: skip
        return 1.8  # the T-2 favourite, scratched before the off
    if r["race_no"] == SCRATCHED_AFTER_T2 and r["race_date"] == SPECIAL_DAY:
        return 2.5 + 1.5 * r["horse_no"]  # tie-free and all longer than 1.8: ranks 2..FIELD
    return r["win_odds"] if np.isfinite(r["win_odds"]) else 30.0


def ts_row(r: dict, odds, *, combination=None, investment=None) -> dict:
    day = r["race_date"]
    post = datetime(day.year, day.month, day.day, 1, 0, tzinfo=timezone.utc) + timedelta(
        minutes=30 * r["race_no"]
    )
    fetched = post - timedelta(minutes=2)
    return {
        "race_id": r["race_id"], "venue": "TOK", "race_no": r["race_no"], "pool": "WIN",
        "combination": combination or str(r["horse_no"]),
        "snapshot_utc": fetched - timedelta(seconds=4), "fetched_utc": fetched, "post_utc": post,
        "minutes_to_post": 2.0, "odds": odds, "odds_hi": None, "investment": investment,
        "status_raw": None if investment else "15時38分現在", "is_final_capture": False,
        "dividend": None, "is_refund": None,
    }  # fmt: skip


# Round-3 fixes (GPT re-check of PR #473): races planted only in the extended data directory.
# v0's development frame leaves out every one of them (dead heats, void, unpriced winner), so the
# v0 file and the fold fits are those of the base world.
EMPTY_RACE_NO = 21  # a races-table row with no runner rows
EMPTY_IN_RANGE, EMPTY_OUT_OF_RANGE = rid(SPECIAL_DAY, EMPTY_RACE_NO), rid(DAYS[1], EMPTY_RACE_NO)
EMBARGO_DAY, FIRST_SCORED_DAY = date(2025, 2, 4), DAYS[4]  # fold 0 trains to 2025-01-29
DH_IN_EMBARGO, DH_AFTER_EMBARGO = rid(EMBARGO_DAY, 1), rid(FIRST_SCORED_DAY, N_NORMAL + 1)
# in fold 1's window, after its kept v0 days: a dead heat; a day whose one v0 race the store now
# contradicts, beside a void race; a day with no v0 race at all (void, unpriced winner)
NO_V0_DAY, EXCLUDED_DAY, VOID_DAY = date(2025, 3, 19), date(2025, 3, 24), date(2025, 3, 26)
DH_NO_V0_REFERENCE = rid(NO_V0_DAY, 1)
EXTRA_DAYS = (EMBARGO_DAY, NO_V0_DAY, EXCLUDED_DAY, VOID_DAY)


def void_race(rng, day: date, race_no: int) -> list[dict]:
    rows = race(rng, day, race_no)
    for r in rows:
        r["finish_pos"] = None
    return rows


def dead_heat_race(rng, day: date, race_no: int) -> list[dict]:
    rows = race(rng, day, race_no)
    by = {r["finish_pos"]: r for r in rows}
    by[2]["finish_pos"] = 1
    by[1]["dead_heat"] = by[2]["dead_heat"] = True
    return rows


def extra_runners(*, now: bool) -> list[dict]:
    rng = np.random.default_rng(473)
    rows = dead_heat_race(rng, EMBARGO_DAY, 1)
    rows += dead_heat_race(rng, FIRST_SCORED_DAY, N_NORMAL + 1)
    rows += dead_heat_race(rng, NO_V0_DAY, 1)
    late = race(rng, EXCLUDED_DAY, 1, not_winner=3)
    if now:  # v0 saw horse 3 run; the store now scratches it
        late[2].update(win_odds=np.nan, finish_pos=None, is_scratched=True)
    rows += late + void_race(rng, EXCLUDED_DAY, 2) + void_race(rng, VOID_DAY, 1)
    unpriced = race(rng, VOID_DAY, 2)
    next(r for r in unpriced if r["finish_pos"] == 1)["win_odds"] = np.nan
    return rows + unpriced


def build_data_dir(base: Path, *, unnumbered: bool = True, extras: bool = False) -> Path:
    assert not any(is_holdout_day("japan", d) for d in (*DAYS, *EXTRA_DAYS))
    then, now = world(unnumbered=unnumbered)
    if extras:
        then += extra_runners(now=False)
        now += extra_runners(now=True)
    data = base / "data"
    store = data / "processed" / "jra"
    store.mkdir(parents=True)
    empty = (
        [{"race_id": EMPTY_IN_RANGE, "race_date": SPECIAL_DAY, "race_no": EMPTY_RACE_NO},
         {"race_id": EMPTY_OUT_OF_RANGE, "race_date": DAYS[1], "race_no": EMPTY_RACE_NO}]
        if extras else []
    )  # fmt: skip
    races = pd.DataFrame(
        [{"race_id": r["race_id"], "race_date": r["race_date"],
          "meeting_id": f"{r['race_date'].isoformat()}:TOK"} for r in now + empty]
    ).drop_duplicates("race_id")  # fmt: skip
    race_schema = pa.schema(
        [("race_id", pa.string()), ("race_date", pa.date32()), ("meeting_id", pa.string())]
    )
    pq.write_table(
        pa.Table.from_pandas(races, preserve_index=False).cast(race_schema),
        store / "races.parquet",
    )
    pq.write_table(runner_table(now), store / "runners.parquet")

    frame, folds = v0_frame_and_folds(then)  # v0 as #114 made it, before the late scratch
    oof = calibration.oof_predict(frame, folds=folds)
    oof["race_id"] = oof["race_id"].astype(str)
    v0_path = data / fb.V0_OOF
    v0_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(oof, preserve_index=False), v0_path)

    handicap = data / "features" / "jra" / "handicap"
    odds_dir = data / "processed" / "jravan" / "odds_timeseries"
    handicap.mkdir(parents=True)
    odds_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for day in sorted({r["race_date"] for r in now}):
        mine = [r for r in now if r["race_date"] == day and r["horse_no"] is not None]
        pd.DataFrame(
            {"race_id": [r["race_id"] for r in mine], "horse_no": [r["horse_no"] for r in mine],
             "handicap_score": rng.uniform(0.02, 0.4, len(mine))}
        ).to_parquet(handicap / f"{day.isoformat()}.parquet", index=False)  # fmt: skip
        ts = [ts_row(r, t2_odds(r)) for r in mine if t2_odds(r) is not None]
        for first in {r["race_id"]: r for r in mine}.values():
            ts.append(ts_row(first, None, combination="__pool__", investment=1.0e7))
        pq.write_table(pa.Table.from_pylist(ts, schema=SCHEMA), odds_dir / f"{day}.parquet")
    return data


def read(data: Path, date_from: date = READ_FROM, date_to: date = LAST_DAY):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(data))
        return fb.read_rows(date_from, date_to, MONTHLY)


@pytest.fixture(scope="module")
def read_back(tmp_path_factory):
    data = build_data_dir(tmp_path_factory.mktemp("reader"))
    rows, folds, counts, _ = read(data)
    return rows, folds, counts, data


# ---------------------------------------------------------------- A. dead heats from v0's fits


def test_a_dead_heat_is_read_with_its_v0_fold_fit_prediction_and_both_winners(read_back):
    rows, _, counts, _ = read_back
    mine = rows[rows["race_id"] == rid(SPECIAL_DAY, DEAD_HEAT)].sort_values("horse_no")
    assert mine["horse_no"].tolist() == list(range(1, FIELD + 1))
    assert int(mine["won"].sum()) == 2
    assert not mine["scratched"].astype(bool).any()

    _, now = world()
    frame, folds = v0_frame_and_folds(now)
    k = next(i for i, f in enumerate(folds) if f.test_start <= SPECIAL_DAY <= f.test_end)
    model = calibration.fit_win_calibration(frame.iloc[folds[k].train_idx])
    race_rows = sorted(
        (r for r in now if r["race_id"] == rid(SPECIAL_DAY, DEAD_HEAT)), key=lambda r: r["horse_no"]
    )
    inv = 1.0 / np.array([r["win_odds"] for r in race_rows])
    want = model.predict_many(inv / inv.sum(), np.zeros(FIELD, dtype=int))
    np.testing.assert_allclose(mine["p_cal"].to_numpy(dtype=float), want, rtol=1e-6)
    assert set(mine["fold"].astype(int)) == {k}
    assert counts["dead_heat_races_refit"] == 1
    assert counts.get("dropped_dead_heat", 0) == 0


def test_a_v0_file_the_fold_fits_do_not_reproduce_is_refused(tmp_path):
    data = build_data_dir(tmp_path, unnumbered=False)  # a refusal for this reason only
    oof = pq.read_table(data / fb.V0_OOF).to_pandas()
    target = oof["race_id"] == rid(date(2025, 3, 5), 1)
    assert target.any()
    oof.loc[target, "p_cal"] = oof.loc[target, "p_cal"] * (
        1.0 + 0.1 * (oof.loc[target, "horse_no"] % 2)
    )
    pq.write_table(pa.Table.from_pandas(oof, preserve_index=False), data / fb.V0_OOF)
    with pytest.raises(ValueError):
        read(data)


# ---------------------------------------------------------------- races v0 cannot score


def test_races_v0_cannot_score_are_dropped_and_counted_by_reason(read_back):
    rows, _, counts, _ = read_back
    present = set(rows["race_id"])
    for race_no, key in ((NO_PRICED_WINNER, "dropped_no_priced_winner"), (VOID, "dropped_void")):
        assert rid(SPECIAL_DAY, race_no) not in present, key
        assert counts[key] == 1, key
    assert counts.get("dropped_not_in_v0_other", 0) == 0


@pytest.mark.parametrize("race_no", [STARTER_PRICED_AT_T2, STARTER_OFF_T2])
def test_a_confirmed_starter_without_a_v0_row_takes_its_race_out(read_back, race_no):
    rows, _, counts, _ = read_back
    assert rid(SPECIAL_DAY, race_no) not in set(rows["race_id"])
    assert counts["dropped_starter_without_v0"] == 2  # races 16 and 17


# ---------------------------------------------------------------- B. v0 rows against the store


def test_a_v0_row_for_a_runner_the_store_now_scratches_takes_its_race_out(read_back):
    rows, _, counts, _ = read_back
    assert rid(LAST_DAY, LATE_SCRATCH) not in set(rows["race_id"])
    assert counts["dropped_v0_disagrees_with_store"] == 1


# ---------------------------------------------------------------- C. unnumbered store runners


def test_unnumbered_store_runners_never_abort_the_read(read_back):
    rows, _, counts, _ = read_back
    kept = rows[rows["race_id"] == rid(SPECIAL_DAY, UNNUMBERED_SCRATCH)]
    assert sorted(kept["horse_no"].astype(int)) == list(range(1, FIELD + 1))
    assert not kept["scratched"].astype(bool).any()
    assert rid(SPECIAL_DAY, UNNUMBERED_STARTER) not in set(rows["race_id"])
    assert counts["dropped_unnumbered_starter"] == 1


# ---------------------------------------------------------------- scratches and the rest


def test_a_runner_scratched_after_t2_ranks_but_is_not_a_starter(read_back):
    rows = read_back[0]
    mine = rows[rows["race_id"] == rid(SPECIAL_DAY, SCRATCHED_AFTER_T2)].set_index("horse_no")
    assert sorted(mine.index.astype(int)) == list(range(1, FIELD + 1))
    assert bool(mine.loc[ODD_ONE, "scratched"]) is True
    assert mine.loc[ODD_ONE, "win_odds_t2"] == pytest.approx(1.8)
    assert not mine.drop(index=ODD_ONE)["scratched"].astype(bool).any()
    ranks = fb.assign_groups(mine.reset_index()).set_index("horse_no")["fav_rank"].astype(int)
    assert ranks.loc[ODD_ONE] == 1
    assert sorted(ranks.drop(index=ODD_ONE)) == list(range(2, FIELD + 1))


def test_every_other_race_is_read_whole_with_v0s_p_cal(read_back):
    rows, _, _, data = read_back
    read_days = [d for d in DAYS if d >= READ_FROM]
    normal = {rid(d, r) for d in read_days for r in range(1, N_NORMAL + 1)}
    kept_specials = {
        rid(SPECIAL_DAY, r) for r in (DEAD_HEAT, SCRATCHED_AFTER_T2, UNNUMBERED_SCRATCH)
    }
    assert set(rows["race_id"]) == normal | kept_specials
    mine = rows[rows["race_id"].isin(normal)]
    assert mine.groupby("race_id").size().eq(FIELD).all()
    assert not mine["scratched"].astype(bool).any()
    assert mine["win_odds_t2"].notna().all() and mine["h"].notna().all()
    oof = pq.read_table(data / fb.V0_OOF).to_pandas().set_index(["race_id", "horse_no"])["p_cal"]
    got = mine.set_index(["race_id", mine["horse_no"].astype(int)])["p_cal"]
    np.testing.assert_allclose(got.to_numpy(dtype=float), oof.loc[got.index].to_numpy(), rtol=1e-6)


# ---------------------------------------------------------------- round 3: the GPT re-check
# 1. a races-table race with no runner rows is void; 2. a dead heat inside D9's one-week embargo
# after its fold fit's training cutoff is dropped; 3. a fold with dead heats and no v0 rows to
# check the refit against drops its dead heats; 4. a span with nothing to score does not raise.


@pytest.fixture(scope="module")
def extended(tmp_path_factory):
    return build_data_dir(tmp_path_factory.mktemp("extended"), extras=True)


def test_a_race_with_no_runner_rows_is_counted_void_once(extended):
    rows, _, counts, _ = read(extended)
    assert EMPTY_IN_RANGE not in set(rows["race_id"])
    assert counts["dropped_void"] == 2  # VOID and the runnerless race; not the one before READ_FROM


def test_a_dead_heat_inside_the_embargo_after_its_fold_fits_cutoff_is_dropped(extended):
    _, now = world()
    _, v0_folds = v0_frame_and_folds(now + extra_runners(now=True))
    k = next(i for i, f in enumerate(v0_folds) if f.test_start <= EMBARGO_DAY <= f.test_end)
    cutoff = v0_folds[k].train_end  # the fold fit's cutoff, not the window's start
    assert (EMBARGO_DAY - cutoff).days < MONTHLY.embargo_days <= (FIRST_SCORED_DAY - cutoff).days
    assert cutoff < v0_folds[k].test_start - timedelta(days=1)

    rows, _, counts, _ = read(extended)
    assert counts.get("dropped_dead_heat_embargo", 0) == 1
    present = set(rows["race_id"])
    assert DH_IN_EMBARGO not in present
    scored = rows[rows["race_id"] == DH_AFTER_EMBARGO]
    assert len(scored) == FIELD and int(scored["won"].sum()) == 2
    assert set(scored["fold"].astype(int)) == {k} and scored["p_cal"].notna().all()
    assert rid(SPECIAL_DAY, DEAD_HEAT) in present
    assert counts["dead_heat_races_refit"] == 2


LATE_MARCH = (date(2025, 3, 13), date(2025, 3, 31))  # fold 1's v0 days before it, Mar 5 and 12


def no_v0_reference_fold() -> int:
    _, now = world()
    _, v0_folds = v0_frame_and_folds(now + extra_runners(now=True))
    return next(i for i, f in enumerate(v0_folds) if f.test_start <= NO_V0_DAY <= f.test_end)


def with_v0_file(extended: Path, base: Path, edit) -> Path:
    """A copy of the extended data directory whose v0 file is `edit(v0 file)`."""
    data = base / "data"
    shutil.copytree(extended, data)
    oof = pq.read_table(data / fb.V0_OOF).to_pandas()
    pq.write_table(pa.Table.from_pandas(edit(oof), preserve_index=False), data / fb.V0_OOF)
    return data


def test_a_dead_heat_is_scored_against_its_folds_v0_rows_outside_the_range(extended):
    k = no_v0_reference_fold()
    oof = pq.read_table(extended / fb.V0_OOF).to_pandas()
    oof["race_date"] = pd.to_datetime(oof["race_id"].str.slice(4, 14)).dt.date
    pinned = oof[oof["fold"] == k]
    lo, hi = LATE_MARCH
    assert min(pinned["race_date"]) < lo  # the fold's v0 rows lie before the range...
    in_range = pinned[(pinned["race_date"] >= lo) & (pinned["race_date"] <= hi)]
    assert set(in_range["race_id"]) == {rid(EXCLUDED_DAY, 1)}  # ...but one, the store contradicts

    rows, _, counts, _ = read(extended, lo, hi)
    assert counts["dropped_v0_disagrees_with_store"] == 1
    assert counts.get("dropped_dead_heat_no_v0_reference", 0) == 0
    assert counts["dead_heat_races_refit"] == 1
    assert set(rows["race_id"]) == {DH_NO_V0_REFERENCE}  # the rows before the range: not scored
    mine = rows.sort_values("horse_no")
    assert mine["horse_no"].astype(int).tolist() == list(range(1, FIELD + 1))
    assert int(mine["won"].sum()) == 2 and set(mine["fold"].astype(int)) == {k}

    _, now = world()
    now += extra_runners(now=True)
    frame, v0_folds = v0_frame_and_folds(now)
    model = calibration.fit_win_calibration(frame.iloc[v0_folds[k].train_idx])
    race_rows = sorted((r for r in now if r["race_id"] == DH_NO_V0_REFERENCE),
                       key=lambda r: r["horse_no"])  # fmt: skip
    inv = 1.0 / np.array([r["win_odds"] for r in race_rows])
    want = model.predict_many(inv / inv.sum(), np.zeros(FIELD, dtype=int))
    got = mine["p_cal"].to_numpy(dtype=float)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_a_folds_refit_is_checked_against_its_v0_rows_outside_the_range(extended, tmp_path):
    def spoil(oof: pd.DataFrame) -> pd.DataFrame:  # a fold-1 race before the range, Mar 5
        target = oof["race_id"] == rid(date(2025, 3, 5), 1)
        assert target.any() and set(oof.loc[target, "fold"]) == {no_v0_reference_fold()}
        oof.loc[target, "p_cal"] = oof.loc[target, "p_cal"] * (
            1.0 + 0.1 * (oof.loc[target, "horse_no"] % 2)
        )
        return oof

    data = with_v0_file(extended, tmp_path, spoil)
    with pytest.raises(ValueError):
        read(data, *LATE_MARCH)  # the same read of the unspoilt file scores (test above)


@pytest.mark.parametrize("span", [LATE_MARCH, (date(2025, 3, 1), date(2025, 3, 31))],
                         ids=["late-march", "all-march"])  # fmt: skip
def test_a_dead_heat_in_a_fold_the_v0_file_holds_no_row_of_is_dropped(extended, tmp_path, span):
    k = no_v0_reference_fold()
    data = with_v0_file(extended, tmp_path, lambda oof: oof[oof["fold"] != k])
    rows, _, counts, _ = read(data, *span)
    assert counts.get("dropped_dead_heat_no_v0_reference", 0) == 1
    assert DH_NO_V0_REFERENCE not in set(rows["race_id"])
    assert counts["dead_heat_races_refit"] == 0


@pytest.mark.parametrize(
    ("day", "want"),
    [(EXCLUDED_DAY, {"dropped_void": 1, "dropped_v0_disagrees_with_store": 1,
                     "dropped_no_priced_winner": 0}),
     (VOID_DAY, {"dropped_void": 1, "dropped_v0_disagrees_with_store": 0,
                 "dropped_no_priced_winner": 1})],
    ids=["v0-rows-all-dropped", "no-v0-rows"],
)  # fmt: skip
def test_a_span_with_no_eligible_race_returns_its_drop_counts(extended, day, want):
    rows, _, counts, _ = read(extended, day, day)
    assert len(rows) == 0
    assert {k: counts[k] for k in want} == want
    assert counts["races_read"] == 0


def test_run_study_on_a_span_with_no_eligible_race_records_failed_variants(extended, tmp_path):
    day = EXCLUDED_DAY
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(extended))
        mp.setattr(fb, "load_country_scheme", lambda country: MONTHLY)
        result = fb.run_study(tmp_path / "out", date_from=day, date_to=day, n_boot=50)
        records = default_registry("japan").records(fb.FAMILY_ID)
    assert result["counts"]["dropped_void"] == 1
    assert result["counts"]["dropped_v0_disagrees_with_store"] == 1
    assert sorted(r.variant for r in records) == sorted(fb.VARIANTS)
    assert {r.status for r in records} == {"failed"}
    assert set(result["errors"]) == set(fb.VARIANTS)


# ---------------------------------------------------------------- D. the families: v1, v2, v3


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_the_v1_spec_builder_still_gives_the_frozen_v1_spec():
    assert fb.FAMILY_ID_V1 == "japan-form-bands-t2-win"
    scheme = load_country_scheme("japan")
    v1 = fb.build_spec(2000, fb.data_description(fb.DATE_FROM, fb.DATE_TO, scheme))
    assert v1["family_id"] == fb.FAMILY_ID_V1
    assert spec_hash(v1) == V1_SPEC_HASH


SAME_IN_V2 = ("ticket", "hypothesis_id", "question", "null", "meld", "bands", "rank_groups",
              "variants", "no_other", "walk_forward", "scoring", "per_group", "inference",
              "holdout", "sources")  # fmt: skip
MAY_DIFFER = {"family_id", "data", "inputs", "rulings", "approved", "supersedes"}


def real_specs() -> tuple[dict, dict, dict]:
    """v1, v2 and v3 as the real run builds them (2000 resamples, the real span and scheme)."""
    span = (fb.DATE_FROM, fb.DATE_TO, load_country_scheme("japan"))
    return (fb.build_spec(2000, fb.data_description(*span)),
            fb.build_spec_v2(2000, fb.data_description_v2(*span)),
            fb.build_spec_v3(2000, fb.data_description_v3(*span)))  # fmt: skip


def test_v2_differs_from_v1_only_in_its_family_and_eligibility():
    assert fb.FAMILY_ID_V2 == "japan-form-bands-t2-win-v2"
    scheme = load_country_scheme("japan")
    v1 = fb.build_spec(2000, fb.data_description(fb.DATE_FROM, fb.DATE_TO, scheme))
    v2 = fb.build_spec_v2(2000, fb.data_description_v2(fb.DATE_FROM, fb.DATE_TO, scheme))
    assert v2["family_id"] == fb.FAMILY_ID_V2
    for key in SAME_IN_V2:
        assert v2[key] == v1[key], key
    differing = {k for k in v1.keys() | v2.keys() if v1.get(k) != v2.get(k)}
    assert {"family_id", "data"} <= differing <= MAY_DIFFER
    described = json.dumps(v2["data"]).lower()
    assert "dead heat" in described and "scratch" in described


def test_the_v2_spec_builder_still_gives_the_frozen_v2_spec():
    assert fb.FAMILY_ID_V2 == "japan-form-bands-t2-win-v2"
    scheme = load_country_scheme("japan")
    v2 = fb.build_spec_v2(2000, fb.data_description_v2(fb.DATE_FROM, fb.DATE_TO, scheme))
    assert v2["family_id"] == fb.FAMILY_ID_V2
    assert spec_hash(v2) == V2_SPEC_HASH


def test_v3_differs_from_v2_only_in_its_family_and_eligibility():
    assert fb.FAMILY_ID == "japan-form-bands-t2-win-v3"
    assert len({fb.FAMILY_ID, fb.FAMILY_ID_V2, fb.FAMILY_ID_V1}) == 3
    v1, v2, v3 = real_specs()
    assert v3["family_id"] == fb.FAMILY_ID
    for key in SAME_IN_V2:  # v2's design, which is v1's
        assert v3[key] == v2[key] == v1[key], key
    differing = {k for k in v2.keys() | v3.keys() if v2.get(k) != v3.get(k)}
    assert {"family_id", "data", "supersedes"} <= differing <= MAY_DIFFER
    assert fb.FAMILY_ID_V2 in v3["supersedes"]
    for key in ("days", "h", "t2", "fold_scheme"):  # v2's sources and span
        assert v3["data"][key] == v2["data"][key], key
    described = json.dumps(v3["data"]).lower()
    assert "dropped_dead_heat_embargo" in described
    assert "dropped_dead_heat_no_v0_reference" in described
    assert any(words in described for words in ("no runner", "without runner", "runnerless"))


def test_run_study_freezes_and_records_under_the_v3_family_only(tmp_path):
    data = build_data_dir(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(data))
        mp.setattr(fb, "load_country_scheme", lambda country: MONTHLY)
        registry = default_registry("japan")
        assert registry.path.resolve().is_relative_to(data.resolve())
        v1, v2, _ = real_specs()  # the earlier families, frozen before this run
        frozen = {fb.FAMILY_ID_V1: registry.freeze_family(fb.FAMILY_ID_V1, v1),
                  fb.FAMILY_ID_V2: registry.freeze_family(fb.FAMILY_ID_V2, v2)}  # fmt: skip
        assert frozen == {fb.FAMILY_ID_V1: V1_SPEC_HASH, fb.FAMILY_ID_V2: V2_SPEC_HASH}
        before = registry.path.read_bytes()
        result = fb.run_study(tmp_path / "out", date_from=READ_FROM, date_to=LAST_DAY, n_boot=50)
        registry = default_registry("japan")
        want = spec_hash(fb.build_spec_v3(50, fb.data_description_v3(READ_FROM, LAST_DAY, MONTHLY)))
        assert registry.frozen_spec_hash(fb.FAMILY_ID) == want == result["spec_hash"]
        for family, spec_sha in frozen.items():  # v1 and v2 untouched
            assert registry.frozen_spec_hash(family) == spec_sha, family
            assert registry.records(family) == [], family
        assert sorted(r.variant for r in registry.records(fb.FAMILY_ID)) == sorted(fb.VARIANTS)
        after = registry.path.read_bytes()
        assert after.startswith(before)
        added = [json.loads(line) for line in after[len(before) :].decode().splitlines()]
        families = {line.get("family_id") or line["record"]["family_id"] for line in added}
        assert added and families == {fb.FAMILY_ID}
    assert result["family_id"] == fb.FAMILY_ID == "japan-form-bands-t2-win-v3"
