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

D. The family the study now records under is a new one:
- `FAMILY_ID == "japan-form-bands-t2-win-v2"`; `FAMILY_ID_V1 == "japan-form-bands-t2-win"`, the
  family frozen and run first, left as it is.
- `build_spec(n_boot, data)` and `data_description(date_from, date_to, scheme)` stay the v1
  builders, byte-identical: the real v1 spec hashes to `V1_SPEC_HASH` below.
- `build_spec_v2(n_boot, data)` and `data_description_v2(date_from, date_to, scheme)` are v2's.
  v2 differs from v1 only in `family_id` and `data` (and optionally `inputs`, `rulings`,
  `approved`, `supersedes`): the variants, bands, rank groups, meld, walk-forward, scoring and
  inference are identical. v2's data description states the revised eligibility (dead heats,
  store-validated starters and scratches).
- `run_study(out_dir, *, date_from, date_to, n_boot)` freezes `build_spec_v2(n_boot,
  data_description_v2(date_from, date_to, scheme))` under `FAMILY_ID` in `default_registry` before
  reading rows and records the 12 variants there; nothing goes to the v1 family. (The test swaps
  `form_bands.load_country_scheme` for a monthly scheme: the synthetic days are three months.)

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


def build_data_dir(base: Path, *, unnumbered: bool = True) -> Path:
    assert not any(is_holdout_day("japan", d) for d in DAYS)
    then, now = world(unnumbered=unnumbered)
    data = base / "data"
    store = data / "processed" / "jra"
    store.mkdir(parents=True)
    races = pd.DataFrame(
        [{"race_id": r["race_id"], "race_date": r["race_date"],
          "meeting_id": f"{r['race_date'].isoformat()}:TOK"} for r in now]
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
    for day in DAYS:
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


def read(data: Path):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(data))
        return fb.read_rows(READ_FROM, LAST_DAY, MONTHLY)


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


# ---------------------------------------------------------------- D. the v2 family


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


def test_v2_differs_from_v1_only_in_its_family_and_eligibility():
    assert fb.FAMILY_ID == "japan-form-bands-t2-win-v2"
    scheme = load_country_scheme("japan")
    v1 = fb.build_spec(2000, fb.data_description(fb.DATE_FROM, fb.DATE_TO, scheme))
    v2 = fb.build_spec_v2(2000, fb.data_description_v2(fb.DATE_FROM, fb.DATE_TO, scheme))
    assert v2["family_id"] == fb.FAMILY_ID
    for key in SAME_IN_V2:
        assert v2[key] == v1[key], key
    differing = {k for k in v1.keys() | v2.keys() if v1.get(k) != v2.get(k)}
    assert {"family_id", "data"} <= differing <= MAY_DIFFER
    described = json.dumps(v2["data"]).lower()
    assert "dead heat" in described and "scratch" in described


def test_run_study_freezes_and_records_under_the_v2_family_only(tmp_path):
    data = build_data_dir(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(data))
        mp.setattr(fb, "load_country_scheme", lambda country: MONTHLY)
        result = fb.run_study(tmp_path / "out", date_from=READ_FROM, date_to=LAST_DAY, n_boot=50)
        registry = default_registry("japan")
        assert registry.path.resolve().is_relative_to(data.resolve())
        want = spec_hash(fb.build_spec_v2(50, fb.data_description_v2(READ_FROM, LAST_DAY, MONTHLY)))
        assert registry.frozen_spec_hash(fb.FAMILY_ID) == want == result["spec_hash"]
        assert registry.frozen_spec_hash(fb.FAMILY_ID_V1) is None
        assert registry.records(fb.FAMILY_ID_V1) == []
        assert sorted(r.variant for r in registry.records(fb.FAMILY_ID)) == sorted(fb.VARIANTS)
    assert result["family_id"] == fb.FAMILY_ID
