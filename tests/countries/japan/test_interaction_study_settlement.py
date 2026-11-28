"""The C2 interaction study's Win settlement is required only where a return is scored (ticket #93,
review round 3; `studies.interactions`).

A JRA store refresh publishes a race's runners before its dividends, so an interrupted backfill
can leave a priced race with no WIN dividend row. Such a race is unsettled, never a race every
starter lost. Round 2 made `win_paid` raise on any such race, which aborts the whole launch even
when the race is a 2019 race used only to train (scoring starts in 2020-Q1). The check belongs
where the returns are read: the scored rows' tail.

Contract pinned here (`horsetrack.countries.japan.studies.interactions`):

- `win_paid(price, numbers, dividends, unit) -> (paid, counts)` (signature unchanged): `paid`
  is one float per row of `price`, in `price`'s row order. In a race with at least one WIN row,
  a starter whose horse number has a WIN row pays dividend / unit (a refund row pays 1.0, the
  stake) and any other starter pays 0.0; a dead heat pays each winner its own row, and the row
  decides, not `won`. A race with no WIN row (rows of other pools do not settle it) no longer
  raises: every one of its starters' `paid` is NaN, never 0, and
  `counts["races_without_a_win_dividend"]` is the number of such races.
- `_tail(p, scored, spec, name) -> (metrics, tables)` reads `paid` on `scored.rows` only (the
  scored folds' test rows, `studies.meta.Scored.of`). A NaN `paid` there, an unsettled race in a
  scored fold, raises ValueError, so `compute_study`'s `study.attempt` records each tail variant
  as failed instead of settling the race as all losers; a NaN on a training-only row, in no
  scored fold's test rows, is never read and does not stop the tail. Its `return_per_100[...]`
  metrics carry 100 x the band's mean return over `n` bets, so they sum to the scored rows'
  total return.

Not pinned: the error messages; a priced starter without a horse number or a malformed WIN row
(still a ValueError); the other counts; whether `compute_study` checks the scored rows before
the tail itself; the claim bands. `compute_study` reads `paid` on the scored rows only, which no
test here runs (its level-1 fits need the stores): the reviewer checks that wiring.

Every frame is synthetic and hand-written; the expected returns come from the dividend table
below by hand, never from the code under test.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.folds import Fold
from horsetrack.countries.japan.studies import interactions as study
from horsetrack.countries.japan.studies.meta import Scored

SPEC = {"bootstrap": {"n_boot": 200, "level": 0.95, "seed": 93}}
UNITS = (100.0, 50.0)

# race_id -> (race_date, meeting_id, [(horse_no, placed first, final Win odds), ...])
RACES = {
    "2019-a": (
        "2019-06-01",
        "m2019",
        [(1, False, 5.2), (2, True, 3.6), (3, False, 8.1), (4, False, 2.4)],
    ),
    "2019-b": ("2019-06-01", "m2019", [(1, True, 1.9), (2, False, 4.4), (3, False, 6.0)]),
    "2020-a": (
        "2020-01-05",
        "m2020a",
        [(1, False, 2.2), (2, False, 7.5), (3, True, 4.4), (4, False, 9.0)],
    ),
    "2020-b": (  # a dead heat between 1 and 4
        "2020-01-05",
        "m2020a",
        [(1, True, 2.4), (2, False, 3.3), (3, False, 12.0), (4, True, 3.0), (5, False, 6.5)],
    ),
    "2020-c": (  # 2 is refunded
        "2020-04-12",
        "m2020b",
        [(1, False, 3.1), (2, False, 4.0), (3, False, 9.9), (4, False, 2.8), (5, True, 12.5)],
    ),
    "2020-d": (  # paid to 1 before 1's disqualification: 2 is placed first and unpaid
        "2020-04-12",
        "m2020b",
        [(1, False, 1.6), (2, True, 5.5), (3, False, 4.8)],
    ),
}
WIN_ROWS = {  # race_id -> [(horse_no, WIN dividend per unit stake)]
    "2019-a": [(2, 360.0)],
    "2019-b": [(1, 190.0)],
    "2020-a": [(3, 450.0)],
    "2020-b": [(1, 230.0), (4, 310.0)],
    "2020-c": [(5, 1230.0)],
    "2020-d": [(1, 160.0)],
}
REFUNDS = {"2020-c": [2]}  # race_id -> refunded horse numbers (a WIN row with is_refund)
TRAINING = ("2019-a", "2019-b")
FOLDS = (  # (test_start, test_end, the races it tests); every earlier race trains it
    (date(2020, 1, 1), date(2020, 3, 31), ("2020-a", "2020-b")),
    (date(2020, 4, 1), date(2020, 6, 30), ("2020-c", "2020-d")),
)
SCORED = tuple(r for _, _, races in FOLDS for r in races)


def starters() -> pd.DataFrame:
    """One row per priced starter with its horse number; horse ids repeat across races and are
    not in horse-number order."""
    rows = []
    for k, (race, (day, meeting, field)) in enumerate(RACES.items()):
        for no, won, odds in field:
            rows.append(
                {
                    "race_id": race,
                    "horse_id": f"h{(7 * no + 3 * k) % 11:02d}",
                    "horse_no": no,
                    "race_date": pd.Timestamp(day),
                    "meeting_id": meeting,
                    "won": won,
                    "win_odds": odds,
                    "p_cal": 1.0 / len(field),
                }
            )
    return pd.DataFrame(rows)


def dividends(unit: float, unsettled=(), place_rows=True) -> pd.DataFrame:
    """The store's dividend rows: every race's WIN rows but the unsettled races', and a PLACE row
    per race (an unsettled race's too, when `place_rows`)."""
    rows = []
    for race in RACES:
        if race not in unsettled:
            for no, dividend in WIN_ROWS[race]:
                rows.append((race, "WIN", str(no), dividend, False))
            for no in REFUNDS.get(race, []):
                rows.append((race, "WIN", str(no), unit, True))
        if race not in unsettled or place_rows:
            rows.append((race, "PLACE", str(WIN_ROWS[race][0][0]), 120.0, False))
    frame = pd.DataFrame(rows, columns=["race_id", "pool", "combination", "dividend", "is_refund"])
    return frame.astype({"is_refund": "boolean"})


def expected_paid(frame: pd.DataFrame, unit: float, unsettled=()) -> np.ndarray:
    """By hand from the tables: NaN in an unsettled race, the stake on a refund, dividend / unit
    on a WIN row, 0 otherwise."""
    out = []
    for race, no in zip(frame["race_id"], frame["horse_no"], strict=True):
        if race in unsettled:
            out.append(math.nan)
        elif no in REFUNDS.get(race, []):
            out.append(1.0)
        else:
            out.append(next((d / unit for n, d in WIN_ROWS[race] if n == no), 0.0))
    return np.array(out, dtype=float)


def settle(frame: pd.DataFrame, unit: float, unsettled=(), place_rows=True):
    price = frame.drop(columns="horse_no")
    numbers = frame[["race_id", "horse_id", "horse_no"]]
    return study.win_paid(price, numbers, dividends(unit, unsettled, place_rows), unit)


def folds_of(frame: pd.DataFrame) -> list[Fold]:
    """Expanding folds over `frame`'s positions: each tests its races and trains on every race
    before its start."""
    out = []
    for start, end, races in FOLDS:
        days = frame["race_date"].dt.date
        out.append(
            Fold(
                train_idx=np.flatnonzero((days < start).to_numpy()),
                test_idx=np.flatnonzero(frame["race_id"].isin(races).to_numpy()),
                train_end=start - timedelta(days=8),
                test_start=start,
                test_end=end,
            )
        )
    return out


def total_return(metrics) -> tuple[float, int]:
    """The tail's total return and bets over its bands, from its return_per_100 metrics."""
    bands = [m for m in metrics if m.metric.startswith("return_per_100[")]
    return sum(m.value * m.n / 100 for m in bands), sum(m.n for m in bands)


# ---------------------------------------------------------------------------------------------
# win_paid marks an unsettled race instead of aborting
# ---------------------------------------------------------------------------------------------


def test_win_paid_marks_an_unsettled_race_and_settles_the_others_at_their_dividends():
    cases = [  # (unit, unsettled races, keep their PLACE rows)
        (100.0, (), True),
        (100.0, ("2019-b",), False),
        (50.0, ("2019-b",), True),
        (50.0, ("2019-a", "2020-c"), True),
        (100.0, ("2020-b", "2019-b", "2020-d"), False),
    ]
    shuffled = starters().sample(frac=1.0, random_state=7).reset_index(drop=True)
    for unit, unsettled, place_rows in cases:
        paid, counts = settle(shuffled, unit, unsettled, place_rows)
        want = expected_paid(shuffled, unit, unsettled)
        label = f"unit {unit}, unsettled {unsettled}"
        assert np.asarray(paid, dtype=float).shape == want.shape, label
        np.testing.assert_allclose(np.asarray(paid, dtype=float), want, err_msg=label)
        in_unsettled = shuffled["race_id"].isin(unsettled).to_numpy()
        assert np.isnan(np.asarray(paid, dtype=float)[in_unsettled]).all(), label  # never 0
        assert counts["races_without_a_win_dividend"] == len(unsettled), label


# ---------------------------------------------------------------------------------------------
# the tail: a training-only unsettled race never reaches it; a scored one fails it
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("unsettled", [("2019-b",), ("2019-a", "2019-b")])
def test_an_unsettled_training_race_does_not_stop_the_tail(unsettled):
    frame = starters()
    for unit in UNITS:
        paid, _ = settle(frame, unit, unsettled, place_rows=False)
        paid = np.asarray(paid, dtype=float)
        training = frame["race_id"].isin(unsettled).to_numpy()
        assert np.isnan(paid[training]).all()

        rows = frame.drop(columns="horse_no").assign(paid=paid)
        scored = Scored.of(rows, folds_of(frame))
        assert set(scored.rows["race_id"]) == set(SCORED)
        metrics, _ = study._tail(rows["p_cal"], scored, SPEC, "stage2:q:base")

        on_scored = frame["race_id"].isin(SCORED).to_numpy()
        want = expected_paid(frame, unit)[on_scored]
        total, bets = total_return(metrics)
        assert bets == int(on_scored.sum())
        assert total == pytest.approx(want.sum(), rel=1e-9)  # dividends / unit and the refund


@pytest.mark.parametrize("race", ["2020-a", "2020-d"])  # one race in each scored fold
def test_an_unsettled_scored_race_fails_the_tail_instead_of_paying_nothing(race):
    frame = starters()
    paid, counts = settle(frame, 100.0, (race,), place_rows=False)
    paid = np.asarray(paid, dtype=float)
    in_race = frame["race_id"].eq(race).to_numpy()
    assert np.isnan(paid[in_race]).all()  # unsettled, not a race every starter lost
    assert not np.isnan(paid[~in_race]).any()
    assert counts["races_without_a_win_dividend"] == 1

    rows = frame.drop(columns="horse_no").assign(paid=paid)
    scored = Scored.of(rows, folds_of(frame))
    with pytest.raises(ValueError):
        study._tail(rows["p_cal"], scored, SPEC, "stage2:q:base")
