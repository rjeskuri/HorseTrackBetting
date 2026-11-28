"""Synthetic runner rows and fits for the horse member on the shadow page (ticket #717).

The member is #438's offset model (`studies/debut_members.py`): P(win) is the softmax within a
race of ln q + sum w x over standardised member columns (`Design`, `fit_offset`,
`predict_offset`). The rows here carry the columns `fit_offset` reads (race_id, won, ln_q,
meeting_id) plus race_date, race_class and horse_no, and three member columns named as #438's
`feature_columns` names them: one JRDB workout column, the body-weight change and the late Win
move. Winners are drawn from softmax(ln q + x @ SIGNAL), so a fit finds non-zero weights; rows of
`flip_days` are drawn from the opposite, stronger signal, so a fit that reads them comes out
different. Some JRDB values are missing, so the fit carries a missing flag too.

`expected_member_p` derives a race's member probabilities with #438's own `fit_offset` and
`predict_offset` on the development rows before the fit day, never with the code under test.
All data is invented here; nothing comes from JRDB or JRA-VAN.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.board.snapshot import DecisionGrid
from horsetrack.countries.japan.models.v0.calibration import WinCalibration
from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.studies import debut_members as dm

JRDB_FEATURE = "workouts_jrdb__ff_best_vs_par_s"
WEIGHT_FEATURE = "body_weight__horse_wt_change_kg"
LATE_FEATURE = "late_win__ln_share_move_t20_t2"
FEATURES = (JRDB_FEATURE, WEIGHT_FEATURE, LATE_FEATURE)
SCALE = {JRDB_FEATURE: 1.5, WEIGHT_FEATURE: 6.0, LATE_FEATURE: 0.3}
SIGNAL = {JRDB_FEATURE: -0.6, WEIGHT_FEATURE: -0.08, LATE_FEATURE: 2.5}
SEGMENTS = ("DEBUT", "MAIDEN")
VENUES = ("TOK", "KYO")

# the column names this helper relies on are #438's own
assert set(FEATURES) <= {c for m in dm.BUILT for c in dm.feature_columns(m)}
assert tuple(dm.CLASSES) == SEGMENTS


def weekend_days(start: date, end: date) -> list[date]:
    """Every Saturday and Sunday from start to end, both included."""
    out, day = [], start
    while day <= end:
        if day.weekday() >= 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def odds_field(rng: np.random.Generator, n: int) -> np.ndarray:
    """n WIN odds with a takeout-like overround, one decimal."""
    raw = rng.gamma(1.2, 1.0, n) + 0.05
    q = raw / raw.sum()
    return np.round(np.maximum(1.1, 0.8 / q), 1)


def synthetic_rows(
    days: Iterable[date],
    *,
    seed: int,
    classes: Sequence[str] = ("DEBUT", "MAIDEN", "C1"),
    races_per_class: int = 3,
    flip_days: Iterable[date] = (),
    missing_jrdb: float = 0.1,
) -> pd.DataFrame:
    """One row per runner of `races_per_class` races per class per day, 8-14 runners each."""
    rng = np.random.default_rng(seed)
    flipped = set(flip_days)
    rows = []
    for day in days:
        no = 0
        for race_class in classes:
            for _ in range(races_per_class):
                no += 1
                venue = VENUES[no % 2]
                n = int(rng.integers(8, 15))
                odds = odds_field(rng, n)
                inv = 1.0 / odds
                q = inv / inv.sum()
                x = {f: rng.normal(0.0, SCALE[f], n) for f in FEATURES}
                sign = -3.0 if day in flipped else 1.0
                u = np.log(q) + sum(sign * SIGNAL[f] * x[f] for f in FEATURES)
                p = np.exp(u - u.max())
                p /= p.sum()
                winner = int(rng.choice(n, p=p))
                jrdb = x[JRDB_FEATURE].copy()
                jrdb[rng.random(n) < missing_jrdb] = np.nan
                for i in range(n):
                    rows.append(
                        {
                            "race_id": f"jra:{day.isoformat()}:{venue}:{no:02d}",
                            "race_date": day,
                            "race_class": race_class,
                            "meeting_id": f"{day.isoformat()}:{venue}",
                            "horse_no": i + 1,
                            "won": i == winner,
                            "ln_q": float(np.log(q[i])),
                            JRDB_FEATURE: float(jrdb[i]),
                            WEIGHT_FEATURE: float(x[WEIGHT_FEATURE][i]),
                            LATE_FEATURE: float(x[LATE_FEATURE][i]),
                        }
                    )
    return pd.DataFrame(rows)


def training_rows(rows: pd.DataFrame, segment: str, today: date) -> pd.DataFrame:
    """The rows a member fit for `segment` on `today` may learn from: the segment's races on
    development days strictly before today."""
    keep = [
        race_class == segment and day < today and not is_holdout_day("japan", day)
        for race_class, day in zip(rows["race_class"], rows["race_date"], strict=True)
    ]
    return rows.loc[keep]


def offset_fit(rows: pd.DataFrame, segment: str, today: date, features=FEATURES):
    """#438's (design, fit) on the training rows."""
    return dm.fit_offset(training_rows(rows, segment, today), list(features))


def race_frame(
    q: Sequence[float], features: Mapping[str, Sequence[float]], race_id: str = "r"
) -> pd.DataFrame:
    """One race's runners as #438's predict_offset reads them (missing columns are NaN)."""
    n = len(q)
    frame = {f: np.asarray(features.get(f, [np.nan] * n), dtype=float) for f in FEATURES}
    return pd.DataFrame({**frame, "race_id": [race_id] * n, "ln_q": np.log(np.asarray(q, float))})


def expected_member_p(
    design_fit, q: Sequence[float], features: Mapping[str, Sequence[float]]
) -> np.ndarray:
    """#438's predict_offset of one race, from (design, fit) of `offset_fit`."""
    design, fit = design_fit
    return dm.predict_offset(design, fit, race_frame(q, features)).to_numpy(dtype=float)


def market_q(odds: Sequence[float]) -> np.ndarray:
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum()


# ---------------------------------------------------------------- the pointer

GRID = DecisionGrid((60, 30, 16, 5, 2, 1), 2)


def base_fits(tag: str) -> tuple[WinCalibration, OrderingFit, NowcastV0]:
    """Small win, ordering and nowcast fits for a pointer (their values do not matter here)."""
    win = WinCalibration(
        knots=(-4.0, -0.5),
        coef=(1.1,),
        fit_id=f"win-{tag}",
        train_start=date(2015, 1, 4),
        train_end=date(2024, 12, 28),
        n_races=120,
    )
    ordering = OrderingFit("discounted_harville", (0.85, 0.7), f"ord-{tag}", 99)
    nowcast = NowcastV0(
        bands=(2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
        grid=GRID,
        prior_n=50.0,
        days=("2026-11-21", "2026-11-22"),
        sigma0={("WIN", tau): 0.05 for tau in GRID.minutes},
        cells={("WIN", 2, 1): DriftCell("WIN", 2, 1, 7, -0.01, 0.04)},
        fit_id=f"now-{tag}",
    )
    return win, ordering, nowcast


def write_pointer(folder: Path, member_entry, *, tag: str = "m1", name: str = "current.json"):
    """The base fits and a pointer naming them, plus `member_entry` as its "member" (no "member"
    key when it is ...). Returns (pointer, (win, ordering, nowcast))."""
    folder.mkdir(parents=True, exist_ok=True)
    win, ordering, nowcast = base_fits(tag)
    names = {"win": "win.json", "ordering": "ordering.json", "nowcast": "nowcast.json"}
    for key, obj in (("win", win), ("ordering", ordering), ("nowcast", nowcast)):
        (folder / names[key]).write_text(obj.to_json(), encoding="utf-8")
    pointer = folder / name
    body = dict(names) if member_entry is ... else {**names, "member": member_entry}
    pointer.write_text(json.dumps(body), encoding="utf-8")
    return pointer, (win, ordering, nowcast)


def default_folder() -> Path:
    return paths.data_dir() / "models" / "japan" / "v0"
