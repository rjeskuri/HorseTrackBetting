"""Synthetic stores, level-2 rows and results for test_competition_study.py (#531, epic #73).

Nothing here calls the code under test (`studies.competition`). Every RNG is seeded. Post times
are `studies.rating.read_results`' stand-in: midnight JST of the race day plus race_no minutes,
in UTC (#94's note: only a cut at a JST day boundary is safe on these stamps).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

JST = "Asia/Tokyo"
PRICE_ERA = date(2014, 6, 7)  # D10: the first priced day (models.v0.store.PRICE_ERA_START)


def softmax_within(u, codes) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / pd.Series(e).groupby(codes).transform("sum").to_numpy()


def post_stamp(day: date, race_no: int) -> pd.Timestamp:
    """Midnight JST of `day` plus race_no minutes, in UTC (read_results' stamp)."""
    return (pd.Timestamp(day).tz_localize(JST) + pd.Timedelta(minutes=race_no)).tz_convert("UTC")


def jst_midnight(day: date) -> pd.Timestamp:
    """00:00 Asia/Tokyo of `day`, in UTC: a fold's cut."""
    return pd.Timestamp(day).tz_localize(JST).tz_convert("UTC")


# --------------------------------------------------------------------------- level-2 rows

ROLES = ("same", "signal", "noise", "both")


def level2_rows(seed: int, members: dict[str, str], base: str) -> pd.DataFrame:
    """Weekend meetings over 2020-2021 (two venues, three races each, 6-12 runners), one row per
    priced starter. win_odds are 0.8 / a market strength's softmax, rounded to 0.1 (at least
    1.1); q is the normalised 1 / win_odds and p_cal = softmax(0.7 ln q), a D4 stand-in. `base`
    is #92's pooled member, an independent log-probability. The winner is drawn from
    softmax(ln q + 0.6 base_u + 0.5 signal): `signal` is what a competition term could add.
    `members` maps each extra member column to its role: 'same' (exactly the base member),
    'signal' (base plus the planted signal), 'noise' (base plus noise the winner never sees)
    or 'both'. Every member column is a log-probability renormalised within race."""
    rng = np.random.default_rng(seed)
    race_ids, race_dates, meetings = [], [], []
    day = date(2020, 1, 4)
    while day <= date(2021, 12, 26):
        for d in (day, day + timedelta(days=1)):
            for venue in ("v1", "v2"):
                for r in range(3):
                    race_ids.append(f"jra:{d:%Y%m%d}:{venue}:{r + 1:02d}")
                    race_dates.append(d)
                    meetings.append(f"{d:%Y%m%d}:{venue}")
        day += timedelta(days=7)
    sizes = rng.integers(6, 13, size=len(race_ids))
    codes = np.repeat(np.arange(len(race_ids)), sizes)
    n = len(codes)
    strength = softmax_within(rng.normal(0.0, 1.0, n), codes)
    odds = np.maximum(1.1, np.round(0.8 / strength, 1))
    inv = 1.0 / odds
    q = inv / pd.Series(inv).groupby(codes).transform("sum").to_numpy()
    ln_q = np.log(q)
    base_u = rng.normal(0.0, 1.0, n)
    signal = rng.normal(0.0, 0.8, n)
    noise = rng.normal(0.0, 0.6, n)
    truth = ln_q + 0.6 * base_u + 0.5 * signal
    noisy = pd.Series(truth + rng.gumbel(size=n))
    won = np.zeros(n, dtype=bool)
    won[noisy.groupby(codes).idxmax().to_numpy()] = True
    utility = {
        "same": base_u,
        "signal": base_u + 0.5 * signal,
        "noise": base_u + noise,
        "both": base_u + 0.5 * signal + 0.5 * noise,
    }
    frame = pd.DataFrame(
        {
            "race_id": np.array(race_ids, dtype=object)[codes],
            "race_date": pd.to_datetime(np.array(race_dates)[codes]),
            "meeting_id": np.array(meetings, dtype=object)[codes],
            "horse_id": [f"h{i:06d}" for i in range(n)],
            "won": won,
            "win_odds": odds,
            "q": q,
            "ln_q": ln_q,
            "p_cal": softmax_within(0.7 * ln_q, codes),
            base: np.log(softmax_within(base_u, codes)),
        }
    )
    for column, role in members.items():
        frame[column] = np.log(softmax_within(utility[role], codes))
    return frame


# --------------------------------------------------------------------------- results


def random_results(seed: int, first: date, last: date) -> pd.DataFrame:
    """Results as `studies.rating.read_results` gives them (race_id, race_date, race_no,
    race_utc, horse_id, finish_pos): two races on each weekend day, 4-7 runners, from a horse
    pool that grows by three new horses every four weeks, so the comparison graph has horses it
    never saw and small components early on."""
    rng = np.random.default_rng(seed)
    rows, pool, week = [], 16, 0
    day = first
    while day <= last:
        if week % 4 == 0:
            pool += 3
        for d in (day, day + timedelta(days=1)):
            for race_no in (1, 2):
                n = int(rng.integers(4, 8))
                recent = max(0, pool - 40)  # older horses retire
                horses = rng.choice(np.arange(recent, pool), size=n, replace=False)
                race_id = f"jra:{d:%Y%m%d}:{race_no:02d}"
                for pos, h in enumerate(rng.permutation(horses), start=1):
                    rows.append((race_id, d, race_no, post_stamp(d, race_no), f"h{h:04d}", pos))
        day += timedelta(days=7)
        week += 1
    out = pd.DataFrame(
        rows, columns=["race_id", "race_date", "race_no", "race_utc", "horse_id", "finish_pos"]
    )
    out["race_date"] = pd.to_datetime(out["race_date"])
    out["race_utc"] = pd.to_datetime(out["race_utc"], utc=True)
    return out


# --------------------------------------------------------------------------- compute_study's inputs

SPARSE_DAYS = (date(2010, 9, 4), date(2011, 1, 8))  # #92's level-1 quarters start at 2011-Q1
DENSE_FIRST, DENSE_LAST = date(2014, 7, 5), date(2016, 6, 26)
PRICED_FROM = date(2015, 1, 1)  # #92's LEVEL1_FROM: level 1 is out of fold from here
ACTIVE = 60  # horses in training at a time


def study_frames(seed: int, members: list[str]) -> dict:
    """compute_study's inputs at a small scale. One race of eight on each SPARSE_DAYS day, then
    each weekend day from DENSE_FIRST to DENSE_LAST two venues x two races of 6-9 runners drawn
    from the ACTIVE newest horses of a pool that grows by two horses every four weeks (so a short
    window leaves idle horses unconnected, and any window the quarter's debutants). Every race has
    one winner, drawn from softmax(strength + 0.6 base + 0.4 signal). Returns 'level1' (race_id,
    race_date, meeting_id, horse_id, finish_pos, won), 'results' (`studies.rating.read_results`'
    columns), 'price' (the priced starters of the races from PRICED_FROM, one non-winner unpriced
    in every fourth race; win_odds = 0.8 / softmax(strength) rounded to 0.1, at least 1.1; q
    normalised over the priced starters; dead_heat False; sorted by date and race on a
    RangeIndex) and 'u': per member, a level-1 utility per level1 row (members[0] is the base;
    the others add the signal and noise in their own measure), NaN before PRICED_FROM."""
    rng = np.random.default_rng(seed)
    races = [(d, "v1", 1, 8, 0) for d in SPARSE_DAYS]
    day = DENSE_FIRST
    while day <= DENSE_LAST:
        newest = ACTIVE + 2 * ((day - DENSE_FIRST).days // 28)
        for d in (day, day + timedelta(days=1)):
            for venue in ("v1", "v2"):
                races += [(d, venue, r, int(rng.integers(6, 10)), newest) for r in (1, 2)]
        day += timedelta(days=7)
    weights = [(0.0, 0.0)] + [((0.5, 0.0, 0.3, 0.8)[k % 4], 0.5) for k in range(len(members) - 1)]
    parts = []
    for k, (d, venue, race_no, n, newest) in enumerate(races):
        horses = rng.choice(np.arange(max(newest - ACTIVE, 0), max(newest, ACTIVE)), n, False)
        strength, base = rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n)
        signal = rng.normal(0.0, 0.8, n)
        winner = int(np.argmax(strength + 0.6 * base + 0.4 * signal + rng.gumbel(size=n)))
        pos = np.empty(n, dtype=int)
        pos[winner] = 1
        pos[np.arange(n) != winner] = rng.permutation(n - 1) + 2
        market = np.exp(strength - strength.max())
        odds = np.maximum(1.1, np.round(0.8 / (market / market.sum()), 1))
        priced = np.ones(n, dtype=bool)
        if k % 4 == 3:
            priced[(winner + 1) % n] = False
        frame = pd.DataFrame(
            {
                "race_id": f"jra:{d:%Y%m%d}:{venue}:{race_no:02d}",
                "race_date": pd.Timestamp(d),
                "race_no": race_no,
                "race_utc": post_stamp(d, race_no),
                "meeting_id": f"{d:%Y%m%d}:{venue}",
                "horse_id": [f"jra:h{h:04d}" for h in horses],
                "finish_pos": pos,
                "won": pos == 1,
                "win_odds": odds,
                "priced": priced,
            }
        )
        for name, (w, sd) in zip(members, weights, strict=True):
            frame[f"u:{name}"] = base + w * signal + rng.normal(0.0, sd, n)
        parts.append(frame)
    every = pd.concat(parts, ignore_index=True)
    every["race_utc"] = pd.to_datetime(every["race_utc"], utc=True)
    level1 = every[["race_id", "race_date", "meeting_id", "horse_id", "finish_pos", "won"]].copy()
    results = every[
        ["race_id", "race_date", "race_no", "race_utc", "horse_id", "finish_pos"]
    ].copy()
    late = every["race_date"].ge(pd.Timestamp(PRICED_FROM)).to_numpy()
    u = {name: np.where(late, every[f"u:{name}"].to_numpy(dtype=float), np.nan) for name in members}
    price = every.loc[late & every["priced"].to_numpy(), ["race_id", "race_date", "horse_id"]]
    price = price.assign(
        win_odds=every["win_odds"],
        won=every["won"],
        dead_heat=False,
        meeting_id=every["meeting_id"],
    )
    inv = 1.0 / price["win_odds"]
    price["q"] = inv / inv.groupby(price["race_id"]).transform("sum")
    price = price.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    return {"level1": level1, "results": results, "price": price, "u": u}


# --------------------------------------------------------------------------- the store

DEV_FIRST, DEV_LAST = date(2013, 1, 5), date(2015, 12, 27)


def write_store(root: Path, holdout_days: list[date], seed: int) -> dict[str, set[str]]:
    """The JRA store's runners and races and the JRA-VAN horse_pedigree under `root`
    (`$HORSETRACK_DATA_DIR`), as parquet: development race days (a weekend every four weeks from
    DEV_FIRST to DEV_LAST, three races of eight a day, every race run with one winner) and races
    on `holdout_days` whose race ids start 'HOLDOUT-', run by the same horses. Returns the race
    ids: 'development', 'price_era' (development races from PRICE_ERA) and 'holdout'."""
    rng = np.random.default_rng(seed)
    days, day = [], DEV_FIRST
    while day <= DEV_LAST:
        days += [day, day + timedelta(days=1)]
        day += timedelta(days=28)
    horses = [f"jra:h{i:03d}" for i in range(80)]
    runners, races = [], []
    ids: dict[str, set[str]] = {"development": set(), "price_era": set(), "holdout": set()}
    for d, poison in [(d, False) for d in days] + [(d, True) for d in holdout_days]:
        for race_no in range(1, 4):
            race_id = f"{'HOLDOUT-' if poison else ''}jra:{d:%Y%m%d}:TOK:{race_no:02d}"
            ids["holdout" if poison else "development"].add(race_id)
            if not poison and d >= PRICE_ERA:
                ids["price_era"].add(race_id)
            races.append(
                (race_id, d, f"{d:%Y%m%d}:TOK", "TOK", race_no, "C1", "turf", 1600)
                + ("GOOD", "age", 8, "run")
            )
            field = rng.choice(len(horses), size=8, replace=False)
            places = rng.permutation(8) + 1
            odds = np.round(rng.uniform(1.5, 60.0, size=8), 1)
            for no, (h, p, o) in enumerate(zip(field, places, odds, strict=True), start=1):
                runners.append(
                    (race_id, d, no, horses[h], int(p), False)
                    + (f"j{rng.integers(12):02d}", f"t{rng.integers(8):02d}", float(o))
                )
    runners = pd.DataFrame(
        runners,
        columns=[
            "race_id",
            "race_date",
            "horse_no",
            "horse_id",
            "finish_pos",
            "is_scratched",
            "jockey_key",
            "trainer_key",
            "win_odds",
        ],
    )
    runners["finish_pos"] = runners["finish_pos"].astype("Int64")
    runners["horse_no"] = runners["horse_no"].astype("Int64")
    races = pd.DataFrame(
        races,
        columns=[
            "race_id",
            "race_date",
            "meeting_id",
            "venue",
            "race_no",
            "race_class",
            "surface",
            "distance_m",
            "going",
            "weight_type",
            "field_size",
            "status",
        ],
    )
    for col in ("race_no", "distance_m", "field_size"):
        races[col] = races[col].astype("Int64")
    pedigree = pd.DataFrame(
        {
            "horse_id": horses,
            "sire_id": [f"jravan:s{i % 9}" for i in range(len(horses))],
            "dam_id": [f"jravan:d{i:03d}" for i in range(len(horses))],
            "dam_sire_id": [f"jravan:s{(i + 4) % 9}" for i in range(len(horses))],
        }
    )
    jra, jravan = root / "processed" / "jra", root / "processed" / "jravan"
    jra.mkdir(parents=True, exist_ok=True)
    jravan.mkdir(parents=True, exist_ok=True)
    runners.to_parquet(jra / "runners.parquet", index=False)
    races.to_parquet(jra / "races.parquet", index=False)
    pedigree.to_parquet(jravan / "horse_pedigree.parquet", index=False)
    return ids
