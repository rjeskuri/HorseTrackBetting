"""Synthetic runs, pedigrees and pedigree tables for test_lineage_run.py (#538, epic #73).

Nothing here calls the code under test (`studies.lineage`'s run). The expected features are
built from #95's `models.lineage.LineagePrior` (library code on main, not the code under test)
fitted at each quarter's start, and from hand counts of each horse's earlier race days. Finishing
orders are Plackett-Luce draws on planted sire x surface x band utilities (ability_synth's Gumbel
sampler). Every RNG is seeded. Every post time is 02:00 to 04:00 UTC on its race date, so "before
a quarter's start" reads the same at 00:00 UTC or 00:00 JST of the start date.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from ability_synth import pl_positions

from horsetrack.core.store import schemas
from horsetrack.countries.japan.jrdb import pedigree as jrdb_pedigree
from horsetrack.countries.japan.models.lineage import LineagePrior

RUN_COLUMNS = [
    "race_id",
    "race_date",
    "race_utc",
    "horse_id",
    "finish_pos",
    "surface",
    "dist_band",
]
SURFACES = ("turf", "dirt")
BANDS = ("sprint", "mile", "middle")  # opaque labels: the cut points are the frozen spec's
STALLIONS = tuple(f"s:{i:02d}" for i in range(10))
RUNNERS = 8
RACES_PER_DAY = 3
THIN = 30  # fewer prior starts than this is thin history (the ticket's THIN_STARTS)

# Five D9 quarters; the first has no earlier run, so its prior is fitted on nothing.
QUARTER_STARTS = (
    date(2023, 10, 1),
    date(2024, 1, 1),
    date(2024, 4, 1),
    date(2024, 7, 1),
    date(2024, 10, 1),
)
LAST_DAY = date(2024, 12, 31)


def post_utc(day: date, race: int) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=2, minutes=40 * race)


def frame(rows) -> pd.DataFrame:
    """The runs frame from (race_id, day, race slot, horse, finish_pos, surface, band) tuples."""
    out = pd.DataFrame(
        {
            "race_id": [r[0] for r in rows],
            "race_date": pd.to_datetime([pd.Timestamp(r[1]) for r in rows]),
            "race_utc": pd.DatetimeIndex([post_utc(r[1], r[2]) for r in rows]),
            "horse_id": pd.Series([r[3] for r in rows], dtype=object),
            "finish_pos": np.array([r[4] for r in rows], dtype=float),
            "surface": pd.Series([r[5] for r in rows], dtype=object),
            "dist_band": pd.Series([r[6] for r in rows], dtype=object),
        }
    )
    return out[RUN_COLUMNS]


def lineage_pedigree(rows) -> pd.DataFrame:
    """The LineagePrior pedigree (horse_id, sire_id, dam_id, dam_sire_id) from (horse, sire,
    damsire) tuples; every dam unknown, so the damsire is the given dam_sire_id."""
    out = pd.DataFrame(list(rows), columns=["horse_id", "sire_id", "dam_sire_id"], dtype=object)
    out.insert(2, "dam_id", None)
    return out.astype(object).where(out.notna(), None)


def race_days() -> list[date]:
    """Every fourth day from the first quarter's start, plus each quarter's first day and the
    day before it, so runs sit on both sides of every boundary."""
    days = {QUARTER_STARTS[0] + timedelta(days=4 * i) for i in range(200)}
    days |= set(QUARTER_STARTS) | {s - timedelta(days=1) for s in QUARTER_STARTS[1:]}
    return sorted(d for d in days if QUARTER_STARTS[0] <= d <= LAST_DAY)


def asof_runs(seed: int, *, horses: int = 300, veterans: int = 2):
    """(runs, pedigree) over the five quarters: RACES_PER_DAY races of RUNNERS runners a race
    day, each horse at most once a day. Each horse has a sire and a damsire among STALLIONS
    (planted sire effects and sire x cell deviations, the damsire at 0.25) and a debut day (the
    first 60 at the first day, the rest spread over the window), so every quarter has debuts and
    thin-history runners. `veterans` horses run every race day's first race, so they pass 30
    prior starts during the window. About 3% of runs have no placing."""
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, 0.6, len(STALLIONS))
    g -= g.mean()
    dev = rng.normal(0.0, 0.5, (len(STALLIONS), len(SURFACES), len(BANDS)))
    days = race_days()
    n = horses + veterans
    sire = rng.integers(0, len(STALLIONS), n)
    damsire = rng.integers(0, len(STALLIONS), n)
    debut = np.r_[np.zeros(60, dtype=int), rng.integers(0, len(days), n - 60)]
    debut[horses:] = 0
    ids = [f"h{i:04d}" for i in range(horses)] + [f"v{i}" for i in range(veterans)]
    own = rng.normal(0.0, 0.3, n)
    rows = []
    for di, day in enumerate(days):
        used: set[int] = set()
        for race in range(RACES_PER_DAY):
            field = list(range(horses, n)) if race == 0 else []
            pool = [h for h in range(horses) if debut[h] <= di and h not in used]
            take = rng.choice(len(pool), RUNNERS - len(field), replace=False)
            field += [pool[t] for t in take]
            used.update(field)
            s, b = int(rng.integers(len(SURFACES))), int(rng.integers(len(BANDS)))
            u = np.array(
                [
                    g[sire[h]]
                    + dev[sire[h], s, b]
                    + 0.25 * (g[damsire[h]] + dev[damsire[h], s, b])
                    + own[h]
                    for h in field
                ]
            )
            pos = pl_positions(rng, u, np.zeros(len(field), dtype=np.int64)).astype(float)
            pos[rng.random(len(field)) < 0.03] = np.nan
            race_id = f"jra:{day:%Y%m%d}r{race}"
            for h, p in zip(field, pos, strict=True):
                rows.append((race_id, day, race, ids[h], p, SURFACES[s], BANDS[b]))
    ped = lineage_pedigree((ids[h], STALLIONS[sire[h]], STALLIONS[damsire[h]]) for h in range(n))
    return frame(rows), ped


VETERAN = "vet"
HOT = "s:hot"  # the veteran's sire, planted well above the field


def veteran_runs(seed: int):
    """(runs, pedigree, starts): a veteran by HOT runs every Saturday from 2024-01-06, 34 times,
    against five others a race; its sixth run has no placing. Each race's others are debut or
    second-start horses, half by HOT and half by field sires; every horse's damsire is a field
    sire. Quarter starts 2024-01-01, 04-01 and 07-01: the veteran's 30th run (29 prior starts,
    2024-07-27) and 31st (30 prior starts, 2024-08-03) both fall in the third quarter, whose
    prior reads the runs before 2024-07-01."""
    rng = np.random.default_rng(seed)
    field_sires = [f"f:{i}" for i in range(6)]
    effect = {HOT: 1.2} | dict(zip(field_sires, rng.normal(0.0, 0.3, 6), strict=True))
    ped_rows = [(VETERAN, HOT, field_sires[0])]
    rows, count = [], 0
    for week in range(34):
        day = date(2024, 1, 6) + timedelta(weeks=week)
        race_id = f"jra:{day:%Y%m%d}r0"
        field = [VETERAN]
        for _ in range(5):
            h = f"y{count:03d}"
            count += 1
            s = HOT if count % 2 else field_sires[int(rng.integers(6))]
            ped_rows.append((h, s, field_sires[int(rng.integers(6))]))
            field.append(h)
        u = np.array([2.0] + [effect[p[1]] for p in ped_rows[-5:]])
        pos = pl_positions(rng, u, np.zeros(6, dtype=np.int64)).astype(float)
        if week == 5:
            pos[0] = np.nan  # the veteran runs and is not placed: still a start
        band = BANDS[week % 2]
        for h, p in zip(field, pos, strict=True):
            rows.append((race_id, day, 0, h, p, "turf", band))
        if week % 3 == 0:  # some of this race's debutants run again two days later
            again = f"jra:{day + timedelta(days=1):%Y%m%d}r1"
            back = field[1:4]
            u2 = np.array([effect[p[1]] for p in ped_rows if p[0] in back])
            pos2 = pl_positions(rng, u2, np.zeros(3, dtype=np.int64)).astype(float)
            for h, p in zip(back, pos2, strict=True):
                rows.append((again, day + timedelta(days=1), 1, h, p, "turf", "mile"))
    starts = (date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1))
    return frame(rows), lineage_pedigree(ped_rows), starts


# --------------------------------------------------------------------------- expected values


def quarter_of(race_date: pd.Series, starts) -> np.ndarray:
    """Per row: the index of the last start on or before its race date."""
    edges = np.array([np.datetime64(s, "D") for s in starts])
    day = race_date.to_numpy().astype("datetime64[D]")
    return np.searchsorted(edges, day, side="right") - 1


def prior_starts(runs: pd.DataFrame) -> pd.Series:
    """Per row, by hand: the horse's runs in `runs` on earlier race days (a debut has 0)."""
    days = runs["race_date"].to_numpy().astype("datetime64[D]")
    horses = runs["horse_id"].to_numpy(dtype=object)
    out = np.zeros(len(runs), dtype=np.int64)
    by_horse: dict = {}
    for i, h in enumerate(horses):
        by_horse.setdefault(h, []).append(i)
    for idx in by_horse.values():
        d = days[idx]
        out[idx] = np.searchsorted(np.sort(d), d, side="left")
    return pd.Series(out, index=runs.index)


def quarter_priors(runs, ped, starts, params) -> pd.Series:
    """Per row: LineagePrior(**params) fitted on the runs before its quarter's start (00:00 UTC
    of the start date), read at the row's horse, surface and band; no trainer effect."""
    q = quarter_of(runs["race_date"], starts)
    if (q < 0).any():
        raise AssertionError("a fixture row precedes the first quarter start")
    out = pd.Series(np.nan, index=runs.index, dtype=float)
    for k, start in enumerate(starts):
        mine = q == k
        if not mine.any():
            continue
        prior = LineagePrior(**params).fit(
            runs, ped, before_utc=pd.Timestamp(start).tz_localize("UTC")
        )
        rows = runs.loc[mine, ["horse_id", "surface", "dist_band"]]
        out.loc[rows.index] = prior.prior_means(rows).to_numpy()
    return out


def expected_feature(runs, ped, starts, params) -> pd.Series:
    """The ticket's feature from first principles: the quarter's prior mean for a runner with
    fewer than THIN prior starts (debut included), 0 otherwise."""
    prior = quarter_priors(runs, ped, starts, params)
    thin = prior_starts(runs).to_numpy() < THIN
    return pd.Series(np.where(thin, prior.to_numpy(), 0.0), index=runs.index)


# --------------------------------------------------------------------------- pedigree tables

STORE_COLUMNS = tuple(schemas.TABLES["horse_pedigree"].columns)
# The columns a UKC (JRDB) merged horse_pedigree carries beyond the netkeiba store schema.
UKC_COLUMNS = tuple(c for c in jrdb_pedigree.OUTPUT_COLUMNS if c not in STORE_COLUMNS)


def netkeiba_table(rows) -> pd.DataFrame:
    """A netkeiba `processed/jra/horse_pedigree` frame in the store schema from (horse, sire
    name, dam name, damsire name) tuples (None: missing): the *_name_norm columns are the
    casefolded names, every id column is empty."""

    def norm(v):
        return None if v is None else v.casefold()

    out = pd.DataFrame(
        {
            "jurisdiction": "jra",
            "horse_id": [r[0] for r in rows],
            "sire_name": [r[1] for r in rows],
            "sire_name_norm": [norm(r[1]) for r in rows],
            "sire_id": None,
            "dam_name": [r[2] for r in rows],
            "dam_name_norm": [norm(r[2]) for r in rows],
            "dam_id": None,
            "dam_sire_name": [r[3] for r in rows],
            "dam_sire_name_norm": [norm(r[3]) for r in rows],
            "dam_sire_id": None,
            "source_race_id": [f"jra:2023{i:04d}" for i in range(len(rows))],
            "scraped_utc": pd.Timestamp("2026-11-01 00:00", tz="UTC"),
        },
        dtype=object,
    )
    out["scraped_utc"] = pd.to_datetime(out["scraped_utc"], utc=True)
    return out[list(STORE_COLUMNS)]


def ukc_table(rows) -> pd.DataFrame:
    """A UKC-built horse_pedigree (jrdb.pedigree.OUTPUT_COLUMNS) from (horse, sire name, dam
    name, damsire name) tuples, with line codes, birth years and paternal hops filled."""
    out = pd.DataFrame(
        {
            "horse_id": [r[0] for r in rows],
            "sire_name": [r[1] for r in rows],
            "sire_name_norm": [jrdb_pedigree.normalise_name(r[1]) for r in rows],
            "dam_name": [r[2] for r in rows],
            "dam_name_norm": [jrdb_pedigree.normalise_name(r[2]) for r in rows],
            "dam_sire_name": [r[3] for r in rows],
            "dam_sire_name_norm": [jrdb_pedigree.normalise_name(r[3]) for r in rows],
            "sire_line_code": "11",
            "dam_sire_line_code": "12",
            "sire_birth_year": 2005,
            "dam_birth_year": 2012,
            "dam_sire_birth_year": 1998,
            "paternal_grandsire_name_norm": "grand",
            "paternal_great_grandsire_name_norm": "great",
        }
    )
    return out[list(jrdb_pedigree.OUTPUT_COLUMNS)].astype(
        {c: "Int64" for c in ("sire_birth_year", "dam_birth_year", "dam_sire_birth_year")}
    )
