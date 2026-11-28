"""Synthetic races for test_interactions.py (#93, epic #73).

Nothing here calls the code under test (`models.interactions`). The expected values are the
planted effects; finishing orders are drawn by Plackett-Luce on the planted utility (the Gumbel
sampler of ability_synth.py). Every RNG is seeded. The designs' sizes were set so that the
reference estimator (#92's `AbilityModel` with the cell as one more entity column) meets the
tolerances by a wide margin over many seeds; that reference is not used by any test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ability_synth import centred, field, labels, pl_positions

FIRST_POST = pd.Timestamp("2024-01-06 01:00", tz="UTC")
RACES_PER_DAY = 24
X_COEF = 0.7
VENUES = ("tokyo", "nakayama", "kyoto", "hanshin")
DISTANCES = np.array([1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 3000, 3200, 3400])


def race_times(codes: np.ndarray) -> pd.DatetimeIndex:
    """One tz-aware UTC post time per row: race c runs on day c // RACES_PER_DAY at slot
    c % RACES_PER_DAY, 30 minutes apart, so race codes are in time order."""
    codes = np.asarray(codes)
    days = pd.to_timedelta(codes // RACES_PER_DAY, unit="D")
    slots = pd.to_timedelta((codes % RACES_PER_DAY) * 30, unit="min")
    return FIRST_POST + days + slots


def after(frame: pd.DataFrame, days: int = 7) -> pd.Timestamp:
    """A post time `days` after frame's last race, for probe races."""
    return pd.Timestamp(frame["race_utc"].max()) + pd.Timedelta(days=days)


# ---------------------------------------------------------------------------------------------
# Trainer x track as a variance component (planted tau)
# ---------------------------------------------------------------------------------------------

# 12 trainers x 4 venues = 48 cells; 9,000 races of 6 to 8 runners, about 1,300 runs per cell
# (about 1,000 in the first 7,000 races, the training window). u = X_COEF * x1 + the trainer's
# main effect (sd TRAINER_SD) + the trainer x venue cell effect.
TT_TRAINERS = 12
TT_RACES = 9000
TT_TRAIN_RACES = 7000
TRAINER_SD = 0.3


def doubly_centred(rng: np.random.Generator, rows: int, cols: int, tau: float) -> np.ndarray:
    """Cell effects with mean 0 in every row and column, scaled so that their sample variance on
    (rows - 1) * (cols - 1) degrees of freedom is exactly tau^2 (all 0 at tau = 0). Row means
    would be a trainer main effect and column means a venue constant, which cancels within a
    race, so the planted tau is the spread the data can identify."""
    if tau == 0:
        return np.zeros((rows, cols))
    g = rng.normal(size=(rows, cols))
    g = g - g.mean(axis=0, keepdims=True)
    g = g - g.mean(axis=1, keepdims=True)
    return g * tau / np.sqrt((g * g).sum() / ((rows - 1) * (cols - 1)))


def trainer_track_frame(seed: int, tau: float) -> tuple[pd.DataFrame, np.ndarray]:
    """(frame, race codes): race_id, race_utc, horse_id, finish_pos, x1, trainer_id, venue."""
    rng = np.random.default_rng(seed)
    codes = field(rng, TT_RACES, sizes=(6, 8))
    n = len(codes)
    venue = rng.integers(0, len(VENUES), size=TT_RACES)[codes]
    trainer = rng.integers(0, TT_TRAINERS, size=n)
    main = centred(rng.normal(0.0, TRAINER_SD, size=TT_TRAINERS))
    cells = doubly_centred(rng, TT_TRAINERS, len(VENUES), tau)
    x1 = rng.normal(size=n)
    u = X_COEF * x1 + main[trainer] + cells[trainer, venue]
    pos = pl_positions(rng, u, codes)
    frame = pd.DataFrame(
        {
            "race_id": labels(codes),
            "race_utc": race_times(codes),
            "horse_id": np.array([f"h:{i:06d}" for i in range(n)], dtype=object),
            "finish_pos": pos.astype(np.int64),
            "x1": x1,
            "trainer_id": np.array([f"t:{i:02d}" for i in trainer], dtype=object),
            "venue": np.array(VENUES, dtype=object)[venue],
        }
    )
    return frame, codes


# ---------------------------------------------------------------------------------------------
# Structured distance aptitude (planted optimum)
# ---------------------------------------------------------------------------------------------

# 900 races of 8 runners: 3 of the 6 focal horses and 5 of the 50 field horses. Each focal horse
# has a peaked aptitude -KAPPA * ((d - optimum) / 1000 m)^2 (about 450 runs over the 12
# distances); the field horses have none. Every horse has a level (sd 0.3). Surface and venue
# are drawn per race and carry no planted effect.
OPTIMA = {"h:f0": 1200, "h:f1": 1200, "h:f2": 2000, "h:f3": 2000, "h:f4": 3000, "h:f5": 3000}
FIELD_HORSES = tuple(f"h:n{i:02d}" for i in range(50))
KAPPA = 2.0
APT_RACES = 900
FOCAL_PER_RACE = 3
FIELD_PER_RACE = 5


def aptitude_frame(seed: int) -> pd.DataFrame:
    """race_id, race_utc, horse_id, finish_pos, x1, distance_m, surface, venue."""
    rng = np.random.default_rng(seed)
    ids = np.array([*OPTIMA, *FIELD_HORSES], dtype=object)
    n_focal = len(OPTIMA)
    picks = [
        np.r_[
            rng.choice(n_focal, FOCAL_PER_RACE, replace=False),
            n_focal + rng.choice(len(FIELD_HORSES), FIELD_PER_RACE, replace=False),
        ]
        for _ in range(APT_RACES)
    ]
    horse = np.concatenate(picks)
    codes = np.repeat(np.arange(APT_RACES), FOCAL_PER_RACE + FIELD_PER_RACE)
    distance = rng.choice(DISTANCES, size=APT_RACES)[codes]
    surface = rng.choice(np.array(["turf", "dirt"], dtype=object), size=APT_RACES)[codes]
    venue = rng.choice(np.array(VENUES, dtype=object), size=APT_RACES)[codes]
    level = rng.normal(0.0, 0.3, size=len(ids))
    optimum = np.array([OPTIMA.get(h, 0) for h in ids], dtype=float)[horse]
    focal = horse < n_focal
    aptitude = np.where(focal, -KAPPA * ((distance - optimum) / 1000.0) ** 2, 0.0)
    x1 = rng.normal(size=len(codes))
    u = X_COEF * x1 + level[horse] + aptitude
    pos = pl_positions(rng, u, codes)
    return pd.DataFrame(
        {
            "race_id": labels(codes, prefix="jra:a"),
            "race_utc": race_times(codes),
            "horse_id": ids[horse],
            "finish_pos": pos.astype(np.int64),
            "x1": x1,
            "distance_m": distance.astype(np.int64),
            "surface": surface,
            "venue": venue,
        }
    )


# ---------------------------------------------------------------------------------------------
# One-factor trainer x track term (east/west, E36)
# ---------------------------------------------------------------------------------------------

# Venues: tokyo and nakayama east (+1), kyoto and hanshin west (-1). Trainers by role: four east
# (+1), four west (-1), two neutral (0), and two sparse ones, one east and one west, that run at
# tokyo and kyoto only. The planted cell effect is EW_EFFECT * s_trainer * s_venue (rank one).
# 3,000 races of 6 to 8 runners. Trainer ids are opaque: the role is not in the id.
EW_EFFECT = 0.4
VENUE_SIGN = {"tokyo": 1, "nakayama": 1, "kyoto": -1, "hanshin": -1}
EW_ROLES = {
    "e1": 1,
    "e2": 1,
    "e3": 1,
    "e4": 1,
    "w1": -1,
    "w2": -1,
    "w3": -1,
    "w4": -1,
    "n1": 0,
    "n2": 0,
    "se": 1,
    "sw": -1,
}
SPARSE_ROLES = ("se", "sw")
SPARSE_VENUES = ("tokyo", "kyoto")
EW_RACES = 3000


def east_west_frame(seed: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """(frame, trainer id by role): race_id, race_utc, horse_id, finish_pos, x1, trainer_id,
    venue."""
    rng = np.random.default_rng(seed)
    roles = list(EW_ROLES)
    perm = rng.permutation(len(roles))
    trainer_of = {role: f"t:{perm[i]:02d}" for i, role in enumerate(roles)}
    codes = field(rng, EW_RACES, sizes=(6, 8))
    n = len(codes)
    venue = np.array(VENUES, dtype=object)[rng.integers(0, len(VENUES), size=EW_RACES)][codes]
    regular = np.array([i for i, r in enumerate(roles) if r not in SPARSE_ROLES])
    everyone = np.arange(len(roles))
    open_to_all = np.isin(venue, SPARSE_VENUES)
    role = np.where(
        open_to_all,
        everyone[rng.integers(0, len(everyone), size=n)],
        regular[rng.integers(0, len(regular), size=n)],
    )
    sign = np.array([EW_ROLES[r] for r in roles], dtype=float)
    main = centred(rng.normal(0.0, 0.2, size=len(roles)))
    venue_sign = np.array([VENUE_SIGN[v] for v in venue], dtype=float)
    x1 = rng.normal(size=n)
    u = X_COEF * x1 + main[role] + EW_EFFECT * sign[role] * venue_sign
    pos = pl_positions(rng, u, codes)
    ids = np.array([trainer_of[r] for r in roles], dtype=object)
    frame = pd.DataFrame(
        {
            "race_id": labels(codes, prefix="jra:e"),
            "race_utc": race_times(codes),
            "horse_id": np.array([f"h:{i:06d}" for i in range(n)], dtype=object),
            "finish_pos": pos.astype(np.int64),
            "x1": x1,
            "trainer_id": ids[role],
            "venue": venue,
        }
    )
    return frame, trainer_of


# ---------------------------------------------------------------------------------------------
# Every block kind on one card (horse x jockey cells of 1 to 10 runs)
# ---------------------------------------------------------------------------------------------

# 400 races of 8 runners from 60 horses and 12 jockeys (distinct within a race); each horse keeps
# one of 6 trainers (horse % 6). Planted: x1, a horse level, a trainer main effect, trainer x
# venue cells (sd 0.2), a per-horse distance slope (sd 0.3); no horse x jockey effect (E35's
# negative control).
MIX_RACES = 400
MIX_TRAIN_RACES = 350


def mixed_frame(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """(frame, race codes): race_id, race_utc, horse_id, finish_pos, x1, jockey_id, trainer_id,
    venue, distance_m, surface."""
    rng = np.random.default_rng(seed)
    runners = 8
    codes = np.repeat(np.arange(MIX_RACES), runners)
    horse = np.concatenate([rng.choice(60, runners, replace=False) for _ in range(MIX_RACES)])
    jockey = np.concatenate([rng.choice(12, runners, replace=False) for _ in range(MIX_RACES)])
    trainer = horse % 6
    venue_idx = rng.integers(0, len(VENUES), size=MIX_RACES)[codes]
    distance = rng.choice(DISTANCES, size=MIX_RACES)[codes]
    surface = rng.choice(np.array(["turf", "dirt"], dtype=object), size=MIX_RACES)[codes]
    level = rng.normal(0.0, 0.3, size=60)
    slope = rng.normal(0.0, 0.3, size=60)
    t_main = centred(rng.normal(0.0, 0.2, size=6))
    cells = rng.normal(0.0, 0.2, size=(6, len(VENUES)))
    x1 = rng.normal(size=len(codes))
    z = (distance - 2000.0) / 700.0
    u = X_COEF * x1 + level[horse] + slope[horse] * z + t_main[trainer] + cells[trainer, venue_idx]
    pos = pl_positions(rng, u, codes)
    frame = pd.DataFrame(
        {
            "race_id": labels(codes, prefix="jra:m"),
            "race_utc": race_times(codes),
            "horse_id": np.array([f"h:{i:02d}" for i in horse], dtype=object),
            "finish_pos": pos.astype(np.int64),
            "x1": x1,
            "jockey_id": np.array([f"j:{i:02d}" for i in jockey], dtype=object),
            "trainer_id": np.array([f"t:{i}" for i in trainer], dtype=object),
            "venue": np.array(VENUES, dtype=object)[venue_idx],
            "distance_m": distance.astype(np.int64),
            "surface": surface,
        }
    )
    return frame, codes


# ---------------------------------------------------------------------------------------------
# Aliasing: horse x trainer when every horse keeps one trainer
# ---------------------------------------------------------------------------------------------

# 50 races of 8 runners; race r runs horses (8r .. 8r + 7) mod 40, so each of the 40 horses runs
# 10 times. A horse's home trainer is horse % 6; with trainer_change its last 5 runs are for
# (home + 1) % 6. Jockeys are drawn per race (8, distinct), or with jockey_constant each horse
# keeps jockey horse % 8 (distinct within a race too). Venue, distance and surface are drawn per
# race.
ALIAS_RACES = 50


def alias_frame(seed: int, *, trainer_change: bool, jockey_constant: bool = False) -> pd.DataFrame:
    """race_id, race_utc, horse_id, finish_pos, x1, jockey_id, trainer_id, venue, distance_m,
    surface."""
    rng = np.random.default_rng(seed)
    runners = 8
    codes = np.repeat(np.arange(ALIAS_RACES), runners)
    horse = (runners * codes + np.tile(np.arange(runners), ALIAS_RACES)) % 40
    run = pd.Series(horse).groupby(horse).cumcount().to_numpy()
    home = horse % 6
    trainer = np.where(trainer_change & (run >= 5), (home + 1) % 6, home)
    if jockey_constant:
        jockey = horse % 8
    else:
        jockey = np.concatenate([rng.permutation(runners) for _ in range(ALIAS_RACES)])
    venue = np.array(VENUES, dtype=object)[rng.integers(0, len(VENUES), size=ALIAS_RACES)]
    distance = rng.choice(DISTANCES, size=ALIAS_RACES)
    surface = rng.choice(np.array(["turf", "dirt"], dtype=object), size=ALIAS_RACES)
    x1 = rng.normal(size=len(codes))
    pos = pl_positions(rng, X_COEF * x1, codes)
    return pd.DataFrame(
        {
            "race_id": labels(codes, prefix="jra:x"),
            "race_utc": race_times(codes),
            "horse_id": np.array([f"h:{i:02d}" for i in horse], dtype=object),
            "finish_pos": pos.astype(np.int64),
            "x1": x1,
            "jockey_id": np.array([f"j:{i}" for i in jockey], dtype=object),
            "trainer_id": np.array([f"t:{i}" for i in trainer], dtype=object),
            "venue": venue[codes],
            "distance_m": distance[codes].astype(np.int64),
            "surface": surface[codes],
        }
    )


# ---------------------------------------------------------------------------------------------
# Trainer sequences for the transfer step
# ---------------------------------------------------------------------------------------------


def sequence_frame(sequences: dict[str, str], seed: int) -> tuple[pd.DataFrame, dict]:
    """One row per run: horse key -> its trainers run by run as letters ("AAABBB"). Runs are 14
    days apart, each horse offset by its own minutes so horses interleave; race ids are opaque
    and not in time order; rows are shuffled on index labels that are not positions. Returns
    (frame, {(horse key, run number from 0): index label})."""
    rng = np.random.default_rng(seed)
    rows = []
    for h, (key, letters) in enumerate(sequences.items()):
        for k, letter in enumerate(letters):
            when = FIRST_POST + pd.Timedelta(days=14 * k) + pd.Timedelta(minutes=7 * h)
            rows.append((key, k, f"h:{h:04d}", f"t:{letter}", when))
    frame = pd.DataFrame(rows, columns=["key", "run", "horse_id", "trainer_id", "race_utc"])
    frame["race_id"] = [f"jra:q{c:06d}" for c in rng.permutation(len(frame))]
    frame.index = pd.Index(70_000 + rng.permutation(len(frame)), name="row")
    frame = frame.iloc[rng.permutation(len(frame))]
    labelled = zip(frame.index, frame["key"], frame["run"], strict=True)
    where = {(key, run): label for label, key, run in labelled}
    return frame.drop(columns=["key", "run"]), where


# ---------------------------------------------------------------------------------------------
# Sire lines under the structured aptitude (#572: the lineage prior on b_h and e_h)
# ---------------------------------------------------------------------------------------------

# Four sire lines with distinct distance slopes b and dirt terms e, each sign of each in two
# lines, so both coefficients average 0 over the lines (a common slope or dirt term cancels
# within a race: only the prior sets its level). The units are the structured block's: b per
# unit z = (d - mean) / sd over the training rows (population sd), e on the dirt indicator.
# Per line: LINE_VETERANS horses that run about 400 times in the training races, one of them (in
# lines s:0 and s:3, the opposites) an outlier LINE_OUTLIER away from its line in both terms;
# LINE_YOUNG horses of YOUNG_RUNS training runs; and one debut horse that never runs in training.
# A horse's own (b, e) is its line's plus a deviation (sd LINE_SD, centred over the veterans and
# over the young horses). Every horse has a level (sd 0.3) and no quadratic term. 1,200 training
# races of 8 runners, at most one young horse in a race; then LINE_TEST_RACES held-out races,
# each with two young horses (their later runs) among six veterans. Distance and surface are
# drawn per race.
SIRE_LINES = {"s:0": (0.9, 0.7), "s:1": (-0.9, 0.7), "s:2": (0.9, -0.7), "s:3": (-0.9, -0.7)}
LINE_VETERANS = 6
LINE_YOUNG = 6
YOUNG_RUNS = 2
LINE_SD = 0.2
LINE_OUTLIER = 1.2
OUTLIER_LINES = ("s:0", "s:3")
LINE_TRAIN_RACES = 1200
LINE_TEST_RACES = 600
LINE_RUNNERS = 8


def lineage_aptitude_frame(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, test, horses). train and test: race_id, race_utc, horse_id, finish_pos, x1,
    distance_m, surface. horses, one row per horse (index horse_id): sire_id, role ('veteran',
    'outlier', 'young' or 'debut'), b and e (its planted own terms, in the units above), and
    n_train (its training runs)."""
    rng = np.random.default_rng(seed)
    rows = []  # horse_id, sire_id, role
    for s, sire in enumerate(SIRE_LINES):
        for i in range(LINE_VETERANS):
            outlier = sire in OUTLIER_LINES and i == 0
            rows.append((f"h:v{s}{i}", sire, "outlier" if outlier else "veteran"))
        for i in range(LINE_YOUNG):
            rows.append((f"h:y{s}{i}", sire, "young"))
        rows.append((f"h:d{s}", sire, "debut"))
    horses = pd.DataFrame(rows, columns=["horse_id", "sire_id", "role"]).set_index("horse_id")
    line = np.array([SIRE_LINES[s] for s in horses["sire_id"]], dtype=float)
    own = line.copy()
    for role in ("veteran", "young"):
        at = (horses["role"] == role).to_numpy()
        dev = rng.normal(0.0, LINE_SD, size=(int(at.sum()), 2))
        own[at] += dev - dev.mean(axis=0)
    out = (horses["role"] == "outlier").to_numpy()
    own[out] -= np.sign(line[out]) * LINE_OUTLIER
    horses["b"], horses["e"] = own[:, 0], own[:, 1]
    level = rng.normal(0.0, 0.3, size=len(horses))

    ids = horses.index.to_numpy(dtype=object)
    veterans = np.flatnonzero(horses["role"].isin(["veteran", "outlier"]).to_numpy())
    young = np.flatnonzero((horses["role"] == "young").to_numpy())
    young_slots = rng.permutation(np.repeat(young, YOUNG_RUNS))
    young_races = rng.choice(LINE_TRAIN_RACES, size=len(young_slots), replace=False)
    picks = []
    for r in range(LINE_TRAIN_RACES):
        hit = np.flatnonzero(young_races == r)
        first = [young_slots[hit[0]]] if len(hit) else []
        rest = rng.choice(veterans, LINE_RUNNERS - len(first), replace=False)
        picks.append(np.r_[first, rest].astype(np.int64))
    for _ in range(LINE_TEST_RACES):
        picks.append(
            np.r_[
                rng.choice(young, 2, replace=False),
                rng.choice(veterans, LINE_RUNNERS - 2, replace=False),
            ].astype(np.int64)
        )
    n_races = LINE_TRAIN_RACES + LINE_TEST_RACES
    horse = np.concatenate(picks)
    codes = np.repeat(np.arange(n_races), LINE_RUNNERS)
    distance = rng.choice(DISTANCES, size=n_races)[codes].astype(float)
    surface = rng.choice(np.array(["turf", "dirt"], dtype=object), size=n_races)[codes]
    in_train = codes < LINE_TRAIN_RACES
    z = (distance - distance[in_train].mean()) / distance[in_train].std()
    dirt = (surface == "dirt").astype(float)
    x1 = rng.normal(size=len(codes))
    u = X_COEF * x1 + level[horse] + own[horse, 0] * z + own[horse, 1] * dirt
    pos = pl_positions(rng, u, codes)
    frame = pd.DataFrame(
        {
            "race_id": labels(codes, prefix="jra:l"),
            "race_utc": race_times(codes),
            "horse_id": ids[horse],
            "finish_pos": pos.astype(np.int64),
            "x1": x1,
            "distance_m": distance.astype(np.int64),
            "surface": surface,
        }
    )
    train, test = frame[in_train], frame[~in_train]
    horses["n_train"] = train["horse_id"].value_counts().reindex(horses.index, fill_value=0)
    return train.reset_index(drop=True), test.reset_index(drop=True), horses
