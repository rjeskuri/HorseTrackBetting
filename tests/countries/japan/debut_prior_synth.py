"""Synthetic runs and scored rows for the trainer debut prior's segment measurement (#573, #93).

Nothing here calls the code under test (`studies.debut_prior`). Expected labels are written out by
hand (`hand_history`) or by a brute-force scan of the horse's earlier rows (`brute_debut`);
expected scores are computed in the tests from the planted probabilities. Every RNG is seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

T0 = pd.Timestamp("2024-02-03 01:00", tz="UTC")
DAY = pd.Timedelta(days=1)

# race id -> (days after T0, class); r4 and r5 share one post time (two venues, one slot)
HAND_RACES = {
    "r1": (0, "DEBUT"),
    "r2": (7, "MAIDEN"),
    "r3": (14, "MAIDEN"),
    "r4": (30, "OP"),
    "r5": (30, "G1"),
    "r6": (40, "G1"),
    "r7": (45, "C2"),
}
# race id -> horses; the debut starters are those with no earlier row of their own
HAND_FIELDS = {
    "r1": ["A", "D", "E"],
    "r2": ["A", "B", "F"],
    "r3": ["A", "B", "C", "F"],
    "r4": ["D", "C", "G"],
    "r5": ["E", "H"],
    "r6": ["A", "B", "D"],
    "r7": ["A", "H", "Z"],
}
# (race, horse) -> is this the horse's debut in the frame, by reading the fields above
HAND_DEBUT = {
    ("r1", "A"): True,
    ("r1", "D"): True,
    ("r1", "E"): True,
    ("r2", "A"): False,
    ("r2", "B"): True,
    ("r2", "F"): True,
    ("r3", "A"): False,
    ("r3", "B"): False,
    ("r3", "C"): True,
    ("r3", "F"): False,
    ("r4", "D"): False,
    ("r4", "C"): False,
    ("r4", "G"): True,
    ("r5", "E"): False,
    ("r5", "H"): True,
    ("r6", "A"): False,
    ("r6", "B"): False,
    ("r6", "D"): False,
    ("r7", "A"): False,
    ("r7", "H"): False,
    ("r7", "Z"): True,
}
# race -> the race-level segment: its class, then whether any starter debuts
HAND_SEGMENT = {
    "r1": "DEBUT/debut",
    "r2": "MAIDEN/debut",
    "r3": "MAIDEN/debut",
    "r4": "OP/debut",
    "r5": "G1/debut",
    "r6": "G1/seasoned",
    "r7": "C2/debut",
}


def hand_history(seed: int = 0) -> pd.DataFrame:
    """The hand-made card: race_id, race_utc, horse_id, race_class, a trainer and a finish. Rows
    are shuffled on index labels that are not positions; the label of (race, horse) is
    `hand_label`."""
    rng = np.random.default_rng(seed)
    rows = []
    for race, horses in HAND_FIELDS.items():
        days, klass = HAND_RACES[race]
        for k, horse in enumerate(horses):
            rows.append(
                {
                    "race_id": f"jra:{race}",
                    "race_utc": T0 + days * DAY,
                    "horse_id": f"h:{horse}",
                    "race_class": klass,
                    "trainer_key": f"t:{horse}",
                    "finish_pos": float(k + 1),
                }
            )
    frame = pd.DataFrame(rows)
    frame.index = pd.Index(9000 + rng.permutation(len(frame)), name="row")
    return frame.iloc[rng.permutation(len(frame))]


def hand_label(frame: pd.DataFrame, race: str, horse: str):
    """The index label of (race, horse) in `frame`."""
    hit = frame.index[(frame["race_id"] == f"jra:{race}") & (frame["horse_id"] == f"h:{horse}")]
    assert len(hit) == 1
    return hit[0]


def random_history(seed: int, *, horses: int = 120, races: int = 260) -> pd.DataFrame:
    """Runs on a random card: race c at T0 + c // 3 days plus 30 minutes x (c % 3) (so slots tie
    nowhere inside a race but several races share a day), 5 to 10 distinct horses a race,
    classes drawn, a few missing. Rows shuffled."""
    rng = np.random.default_rng(seed)
    classes = np.array(["DEBUT", "MAIDEN", "C1", "C2", "OP", "G3", "G1"], dtype=object)
    rows = []
    for c in range(races):
        when = T0 + (c // 3) * DAY + pd.Timedelta(minutes=30 * (c % 3))
        klass = classes[rng.integers(0, len(classes))]
        if rng.random() < 0.05:
            klass = None
        for h in rng.choice(horses, size=int(rng.integers(5, 11)), replace=False):
            rows.append((f"jra:q{c:04d}", when, f"h:{h:03d}", klass))
    frame = pd.DataFrame(rows, columns=["race_id", "race_utc", "horse_id", "race_class"])
    frame["race_class"] = frame["race_class"].astype(object)
    return frame.iloc[rng.permutation(len(frame))]


def brute_debut(frame: pd.DataFrame) -> pd.Series:
    """True where no other row of the same horse has a strictly earlier race_utc, by a scan of
    every pair (never by a groupby the code under test could share)."""
    out = []
    horses = frame["horse_id"].tolist()
    times = frame["race_utc"].tolist()
    for h, t in zip(horses, times, strict=True):
        out.append(not any(h == h2 and t2 < t for h2, t2 in zip(horses, times, strict=True)))
    return pd.Series(out, index=frame.index, dtype=bool)


def brute_segment(frame: pd.DataFrame, debut: pd.Series) -> pd.Series:
    """`<class>/<debut|seasoned>` per row: the race's class (`unknown` when missing), then
    `debut` when any starter of the race debuts."""
    any_debut = debut.groupby(frame["race_id"]).transform("any")
    klass = frame["race_class"].map(lambda v: v if isinstance(v, str) else "unknown")
    return klass + "/" + any_debut.map({True: "debut", False: "seasoned"})


# ---------------------------------------------------------------------------------------------
# Scored rows: out-of-fold probabilities of the trainer_debut member and the base at the final price
# ---------------------------------------------------------------------------------------------

SEGMENT_PLAN = {  # segment -> (races, planted log-odds lift of the member on the winner)
    "DEBUT/debut": (60, 0.30),
    "MAIDEN/debut": (45, 0.10),
    "MAIDEN/seasoned": (30, -0.15),
    "C1/seasoned": (25, 0.0),
    "G1/seasoned": (8, 0.05),
}
RACES_PER_MEETING = 3


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def scored_rows(seed: int, *, paid_factor: float = 1.0) -> pd.DataFrame:
    """One row per runner: race_id, race_utc, horse_id, segment, meeting_id, won, win_odds, paid,
    p_member, p_base. p_base is a softmax of random logits; the winner is drawn from it; the
    member's logits add `lift` to the winner and noise to all, then are renormalised. `paid` is
    the return per unit stake at the recorded dividend: paid_factor x the odds on the winner, 0
    on a loser."""
    rng = np.random.default_rng(seed)
    rows = []
    race_no = 0
    for segment, (n_races, lift) in SEGMENT_PLAN.items():
        for _ in range(n_races):
            n = int(rng.integers(6, 13))
            base = softmax(rng.normal(0.0, 0.8, size=n))
            win = int(rng.choice(n, p=base))
            logits = np.log(base) + rng.normal(0.0, 0.15, size=n)
            logits[win] += lift
            member = softmax(logits)
            odds = np.round(0.8 / base * rng.uniform(0.9, 1.1, size=n), 1) + 1.0
            meeting = f"m:{race_no // RACES_PER_MEETING:03d}"
            when = T0 + (race_no // 12) * DAY + pd.Timedelta(minutes=30 * (race_no % 12))
            for k in range(n):
                rows.append(
                    {
                        "race_id": f"jra:s{race_no:04d}",
                        "race_utc": when,
                        "horse_id": f"h:{race_no:04d}{k:02d}",
                        "segment": segment,
                        "meeting_id": meeting,
                        "won": k == win,
                        "win_odds": float(odds[k]),
                        "paid": float(paid_factor * odds[k]) if k == win else 0.0,
                        "p_member": float(member[k]),
                        "p_base": float(base[k]),
                    }
                )
            race_no += 1
    frame = pd.DataFrame(rows)
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
