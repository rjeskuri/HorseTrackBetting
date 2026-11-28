"""The horse member of the shadow page (ticket #717): #438's debut/maiden combined member, fitted
for one segment on development days before the fit day.

The model is #438's (`studies/debut_members.py`): a conditional logit with the final-price
offset, u = ln q + sum w x over the standardised member columns (`Design`, `fit_offset`), so
w = 0 is the market exactly; `predict_offset` gives P(win) within a race. `fit_member` fits it on
the rows of one race class, DEBUT or MAIDEN (`debut_members.CLASSES`), of the development days
from #438's price era (`select_races`: holdout days are never read) strictly before `today`.
The fit keeps #438's design (each column's training mean and sd, the missing flags, the kept
columns) and its weights; `MemberFit.predict` rebuilds the design and calls `predict_offset`.

Display only: the shadow feed shows the member's P(win) beside the market's (`p_member`), and
nothing reads it into a value, a flag, `p_model` or the shadow log. The real-data refit command
is not here; a fit file is named by the v0 pointer's "member" entry (models/v0/loader.py).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from horsetrack.countries.japan.studies import debut_members as dm

MODEL = "debut_maiden_member_v0"
SEGMENTS = tuple(dm.CLASSES)


@dataclass(frozen=True)
class MemberFit:
    """#438's offset model of one segment: the design (`stats`: (feature, mean, sd) per feature
    with a known training value; `flags`: the features with a missing flag; `names`: the kept
    design columns) and one weight per kept column (`coef`)."""

    segment: str
    features: tuple[str, ...]
    stats: tuple[tuple[str, float, float], ...]
    flags: tuple[str, ...]
    names: tuple[str, ...]
    coef: tuple[float, ...]
    train_start: date
    train_end: date
    n_races: int
    fit_id: str

    def design(self) -> dm.Design:
        return dm.Design({f: (mean, sd) for f, mean, sd in self.stats}, self.flags, self.names)

    def predict(self, q: Sequence[float], features: pd.DataFrame) -> np.ndarray:
        """P(win) per runner of one race: `q` the market's normalised Win probabilities, `features`
        the member columns in the same runner order (a column left out, or a NaN, is missing)."""
        q = np.asarray(q, dtype=float)
        if q.ndim != 1 or len(features) != len(q):
            raise ValueError(f"{len(q)} probabilities for {len(features)} runners")
        frame = {
            f: (
                features[f].to_numpy(dtype=float)
                if f in features.columns
                else np.full(len(q), np.nan)
            )
            for f in self.features
        }
        rows = pd.DataFrame({**frame, "race_id": ["race"] * len(q), "ln_q": np.log(q)})
        return dm.predict_offset(self.design(), self, rows).to_numpy(dtype=float)

    def _content(self) -> dict:
        return {
            "model": MODEL,
            "segment": self.segment,
            "features": list(self.features),
            "stats": [[f, mean, sd] for f, mean, sd in self.stats],
            "flags": list(self.flags),
            "names": list(self.names),
            "coef": list(self.coef),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "n_races": self.n_races,
        }

    def to_json(self) -> str:
        return json.dumps({"fit_id": self.fit_id, **self._content()}, indent=2)

    @classmethod
    def from_json(cls, s: str) -> MemberFit:
        data = json.loads(s)
        if not isinstance(data, dict) or data.get("model") != MODEL:
            raise ValueError(f"not a {MODEL} fit")
        return cls(
            segment=str(data["segment"]),
            features=tuple(str(f) for f in data["features"]),
            stats=tuple((str(f), float(mean), float(sd)) for f, mean, sd in data["stats"]),
            flags=tuple(str(f) for f in data["flags"]),
            names=tuple(str(n) for n in data["names"]),
            coef=tuple(float(c) for c in data["coef"]),
            train_start=date.fromisoformat(data["train_start"]),
            train_end=date.fromisoformat(data["train_end"]),
            n_races=int(data["n_races"]),
            fit_id=str(data["fit_id"]),
        )


@dataclass(frozen=True)
class MemberEntry:
    """One segment of the pointer's "member" entry: the fit and the p/q threshold k the
    leaderboard ranks best (a horse with p_member / q >= k is marked)."""

    fit: MemberFit
    k: float


def fit_member(
    rows: pd.DataFrame, *, segment: str, today: date, features: Sequence[str]
) -> MemberFit:
    """#438's offset model on the rows of race class `segment` on development days strictly
    before `today`. `rows` has race_id, race_date, race_class, meeting_id, won, ln_q and the
    `features` columns (#438's `feature_columns` names)."""
    if segment not in SEGMENTS:
        raise ValueError(f"segment {segment!r} is not one of {SEGMENTS}")
    picked = dm.select_races(rows)  # DEBUT and MAIDEN on development days of the price era
    days = pd.to_datetime(picked["race_date"]).dt.date
    keep = (picked["race_class"] == segment).to_numpy(dtype=bool) & (days < today).to_numpy(
        dtype=bool
    )
    train = picked.loc[keep]
    if train.empty:
        raise ValueError(f"no {segment} race on a development day before {today}")
    design, fit = dm.fit_offset(train, list(features))
    train_days = days[keep]
    content = MemberFit(
        segment=segment,
        features=tuple(features),
        stats=tuple((f, float(m), float(s)) for f, (m, s) in design.stats.items()),
        flags=tuple(design.flags),
        names=tuple(design.names),
        coef=tuple(float(c) for c in fit.coef),
        train_start=min(train_days),
        train_end=max(train_days),
        n_races=int(train["race_id"].nunique()),
        fit_id="",
    )
    digest = hashlib.sha256(json.dumps(content._content()).encode()).hexdigest()[:12]
    return dataclasses.replace(content, fit_id=f"member-{segment.lower()}-{digest}")
