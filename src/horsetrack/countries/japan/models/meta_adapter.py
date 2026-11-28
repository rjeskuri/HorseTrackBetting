"""A C4 meta fit behind the C6 decider's ``meta[r].predict(q, view)`` (#618).

``MetaAdapter(fit, members)``: ``fit`` a ``models.meta.MetaFit``, ``members`` a frame of one row
per runner with ``race_id``, ``horse_no`` (the saddle number a Win combination names), the fit's
member columns and its segment columns; other columns and other races are allowed.
``predict(q, view)`` is P(win) of the runners on ``view.pools["WIN"]``, one value per Win
combination in the board's order, from ``fit.predict`` on those runners' rows with ``q`` (one Win
draw's share row, in the board's order). The rows are those of ``view.race_id`` only, matched by
horse number, never by row order; a runner of the race not on the Win board (scratched) is left
out, so the values sum to 1 over the board. A Win-board runner without a row, or with two, is a
ValueError.

The exotic side needs no adapter: ``models.exotic.Ordering.trifecta(p_win)`` already is the
decider's ``exotic[r].trifecta(p_win)``, indexed as the adapter's output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from horsetrack.countries.japan.models.decision import WIN, _horses
from horsetrack.countries.japan.models.meta import MetaFit

if TYPE_CHECKING:
    from horsetrack.countries.japan.models.nowcast import RaceView

KEYS = ("race_id", "horse_no")


class MetaAdapter:
    """One C4 refit as the decider's ``meta[r]``: ``predict(q, view)`` gives P(win) of the Win
    board's runners in board order (see the module doc)."""

    def __init__(self, fit: MetaFit, members: pd.DataFrame) -> None:
        if not isinstance(fit, MetaFit):
            raise TypeError(f"fit must be a MetaFit, got {type(fit).__name__}")
        needed = [*KEYS, *fit.spec.members, *fit.spec.segment_cols]
        missing = [c for c in needed if c not in members.columns]
        if missing:
            raise KeyError(f"the members frame has no column(s) {missing}")
        self.fit = fit
        self._members = members[needed].reset_index(drop=True)
        self._boards: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}

    def _rows(self, race_id: str, combinations: tuple[str, ...]) -> pd.DataFrame:
        """The members' rows of ``race_id``'s Win-board runners, in board order."""
        cached = self._boards.get((race_id, combinations))
        if cached is not None:
            return cached
        race = self._members[self._members["race_id"] == race_id]
        if race.empty:
            raise ValueError(f"the members frame has no row of {race_id}")
        numbers = race["horse_no"].to_numpy()
        order = []
        for label in combinations:
            horses = _horses(label)
            if len(horses) != 1:
                raise ValueError(f"{race_id}: Win combination {label!r} names one horse")
            at = np.flatnonzero(numbers == horses[0])
            if len(at) != 1:
                raise ValueError(f"{race_id}: horse {label} has {len(at)} member rows, not one")
            order.append(at[0])
        rows = race.iloc[order].reset_index(drop=True)
        self._boards[(race_id, combinations)] = rows
        return rows

    def predict(self, q: np.ndarray, view: RaceView) -> np.ndarray:
        """P(win) of ``view``'s Win-board runners in board order from the Win draw ``q``."""
        if WIN not in view.pools:
            raise ValueError(f"{view.race_id}: no Win board to price")
        rows = self._rows(view.race_id, tuple(view.pools[WIN].combinations))
        q = np.asarray(q, dtype=float)
        if q.shape != (len(rows),):
            raise ValueError(f"q has shape {q.shape}, the Win board {len(rows)} runners")
        return self.fit.predict(rows, q)
