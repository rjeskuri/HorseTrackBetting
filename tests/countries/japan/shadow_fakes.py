"""Synthetic race days and fake v0 models for the shadow-feed tests (ticket #121).

`SyntheticDay` writes odds logs (one gzip member per capture) and `<VENUE>.races.json` post-time
sidecars under the data dir, and keeps every row it wrote so a test can rebuild the board at t with
`board_at`. Prices are invented and move with a capture's `factor`.

The fake models are deterministic so expected values can be worked out here by hand:
- `FakeNowcast.draws` returns the board's own figures times (1 + tau_min / 200) on every draw and
  records (race_id, pool, tau_min, capture_utc) of each call: the spy on the bucket used and on
  which races were computed.
- `FakeWin.predict(q) = q**1.3 / sum(q**1.3)`; `FakeOrdering.place_probs(p, k) = min(1, k p)` and
  `pair_probs(p, k)[a, b] = min(1, 3 k p_a p_b)` (zero diagonal).
`expected_values` derives each key's value from the capture's figures, the tau and the places paid
(PLACE: 3 at 8+ starters, 2 at 5-7; Wide: 3 at 4+), never from the code under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.models.v0.shadow_values import PoolDraws, ShadowModels

UTC = timezone.utc
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
IDS = {"win": "win-a1", "ordering": "ord-b2", "nowcast": "now-c3"}
BANNED_KEY_WORDS = ("stake", "bet", "kelly", "amount", "wager")


# ---------------------------------------------------------------- the synthetic day


@dataclass(frozen=True)
class Race:
    day: date
    venue: str
    no: int
    post: datetime
    base: tuple[float, ...]

    @property
    def race_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{self.venue}:{self.no:02d}"


def figures(race: Race, factor: float) -> tuple[dict, dict, dict]:
    """(win, place, wide) of one capture: {h: odds}, {h: (low, high)}, {(a, b): (low, high)}."""
    win = {h: round(o * factor, 2) for h, o in enumerate(race.base, 1)}
    place = {}
    for h, o in win.items():
        low = round(1.0 + (o - 1.0) * 0.22, 2)
        place[h] = (low, round(low * 1.7, 2))
    wide = {}
    for a, b in combinations(sorted(win), 2):
        low = round(1.05 + 0.12 * (win[a] + win[b]), 2)
        wide[(a, b)] = (low, round(low * 1.45, 2))
    return win, place, wide


def capture_rows(race: Race, at: datetime, factor: float) -> list[dict]:
    win, place, wide = figures(race, factor)

    def row(pool: str, combination: str, odds: float, hi: float | None = None) -> dict:
        out = {
            "snapshot_utc": at.astimezone(UTC).isoformat(),
            "fetched_utc": at.astimezone(UTC).isoformat(),
            "race_id": race.race_id,
            "pool": pool,
            "combination": combination,
            "odds": odds,
            "status_raw": "10時00分現在",
            "page_state": None,
            "source_update_utc": (at - timedelta(seconds=10)).astimezone(UTC).isoformat(),
        }
        if hi is not None:
            out["odds_hi"] = hi
        return out

    rows = [row(WIN, str(h), o) for h, o in win.items()]
    rows += [row(PLACE, str(h), lo, hi) for h, (lo, hi) in place.items()]
    rows += [row(WIDE, f"{a},{b}", lo, hi) for (a, b), (lo, hi) in wide.items()]
    return rows


def log_path(day: date, venue: str) -> Path:
    return paths.odds_snapshot_path("jra", day.isoformat(), venue)


@dataclass
class SyntheticDay:
    day: date
    races: list[Race]
    rows: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        for venue in sorted({r.venue for r in self.races}):
            posts = {r.no: r.post for r in self.races if r.venue == venue}
            sidecar = log_path(self.day, venue).with_name(f"{venue}.races.json")
            first = min(posts.values())
            odds_log.write_post_times(sidecar, posts, first - timedelta(hours=3))

    def capture(self, race: Race, at: datetime, factor: float) -> None:
        rows = capture_rows(race, at, factor)
        odds_log.append_capture(log_path(self.day, race.venue), rows)
        self.rows.extend(rows)


# ---------------------------------------------------------------- fake models


def win_p(q: np.ndarray) -> np.ndarray:
    w = np.asarray(q, dtype=float) ** 1.3
    return w / w.sum()


def place_p(p: np.ndarray, places: int) -> np.ndarray:
    return np.minimum(1.0, places * np.asarray(p, dtype=float))


def pair_p(p: np.ndarray, places: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = np.minimum(1.0, 3.0 * places * np.outer(p, p))
    np.fill_diagonal(m, 0.0)
    return m


def tau_factor(tau_min: int) -> float:
    return 1.0 + tau_min / 200.0


class FakeWin:
    def predict(self, q, race=None):
        return win_p(q)


class FakeOrdering:
    def place_probs(self, p, places):
        return place_p(p, places)

    def pair_probs(self, p, places):
        return pair_p(p, places)


class FakeNowcast:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, datetime]] = []

    def draws(self, board, *, pool, tau_min, shocks):
        self.calls.append((board.race_id, pool, tau_min, board.capture_utc))
        n_draws = len(next(iter(shocks.values())))
        f = tau_factor(tau_min)
        if pool == WIN:
            keys = sorted(board.win)
            lows = [board.win[h] for h in keys]
            highs = None
            combos = tuple(str(h) for h in keys)
        elif pool == PLACE:
            keys = sorted(board.place)
            lows = [board.place[h][0] for h in keys]
            highs = [board.place[h][1] for h in keys]
            combos = tuple(str(h) for h in keys)
        else:
            keys = sorted(board.wide)
            lows = [board.wide[k][0] for k in keys]
            highs = [board.wide[k][1] for k in keys]
            combos = tuple(f"{a},{b}" for a, b in keys)
        odds = np.tile(np.asarray(lows, dtype=float) * f, (n_draws, 1))
        hi = None
        if highs is not None:
            hi = np.tile(
                np.asarray([np.nan if h is None else h for h in highs], dtype=float) * f,
                (n_draws, 1),
            )
        return PoolDraws(pool=pool, combinations=combos, odds=odds, odds_hi=hi)

    def races_called(self, since: int = 0) -> set[str]:
        return {race_id for race_id, *_ in self.calls[since:]}

    def taus(self, race_id: str) -> set[int]:
        return {tau for rid, _, tau, _ in self.calls if rid == race_id}


def fake_models() -> ShadowModels:
    return ShadowModels(
        win=FakeWin(), ordering=FakeOrdering(), nowcast=FakeNowcast(), ids=dict(IDS)
    )


# ---------------------------------------------------------------- expected values


def _places_place(starters: int) -> int:
    return 3 if starters >= 8 else 2 if starters >= 5 else 0


def _places_wide(starters: int) -> int:
    return 3 if starters >= 4 else 0


def expected_values(race: Race, factor: float, tau: int) -> dict[str, dict[str, dict]]:
    """{pool: {combination: {value field: value}}} for the capture of `race` at `factor`, read at
    bucket `tau` by the fake models."""
    win, place, wide = figures(race, factor)
    runners = sorted(win)
    f = tau_factor(tau)
    odds = np.array([win[h] for h in runners]) * f
    inv = 1.0 / odds
    p = win_p(inv / inv.sum())
    pos = {h: j for j, h in enumerate(runners)}
    out: dict[str, dict[str, dict]] = {WIN: {}, PLACE: {}, WIDE: {}}
    for h in runners:
        v = float(p[pos[h]] * odds[pos[h]])
        out[WIN][str(h)] = {"value_mean": v, "value_q05": v, "value_q95": v}
    pp = place_p(p, _places_place(len(runners)))
    for h, (lo, hi) in place.items():
        out[PLACE][str(h)] = {
            "value_at_low": float(pp[pos[h]] * lo * f),
            "value_at_high": float(pp[pos[h]] * hi * f),
        }
    pw = pair_p(p, _places_wide(len(runners)))
    for (a, b), (lo, hi) in wide.items():
        out[WIDE][f"{a},{b}"] = {
            "value_at_low": float(pw[pos[a], pos[b]] * lo * f),
            "value_at_high": float(pw[pos[a], pos[b]] * hi * f),
        }
    return out


def instant(text: str) -> datetime:
    return datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)


def all_keys(obj) -> list[str]:
    """Every dict key anywhere in a JSON-like object, nested lists included."""
    out: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.append(str(key))
            out.extend(all_keys(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(all_keys(value))
    return out
