"""Synthetic race days with a planted day variance at one level, for the day-variance tests (#224).

A level's score in a race is #138's: a finisher in place `pos` among `nf` finishers scores
(nf - pos) / (nf - 1) - 0.5. Every runner here finishes (nf = the field), its draw is its
horse_no, and its first-corner position is a random permutation of the field, independent of the
draw. So the draw thirds and the running styles are #129's thirds rules over the field.

Each (day, venue, surface) group gets a planted day effect `u`, in score units, at one target
level: the expected mean score of the target level's runners in every race of the group is
exactly `u`. The construction places the target level's c runners on a contiguous block of
places whose start k is randomised between two neighbours, so the block's mean score
(nf - (k + (c - 1) / 2)) / (nf - 1) - 0.5, linear in k, has exactly the wanted expectation; the
other runners take the remaining places in random order. Hence, given `u`:

- the target level's expected race mean is `u`;
- any other level of the target's factor has expected race mean -c u / (nf - c) (the scores of a
  race sum to 0 and the rest is a random order);
- every level of the other factor has expected race mean 0 (independent of the places).

Modes of a race (the race-to-race noise of the target's race mean around `u`):

- "pinned": always the block of mean u; the race mean barely varies (randomised rounding only),
  far below a random order's variance.
- "mixed": half the races are a random order (mean 0), half the block of mean 2u.
- "extreme": with probability |u| / M the block at the top (u > 0) or bottom (M the top block's
  mean score), otherwise a random order; above a random order's variance.

Synthetic only: no page, no JRDB input.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from datetime import date

from handicap_store import R, Store

VENUES = ("TOK", "NAK")
SURFACES = ("turf", "dirt")
CELL = dict(distance=1600, going="GOOD", course="A")


def draw_group(draw: int, n: int) -> str:
    if 3 * (draw - 1) <= n - 1:
        return "inner"
    if 3 * (draw - 1) <= 2 * (n - 1):
        return "middle"
    return "outer"


def style_of(p: int, n: int) -> str:
    if p == 1:
        return "front"
    if 3 * (p - 1) <= n - 1:
        return "stalker"
    if 3 * (p - 1) <= 2 * (n - 1):
        return "midpack"
    return "closer"


def block_start(n: int, c: int, mean: float) -> float:
    """The (fractional) first place of a block of c places with that mean score."""
    return n - (n - 1) * (mean + 0.5) - (c - 1) / 2


def top_mean(n: int, c: int) -> float:
    return 0.5 - (c - 1) / (2 * (n - 1))


def planted_race(
    s: Store,
    rng: random.Random,
    day: date,
    *,
    venue: str,
    surface: str,
    field: int,
    target: tuple[str, str],
    u: float,
    mode: str,
) -> str:
    n = field
    corners = list(range(1, n + 1))
    rng.shuffle(corners)
    factor, level = target

    def level_of(i: int) -> str:
        return draw_group(i + 1, n) if factor == "draw" else style_of(corners[i], n)

    chosen = [i for i in range(n) if level_of(i) == level]
    rest = [i for i in range(n) if level_of(i) != level]
    c = len(chosen)
    assert c >= 1, f"fixture: no {target} runner in a field of {n}"

    block = None  # the target block's wanted mean score; None: a random order
    if mode == "pinned":
        block = u
    elif mode == "mixed":
        block = 2 * u if rng.random() < 0.5 else None
    elif mode == "extreme":
        m = top_mean(n, c)
        assert abs(u) <= m, f"fixture: |u| {u} above the top block's mean {m}"
        if rng.random() < abs(u) / m:
            block = math.copysign(m, u)
    else:
        raise AssertionError(f"fixture: unknown mode {mode}")

    places = list(range(1, n + 1))
    if block is None:
        rng.shuffle(places)
        pos = dict(zip(range(n), places, strict=True))
    else:
        k_star = block_start(n, c, block)
        assert 1 - 1e-9 <= k_star <= n - c + 1 + 1e-9, f"fixture: block {block} does not fit"
        k = math.floor(k_star + 1e-12)
        if rng.random() < k_star - k:
            k += 1
        k = min(max(k, 1), n - c + 1)
        mine = list(range(k, k + c))
        others = [p for p in places if p not in mine]
        rng.shuffle(mine)
        rng.shuffle(others)
        pos = dict(zip(chosen, mine, strict=True)) | dict(zip(rest, others, strict=True))
    base = len(s.runners)
    runners = [
        R(f"h:{base + i}", pos=pos[i], corners=[corners[i], corners[i]], no=i + 1) for i in range(n)
    ]
    return s.race(day, runners, venue=venue, surface=surface, distance=CELL["distance"],
                  going=CELL["going"], course=CELL["course"])  # fmt: skip


Group = tuple[date, str, str]  # (day, venue, surface)


def planted_history(
    seed: int,
    days: list[date],
    *,
    target: tuple[str, str],
    sd: float,
    races: int | Callable[[int, random.Random], int] = 6,
    field: int | Callable[[random.Random], int] = 12,
    mode: str | Callable[[int], str] = "mixed",
    store: Store | None = None,
) -> tuple[Store, dict[Group, float], dict[Group, int]]:
    """Every day of `days` at both venues and both surfaces: one group each, with a day effect
    u ~ Uniform(-sd sqrt 3, sd sqrt 3) (sd `sd`, bounded so every block fits) at `target`.
    `races`, `field` and `mode` may vary by group index g (and the rng). Returns the store, the
    planted u per group and the number of races per group."""
    rng = random.Random(seed)
    s = store if store is not None else Store()
    planted: dict[Group, float] = {}
    sizes: dict[Group, int] = {}
    half = sd * math.sqrt(3)
    g = 0
    for day in days:
        for venue in VENUES:
            for surface in SURFACES:
                u = rng.uniform(-half, half) if sd > 0 else 0.0
                r = races(g, rng) if callable(races) else races
                m = mode(g) if callable(mode) else mode
                for _ in range(r):
                    n = field(rng) if callable(field) else field
                    planted_race(s, rng, day, venue=venue, surface=surface, field=n,
                                 target=target, u=u, mode=m)  # fmt: skip
                planted[(day, venue, surface)] = u
                sizes[(day, venue, surface)] = r
                g += 1
    return s, planted, sizes


def realized_sd(planted: dict[Group, float], scale: float = 1.0) -> float:
    """The planted day effects' sd around each (venue, surface)'s own mean (the prior absorbs
    a course's mean), times `scale`; groups weighted equally."""
    by_course: dict[tuple[str, str], list[float]] = {}
    for (_, venue, surface), u in planted.items():
        by_course.setdefault((venue, surface), []).append(u)
    sq = [(u - sum(us) / len(us)) ** 2 for us in by_course.values() for u in us]
    return scale * math.sqrt(sum(sq) / len(sq))
