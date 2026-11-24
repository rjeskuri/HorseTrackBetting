"""A synthetic JRA store for the handicap-factor tests (#129).

`Store` holds store-shaped `races` and `runners` rows built race by race from explicit runners
(`R`), so a test can state each value it later checks by hand. `random_store` builds a varied,
deterministic history for the whole-table tests (leakage, flags, the per-day writer). `write`
puts both tables under tmp_path and points HORSETRACK_DATA_DIR there; the code under test reads
them back through #114's scoped reader, as tests/countries/japan/handicap/test_pp_lines.py does.

Synthetic data only: no page, no JRDB input.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from horsetrack.core.eval import holdout
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.models.v0 import store as v0_store

DATA_VAR = "HORSETRACK_DATA_DIR"
JST = ZoneInfo("Asia/Tokyo")
T = date(2025, 6, 15)  # the race day of most target races (a Sunday)


def rid(day: date, venue: str, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{race_no:02d}"


def at_jst(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), tzinfo=JST)


def weekend_days(first: date, last: date) -> list[date]:
    """Every Saturday and Sunday from `first` to `last`, inclusive."""
    days, d = [], first
    while d <= last:
        if d.weekday() in (5, 6):
            days.append(d)
        d += timedelta(days=1)
    return days


@dataclass
class R:
    """One runner of a race. pos None: did not finish (a start, not a scratch)."""

    horse: str
    pos: int | None = None
    time: float | None = None  # default: winner time + 0.2 s per place behind
    l3f: float | None = None  # default 35.0 for a finisher
    corners: list[int] | None = None  # default [pos, pos] for a finisher, [] otherwise
    jockey: str | None = None  # default "j:<horse>"
    trainer: str | None = None  # default "t:<horse>"
    carried: float | None = 55.0
    body: int | None = 480
    body_chg: int | None = 0
    scratched: bool = False
    no: int | None = None  # horse_no and draw; default: the next free number


class Store:
    def __init__(self):
        self.races: list[dict] = []
        self.runners: list[dict] = []
        self.races_by_day: dict[date, list[str]] = {}
        self._next_no: dict[tuple[date, str], int] = {}

    def race(
        self,
        day: date,
        runners: list[R],
        *,
        venue: str = "TOK",
        race_no: int | None = None,
        surface: str = "turf",
        distance: int = 1600,
        going: str | None = "GOOD",
        cls: str | None = "C2",
        course: str | None = "outer",
        winner_time: float | None = None,
        fill_to: int | None = None,
    ) -> str:
        """Add a run race. `fill_to` adds filler runners until that many are not scratched,
        taking the finishing places the named runners leave free."""
        if race_no is None:
            race_no = self._next_no.get((day, venue), 1)
        self._next_no[(day, venue)] = race_no + 1
        race_id = rid(day, venue, race_no)
        if any(r["race_id"] == race_id for r in self.races):
            raise AssertionError(f"fixture bug: duplicate race {race_id}")

        runners = [copy.copy(r) for r in runners]
        if fill_to is not None:
            taken = {r.pos for r in runners if r.pos is not None}
            free = [p for p in range(1, fill_to + 1) if p not in taken]
            starters = sum(not r.scratched for r in runners)
            for k, p in enumerate(free[: max(fill_to - starters, 0)]):
                runners.append(R(f"f:{race_id}:{k}", pos=p))
        if winner_time is None:
            timed = [r.time for r in runners if r.pos == 1 and r.time is not None]
            winner_time = timed[0] if timed else 95.0

        used = {r.no for r in runners if r.no is not None}
        nxt = 1
        for r in runners:
            if r.no is None:
                while nxt in used:
                    nxt += 1
                r.no = nxt
                used.add(nxt)
        field = sum(not r.scratched for r in runners)

        self.races_by_day.setdefault(day, []).append(race_id)
        self.races.append(
            {
                "jurisdiction": "jra", "race_id": race_id, "meeting_id": f"jra:{day}:{venue}",
                "race_date": day, "venue": venue, "race_no": race_no, "status": "run",
                "post_time_local": f"{9 + race_no // 2:02d}:{(race_no * 25) % 60:02d}",
                "surface": surface, "distance_m": distance, "course_config": course,
                "going": going, "race_class": cls, "field_size": field,
                "finish_time_s": winner_time,
            }
        )  # fmt: skip
        for r in runners:
            finished = r.pos is not None and not r.scratched
            if finished:
                t = r.time if r.time is not None else round(winner_time + 0.2 * (r.pos - 1), 2)
                l3f = r.l3f if r.l3f is not None else 35.0
                corners = r.corners if r.corners is not None else [r.pos, r.pos]
            else:
                t, l3f = None, None
                corners = [] if r.corners is None or r.scratched else r.corners
            self.runners.append(
                {
                    "jurisdiction": "jra", "race_id": race_id, "meeting_id": f"jra:{day}:{venue}",
                    "race_date": day, "venue": venue, "race_no": race_no, "dead_heat": False,
                    "horse_id": r.horse, "horse_no": r.no, "draw": r.no,
                    "horse_name": r.horse.upper(), "bracket": 1 + (r.no - 1) // 2,
                    "jockey_key": r.jockey or f"j:{r.horse}", "jockey_name": "J",
                    "trainer_key": r.trainer or f"t:{r.horse}", "trainer_name": "T",
                    "actual_wt_kg": r.carried, "horse_wt_kg": r.body,
                    "horse_wt_change_kg": r.body_chg,
                    "finish_pos": r.pos if finished else None,
                    "placing_raw": str(r.pos) if finished else ("取消" if r.scratched else "中止"),
                    "lbw_lengths": (0.5 * (r.pos - 1)) if finished else None,
                    "finish_time_s": t, "last_3f_s": l3f, "running_positions": corners,
                    "win_odds": (2.0 + r.no) if not r.scratched else None,
                    "win_fav_rank": r.no if not r.scratched else None,
                    "is_scratched": r.scratched,
                }
            )  # fmt: skip
        return race_id

    def without_day(self, day: date) -> Store:
        other = copy.deepcopy(self)
        other.races = [r for r in other.races if r["race_date"] != day]
        other.runners = [r for r in other.runners if r["race_date"] != day]
        other.races_by_day.pop(day, None)
        return other

    def won(self) -> dict[tuple[str, int], bool]:
        """(race_id, horse_no) -> won, for every declared runner."""
        return {
            (r["race_id"], r["horse_no"]): r["finish_pos"] == 1
            for r in self.runners
            if not r["is_scratched"]
        }

    def write(self, tmp_path: Path, monkeypatch, sub: str = "data") -> Path:
        data = (tmp_path / sub).resolve()
        monkeypatch.setenv(DATA_VAR, str(data))
        jra = data / "processed" / "jra"
        jra.mkdir(parents=True)
        for name, rows in (("races", self.races), ("runners", self.runners)):
            spec = schemas.TABLES[name].columns
            schema = pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in spec.items()])
            shuffled = list(rows)
            random.Random(7).shuffle(shuffled)  # nothing may rely on the file's row order
            pq.write_table(pa.Table.from_pylist(shuffled, schema=schema), jra / f"{name}.parquet")
        return data


def patch_holdout(monkeypatch, days, *modules):
    """Mark exactly `days` holdout wherever the reader (or the code under test) looks it up."""
    marked = set(days)

    def fake(country, day):
        return day in marked

    monkeypatch.setattr(holdout, "is_holdout_day", fake)
    monkeypatch.setattr(v0_store, "is_holdout_day", fake, raising=False)
    for module in modules:
        monkeypatch.setattr(module, "is_holdout_day", fake, raising=False)


# ---------------------------------------------------------------------------------------------
# A varied, deterministic history
# ---------------------------------------------------------------------------------------------

VENUES = ("TOK", "NAK")
SURFACES = ("turf", "dirt")
DISTANCES = (1400, 1600, 2000)
CLASSES = ("C1", "C2", "C3")
GOINGS = ("GOOD", "GOOD TO YIELDING", "SOFT", "HEAVY")
COURSES = ("inner", "outer", None)


def random_race(s: Store, rng: random.Random, day: date, horses: list[str], **race) -> str:
    """A race of `horses` with random results (some did not finish, some scratched)."""
    status = []
    for _ in horses:
        u = rng.random()
        status.append("scr" if u < 0.05 else "dnf" if u < 0.10 else "run")
    if "run" not in status:
        status[0] = "run"
    finishers = [i for i, st in enumerate(status) if st == "run"]
    places = list(range(1, len(finishers) + 1))
    rng.shuffle(places)
    place_of = dict(zip(finishers, places, strict=True))
    distance = race.get("distance", 1600)
    winner = round(distance * 0.06 + rng.uniform(-0.8, 0.8), 1)
    field = len(horses) - status.count("scr")
    n_corners = 2 if distance <= 1400 else 4
    runners = []
    for i, h in enumerate(horses):
        pos = place_of.get(i)
        first = rng.randint(1, field)
        corners = [first] + [rng.randint(1, field) for _ in range(n_corners - 1)]
        runners.append(
            R(
                h, pos=pos,
                time=None if pos is None else round(winner + 0.15 * (pos - 1) + 0.05 * (i % 3), 2),
                l3f=round(rng.uniform(33.5, 37.5), 1) if pos is not None else None,
                corners=corners if status[i] != "scr" else [],
                jockey=f"jockey:{rng.randrange(6)}", trainer=f"trainer:{rng.randrange(4)}",
                carried=rng.choice([54.0, 55.0, 56.0, 57.0]), body=rng.randint(440, 520),
                body_chg=rng.choice([-6, -4, -2, 0, 2, 4, 6, None]),
                scratched=status[i] == "scr",
            )
        )  # fmt: skip
    return s.race(day, runners, winner_time=winner, **race)


def random_store(
    days: list[date],
    *,
    seed: int = 11,
    races_per_day: int = 2,
    field: int = 8,
    pool: int = 24,
    target: str | None = None,
) -> Store:
    """Races on every day of `days` from a pool of horses, jockeys and trainers. With `target`
    (a race id on one of `days`), that race is added too, at TOK turf 1600 C2 outer GOOD, with
    seven pool horses, the debutant h:new and the scratched h:out."""
    rng = random.Random(seed)
    s = Store()
    horses = [f"h:{i:02d}" for i in range(pool)]
    for day in days:
        for k in range(races_per_day):
            random_race(
                s, rng, day, rng.sample(horses, field), venue=VENUES[k % 2], race_no=1 + k,
                surface=rng.choice(SURFACES), distance=rng.choice(DISTANCES),
                cls=rng.choice(CLASSES), going=rng.choice(GOINGS), course=rng.choice(COURSES),
            )  # fmt: skip
    if target is not None:
        _, day_s, venue, no = target.split(":")
        horses_t = rng.sample(horses, 7) + ["h:new"]
        random_race(
            s, rng, date.fromisoformat(day_s), horses_t, venue=venue, race_no=int(no),
            surface="turf", distance=1600, cls="C2", going="GOOD", course="outer",
        )  # fmt: skip
        # h:out is scratched from the target race itself
        target_rows = [r for r in s.runners if r["race_id"] == target]
        out = copy.deepcopy(target_rows[0])
        out.update(
            horse_id="h:out", horse_no=max(r["horse_no"] for r in target_rows) + 1,
            finish_pos=None, placing_raw="取消", finish_time_s=None, last_3f_s=None,
            running_positions=[], win_odds=None, win_fav_rank=None, is_scratched=True,
        )  # fmt: skip
        out["draw"] = out["horse_no"]
        s.runners.append(out)
    return s


# The card of a race: what is known before it runs. A leakage perturbation keeps these.
CARD_RUNNER = ("horse_id", "horse_no", "draw", "bracket", "jockey_key", "jockey_name",
               "trainer_key", "trainer_name", "actual_wt_kg", "horse_wt_kg",
               "horse_wt_change_kg", "is_scratched")  # fmt: skip
CARD_RACE = ("venue", "surface", "distance_m", "course_config", "going", "race_class",
             "field_size", "race_no", "post_time_local")  # fmt: skip


def perturb(s: Store, pick, *, keep_card: str | None = None) -> Store:
    """A copy of `s` whose results on the picked race days are changed: finishing order reversed,
    times, last 600 m, corner positions, odds and (outside the `keep_card` race) the race's
    conditions and its connections, draws and weights. A race-day predicate `pick(day)`."""
    other = copy.deepcopy(s)
    by_race: dict[str, list[dict]] = {}
    for r in other.runners:
        if pick(r["race_date"]):
            by_race.setdefault(r["race_id"], []).append(r)
    winner_time = {}
    for race_id, rows in by_race.items():
        finishers = sorted(
            (r for r in rows if r["finish_pos"] is not None), key=lambda r: r["finish_pos"]
        )
        m = len(finishers)
        for r in finishers:
            new = m + 1 - r["finish_pos"]
            r["finish_pos"] = new
            r["placing_raw"] = str(new)
            r["finish_time_s"] = round(r["finish_time_s"] + 3.7 + 0.5 * new, 2)
            r["last_3f_s"] = round((r["last_3f_s"] or 35.0) - 1.3 - 0.1 * new, 2)
            r["lbw_lengths"] = 0.5 * (new - 1) + 0.25
            r["running_positions"] = [max(1, m + 1 - p) for p in r["running_positions"]][::-1]
            r["win_odds"] = (r["win_odds"] or 3.0) * 2.5
            r["win_fav_rank"] = m + 1 - (r["win_fav_rank"] or 1)
            if new == 1:
                winner_time[race_id] = r["finish_time_s"]
        if race_id != keep_card:
            for r in rows:
                r["jockey_key"] = "jockey:leak"
                r["trainer_key"] = "trainer:leak"
                r["actual_wt_kg"] = 60.0
                r["horse_wt_change_kg"] = 12
    for race in other.races:
        if race["race_id"] in by_race:
            race["finish_time_s"] = winner_time.get(race["race_id"], race["finish_time_s"])
            if race["race_id"] != keep_card:
                race["going"] = "HEAVY" if race["going"] != "HEAVY" else "GOOD"
                race["race_class"] = "G1"
    return other
