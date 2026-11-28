"""The sealed ladder log of a Japan race day (#628), written by shadow-day beside the shadow log.

``data_dir()/shadow/japan/<YYYY-MM-DD>.ladder.jsonl`` keeps, at each ladder checkpoint k in
(20, 10, 5, 2) of each race, the odds observed at post - k with the shadow page's projection of
every later checkpoint and the final (`board.shadow_feed._projector`), and after the post each
key's final odds, so the nowcast can be scored on the day's file alone. The file has the shadow
log's hash chain, seal and ``day_status`` (`shadow.log.ChainedLog`); its records are keyed by
(race_id, pool, combination, checkpoint_min). Display data only: nothing here feeds a value, a
flag or the shadow log. The format is in docs/specs/shadow-log.md, "The ladder log".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.board.shadow_feed import LADDER_MINUTES, _projector
from horsetrack.countries.japan.board.snapshot import PLACE, WIDE, WIN, RaceBoard, final_board
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels
from horsetrack.countries.japan.shadow.log import (
    MODES,
    NO_BOARD,
    STATUSES,
    ChainedLog,
    _check_day,
    _instant,
    _number,
    _stake_like_key,
    shadow_log_path,
)

CHECKPOINTS = LADDER_MINUTES  # (20, 10, 5, 2): minutes before the post
FINAL = 0  # checkpoint_min of a final record
ANY = "*"  # pool and combination of a no_board record: it stands for the whole checkpoint
KEY_FIELDS = ("race_id", "pool", "combination", "checkpoint_min")
REQUIRED = (
    *KEY_FIELDS,
    "status",
    "odds",
    "fetched_utc",
    "board_age_s",
    "projected",
    "nowcast_id",
    "mode",
    "post_utc",
    "written_utc",
    "git_sha",
)


def ladder_log_path(day: date) -> Path:
    """``data_dir()/shadow/japan/<YYYY-MM-DD>.ladder.jsonl``, beside the day's shadow log."""
    _check_day(day)
    return shadow_log_path(day).with_name(f"{day.isoformat()}.ladder.jsonl")


def _refusal(record: Any) -> str | None:
    """Why a ladder record is refused, or None."""
    if not isinstance(record, dict):
        return "is not a dict"
    if "day_status" in record:
        return "sets day_status, which the log sets"
    missing = [name for name in REQUIRED if name not in record]
    if missing:
        return f"lacks {', '.join(missing)}"
    for name in ("race_id", "pool", "combination", "git_sha"):
        if not isinstance(record[name], str) or not record[name]:
            return f"has a {name} that is not non-empty text"
    if type(record["checkpoint_min"]) is not int or record["checkpoint_min"] not in (
        *CHECKPOINTS,
        FINAL,
    ):
        return f"has a checkpoint_min other than {', '.join(map(str, (*CHECKPOINTS, FINAL)))}"
    if record["mode"] not in MODES:
        return f"has a mode other than {' or '.join(MODES)}"
    if record["status"] not in STATUSES:
        return f"has a status other than {' or '.join(STATUSES)}"
    odds = record["odds"]
    if record["status"] == NO_BOARD:
        if odds is not None or record["projected"]:
            return "is a no_board record with odds or projections"
    elif not (_number(odds) and odds > 0):
        return "has odds that are not a number above 0"
    projected = record["projected"]
    if not isinstance(projected, dict) or any(
        not isinstance(j, str) or not (v is None or _number(v)) for j, v in projected.items()
    ):
        return "has projections that are not {checkpoint: odds or null}"
    for name in ("post_utc", "written_utc"):
        if _instant(record[name]) is None:
            return f"has a {name} that is not ISO text with a UTC offset"
    if record["fetched_utc"] is not None and _instant(record["fetched_utc"]) is None:
        return "has a fetched_utc that is neither null nor ISO text with a UTC offset"
    if record["nowcast_id"] is not None and not isinstance(record["nowcast_id"], str):
        return "has a nowcast_id that is neither text nor null"
    name = _stake_like_key(record)
    if name is not None:
        return f"has the stake-like key {name!r}"
    return None


class LadderLog(ChainedLog):
    """One race day's ladder log: the shadow log's chain, seal and day_status, keyed by
    (race_id, pool, combination, checkpoint_min). `read()` raises HoldoutSealed on a holdout day."""

    SUFFIX = ".ladder.jsonl"
    KEY_FIELDS = KEY_FIELDS

    def _refusal(self, record: Any) -> str | None:
        return _refusal(record)


def checkpoint_age_limits(limits: Mapping[int, float]) -> dict[int, float]:
    """Each checkpoint's board-age limit in seconds from shadow-day's per-tau limits: the tau's own
    limit when it has one, else linear in tau between the nearest taus below and above it (the
    nearest one's beyond either end)."""
    taus = sorted(limits)
    out = {}
    for k in CHECKPOINTS:
        below, above = [t for t in taus if t < k], [t for t in taus if t > k]
        if k in limits:
            out[k] = float(limits[k])
        elif not below or not above:
            out[k] = float(limits[above[0] if above else below[-1]])
        else:
            lo, hi = below[-1], above[0]
            out[k] = limits[lo] + (k - lo) / (hi - lo) * (limits[hi] - limits[lo])
    return out


# ---------------------------------------------------------------- what shadow-day keeps per race

Ref = tuple[datetime, int, dict[str, Any]]  # (snapshot_utc, seq, row)


def _offer(slot: dict[tuple, Ref], key: tuple, entry: Ref) -> None:
    """Keep the latest row per key by capture time; at the same instant the later row wins."""
    current = slot.get(key)
    if current is None or entry[:2] >= current[:2]:
        slot[key] = entry


class _RaceLadder:
    """One race's undecided checkpoints, each with the latest row per board key captured at or
    before post - k (`shadow_feed._DayRows.refs`' rule), the latest row per key overall (to re-time
    a moved post) and, until its final is written, the rows of its captures showing a final."""

    def __init__(self, race_id: str) -> None:
        self.race_id = race_id
        self.post: datetime | None = None
        self.slots: dict[int, dict[tuple, Ref]] = {}
        self.newest: dict[tuple, Ref] = {}
        self.done: set[int] = set()
        self.shown: set[str] = set()  # the pools on its checkpoint records
        self.finals: set[tuple[str, str]] = set()  # the (pool, combination) of its final records
        self.final_snaps: set[datetime] = set()
        self.final_rows: list[dict[str, Any]] = []
        self.final_new = False  # final rows arrived since its final records were last formed

    def schedule(self, post: datetime) -> None:
        if post == self.post:
            return
        known = {e[1]: (k, e) for s in (self.newest, *self.slots.values()) for k, e in s.items()}
        self.post, self.slots = post, {}
        for k in CHECKPOINTS:
            if k not in self.done:
                slot: dict[tuple, Ref] = {}
                for key, entry in (known[seq] for seq in sorted(known)):
                    if entry[0] <= post - timedelta(minutes=k):
                        _offer(slot, key, entry)
                self.slots[k] = slot

    def add(self, key: tuple, entry: Ref) -> None:
        _offer(self.newest, key, entry)
        for k, slot in self.slots.items():
            if self.post is not None and entry[0] <= self.post - timedelta(minutes=k):
                _offer(slot, key, entry)

    def decided(self, k: int) -> None:
        self.done.add(k)
        self.slots.pop(k, None)
        if not self.slots and self.post is not None:
            self.newest = {}

    def final_complete(self) -> bool:
        """Its final is written for WIN and for every pool its checkpoints showed."""
        return ({WIN} | self.shown) <= {pool for pool, _ in self.finals}


def _keys(board: RaceBoard) -> list[tuple[str, tuple, str, float]]:
    """(pool, board key runners, combination text, low figure) of every key on the board."""
    out = []
    for pool, figures in ((WIN, board.win), (PLACE, board.place), (WIDE, board.wide)):
        for runners, figure in sorted(figures.items()):
            text = f"{runners[0]},{runners[1]}" if pool == WIDE else str(runners)
            out.append((pool, runners, text, float(figure[0] if pool != WIN else figure)))
    return out


def _iso(at: datetime | None) -> str | None:
    return None if at is None else at.astimezone(timezone.utc).isoformat()


class LadderDay:
    """The ladder log's side of a shadow-day run: fed the rows and post times shadow-day reads,
    `step` writes each checkpoint once it is due (post - k + grace_s) and each final once its board
    is in; a key already in the log is not written again, so a restart appends only what is
    missing."""

    def __init__(
        self,
        log: LadderLog,
        *,
        models: ShadowModels | None,
        grace_s: float,
        max_board_age_s: Mapping[int, float],
        max_lag_s: float,
        git_sha: str,
    ) -> None:
        self.log = log
        self.grace_s, self.max_lag_s, self.git_sha = grace_s, max_lag_s, git_sha
        self.limits = checkpoint_age_limits(max_board_age_s)
        self.project = _projector(models)
        self.nowcast_id = None if models is None else models.ids.get("nowcast")
        self.races: dict[str, _RaceLadder] = {}
        self.seq = 0
        self.logged: dict[str, list[tuple]] = {}
        for key in log.logged_keys():
            self.logged.setdefault(key[0], []).append(key)

    def _race(self, race_id: str) -> _RaceLadder:
        race = self.races.get(race_id)
        if race is None:
            race = self.races[race_id] = _RaceLadder(race_id)
            for _, pool, combination, k in self.logged.pop(race_id, []):
                if k == FINAL:
                    race.finals.add((pool, combination))
                else:
                    race.done.add(k)
                    if pool != ANY:
                        race.shown.add(pool)
        return race

    def schedule(self, race_id: str, post: datetime) -> None:
        self._race(race_id).schedule(post)

    def add(self, rows: list[dict[str, Any]]) -> None:
        """One poll's rows: a capture is one gzip member, read whole, so a final capture's rows
        (its SCR rows and other pools included) are known by their snapshot_utc."""
        parsed = []
        for row in rows:
            race_id, key = row.get("race_id"), snapshot._key(row)
            snap = _instant(row.get("snapshot_utc"))
            if not isinstance(race_id, str) or key is None or snap is None:
                continue
            if key[1] == snapshot.POOL_TOTAL:
                continue
            race = self._race(race_id)
            if snapshot._is_final(row):
                race.final_snaps.add(snap)
            parsed.append((race, key, snap, row))
        for race, key, snap, row in parsed:
            self.seq += 1
            if snap in race.final_snaps:
                if not race.final_complete():
                    race.final_rows.append(row)
                    race.final_new = True
            elif race.post is None or race.slots:
                race.add(key, (snap, self.seq, row))

    def _common(self, race: _RaceLadder, k: int, at: datetime, written: datetime) -> dict:
        lag = (written - at).total_seconds()
        return {
            "race_id": race.race_id,
            "checkpoint_min": k,
            "post_utc": _iso(race.post),
            "written_utc": _iso(written),
            "mode": "live" if lag <= self.max_lag_s else "replay",
            "nowcast_id": self.nowcast_id,
            "git_sha": self.git_sha,
        }

    def _checkpoint(self, race: _RaceLadder, k: int, written: datetime) -> list[dict]:
        post = race.post
        assert post is not None
        at = post - timedelta(minutes=k)
        latest = {
            key: (snapshot._available(row) or snap, row)
            for key, (snap, _, row) in race.slots[k].items()
        }
        board = snapshot._build(race.race_id, latest, at, post)
        common = self._common(race, k, at, written)
        age = None if board is None else (at - board.capture_utc).total_seconds()
        if board is None or age > self.limits[k]:
            empty = {"pool": ANY, "combination": ANY, "status": NO_BOARD, "odds": None}
            return [{**common, **empty, "fetched_utc": None, "board_age_s": age, "projected": {}}]
        later = [j for j in CHECKPOINTS if j < k] + [FINAL]
        out = []
        for pool, runners, combination, odds in _keys(board):
            fetched = latest[(pool, runners)][0]
            tau = max(0.0, (post - fetched) / timedelta(minutes=1))
            out.append(
                {
                    **common,
                    "pool": pool,
                    "combination": combination,
                    "status": "ok",
                    "odds": odds,
                    "fetched_utc": _iso(fetched),
                    "board_age_s": age,
                    "projected": {str(j): self.project(pool, odds, tau, j) for j in later},
                }
            )
        return out

    def _final(self, race: _RaceLadder, written: datetime) -> list[dict]:
        board = final_board(race.final_rows, race.race_id, post_utc=race.post)
        if board is None:
            return []
        common = self._common(race, FINAL, board.at_utc, written)
        return [
            {
                **common,
                "pool": pool,
                "combination": combination,
                "status": "ok",
                "odds": odds,
                "fetched_utc": _iso(board.at_utc),
                "board_age_s": None,
                "projected": {},
            }
            for pool, _, combination, odds in _keys(board)
            if (pool, combination) not in race.finals
        ]

    def step(self, t: datetime, clock: Callable[[], datetime]) -> int:
        """Write every checkpoint due at `t` and every final now in, in one append; returns the
        number of records written."""
        grace = timedelta(seconds=self.grace_s)
        due = [
            (race, k)
            for race in self.races.values()
            if race.post is not None
            for k in list(race.slots)
            if t >= race.post - timedelta(minutes=k) + grace
        ]
        finals = [
            r
            for r in self.races.values()
            if r.post is not None and r.final_new and not r.final_complete()
        ]
        if not due and not finals:
            return 0
        written = clock()
        batch = [record for race, k in due for record in self._checkpoint(race, k, written)]
        ended = [(race, self._final(race, written)) for race in finals]
        batch += [record for _, records in ended for record in records]
        self.log.append(batch)
        for race, k in due:
            race.decided(k)
        for record in batch:
            race = self.races[record["race_id"]]
            if record["checkpoint_min"] == FINAL:
                race.finals.add((record["pool"], record["combination"]))
            elif record["status"] == "ok":
                race.shown.add(record["pool"])
        for race, _ in ended:
            race.final_new = False
            if race.final_complete():
                race.final_rows, race.final_snaps = [], set()
        return len(batch)

    def complete(self) -> bool:
        """Every scheduled race has its checkpoints and its final written."""
        return all(
            not r.slots and r.final_complete() for r in self.races.values() if r.post is not None
        )
