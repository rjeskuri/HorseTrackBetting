"""The best bets of a venue's next race, read on the nowcast's predicted final board (ticket #714).

Rules (docs/specs/board-feed.md, "Best bets"; the owner's selection rule, D19):
- Eligible rows (`eligible`): the private leaderboard's rows of selection `predicted_final@T-k` at
  the current checkpoint k, or `decision_time`, with a shrunk return, n >= 100 (whatever its
  n_unit) and n_meetings >= 20. Licensed rows count: the page is token-gated and not in the
  repository. look_ahead, card_only and unknown rows never.
- The checkpoint (`checkpoint_for`): T-10 while more than 5 min to the post, T-5 while more than 2,
  T-2 from post - 2 on.
- Matching (`match`): a row whose rule is a `predicted_final.RULES` rule of its family is read
  with that rule's own picker on the projected board; any other row (no matcher for its family, an
  unknown or an unmeasured rule) is listed as unmatched, never dropped. A matched rule that buys
  nothing on the board is neither. Each ticket (pool, combination) appears once, under its row of
  the highest shrunk return per 100 yen, the other rows buying it as its duplicates; tickets are
  ranked by that shrunk return. The highlight is "strong" at 100 or more, "muted" below. The
  page's Model % plays no part.
- The board (`build_best_bets`): the venue's first race whose post time (the post-time sidecar, so
  a delayed post counts) is after now. Before post - 2 the board as of now (provisional); from post
  - 2 on the latest row of each price captured at or before post - 2 (frozen: the ladder's T-2
  rule, so a capture after post - 2 changes nothing). Either is projected to the final with
  `models.nowcast` by `predicted_final.project_board`, tau the minutes from the board's capture to
  the post.
- Horse numbers carry their names from the day's card log (`<VENUE>.cards.jsonl.gz`, the logger's
  card captures) when it has them; the card's class and surface feed the rules' race cells.

Information only: nothing here places or sizes a bet.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Iterable, MutableMapping
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from horsetrack.core.eval.leaderboard import LeaderboardRow, from_json
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.board.feed import utc_now
from horsetrack.countries.japan.board.shadow_feed import _brackets, _DayRows, _iso
from horsetrack.countries.japan.board.snapshot import (
    RaceBoard,
    board_at,
    card_race_classes,
    jra_day_dir,
    post_times_for,
)
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels
from horsetrack.countries.japan.research import predicted_final as pf

log = logging.getLogger(__name__)

DECISION_TIME = "decision_time"
MIN_N, MIN_MEETINGS = 100, 20
STRONG_AT = 100.0  # shrunk return per 100 yen from which the highlight is strong
FREEZE_MIN = 2  # the checkpoint at which the list freezes
CARD_SUFFIX = ".cards.jsonl.gz"
BRACKET_POOLS = frozenset({"BRACKET_QUINELLA"})  # combinations of brackets, not horses


def checkpoint_for(minutes_to_post: float) -> int:
    """10 while more than 5 min to the post, 5 while more than 2, 2 from post - 2 on."""
    if minutes_to_post > 5:
        return 10
    if minutes_to_post > FREEZE_MIN:
        return 5
    return FREEZE_MIN


def _number(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def eligible(rows: Iterable[LeaderboardRow], checkpoint_min: int) -> list[LeaderboardRow]:
    """The rows that may be best bets at T-k (module docstring), each option once."""
    wanted = {pf.selection_label(checkpoint_min), DECISION_TIME}
    out, seen = [], set()
    for row in rows:
        if row.option_id in seen or row.selection not in wanted:
            continue
        if not _number(row.shrunk_per_100):
            continue
        if (
            row.n is None
            or row.n < MIN_N
            or row.n_meetings is None
            or row.n_meetings < MIN_MEETINGS
        ):
            continue
        seen.add(row.option_id)
        out.append(row)
    return out


def _matcher(row: LeaderboardRow) -> pf.Rule | None:
    """The #713 rule that reads `row` on a projected board, or None (no matcher)."""
    rule = pf.RULES.get(row.rule)
    if rule is None or rule.family_id != row.family_id or rule.pick is None:
        return None
    return rule


def _figures(row: LeaderboardRow) -> dict[str, Any]:
    return {
        "option_id": row.option_id,
        "family_id": row.family_id,
        "rule": row.rule,
        "selection": row.selection,
        "shrunk_per_100": row.shrunk_per_100,
        "return_per_100": row.return_per_100,
        "ci_lo": row.ci_lo,
        "ci_hi": row.ci_hi,
        "ci_level": row.ci_level,
        "n": row.n,
        "n_unit": row.n_unit,
        "n_meetings": row.n_meetings,
        "variants_searched": row.variants_searched,
        "licensed": row.licensed,
    }


def _by_shrunk(row: LeaderboardRow) -> tuple[float, str]:
    return (-float(row.shrunk_per_100 or 0.0), row.option_id)


def _unmatched(rows: Iterable[LeaderboardRow]) -> list[dict[str, Any]]:
    return [
        {**_figures(r), "reason": "no matcher for this family or rule"}
        for r in sorted(rows, key=_by_shrunk)
        if _matcher(r) is None
    ]


def _combination_order(key: tuple[str, str]) -> tuple:
    return (key[0], tuple(int(h) for h in key[1].split(",")))


def match(
    rows: Iterable[LeaderboardRow],
    board: RaceBoard,
    *,
    race: pf.RaceInput,
    names: dict[int, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """{"best_bets": [...], "unmatched": [...]} of the eligible `rows` read on the projected
    `board` (module docstring). `names`: horse number -> name, shown beside each ticket."""
    rows = list(rows)
    buyers: dict[tuple[str, str], list[LeaderboardRow]] = {}
    for row in rows:
        rule = _matcher(row)
        if rule is None:
            continue
        tickets = pf.select(rule.rule_id, board, race=race)
        if tickets is None:
            continue
        for combo in tickets.combinations:
            buyers.setdefault((tickets.pool, ",".join(str(h) for h in combo)), []).append(row)
    entries = []
    for key, found in buyers.items():
        found.sort(key=_by_shrunk)
        entries.append((key, found))
    entries.sort(key=lambda e: (_by_shrunk(e[1][0])[0], _combination_order(e[0])))
    best = []
    for rank, ((pool, combination), found) in enumerate(entries, 1):
        top = found[0]
        horses = None
        if pool not in BRACKET_POOLS:
            horses = [
                {"no": int(h), "name": (names or {}).get(int(h))} for h in combination.split(",")
            ]
        best.append(
            {
                "rank": rank,
                "pool": pool,
                "combination": combination,
                "horses": horses,
                **_figures(top),
                "description": pf.RULES[top.rule].description,
                "highlight": "strong" if float(top.shrunk_per_100) >= STRONG_AT else "muted",
                "duplicates": [
                    {**_figures(r), "description": pf.RULES[r.rule].description} for r in found[1:]
                ],
            }
        )
    return {"best_bets": best, "unmatched": _unmatched(rows)}


# ---------------------------------------------------------------- the day's card


class _Card:
    """The latest card capture of each race in the venue's card log: its class, surface and the
    horse names by number. Read on with a `LogTail`, so a rebuild parses only new members."""

    def __init__(self, path: Path) -> None:
        self.tail = odds_log.LogTail(path)
        self.races: dict[str, tuple[str, dict[str, Any]]] = {}
        self.runners: dict[str, dict[int, tuple[str, str]]] = {}

    def update(self) -> None:
        for row in self.tail.poll():
            race_id, at = row.get("race_id"), str(row.get("snapshot_utc") or "")
            if not isinstance(race_id, str):
                continue
            if row.get("table") == "races":
                if race_id not in self.races or at >= self.races[race_id][0]:
                    self.races[race_id] = (at, row)
            elif row.get("table") == "racecard_runners":
                no, name = row.get("horse_no"), row.get("horse_name")
                if isinstance(no, int) and isinstance(name, str) and name:
                    slot = self.runners.setdefault(race_id, {})
                    if no not in slot or at >= slot[no][0]:
                        slot[no] = (at, name)

    def race(self, race_id: str) -> dict[str, Any]:
        return self.races.get(race_id, ("", {}))[1]

    def names(self, race_id: str) -> dict[int, str]:
        return {no: name for no, (_, name) in self.runners.get(race_id, {}).items()}


# ---------------------------------------------------------------- the venue's next race


def _next_race(posts: dict[int, datetime], now_utc: datetime) -> tuple[int, datetime] | None:
    ahead = [(post, no) for no, post in posts.items() if post > now_utc]
    if not ahead:
        return None
    post, no = min(ahead)
    return no, post


def _frozen_board(day_rows: _DayRows, race_id: str, post: datetime) -> RaceBoard | None:
    """The latest row of each price captured at or before post - 2 (the ladder's T-2 rule)."""
    refs = day_rows.refs.get(race_id, {}).get(FREEZE_MIN, {})
    rows = [row for _, _, row in sorted(refs.values(), key=lambda e: e[1])]
    if not rows:
        return None
    available = [a for row in rows if (a := snapshot._available(row)) is not None]
    return board_at(rows, race_id, max(available), post_utc=post) if available else None


def _empty(race_date: date, venue: str, now_utc: datetime) -> dict[str, Any]:
    return {
        "race_date": race_date.isoformat(),
        "venue": venue,
        "generated_utc": _iso(now_utc),
        "race": None,
        "checkpoint_min": None,
        "frozen": False,
        "board_capture_utc": None,
        "note": None,
        "best_bets": [],
        "unmatched": [],
    }


def build_best_bets(
    race_date: date,
    *,
    venue: str,
    leaderboard: Iterable[LeaderboardRow],
    models: ShadowModels | None,
    now: Callable[[], datetime] = utc_now,
    day_dir: Path | None = None,
    cache: MutableMapping | None = None,
) -> dict[str, Any]:
    """The best bets of `venue`'s next race as of now() (module docstring)."""
    now_utc = now().astimezone(timezone.utc)
    out = _empty(race_date, venue, now_utc)
    posts = post_times_for(race_date, venue, day_dir=day_dir)
    found = _next_race(posts, now_utc)
    if found is None:
        return out
    race_no, post = found
    race_id = f"jra:{race_date.isoformat()}:{venue}:{race_no:02d}"
    minutes = (post - now_utc) / timedelta(minutes=1)
    k = checkpoint_for(minutes)
    frozen = minutes <= FREEZE_MIN
    rows = eligible(leaderboard, k)
    out.update(
        race={"race_id": race_id, "race_no": race_no, "post_utc": _iso(post)},
        checkpoint_min=k,
        frozen=frozen,
        unmatched=_unmatched(rows),
    )

    cache = {} if cache is None else cache
    key = ("best-bets-logs", race_date, day_dir, venue)
    day_rows = cache.get(key)
    if day_rows is None:
        day_rows = cache[key] = _DayRows(race_date, day_dir)
    day_rows.update([venue], now_utc, {venue: posts})
    if frozen:
        board = _frozen_board(day_rows, race_id, post)
    else:
        board = board_at(day_rows.rows(race_id), race_id, now_utc, post_utc=post)
    if board is None:
        out["note"] = "no board captured yet"
        return out
    out["board_capture_utc"] = _iso(board.capture_utc)
    nowcast = None if models is None else models.nowcast
    if nowcast is None:
        out["note"] = "no nowcast: no predicted final board"
        return out

    card_key = ("best-bets-card", race_date, day_dir, venue)
    card = cache.get(card_key)
    if card is None:
        card = cache[card_key] = _Card(jra_day_dir(race_date, day_dir) / f"{venue}{CARD_SUFFIX}")
    try:
        card.update()
    except Exception:  # the card adds names and cells only: go on without the new members
        log.exception("%s: card log unreadable; best bets without it", venue)
    on_card = card.race(race_id)
    race_class = on_card.get("race_class") or card_race_classes(race_date, venue).get(race_no)
    tau = max(0.0, (post - board.capture_utc) / timedelta(minutes=1))
    projected = pf.project_board(board, nowcast=nowcast, checkpoint_min=tau)
    fielded = set(board.win) | set(board.place) | set(board.scratched)
    drawn = max(fielded | {h for pair in board.wide for h in pair}, default=0)
    race = pf.RaceInput(
        race_id=race_id,
        race_date=race_date,
        meeting_id=race_id.rsplit(":", 1)[0],
        race_class=str(race_class or ""),
        surface=str(on_card.get("surface") or ""),
        boards={},
        final=None,
        paid=pd.DataFrame(columns=["race_id", "pool", "combination", "dividend", "is_refund"]),
        brackets=_brackets(drawn) or {},
    )
    out.update(match(rows, projected, race=race, names=card.names(race_id)))
    return out


# ---------------------------------------------------------------- the leaderboard file


class LeaderboardFile:
    """The rows of the private leaderboard JSON (`leaderboard.to_json`'s text) at `path`, read
    again whenever the file's size or mtime changes; [] while there is no file. A file that does
    not parse keeps the last rows read (a write in progress)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._rows: list[LeaderboardRow] = []

    def rows(self) -> list[LeaderboardRow]:
        with self._lock:
            try:
                st = self.path.stat()
            except FileNotFoundError:
                self._stamp, self._rows = None, []
                return []
            stamp = (st.st_size, st.st_mtime_ns)
            if stamp != self._stamp:
                try:
                    self._rows = from_json(self.path.read_text(encoding="utf-8"))
                    self._stamp = stamp
                except (OSError, ValueError):
                    log.exception("leaderboard %s unreadable; keeping the last rows", self.path)
            return list(self._rows)
