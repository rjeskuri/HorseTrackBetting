"""The owner's manual bet log per Japan race day, hash-chained, with the race-day cap (#453).

``data_dir()/bets/japan/<YYYY-MM-DD>.jsonl``, the one log of the day, holds one JSON line per bet
the owner placed by hand, each line carrying a sha256 of itself and chained to the one before it by
sha256, as the shadow log chains its lines (docs/specs/shadow-log.md). A placed bet is a fact: a
bet past the day cap or off the flat stake is recorded and flagged, never refused; ``check`` is the
gate before betting. Stakes are not outcomes, so a holdout day's bets are read back and counted
like any other day's. The format is in docs/specs/bet-log.md. The log is the owner's financial
record and stays outside the repository.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from horsetrack.core import env, paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.reports import is_inside

COUNTRY = "japan"
SOURCE = "jra"
POOLS = (
    "WIN",
    "PLACE",
    "BRACKET_QUINELLA",
    "QUINELLA",
    "QUINELLA_PLACE",
    "FORECAST",
    "TRIO",
    "TIERCE",
    "WIN5",
)
HOLDOUT = "holdout"
DEVELOPMENT = "development"
FIRST_SEQ = 1
ZERO_SHA = "0" * 64
REQUIRED = ("race_id", "pool", "combination", "stake_yen", "placed_utc")
OPTIONAL = ("option_id", "note")
ADDED = ("seq", "written_utc", "day_status", "over_cap", "off_flat")
LINE_KEYS = frozenset({"seq", "prev_sha256", "record", "sha256"})
WIN5 = "WIN5"  # the pool, keyed by its day (jra:<day>:WIN5) as research/win5.py keys it

# [0-9] and [A-Z], not \d and \w: those match full-width digits and letters too.
_RACE_ID = re.compile(
    r"jra:(?P<day>[0-9]{4}-[0-9]{2}-[0-9]{2}):(?:(?P<venue>[A-Z]+):(?:0[1-9]|[1-9][0-9])|WIN5)"
)
_COMBINATION = re.compile(r"[1-9][0-9]?(?:,[1-9][0-9]?)*")


class BetRefused(ValueError):
    """A bet the log does not accept; nothing is written."""


class BetLogTampered(ValueError):
    """The file is not the chain this log wrote: a line was edited, inserted or deleted."""


class BetLogInRepo(RuntimeError):
    """The bet log was pointed inside the repository; it lives under $HORSETRACK_DATA_DIR."""


@dataclass(frozen=True)
class Limits:
    day_cap_yen: int
    flat_stake_yen: int | None

    def __post_init__(self) -> None:
        cap, flat = self.day_cap_yen, self.flat_stake_yen
        if type(cap) is not int or cap < 0:
            raise ValueError(f"day_cap_yen must be a whole number of yen >= 0: {cap!r}")
        if flat is not None and _stake_refusal(flat) is not None:
            raise ValueError(f"flat_stake_yen {_stake_refusal(flat)}")


@dataclass(frozen=True)
class CapCheck:
    fits: bool
    staked_yen: int
    remaining_yen: int
    off_flat: bool


@dataclass(frozen=True)
class _State:
    records: list[dict[str, Any]]
    staked_yen: int
    held: bool  # some bet was logged as holdout
    next_seq: int
    last_sha: str


def load_limits() -> Limits:
    """The committed day cap and flat stake, from countries/japan/config/betting.yaml."""
    raw = load_country_yaml(COUNTRY, "betting")
    if not {"day_cap_yen"} <= set(raw) <= {"day_cap_yen", "flat_stake_yen"}:
        raise ValueError(f"betting.yaml takes day_cap_yen and flat_stake_yen, got {sorted(raw)}")
    return Limits(day_cap_yen=raw["day_cap_yen"], flat_stake_yen=raw.get("flat_stake_yen"))


def _check_day(day: Any) -> None:
    if not isinstance(day, date) or isinstance(day, datetime):
        raise TypeError(f"day must be a date, got {day!r}")


def bet_log_path(day: date) -> Path:
    """``data_dir()/bets/japan/<YYYY-MM-DD>.jsonl``, the data dir read on every call."""
    _check_day(day)
    return paths.data_dir() / "bets" / COUNTRY / f"{day.isoformat()}.jsonl"


# ---------------------------------------------------------------- encoding and checks


def _encode(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def _decode(line: bytes) -> Any:
    """The JSON of ``line``; ValueError unless ``line`` is exactly how this log encodes it."""
    obj = json.loads(line)
    if _encode(obj) != line:
        raise ValueError("not in the log's own encoding")
    return obj


def _sha(line: bytes) -> str:
    return hashlib.sha256(line).hexdigest()


def _instant(value: Any) -> datetime | None:
    """ISO 8601 text with a UTC offset (``Z`` or ``+HH:MM``) as an aware datetime, else None."""
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return None
    return stamp if stamp.utcoffset() is not None else None


def _jra() -> dict[str, Any]:
    return load_country_yaml(COUNTRY, SOURCE)


def _kind(pool: str) -> str:
    return _jra()["pools"][pool]["kind"]


def _legs_in_order(pool: str) -> bool:
    """Whether the legs are in an order the dividend depends on: finishing order (the
    ``ordered_*`` kinds) or race order (WIN5's ``multi_race_*``)."""
    return _kind(pool).startswith(("ordered_", "multi_race_"))


def _repeats_allowed(pool: str) -> bool:
    """Whether a number may repeat: the legs are brackets (two horses of one bracket) or races
    (WIN5), not horses of one race."""
    kind = _kind(pool)
    return kind.endswith("_bracket") or kind.startswith("multi_race_")


def _in_store_order(pool: str, combination: str) -> str:
    """``combination`` as the store holds it (jravan/dividends.py): as given when its legs are in
    order, else ascending by number."""
    if _legs_in_order(pool):
        return combination
    return ",".join(sorted(combination.split(","), key=int))


def _stake_refusal(stake: Any) -> str | None:
    """Why ``stake`` is not a stake JRA sells (whole units of its unit stake), or None."""
    unit = int(_jra()["unit_stake"])
    if type(stake) is not int or stake < unit or stake % unit:
        return f"is not a whole multiple of {unit} yen, at least {unit}: {stake!r}"
    return None


def _refusal(bet: Any, day: date) -> str | None:
    """Why ``bet`` is refused as a bet of ``day``, or None."""
    if not isinstance(bet, dict):
        return "is not a dict"
    unknown = sorted(repr(key) for key in bet if key not in REQUIRED + OPTIONAL)
    if unknown:
        return f"has keys the log does not take (it sets {', '.join(ADDED)}): {', '.join(unknown)}"
    missing = [name for name in REQUIRED if name not in bet]
    if missing:
        return f"lacks {', '.join(missing)}"
    race_id = bet["race_id"]
    match = _RACE_ID.fullmatch(race_id) if isinstance(race_id, str) else None
    if match is None:
        return f"has a race_id not of the form jra:<YYYY-MM-DD>:<VENUE>:<NN> or :WIN5: {race_id!r}"
    if match["day"] != day.isoformat():
        return f"has a race_id of another day than {day.isoformat()}: {race_id!r}"
    if match["venue"] is not None and match["venue"] not in _jra()["venues"]:
        return f"has a race_id at an unknown venue: {race_id!r}"
    pool = bet["pool"]
    if not isinstance(pool, str) or pool not in POOLS:
        return f"has a pool other than {', '.join(POOLS)}: {pool!r}"
    if (pool == WIN5) != (match["venue"] is None):
        return f"has the race_id {race_id!r} for {pool}: WIN5, and only WIN5, takes jra:<day>:WIN5"
    combination = bet["combination"]
    if not isinstance(combination, str) or not _COMBINATION.fullmatch(combination):
        return f"has a combination that is not horse numbers joined by ',': {combination!r}"
    numbers = combination.split(",")
    legs = _jra()["pools"][pool]["legs"]
    if len(numbers) != legs:
        return f"has a combination of other than {legs} numbers for {pool}: {combination!r}"
    if len(set(numbers)) != len(numbers) and not _repeats_allowed(pool):
        return f"has a combination with a number twice, which {pool} does not sell: {combination!r}"
    reason = _stake_refusal(bet["stake_yen"])
    if reason is not None:
        return f"has a stake_yen that {reason}"
    if _instant(bet["placed_utc"]) is None:
        return f"has a placed_utc that is not ISO text with a UTC offset: {bet['placed_utc']!r}"
    option_id = bet.get("option_id")
    if option_id is not None and (not isinstance(option_id, str) or not option_id.strip()):
        return f"has an option_id that is neither non-blank text nor null: {option_id!r}"
    note = bet.get("note")
    if note is not None and not isinstance(note, str):
        return f"has a note that is neither text nor null: {note!r}"
    return None


def _stored_refusal(record: Any, seq: int, day: date) -> str | None:
    """Why a record read back is not one this log writes as bet ``seq`` of ``day``, or None."""
    if not isinstance(record, dict):
        return "is not a dict"
    keys = set(record)
    if not set(REQUIRED + ADDED) <= keys <= set(REQUIRED + OPTIONAL + ADDED):
        return f"has the keys {sorted(keys)}"
    if not (
        type(record["seq"]) is int
        and record["seq"] == seq
        and record["day_status"] in (HOLDOUT, DEVELOPMENT)
        and type(record["over_cap"]) is bool
        and type(record["off_flat"]) is bool
    ):
        return "has a seq, day_status or flag this log does not write"
    reason = _refusal({key: record[key] for key in keys - set(ADDED)}, day)  # a bet of the day
    if reason is not None:
        return reason
    if record["combination"] != _in_store_order(record["pool"], record["combination"]):
        return f"has a combination not in the store's order: {record['combination']!r}"
    return None


def _parse(raw: bytes, path: Path, day: date) -> _State:
    """The verified content of a log file; BetLogTampered on any break in the chain."""
    if raw and not raw.endswith(b"\n"):
        raise BetLogTampered(f"{path}: the last line is cut off")
    lines = raw[:-1].split(b"\n") if raw else []
    records: list[dict[str, Any]] = []
    prev = ZERO_SHA
    for n, line in enumerate(lines, 1):
        where = f"{path}, line {n}"
        try:
            obj = _decode(line)
        except (TypeError, ValueError, RecursionError):
            obj = None
        if not isinstance(obj, dict) or set(obj) != LINE_KEYS:
            raise BetLogTampered(f"{where}: not a line this log wrote")
        if obj["sha256"] != _sha(_encode({k: v for k, v in obj.items() if k != "sha256"})):
            raise BetLogTampered(f"{where}: its sha256 is not that of the line")
        if obj["prev_sha256"] != prev:
            raise BetLogTampered(f"{where}: does not chain onto the line before it")
        seq = FIRST_SEQ + n - 1
        if type(obj["seq"]) is not int or obj["seq"] != seq:
            raise BetLogTampered(f"{where}: seq out of order")
        reason = _stored_refusal(obj["record"], seq, day)
        if reason is not None:
            raise BetLogTampered(f"{where}: not a bet record this log wrote: it {reason}")
        records.append(obj["record"])
        prev = _sha(line)
    return _State(
        records=records,
        staked_yen=sum(record["stake_yen"] for record in records),
        held=any(record["day_status"] == HOLDOUT for record in records),
        next_seq=FIRST_SEQ + len(lines),
        last_sha=prev,
    )


def _fsync_dir(path: Path) -> None:
    """Make a new file's directory entry durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------- the log


class BetLog:
    """One race day's bet log. Every call reads and verifies the whole file first."""

    def __init__(self, path: Path, *, day: date, limits: Limits) -> None:
        _check_day(day)
        if not isinstance(limits, Limits):
            raise TypeError(f"limits must be Limits, got {limits!r}")
        path = Path(path)
        root = env.repo_root()
        if is_inside(path, root):
            raise BetLogInRepo(f"{path} is inside the repository at {root}: bets stay outside it")
        own = bet_log_path(day)  # one log per day, so check and record count the same stakes
        if path.resolve() != own.resolve():
            raise ValueError(f"{path} is not the bet log of {day.isoformat()}, which is {own}")
        self._path = own
        self._day = day
        self._limits = limits

    def _load(self) -> _State:
        try:
            with open(self._path, "rb") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH)  # no half-written line from a writer
                raw = fh.read()
        except FileNotFoundError:
            raw = b""
        return _parse(raw, self._path, self._day)

    def _status(self, state: _State) -> str:
        """Fails closed: holdout when the rule says so now or any bet was logged as holdout."""
        return HOLDOUT if state.held or is_holdout_day(COUNTRY, self._day) else DEVELOPMENT

    def _off_flat(self, stake_yen: int) -> bool:
        flat = self._limits.flat_stake_yen
        return flat is not None and stake_yen != flat

    def check(self, stake_yen: int) -> CapCheck:
        """The day's staked total, the cap left before ``stake_yen`` and whether it fits; writes
        nothing. ``remaining_yen`` is negative once the day is past the cap."""
        staked = self._load().staked_yen
        reason = _stake_refusal(stake_yen)
        if reason is not None:
            raise BetRefused(f"the stake {reason}")
        cap = self._limits.day_cap_yen
        return CapCheck(
            fits=staked + stake_yen <= cap,
            staked_yen=staked,
            remaining_yen=cap - staked,
            off_flat=self._off_flat(stake_yen),
        )

    def record(self, bet: dict) -> dict:
        """Append one placed bet and return the stored record, its combination in the store's
        order, or raise BetRefused and write nothing. A bet past the cap or off the flat stake is
        recorded and flagged."""
        reason = _refusal(bet, self._day)
        if reason is not None:
            raise BetRefused(f"the bet {reason}; nothing was written")
        stake = bet["stake_yen"]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        created = not self._path.exists()
        with open(self._path, "a+b") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)  # read, verify and append as one step
            fh.seek(0)
            state = _parse(fh.read(), self._path, self._day)
            record = {
                **bet,
                "combination": _in_store_order(bet["pool"], bet["combination"]),
                "seq": state.next_seq,
                "written_utc": datetime.now(timezone.utc).isoformat(),
                "day_status": self._status(state),
                "over_cap": state.staked_yen + stake > self._limits.day_cap_yen,
                "off_flat": self._off_flat(stake),
            }
            body = {"seq": state.next_seq, "prev_sha256": state.last_sha, "record": record}
            line = _encode({**body, "sha256": _sha(_encode(body))})
            stored = _decode(line)["record"]  # it must read back as written
            fh.seek(0, os.SEEK_END)
            fh.write(line + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        if created:
            _fsync_dir(self._path.parent)
        return stored

    def read(self) -> list[dict]:
        """The day's bets in order, each as stored; on any day, holdout days too."""
        return [dict(record) for record in self._load().records]

    def staked_yen(self) -> int:
        """The day's staked total, every bet in the file counted."""
        return self._load().staked_yen
