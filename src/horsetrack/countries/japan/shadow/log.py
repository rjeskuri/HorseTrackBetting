"""A sealed, append-only shadow log per Japan race day (#119).

``data_dir()/shadow/japan/<YYYY-MM-DD>.jsonl`` holds one JSON line per shadow record, each line
chained to the one before it by sha256, and a seal line once the day is closed. The log sets each
record's ``day_status`` from the holdout rule (#71). A holdout day's records are written and
verified here but never read back: settlement and calibration reach them through
``LockedHoldout.open`` (#101). Records are versioned (#564): ``record_version`` 2 carries
``p_model``; a record without the field is version 1 and has no ``p_model``. The format is in
docs/specs/shadow-log.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day

COUNTRY = "japan"
HOLDOUT = "holdout"
DEVELOPMENT = "development"
NO_BOARD = "no_board"
FIRST_SEQ = 1
ZERO_SHA = "0" * 64
KEY_FIELDS = ("race_id", "pool", "combination", "tau_min")
REQUIRED = (
    *KEY_FIELDS,
    "decision_utc",
    "board_capture_utc",
    "written_utc",
    "mode",
    "status",
    "models",
    "git_sha",
)
MODES = ("live", "replay")
STATUSES = ("ok", NO_BOARD)
STAKE_LIKE = ("stake", "bet", "kelly", "amount", "wager")
RECORD_VERSION = 2  # the version a record carrying p_model states (#564); absent means 1
RECORD_VERSIONS = (1, RECORD_VERSION)

Key = tuple[str, str, str, int]


class RecordRefused(ValueError):
    """A record the log does not accept; the whole batch is refused and nothing is written."""


class LogSealed(ValueError):
    """The day's log is sealed: nothing more is appended."""


class ShadowLogTampered(ValueError):
    """The file is not the chain this log wrote: a line was edited, inserted or deleted."""


class HoldoutSealed(PermissionError):
    """A holdout day's records are not read through this module (#71, #101)."""


@dataclass(frozen=True)
class LogSummary:
    n_records: int
    by_tau: dict[int, int]
    no_board: int
    sealed: bool
    day_status: str


@dataclass(frozen=True)
class _State:
    records: list[dict[str, Any]]
    keys: set[Key]
    sealed: bool
    held: bool  # some record was logged as holdout
    next_seq: int
    last_sha: str
    size: int


def _check_day(day: Any) -> None:
    if not isinstance(day, date) or isinstance(day, datetime):
        raise TypeError(f"day must be a date, got {day!r}")


def shadow_log_path(day: date) -> Path:
    """``data_dir()/shadow/japan/<YYYY-MM-DD>.jsonl``, the data dir read on every call."""
    _check_day(day)
    return paths.data_dir() / "shadow" / COUNTRY / f"{day.isoformat()}.jsonl"


def day_status(day: date) -> str:
    """'holdout' when the committed holdout rule holds ``day``, else 'development'."""
    _check_day(day)
    return HOLDOUT if is_holdout_day(COUNTRY, day) else DEVELOPMENT


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


def record_version(record: dict[str, Any]) -> int:
    """The record's ``record_version``; 1 for a record without one (written before #564)."""
    return record.get("record_version", 1)


def _number(value: Any) -> bool:
    """A finite int or float, not a bool."""
    return type(value) in (int, float) and math.isfinite(value)


def _version_refusal(record: dict[str, Any]) -> str | None:
    """Why the record's version, ``p_model`` or ``value_model`` is refused, or None."""
    if "record_version" in record and (
        type(record["record_version"]) is not int or record["record_version"] not in RECORD_VERSIONS
    ):
        return f"has a record_version other than {' or '.join(map(str, RECORD_VERSIONS))}"
    version = record_version(record)
    if version == RECORD_VERSION and "p_model" not in record:
        return f"is version {RECORD_VERSION} but lacks p_model"
    if version == 1 and "p_model" in record:
        return "has a p_model but is version 1"
    p_model = record.get("p_model")
    if p_model is not None and not (_number(p_model) and 0 <= p_model <= 1):
        return "has a p_model that is neither null nor a number in [0, 1]"
    value = record.get("value_model")
    if value is not None and not (_number(value) and value >= 0):
        return "has a value_model that is neither null nor a finite number >= 0"
    return None


def _stake_like_key(value: Any) -> str | None:
    """The first key, at any depth, that names a stake, bet, kelly, amount or wager."""
    if isinstance(value, dict):
        for name, inner in value.items():
            if any(part in str(name).lower() for part in STAKE_LIKE):
                return str(name)
            found = _stake_like_key(inner)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for inner in value:
            found = _stake_like_key(inner)
            if found is not None:
                return found
    return None


def _refusal(record: Any) -> str | None:
    """Why ``record`` is refused, or None."""
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
    if type(record["tau_min"]) is not int:
        return "has a tau_min that is not an integer"
    if record["mode"] not in MODES:
        return f"has a mode other than {' or '.join(MODES)}"
    if record["status"] not in STATUSES:
        return f"has a status other than {' or '.join(STATUSES)}"
    if record["models"] is not None and not isinstance(record["models"], dict):
        return "has models that are neither fit ids nor null"
    reason = _version_refusal(record)
    if reason is not None:
        return reason
    decision = _instant(record["decision_utc"])
    if decision is None:
        return "has a decision_utc that is not ISO text with a UTC offset"
    if _instant(record["written_utc"]) is None:
        return "has a written_utc that is not ISO text with a UTC offset"
    board = record["board_capture_utc"]
    if board is None:
        if record["status"] != NO_BOARD:
            return "has a null board_capture_utc but status is not no_board"
    else:
        captured = _instant(board)
        if captured is None:
            return "has a board_capture_utc that is not ISO text with a UTC offset"
        if captured > decision:
            return "has a board_capture_utc after its decision_utc (look-ahead)"
    name = _stake_like_key(record)
    if name is not None:
        return f"has the stake-like key {name!r}"
    return None


def _stored_key(record: Any, key_fields: tuple[str, ...] = KEY_FIELDS) -> Key | None:
    """The key of a record line read back, or None when it is not one this log writes."""
    if not isinstance(record, dict) or record.get("day_status") not in (HOLDOUT, DEVELOPMENT):
        return None
    key = tuple(record.get(name) for name in key_fields)
    if not all(isinstance(part, str) for part in key[:3]) or type(key[3]) is not int:
        return None
    return key


def _parse(raw: bytes, path: Path, key_fields: tuple[str, ...] = KEY_FIELDS) -> _State:
    """The verified content of a log file; ShadowLogTampered on any break in the chain.
    `key_fields`: the four fields that key a record, each key at most once in the file."""
    if raw and not raw.endswith(b"\n"):
        raise ShadowLogTampered(f"{path}: the last line is cut off")
    lines = raw[:-1].split(b"\n") if raw else []
    records: list[dict[str, Any]] = []
    keys: set[Key] = set()
    sealed, prev = False, ZERO_SHA
    for n, line in enumerate(lines, 1):
        where = f"{path}, line {n}"
        if sealed:
            raise ShadowLogTampered(f"{where}: a line after the seal")
        try:
            obj = _decode(line)
        except (TypeError, ValueError):
            obj = None
        if not isinstance(obj, dict):
            raise ShadowLogTampered(f"{where}: not a line this log wrote")
        if obj.get("prev_sha256") != prev:
            raise ShadowLogTampered(f"{where}: does not chain onto the line before it")
        seq = obj.get("seq")
        if type(seq) is not int or seq != FIRST_SEQ + n - 1:
            raise ShadowLogTampered(f"{where}: seq out of order")
        if set(obj) == {"seq", "prev_sha256", "record"}:
            key = _stored_key(obj["record"], key_fields)
            if key is None or key in keys:
                raise ShadowLogTampered(f"{where}: a malformed or repeated record")
            keys.add(key)
            records.append(obj["record"])
        elif set(obj) == {"seq", "prev_sha256", "seal"}:
            seal = obj["seal"]
            if (
                not isinstance(seal, dict)
                or set(seal) != {"n_records", "sealed_utc"}
                or type(seal["n_records"]) is not int
                or seal["n_records"] != len(records)
                or _instant(seal["sealed_utc"]) is None
            ):
                raise ShadowLogTampered(f"{where}: the seal does not match the records")
            sealed = True
        else:
            raise ShadowLogTampered(f"{where}: neither a record nor a seal")
        prev = _sha(line)
    held = any(record["day_status"] == HOLDOUT for record in records)
    return _State(records, keys, sealed, held, FIRST_SEQ + len(lines), prev, len(raw))


# ---------------------------------------------------------------- the log


class ChainedLog:
    """One race day's hash-chained, sealed log; every call reads and verifies the whole file
    first. A subclass names its file suffix, its four key fields and its record check."""

    SUFFIX = ".jsonl"
    KEY_FIELDS: tuple[str, ...] = KEY_FIELDS

    def __init__(self, path: Path, *, day: date) -> None:
        _check_day(day)
        path = Path(path)
        if path.name != f"{day.isoformat()}{self.SUFFIX}":
            raise ValueError(f"{path.name} is not the log of {day.isoformat()}")
        self._path = path
        self._day = day

    def _load(self) -> _State:
        try:
            raw = self._path.read_bytes()
        except FileNotFoundError:
            raw = b""
        return _parse(raw, self._path, self.KEY_FIELDS)

    def _refusal(self, record: Any) -> str | None:
        """Why ``record`` is refused, or None."""
        return _refusal(record)

    def _status(self, state: _State) -> str:
        """Fails closed: holdout when the rule says so now or any record was logged as holdout."""
        return HOLDOUT if state.held or day_status(self._day) == HOLDOUT else DEVELOPMENT

    def _write(self, payload: bytes, verified_size: int) -> None:
        """One append-only write onto exactly the bytes verified, flushed and fsynced."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "ab") as fh:
            if os.fstat(fh.fileno()).st_size != verified_size:
                raise ShadowLogTampered(f"{self._path} changed after it was verified")
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())

    def append(self, records: list[dict]) -> int:
        """Append the whole batch, or raise RecordRefused and write nothing. Returns the count."""
        batch = list(records)
        state = self._load()
        if state.sealed:
            raise LogSealed(f"{self._path} is sealed")
        if not batch:
            return 0
        status = self._status(state)
        seen = set(state.keys)
        seq, prev = state.next_seq, state.last_sha
        lines = []
        for i, record in enumerate(batch, 1):
            reason = self._refusal(record)
            if reason is None:
                key = tuple(record[name] for name in self.KEY_FIELDS)
                if key in seen:
                    reason = f"repeats the logged key {key}"
            if reason is None:
                obj = {"seq": seq, "prev_sha256": prev, "record": {**record, "day_status": status}}
                try:
                    line = _encode(obj)
                    _decode(line)  # it must read back as written
                except (TypeError, ValueError):
                    reason = "is not plain JSON"
            if reason is not None:
                raise RecordRefused(f"record {i} of {len(batch)} {reason}; nothing was written")
            seen.add(key)
            lines.append(line)
            seq, prev = seq + 1, _sha(line)
        self._write(b"".join(line + b"\n" for line in lines), state.size)
        return len(lines)

    def seal(self, *, sealed_utc: datetime) -> str:
        """Close the day with a seal line; returns its sha256. Later appends raise LogSealed."""
        if not isinstance(sealed_utc, datetime):
            raise TypeError(f"sealed_utc must be a datetime, got {sealed_utc!r}")
        if sealed_utc.utcoffset() is None:
            raise ValueError("sealed_utc must be timezone-aware")
        state = self._load()
        if state.sealed:
            raise LogSealed(f"{self._path} is already sealed")
        seal = {
            "n_records": len(state.records),
            "sealed_utc": sealed_utc.astimezone(timezone.utc).isoformat(),
        }
        line = _encode({"seq": state.next_seq, "prev_sha256": state.last_sha, "seal": seal})
        self._write(line + b"\n", state.size)
        return _sha(line)

    def logged_keys(self) -> set[Key]:
        """The KEY_FIELDS keys in the file ((race_id, pool, combination, tau_min) for the shadow
        log), on any day."""
        return set(self._load().keys)

    def read(self) -> list[dict]:
        """The records of a development day, in order. HoldoutSealed on a holdout day."""
        if day_status(self._day) == HOLDOUT:
            raise HoldoutSealed(f"{self._day.isoformat()} is a holdout day: not read here")
        state = self._load()
        if state.held:
            raise HoldoutSealed(f"{self._path} was logged as a holdout day: not read here")
        return state.records

    def verify(self) -> LogSummary:
        """Check the chain and return counts only."""
        state = self._load()
        by_tau = Counter(record[self.KEY_FIELDS[3]] for record in state.records)
        return LogSummary(
            n_records=len(state.records),
            by_tau=dict(sorted(by_tau.items())),
            no_board=sum(record.get("status") == NO_BOARD for record in state.records),
            sealed=state.sealed,
            day_status=self._status(state),
        )


class ShadowLog(ChainedLog):
    """One race day's shadow log. Every call reads and verifies the whole file first."""
