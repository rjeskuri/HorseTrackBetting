"""The append-only experiment registry and its committed summary (#74).

Every run, failures included, is one JSON line in ``$HORSETRACK_DATA_DIR/registry/<country>.jsonl``
(``default_registry``). A line is canonical JSON (sorted keys, no spaces, ASCII) holding
``prev``, the sha256 hex of the previous line's bytes (64 zeros on the first line), and ``seal``,
the sha256 hex of the line's canonical JSON without ``seal``. ``verify()`` therefore catches an
edited byte on any line, and a deleted, moved or replayed line before the last. The file is only
opened to read or to append, under an exclusive lock, so concurrent writers cannot fork the chain.

A run's id is the sha256 hex of its record's canonical JSON: the same run twice is refused. A
family's pre-registered spec is frozen with ``freeze_family``; the freeze is a line of its own,
not a run. ``write_summary`` writes the committed, aggregate-only view: family, E id, variant,
status and per-fold metric, value, CI and n, with the values of licensed-source runs withheld.
A record's ``notes`` (run metadata, such as why a measurement could not be made) are outside its
variant identity and the summary; a record without notes is written without the key, so its line
and id are those it had before ``notes`` existed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from horsetrack.core import env, paths
from horsetrack.core.eval.holdout import DEFAULT_COUNTRIES_DIR, LOG_NAME
from horsetrack.core.eval.reports import (
    WITHHELD_SOURCES,
    LicensedOutputInRepo,
    is_inside,
    is_withheld,
    write_report,
)

try:  # POSIX only; without it concurrent appends are not serialised
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "WITHHELD_SOURCES",
    "FamilyAlreadyFrozen",
    "FoldMetric",
    "RecordRefused",
    "Registry",
    "RegistryTampered",
    "RunRecord",
    "default_registry",
    "summary_path",
    "write_summary",
]

STATUSES = frozenset({"failed", "scored", "rejected", "adopted"})
WITHHELD_LABEL = "withheld (licensed source)"
GENESIS = "0" * 64

_HYPOTHESIS_ID = re.compile(r"[ELP][0-9]+")
_COUNTRY = re.compile(r"[a-z][a-z0-9_]*")


class RecordRefused(ValueError):
    """The record (or freeze) breaks a registry rule; nothing was written."""


class RegistryTampered(ValueError):
    """The registry file is not the chain of lines the registry wrote."""


class FamilyAlreadyFrozen(ValueError):
    """The family is frozen with another spec."""


@dataclass(frozen=True)
class FoldMetric:
    fold: str
    metric: str
    value: float
    lo: float | None
    hi: float | None
    n: int


@dataclass(frozen=True)
class RunRecord:
    hypothesis_id: str
    family_id: str
    variant: str
    ticket: int
    git_sha: str
    data_builds: dict[str, str]
    sources: tuple[str, ...]
    fold_scheme: str
    components: dict[str, str]
    params: dict
    metrics: tuple[FoldMetric, ...]
    status: str
    holdout_open: str | None = None
    owner_signoff: int | None = None
    notes: dict[str, str] | None = None  # run metadata, outside the variant identity


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("ascii") if isinstance(text, str) else text).hexdigest()


def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record_data(record: RunRecord) -> dict:
    """The record as its line holds it: ``asdict`` without ``notes`` when they are None, so a
    record without notes has the id it had before the field existed, and an older line reads back
    with notes None."""
    data = asdict(record)
    if data["notes"] is None:
        del data["notes"]
    return data


def _record_from(data: dict) -> RunRecord:
    fields = dict(data)
    fields["sources"] = tuple(fields["sources"])
    fields["metrics"] = tuple(FoldMetric(**metric) for metric in fields["metrics"])
    return RunRecord(**fields)


def _feature_tables(builds: dict[str, str]) -> dict[str, list]:
    """Each build name's feature-table identity, the part of its value that counts toward a
    variant. ``"<identity>@<refresh>"`` splits at its last ``@`` and keeps ``[identity, True]``:
    a newer refresh of the same table is the same variant. A value with no ``@``, or with nothing
    on one side of its last ``@``, is identity in full, ``[value, False]``, so ``"X"``, ``"X@"``
    and ``"@r"`` each differ from ``"X@r"`` and from one another."""
    tables = {}
    for name, value in builds.items():
        identity, _, refresh = value.rpartition("@")
        tables[name] = [identity, True] if identity and refresh else [value, False]
    return tables


def _licensed(record: RunRecord) -> bool:
    """Fails closed: a withheld name among the sources, or naming a data build or a component,
    makes the run licensed."""
    builds = record.data_builds if isinstance(record.data_builds, dict) else {}
    components = record.components if isinstance(record.components, dict) else {}
    named = [*builds, *components, *components.values()]
    return is_withheld(record.sources) or is_withheld(tuple(str(name) for name in named))


# --------------------------------------------------------------------------- validation


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _names(mapping: Any) -> bool:
    return isinstance(mapping, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
    )


def _metric_ok(m: Any) -> bool:
    return (
        isinstance(m, FoldMetric)
        and _text(m.fold)
        and _text(m.metric)
        and _finite(m.value)
        and all(bound is None or _finite(bound) for bound in (m.lo, m.hi))
        and isinstance(m.n, int)
        and not isinstance(m.n, bool)
        and m.n >= 0
    )


def _problem(r: RunRecord) -> str | None:
    """Why the record breaks a rule that needs no file, or None."""
    if not (isinstance(r.hypothesis_id, str) and _HYPOTHESIS_ID.fullmatch(r.hypothesis_id)):
        return f"hypothesis_id must be E, L or P followed by digits, got {r.hypothesis_id!r}"
    for name in ("family_id", "variant", "git_sha", "fold_scheme"):
        if not _text(getattr(r, name)):
            return f"{name} must be a non-blank string, got {getattr(r, name)!r}"
    if not _positive_int(r.ticket):
        return f"ticket must be a positive issue number, got {r.ticket!r}"
    if not (isinstance(r.sources, tuple) and all(_text(s) for s in r.sources)):
        return f"sources must be a tuple of source names, got {r.sources!r}"
    if not (_names(r.data_builds) and _names(r.components)):
        return "data_builds and components must map names to strings"
    if not isinstance(r.params, dict):
        return f"params must be a dict, got {type(r.params).__name__}"
    if not (isinstance(r.metrics, tuple) and all(_metric_ok(m) for m in r.metrics)):
        return "metrics must be a tuple of FoldMetric with names, finite values and n >= 0"
    if r.status not in STATUSES:
        return f"status must be one of {sorted(STATUSES)}, got {r.status!r}"
    if r.owner_signoff is not None and not _positive_int(r.owner_signoff):
        return f"owner_signoff must be a positive ticket number, got {r.owner_signoff!r}"
    if r.status == "adopted" and r.owner_signoff is None:
        return "an adopted run needs owner_signoff, the ticket where the owner signed adoption"
    if r.holdout_open is not None and not isinstance(r.holdout_open, str):
        return f"holdout_open must be a line hash (str), got {r.holdout_open!r}"
    if r.notes is not None and not _names(r.notes):
        return f"notes must be None or map names to strings, got {r.notes!r}"
    try:
        if _record_from(json.loads(_canonical(_record_data(r)))) != r:
            return "the record would not read back unchanged (tuples or non-str keys in params?)"
    except (TypeError, ValueError) as exc:
        return f"the record is not plain JSON: {exc}"
    return None


# --------------------------------------------------------------------------- the chain


@dataclass
class _Chain:
    head: str = GENESIS
    runs: list[RunRecord] = field(default_factory=list)
    run_ids: set[str] = field(default_factory=set)
    freezes: dict[str, str] = field(default_factory=dict)


def _open_line(raw: bytes, number: int) -> dict:
    """The line's entry (without its seal) when the line is byte-for-byte as written."""
    try:
        entry = json.loads(raw.decode("ascii"))
        if not isinstance(entry, dict) or _canonical(entry).encode("ascii") != raw:
            raise ValueError("not canonical")
        seal = entry.pop("seal")
        if seal != _sha256(_canonical(entry)):
            raise ValueError("seal mismatch")
        return entry
    except (UnicodeDecodeError, ValueError, KeyError, TypeError, RecursionError) as exc:
        raise RegistryTampered(f"line {number} was changed after it was written") from exc


def _parse(data: bytes) -> _Chain:
    chain = _Chain()
    if not data:
        return chain
    if not data.endswith(b"\n"):
        raise RegistryTampered("the last line is incomplete or was edited (no final newline)")
    for number, raw in enumerate(data[:-1].split(b"\n"), start=1):
        entry = _open_line(raw, number)
        if entry.get("prev") != chain.head:
            raise RegistryTampered(
                f"line {number} does not follow the line before it: a line was deleted, moved "
                "or replayed"
            )
        try:
            if entry["kind"] == "run":
                run_id, record = entry["run_id"], _record_from(entry["record"])
                if run_id != _sha256(_canonical(entry["record"])) or run_id in chain.run_ids:
                    raise ValueError("run id")
                chain.runs.append(record)
                chain.run_ids.add(run_id)
            elif entry["kind"] == "freeze":
                family, spec_hash = entry["family_id"], entry["spec_hash"]
                if spec_hash != _spec_hash(entry["spec"]) or family in chain.freezes:
                    raise ValueError("freeze")
                chain.freezes[family] = spec_hash
            else:
                raise ValueError("kind")
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryTampered(f"line {number} is not a registry entry") from exc
        chain.head = _sha256(raw)
    return chain


class Registry:
    """One country's run registry: append-only, hash-chained JSON lines."""

    def __init__(self, path: Path | str, *, holdout_log: Path | str) -> None:
        self._path = Path(path)
        self._holdout_log = Path(holdout_log)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def holdout_log(self) -> Path:
        return self._holdout_log

    def append(self, record: RunRecord) -> str:
        """Add one run and return its id; ``RecordRefused`` (nothing written) when it breaks a
        rule or is already in the registry."""
        if not isinstance(record, RunRecord):
            raise TypeError(f"append takes a RunRecord, got {type(record).__name__}")
        problem = _problem(record)
        if problem is None and record.holdout_open is not None:
            if record.holdout_open not in self._holdout_line_hashes():
                problem = (
                    f"holdout_open {record.holdout_open!r} is not the sha256 of a line of "
                    f"{self._holdout_log}"
                )
        if problem is not None:
            raise RecordRefused(problem)
        if _licensed(record) and is_inside(self._path, env.repo_root()):
            raise LicensedOutputInRepo(f"the registry {self._path} is inside the repository")

        data = _record_data(record)
        run_id = _sha256(_canonical(data))

        def entry(chain: _Chain) -> dict:
            if run_id in chain.run_ids:
                raise RecordRefused(f"run {run_id} is already in the registry")
            return {"kind": "run", "run_id": run_id, "record": data}

        self._append(entry)
        return run_id

    def records(self, family_id: str | None = None) -> list[RunRecord]:
        """Every run in append order, or only the runs of ``family_id``."""
        runs = self._read().runs
        return runs if family_id is None else [r for r in runs if r.family_id == family_id]

    def family_size(self, family_id: str) -> int:
        """How many variants the family has tried: its distinct run specifications
        ``(hypothesis_id, variant, fold_scheme, sources, components, params)`` plus the
        feature-table identity of each data build (``_feature_tables``), compared as canonical
        JSON so mapping key order does not matter. Each counts once whatever its status or number
        of runs, failed ones included; a change to params, components, fold_scheme, sources or a
        build's identity, or an added, removed or renamed build, is a new variant even under the
        same label. A later decision, retry or rerun (another status, owner_signoff, metrics,
        holdout_open, notes, git_sha or ticket, or a refresh of the same builds) is another run in
        ``records()``, not another variant.
        """
        return len(
            {
                _canonical(
                    [
                        r.hypothesis_id,
                        r.variant,
                        r.fold_scheme,
                        r.sources,
                        r.components,
                        r.params,
                        _feature_tables(r.data_builds),
                    ]
                )
                for r in self.records(family_id)
            }
        )

    def freeze_family(self, family_id: str, spec: dict) -> str:
        """Freeze the family's pre-registered spec and return its hash (sha256 hex of the spec's
        sorted, compact JSON). Freezing again with an equal spec is a no-op; another spec raises
        ``FamilyAlreadyFrozen``."""
        if not _text(family_id):
            raise RecordRefused(f"family_id must be a non-blank string, got {family_id!r}")
        if not isinstance(spec, dict):
            raise TypeError(f"spec must be a dict, got {type(spec).__name__}")
        try:
            spec_hash = _spec_hash(spec)
            stored = json.loads(_canonical(spec))
        except (TypeError, ValueError) as exc:
            raise RecordRefused(f"the spec is not plain JSON: {exc}") from exc
        if _spec_hash(stored) != spec_hash:
            raise RecordRefused("the spec would not read back to the same hash")

        def entry(chain: _Chain) -> dict | None:
            frozen = chain.freezes.get(family_id)
            if frozen == spec_hash:
                return None
            if frozen is not None:
                raise FamilyAlreadyFrozen(f"family {family_id!r} is frozen with spec {frozen}")
            return {
                "kind": "freeze",
                "family_id": family_id,
                "spec": stored,
                "spec_hash": spec_hash,
            }

        self._append(entry)
        return spec_hash

    def frozen_spec_hash(self, family_id: str) -> str | None:
        return self._read().freezes.get(family_id)

    def verify(self) -> None:
        """Raise ``RegistryTampered`` unless the file is the chain the registry wrote."""
        self._read()

    def _holdout_line_hashes(self) -> set[str]:
        try:
            data = self._holdout_log.read_bytes()
        except FileNotFoundError:
            return set()
        return {_sha256(line) for line in data.split(b"\n") if line}

    def _read(self) -> _Chain:
        try:
            handle = open(self._path, "rb")
        except FileNotFoundError:
            return _Chain()
        with handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_SH)
            return _parse(handle.read())

    def _append(self, make_entry: Callable[[_Chain], dict | None]) -> None:
        """Under an exclusive lock: verify the chain, then append ``make_entry``'s entry (when it
        returns one) as one sealed line linked to the last."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            chain = _parse(handle.read())
            entry = make_entry(chain)
            if entry is None:
                return
            entry = {**entry, "prev": chain.head, "utc": datetime.now(timezone.utc).isoformat()}
            line = _canonical({**entry, "seal": _sha256(_canonical(entry))})
            handle.write(line.encode("ascii") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def default_registry(country: str) -> Registry:
    """The registry of ``country`` under the data directory, linked to its #71 holdout log."""
    if not (isinstance(country, str) and _COUNTRY.fullmatch(country)):
        raise ValueError(f"not a country name: {country!r}")
    if not (DEFAULT_COUNTRIES_DIR / country).is_dir():
        raise ValueError(f"no country folder for {country!r} under {DEFAULT_COUNTRIES_DIR}")
    path = paths.data_dir() / "registry" / f"{country}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return Registry(path, holdout_log=DEFAULT_COUNTRIES_DIR / country / LOG_NAME)


def summary_path(country: str) -> Path:
    """Where the committed summary of ``country``'s registry lives."""
    return env.repo_root() / "docs" / "experiments" / f"registry-{country}.md"


# --------------------------------------------------------------------------- summary


def _cell(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def write_summary(registry: Registry, path: Path | str) -> None:
    """Write the aggregate-only summary of ``registry`` to ``path`` (normally
    ``summary_path(country)``): per run its family, E id, variant, status and per-fold metric,
    value, CI and n, and nothing else. A licensed-source run shows only its identity and
    ``withheld (licensed source)``."""
    families: dict[str, list[RunRecord]] = {}
    for record in registry.records():
        families.setdefault(record.family_id, []).append(record)

    lines = [
        "# Experiment registry summary",
        "",
        "Written by `horsetrack.core.eval.registry.write_summary` from the run registry under",
        "the data directory; do not edit by hand. Every run is listed, failures included. A run",
        "built on a licensed source shows no values here.",
    ]
    for family, records in families.items():
        lines += [
            "",
            f"## {_cell(family)}",
            "",
            "| E id | Variant | Status | Fold | Metric | Value | CI | n |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in records:
            who = f"| {_cell(r.hypothesis_id)} | {_cell(r.variant)} | {_cell(r.status)} |"
            if _licensed(r):
                lines.append(f"{who} {WITHHELD_LABEL} | | | | |")
                continue
            if not r.metrics:
                lines.append(f"{who} - | - | - | - | - |")
            for m in r.metrics:
                ci = "-" if m.lo is None and m.hi is None else f"[{_number(m.lo)}, {_number(m.hi)}]"
                values = f"{_number(m.value)} | {ci} | {m.n}"
                lines.append(f"{who} {_cell(m.fold)} | {_cell(m.metric)} | {values} |")
    write_report(Path(path), "\n".join(lines) + "\n", sources=())
