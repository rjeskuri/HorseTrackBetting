"""#116's drift nowcast refitted on the JRA-VAN odds history (#500): `horsetrack japan nowcast-fit
--from-store`.

Pairs: the (low figure at t, final low figure) pairs of every development day of
`processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet`, built exactly as `studies/nowcast.py`
builds its `drift_v0` examples (`read_day`, `models.nowcast.extract`, `drift_examples`), one day
at a time and with column-scoped reads, each day's frame dropped once its pairs are built. D10:
days from `PRICE_ERA_START`. D14: holdout days are dropped by file name, before any read.
nowcast_v0.yaml's `max_board_age_s` is not applied (it was set for a ~30 s logger; the JRA-VAN
paths are 5-minute), and the manifest says so.

Fit: `NowcastV0.fit` on the WIN and PLACE pairs with nowcast_v0.yaml's bands and prior_n and the
shipped decision grid; a day with no WIN or PLACE pair (a quinella-only file) is not one of the
fit's days, and a pool with no pair at some grid tau (a partial backfill) refuses the fit. The
JRA-VAN series has no Wide pool, so the QUINELLA_PLACE cells and sigma0 are copied unchanged from
our logged-days nowcast fit: the one `current.json` names or, when it names a fit of this command,
the logged fit that fit's manifest names. A fit of this command is told apart by its manifest's
schema (`pool_sources`, `wide_fit_file`), not by having one: the logged fitter (`run_fit`) writes a
manifest too. Without such a v0 fit the command refuses and writes nothing. The merged fit's id
is the sha256 of its JSON, as `NowcastV0.fit` names a fit.

Provenance: the data build is a digest of the selected development day files (name, size,
content), so it does not depend on where HORSETRACK_DATA_DIR is and never reads a holdout file or
the races table.

Outputs (JRA-VAN-derived, D21, through `write_report`, which refuses the repository):
`$HORSETRACK_DATA_DIR/models/japan/v0/nowcast-<fit_id>.json` and `manifest-nowcast-<fit_id>.json`;
then `current.json`'s "nowcast" entry names the new file (every other entry and the old fit file
kept). One registry record per fit (family `japan-v0-nowcast-drift`), a failed fit too (an
unreadable day file included, keyed by the files' names, sizes and mtimes since they could not be
hashed), and one success record per (fit_id, data build, Wide source): a new data build, or a
corrected logged fit with the same Wide cells, that leaves the fit unchanged appends a record of
its own, which the manifest then names, while an unchanged rerun, or a retry after a crash
between the record and the pointer, completes the writes without a second record. A rerun of the fit
the pointer already names keeps the first run's Wide source and rollback target
(`previous_nowcast`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from pandas.api.types import union_categoricals

from horsetrack.core import config
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import (
    RecordRefused,
    Registry,
    RunRecord,
    _canonical,
    _record_data,
    _sha256,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.board.snapshot import load_decision_grid
from horsetrack.countries.japan.models import nowcast as nc
from horsetrack.countries.japan.models.v0.loader import current_pointer
from horsetrack.countries.japan.models.v0.nowcast import CONFIG_STEM, PLACE, WIDE, WIN, NowcastV0
from horsetrack.countries.japan.models.v0.ordering import _git_sha
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import nowcast as study

COUNTRY = "japan"
TICKET = 500
SOURCE = study.SOURCE  # "jravan"
HYPOTHESIS_ID = "E19"  # the nowcast hypothesis under which the study scores this drift
FAMILY_ID = "japan-v0-nowcast-drift"
HISTORY_POOLS = (WIN, PLACE)
POOL_SOURCES = {WIN: SOURCE, PLACE: SOURCE, WIDE: "logged"}


class NoWideSource(RuntimeError):
    """The pointer names no v0 nowcast fit with Wide cells to carry over."""


class MissingHorizons(ValueError):
    """WIN or PLACE has no pairs at some tau of the decision grid."""


@dataclasses.dataclass(frozen=True)
class WideSource:
    """The logged-days fit the Wide cells come from, and the pointer as it was before the run."""

    names: dict  # current.json's entries
    fit: NowcastV0  # the logged fit, which holds the Wide cells
    fit_name: str  # its file, as current.json (or the pointed fit's manifest) names it
    pointed_id: str  # the fit current.json names now: the logged fit or one of ours
    pointed_manifest: dict | None  # that fit's manifest when it is one of ours


def development_files() -> tuple[list[tuple[str, Path]], int]:
    """(day, path) of every JRA-VAN day from PRICE_ERA_START that is not a holdout day, and how
    many holdout days were skipped. Only file names are looked at here."""
    files = study.day_files(PRICE_ERA_START, None)
    keep = [(d, p) for d, p in files if not is_holdout_day(COUNTRY, date.fromisoformat(d))]
    return keep, len(files) - len(keep)


def data_builds(files: list[tuple[str, Path]]) -> dict[str, str]:
    """A digest of the selected development day files, each day's name, size and content: a
    change to one of them changes it, and nothing else does (holdout or pre-era files, the races
    table, where the data dir is)."""
    digest = hashlib.sha256()
    for day, path in files:
        content = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                content.update(chunk)
        digest.update(f"{day}:{path.stat().st_size}:{content.hexdigest()}\n".encode())
    build = f"processed/{SOURCE}/odds_timeseries@{digest.hexdigest()[:16]}"
    return {"jravan_odds_timeseries": f"{build} ({len(files)} development days)"}


def unhashed_builds(files: list[tuple[str, Path]]) -> dict[str, str]:
    """The data build of a run whose files were not hashed: a digest of each selected day's name,
    size and mtime (stat works on an unreadable file too), so the same failure on a refreshed store
    is a new run and on unchanged files the same one. A file gone since discovery digests as
    `missing`, so the run still reaches its failed-record handler."""
    digest = hashlib.sha256()
    for day, path in files:
        try:
            st = path.stat()
        except OSError:
            digest.update(f"{day}:missing\n".encode())
            continue
        digest.update(f"{day}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    build = f"unhashed@{digest.hexdigest()[:16]}"
    return {"jravan_odds_timeseries": f"{build} ({len(files)} development days)"}


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """The days' examples in one frame, `day` and `pool` kept categorical. Days with no pairs are
    left out: their empty categoricals have object categories, which pandas 3 will not union with
    the others' `str` categories."""
    parts = [p for p in parts if len(p)]
    if not parts:
        raise ValueError(f"no drift pairs on any {SOURCE} development day")
    return pd.DataFrame(
        {
            "day": union_categoricals([p["day"] for p in parts]),
            "pool": union_categoricals([p["pool"] for p in parts]),
            "tau_min": np.concatenate([p["tau_min"].to_numpy() for p in parts]),
            "odds_t": np.concatenate([p["odds_t"].to_numpy() for p in parts]),
            "odds_final": np.concatenate([p["odds_final"].to_numpy() for p in parts]),
        }
    )


def store_examples(files: list[tuple[str, Path]] | None = None) -> pd.DataFrame:
    """The study's drift_v0 pairs of every development day (default: `development_files()`)."""
    files = development_files()[0] if files is None else files
    if not files:
        raise ValueError(f"no {SOURCE} development days in the store")
    parts = []
    for day, path in files:
        frame = study.read_day(path)
        cases, _, _ = nc.extract(frame)
        del frame
        parts.append(study.drift_examples(cases, [day]))
        del cases
    return _concat(parts)


def _manifest_path(folder: Path, fit_id: str) -> Path:
    return folder / f"manifest-nowcast-{fit_id}.json"


def _is_history_manifest(manifest: object) -> bool:
    """Whether a manifest is one this command writes; the logged fitter's has neither key."""
    return isinstance(manifest, dict) and (
        "pool_sources" in manifest or "wide_fit_file" in manifest
    )


def _load_v0(folder: Path, name: str, what: str) -> NowcastV0:
    path = Path(name) if Path(name).is_absolute() else folder / name
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise NoWideSource(f"the {what} {path} is missing") from exc
    if json.loads(text).get("model") != "nowcast_v0":
        raise NoWideSource(f"the {what} {path} is not a v0 drift fit; no Wide cells")
    return NowcastV0.from_json(text)


def _wide_from_current(pointer: Path) -> WideSource:
    """The pointer's names and the logged v0 fit to take Wide from: the fit the pointer names or,
    when that fit is one of ours (its manifest has this command's schema), the logged fit its
    manifest names. The logged fit must hold Wide cells."""
    try:
        names = json.loads(pointer.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = f"no pointer at {pointer}: no current nowcast fit to take Wide from"
        raise NoWideSource(msg) from exc
    name = names.get("nowcast") if isinstance(names, dict) else None
    if name is None:
        raise NoWideSource(f"{pointer} names no nowcast fit to take Wide from")
    folder = pointer.parent
    pointed = _load_v0(folder, name, "current nowcast fit")
    manifest_path = _manifest_path(folder, pointed.fit_id)
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _is_history_manifest(manifest):
            manifest = None  # the logged fitter's own manifest: the pointed fit is the logged fit
    if manifest is not None:  # a fit of this command: its manifest names the logged fit
        wide_name, wide_id = manifest.get("wide_fit_file"), manifest.get("wide_fit_id")
        if not wide_name or not wide_id:
            raise NoWideSource(f"{manifest_path} names no logged fit to take Wide from")
        logged = _load_v0(folder, wide_name, "logged nowcast fit")
        if logged.fit_id != wide_id:
            msg = f"{wide_name} holds fit {logged.fit_id}, not {wide_id} as {manifest_path} says"
            raise NoWideSource(msg)
    else:
        logged, wide_name = pointed, name
    if not any(key[0] == WIDE for key in logged.cells):
        raise NoWideSource(f"the current nowcast fit {logged.fit_id} has no {WIDE} cells")
    return WideSource(names, logged, wide_name, pointed.fit_id, manifest)


def _with_wide(history: NowcastV0, wide: NowcastV0) -> NowcastV0:
    """`history`'s WIN and PLACE cells with `wide`'s Wide cells and sigma0, renamed by its JSON."""
    cells = {k: c for k, c in history.cells.items() if k[0] != WIDE}
    cells.update({k: c for k, c in wide.cells.items() if k[0] == WIDE})
    sigma0 = {k: s for k, s in history.sigma0.items() if k[0] != WIDE}
    sigma0.update({k: s for k, s in wide.sigma0.items() if k[0] == WIDE})
    days = tuple(sorted({*history.days, *wide.days}))
    merged = dataclasses.replace(history, days=days, sigma0=sigma0, cells=cells, fit_id="")
    digest = hashlib.sha256(merged.to_json().encode()).hexdigest()[:12]
    return dataclasses.replace(merged, fit_id=digest)


def _missing_horizons(examples: pd.DataFrame, grid) -> dict[str, list[int]]:
    """The grid taus at which WIN or PLACE has no pair, by pool."""
    missing = {}
    for pool in HISTORY_POOLS:
        present = set(examples.loc[examples["pool"] == pool, "tau_min"].astype(int).tolist())
        gaps = [int(t) for t in grid.minutes if int(t) not in present]
        if gaps:
            missing[pool] = gaps
    return missing


def _record(*, status: str, params: dict, builds: dict, git_sha: str, wide_fit_id: str):
    return RunRecord(
        hypothesis_id=HYPOTHESIS_ID,
        family_id=FAMILY_ID,
        variant="jravan-history-wide-logged",
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme="none (one fit on every development day)",
        components={"drift": "models.v0.nowcast.NowcastV0", "wide_fit": wide_fit_id},
        params=params,
        metrics=(),
        status=status,
    )


def _success_run_id(registry: Registry, fit_id: str, builds: dict, wide_fit_id: str) -> str | None:
    """The run id of `fit_id`'s success record on the data build `builds` with Wide taken from the
    logged fit `wide_fit_id`, when the registry holds one already."""
    for record in registry.records(FAMILY_ID):
        if (
            record.status != "failed"
            and record.params.get("fit_id") == fit_id
            and record.data_builds == builds
            and record.params.get("wide_fit_id") == wide_fit_id
        ):
            return _sha256(_canonical(_record_data(record)))  # as Registry.append names a run
    return None


def _write_pointer(pointer: Path, names: dict) -> None:
    tmp = pointer.with_name(f".{pointer.name}.tmp")
    tmp.write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, pointer)


def fit_from_store() -> dict:
    """Fit, write the fit, record the run, write its manifest and point current.json at the fit;
    returns the manifest. Raises NoWideSource, before any day is read, without a Wide source, and
    MissingHorizons (after a failed record, nothing else written) when WIN or PLACE has no pair at
    a grid tau. Rerunning on unchanged inputs, or after a crash part way, finishes the writes and
    appends no second success record."""
    pointer = current_pointer()
    source = _wide_from_current(pointer)
    wide = source.fit
    cfg = config.load_country_yaml(COUNTRY, CONFIG_STEM)
    bands = tuple(float(b) for b in cfg["bands"])
    prior_n = float(cfg["prior_n"])
    grid = load_decision_grid()
    files, held = development_files()
    # until the files are hashed, inside the try below, so an unreadable one is a failed run
    builds = unhashed_builds(files)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    params = {
        "jravan_days": [
            d for d, _ in files
        ],  # the selected days, until the pairs say which gave any
        "pool_sources": POOL_SOURCES,
        "wide_fit_id": wide.fit_id,
        "bands": list(bands),
        "prior_n": prior_n,
        "grid": list(grid.minutes),
        "max_board_age_s_applied": False,
    }

    try:
        builds = data_builds(files)
        examples = store_examples(files)
        examples = examples[examples["pool"].isin(HISTORY_POOLS)]
        missing = _missing_horizons(examples, grid)
        if missing:
            gaps = "; ".join(f"{p} at tau {', '.join(map(str, t))} min" for p, t in missing.items())
            raise MissingHorizons(f"no {SOURCE} pairs on the decision grid for {gaps}")
        history = NowcastV0.fit(examples, bands=bands, grid=grid, prior_n=prior_n)
    except Exception as exc:
        failed = {**params, "fit_id": "", "error": f"{type(exc).__name__}: {exc}"}
        record = _record(
            status="failed", params=failed, builds=builds, git_sha=git_sha, wide_fit_id=wide.fit_id
        )
        try:
            registry.append(record)
        except RecordRefused:
            pass  # the same failed run, on the same inputs and code, is in the registry already
        write_summary(registry, summary_dir(COUNTRY))
        raise
    days = list(history.days)  # the development days that gave a WIN or PLACE pair
    params["jravan_days"] = days
    model = _with_wide(history, wide)
    cells_per_pool = {pool: sum(k[0] == pool for k in model.cells) for pool in (WIN, PLACE, WIDE)}
    examples_per_pool = {pool: int((examples["pool"] == pool).sum()) for pool in HISTORY_POOLS}
    del examples

    folder = pointer.parent
    fit_path = folder / f"nowcast-{model.fit_id}.json"
    manifest_path = _manifest_path(folder, model.fit_id)
    if model.fit_id == source.pointed_id and source.pointed_manifest is not None:
        first = source.pointed_manifest  # an unchanged rerun: keep the first run's provenance
        previous, created = first.get("previous_nowcast"), first.get("created_utc")
    else:
        previous, created = source.names.get("nowcast"), None
    write_report(fit_path, model.to_json(), sources=(SOURCE,))
    run_id = _success_run_id(registry, model.fit_id, builds, wide.fit_id)
    if run_id is None:
        run_id = registry.append(
            _record(
                status="scored",
                params={
                    **params,
                    "fit_id": model.fit_id,
                    "cells_per_pool": cells_per_pool,
                    "examples_per_pool": examples_per_pool,
                },
                builds=builds,
                git_sha=git_sha,
                wide_fit_id=wide.fit_id,
            )
        )
    write_summary(registry, summary_dir(COUNTRY))
    manifest = {
        "fit_id": model.fit_id,
        "fit_path": str(fit_path),
        "manifest_path": str(manifest_path),
        "jravan_days": days,
        "holdout_days_skipped": held,
        "pool_sources": POOL_SOURCES,
        "wide_fit_id": wide.fit_id,
        "wide_fit_file": source.fit_name,
        "max_board_age_s_applied": False,
        "config": {"bands": list(bands), "prior_n": prior_n},
        "grid": list(grid.minutes),
        "days": list(model.days),
        "cells_per_pool": cells_per_pool,
        "examples_per_pool": examples_per_pool,
        "examples_per_cell": [
            {"pool": c.pool, "tau_min": c.tau_min, "band": c.band, "n": c.n}
            for c in model.cells.values()
        ],
        "data_builds": builds,
        "previous_nowcast": previous,
        "pointer": str(pointer),
        "run_id": run_id,
        "git_sha": git_sha,
        "created_utc": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_report(manifest_path, json.dumps(manifest, indent=2) + "\n", sources=(SOURCE,))
    _write_pointer(pointer, {**source.names, "nowcast": fit_path.name})
    return manifest


def nowcast_fit_command(
    from_store: Annotated[
        bool,
        typer.Option(
            "--from-store",
            help=f"Refit on the {SOURCE} odds history (Wide kept from the current fit) and point "
            "current.json at it.",
        ),
    ] = False,
) -> None:
    """Refit #116's drift nowcast on the JRA-VAN odds history and point the shadow at it."""
    if not from_store:
        typer.echo(
            "only --from-store is supported here; the logged-days fit is "
            "`python -m horsetrack.countries.japan.models.v0.nowcast --fit`",
            err=True,
        )
        raise typer.Exit(2)
    try:
        manifest = fit_from_store()
    except (NoWideSource, MissingHorizons) as exc:
        typer.echo(f"refused: {exc}; nothing written to the models folder", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"nowcast fit {manifest['fit_id']} on {len(manifest['jravan_days'])} {SOURCE} days "
        f"(Wide from {manifest['wide_fit_id']}): {manifest['fit_path']}"
    )
