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
shipped decision grid. The JRA-VAN series has no Wide pool, so the QUINELLA_PLACE cells and sigma0
are copied unchanged from the nowcast fit `current.json` names (our logged days); without such a
v0 fit the command refuses and writes nothing. The merged fit's id is the sha256 of its JSON, as
`NowcastV0.fit` names a fit.

Outputs (JRA-VAN-derived, D21, through `write_report`, which refuses the repository):
`$HORSETRACK_DATA_DIR/models/japan/v0/nowcast-<fit_id>.json` and `manifest-nowcast-<fit_id>.json`;
then `current.json`'s "nowcast" entry names the new file (every other entry and the old fit file
kept). One registry record per fit (family `japan-v0-nowcast-drift`), a failed fit too.
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
from horsetrack.core.eval.registry import RunRecord, default_registry, summary_dir, write_summary
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


def development_files() -> tuple[list[tuple[str, Path]], int]:
    """(day, path) of every JRA-VAN day from PRICE_ERA_START that is not a holdout day, and how
    many holdout days were skipped. Only file names are looked at here."""
    files = study.day_files(PRICE_ERA_START, None)
    keep = [(d, p) for d, p in files if not is_holdout_day(COUNTRY, date.fromisoformat(d))]
    return keep, len(files) - len(keep)


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """The days' examples in one frame, `day` and `pool` kept categorical."""
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


def _wide_from_current(pointer: Path) -> tuple[dict, NowcastV0]:
    """The pointer's names and the v0 nowcast fit it names, which must hold Wide cells."""
    try:
        names = json.loads(pointer.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = f"no pointer at {pointer}: no current nowcast fit to take Wide from"
        raise NoWideSource(msg) from exc
    name = names.get("nowcast") if isinstance(names, dict) else None
    if name is None:
        raise NoWideSource(f"{pointer} names no nowcast fit to take Wide from")
    path = Path(name) if Path(name).is_absolute() else pointer.parent / name
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise NoWideSource(f"the current nowcast fit {path} is missing") from exc
    if json.loads(text).get("model") != "nowcast_v0":
        raise NoWideSource(f"the current nowcast fit {path} is not a v0 drift fit; no Wide cells")
    current = NowcastV0.from_json(text)
    if not any(key[0] == WIDE for key in current.cells):
        raise NoWideSource(f"the current nowcast fit {current.fit_id} has no {WIDE} cells")
    return names, current


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


def _write_pointer(pointer: Path, names: dict) -> None:
    tmp = pointer.with_name(f".{pointer.name}.tmp")
    tmp.write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, pointer)


def fit_from_store() -> dict:
    """Fit, write the fit and its manifest, point current.json at the fit, record the run;
    returns the manifest. Raises NoWideSource, before any day is read, without a Wide source."""
    pointer = current_pointer()
    names, wide = _wide_from_current(pointer)
    cfg = config.load_country_yaml(COUNTRY, CONFIG_STEM)
    bands = tuple(float(b) for b in cfg["bands"])
    prior_n = float(cfg["prior_n"])
    grid = load_decision_grid()
    files, held = development_files()
    days = [d for d, _ in files]
    builds = study.data_builds(files)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    params = {
        "jravan_days": days,
        "pool_sources": POOL_SOURCES,
        "wide_fit_id": wide.fit_id,
        "bands": list(bands),
        "prior_n": prior_n,
        "grid": list(grid.minutes),
        "max_board_age_s_applied": False,
    }

    try:
        examples = store_examples(files)
        examples = examples[examples["pool"].isin(HISTORY_POOLS)]
        history = NowcastV0.fit(examples, bands=bands, grid=grid, prior_n=prior_n)
    except Exception as exc:
        failed = {**params, "fit_id": "", "error": f"{type(exc).__name__}: {exc}"}
        registry.append(
            _record(
                status="failed", params=failed, builds=builds, git_sha=git_sha,
                wide_fit_id=wide.fit_id,
            )
        )  # fmt: skip
        write_summary(registry, summary_dir(COUNTRY))
        raise
    model = _with_wide(history, wide)
    cells_per_pool = {pool: sum(k[0] == pool for k in model.cells) for pool in (WIN, PLACE, WIDE)}
    examples_per_pool = {pool: int((examples["pool"] == pool).sum()) for pool in HISTORY_POOLS}
    del examples

    folder = pointer.parent
    fit_path = folder / f"nowcast-{model.fit_id}.json"
    manifest_path = folder / f"manifest-nowcast-{model.fit_id}.json"
    write_report(fit_path, model.to_json(), sources=(SOURCE,))
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
        "previous_nowcast": names.get("nowcast"),
        "pointer": str(pointer),
        "run_id": run_id,
        "git_sha": git_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_report(manifest_path, json.dumps(manifest, indent=2) + "\n", sources=(SOURCE,))
    _write_pointer(pointer, {**names, "nowcast": fit_path.name})
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
    except NoWideSource as exc:
        typer.echo(f"refused: {exc}; nothing written", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"nowcast fit {manifest['fit_id']} on {len(manifest['jravan_days'])} {SOURCE} days "
        f"(Wide from {manifest['wide_fit_id']}): {manifest['fit_path']}"
    )
