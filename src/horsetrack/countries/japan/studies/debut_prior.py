"""#93's trainer debut prior measured by segment at the final Win price (#573):
`horsetrack japan debut-prior`.

The `trainer_debut` member of `studies.interactions` (#93: a debut indicator and a variance
component per (trainer, debut) cell) is cut by the race's segment, `<race_class>/debut` when any
starter of the race debuts and `<race_class>/seasoned` otherwise, and each segment is scored at the
final Win price from #93's out-of-fold level-1 export: the ΔLL of the member's stage-2 meta against
the base's (`ln p_member(winner) - ln p_base(winner)`, a dead heat's k winners 1/k each, D6), with
D7's one-sided meeting bounds, and the tail return of the segment's top-claim bets at the recorded
Win dividends. The cut is frozen before any scored read (`docs/specs/debut-prior-prereg.md`).

The debut label reads prior runs only: a horse's runs strictly before the race's post time, never
the race itself, another run at the same post time (the tie rule of `models.interactions.
cell_sizes`) or a later run. It is computed on the whole level-1 history from 2010 and the scored
window is cut afterwards, so a horse's runs before the window still count. The stage-2 members are
rebuilt here from the export's level-1 predictions exactly as `studies.interactions` builds them
(D4's calibrated market out of fold, #87's stage-2 meta on D9's folds); no level-1 model is fitted.

The inputs are the JRA results store only (netkeiba results, no JRDB, no JRA-VAN), so the figures
are public. Holdout days are never read; the holdout is not opened. Nothing is adopted (D16, D19):
every segment is a ranked row, and its tail return a leaderboard row (the registry's convention).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from horsetrack.core import paths
from horsetrack.core.config import load_source
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.studies import ability as _ability
from horsetrack.countries.japan.studies import interactions as ix
from horsetrack.countries.japan.studies.calibration import (
    _markdown_table,
    _metric,
    bootstrap_settings,
    store_builds,
)
from horsetrack.countries.japan.studies.meta import (
    Scored,
    _meta_oof,
    baseline_oof,
    read_development,
    restrict,
    store_frame,
)

COUNTRY = "japan"
TICKET = 573
FAMILY_ID = "japan-debut-prior-segments-final-win"
SOURCES = ("jra",)
MEMBER = "trainer_debut"
BASE = ix.BASE
MAIDEN_CLASSES = ("DEBUT", "MAIDEN")
TOP = 0.99  # the tail: bets whose claim p x O_final is at or above this quantile of the segment's
HYPOTHESIS = "E93"
UNKNOWN = "unknown"
EXPORT = "interaction_export"
ARTIFACTS = "debut_prior_artifacts"


# --------------------------------------------------------------------------- the pure cut


def debut_labels(frame: pd.DataFrame) -> pd.Series:
    """True where the horse has no row of `frame` at a strictly earlier race_utc. It reads
    race_utc and horse_id only, never an outcome or the order of the rows."""
    first = frame.groupby("horse_id", sort=False, dropna=False)["race_utc"].transform("min")
    return (frame["race_utc"] == first).astype(bool)


def label_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Each row's `debut` (`debut_labels`), `maiden` (the race is a DEBUT or MAIDEN class) and
    `segment`, one string per race: `<race_class>/debut` when any starter debuts, else
    `<race_class>/seasoned`; a missing class reads `unknown`."""
    debut = debut_labels(frame)
    klass = frame["race_class"]
    maiden = klass.isin(MAIDEN_CLASSES).astype(bool)
    named = klass.where(klass.notna(), UNKNOWN).astype(str)
    any_debut = debut.groupby(frame["race_id"].to_numpy()).transform("any").to_numpy(dtype=bool)
    segment = named + "/" + pd.Series(np.where(any_debut, "debut", "seasoned"), index=frame.index)
    return pd.DataFrame({"debut": debut, "maiden": maiden, "segment": segment}, index=frame.index)


def _race_gains(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Each race's gain ln p_member(winner) - ln p_base(winner) (a dead heat's winners 1/k each)
    and its meeting, by race_id; a race with no winner has no gain."""
    won = rows.loc[rows["won"].to_numpy(dtype=bool)]
    gain = np.log(won["p_member"].to_numpy(dtype=float)) - np.log(
        won["p_base"].to_numpy(dtype=float)
    )
    per_race = pd.Series(gain, index=won["race_id"].to_numpy()).groupby(level=0).mean().sort_index()
    meeting = rows.groupby("race_id")["meeting_id"].first().reindex(per_race.index)
    return per_race, meeting


def score_segments(
    rows: pd.DataFrame, *, n_boot: int, level: float, seed: int, top: float = TOP
) -> pd.DataFrame:
    """One ranked row per segment (see the module's test for the columns): the mean per-race ΔLL
    with `paired_bootstrap` bounds over meetings, and the return per ¥100 of the segment's bets at
    or above the `top` quantile of the claim p_member x win_odds, read at `paid`, with the same
    bounds over meetings. Ranked by delta_ll, highest first; no segment is dropped."""
    out = []
    for segment, sub in rows.groupby("segment", sort=True):
        gain, meeting = _race_gains(sub)
        iv = paired_bootstrap(gain, meeting, n_boot=n_boot, level=level, seed=seed)
        claim = (sub["p_member"] * sub["win_odds"]).to_numpy(dtype=float)
        paid = sub["paid"].to_numpy(dtype=float)
        if np.isnan(paid).any():
            raise ValueError(f"{segment}: a bet has no recorded dividend (an unsettled race)")
        picked = claim >= np.quantile(claim, top)
        ret = pd.Series(100.0 * paid[picked])
        tail = paired_bootstrap(
            ret,
            pd.Series(sub["meeting_id"].to_numpy()[picked]),
            n_boot=n_boot,
            level=level,
            seed=seed,
        )
        out.append(
            {
                "segment": segment,
                "races": len(gain),
                "meetings": int(sub["meeting_id"].nunique()),
                "bets": len(sub),
                "delta_ll": iv.mean,
                "delta_ll_lo": iv.lo,
                "delta_ll_hi": iv.hi,
                "tail_bets": int(picked.sum()),
                "tail_return": tail.mean,
                "tail_lo": tail.lo,
                "tail_hi": tail.hi,
                "tail_meetings": tail.n_clusters,
            }
        )
    table = pd.DataFrame(out).sort_values(["delta_ll", "segment"], ascending=[False, True])
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True)


def build_spec(bootstrap: dict) -> dict:
    """The family's pre-registered spec, plain JSON, frozen before any scored read."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "member": MEMBER,
        "base": BASE,
        "bootstrap": dict(bootstrap),
        "segments": {
            "axes": ["debut", "maiden", "race_class"],
            "maiden_classes": list(MAIDEN_CLASSES),
            "label": "<race_class>/debut when any starter has no earlier run, else /seasoned",
            "history": "the whole level-1 history from 2010-01-01; the scored window cut after",
        },
        "score": {
            "delta_ll": "mean per race of ln p_member(winner) - ln p_base(winner), dead heats 1/k",
            "p": "#87's stage-2 meta of each member at the final Win price (studies.interactions)",
            "folds": "D9's level-2 folds from 2020-Q1 over #93's out-of-fold level-1 export",
        },
        "tail": {"top": TOP},
    }


# --------------------------------------------------------------------------- the real run


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def latest_export() -> Path:
    """The export of the latest scored record of #93's v2 family (its ARTIFACTS build names the
    stem), under the data dir's features."""
    records = [
        r
        for r in default_registry(COUNTRY).records(ix.FAMILY_ID)
        if r.status == "scored" and ix.ARTIFACTS in r.data_builds
    ]
    if not records:
        raise FileNotFoundError(f"no scored record of {ix.FAMILY_ID} names an export")
    stem = records[-1].data_builds[ix.ARTIFACTS].rsplit("@", 1)[1]
    path = paths.data_dir() / "features" / COUNTRY / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"#93's export {path.name} is not under the data dir's features")
    return path


def race_segments(history: pd.DataFrame) -> pd.Series:
    """Each race's segment, from the whole level-1 history."""
    labels = label_segments(history[["race_id", "race_utc", "horse_id", "race_class"]])
    return labels["segment"].groupby(history["race_id"].to_numpy()).first()


def scored_rows(export: pd.DataFrame, history: pd.DataFrame, log=print) -> pd.DataFrame:
    """The scored runners with each member's stage-2 probability at the final price: #93's
    `compute_study` level 2, from the export's level-1 predictions."""
    scheme = load_country_scheme(COUNTRY)
    runners, races = read_development()
    price, _ = store_frame(runners, races)
    del runners, races
    numbers, dividends = ix.read_win_dividends(price["race_id"].unique())
    stake = float(load_source(COUNTRY, ix.SOURCE).unit_stake)
    price["paid"], _ = ix.win_paid(price, numbers, dividends, stake)
    del numbers, dividends
    folds = walk_forward(price["race_date"], scheme=scheme)
    price = price.assign(p_cal=baseline_oof(price, folds))
    u = {m: export[f"u_{m}"].to_numpy() for m in ix.MEMBERS}
    rows, _ = ix.member_frame(price, export, u, _ability.LEVEL1_FROM)
    rfolds = restrict(folds, rows.index.to_numpy(), len(price))
    rows = rows.reset_index(drop=True)
    scored = Scored.of(rows, rfolds)
    log(f"{scored.n_races} scored races over {len(rfolds)} folds")
    sr = scored.rows
    p = {m: _meta_oof(rows, rfolds, ix.STAGE2[m], "ln_q")[0] for m in (BASE, MEMBER)}
    out = pd.DataFrame(
        {
            "race_id": sr["race_id"].to_numpy(),
            "meeting_id": sr["meeting_id"].to_numpy(),
            "won": sr["won"].to_numpy(dtype=bool),
            "win_odds": sr["win_odds"].to_numpy(dtype=float),
            "paid": sr["paid"].to_numpy(dtype=float),
        },
        index=sr.index,
    )
    for column, member in (("p_member", MEMBER), ("p_base", BASE)):
        v = p[member].reindex(sr.index)
        out[column] = (v / v.groupby(out["race_id"]).transform("sum")).to_numpy()
    segment = race_segments(history)
    out["segment"] = out["race_id"].map(segment)
    missing = out["segment"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} scored runners are in a race with no segment")
    if out[["p_member", "p_base"]].isna().any().any():
        raise ValueError("a scored runner has no stage-2 probability")
    return out.reset_index(drop=True)


def _records(table: pd.DataFrame, boot: dict, builds: dict, git_sha: str, spec_hash: str):
    """One registry record per segment; its development metrics follow the leaderboard's
    convention (return_per_100 with its bounds, n_meetings)."""
    fold_scheme = "walk_forward:D9 level-2 folds from 2020-Q1"
    out = []
    for r in table.itertuples(index=False):
        metrics = (
            _metric("development", "delta_ll", r.delta_ll, r.races, r.delta_ll_lo, r.delta_ll_hi),
            _metric(
                "development", "return_per_100", r.tail_return, r.tail_bets, r.tail_lo, r.tail_hi
            ),
            _metric("development", "n_meetings", r.tail_meetings, r.tail_meetings),
        )
        out.append(
            RunRecord(
                hypothesis_id=HYPOTHESIS,
                family_id=FAMILY_ID,
                variant=f"segment:{r.segment}",
                ticket=TICKET,
                git_sha=git_sha,
                data_builds=builds,
                sources=SOURCES,
                fold_scheme=fold_scheme,
                components={
                    "model": "models/interactions.py",
                    "study": "studies/debut_prior.py",
                    "member": MEMBER,
                    "comparator": f"stage2:q:{BASE}",
                    "score": "paired_winner_log_loss",
                    "segment": str(r.segment),
                },
                params={
                    "spec_sha256": spec_hash,
                    "rank": int(r.rank),
                    "races": int(r.races),
                    "bets": int(r.bets),
                    "tail_top": TOP,
                    "bootstrap": boot,
                    "leaderboard": {
                        "kind": "member",
                        "pool": "WIN",
                        "rule": (
                            f"trainer_debut stage-2 bets at the top {1 - TOP:.0%} of claim "
                            f"p x O_final in {r.segment}, flat unit stakes"
                        ),
                        "selection": "look_ahead",
                        "payout_rate": None,
                        "ci_level": boot["level"],
                        "n_unit": "bets",
                    },
                },
                metrics=metrics,
                status="scored",
            )
        )
    return out


def render_report(table: pd.DataFrame, spec: dict, about: dict) -> str:
    shown = table.rename(
        columns={
            "delta_ll": "ΔLL",
            "delta_ll_lo": "ΔLL lo",
            "delta_ll_hi": "ΔLL hi",
            "tail_return": "tail return per ¥100",
            "tail_lo": "tail lo",
            "tail_hi": "tail hi",
        }
    )
    lines = [
        f"# The trainer debut prior by segment at the final Win price ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "- development rows only: holdout days are never read and the holdout is not opened",
        f"- bootstrap: {json.dumps(spec['bootstrap'], sort_keys=True)}; tail: top {1 - TOP:.0%}",
        "",
        "## Segments, ranked by ΔLL",
        "",
        _markdown_table(shown.to_dict("records")),
        "",
    ]
    return "\n".join(lines)


def run_study(*, export: Path | None = None, out_dir: Path | None = None, log=print) -> dict:
    """Freeze the spec, score every segment from #93's export, append one registry record per
    segment and write the table through `write_report`. Nothing is read before the freeze."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    spec = build_spec({**bootstrap_settings(), "seed": TICKET})
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any outcome is read
    git_sha = _git_sha()
    export_path = latest_export() if export is None else Path(export)
    builds = store_builds() | {
        f"{ix.SOURCE}_dividends": ix._file_build("dividends"),
        EXPORT: f"features/{COUNTRY}/{export_path.name}@{_digest(export_path)}",
    }
    boot = spec["bootstrap"]
    try:
        runners, races = ix.read_level1()
        history, _ = ix.level1_frame(runners, races)
        del runners, races
        rows = scored_rows(pd.read_parquet(export_path), history, log=log)
        table = score_segments(
            rows, n_boot=boot["n_boot"], level=boot["level"], seed=boot["seed"], top=TOP
        )
    except Exception as exc:  # a failed run is a registry record too
        registry.append(
            RunRecord(
                hypothesis_id=HYPOTHESIS,
                family_id=FAMILY_ID,
                variant="run:aborted",
                ticket=TICKET,
                git_sha=git_sha,
                data_builds=builds,
                sources=SOURCES,
                fold_scheme="walk_forward:D9 level-2 folds from 2020-Q1",
                components={"study": "studies/debut_prior.py", "member": MEMBER},
                params={
                    "spec_sha256": spec_hash,
                    "error": f"{type(exc).__name__}: {exc}",
                    "aborted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                metrics=(),
                status="failed",
            )
        )
        _ability.write_committed_summary(registry, families=(FAMILY_ID,))
        raise
    records = _records(table, boot, builds, git_sha, spec_hash)
    held = {r.variant for r in registry.records(FAMILY_ID) if r.status == "scored"}
    for record in records:
        if record.variant not in held:
            registry.append(record)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    stem = f"debut-prior-{COUNTRY}-{spec_hash[:12]}"
    report = out_dir / f"{stem}.md"
    write_report(report, render_report(table, spec, about), sources=SOURCES)
    write_report(
        out_dir / f"{stem}.json",
        json.dumps(table.to_dict("records"), indent=1, default=str),
        sources=SOURCES,
    )
    _ability.write_committed_summary(registry, families=(FAMILY_ID,))
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "segments": len(table),
        "records_appended": sum(r.variant not in held for r in records),
    }


def debut_prior_command(
    export: Annotated[
        Path | None,
        typer.Option(
            "--export", help="#93's level-1 export parquet (default: the latest v2 record's)."
        ),
    ] = None,
) -> None:
    """Measure #93's trainer debut prior by debut, maiden and class at the final Win price."""
    typer.echo(json.dumps(run_study(export=export, log=typer.echo), indent=2))
