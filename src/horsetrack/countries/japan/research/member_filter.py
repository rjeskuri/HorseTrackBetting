"""Each #713 rule combined with a 'horse beats the crowd' member filter (#716, pre-registered).

A combination (`COMBINATIONS`) is one rule of #713 (`predicted_final.RULES`), one member
(`MEMBERS`, each giving p_member per runner) and one threshold k of `K_GRID`. At each checkpoint
of `CHECKPOINTS` the rule picks its tickets on #713's projected board (`project_board`, the rule's
own picker) and the combination keeps the tickets that pass `qualifying`: every horse on the ticket
(every runner of a bracket leg's bracket, any one of them) has p_member / q >= k, q the crowd's
normalised win probability on the projected Win board (`crowd_q`). Every ticket stakes 100 yen and
is settled at the race's final dividends as #713 settles them. Its gain over the rule alone
(`paired_gain`) is the difference of the two returns per 100 yen on the races the member covers
where the rule bought, bounded by one meeting bootstrap shared by both sides.

`run_study` refuses (`FamilyNotFrozen`) before it reads a race unless every family of `FAMILIES`
(one per member x #713 family) is frozen with `family_spec`, refuses a member `MEMBERS` does not
register (ValueError) and an out_dir inside the repository, drops every holdout day (the holdout is
never opened), writes the private report through `write_report` and appends one registry record per
(family, checkpoint) whose params name every combination; `leaderboard_backfill` reads them as one
`predicted_final@T-k` row per combination and checkpoint. A member registered unavailable, a member
without predictions and a #713 rule no projected board reads are unmeasured rows. D21: the figures
are JRA-VAN-derived and stay under the data directory. Reading the store, the logs and the members'
predictions into the run's inputs is separate (not in this module).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import typer

from horsetrack.core import env
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    Registry,
    RunRecord,
    _canonical,
    _sha256,
    _spec_hash,
    default_registry,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_inside, write_report
from horsetrack.countries.japan.betting.formations import TicketSet
from horsetrack.countries.japan.board.snapshot import RaceBoard
from horsetrack.countries.japan.models.v0.nowcast import NowcastV0
from horsetrack.countries.japan.research import predicted_final as pf
from horsetrack.countries.japan.research import spread_scan

COUNTRY, TICKET, HYPOTHESIS_ID, STUDY = "japan", 716, "P716", "member-filter"
FAMILY_PREFIX = "japan-716-member-filter"
SOURCES = pf.SOURCES
FOLD_SCHEME = pf.FOLD_SCHEME
CHECKPOINTS = pf.CHECKPOINTS
K_GRID = (1.0, 1.25, 1.5, 2.0)  # the filter's threshold on p_member / q
BRACKET_POOL = "BRACKET_QUINELLA"
DEFAULT_SEED, DEFAULT_N_BOOT = pf.DEFAULT_SEED, pf.DEFAULT_N_BOOT
DEVELOPMENT = pf.DEVELOPMENT
NO_MATCH = "no qualifying ticket on the projected boards"


class FamilyNotFrozen(pf.FamilyNotFrozen):
    """A family of `FAMILIES` is not frozen with `family_spec`: nothing was read or written."""


# --------------------------------------------------------------------------- the pre-registration


@dataclass(frozen=True)
class Member:
    member_id: str
    ticket: int
    description: str
    unavailable: str | None = None  # why it cannot be scored yet


MEMBERS: dict[str, Member] = {
    m.member_id: m
    for m in (
        Member("debut-maiden-combined", 438, "#438's debut and maiden combined member"),
        Member("c4-meta", 87, "#87's C4 meta-model (form)"),
        Member("c2-ability", 92, "#92's C2 ability member"),
        Member("c2-interactions", 93, "#93's C2 interactions member"),
        Member(
            "jrdb-comment",
            715,
            "#715's JRDB horse-comment member",
            "#715's member has not landed: it gives no p_member yet",
        ),
    )
}


@dataclass(frozen=True)
class Combination:
    combination_id: str
    rule_id: str
    member_id: str
    k: float
    family_id: str


def combination_id(rule_id: str, member_id: str, k: float) -> str:
    """The id of rule x member x k: '<rule>|<member>|k<k>'."""
    return f"{rule_id}|{member_id}|k{float(k):g}"


def _family_id(member_id: str, rule_family: str) -> str:
    return f"{FAMILY_PREFIX}-{member_id}-{rule_family.removeprefix(pf.FAMILY_PREFIX + '-')}"


_FAMILY_PARTS: dict[str, tuple[str, str]] = {
    _family_id(m, f): (m, f) for m in MEMBERS for f in pf.FAMILIES
}
COMBINATIONS: dict[str, Combination] = {
    combination_id(r, m, k): Combination(combination_id(r, m, k), r, m, k, family)
    for family, (m, rule_family) in _FAMILY_PARTS.items()
    for r in pf.FAMILIES[rule_family]
    for k in K_GRID
}
FAMILIES: dict[str, tuple[str, ...]] = {
    family: tuple(c for c, combo in COMBINATIONS.items() if combo.family_id == family)
    for family in _FAMILY_PARTS
}


def _common_spec() -> dict[str, Any]:
    return {
        "study": STUDY,
        "ticket": TICKET,
        "hypothesis_id": HYPOTHESIS_ID,
        "sources": list(SOURCES),
        "hypothesis": "some #713 rule, kept to the tickets whose horses a member rates at least k "
        "times the crowd, returns more per 100 yen at the real final dividends than the rule alone "
        "on the same races; the null, the starting point: no filter adds return",
        "rules": "#713's predicted-final rules as frozen (research/predicted_final.py), by the "
        "spec hash of every #713 family spec",
        "rules_spec_hash": pf.spec_hash(),
        "checkpoints_min": list(CHECKPOINTS),
        "selection_labels": [pf.selection_label(k) for k in CHECKPOINTS],
        "k_grid": list(K_GRID),
        "members": {m.member_id: m.ticket for m in MEMBERS.values()},
        "combination_id": "'<rule>|<member>|k<k>'; every rule x member x k, unmeasured ones too",
        "projection": "#713's project_board of the observed board at post - k with the nowcast "
        "named at run time; each rule picks on the projection as #713 frozen it",
        "crowd_q": "q_h = (1 / O_h) / sum_j (1 / O_j) over the projected Win board's priced "
        "runners (finite, positive prices)",
        "p_member": "the member's win probability per runner of the race, taken as given; a race "
        "the member has no entry for is not covered by it",
        "filter": "a horse passes when its p_member and q are finite, q > 0 and p_member / q >= k "
        "(the boundary passes); a horse without either never passes. A ticket qualifies when "
        "every horse of its combination passes, in every pool and whatever its position "
        "(#713's tickets mark no axis); a bracket leg passes when any runner of the bracket passes",
        "stake_yen_per_ticket": pf.STAKE,
        "settlement": "as #713: each kept ticket returns its pool's paying dividends of its "
        "combination, its stake on a refund, else 0; a race without a dividends row of the "
        "rule's pool is no bet",
        "metrics": {
            "return_per_100": "100 x returned / staked over the combination's kept tickets",
            "n_bets": "races with a kept ticket; n_tickets the kept tickets; n_meetings theirs",
            "ci": "#713's D7 meeting bootstrap over the combination's own meetings",
            "gain": "over the races the member covers where the rule bought: 100 x combination "
            "returned / staked - 100 x rule returned / staked (ratios of sums); none when the "
            "combination staked nothing",
            "gain_ci": "one meeting bootstrap of the rule's covered meetings, rng.multinomial(m, "
            "uniform) per replicate with numpy default_rng(seed), the same weights on both "
            "sides; the difference's (1 - level) and level quantiles, one-sided each, over the "
            "replicates where the combination stakes; none under two meetings of combination bets",
            "level": pf._level(),
            "run_of_record": {"seed": DEFAULT_SEED, "n_boot": DEFAULT_N_BOOT},
        },
        "unmeasured": "a member registered unavailable, a member without predictions in the run "
        "and a #713 rule no projected board reads are rows with a note, never dropped",
        "data": "development race days only: is_holdout_day days are dropped and LockedHoldout is "
        "never opened",
        "leaderboard": "one row per combination x checkpoint: rule the combination id, segment "
        "'all', pool the rule's, selection predicted_final@T-k, withheld (D21)",
        "verdicts": "none: every combination x checkpoint is a ranked row (owner reframe, "
        "2026-11-27)",
    }


def family_spec(family_id: str) -> dict[str, Any]:
    """The plain-JSON spec `family_id` (one member x one #713 family) is frozen with."""
    if family_id not in FAMILIES:
        raise KeyError(f"unknown family {family_id!r} (have {sorted(FAMILIES)})")
    member_id, rule_family = _FAMILY_PARTS[family_id]
    member = MEMBERS[member_id]
    rules = {
        r: {"pool": pf.RULES[r].pool, "unmeasured": pf.RULES[r].unmeasured}
        for r in pf.FAMILIES[rule_family]
    }
    return {
        "family_id": family_id,
        "member": {
            "member_id": member_id,
            "ticket": member.ticket,
            "member": member.description,
            "unavailable": member.unavailable,
        },
        "rule_family": rule_family,
        "family_rules": rules,
        **_common_spec(),
    }


def spec_hash() -> str:
    return _spec_hash({family: family_spec(family) for family in FAMILIES})


def _check_frozen(registry: Registry) -> None:
    for family_id in FAMILIES:
        if registry.frozen_spec_hash(family_id) != _spec_hash(family_spec(family_id)):
            raise FamilyNotFrozen(
                f"{family_id} is not frozen with its spec: run `horsetrack japan member-filter "
                "freeze` first; nothing was read"
            )


# --------------------------------------------------------------------------- the filter


def crowd_q(board: RaceBoard) -> dict[int, float]:
    """The crowd's win probability per runner of the (projected) Win board, normalised."""
    inv = {}
    for h, odds in board.win.items():
        o = float(odds) if odds is not None else math.nan
        if math.isfinite(o) and o > 0:
            inv[h] = 1.0 / o
    total = sum(inv.values())
    return {h: v / total for h, v in inv.items()} if total > 0 else {}


def _passes(p: Any, q: Any, k: float) -> bool:
    try:
        p, q = float(p), float(q)
    except (TypeError, ValueError):
        return False
    return math.isfinite(p) and math.isfinite(q) and q > 0 and p / q >= k


def _legs(combo: Any) -> tuple[int, ...]:
    if isinstance(combo, (int, np.integer)):
        return (int(combo),)
    if isinstance(combo, str):
        return tuple(int(x) for x in combo.split(","))
    return tuple(int(x) for x in combo)


def qualifying(
    tickets: TicketSet,
    *,
    p_member: Mapping[int, Any],
    q: Mapping[int, Any],
    k: float,
    brackets: Mapping[int, int] | None = None,
) -> TicketSet | None:
    """The tickets of `tickets` whose every leg passes the filter (module docstring); None when
    none does."""
    passing = {int(h) for h in q if _passes(p_member.get(h), q[h], k)}
    if tickets.pool == BRACKET_POOL:
        passing = {int(b) for h, b in (brackets or {}).items() if int(h) in passing}
    kept = tuple(c for c in tickets.combinations if set(_legs(c)) <= passing)
    return TicketSet(tickets.pool, kept) if kept else None


# --------------------------------------------------------------------------- the paired gain


@dataclass(frozen=True)
class Gain:
    gain: float | None
    lo: float | None
    hi: float | None


def paired_gain(
    bets: Iterable[tuple[str, float, float, float, float]], *, seed: int, n_boot: int, level: float
) -> Gain:
    """The combination's return per 100 yen less the rule's on the same races, and its one-sided
    meeting-bootstrap bounds with one set of meeting weights for both sides (module docstring).
    `bets`: (meeting_id, rule_staked, rule_returned, combo_staked, combo_returned) per race."""
    by_meeting: dict[str, np.ndarray] = {}
    for meeting, *amounts in bets:
        acc = by_meeting.setdefault(meeting, np.zeros(4))
        acc += np.asarray(amounts, dtype=float)
    names = sorted(by_meeting)
    if not names:
        return Gain(None, None, None)
    table = np.array([by_meeting[m] for m in names])
    rule_staked, rule_returned, staked, returned = table.sum(axis=0)
    if staked <= 0 or rule_staked <= 0:
        return Gain(None, None, None)
    gain = float(100.0 * returned / staked - 100.0 * rule_returned / rule_staked)
    if np.count_nonzero(table[:, 2] > 0) < 2:
        return Gain(gain, None, None)
    m = len(names)
    w = np.random.default_rng(seed).multinomial(m, np.full(m, 1 / m), size=n_boot).astype(float)
    sums = w @ table
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = 100.0 * sums[:, 3] / sums[:, 2] - 100.0 * sums[:, 1] / sums[:, 0]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return Gain(gain, None, None)
    lo, hi = np.quantile(diff, [1 - level, level])
    return Gain(gain, float(lo), float(hi))


# --------------------------------------------------------------------------- the run


@dataclass
class StudyResult:
    rows: list[dict[str, Any]]
    report_path: Path
    run_ids: dict[str, str]
    counts: dict[str, int]


def _note(combo: Combination, covered: set[str], kept: bool) -> str | None:
    reasons = [pf.RULES[combo.rule_id].unmeasured, MEMBERS[combo.member_id].unavailable]
    if not any(reasons) and combo.member_id not in covered:
        reasons.append(f"no predictions from {combo.member_id} in this run")
    if not any(reasons) and not kept:
        reasons.append(NO_MATCH)
    return "; ".join(r for r in reasons if r) or None


def _row(combo: Combination, k: int, bets: list[tuple], covered: set[str], **boot) -> dict:
    """`bets`: (meeting, rule_staked, rule_returned, staked, returned, n_tickets) per race the
    rule bought on a race the member covers."""
    by_meeting: dict[str, list[float]] = {}
    for meeting, _, _, cost, ret, _ in bets:
        if cost > 0:
            acc = by_meeting.setdefault(meeting, [0.0, 0.0])
            acc[0], acc[1] = acc[0] + cost, acc[1] + ret
    staked = float(sum(b[3] for b in bets))
    returned = float(sum(b[4] for b in bets))
    lo, hi = pf._bootstrap(by_meeting, **boot)
    gain = paired_gain([b[:5] for b in bets], **boot)
    return {
        "combination_id": combo.combination_id,
        "rule_id": combo.rule_id,
        "member_id": combo.member_id,
        "k": combo.k,
        "checkpoint_min": k,
        "family_id": combo.family_id,
        "pool": pf.RULES[combo.rule_id].pool,
        "selection": pf.selection_label(k),
        "n_bets": sum(1 for b in bets if b[3] > 0),
        "n_tickets": int(sum(b[5] for b in bets)),
        "n_meetings": len(by_meeting),
        "staked_yen": staked,
        "returned_yen": returned,
        "return_per_100": 100.0 * returned / staked if staked > 0 else None,
        "ci_lo": lo,
        "ci_hi": hi,
        "gain": gain.gain,
        "gain_lo": gain.lo,
        "gain_hi": gain.hi,
        "note": _note(combo, covered, staked > 0),
    }


def _fingerprint(predictions: Mapping[str, Mapping], scored: list[str]) -> str:
    """The members' p_member on the scored races, as canonical text (NaN as its repr)."""
    items = sorted(
        [m, r, sorted([int(h), repr(float(v)) if v is not None else None] for h, v in p.items())]
        for m, by_race in predictions.items()
        for r in scored
        if (p := by_race.get(r)) is not None
    )
    return _sha256(_canonical(items))[:16]


def run_study(
    *,
    races: Iterable[pf.RaceInput],
    predictions: Mapping[str, Mapping[str, Mapping[int, float]]],
    nowcast: NowcastV0,
    registry: Registry,
    out_dir: Path | str,
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
    sources: tuple[str, ...] = SOURCES,
) -> StudyResult:
    """Score every combination at every checkpoint on the development races (module docstring)."""
    _check_frozen(registry)
    out_dir = Path(out_dir)
    if is_inside(out_dir, env.repo_root()):
        raise LicensedOutputInRepo(f"{out_dir} is inside the repository; write under the data dir")
    unknown = sorted(set(predictions) - set(MEMBERS))
    if unknown:
        raise ValueError(f"members {unknown} are not pre-registered (have {sorted(MEMBERS)})")
    level = pf._level()
    members = {m: p for m, p in predictions.items() if MEMBERS[m].unavailable is None}
    measured = [rule for rule in pf.RULES.values() if rule.pick is not None]
    bets: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    counts: Counter = Counter()
    scored: list[str] = []
    for race in races:
        if is_holdout_day(COUNTRY, race.race_date):
            counts["race: holdout day, dropped"] += 1
            continue
        own = race.paid.loc[race.paid["race_id"].to_numpy() == race.race_id]
        paid = spread_scan._paid(own)
        scored.append(race.race_id)
        covering = {m: p[race.race_id] for m, p in members.items() if race.race_id in p}
        for k in CHECKPOINTS:
            board = race.boards.get(k)
            if board is None:
                counts[f"race: no board at T-{k}"] += 1
                continue
            projected = pf.project_board(board, nowcast=nowcast, checkpoint_min=k)
            q = crowd_q(projected)
            for rule in measured:
                tickets = pf.select(rule.rule_id, projected, race=race)
                if tickets is None or not tickets.combinations or tickets.pool not in paid:
                    continue
                base = spread_scan._settle(tickets, paid)
                for m, p in covering.items():
                    for x in K_GRID:
                        kept = qualifying(tickets, p_member=p, q=q, k=x, brackets=race.brackets)
                        got = spread_scan._settle(kept, paid) if kept else None
                        bets[(combination_id(rule.rule_id, m, x), k)].append(
                            (
                                race.meeting_id,
                                base.cost,
                                base.ret,
                                got.cost if got else 0.0,
                                got.ret if got else 0.0,
                                kept.n_tickets if kept else 0,
                            )
                        )
    counts["race: scored"] = len(scored)
    covered = set(members)
    boot = {"seed": seed, "n_boot": n_boot, "level": level}
    rows = [
        _row(combo, k, bets.get((cid, k), []), covered, **boot)
        for cid, combo in COMBINATIONS.items()
        for k in CHECKPOINTS
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = spec_hash()
    report_path = out_dir / f"member-filter-japan-{digest[:12]}-{stamp}.md"
    write_report(report_path, _report(rows, counts, nowcast, digest, seed, n_boot), sources=sources)
    builds = {
        "nowcast": f"nowcast-v0@{nowcast.fit_id or 'unidentified'}",
        "races": f"race-inputs@{_sha256(_canonical(sorted(scored)))[:16]}",
        "predictions": f"member-predictions@{_fingerprint(predictions, scored)}",
        pf.REPORT_BUILD: f"member-filter-report@{report_path.name}",
    }
    by_record: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_record[(row["family_id"], row["checkpoint_min"])].append(row)
    git_sha = _git_sha()
    notes = {"run_utc": stamp, "seed": str(seed), "n_boot": str(n_boot)}  # run, not variant
    run_ids = {}
    for family_id in FAMILIES:
        for k in CHECKPOINTS:
            record = _record(family_id, k, by_record[(family_id, k)], builds, git_sha, sources)
            run_ids[f"{family_id}@T-{k}"] = registry.append(replace(record, notes=notes))
    return StudyResult(rows, report_path, run_ids, dict(counts))


def _record(
    family_id: str,
    k: int,
    rows: list[dict],
    builds: dict[str, str],
    git_sha: str,
    sources: tuple[str, ...],
) -> RunRecord:
    level = pf._level()
    metrics = []
    for row in rows:
        if row["return_per_100"] is None:
            continue
        cid, n = row["combination_id"], row["n_bets"]
        metrics += [
            FoldMetric(
                DEVELOPMENT,
                f"return_per_100[{cid}]",
                row["return_per_100"],
                row["ci_lo"],
                row["ci_hi"],
                n,
            ),
            FoldMetric(DEVELOPMENT, f"n_meetings[{cid}]", row["n_meetings"], None, None, n),
            FoldMetric(DEVELOPMENT, f"n_tickets[{cid}]", row["n_tickets"], None, None, n),
            FoldMetric(DEVELOPMENT, f"staked_yen[{cid}]", row["staked_yen"], None, None, n),
            FoldMetric(
                DEVELOPMENT, f"gain_per_100[{cid}]", row["gain"], row["gain_lo"], row["gain_hi"], n
            ),
        ]
    member_id, rule_family = _FAMILY_PARTS[family_id]
    combos = {
        row["combination_id"]: {
            "rule_id": row["rule_id"],
            "member_id": row["member_id"],
            "k": row["k"],
            "pool": row["pool"],
            "unmeasured": pf.RULES[row["rule_id"]].unmeasured or MEMBERS[member_id].unavailable,
        }
        for row in rows
    }
    return RunRecord(
        hypothesis_id=HYPOTHESIS_ID,
        family_id=family_id,
        variant=pf.selection_label(k),
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=dict(builds),
        sources=tuple(sources),
        fold_scheme=FOLD_SCHEME,
        components={
            "projector": "board.shadow_feed._projector",
            "nowcast": "models.v0.NowcastV0",
            "member": member_id,
        },
        params={
            "checkpoint_min": k,
            "selection": pf.selection_label(k),
            "stake_yen": pf.STAKE,
            "ci_level": level,
            "k_grid": list(K_GRID),
            "member_id": member_id,
            "rule_family": rule_family,
            "rules_spec_hash": pf.spec_hash(),
            "rules": combos,  # the leaderboard reads each combination as a rule row
        },
        metrics=tuple(metrics),
        status="scored" if metrics else "failed",
    )


# --------------------------------------------------------------------------- the report


def _num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _bounds(lo: float | None, hi: float | None) -> str:
    return "—" if lo is None else f"[{_num(lo)}, {_num(hi)}]"


def _report(rows, counts, nowcast, digest, seed, n_boot) -> str:
    measured = [r for r in rows if r["return_per_100"] is not None]
    lines = [
        "# #713's rules with a 'horse beats the crowd' member filter (Japan, #716)",
        "",
        "Private: built on JRA-VAN (D21). Never commit this file.",
        "",
        f"- spec hash `{digest}`; #713 rules spec `{pf.spec_hash()}`; nowcast fit "
        f"`{nowcast.fit_id or 'unidentified'}`; seed {seed}, {n_boot} bootstrap replicates",
        *(f"- {name}: {value}" for name, value in sorted(counts.items())),
        "",
        "## Hypothesis",
        "",
        "Keeping a rule's tickets to those whose horses a member rates at least k times the crowd "
        "raises its return per 100 yen at the real final dividends over the rule alone on the same "
        "races. The null (no filter adds return) is the starting point.",
        "",
        "## Variants tried",
        "",
        f"{len(pf.RULES)} rules x {len(MEMBERS)} members x {len(K_GRID)} k x {len(CHECKPOINTS)} "
        f"checkpoints = {len(rows)} rows, {len(measured)} with a bet; every row, unmeasured ones "
        "included. Gain: the combination's return per 100 less the rule's on the races the member "
        "covers, with its paired meeting-bootstrap bounds.",
        "",
        "| Combination | T-k | Bets | Tickets | Meetings | Staked | Returned | Per 100 | CI | "
        "Gain | Gain CI | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['combination_id']}` | {r['checkpoint_min']} | {r['n_bets']} | "
            f"{r['n_tickets']} | {r['n_meetings']} | {r['staked_yen']:.0f} | "
            f"{r['returned_yen']:.0f} | {_num(r['return_per_100'])} | "
            f"{_bounds(r['ci_lo'], r['ci_hi'])} | {_num(r['gain'])} | "
            f"{_bounds(r['gain_lo'], r['gain_hi'])} | {r['note'] or ''} |"
        )
    lines += [
        "",
        "## What this rules out",
        "",
        "Read the rows above against the null; nothing is adopted or rejected (D19).",
        "",
        "## Leaderboard rows",
        "",
        "One row per combination x checkpoint, selection predicted_final@T-k, withheld (D21).",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- the command

member_filter_app = typer.Typer(
    no_args_is_help=True, help="#716: #713's rules with a 'horse beats the crowd' member filter."
)


@member_filter_app.command("freeze")
def freeze_command() -> None:
    """Freeze every family's spec in the japan registry (irreversible; again is a no-op)."""
    registry = default_registry(COUNTRY)
    for family_id in FAMILIES:
        typer.echo(
            f"frozen {family_id} {registry.freeze_family(family_id, family_spec(family_id))}"
        )
    typer.echo(f"spec_hash={spec_hash()}")
