"""One leaderboard of every option, shrunk within its pool, ranked, written twice (#449).

A row is one option the project has measured or will measure (a rule, a comparator, a cell, a
model, a member) in a pool and a segment: its expected return per ¥100 staked at the final price,
the CI beside it (each bound one-sided at ``ci_level``, D7 style) and the number of variants its
family searched. Nothing is dropped: an option with no measured return (a failed or pending run)
is a row too, and ranks after every measured row, unranked. A descriptive row (``descriptive``,
#451) is a measurement that is not a return per ¥100 staked on a ticket, such as WIN5's pool-level
day value: it takes no part in its pool's shrinkage and gets no rank, and the Markdown texts show
it with its value and bounds in its own section, "Descriptive, not ranked", after the ranked
table and never in it.

Shrinkage (``shrink``). Within each pool, over the k rows with a return ``x_i`` and a standard
error ``se_i`` (given, or recovered from the one-sided bounds by ``LeaderboardRow.se``),
DerSimonian-Laird:

    w_i = 1 / se_i^2,  x_bar = sum(w_i x_i) / sum(w_i),  Q = sum(w_i (x_i - x_bar)^2),
    C = sum(w_i) - sum(w_i^2) / sum(w_i),  tau^2 = max(0, (Q - (k - 1)) / C),
    mu = sum(x_i / (se_i^2 + tau^2)) / sum(1 / (se_i^2 + tau^2)),
    shrunk_i = mu + tau^2 / (tau^2 + se_i^2) * (x_i - mu).

C is evaluated as 2 sum_{i<j} w_i w_j / sum(w) and Q around the pool's most precise row: the same
quantities, free of the cancellation that zeroes C as written when one se is far below the rest.

A pool of one keeps its raw return. A row without a pool or a positive se gets no shrunk value
and ranks on its raw return. A descriptive row gets no shrunk value and is left out of its pool.

Rows of one pool overlap: a segment's bets are also bets of the pool's 'all' row, and the variants
of a family share races. DerSimonian-Laird assumes independent estimates, so here tau^2 is a
ranking device that pulls noisy rows toward their pool, not inference: no CI, test or claim is
built on it.

The target is the pool's own mean, not its payout rate. Flat boxes of exotic pools return far
below the payout rate because longshots are overbet (the ledger's #81 rows), so shrinking toward
the payout rate would lift exactly the rows the market prices worst.

Withholding (D21). A row is withheld when it is marked licensed or its sources name JRDB or
JRA-VAN in any spelling (``core.eval.reports.is_withheld``). ``public_view`` drops withheld rows
and reshrinks the rest over themselves alone, so no public shrunk value or rank depends on a
licensed row; the public text lists withheld rows by option id and family only. Its variants
total still counts withheld families, by the ticket's decision (#449, Context): it adds each
family's variants searched over every family, so a public-only total would understate the search
the multiple-testing count exists to disclose. The private
writer passes every row's sources to ``write_report`` and itself refuses a path inside the
repository when any row is withheld, so a row marked licensed under a public source name is
refused too.

From the registry (``rows_from_registry``, #450). The run registry is the leaderboard's source:
every non-holdout run gives measured rows from the convention of ``docs/specs/leaderboard.md``,
an adapter's rows, or one unmeasured row, failures included; each option's latest row in append
order is its row.

Generic, with no country input, like ``segment_roi`` and ``snooping``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import norm

from horsetrack.core import env
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord, is_licensed_record
from horsetrack.core.eval.reports import (
    LicensedOutputInRepo,
    is_inside,
    is_withheld,
    write_report,
)

KINDS = ("rule", "comparator", "cell", "model", "member", "other")
SELECTIONS = ("decision_time", "card_only", "look_ahead", "unknown")
N_UNITS = ("bets", "races", "days")

_WRITER = "horsetrack.core.eval.leaderboard"
_NONE = "—"
_FLOATS = (
    "return_per_100",
    "ci_lo",
    "ci_hi",
    "ci_level",
    "se_per_100",
    "payout_rate",
    "shrunk_per_100",
)


class RowRefused(ValueError):
    """A leaderboard row (or a set of rows) is malformed."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class LeaderboardRow:
    option_id: str
    family_id: str
    ticket: int | None
    kind: str
    pool: str | None
    segment: str
    rule: str
    selection: str
    sources: tuple[str, ...]
    variants_searched: int
    status: str
    return_per_100: float | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    ci_level: float | None = None
    se_per_100: float | None = None
    n: int | None = None
    n_unit: str | None = None
    n_meetings: int | None = None
    payout_rate: float | None = None
    shrunk_per_100: float | None = None
    licensed: bool = False
    note: str | None = None
    descriptive: bool = False

    def __post_init__(self) -> None:
        problem = self._problem()
        if problem is not None:
            raise RowRefused(f"leaderboard row {self.option_id!r}: {problem}")

    def _problem(self) -> str | None:
        if not _is_text(self.option_id):
            return "option_id must be a non-blank string"
        if not _is_text(self.family_id):
            return "family_id must be a non-blank string"
        if self.ticket is not None and not _is_int(self.ticket):
            return f"ticket must be an int or None, got {self.ticket!r}"
        if self.kind not in KINDS:
            return f"kind {self.kind!r} is not one of {KINDS}"
        if self.pool is not None and not _is_text(self.pool):
            return "pool must be a non-blank string or None"
        for name in ("segment", "rule", "status"):
            if not isinstance(getattr(self, name), str):
                return f"{name} must be a string"
        if self.selection not in SELECTIONS:
            return f"selection {self.selection!r} is not one of {SELECTIONS}"
        if not isinstance(self.sources, tuple) or not all(isinstance(s, str) for s in self.sources):
            return f"sources must be a tuple of source names, got {self.sources!r}"
        if not _is_int(self.variants_searched) or self.variants_searched < 1:
            return f"variants_searched must be an int of at least 1, got {self.variants_searched!r}"
        for name in _FLOATS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{name} must be a number or None, got {value!r}"
            if not math.isfinite(value):
                return f"{name} must be finite, got {value!r}"
        if self.ci_lo is not None and self.ci_hi is not None and self.ci_lo > self.ci_hi:
            return f"ci_lo {self.ci_lo} is above ci_hi {self.ci_hi}"
        if self.ci_level is not None and not 0.5 < self.ci_level < 1:
            return f"ci_level must lie strictly between 0.5 and 1, got {self.ci_level}"
        if self.se_per_100 is not None and self.se_per_100 < 0:
            return f"se_per_100 must not be negative, got {self.se_per_100}"
        for name in ("n", "n_meetings"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 0):
                return f"{name} must be a non-negative int or None, got {value!r}"
        if self.n_unit is not None and self.n_unit not in N_UNITS:
            return f"n_unit {self.n_unit!r} is not one of {N_UNITS}"
        if not isinstance(self.licensed, bool):
            return f"licensed must be a bool, got {self.licensed!r}"
        if self.note is not None and not isinstance(self.note, str):
            return "note must be a string or None"
        if not isinstance(self.descriptive, bool):
            return f"descriptive must be a bool, got {self.descriptive!r}"
        return None

    @property
    def withheld(self) -> bool:
        """Marked licensed, or built on a withheld source in any spelling (D21)."""
        return self.licensed or is_withheld(self.sources)

    @property
    def measured(self) -> bool:
        return self.return_per_100 is not None

    def se(self) -> float | None:
        """se_per_100; else from both one-sided bounds; else from one bound and the return."""
        if self.se_per_100 is not None:
            return self.se_per_100
        if self.ci_level is None:
            return None
        z = float(norm.ppf(self.ci_level))
        if self.ci_lo is not None and self.ci_hi is not None:
            return (self.ci_hi - self.ci_lo) / (2 * z)
        bound = self.ci_lo if self.ci_lo is not None else self.ci_hi
        if bound is None or self.return_per_100 is None:
            return None
        return abs(self.return_per_100 - bound) / z


@dataclass(frozen=True)
class RankedRow:
    rank: int | None
    row: LeaderboardRow


def _dersimonian_laird(xs: list[float], variances: list[float]) -> list[float]:
    k = len(xs)
    if k == 1:
        return list(xs)
    w = [1.0 / v for v in variances]
    sw = sum(w)
    # The Context's formulas, arranged so that a pool mixing a near-exact row with ordinary ones
    # (se 1e-8 beside se 5: weights 1e16 beside 0.04) loses nothing to cancellation.
    # x_bar is taken relative to the most precise row p: x_bar - x_p = sum(w_i (x_i - x_p)) / sum(w)
    # holds no huge term, so Q's term for p, w_p (x_p - x_bar)^2, is not an ulp of x_bar squared
    # and multiplied by 1e16.
    p = max(range(k), key=w.__getitem__)
    d = [xi - xs[p] for xi in xs]
    shift = sum(wi * di for wi, di in zip(w, d, strict=True)) / sw
    q = sum(wi * (di - shift) ** 2 for wi, di in zip(w, d, strict=True))
    # C = sum(w) - sum(w^2) / sum(w) = 2 * sum_{i<j} w_i w_j / sum(w). As written, the subtraction
    # rounds to 0 there and tau^2 divides by it; here every term is positive, so nothing cancels.
    c = 0.0
    before = 0.0
    for wi in w:
        c += wi / sw * before
        before += wi
    c *= 2.0
    tau2 = max(0.0, (q - (k - 1)) / c)
    mu = sum(x / (v + tau2) for x, v in zip(xs, variances, strict=True)) / sum(
        1.0 / (v + tau2) for v in variances
    )
    return [mu + tau2 / (tau2 + v) * (x - mu) for x, v in zip(xs, variances, strict=True)]


def shrink(rows: Iterable[LeaderboardRow]) -> list[LeaderboardRow]:
    """Copies of ``rows``, in order, with ``shrunk_per_100`` set per pool (any incoming value is
    ignored); descriptive rows, and rows without a return, a pool or a positive se, get None and
    are left out of their pool's estimate."""
    rows = list(rows)
    pools: dict[str, list[int]] = {}
    variances: dict[int, float] = {}
    for i, row in enumerate(rows):
        se = row.se()
        rankable = row.measured and not row.descriptive and row.pool is not None
        if rankable and se is not None and se > 0:
            pools.setdefault(row.pool, []).append(i)
            variances[i] = se * se
    shrunk: list[float | None] = [None] * len(rows)
    for members in pools.values():
        values = _dersimonian_laird(
            [rows[i].return_per_100 for i in members], [variances[i] for i in members]
        )
        for i, value in zip(members, values, strict=True):
            shrunk[i] = value
    return [
        dataclasses.replace(row, shrunk_per_100=value)
        for row, value in zip(rows, shrunk, strict=True)
    ]


def _refuse_duplicates(rows: list[LeaderboardRow]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.option_id in seen:
            raise RowRefused(f"option id {row.option_id!r} appears twice")
        seen.add(row.option_id)


def rank(rows: Iterable[LeaderboardRow]) -> list[RankedRow]:
    """Measured rows by their shrunk value (else raw) descending, ties by raw return descending
    then option id, ranked 1..k; then measured descriptive rows, unranked, by family id then
    option id; then unmeasured rows (descriptive or not), unranked, by family id then option id.
    The shrunk value on each row is used as it is: ``rank`` does not shrink."""
    rows = list(rows)
    _refuse_duplicates(rows)

    def key(row: LeaderboardRow) -> tuple[float, float, str]:
        value = row.shrunk_per_100 if row.shrunk_per_100 is not None else row.return_per_100
        return (-value, -row.return_per_100, row.option_id)

    def by_family(row: LeaderboardRow) -> tuple[str, str]:
        return (row.family_id, row.option_id)

    ranked = sorted((r for r in rows if r.measured and not r.descriptive), key=key)
    descriptive = sorted((r for r in rows if r.measured and r.descriptive), key=by_family)
    unmeasured = sorted((r for r in rows if not r.measured), key=by_family)
    return [RankedRow(i, r) for i, r in enumerate(ranked, start=1)] + [
        RankedRow(None, r) for r in (*descriptive, *unmeasured)
    ]


def public_view(rows: Iterable[LeaderboardRow]) -> list[LeaderboardRow]:
    """The rows that are not withheld, reshrunk over themselves alone."""
    return shrink([r for r in rows if not r.withheld])


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|") or _NONE


def _figure(value: float | None) -> str:
    return _NONE if value is None else f"{value:.2f}"


def _ci(row: LeaderboardRow) -> str:
    if row.ci_lo is not None and row.ci_hi is not None:
        span = f"{row.ci_lo:.2f} to {row.ci_hi:.2f}"
    elif row.ci_lo is not None:
        span = f"≥ {row.ci_lo:.2f}"
    elif row.ci_hi is not None:
        span = f"≤ {row.ci_hi:.2f}"
    else:
        return _NONE
    return span if row.ci_level is None else f"{span} ({row.ci_level * 100:g}%)"


def _n(row: LeaderboardRow) -> str:
    if row.n is None:
        return _NONE
    return f"{row.n} {row.n_unit}" if row.n_unit is not None else str(row.n)


def _variants_total(rows: list[LeaderboardRow]) -> int:
    """The sum over distinct families of each family's variants searched (its largest count).

    Withheld families count too, in the public text as in the private one: the ticket's Context
    (#449) sets the total as "the sum over distinct families, withheld ones included". The total
    exists for multiple-testing honesty, and a public-only total would understate how many
    variants were searched in all. Every figure of a withheld row (return,
    bounds, se, n, n_meetings, payout_rate, shrunk value, rule, note, sources) stays out of the
    public text, which names withheld rows by option id and family only.
    """
    families: dict[str, int] = {}
    for row in rows:
        families[row.family_id] = max(families.get(row.family_id, 0), row.variants_searched)
    return sum(families.values())


_COLUMNS = (
    "Rank",
    "Option",
    "Kind",
    "Pool",
    "Segment",
    "Selection",
    "Return per ¥100 staked",
    "CI",
    "Shrunk",
    "n",
    "Variants searched",
    "Sources",
)
_ALIGN = ("---:", "---", "---", "---", "---", "---", "---:", "---", "---:", "---:", "---:", "---")
# The descriptive section's table: no Shrunk column (a descriptive row is never shrunk), and a
# Rank column that always shows ``_NONE``, so each row reads as unranked.
_DESCRIPTIVE_HEADING = "## Descriptive, not ranked"
_DESCRIPTIVE_COLUMNS = (
    "Rank",
    "Option",
    "Kind",
    "Pool",
    "Segment",
    "Selection",
    "Pool value per ¥100 (descriptive)",
    "CI",
    "n",
    "Variants searched",
    "Sources",
)
_DESCRIPTIVE_ALIGN = _ALIGN[:8] + _ALIGN[9:]  # _ALIGN without Shrunk's


def _table_line(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown(rows: Iterable[LeaderboardRow], *, title: str, public: bool) -> str:
    """The leaderboard as Markdown: shrunk and ranked over every row, or, when ``public``, over
    the public view alone, with the withheld rows listed by option id and family only.

    The ranked table holds the rows that are not descriptive (ranked, then unmeasured); the
    descriptive rows follow in their own section, unranked, in ``rank``'s order (measured by
    family id then option id, then unmeasured); the public withheld list comes last."""
    rows = list(rows)
    _refuse_duplicates(rows)
    shown = public_view(rows) if public else shrink(rows)
    lines = [
        f"# {title}",
        "",
        f"Written by `{_WRITER}`; do not edit by hand.",
        "",
        "Rows rank by expected return per ¥100 staked at the final price, shrunk toward their "
        "pool's own mean (DerSimonian-Laird) or raw where a row has no standard error, highest "
        "first; rows with no measured return come last, unranked. Descriptive rows (a "
        "measurement that is not a return per ¥100 staked on a ticket, such as a pool-level "
        "value) are not ranked: they are left out of the shrinkage and shown with their value "
        "and bounds in their own section after the ranked table. Each CI bound is one-sided at "
        "the level shown.",
        "",
        # Over every row, withheld ones included, public or not, by the ticket's decision (#449,
        # Context): a public-only total would understate the search. No other figure of a
        # withheld row reaches the public text (see ``_variants_total``).
        f"Variants searched in total: {_variants_total(rows)} (each family's variants searched, "
        "its largest count across its rows, added over every family, withheld families "
        "included).",
        "",
    ]
    if public:
        lines += [
            "Public version: rows built on JRDB or JRA-VAN, or marked licensed, are withheld "
            "(D21) and listed by option id below; every shrunk value and rank here is computed "
            "over the public rows alone.",
            "",
        ]
    ordered = rank(shown)
    lines += [_table_line(_COLUMNS), _table_line(_ALIGN)]
    for ranked in ordered:
        row = ranked.row
        if row.descriptive:
            continue
        lines.append(
            _table_line(
                (
                    _NONE if ranked.rank is None else str(ranked.rank),
                    _cell(row.option_id),
                    row.kind,
                    _cell(row.pool or ""),
                    _cell(row.segment),
                    row.selection,
                    _figure(row.return_per_100),
                    _ci(row),
                    _figure(row.shrunk_per_100),
                    _n(row),
                    str(row.variants_searched),
                    _cell(", ".join(row.sources)),
                )
            )
        )
    descriptive = [ranked.row for ranked in ordered if ranked.row.descriptive]
    if descriptive:
        lines += ["", _DESCRIPTIVE_HEADING, ""]
        lines += [_table_line(_DESCRIPTIVE_COLUMNS), _table_line(_DESCRIPTIVE_ALIGN)]
        for row in descriptive:
            lines.append(
                _table_line(
                    (
                        _NONE,
                        _cell(row.option_id),
                        row.kind,
                        _cell(row.pool or ""),
                        _cell(row.segment),
                        row.selection,
                        _figure(row.return_per_100),
                        _ci(row),
                        _n(row),
                        str(row.variants_searched),
                        _cell(", ".join(row.sources)),
                    )
                )
            )
    if public:
        withheld = sorted((r for r in rows if r.withheld), key=lambda r: (r.family_id, r.option_id))
        lines += ["", "## Withheld (D21)", ""]
        lines += [f"- `{r.option_id}` (family `{r.family_id}`)" for r in withheld] or ["None."]
    return "\n".join(lines) + "\n"


def to_json(rows: Iterable[LeaderboardRow]) -> str:
    return json.dumps(
        [dataclasses.asdict(r) for r in rows], ensure_ascii=False, allow_nan=False, indent=2
    )


def from_json(text: str) -> list[LeaderboardRow]:
    """Rows from ``to_json``'s text; anything that is not a list of valid rows is refused."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RowRefused(f"leaderboard JSON does not parse: {exc}") from exc
    if not isinstance(data, list):
        raise RowRefused("leaderboard JSON must be a list of rows")
    rows = []
    for item in data:
        if not isinstance(item, dict):
            raise RowRefused(f"a leaderboard row must be an object, got {item!r}")
        fields = dict(item)
        if isinstance(fields.get("sources"), list):
            fields["sources"] = tuple(fields["sources"])
        try:
            rows.append(LeaderboardRow(**fields))
        except TypeError as exc:  # a missing or unknown field
            raise RowRefused(f"leaderboard row {item.get('option_id')!r}: {exc}") from exc
    return rows


def write_leaderboard(
    path: Path, rows: Iterable[LeaderboardRow], *, title: str, public: bool
) -> Path:
    """Render and write through ``write_report`` with the union of the rendered rows' sources:
    every row when private, the public rows when public. A path inside the repository is refused
    when any rendered row is withheld, by its sources or by its licensed mark."""
    rows = list(rows)
    text = render_markdown(rows, title=title, public=public)
    rendered = [r for r in rows if not r.withheld] if public else rows
    sources = tuple(sorted({s for r in rendered for s in r.sources}))
    if any(r.withheld for r in rendered):
        root = env.repo_root()
        if is_inside(path, root):
            raise LicensedOutputInRepo(
                f"{path} is inside the repository at {root} and the leaderboard holds a withheld "
                "row (D21); write it under $HORSETRACK_DATA_DIR"
            )
    return write_report(path, text, sources=sources)


# --------------------------------------------------------------------------- from the registry

Adapter = Callable[[RunRecord, int], list[LeaderboardRow]]

BLOCK_FIELDS = ("kind", "pool", "rule", "selection", "payout_rate", "ci_level", "n_unit")
_DEVELOPMENT = "development"
_ALL = "all"
_CONVENTION_METRIC = re.compile(r"(return_per_100|se_per_100|n_meetings)(?:\[(.+)\])?")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _block_problem(block: object) -> str | None:
    """Why ``params['leaderboard']`` does not follow the convention, or None."""
    if not isinstance(block, Mapping):
        return (
            f"params['leaderboard'] must be a mapping of {BLOCK_FIELDS}, got {type(block).__name__}"
        )
    missing = [name for name in BLOCK_FIELDS if name not in block]
    if missing:
        return f"params['leaderboard'] lacks {missing}"
    unknown = sorted(str(name) for name in block if name not in BLOCK_FIELDS)
    if unknown:
        return f"params['leaderboard'] has unknown fields {unknown}"
    if block["kind"] not in KINDS:
        return f"kind {block['kind']!r} is not one of {KINDS}"
    if block["pool"] is not None and not _is_text(block["pool"]):
        return f"pool must be a non-blank string or None, got {block['pool']!r}"
    if not _is_text(block["rule"]):
        return f"rule must be a non-blank string, got {block['rule']!r}"
    if block["selection"] not in SELECTIONS:
        return f"selection {block['selection']!r} is not one of {SELECTIONS}"
    if block["payout_rate"] is not None and not _finite_number(block["payout_rate"]):
        return f"payout_rate must be a finite number or None, got {block['payout_rate']!r}"
    level = block["ci_level"]
    if not (_finite_number(level) and 0.5 < level < 1):
        return f"ci_level must be a number strictly between 0.5 and 1, got {level!r}"
    if block["n_unit"] not in N_UNITS:
        return f"n_unit {block['n_unit']!r} is not one of {N_UNITS}"
    return None


def _development_metrics(record: RunRecord) -> dict[str, dict[str, FoldMetric]]:
    """The record's development-fold convention metrics, by name then segment."""
    found: dict[str, dict[str, FoldMetric]] = {}
    for metric in record.metrics:
        match = _CONVENTION_METRIC.fullmatch(metric.metric)
        if metric.fold == _DEVELOPMENT and match is not None:
            found.setdefault(match[1], {})[match[2] or _ALL] = metric
    return found


def _option_id(record: RunRecord, segment: str) -> str:
    base = f"{record.family_id}/{record.variant}"
    return base if segment == _ALL else f"{base}[{segment}]"


def _meetings(metric: FoldMetric | None, segment: str) -> int | None:
    if metric is None:
        return None
    if not (_finite_number(metric.value) and float(metric.value).is_integer()):
        raise RowRefused(f"n_meetings[{segment}] is not a whole number: {metric.value}")
    return int(metric.value)


def _convention_rows(record: RunRecord, block: Mapping, size: int) -> list[LeaderboardRow]:
    found = _development_metrics(record)
    se = found.get("se_per_100", {})
    meetings = found.get("n_meetings", {})
    licensed = is_licensed_record(record)
    return [
        LeaderboardRow(
            option_id=_option_id(record, segment),
            family_id=record.family_id,
            ticket=record.ticket,
            kind=block["kind"],
            pool=block["pool"],
            segment=segment,
            rule=block["rule"],
            selection=block["selection"],
            sources=tuple(record.sources),
            variants_searched=size,
            status=record.status,
            return_per_100=metric.value,
            ci_lo=metric.lo,
            ci_hi=metric.hi,
            ci_level=block["ci_level"],
            se_per_100=se[segment].value if segment in se else None,
            n=metric.n,
            n_unit=block["n_unit"],
            n_meetings=_meetings(meetings.get(segment), segment),
            payout_rate=block["payout_rate"],
            licensed=licensed,
        )
        for segment, metric in found.get("return_per_100", {}).items()
    ]


def _record_rows(
    record: RunRecord, size: int, adapter: Adapter | None
) -> tuple[list[LeaderboardRow], str | None]:
    """The record's rows (the adapter's, else the convention's) and why it has none, if known.
    An adapter's rows are marked licensed when the record is, keeping an adapter's own mark."""
    try:
        has_block = "leaderboard" in record.params
        block = record.params.get("leaderboard")
        problem = _block_problem(block) if has_block else None
        if adapter is not None:
            licensed = is_licensed_record(record)
            made = [
                dataclasses.replace(row, licensed=row.licensed or licensed)
                for row in adapter(record, size)
            ]
            return made, None if made else problem
        if not has_block or problem is not None:
            return [], problem
        return _convention_rows(record, block, size), None
    except RowRefused as exc:
        return [], str(exc)


def _unmeasured_row(record: RunRecord, size: int, note: str | None) -> LeaderboardRow:
    block = record.params.get("leaderboard")
    block = block if isinstance(block, Mapping) else {}
    kind = block.get("kind")
    pool = block.get("pool")
    return LeaderboardRow(
        option_id=_option_id(record, _ALL),
        family_id=record.family_id,
        ticket=record.ticket,
        kind=kind if kind in KINDS else "other",
        pool=pool if _is_text(pool) else None,
        segment=_ALL,
        rule=record.variant,
        selection="unknown",
        sources=tuple(record.sources),
        variants_searched=size,
        status=record.status,
        licensed=is_licensed_record(record),
        note=note,
    )


def rows_from_registry(
    registry: Registry, *, adapters: Mapping[str, Adapter] | None = None
) -> list[LeaderboardRow]:
    """Every (family, variant) of ``registry`` as leaderboard rows, unshrunk; read through
    ``Registry.records()``, never written.

    A run with ``holdout_open`` set gives no row, so a variant run only on the holdout has none.
    Each other run gives its rows: an adapter's when ``adapters`` holds its family (called with
    the run and the family size), else one measured row per development-fold ``return_per_100``
    metric under a well-formed ``params['leaderboard']`` block (``docs/specs/leaderboard.md``).
    A run that gives no row (failed, no development return, a malformed block, an adapter that
    returns none) gives one unmeasured row, the '<family>/<variant>' option, with the reason in
    ``note`` when its block is malformed. Rows resolve per option id: each option's latest row in
    append order is its row, adapter or not, so a rerun replaces only the options it gives (its
    unmeasured row replaces '<family>/<variant>' alone) and an option a later run does not give
    keeps its earlier row. Every
    row carries the family size as its variants searched, the run's ticket, sources and status
    (old 'rejected' and 'adopted' runs read like 'scored'), and ``licensed`` as
    ``is_licensed_record`` (an adapter's row also keeps its own True).
    """
    adapters = adapters or {}
    families: dict[str, list[RunRecord]] = {}
    for record in registry.records():
        if record.holdout_open is None:
            families.setdefault(record.family_id, []).append(record)
    latest: dict[str, LeaderboardRow] = {}
    for family_id, runs in families.items():
        size = registry.family_size(family_id)
        adapter = adapters.get(family_id)
        for record in runs:
            made, problem = _record_rows(record, size, adapter)
            for row in made or [_unmeasured_row(record, size, problem)]:
                latest[row.option_id] = row
    return list(latest.values())
