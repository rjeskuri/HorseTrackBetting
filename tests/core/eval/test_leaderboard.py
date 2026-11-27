"""Ticket #449: one leaderboard of every option, shrunk within its pool, ranked, written twice.

Contract pinned here (the coder implements what these tests call):

- ``LeaderboardRow`` (frozen dataclass) with the fields of the ticket's Context; ``__post_init__``
  raises ``RowRefused`` (a ValueError) on a bad field: an unknown kind, selection or n unit,
  ``ci_lo > ci_hi``, a negative n, a blank option id, ``variants_searched < 1``.
- ``row.withheld``: licensed, or ``core.eval.reports.is_withheld(sources)``; ``row.measured``:
  ``return_per_100 is not None``; ``row.se()``: se_per_100, else ``(ci_hi - ci_lo) / (2z)``, else
  ``|return - bound| / z`` with one bound, ``z = norm.ppf(ci_level)``, else None.
- ``shrink(rows)``: copies with ``shrunk_per_100`` set by DerSimonian-Laird per pool over the rows
  with a return, a se and a pool; a pool of one keeps its raw return; a row without a se or a pool
  gets None; an incoming ``shrunk_per_100`` is ignored.
- ``rank(rows)``: measured rows by (shrunk, else raw) descending, ties by raw descending then
  option id ascending, ranks 1..k; then unmeasured rows, rank None, by family id then option id.
  Duplicate option ids raise ``RowRefused``.
- ``public_view(rows)``: the rows that are not withheld, reshrunk over themselves alone.
- ``render_markdown(rows, *, title, public)``: a table with each row's CI, shrunk value and
  variants searched (figures to 2 decimals); the total variants searched is the sum over distinct
  families, withheld ones included; public text holds no figure of a withheld row and lists the
  withheld option ids and families under ``## Withheld (D21)``.
- ``write_leaderboard(path, rows, *, title, public)`` writes through ``write_report`` with the union
  of the rendered rows' sources; ``to_json`` / ``from_json`` round-trip rows exactly.

Every row is synthetic and built here. Expected shrunk values come from ``dersimonian_laird``
below, written from the ticket's formulas, and expected standard errors from the standard
library's ``NormalDist``; nothing expected is computed by the module under test. Every write goes
under ``tmp_path``, except one refused write aimed at the real repository, which is cleaned up.
"""

from __future__ import annotations  # noqa: I001 (leaderboard sorts as third-party until it exists)

import dataclasses
import json
import random
import re
import uuid
from pathlib import Path
from statistics import NormalDist

import pytest

import horsetrack.core.eval.leaderboard as leaderboard
from horsetrack.core import env
from horsetrack.core.eval import reports
from horsetrack.core.eval.leaderboard import (
    KINDS,
    N_UNITS,
    SELECTIONS,
    LeaderboardRow,
    RankedRow,
    RowRefused,
    from_json,
    public_view,
    rank,
    render_markdown,
    shrink,
    to_json,
    write_leaderboard,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo

TOL = 1e-9
TITLE = "Synthetic leaderboard"


# --------------------------------------------------------------------------- helpers


def make_row(
    option_id: str,
    ret: float | None = None,
    se: float | None = None,
    *,
    pool: str | None = "WIN",
    family: str = "fam-a",
    sources: tuple[str, ...] = ("jra",),
    variants: int = 1,
    status: str | None = None,
    **extra,
) -> LeaderboardRow:
    return LeaderboardRow(
        option_id=option_id,
        family_id=family,
        ticket=449,
        kind=extra.pop("kind", "rule"),
        pool=pool,
        segment=extra.pop("segment", "all"),
        rule=extra.pop("rule", f"flat bet on {option_id}"),
        selection=extra.pop("selection", "decision_time"),
        sources=sources,
        variants_searched=variants,
        status=status if status is not None else ("scored" if ret is not None else "failed"),
        return_per_100=ret,
        se_per_100=se,
        **extra,
    )


def z_of(level: float) -> float:
    return NormalDist().inv_cdf(level)


def expected_se(row: LeaderboardRow) -> float | None:
    """The ticket's se rule, written independently of the module."""
    if row.se_per_100 is not None:
        return row.se_per_100
    if row.ci_level is None:
        return None
    z = z_of(row.ci_level)
    if row.ci_lo is not None and row.ci_hi is not None:
        return (row.ci_hi - row.ci_lo) / (2 * z)
    if row.return_per_100 is None:
        return None
    if row.ci_lo is not None:
        return abs(row.return_per_100 - row.ci_lo) / z
    if row.ci_hi is not None:
        return abs(row.ci_hi - row.return_per_100) / z
    return None


def dersimonian_laird(xs: list[float], ses: list[float]) -> tuple[float, float, list[float]]:
    """(tau^2, mu, shrunk values) from the ticket's formulas, for k >= 2 rows."""
    k = len(xs)
    w = [1.0 / s**2 for s in ses]
    sw = sum(w)
    xbar = sum(wi * xi for wi, xi in zip(w, xs, strict=True)) / sw
    q = sum(wi * (xi - xbar) ** 2 for wi, xi in zip(w, xs, strict=True))
    c = sw - sum(wi**2 for wi in w) / sw
    tau2 = max(0.0, (q - (k - 1)) / c)
    mu = sum(x / (s**2 + tau2) for x, s in zip(xs, ses, strict=True)) / sum(
        1.0 / (s**2 + tau2) for s in ses
    )
    shrunk = [mu + tau2 / (tau2 + s**2) * (x - mu) for x, s in zip(xs, ses, strict=True)]
    return tau2, mu, shrunk


def expected_shrunk(rows: list[LeaderboardRow]) -> dict[str, float | None]:
    """option id -> the shrunk value the ticket's formulas give, per pool."""
    out: dict[str, float | None] = {r.option_id: None for r in rows}
    pools: dict[str, list[LeaderboardRow]] = {}
    for r in rows:
        if r.return_per_100 is not None and r.pool is not None and expected_se(r) is not None:
            pools.setdefault(r.pool, []).append(r)
    for members in pools.values():
        if len(members) == 1:
            out[members[0].option_id] = members[0].return_per_100
            continue
        _, _, values = dersimonian_laird(
            [m.return_per_100 for m in members], [expected_se(m) for m in members]
        )
        for m, v in zip(members, values, strict=True):
            out[m.option_id] = v
    return out


def by_id(rows: list[LeaderboardRow]) -> dict[str, LeaderboardRow]:
    return {r.option_id: r for r in rows}


def assert_shrunk(actual: list[LeaderboardRow], expected: dict[str, float | None]) -> None:
    got = by_id(actual)
    assert set(got) == set(expected)
    for option_id, value in expected.items():
        if value is None:
            assert got[option_id].shrunk_per_100 is None, option_id
        else:
            assert got[option_id].shrunk_per_100 == pytest.approx(value, abs=TOL), option_id


def ranked_ids(ranked: list[RankedRow]) -> list[tuple[int | None, str]]:
    return [(r.rank, r.row.option_id) for r in ranked]


def table_line(text: str, option_id: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.lstrip().startswith("|") and option_id in ln]
    assert len(lines) == 1, f"{option_id}: expected one table line, got {lines}"
    return lines[0]


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def has_number(text: str, number: int) -> bool:
    return re.search(rf"(?<![\d.,]){number}(?![\d.,]*\d)", text) is not None


def has_n(text: str, n: int) -> bool:
    """Whether a count appears, with or without thousands separators."""
    return str(n) in text or f"{n:,}" in text


def withheld_section(text: str) -> tuple[str, str]:
    """(text before the withheld heading, the section's own text up to the next heading)."""
    lines = text.splitlines()
    heads = [
        i for i, ln in enumerate(lines) if re.fullmatch(r"#+\s*withheld \(d21\)\s*", ln.lower())
    ]
    assert len(heads) == 1, "expected one `## Withheld (D21)` heading"
    start = heads[0]
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("#")), len(lines))
    return "\n".join(lines[:start]), "\n".join(lines[start + 1 : end])


def tree(root: Path) -> dict[str, bytes | None]:
    return {
        str(p.relative_to(root)): (None if p.is_dir() else p.read_bytes())
        for p in sorted(root.rglob("*"))
    }


def three_win_rows(**extra) -> list[LeaderboardRow]:
    return [
        make_row("win-70", 70.0, 5.0, **extra),
        make_row("win-80", 80.0, 5.0, **extra),
        make_row("win-90", 90.0, 5.0, **extra),
    ]


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A synthetic repository that ``write_report``'s default root resolves to."""
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n', encoding="utf-8")
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    return root


@pytest.fixture
def sources_seen(monkeypatch) -> list[tuple[str, ...]]:
    """The ``sources`` of every ``write_report`` call, which still writes (or refuses)."""
    seen: list[tuple[str, ...]] = []
    real = reports.write_report

    def spy(path, text, *, sources, repo_root=None):
        seen.append(tuple(sources))
        return real(path, text, sources=sources, repo_root=repo_root)

    monkeypatch.setattr(reports, "write_report", spy)
    monkeypatch.setattr(leaderboard, "write_report", spy, raising=False)
    return seen


# --------------------------------------------------------------------------- ranking (broad)


def test_rows_rank_by_shrunk_return_with_ci_and_variants_beside_it():
    # Four measured WIN rows, se from 95% one-sided bounds; the noisy 97 shrinks below 88.
    spec = {  # option id: (return, ci_lo, ci_hi, variants searched, n)
        "opt-alpha": (97.0, 40.0, 150.0, 13, 410),
        "opt-bravo": (88.0, 84.71, 91.29, 17, 5230),
        "opt-charlie": (81.5, 76.12, 86.88, 29, 2140),
        "opt-delta": (69.0, 65.83, 72.17, 41, 6020),
    }
    measured = [
        make_row(oid, ret, ci_lo=lo, ci_hi=hi, ci_level=0.95, variants=v, n=n, n_unit="bets")
        for oid, (ret, lo, hi, v, n) in spec.items()
    ]
    unmeasured = [
        make_row("opt-zulu", family="fam-a", variants=53),
        make_row("opt-echo", family="fam-b", variants=67),
    ]
    rows = [unmeasured[1], measured[2], measured[0], unmeasured[0], measured[3], measured[1]]

    want = expected_shrunk(rows)
    order = sorted(spec, key=lambda oid: -want[oid])
    assert order == ["opt-bravo", "opt-alpha", "opt-charlie", "opt-delta"]  # premise: not raw

    ranked = rank(shrink(rows))
    assert ranked_ids(ranked) == [(i + 1, oid) for i, oid in enumerate(order)] + [
        (None, "opt-zulu"),
        (None, "opt-echo"),
    ]
    for r in ranked:
        assert isinstance(r, RankedRow)
        if r.rank is not None:
            assert r.row.shrunk_per_100 == pytest.approx(want[r.row.option_id], abs=TOL)

    text = render_markdown(rows, title=TITLE, public=False)
    positions = [text.index(table_line(text, oid)) for oid in [*order, "opt-zulu", "opt-echo"]]
    assert positions == sorted(positions)
    for position, oid in enumerate(order, start=1):
        line = table_line(text, oid)
        ret, lo, hi, variants, n = spec[oid]
        assert cells(line)[0] == str(position)
        assert f"{lo:.2f}" in line and f"{hi:.2f}" in line
        assert f"{ret:.2f}" in line and f"{want[oid]:.2f}" in line
        assert str(variants) in cells(line)
    for oid, variants in (("opt-zulu", 53), ("opt-echo", 67)):
        line = table_line(text, oid)
        assert not cells(line)[0].isdigit()
        assert str(variants) in cells(line)


def test_ranking_orders_measured_rows_by_shrunk_else_raw_and_never_drops_a_row():
    rows = [
        make_row("tie-b", 85.0, pool="WIN", shrunk_per_100=80.0),
        make_row("u-z", family="fam-a"),
        make_row("zero", 0.0, pool="PLACE"),
        make_row("m-raw", 90.0, pool=None),
        make_row("u-a", family="fam-b"),
        make_row("tie-a", 85.0, pool="PLACE", shrunk_per_100=80.0),
        make_row("mid", 60.0, pool="QUINELLA", shrunk_per_100=82.0),
        make_row("u-m", family="fam-a"),
        make_row("tie-raw-hi", 88.0, pool="WIN", shrunk_per_100=80.0),
        make_row("m-hi", 70.0, pool="WIN5", shrunk_per_100=95.0),
    ]
    assert [r.measured for r in rows] == [r.return_per_100 is not None for r in rows]
    assert by_id(rows)["zero"].measured  # a return of 0.0 is measured

    expected = [
        (1, "m-hi"),
        (2, "m-raw"),
        (3, "mid"),
        (4, "tie-raw-hi"),
        (5, "tie-a"),
        (6, "tie-b"),
        (7, "zero"),
        (None, "u-m"),
        (None, "u-z"),
        (None, "u-a"),
    ]
    assert ranked_ids(rank(rows)) == expected
    assert ranked_ids(rank(list(reversed(rows)))) == expected
    assert [r.row for r in rank(rows)] == [by_id(rows)[oid] for _, oid in expected]


# --------------------------------------------------------------------------- shrinkage


def _fixed_pools() -> list[LeaderboardRow]:
    junk = {"shrunk_per_100": 999.0}  # an incoming value is ignored
    return [
        make_row("w1", 62.3, 3.1, **junk),
        make_row("w2", 71.8, 7.4, **junk),
        make_row("w3", 80.4, 2.2, **junk),
        make_row("w4", 95.1, 12.5, **junk),
        make_row("w5", 77.0, ci_lo=70.0, ci_hi=84.0, ci_level=0.95, **junk),
        make_row("w-no-se", 99.0, **junk),
        make_row("w-unmeasured", **junk),
        make_row("p1", 88.2, 1.5, pool="PLACE", **junk),
        make_row("p2", 64.0, ci_lo=62.0, ci_level=0.9, pool="PLACE", **junk),
        make_row("p3", 79.9, 6.0, pool="PLACE", **junk),
        make_row("n1", 50.0, 4.0, pool="WIN5", **junk),
        make_row("n2", 51.0, ci_hi=65.0, ci_level=0.975, pool="WIN5", **junk),
        make_row("n3", 49.5, 6.0, pool="WIN5", **junk),
        make_row("f1", 41.0, 9.0, pool="FORECAST", **junk),
        make_row("no-pool", 77.7, 3.0, pool=None, **junk),
    ]


def _random_pools(seed: int) -> list[LeaderboardRow]:
    rng = random.Random(seed)
    rows = []
    for p in range(6):
        for i in range(rng.randint(2, 7)):
            x = round(rng.uniform(40.0, 140.0), 3)
            how = rng.choice(["se", "two-sided", "lower", "upper"])
            level = rng.choice([0.9, 0.95, 0.975])
            a, b = rng.uniform(0.5, 30.0), rng.uniform(0.5, 30.0)
            extra = {
                "se": {"se": rng.uniform(0.5, 25.0)},
                "two-sided": {"ci_lo": x - a, "ci_hi": x + b, "ci_level": level},
                "lower": {"ci_lo": x - a, "ci_level": level},
                "upper": {"ci_hi": x + b, "ci_level": level},
            }[how]
            rows.append(make_row(f"r{p}-{i}", x, pool=f"POOL{p}", **extra))
    return rows


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(_fixed_pools(), id="fixed"),
        pytest.param(_random_pools(449), id="random-449"),
        pytest.param(_random_pools(2026), id="random-2026"),
    ],
)
def test_shrinkage_matches_dersimonian_laird_per_pool(rows):
    out = shrink(rows)
    assert len(out) == len(rows)
    assert_shrunk(out, expected_shrunk(rows))
    for before in rows:  # copies: only shrunk_per_100 changes
        after = by_id(out)[before.option_id]
        assert after == dataclasses.replace(before, shrunk_per_100=after.shrunk_per_100)
    assert_shrunk(shrink(list(reversed(rows))), expected_shrunk(rows))


def test_the_fixed_pools_cover_both_branches_of_tau2():
    rows = by_id(_fixed_pools())

    def tau2(ids):
        return dersimonian_laird(
            [rows[i].return_per_100 for i in ids], [expected_se(rows[i]) for i in ids]
        )

    assert tau2(["w1", "w2", "w3", "w4", "w5"])[0] > 0
    assert tau2(["p1", "p2", "p3"])[0] > 0
    t2, mu, _ = tau2(["n1", "n2", "n3"])
    assert t2 == 0 and abs(mu - (50.0 + 51.0 + 49.5) / 3) > 0.01  # weighted, not simple mean
    for oid in ("n1", "n2", "n3"):
        assert by_id(shrink(list(rows.values())))[oid].shrunk_per_100 == pytest.approx(mu, abs=TOL)


def test_shrinkage_pulls_rows_toward_the_pool_mean():
    rows = three_win_rows()
    tau2, mu, _ = dersimonian_laird([70.0, 80.0, 90.0], [5.0, 5.0, 5.0])
    assert tau2 == pytest.approx(75.0, abs=TOL) and mu == pytest.approx(80.0, abs=TOL)

    got = by_id(shrink(rows))
    assert [got[o].shrunk_per_100 for o in ("win-70", "win-80", "win-90")] == pytest.approx(
        [72.5, 80.0, 87.5], abs=TOL
    )
    for oid, x in (("win-70", 70.0), ("win-90", 90.0)):  # tau^2 implied by the shrinkage factor
        factor = (got[oid].shrunk_per_100 - 80.0) / (x - 80.0)
        assert factor * 25.0 / (1.0 - factor) == pytest.approx(75.0, abs=1e-6)


def test_pure_noise_shrinks_every_row_to_the_pool_mean():
    rows = [
        make_row("noise-a", 78.0, 5.0),
        make_row("noise-b", 80.0, 5.0),
        make_row("noise-c", 82.0, 5.0),
    ]
    assert dersimonian_laird([78.0, 80.0, 82.0], [5.0] * 3)[0] == 0.0
    out = shrink(rows)
    for r in out:
        assert r.shrunk_per_100 == pytest.approx(80.0, abs=TOL)
    assert ranked_ids(rank(out)) == [(1, "noise-c"), (2, "noise-b"), (3, "noise-a")]


def test_a_noisier_row_shrinks_more():
    ses = [2.0, 5.0, 10.0, 20.0]
    rows = [make_row(f"hi-se{int(s)}", 90.0, s) for s in ses]
    rows += [make_row(f"lo-se{int(s)}", 70.0, s) for s in ses]  # mirrors pin the pool mean at 80
    tau2, mu, _ = dersimonian_laird([r.return_per_100 for r in rows], [r.se_per_100 for r in rows])
    assert tau2 > 0 and mu == pytest.approx(80.0, abs=TOL)

    out = shrink(rows)
    assert_shrunk(out, expected_shrunk(rows))
    got = by_id(out)
    distance = [got[f"hi-se{int(s)}"].shrunk_per_100 - 80.0 for s in ses]
    assert all(0 < d < 10.0 for d in distance)
    assert distance == sorted(distance, reverse=True)  # larger se, nearer the mean
    assert distance[-1] < distance[0] - 1.0  # se 20 ends well nearer the mean than se 2


def test_pools_shrink_separately():
    win = three_win_rows()
    place = [
        make_row("place-a", 55.0, 3.0, pool="PLACE"),
        make_row("place-b", 75.0, 6.0, pool="PLACE"),
        make_row("place-c", 61.0, 2.5, pool="PLACE"),
        make_row("place-d", 300.0, 1.0, pool="PLACE"),
    ]
    together = dersimonian_laird(
        [r.return_per_100 for r in win + place], [r.se_per_100 for r in win + place]
    )[2]
    assert abs(together[0] - 72.5) > 1.0  # premise: one pooled fit would move the WIN rows

    alone = by_id(shrink(win))
    mixed = shrink(place[:2] + win + place[2:])
    for oid, value in zip(("win-70", "win-80", "win-90"), (72.5, 80.0, 87.5), strict=True):
        assert by_id(mixed)[oid].shrunk_per_100 == pytest.approx(value, abs=TOL)
        assert by_id(mixed)[oid].shrunk_per_100 == pytest.approx(alone[oid].shrunk_per_100, abs=TOL)
    assert_shrunk(mixed, expected_shrunk(win) | expected_shrunk(place))


def test_a_lone_row_and_a_row_without_se_keep_their_raw_return():
    rows = [
        make_row("lone-forecast", 55.0, 4.0, pool="FORECAST"),
        make_row("win-no-se", 85.0),
        *three_win_rows(),
        make_row("place-se", 60.0, 4.0, pool="PLACE"),
        make_row("place-no-se", 90.0, pool="PLACE"),
        make_row("no-pool", 77.0, 3.0, pool=None),
    ]
    got = by_id(shrink(rows))
    assert got["lone-forecast"].shrunk_per_100 == pytest.approx(55.0, abs=TOL)
    assert got["place-se"].shrunk_per_100 == pytest.approx(60.0, abs=TOL)  # its pool has k = 1
    for oid in ("win-no-se", "place-no-se", "no-pool"):
        assert got[oid].shrunk_per_100 is None
        assert got[oid].return_per_100 == by_id(rows)[oid].return_per_100
    assert [got[o].shrunk_per_100 for o in ("win-70", "win-80", "win-90")] == pytest.approx(
        [72.5, 80.0, 87.5], abs=TOL
    )
    assert [oid for _, oid in ranked_ids(rank(list(got.values())))] == [
        "place-no-se",  # raw 90
        "win-90",  # shrunk 87.5
        "win-no-se",  # raw 85
        "win-80",  # shrunk 80
        "no-pool",  # raw 77
        "win-70",  # shrunk 72.5
        "place-se",  # 60
        "lone-forecast",  # 55
    ]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param({"ret": 100.0, "ci_lo": 90.0, "ci_hi": 110.0, "ci_level": 0.95},
                     20 / (2 * z_of(0.95)), id="both-bounds"),
        pytest.param({"ret": 100.0, "ci_lo": 90.0, "ci_level": 0.95}, 10 / z_of(0.95), id="lower"),
        pytest.param({"ret": 100.0, "ci_hi": 110.0, "ci_level": 0.95}, 10 / z_of(0.95), id="upper"),
        pytest.param({"ret": 50.0, "ci_lo": 41.0, "ci_hi": 63.0, "ci_level": 0.975},
                     22 / (2 * z_of(0.975)), id="asymmetric-both"),
        pytest.param({"ret": 50.0, "ci_lo": 41.0, "ci_level": 0.975},
                     9 / z_of(0.975), id="asym-lower"),
        pytest.param({"ret": 50.0, "ci_hi": 63.0, "ci_level": 0.975},
                     13 / z_of(0.975), id="asym-upper"),
        pytest.param({"ret": 12.0, "ci_lo": 0.0, "ci_level": 0.9}, 12 / z_of(0.9), id="zero-bound"),
        pytest.param({"ret": 100.0, "se": 3.3, "ci_lo": 90.0, "ci_hi": 110.0, "ci_level": 0.95},
                     3.3, id="se-given-wins"),
        pytest.param({"ret": 100.0}, None, id="nothing"),
    ],
)  # fmt: skip
def test_se_comes_from_one_sided_bounds_when_not_given(fields, expected):
    fields = dict(fields)
    row = make_row("se-row", fields.pop("ret"), fields.pop("se", None), **fields)
    if expected is None:
        assert row.se() is None
    else:
        assert row.se() == pytest.approx(expected, abs=TOL)


def test_the_ticket_se_example_is_6_0796():
    both = make_row("both", 100.0, ci_lo=90.0, ci_hi=110.0, ci_level=0.95)
    lower = make_row("lower", 100.0, ci_lo=90.0, ci_level=0.95)
    assert both.se() == pytest.approx(20 / (2 * 1.6448536), abs=1e-6)
    assert both.se() == pytest.approx(6.0796, abs=1e-4)
    assert lower.se() == pytest.approx(both.se(), abs=TOL)


# --------------------------------------------------------------------------- withholding


@pytest.mark.parametrize(
    ("sources", "licensed", "withheld"),
    [
        pytest.param(("JRA-VAN",), False, True, id="JRA-VAN"),
        pytest.param(("jrdb-kyi",), False, True, id="jrdb-kyi"),
        pytest.param(("netkeiba", "JRDB"), False, True, id="second-source"),
        pytest.param(("ＪＲＡ－ＶＡＮ",), False, True, id="fullwidth"),
        pytest.param(("Jra Van",), False, True, id="spaced"),
        pytest.param(("jra",), True, True, id="licensed-flag"),
        pytest.param(("jra",), False, False, id="jra"),
        pytest.param(("netkeiba", "jra"), False, False, id="netkeiba-jra"),
        pytest.param(("nar",), False, False, id="nar"),
    ],
)
def test_a_withheld_source_in_any_spelling_withholds_the_row(sources, licensed, withheld):
    row = make_row("candidate", 75.0, 5.0, sources=sources, licensed=licensed)
    assert row.withheld == withheld
    anchor = make_row("anchor", 80.0, 5.0)
    kept = {r.option_id for r in public_view([anchor, row])}
    assert kept == ({"anchor"} if withheld else {"anchor", "candidate"})


def test_the_public_view_reshrinks_without_withheld_rows():
    public = three_win_rows()
    secret = make_row("win-secret", 200.0, 5.0, sources=("jrdb",))
    rows = [*public, secret]

    view = public_view(rows)
    assert {r.option_id for r in view} == {"win-70", "win-80", "win-90"}
    assert [by_id(view)[o].shrunk_per_100 for o in ("win-70", "win-80", "win-90")] == pytest.approx(
        [72.5, 80.0, 87.5], abs=TOL
    )
    private = shrink(rows)
    assert_shrunk(private, expected_shrunk(rows))
    for oid, public_value in (("win-70", 72.5), ("win-90", 87.5)):
        assert abs(by_id(private)[oid].shrunk_per_100 - public_value) > 1.0
    # A view of already-shrunk rows reshrinks: the private values do not leak through.
    assert_shrunk(public_view(private), {"win-70": 72.5, "win-80": 80.0, "win-90": 87.5})


WITHHELD_SETS = [
    pytest.param([make_row("x-win", 200.0, 5.0, sources=("jrdb",))], id="same-pool"),
    pytest.param(
        [make_row("x-lic", 10.0, 1.0, pool="PLACE", sources=("jra",), licensed=True)],
        id="licensed-place",
    ),
    pytest.param(
        [
            make_row("x-trio", 33.0, 2.0, pool="TRIO", sources=("JRA-VAN",)),
            make_row("x-nopool", 150.0, 2.0, pool=None, sources=("jrdb-kyi",)),
            make_row("x-failed", sources=("jrdb",), family="fam-x"),
        ],
        id="new-pool-no-pool-unmeasured",
    ),
]


@pytest.mark.parametrize("secret", WITHHELD_SETS)
def test_adding_withheld_rows_changes_no_public_shrunk_value_or_rank(secret):
    base = [*_fixed_pools(), make_row("u-1", family="fam-z")]
    rows = base[:4] + secret + base[4:]

    def summary(ranked):
        return [(r.rank, r.row.option_id, r.row.shrunk_per_100) for r in ranked]

    alone, mixed = summary(rank(public_view(base))), summary(rank(public_view(rows)))
    assert [(a, b) for a, b, _ in mixed] == [(a, b) for a, b, _ in alone]
    for (_, _, x), (_, _, y) in zip(mixed, alone, strict=True):
        assert x == y or x == pytest.approx(y, abs=1e-12)

    text_alone = render_markdown(base, title=TITLE, public=True)
    text_mixed = render_markdown(rows, title=TITLE, public=True)
    for r in base:
        assert table_line(text_mixed, r.option_id) == table_line(text_alone, r.option_id)


def test_the_public_text_carries_no_withheld_figure():
    public = [
        make_row("win-70", 70.0, 5.0, ci_lo=61.78, ci_hi=78.22, ci_level=0.95, n=1200,
                 n_unit="bets", variants=3),
        make_row("win-80", 80.0, 5.0, ci_lo=71.78, ci_hi=88.22, ci_level=0.95, n=1350,
                 n_unit="bets", variants=3),
        make_row("win-90", 90.0, 5.0, ci_lo=81.78, ci_hi=98.22, ci_level=0.95, n=1500,
                 n_unit="bets", variants=3),
    ]  # fmt: skip
    secret = make_row(
        "opt-secret", 123.456, family="fam-secret", sources=("jrdb-zq7feed",), ci_lo=110.111,
        ci_hi=135.789, ci_level=0.95, n=98765, n_unit="bets", n_meetings=77, payout_rate=0.8,
        rule="zq7 rule in words", note="zq7 note", variants=5,
    )  # fmt: skip
    licensed = make_row(
        "opt-licensed", 64.321, 2.0, pool="PLACE", family="fam-licensed", sources=("jra",),
        licensed=True, n=54321, n_unit="races", variants=5,
    )  # fmt: skip
    rows = [public[0], secret, public[1], licensed, public[2]]

    private_values = expected_shrunk(rows)
    forbidden = {"123.456", "123.46", "110.11", "135.79", "98765", "64.32", "54321", "zq7"}
    forbidden |= {"98,765", "54,321"}
    forbidden |= {f"{private_values[o]:.2f}" for o in ("win-70", "win-80", "win-90", "opt-secret")}
    assert not forbidden & {"72.50", "80.00", "87.50"}  # premise: private values differ

    text = render_markdown(rows, title=TITLE, public=True)
    for figure in sorted(forbidden):
        assert figure not in text, figure
    for figure in ("72.50", "80.00", "87.50", "61.78", "98.22"):
        assert figure in text, figure

    before, section = withheld_section(text)
    for oid, family in (("opt-secret", "fam-secret"), ("opt-licensed", "fam-licensed")):
        assert oid in section and family in section
        assert oid not in before
    for r in public:
        assert r.option_id in before and r.option_id not in section

    private = render_markdown(rows, title=TITLE, public=False)
    assert "123.46" in private and has_n(private, 98765)  # the private text holds every row


def test_variants_searched_total_sums_distinct_families_including_withheld():
    rows = [
        make_row("a-1", 70.0, 5.0, family="fam-a", variants=113),
        make_row("a-2", 80.0, 5.0, family="fam-a", variants=113),
        make_row("b-1", 90.0, 5.0, family="fam-b", variants=227),
        make_row("w-1", 65.0, 5.0, family="fam-w", variants=451, sources=("jrdb",)),
    ]
    for public in (True, False):
        text = render_markdown(rows, title=TITLE, public=public)
        assert has_number(text, 113 + 227 + 451), public
        assert not has_number(text, 113 + 113 + 227 + 451), public  # not summed over rows
    assert not has_number(render_markdown(rows, title=TITLE, public=True), 451)


# --------------------------------------------------------------------------- writers


def _rows_with(secret: LeaderboardRow) -> list[LeaderboardRow]:
    return [*three_win_rows(n=1200, n_unit="bets"), secret]


SECRETS = [
    make_row("opt-secret", 123.456, 4.0, sources=("jrdb",), n=98765, n_unit="bets"),
    make_row("opt-secret", 123.456, 4.0, sources=("netkeiba", "JRA-VAN"), n=98765, n_unit="bets"),
    make_row("opt-secret", 123.456, 4.0, sources=("jra",), licensed=True, n=98765, n_unit="bets"),
]


def test_the_private_leaderboard_refuses_the_repository(repo, tmp_path):
    for secret in SECRETS:
        rows = _rows_with(secret)
        before = tree(repo)
        with pytest.raises(LicensedOutputInRepo):
            write_leaderboard(repo / "docs" / "x.md", rows, title=TITLE, public=False)
        assert tree(repo) == before

        target = tmp_path / "data" / "reports" / f"{secret.sources[-1]}-{secret.licensed}.md"
        assert write_leaderboard(target, rows, title=TITLE, public=False) == target
        text = target.read_text(encoding="utf-8")
        assert "123.46" in text and has_n(text, 98765)
        assert tree(repo) == before


def test_the_private_leaderboard_refuses_the_real_repository(tmp_path):
    root = env.repo_root()
    docs = root / "docs"
    assert docs.is_dir()
    name = f"_t449_probe_{uuid.uuid4().hex}.md"
    target = docs / name
    assert not target.exists()
    try:
        with pytest.raises(LicensedOutputInRepo):
            write_leaderboard(target, _rows_with(SECRETS[0]), title=TITLE, public=False)
        assert [p.name for p in docs.iterdir() if name in p.name] == []
    finally:
        for p in docs.iterdir():
            if name in p.name:
                p.unlink()


def test_the_private_writer_passes_every_rows_sources_to_write_report(tmp_path, sources_seen):
    rows = [
        make_row("s-1", 70.0, 5.0, sources=("jra",)),
        make_row("s-2", 80.0, 5.0, sources=("netkeiba", "jra")),
        make_row("s-3", 90.0, 5.0, pool="PLACE", sources=("nar",)),
        make_row("s-4", 60.0, 5.0, sources=("jrdb",)),
        make_row("s-5", sources=("JRA-VAN", "jra")),
    ]
    target = tmp_path / "private" / "leaderboard.md"
    assert write_leaderboard(target, rows, title=TITLE, public=False) == target
    assert target.is_file()
    assert len(sources_seen) == 1
    assert set(sources_seen[0]) >= {"jra", "netkeiba", "nar", "jrdb", "JRA-VAN"}


def test_the_public_writer_writes_only_public_rows_into_the_repository(repo, sources_seen):
    public = [
        make_row("s-1", 70.0, 5.0, sources=("jra",), n=1200, n_unit="bets"),
        make_row("s-2", 80.0, 5.0, sources=("netkeiba", "jra"), n=1350, n_unit="bets"),
        make_row("s-3", 90.0, 5.0, sources=("nar",), n=1500, n_unit="bets"),
    ]
    secrets = [
        make_row("opt-secret", 123.456, 5.0, sources=("jrdb-kyi",), n=98765, n_unit="bets",
                 ci_lo=110.111, ci_hi=135.789, ci_level=0.95),
        make_row("opt-licensed", 64.321, 2.0, pool="PLACE", sources=("keibalab",), licensed=True,
                 n=54321, n_unit="races"),
    ]  # fmt: skip
    cases = [
        ("mixed.md", [public[0], secrets[0], public[1], secrets[1], public[2]], public),
        ("public-only.md", public, public),
        ("withheld-only.md", secrets, []),
    ]
    for name, rows, kept in cases:
        sources_seen.clear()
        target = repo / "docs" / "leaderboards" / name
        assert write_leaderboard(target, rows, title=TITLE, public=True) == target
        assert len(sources_seen) == 1
        assert set(sources_seen[0]) == {s for r in kept for s in r.sources}, name

        text = target.read_text(encoding="utf-8")
        assert text == render_markdown(rows, title=TITLE, public=True)
        for figure in (*("123.456", "123.46", "110.11", "135.79", "98765", "98,765"),
                       *("64.32", "54321", "54,321")):  # fmt: skip
            assert figure not in text, (name, figure)
        if kept:
            for figure in ("72.50", "80.00", "87.50"):
                assert figure in text, (name, figure)
    assert sorted(p.name for p in (repo / "docs" / "leaderboards").iterdir()) == sorted(
        name for name, _, _ in cases
    )


# --------------------------------------------------------------------------- refusals


GOOD = {
    "option_id": "opt-good",
    "family_id": "fam-good",
    "ticket": 449,
    "kind": "rule",
    "pool": "WIN",
    "segment": "class:maiden",
    "rule": "flat ¥100 on the favourite",
    "selection": "decision_time",
    "sources": ("jra",),
    "variants_searched": 3,
    "status": "scored",
    "return_per_100": 90.0,
    "ci_lo": 80.0,
    "ci_hi": 100.0,
    "ci_level": 0.95,
    "n": 400,
    "n_unit": "bets",
}


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"kind": "bogus"}, id="kind"),
        pytest.param({"selection": "hindsight"}, id="selection"),
        pytest.param({"ci_lo": 101.0, "ci_hi": 99.0}, id="ci-lo-above-ci-hi"),
        pytest.param({"n": -1}, id="negative-n"),
        pytest.param({"option_id": ""}, id="empty-option-id"),
        pytest.param({"option_id": "   "}, id="blank-option-id"),
        pytest.param({"variants_searched": 0}, id="variants-0"),
        pytest.param({"variants_searched": -2}, id="variants-negative"),
        pytest.param({"n_unit": "horses"}, id="n-unit"),
    ],
)
def test_bad_rows_are_refused(bad):
    LeaderboardRow(**GOOD)  # premise: the base row is valid
    with pytest.raises(RowRefused):
        LeaderboardRow(**(GOOD | bad))
    assert issubclass(RowRefused, ValueError)
    with pytest.raises(RowRefused):
        rank([LeaderboardRow(**GOOD), LeaderboardRow(**(GOOD | {"return_per_100": 50.0}))])


def test_every_listed_kind_selection_and_unit_is_accepted():
    assert set(KINDS) == {"rule", "comparator", "cell", "model", "member", "other"}
    assert set(SELECTIONS) == {"decision_time", "card_only", "look_ahead", "unknown"}
    assert set(N_UNITS) == {"bets", "races", "days"}
    for kind in KINDS:
        assert LeaderboardRow(**(GOOD | {"kind": kind})).kind == kind
    for selection in SELECTIONS:
        assert LeaderboardRow(**(GOOD | {"selection": selection})).selection == selection
    for unit in (*N_UNITS, None):
        assert LeaderboardRow(**(GOOD | {"n_unit": unit})).n_unit == unit


# --------------------------------------------------------------------------- JSON


def test_rows_round_trip_through_json():
    rows = [
        LeaderboardRow(
            option_id="opt-full",
            family_id="fam-full",
            ticket=449,
            kind="comparator",
            pool="WIN5",
            segment="class:maiden",
            rule="box of 4 around the favourite, ¥100 flat",
            selection="card_only",
            sources=("jrdb-kyi", "jra"),
            variants_searched=12,
            status="scored",
            return_per_100=0.0,
            ci_lo=0.0,
            ci_hi=3.25,
            ci_level=0.975,
            se_per_100=1e-3,
            n=31,
            n_unit="days",
            n_meetings=12,
            payout_rate=0.7,
            shrunk_per_100=0.1 + 0.2,
            licensed=True,
            note="勝率 note ¥",
        ),
        LeaderboardRow(
            option_id="opt-minimal",
            family_id="fam-min",
            ticket=None,
            kind="other",
            pool=None,
            segment="all",
            rule="nothing yet",
            selection="unknown",
            sources=(),
            variants_searched=1,
            status="failed",
        ),
        make_row("opt-mid", 87.654321, ci_lo=80.0, ci_level=0.9, n=1200, n_unit="bets"),
    ]
    text = to_json(rows)
    assert isinstance(text, str)
    json.loads(text)  # valid JSON
    back = from_json(text)
    assert back == rows
    assert [type(r) for r in back] == [LeaderboardRow] * 3
    assert isinstance(back[0].sources, tuple) and back[0].sources == ("jrdb-kyi", "jra")
    assert type(back[0].n) is int and type(back[0].ticket) is int
    assert back[0].return_per_100 == 0.0 and back[0].ci_lo == 0.0
    assert back[0].shrunk_per_100 == 0.1 + 0.2
    assert back[1].ticket is None and back[1].pool is None and back[1].return_per_100 is None
    assert back[0].withheld and not back[2].withheld
    assert from_json(to_json([])) == []
    assert from_json(to_json(shrink(rows))) == shrink(rows)

    with pytest.raises(RowRefused):  # a tampered file is refused like a bad row
        from_json(text.replace('"comparator"', '"bogus"'))
