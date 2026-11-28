"""Ticket #455: `horsetrack japan bets shortlist`, the top of the private leaderboard beside the
day's cap.

Contract pinned here, module ``horsetrack.countries.japan.bets.shortlist`` and the command on
``bets_app`` (always run through the root CLI, so the registration is tested too):

- ``shortlist(rows, *, top) -> list[RankedRow]``: ``rows`` are as ``build_rows()`` returns them,
  UNSHRUNK with ``se_per_100`` set. ``shortlist`` itself shrinks the whole row set within pool
  (``leaderboard.shrink``: a pool's shrinkage needs every row of the pool, not just the top),
  ranks it (``leaderboard.rank``) and keeps the first ``top``: the ``top`` best measured, ranked
  rows of ``rank(shrink(rows))``, in rank order (ranks 1..k), each with its shrunk return, as the
  private leaderboard ranks them. Unmeasured and descriptive (unranked) rows are never on it,
  however large ``top`` is.
- ``shortlist_text(ranked, *, day, check, limits, played) -> str``: one entry per option, in rank
  order, with its option id, pool, segment, rule, selection, shrunk return per 100 yen, CI and
  variants searched, and its played bets (n, staked, return per 100 yen from ``played``, a
  ``played_by_option`` frame, or None when nothing was played); a look-ahead option carries the
  label 'look-ahead'. The head shows the day's staked total, 'remaining ¥N' (the day cap minus the
  staked total) and 'flat stake: ¥N' or 'flat stake: not set'. The CI shows as the private
  leaderboard shows it: both bounds, or the one bound a row has with its sense (>= for a lower
  bound, <= for an upper one), and the row's own ``ci_level``, never a fixed 95%, never 'n/a'
  when a bound exists.
- ``bets shortlist --date D [--top 10]``: rows from ``leaderboard_build.build_rows()``, the day's
  staked total from its bet log, the played bets of every ``<day>.settled.json`` under the bets
  dir. It prints and writes nothing: not a file, not a directory, not even the registry folder of
  a fresh data root.

Everything is synthetic. Most command tests patch ``build_rows`` (it reads the private registry);
the last tests run the real ``build_rows`` path, unpatched, on an empty and on a seeded temp data
dir. The bet log and the settled files are written through the project's own APIs.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.leaderboard import LeaderboardRow, RankedRow, rank, shrink
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord
from horsetrack.countries.japan.bets.cli import settled_path
from horsetrack.countries.japan.bets.log import BetLog, CapCheck, Limits, bet_log_path
from horsetrack.countries.japan.bets.settle import played_by_option, settle_bets
from horsetrack.countries.japan.bets.shortlist import shortlist, shortlist_text
from horsetrack.countries.japan.research import leaderboard_build

REPO = Path(__file__).resolve().parents[4]
DAY = date(2026, 12, 5)
EARLIER = date(2026, 10, 31)
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
runner = CliRunner()


@pytest.fixture(autouse=True)
def D(tmp_path, monkeypatch) -> Path:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    monkeypatch.chdir(REPO)
    return data


def row(
    option_id: str,
    raw: float | None,
    se: float | None = 5.0,
    *,
    selection: str = "decision_time",
    variants: int = 1,
    pool: str = "WIN",
    segment: str = "all",
    rule: str = "flat",
    descriptive: bool = False,
    ci_level: float | None = 0.95,
    ci: tuple[float | None, float | None] | None = None,
) -> LeaderboardRow:
    """A row as ``build_rows()`` returns it: the raw return and its se, no shrunk value."""
    if ci is None:
        ci = (None, None) if raw is None else (round(raw - 11.1, 1), round(raw + 12.3, 1))
    return LeaderboardRow(
        option_id=option_id,
        family_id=f"fam-{option_id}",
        ticket=None,
        kind="rule",
        pool=pool,
        segment=segment,
        rule=rule,
        selection=selection,
        sources=("jra",),
        variants_searched=variants,
        status="measured" if raw is not None else "unmeasured",
        return_per_100=raw,
        ci_lo=ci[0],
        ci_hi=ci[1],
        ci_level=None if raw is None else ci_level,
        se_per_100=se if raw is not None else None,
        descriptive=descriptive,
    )


# Twelve WIN options, deliberately not in rank order, and three QUINELLA options. The se differs by
# row: o12 has the highest raw return and a huge se, o01 the second highest and a large one, o09
# the lowest raw return and a large one. Shrinkage (within pool, over every row) pulls the noisy
# rows to the pool mean: o12 falls from first on the raw return to sixth, o01 from second to third,
# o09 climbs from last to ninth, and o10 and o11, tenth and eleventh on the raw return, drop out
# of the top ten.
RAW = {
    "o07": 91.4, "o02": 104.2, "o11": 88.6, "o05": 97.8, "o01": 112.5, "o09": 60.0,
    "o12": 131.0, "o03": 101.7, "o08": 93.9, "o04": 99.4, "o10": 86.2, "o06": 95.3,
}  # fmt: skip
SE = {k: round(3.0 + 0.4 * i, 1) for i, k in enumerate(RAW)} | {
    "o12": 30.0,
    "o09": 22.0,
    "o01": 14.0,
}
QUINELLA = {"q1": (76.0, 4.0), "q2": (71.5, 6.0), "q3": (80.2, 9.0)}
UNMEASURED = ["u1", "u2", "u3"]


def board_rows() -> list[LeaderboardRow]:
    """Fifteen measured rows, three unmeasured, one descriptive; the order is scrambled."""
    measured = [
        row(k, RAW[k], SE[k], variants=100 + 7 * i, rule=f"rule-{k}", segment=f"seg-{k}")
        for i, k in enumerate(RAW)
    ]
    quinella = [
        row(k, raw, se, pool="QUINELLA", variants=20 + i, rule=f"rule-{k}", segment=f"seg-{k}")
        for i, (k, (raw, se)) in enumerate(QUINELLA.items())
    ]
    unmeasured = [row(k, None) for k in UNMEASURED]
    descriptive = [row("d1", 500.0, None, descriptive=True)]
    return [
        *unmeasured[:1], *measured[:6], *descriptive, *quinella[:2], *unmeasured[1:],
        *measured[6:], *quinella[2:],
    ]  # fmt: skip


def variants_of(option_id: str) -> int:
    return next(r.variants_searched for r in board_rows() if r.option_id == option_id)


# The private leaderboard's own order and shrunk values, ``rank(shrink(rows))``, derived from the
# library the leaderboard uses and never from the code under test.
LEADERBOARD = [r for r in rank(shrink(board_rows())) if r.rank is not None]
IN_RANK_ORDER = [r.row.option_id for r in LEADERBOARD]
SHRUNK = {r.row.option_id: r.row.shrunk_per_100 for r in LEADERBOARD}
RAW_ORDER = [r.row.option_id for r in rank(board_rows()) if r.rank is not None]
N_MEASURED = len(RAW) + len(QUINELLA)


def test_premise_shrinkage_changes_the_order_and_the_top_ten():
    """The fixture is only a test when shrinking reorders: the raw return and the shrunk value
    disagree about who is first, who is in the top ten, and the QUINELLA rows' values."""
    assert IN_RANK_ORDER[:10] != RAW_ORDER[:10]
    assert RAW_ORDER[0] == "o12" and IN_RANK_ORDER.index("o12") >= 4
    assert "o09" in IN_RANK_ORDER[:10] and "o09" not in RAW_ORDER[:10]
    assert SHRUNK["o12"] != RAW["o12"] and abs(SHRUNK["o12"] - RAW["o12"]) > 5
    # a pool of its own: shrunk over its three rows, not over the twelve of WIN
    assert 70 < SHRUNK["q1"] < 82


# ---------------------------------------------------------------- shortlist


def test_the_shortlist_takes_the_top_ranked_measured_options():
    rows = board_rows()
    got = shortlist(rows, top=10)
    assert [r.row.option_id for r in got] == IN_RANK_ORDER[:10]
    assert [r.rank for r in got] == list(range(1, 11))
    assert all(isinstance(r, RankedRow) for r in got)
    assert not {r.row.option_id for r in got} & set(UNMEASURED)
    # the private leaderboard's own ranking of the shrunk rows, not a sort of the raw return
    assert got == rank(shrink(rows))[:10]
    assert [r.row.option_id for r in got] != RAW_ORDER[:10]
    # every listed row carries its shrunk value, set from its pool
    for item in got:
        assert item.row.shrunk_per_100 == pytest.approx(SHRUNK[item.row.option_id])


def test_the_shortlist_shrinks_over_every_row_before_it_truncates():
    """Shrinkage is within pool over all rows: shrinking only the top few would give other values
    (here o12's, whose shrunk value falls with the rows that were dropped)."""
    top3 = shortlist(board_rows(), top=3)
    assert [r.row.option_id for r in top3] == IN_RANK_ORDER[:3]
    for item in top3:
        assert item.row.shrunk_per_100 == pytest.approx(SHRUNK[item.row.option_id])
    only_top = rank(shrink([r.row for r in LEADERBOARD[:5]]))
    assert [r.row.shrunk_per_100 for r in only_top] != pytest.approx(
        [SHRUNK[r.row.option_id] for r in only_top]
    ), "premise: the pool of the first five shrinks differently from the whole pool"


@pytest.mark.parametrize("top", [1, 3, 7, N_MEASURED, 50])
def test_the_shortlist_is_as_long_as_top_up_to_the_measured_options(top):
    got = shortlist(board_rows(), top=top)
    want = IN_RANK_ORDER[: min(top, N_MEASURED)]
    assert [r.row.option_id for r in got] == want
    assert [r.rank for r in got] == list(range(1, len(want) + 1))


def test_the_shortlist_never_holds_unmeasured_or_unranked_rows():
    got = shortlist(board_rows(), top=50)
    ids = {r.row.option_id for r in got}
    assert ids == set(RAW) | set(QUINELLA), "the measured, ranked rows and nothing else"
    assert all(r.rank is not None and r.row.measured for r in got)
    assert shortlist([row(k, None) for k in UNMEASURED], top=10) == []


# ---------------------------------------------------------------- shortlist_text


def check_of(staked: int, cap: int = 30_000) -> CapCheck:
    return CapCheck(fits=staked + 1_000 <= cap, staked_yen=staked, remaining_yen=cap - staked,
                    off_flat=False)  # fmt: skip


def text_of(ranked, *, staked=0, flat=None, played=None, day=DAY) -> str:
    out = shortlist_text(
        ranked,
        day=day,
        check=check_of(staked),
        limits=Limits(day_cap_yen=30_000, flat_stake_yen=flat),
        played=played,
    )
    return _ANSI.sub("", out)


def blocks(text: str, ids: list[str]) -> dict[str, str]:
    """The lines of ``text`` from the line of each option id to the line of the next one, with the
    ids required to appear in the given order."""
    starts = []
    for option_id in ids:
        found = re.search(rf"(?<![\w-]){re.escape(option_id)}(?![\w-])", text)
        assert found, f"{option_id} is not in the shortlist:\n{text}"
        starts.append(text.rfind("\n", 0, found.start()) + 1)
    assert starts == sorted(starts) and len(set(starts)) == len(starts), (
        f"the options are not one per line group in rank order:\n{text}"
    )
    ends = [*starts[1:], len(text)]
    return {i: text[a:b] for i, a, b in zip(ids, starts, ends, strict=True)}


def figure(block: str, value: float) -> bool:
    """Whether ``block`` prints ``value`` to one or two decimals."""
    return re.search(rf"(?<![\d.]){re.escape(f'{value:.1f}')}0?(?!\d)", block) is not None


def whole(block: str, value: int) -> bool:
    flat = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", block)
    return re.search(rf"(?<![\d.]){value}(?![\d.])", flat) is not None


def test_each_option_shows_its_rank_fields_in_rank_order():
    ranked = shortlist(board_rows(), top=50)
    text = text_of(ranked)
    got = blocks(text, IN_RANK_ORDER)
    by_id = {r.option_id: r for r in board_rows()}
    for k in IN_RANK_ORDER:
        block, source = got[k], by_id[k]
        assert figure(block, SHRUNK[k]), f"{k}: shrunk return missing:\n{block}"
        assert "n/a" not in block, f"{k}: a figure is missing:\n{block}"
        assert figure(block, source.ci_lo) and figure(block, source.ci_hi), (
            f"{k}: CI missing:\n{block}"
        )
        assert whole(block, variants_of(k)), f"{k}: variants searched missing:\n{block}"
        assert source.pool in block and f"seg-{k}" in block and f"rule-{k}" in block, block
        assert re.search(r"decision[-_ ]time", block, re.I), f"{k}: selection missing:\n{block}"
    # a noisy row shows its shrunk value, not its raw return
    assert not figure(got["o12"], RAW["o12"]) and not figure(got["o09"], RAW["o09"])
    for k in ["d1", *UNMEASURED]:
        assert not re.search(rf"(?<![\w-]){k}(?![\w-])", text), f"{k} must not be listed"


@pytest.mark.parametrize(
    ("staked", "remaining"),
    [(12_000, 18_000), (0, 30_000), (29_500, 500), (7_300, 22_700)],
)
def test_the_shortlist_shows_the_remaining_cap_and_flat_stake(staked, remaining):
    ranked = shortlist(board_rows(), top=3)
    out = text_of(ranked, staked=staked, flat=None)
    assert f"remaining ¥{remaining:,}" in out, out
    assert "flat stake: not set" in out, out
    if staked:
        assert f"{staked:,}" in out, f"the day's staked total is not shown:\n{out}"


@pytest.mark.parametrize("flat", [1_000, 500])
def test_the_shortlist_shows_the_flat_stake_when_set(flat):
    out = text_of(shortlist(board_rows(), top=2), staked=12_000, flat=flat)
    assert "flat stake: not set" not in out
    line = next((ln for ln in out.splitlines() if "flat stake" in ln), "")
    assert whole(line, flat), f"the flat stake {flat} is not on its line:\n{out}"
    assert "remaining ¥18,000" in out


def settled_frame(bets: list[tuple[str | None, int, float, str]]) -> pd.DataFrame:
    """(option_id, stake, paid, outcome) per bet."""
    frame = pd.DataFrame(bets, columns=["option_id", "stake_yen", "paid_yen", "outcome"])
    frame["return_per_100"] = frame["paid_yen"] / frame["stake_yen"] * 100
    return frame


def test_played_bets_appear_beside_their_option():
    ranked = shortlist(board_rows(), top=5)
    top5 = IN_RANK_ORDER[:5]
    a, b, c, e, f = top5
    played = played_by_option(
        settled_frame(
            [
                (a, 1_000, 5_000.0, "won"),
                (a, 1_000, 0.0, "lost"),
                (a, 2_000, 5_000.0, "won"),  # a: 3 bets, 4,000 staked, paid 10,000: 250 per 100
                (b, 1_000, 500.0, "won"),
                (b, 3_000, 0.0, "lost"),  # b: 2 bets, 4,000 staked, paid 500: 12.5 per 100
                (f, 1_500, 3_300.0, "won"),  # f: 1 bet, 1,500 staked: 220 per 100
                ("o99-not-listed", 1_000, 0.0, "lost"),  # an option outside the shortlist
                (None, 9_000, 0.0, "lost"),  # the owner's own pick: off-board
            ]
        )
    )
    out = text_of(ranked, staked=21_000, played=played)
    got = blocks(out, top5)
    assert whole(got[a], 3) and whole(got[a], 4_000) and figure(got[a], 250.0), got[a]
    assert whole(got[b], 2) and whole(got[b], 4_000) and figure(got[b], 12.5), got[b]
    assert whole(got[f], 1) and whole(got[f], 1_500) and figure(got[f], 220.0), got[f]
    for k in (c, e):  # nothing played on them: none, and no one else's figures
        assert re.search(r"\bnone\b|\bno bets\b|\bnot played\b", got[k], re.I), got[k]
        assert not figure(got[k], 250.0) and not figure(got[k], 12.5), got[k]
    assert "o99-not-listed" not in out


def test_a_shortlist_without_played_bets_says_none_for_every_option():
    ranked = shortlist(board_rows(), top=4)
    for played in (None, played_by_option(settled_frame([]))):
        out = text_of(ranked, staked=0, played=played)
        for k, block in blocks(out, IN_RANK_ORDER[:4]).items():
            assert re.search(r"\bnone\b|\bno bets\b|\bnot played\b", block, re.I), (k, block)


def test_look_ahead_options_are_labelled():
    rows = [
        row("la-top", 120.0, selection="look_ahead"),
        row("dt-second", 110.0, selection="decision_time"),
        row("la-third", 105.0, selection="look_ahead"),
        row("co-fourth", 100.0, selection="card_only"),
        row("un-fifth", 95.0, selection="unknown"),
        row("dt-sixth", 90.0, selection="decision_time"),
    ]
    out = text_of(shortlist(rows, top=6))
    got = blocks(out, ["la-top", "dt-second", "la-third", "co-fourth", "un-fifth", "dt-sixth"])
    label = re.compile(r"look-ahead", re.I)
    assert label.search(got["la-top"]) and label.search(got["la-third"]), out
    # the last option's block runs to the end of the text, a footer there is not its own
    for k in ("dt-second", "co-fourth", "un-fifth"):
        assert not label.search(got[k]), f"{k} is not look-ahead:\n{got[k]}"
    assert len(label.findall(got["la-top"])) >= 1


# ---------------------------------------------------------------- the CI as the leaderboard


def ci_block(raw: float, *, ci, ci_level, se: float = 5.0) -> str:
    """The text block of one row with the given bounds (a pair, one side possibly None)."""
    rows = [row("ci-row", raw, se, ci=ci, ci_level=ci_level), row("ci-other", raw - 40.0, 5.0)]
    # shrunk here already, so a failure is about the CI and not about the shrinkage (``shrink`` is
    # idempotent: it ignores an incoming shrunk value)
    got = blocks(text_of(shortlist(shrink(rows), top=2)), ["ci-row", "ci-other"])
    return got["ci-row"]


@pytest.mark.parametrize("level", [0.90, 0.975, 0.99])
@pytest.mark.parametrize("bound", ["lo", "hi"])
def test_a_one_sided_ci_shows_its_bound_its_sense_and_its_level(bound, level):
    lo, hi = (101.7, None) if bound == "lo" else (None, 123.4)
    block = ci_block(110.0, ci=(lo, hi), ci_level=level)
    value = lo if bound == "lo" else hi
    assert figure(block, value), f"the one bound is not shown:\n{block}"
    assert "n/a" not in block, f"a row with a bound has a CI:\n{block}"
    sense, other = (r"≥|>=", r"≤|<=") if bound == "lo" else (r"≤|<=", r"≥|>=")
    assert re.search(sense, block), f"the bound's sense (one-sided) is not shown:\n{block}"
    assert not re.search(other, block), block
    shown = f"{level * 100:g}%"  # 90%, 97.5%, 99%: as the private leaderboard prints the level
    assert re.search(rf"(?<![\d.]){re.escape(shown)}", block), f"level {shown} missing:\n{block}"
    for fixed in ("95%", "98%", "97%"):
        if fixed != shown:
            assert fixed not in block, f"the CI level is not the row's {shown}:\n{block}"


@pytest.mark.parametrize("level", [0.90, 0.975, 0.99])
def test_a_two_sided_ci_shows_both_bounds_at_the_rows_level(level):
    block = ci_block(110.0, ci=(98.7, 124.3), ci_level=level)
    assert figure(block, 98.7) and figure(block, 124.3), block
    assert "n/a" not in block
    assert f"{level * 100:g}%" in block and "95%" not in block, block


# ---------------------------------------------------------------- the command


def shortlist_cmd(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["japan", "bets", "shortlist", *args])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, _ANSI.sub("", result.output)


@pytest.fixture
def patched_rows(monkeypatch):
    """Serve ``rows`` as ``build_rows()``, unshrunk as the real one returns them: patched where it
    lives and where a caller may have bound it, since the registry it reads is the owner's."""

    def serve(rows: list[LeaderboardRow]) -> None:
        def build_rows(*args, **kwargs):
            return list(rows)

        build = importlib.import_module("horsetrack.countries.japan.research.leaderboard_build")
        monkeypatch.setattr(build, "build_rows", build_rows)
        for name in (
            "horsetrack.countries.japan.bets.shortlist",
            "horsetrack.countries.japan.bets.cli",
        ):
            monkeypatch.setattr(
                importlib.import_module(name), "build_rows", build_rows, raising=False
            )

    return serve


DIVIDENDS = pd.DataFrame(
    [
        (f"jra:{d.isoformat()}:NAK:11", "WIN", comb, div, False)
        for d in (DAY, EARLIER)
        for comb, div in (("4", 500.0), ("6", 250.0))
    ],
    columns=["race_id", "pool", "combination", "dividend", "is_refund"],
)


def put(day: date, race_no: int, combination: str, stake: int, option_id: str | None) -> None:
    bet = {
        "race_id": f"jra:{day.isoformat()}:NAK:{race_no:02d}",
        "pool": "WIN",
        "combination": combination,
        "stake_yen": stake,
        "placed_utc": f"{day.isoformat()}T01:00:00Z",
    }
    if option_id is not None:
        bet["option_id"] = option_id
    BetLog(bet_log_path(day), day=day, limits=Limits(30_000, None)).record(bet)


def settle(day: date) -> None:
    log = BetLog(bet_log_path(day), day=day, limits=Limits(30_000, None))
    settled = settle_bets(pd.DataFrame(log.read()), DIVIDENDS)
    path = settled_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(settled.to_json(orient="records", force_ascii=False, indent=1) + "\n")


def seed_two_days() -> None:
    """DAY stakes 12,000: o01 won (WIN 4 at 500), o02 won (WIN 6 at 250), 10,000 off-board.
    EARLIER stakes 2,000 on o01: one won, one lost. Both days settled."""
    assert not is_holdout_day("japan", DAY) and not is_holdout_day("japan", EARLIER)
    put(DAY, 11, "4", 1_000, "o01")
    put(DAY, 11, "6", 1_000, "o02")
    put(DAY, 10, "3", 10_000, None)
    put(EARLIER, 11, "4", 1_000, "o01")
    put(EARLIER, 11, "5", 1_000, "o01")
    settle(DAY)
    settle(EARLIER)


def test_the_command_lists_the_top_ten_with_the_days_cap_and_played_bets(patched_rows):
    seed_two_days()
    patched_rows(board_rows())  # unshrunk, with se: as build_rows() returns them
    code, out = shortlist_cmd("--date", DAY.isoformat())
    assert code == 0, out
    got = blocks(out, IN_RANK_ORDER[:10])  # default top is 10, in the leaderboard's shrunk order
    for k in [*IN_RANK_ORDER[10:], *UNMEASURED, "d1"]:
        assert not re.search(rf"(?<![\w-]){k}(?![\w-])", out), f"{k} must not be listed"
    for k, block in got.items():  # the shrunk values, not 'n/a' and not the raw return
        assert figure(block, SHRUNK[k]), f"{k}: shrunk return missing:\n{block}"
        assert "n/a" not in block, block
    # the day's own log gives the cap: 12,000 staked on DAY, not the 14,000 of both days
    assert "remaining ¥18,000" in out, out
    assert "12,000" in out
    flat_line = next((ln for ln in out.splitlines() if "flat stake" in ln), "")
    assert whole(flat_line, 1_000) and "not set" not in flat_line, out  # betting.yaml's flat stake
    # played bets come from every settled file: o01 is 3 bets, 3,000 staked, 10,000 paid (333.3)
    o01 = got["o01"]
    assert whole(o01, 3) and whole(o01, 3_000) and figure(o01, 333.3), o01
    # o02: 1 bet, 1,000 staked, 2,500 paid (250.0), on DAY only
    o02 = got["o02"]
    assert whole(o02, 1) and whole(o02, 1_000) and figure(o02, 250.0), o02
    for k in ("o03", "o05"):
        assert re.search(r"\bnone\b|\bno bets\b|\bnot played\b", got[k], re.I), got[k]


def test_the_command_ranks_by_the_shrunk_return_not_the_raw_one(patched_rows):
    patched_rows(board_rows())
    code, out = shortlist_cmd("--date", DAY.isoformat(), "--top", "6")
    assert code == 0, out
    blocks(out, IN_RANK_ORDER[:6])
    # o12 has the best raw return and is only sixth after shrinkage: the order is the shrunk one
    assert IN_RANK_ORDER[0] != "o12"
    assert out.index("o02") < out.index("o03") < out.index("o01") < out.index("o12")
    # o10 and o11 are tenth and eleventh on the raw return and out of the top ten once shrunk
    code, full = shortlist_cmd("--date", DAY.isoformat())
    assert code == 0 and "o09" in full and "o10" not in full and "o11" not in full, full


@pytest.mark.parametrize("top", [1, 3, 15])
def test_the_command_takes_top(patched_rows, top):
    patched_rows(board_rows())
    code, out = shortlist_cmd("--date", DAY.isoformat(), "--top", str(top))
    assert code == 0, out
    blocks(out, IN_RANK_ORDER[:top])
    for k in IN_RANK_ORDER[top:]:
        assert not re.search(rf"(?<![\w-]){k}(?![\w-])", out), f"{k} is past --top {top}"


def test_the_command_on_a_day_without_bets_shows_the_whole_cap_and_creates_nothing(patched_rows, D):
    patched_rows(board_rows())
    code, out = shortlist_cmd("--date", DAY.isoformat(), "--top", "2")
    assert code == 0, out
    assert "remaining ¥30,000" in out, out
    for block in blocks(out, IN_RANK_ORDER[:2]).values():
        assert re.search(r"\bnone\b|\bno bets\b|\bnot played\b", block, re.I), block
    assert not D.exists(), "no bets and nothing to write: the data dir is not even created"


def test_the_command_refuses_a_bad_date(patched_rows, D):
    patched_rows(board_rows())
    code, out = shortlist_cmd("--date", "2026-13-45")
    assert code != 0 and "Traceback" not in out
    assert not D.exists()


def snapshot(root: Path, skip: set[str] = frozenset()) -> dict[str, tuple]:
    """Every path under ``root`` (directories too) with its size, mtime and bytes."""
    found: dict[str, tuple] = {}
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip and not d.endswith(".egg-info")]
        if Path(current) == REPO / ".claude":
            dirs[:] = [d for d in dirs if d != "worktrees"]
        for name in [*dirs, *files]:
            path = Path(current) / name
            stat = path.lstat()
            body = path.read_bytes() if path.is_file() and stat.st_size < 5_000_000 else None
            found[str(path)] = (stat.st_size, stat.st_mtime_ns, body)
    return found


def test_the_command_writes_nothing(patched_rows, D):
    seed_two_days()
    patched_rows(board_rows())
    before_data = snapshot(D)
    before_repo = snapshot(REPO, SKIP_DIRS)
    assert before_data, "premise: the seeded bet logs and settled files exist"
    code, out = shortlist_cmd("--date", DAY.isoformat())
    assert code == 0, out
    assert snapshot(D) == before_data, "the data dir changed"
    assert snapshot(REPO, SKIP_DIRS) == before_repo, "the repository changed"


# ---------------------------------------------------------------- the real build_rows path


GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
BUILD_HASH = "ab" * 32
POOL_SCAN = "japan-pool-scan-v1"
FAMILY_L = "japan-family-l-exotic-pool-choice"
CELLS = [
    {
        "study": "N1",
        "family_id": POOL_SCAN,
        "cell_id": cell_id,
        "pool": "WIN",
        "payout_rate": 0.8,
        "n_bets": 1200,
        "n_race_days": 300,
        "n_meetings": 90,
        "mean_return": mean,
        "se_mean_return": se,
        "shrunk_edge": None,
        "mean_odds": 7.1,
        "sources": ["netkeiba"],
    }
    for cell_id, mean, se in (
        ("WIN|class:maiden|band:5-10", 0.84, 0.03),
        ("WIN|class:open|band:10-20", 0.97, 0.12),
        ("WIN|class:novice|band:2-5", 0.9, 0.02),
    )
]
CELL_IDS = [f"{POOL_SCAN}/{c['cell_id']}" for c in CELLS]


def dev(metric: str, value: float, lo=None, hi=None, n: int = 0) -> FoldMetric:
    return FoldMetric(fold="development", metric=metric, value=value, lo=lo, hi=hi, n=n)


def seed_registry_and_cells(data: Path, tmp_path: Path) -> Registry:
    """A registry file at the default path holding one Family L run, and a candidate-cell file at
    the default directory; both under the data dir, both synthetic."""
    path = data / "registry" / "japan.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = Registry(path, holdout_log=tmp_path / "holdout_log.jsonl")
    rule, comparator = 0.93, 0.88
    registry.append(
        RunRecord(
            hypothesis_id="L1",
            family_id=FAMILY_L,
            variant="L1:TRIO:threshold_20:final_look_ahead",
            ticket=81,
            git_sha=GIT_SHA,
            data_builds={"results": BUILD_HASH},
            sources=("netkeiba",),
            fold_scheme="development-only",
            components={"rule": "synthetic"},
            params={"pool": "TRIO", "stake_yen": 100, "threshold": 20},
            metrics=(
                dev("paired_return_difference", rule - comparator, 0.04, 0.06, 900),
                dev("rule_mean_return", rule, None, None, 900),
                dev("comparator_mean_return", comparator, None, None, 900),
                dev("meetings", 40.0, None, None, 900),
            ),
            status="scored",
        )
    )
    cells = data / "reports" / "candidate-cells"
    cells.mkdir(parents=True, exist_ok=True)
    (cells / "n1.json").write_text(json.dumps(CELLS, indent=1) + "\n", encoding="utf-8")
    return registry


def test_the_real_build_rows_path_on_a_fresh_data_root_creates_nothing(D):
    """No patch: the command reads the registry that is not there, and it makes no registry
    folder, no reports folder, not even the data root."""
    assert not D.exists()
    before_repo = snapshot(REPO, SKIP_DIRS)
    code, out = shortlist_cmd("--date", DAY.isoformat())
    assert code == 0, out
    assert "remaining ¥30,000" in out, out
    assert not D.exists(), f"the command created {sorted(snapshot(D)) if D.exists() else D}"
    assert snapshot(REPO, SKIP_DIRS) == before_repo, "the repository changed"


def test_the_real_build_rows_path_on_an_empty_data_dir_creates_nothing(D):
    D.mkdir()
    before = snapshot(D)
    assert before == {}
    code, out = shortlist_cmd("--date", DAY.isoformat())
    assert code == 0, out
    assert snapshot(D) == {}, f"the command created {sorted(snapshot(D))} under the data dir"


def test_the_real_build_rows_path_on_a_seeded_data_dir_lists_the_shrunk_top_and_writes_nothing(
    D, tmp_path
):
    seed_two_days()
    registry = seed_registry_and_cells(D, tmp_path)
    # the leaderboard's own rows of this data, as the private leaderboard ranks them
    built = leaderboard_build.build_rows(
        registry=registry, candidate_dir=D / "reports" / "candidate-cells"
    )
    want = [r for r in rank(shrink(built)) if r.rank is not None]
    assert set(CELL_IDS) <= {r.row.option_id for r in want}, "premise: the cells are measured rows"
    assert any(r.row.option_id not in CELL_IDS for r in want), "premise: the registry gives a row"
    before_data = snapshot(D)
    before_repo = snapshot(REPO, SKIP_DIRS)
    code, out = shortlist_cmd("--date", DAY.isoformat(), "--top", "50")
    assert code == 0, out
    got = blocks(out, [r.row.option_id for r in want])
    for r in want:
        block = got[r.row.option_id]
        if r.row.shrunk_per_100 is None:
            # the library gives no shrunk value to a row without a se (the registry's Family L
            # rows carry no CI on their rule return): the leaderboard ranks it on its raw return
            assert "n/a" in block, block
        else:
            assert figure(block, r.row.shrunk_per_100), block
    shrunk = [r for r in want if r.row.shrunk_per_100 is not None]
    assert {r.row.option_id for r in shrunk} >= set(CELL_IDS), "premise: the cells are shrunk"
    assert snapshot(D) == before_data, "the data dir changed"
    assert snapshot(REPO, SKIP_DIRS) == before_repo, "the repository changed"
