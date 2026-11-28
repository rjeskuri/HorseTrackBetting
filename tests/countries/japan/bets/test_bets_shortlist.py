"""Ticket #455: `horsetrack japan bets shortlist`, the top of the private leaderboard beside the
day's cap.

Contract pinned here, module ``horsetrack.countries.japan.bets.shortlist`` and the command on
``bets_app`` (always run through the root CLI, so the registration is tested too):

- ``shortlist(rows, *, top) -> list[RankedRow]``: the ``top`` best measured, ranked rows of
  ``leaderboard.rank(rows)``, in rank order (ranks 1..k). Unmeasured and descriptive (unranked)
  rows are never on it, however large ``top`` is.
- ``shortlist_text(ranked, *, day, check, limits, played) -> str``: one entry per option, in rank
  order, with its option id, pool, segment, rule, selection, shrunk return per 100 yen, CI and
  variants searched, and its played bets (n, staked, return per 100 yen from ``played``, a
  ``played_by_option`` frame, or None when nothing was played); a look-ahead option carries the
  label 'look-ahead'. The head shows the day's staked total, 'remaining ¥N' (the day cap minus the
  staked total) and 'flat stake: ¥N' or 'flat stake: not set'.
- ``bets shortlist --date D [--top 10]``: rows from ``leaderboard_build.build_rows()``, the day's
  staked total from its bet log, the played bets of every ``<day>.settled.json`` under the bets
  dir. It prints and writes nothing.

Everything is synthetic. ``build_rows`` is patched (it reads the private registry); the bet log and
the settled files are written through the project's own APIs under a temp data dir.
"""

from __future__ import annotations

import importlib
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from horsetrack.countries.japan.bets.shortlist import shortlist, shortlist_text
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.leaderboard import LeaderboardRow, RankedRow, rank
from horsetrack.countries.japan.bets.cli import settled_path
from horsetrack.countries.japan.bets.log import BetLog, CapCheck, Limits, bet_log_path
from horsetrack.countries.japan.bets.settle import played_by_option, settle_bets

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
    shrunk: float | None,
    raw: float | None = None,
    *,
    selection: str = "decision_time",
    variants: int = 1,
    pool: str = "WIN",
    segment: str = "all",
    rule: str = "flat",
    descriptive: bool = False,
) -> LeaderboardRow:
    measured = shrunk is not None or raw is not None
    raw = raw if raw is not None else shrunk
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
        status="measured" if measured else "unmeasured",
        return_per_100=raw,
        ci_lo=None if raw is None else round(raw - 11.1, 1),
        ci_hi=None if raw is None else round(raw + 12.3, 1),
        ci_level=None if raw is None else 0.95,
        shrunk_per_100=None if descriptive else shrunk,
        descriptive=descriptive,
    )


# twelve measured options with distinct shrunk values, deliberately not in rank order; the raw
# return is not monotone in the shrunk value (a high raw row with a low shrunk value)
SHRUNK = {
    "o07": 91.4, "o02": 104.2, "o11": 88.6, "o05": 97.8, "o01": 112.5, "o09": 83.1,
    "o12": 79.3, "o03": 101.7, "o08": 93.9, "o04": 99.4, "o10": 86.2, "o06": 95.3,
}  # fmt: skip
RAW = {**{k: v + 2.0 for k, v in SHRUNK.items()}, "o12": 131.0, "o09": 60.0}
IN_RANK_ORDER = sorted(SHRUNK, key=lambda k: -SHRUNK[k])
UNMEASURED = ["u1", "u2", "u3"]


def twelve_and_three() -> list[LeaderboardRow]:
    measured = [
        row(k, SHRUNK[k], RAW[k], variants=100 + 7 * i, rule=f"rule-{k}", segment=f"seg-{k}")
        for i, k in enumerate(SHRUNK)
    ]
    unmeasured = [row(k, None) for k in UNMEASURED]
    descriptive = [row("d1", None, 500.0, descriptive=True)]
    return [*unmeasured[:1], *measured[:6], *descriptive, *unmeasured[1:], *measured[6:]]


# ---------------------------------------------------------------- shortlist


def test_the_shortlist_takes_the_top_ranked_measured_options():
    rows = twelve_and_three()
    got = shortlist(rows, top=10)
    assert [r.row.option_id for r in got] == IN_RANK_ORDER[:10]
    assert [r.rank for r in got] == list(range(1, 11))
    assert all(isinstance(r, RankedRow) for r in got)
    assert not {r.row.option_id for r in got} & set(UNMEASURED)
    # the same rows as the leaderboard's own ranking, not a re-sort of the raw return
    assert got == rank(rows)[:10]


@pytest.mark.parametrize("top", [1, 3, 7, 12, 50])
def test_the_shortlist_is_as_long_as_top_up_to_the_measured_options(top):
    got = shortlist(twelve_and_three(), top=top)
    want = IN_RANK_ORDER[: min(top, 12)]
    assert [r.row.option_id for r in got] == want
    assert [r.rank for r in got] == list(range(1, len(want) + 1))


def test_the_shortlist_never_holds_unmeasured_or_unranked_rows():
    got = shortlist(twelve_and_three(), top=50)
    ids = {r.row.option_id for r in got}
    assert ids == set(SHRUNK), "twelve measured, ranked rows and nothing else"
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
    ranked = shortlist(twelve_and_three(), top=10)
    text = text_of(ranked)
    got = blocks(text, IN_RANK_ORDER[:10])
    for k in IN_RANK_ORDER[:10]:
        block = got[k]
        raw = RAW[k]
        assert figure(block, SHRUNK[k]), f"{k}: shrunk return missing:\n{block}"
        assert figure(block, round(raw - 11.1, 1)) and figure(block, round(raw + 12.3, 1)), (
            f"{k}: CI missing:\n{block}"
        )
        n_variants = 100 + 7 * list(SHRUNK).index(k)
        assert whole(block, n_variants), f"{k}: variants searched missing:\n{block}"
        assert "WIN" in block and f"seg-{k}" in block and f"rule-{k}" in block, block
        assert re.search(r"decision[-_ ]time", block, re.I), f"{k}: selection missing:\n{block}"
    for k in [*IN_RANK_ORDER[10:], *UNMEASURED, "d1"]:
        assert not re.search(rf"(?<![\w-]){k}(?![\w-])", text), f"{k} must not be listed"


@pytest.mark.parametrize(
    ("staked", "remaining"),
    [(12_000, 18_000), (0, 30_000), (29_500, 500), (7_300, 22_700)],
)
def test_the_shortlist_shows_the_remaining_cap_and_flat_stake(staked, remaining):
    ranked = shortlist(twelve_and_three(), top=3)
    out = text_of(ranked, staked=staked, flat=None)
    assert f"remaining ¥{remaining:,}" in out, out
    assert "flat stake: not set" in out, out
    if staked:
        assert f"{staked:,}" in out, f"the day's staked total is not shown:\n{out}"


@pytest.mark.parametrize("flat", [1_000, 500])
def test_the_shortlist_shows_the_flat_stake_when_set(flat):
    out = text_of(shortlist(twelve_and_three(), top=2), staked=12_000, flat=flat)
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
    ranked = shortlist(twelve_and_three(), top=5)
    top5 = IN_RANK_ORDER[:5]  # o01, o02, o03, o05(97.8)... in rank order
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
    ranked = shortlist(twelve_and_three(), top=4)
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


# ---------------------------------------------------------------- the command


def shortlist_cmd(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["japan", "bets", "shortlist", *args])
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, _ANSI.sub("", result.output)


@pytest.fixture
def patched_rows(monkeypatch):
    """Serve ``rows`` as ``build_rows()``: patched where it lives and where a caller may have
    bound it, since the registry it reads is the owner's private one."""

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
    patched_rows(twelve_and_three())
    code, out = shortlist_cmd("--date", DAY.isoformat())
    assert code == 0, out
    got = blocks(out, IN_RANK_ORDER[:10])  # default top is 10
    for k in [*IN_RANK_ORDER[10:], *UNMEASURED, "d1"]:
        assert not re.search(rf"(?<![\w-]){k}(?![\w-])", out), f"{k} must not be listed"
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


@pytest.mark.parametrize("top", [1, 3, 12])
def test_the_command_takes_top(patched_rows, top):
    patched_rows(twelve_and_three())
    code, out = shortlist_cmd("--date", DAY.isoformat(), "--top", str(top))
    assert code == 0, out
    blocks(out, IN_RANK_ORDER[:top])
    for k in IN_RANK_ORDER[top:]:
        assert not re.search(rf"(?<![\w-]){k}(?![\w-])", out), f"{k} is past --top {top}"


def test_the_command_on_a_day_without_bets_shows_the_whole_cap_and_creates_nothing(patched_rows, D):
    patched_rows(twelve_and_three())
    code, out = shortlist_cmd("--date", DAY.isoformat(), "--top", "2")
    assert code == 0, out
    assert "remaining ¥30,000" in out, out
    for block in blocks(out, IN_RANK_ORDER[:2]).values():
        assert re.search(r"\bnone\b|\bno bets\b|\bnot played\b", block, re.I), block
    assert not D.exists(), "no bets and nothing to write: the data dir is not even created"


def test_the_command_refuses_a_bad_date(patched_rows, D):
    patched_rows(twelve_and_three())
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
    patched_rows(twelve_and_three())
    before_data = snapshot(D)
    before_repo = snapshot(REPO, SKIP_DIRS)
    assert before_data, "premise: the seeded bet logs and settled files exist"
    code, out = shortlist_cmd("--date", DAY.isoformat())
    assert code == 0, out
    assert snapshot(D) == before_data, "the data dir changed"
    assert snapshot(REPO, SKIP_DIRS) == before_repo, "the repository changed"
