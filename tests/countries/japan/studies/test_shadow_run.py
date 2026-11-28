"""Ticket #617: the shadow-calibration runner, rerun as ``japan-shadow-calibration-v2``.

Contract pinned here (module ``horsetrack.countries.japan.studies.shadow_run`` unless named):

- ``FAMILY_ID == "japan-shadow-calibration-v2"``; ``build_spec()`` is its plain-JSON spec with
  - ``days``: ISO dates, sorted and unique, every one on or after 2026-11-28 and a development day
    by the committed rule (``is_holdout_day`` false), 2026-11-28 among them;
  - ``nowcast``: ``{"fit_id": "61413e8a77a6", "fitted_through": <ISO date>, ...}``, the fit the
    scored claims used and the last day of its training data, before the first study day (out of
    sample of every study day);
  - ``drift.pairs``: E20 as explicit pairs ``{"tau_min": t, "lag_min": lag}``, each t and t + lag
    on the calibration grid, no pair twice, v1's comparisons (2, 3), (5, 11), (2, 14), (16, 14)
    among them as separate pairs;
  - ``sources`` naming ``jravan`` beside ``jra`` (the nowcast fit is on JRA-VAN's odds series).
- ``run_study`` refuses (RuntimeError) before reading a shadow record, a dividend, a board or a
  bet unless ``FAMILY_ID`` is frozen with exactly ``build_spec()``. Its E20 variants are one per
  (pair, odds level): params ``tau_min``, ``lag_min`` and ``partial_log_odds`` (False: the plain
  regression; True: with ln board_odds at t as a covariate). The pick rule's return per 100 has a
  meeting interval with ``n_clusters[meeting]``; on one meeting it is not estimable (no zero-width
  interval). ``pick_growth`` keeps a row for a day that lost its whole money. Records carry the
  spec's sources.
- ``_picks(settled, tau, floor)``: WIN at ``tau``, value_q05 strictly above ``floor``, empty flags,
  never a holdout record; ``rank`` 1 = the highest value_q05 within a decision time.
- ``_bets(limits, out, played)()``: ``out["day_stakes"]`` one row per bet-log day with the day's
  staked total and ``over_cap`` (above the cap only); holdout-day bets settle 'holdout'.
- ``models.shadow.selection_drift(..., tau_minutes=None, partial_log_odds=False)``: with
  ``tau_minutes`` only records at that t are regressed (one pair, never pooled over t);
  ``partial_log_odds`` adds ln(board_odds at t) as a covariate and reports the score's coefficient.

The artifact id leaving out invocation metadata is pinned in ``japan/test_shadow.py``
(an identical rerun keeps its report; another leaderboard is another artifact). Every record, bet
and dividend here is synthetic; HORSETRACK_DATA_DIR points at tmp_path in every test.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import default_registry
from horsetrack.countries.japan.bets.log import BetLog, bet_log_path, load_limits
from horsetrack.countries.japan.models.shadow import flat_stakes, selection_drift, settle_shadow
from horsetrack.countries.japan.studies import shadow_run

UTC = timezone.utc
D28 = date(2026, 11, 28)
V1_FAMILY = "japan-shadow-calibration-v1"
V1_DAYS = {"2026-11-21", "2026-11-22"}
NOWCAST_FIT = "61413e8a77a6"  # the fit the 2026-11-28 records name
MODELS = {"win": "aac501f927b0", "ordering": "2b822d643cf7", "nowcast": NOWCAST_FIT}
DIVIDEND_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
GRID = (2, 5, 16, 30)  # the taus the drift fixtures log
BOOT = {"n_boot": 1_000, "level": 0.95, "seed": 617}


@pytest.fixture(autouse=True)
def D(tmp_path, monkeypatch) -> Path:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    return data


def first_held_day() -> date:
    day = D28
    while not is_holdout_day("japan", day):
        day += timedelta(days=1)
    return day


def race(venue: str, no: int, day: date = D28) -> str:
    return f"jra:{day.isoformat()}:{venue}:{no:02d}"


def post_time(race_id: str) -> datetime:
    day = date.fromisoformat(race_id.split(":")[1])
    venue_offset = {"TOK": 0, "KYO": 5, "NAK": 10, "HAN": 15}[race_id.split(":")[2]]
    no = int(race_id.rsplit(":", 1)[1])
    return datetime(day.year, day.month, day.day, 1, 0, tzinfo=UTC) + timedelta(
        minutes=30 * no + venue_offset
    )


def record(
    race_id: str,
    combination: str,
    *,
    tau: int = 2,
    pool: str = "WIN",
    value_mean: float | None = 0.9,
    value_q05: float | None = None,
    board_odds: float | None = 3.4,
    flags: list | None = None,
    day_status: str = "development",
) -> dict:
    """A v0 shadow-day record as ``ShadowLog.read()`` returns it (docs/specs/shadow-log.md)."""
    decided = post_time(race_id) - timedelta(minutes=tau)
    return {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "tau_min": tau,
        "decision_utc": decided.isoformat(),
        "board_capture_utc": (decided - timedelta(seconds=16)).isoformat(),
        "written_utc": (decided + timedelta(seconds=16)).isoformat(),
        "mode": "live",
        "status": "ok",
        "models": dict(MODELS),
        "git_sha": "0123abc",
        "board_odds": board_odds,
        "board_odds_hi": None,
        "value_mean": value_mean,
        "value_q05": value_q05,
        "value_q95": None,
        "value_at_low": None,
        "value_at_high": None,
        "flags": [] if flags is None else flags,
        "n_draws": 400,
        "seed": 1,
        "day_status": day_status,
    }


def paying(race_id: str, combination: str, dividend: float, pool: str = "WIN") -> dict:
    return {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "dividend": float(dividend),
        "is_refund": False,
    }


def dividends(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DIVIDEND_COLUMNS)


def null(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def day_text(value) -> str:
    return str(value)[:10]


def run_v2(monkeypatch, tmp_path: Path, records: list[dict], paid: list[dict], finals=None):
    """``run_study`` on the frozen v2 spec over synthetic inputs; returns its payload."""
    by_day: dict[date, list[dict]] = {}
    for row in paid:
        by_day.setdefault(date.fromisoformat(row["race_id"].split(":")[1]), []).append(row)
    monkeypatch.setattr(shadow_run, "read_shadow", lambda days: pd.DataFrame(records))
    monkeypatch.setattr(shadow_run, "day_dividends", lambda day: dividends(by_day.get(day, [])))
    monkeypatch.setattr(shadow_run, "final_win_odds", lambda days: dict(finals or {}))
    monkeypatch.setattr(shadow_run, "_git_sha", lambda: "0123abc" + "0" * 33)
    build = paths.processed_dir("jra") / "_build.json"
    build.parent.mkdir(parents=True, exist_ok=True)
    build.write_text(json.dumps({"snapshot_id": "synthetic-617"}), encoding="utf-8")
    shadow_run.freeze()
    out = (tmp_path / "out").resolve()
    out.mkdir(exist_ok=True)
    return shadow_run.run_study(out_dir=out)


def variant(payload: dict, name: str) -> dict:
    found = [v for v in payload["variants"] if v["variant"] == name]
    assert len(found) == 1, f"{name}: {len(found)} variant(s) of that name"
    return found[0]


def metrics_of(v: dict) -> dict[str, dict]:
    return {m["metric"]: m for m in v["metrics"]}


# ---------------------------------------------------------------- the v2 spec


def test_the_v2_spec_names_only_development_days_an_out_of_sample_nowcast_and_explicit_pairs():
    assert shadow_run.FAMILY_ID == "japan-shadow-calibration-v2"
    spec = shadow_run.build_spec()
    assert json.loads(json.dumps(spec)) == spec, "the spec is plain JSON"
    assert spec["family_id"] == shadow_run.FAMILY_ID

    days = spec["days"]
    assert days and days == sorted(set(days)), f"days are sorted and unique: {days}"
    parsed = [date.fromisoformat(d) for d in days]
    assert all(d >= D28 for d in parsed), f"a day before 2026-11-28: {days}"
    held = [d.isoformat() for d in parsed if is_holdout_day("japan", d)]
    assert not held, f"the spec names holdout day(s) {held}"
    assert D28.isoformat() in days
    assert not V1_DAYS & set(days), "v1's days are not v2's"

    nowcast = spec["nowcast"]
    assert nowcast["fit_id"] == NOWCAST_FIT
    through = date.fromisoformat(nowcast["fitted_through"])
    assert through < min(parsed), (
        f"the nowcast fit's data run to {through}: not out of sample of the first day {days[0]}"
    )

    taus = set(spec["calibration"]["taus"])
    assert spec["calibration"]["primary_tau"] == 2
    pairs = [(p["tau_min"], p["lag_min"]) for p in spec["drift"]["pairs"]]
    assert len(pairs) == len(set(pairs)), f"a pair twice: {pairs}"
    for tau, lag in pairs:
        assert lag > 0 and tau in taus and tau + lag in taus, f"({tau}, {lag}) is off the grid"
    assert {(2, 3), (5, 11), (2, 14), (16, 14)} <= set(pairs), (
        f"v1's comparisons, lag 14 split into its two pairs, are not all named: {pairs}"
    )

    assert {"jra", "jravan"} <= set(spec["sources"]), "the claims rest on a JRA-VAN nowcast fit"


# ---------------------------------------------------------------- the frozen-spec guard


def test_the_run_reads_nothing_unless_the_v2_family_is_frozen_with_exactly_this_spec(
    tmp_path, monkeypatch
):
    assert shadow_run.FAMILY_ID == "japan-shadow-calibration-v2"
    reads: list[str] = []
    monkeypatch.setattr(
        shadow_run, "read_shadow", lambda days: reads.append("shadow") or pd.DataFrame()
    )
    monkeypatch.setattr(
        shadow_run, "day_dividends", lambda day: reads.append("dividends") or dividends([])
    )
    monkeypatch.setattr(shadow_run, "final_win_odds", lambda days: reads.append("boards") or {})
    monkeypatch.setattr(
        shadow_run, "read_bets", lambda days: reads.append("bets") or pd.DataFrame()
    )
    monkeypatch.setattr(shadow_run, "_git_sha", lambda: "0123abc" + "0" * 33)
    spec = shadow_run.build_spec()

    def not_frozen(registry):
        pass

    def frozen_with_other_days(registry):
        registry.freeze_family(shadow_run.FAMILY_ID, spec | {"days": sorted(V1_DAYS)})

    def only_v1_frozen(registry):  # v1's freeze, even of this very spec, does not unlock v2
        registry.freeze_family(V1_FAMILY, spec)

    def spec_changed_after_freeze(registry):
        shadow_run.freeze()
        changed = json.loads(json.dumps(spec))
        changed["pick_rule"]["value_q05_above"] = 1.05
        monkeypatch.setattr(shadow_run, "build_spec", lambda: changed)

    cases = (not_frozen, frozen_with_other_days, only_v1_frozen, spec_changed_after_freeze)
    original = shadow_run.build_spec
    for case in cases:
        monkeypatch.setattr(shadow_run, "build_spec", original)
        data = (tmp_path / case.__name__).resolve()
        monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
        registry = default_registry("japan")
        case(registry)
        out = (tmp_path / f"out-{case.__name__}").resolve()
        out.mkdir()
        reads.clear()
        with pytest.raises(RuntimeError):
            shadow_run.run_study(out_dir=out)
        assert reads == [], f"{case.__name__}: read {reads} before refusing"
        assert not default_registry("japan").records(shadow_run.FAMILY_ID), case.__name__
        assert not list(out.iterdir()), f"{case.__name__}: wrote a report"

    # The control: frozen with exactly build_spec(), the run reads its inputs.
    monkeypatch.setattr(shadow_run, "build_spec", original)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str((tmp_path / "frozen").resolve()))
    build = paths.processed_dir("jra") / "_build.json"
    build.parent.mkdir(parents=True)
    build.write_text(json.dumps({"snapshot_id": "synthetic-617"}), encoding="utf-8")
    shadow_run.freeze()
    out = (tmp_path / "out-frozen").resolve()
    out.mkdir()
    reads.clear()
    shadow_run.run_study(out_dir=out)
    assert "shadow" in reads and "dividends" in reads


# ---------------------------------------------------------------- the pick rule


def test_picks_are_unflagged_development_win_records_above_the_floor_ranked_by_q05():
    a, a5, b = race("TOK", 1), race("TOK", 1), race("KYO", 1)
    held = race("TOK", 1, first_held_day())
    rows = [
        ("A1", record(a, "1", value_q05=1.20)),
        ("A2", record(a, "2", value_q05=1.35)),
        ("A3", record(a, "3", value_q05=1.0001)),
        ("A4", record(a, "4", value_q05=1.00)),  # at the floor, not above it
        ("A5", record(a, "5", value_q05=0.90)),
        ("A6", record(a, "6", value_q05=None)),
        ("A7", record(a, "7", value_q05=1.50, flags=["tail"])),
        ("A2@5", record(a5, "2", tau=5, value_q05=1.60)),
        ("A1 place", record(a, "1", pool="PLACE", value_mean=None, value_q05=1.40)),
        ("B3", record(b, "3", value_q05=1.10)),
        ("B8", record(b, "8", value_q05=1.05)),
        ("held", record(held, "1", value_q05=1.70, day_status="holdout")),
    ]
    settled = settle_shadow(pd.DataFrame([r for _, r in rows]), dividends([]))
    names = {(r["race_id"], r["pool"], r["combination"], r["tau_min"]): name for name, r in rows}
    # (tau, floor) -> {pick: rank}, worked out from the table by hand
    expected = {
        (2, 1.0): {"A1": 2, "A2": 1, "A3": 3, "B3": 1, "B8": 2},
        (2, 1.1): {"A1": 2, "A2": 1},
        (2, 1.25): {"A2": 1},
        (5, 1.0): {"A2@5": 1},
    }
    for (tau, floor), want in expected.items():
        picks = shadow_run._picks(settled, tau, floor)
        got = {
            names[(r["race_id"], r["pool"], r["combination"], r["tau_min"])]: int(r["rank"])
            for r in picks.to_dict("records")
        }
        assert got == want, f"tau {tau}, floor {floor}"


def test_picks_take_the_race_day_cap_by_decision_time_then_highest_q05():
    limits = load_limits()
    per_race = 16  # 32 picks of the flat stake over two decision times: two past the cap
    assert 2 * per_race * limits.flat_stake_yen == limits.day_cap_yen + 2 * limits.flat_stake_yen
    early, late = race("TOK", 1), race("TOK", 2)  # TOK 1 decides 30 minutes before TOK 2
    q05 = {h: 1.0 + h / 100 for h in range(1, per_race + 1)}
    records = [record(r, str(h), value_q05=q) for r in (late, early) for h, q in q05.items()]
    picks = shadow_run._picks(settle_shadow(pd.DataFrame(records), dividends([])), 2, 1.0)
    staked = flat_stakes(picks, limits=limits)
    unstaked = {
        (r["race_id"], r["combination"]) for r in staked.to_dict("records") if r["stake_yen"] == 0
    }
    assert unstaked == {(late, "1"), (late, "2")}, "the later race's two lowest value_q05 miss"


def single_and_two_meeting_picks() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Pick-rule worlds: TOK alone (picks return 500, 0, 300, 0) and TOK with KYO (800, 0, 0)."""
    tok, tok_paid = [], []
    for no, (winner, dividend) in enumerate((("1", 500), ("2", 300), ("1", 300), ("3", 900)), 1):
        r = race("TOK", no)
        tok += [
            record(r, "1", value_mean=1.30, value_q05=1.10),
            record(r, "2", value_mean=0.90, value_q05=0.80),
            record(r, "3", value_mean=0.70, value_q05=0.50),
        ]
        tok_paid.append(paying(r, winner, dividend))
    kyo, kyo_paid = [], []
    for no, (winner, dividend) in enumerate((("1", 800), ("2", 250), ("3", 400)), 1):
        r = race("KYO", no)
        kyo += [
            record(r, "1", value_mean=1.25, value_q05=1.05),
            record(r, "2", value_mean=0.95, value_q05=0.85),
            record(r, "3", value_mean=0.80, value_q05=0.60),
        ]
        kyo_paid.append(paying(r, winner, dividend))
    return tok, tok_paid, kyo, kyo_paid


def test_a_single_meeting_pick_set_is_not_estimable(tmp_path, monkeypatch):
    tok, tok_paid, _, _ = single_and_two_meeting_picks()
    got = run_v2(monkeypatch, tmp_path, tok, tok_paid)
    assert got["family_id"] == "japan-shadow-calibration-v2"
    pick = variant(got, shadow_run.PICK_OPTION)
    m = metrics_of(pick)
    assert m["n_clusters[meeting]"]["value"] == 1, "the one meeting is counted, not absent"
    ret = m.get("return_per_100")
    if ret is not None:
        assert ret["value"] == pytest.approx((500 + 0 + 300 + 0) / 4)
        assert ret["lo"] is None and ret["hi"] is None, (
            f"a one-meeting bootstrap gave the interval [{ret['lo']}, {ret['hi']}]"
        )
    assert "not estimable" in pick["params"].get("note", ""), pick["params"]


def test_a_two_meeting_pick_set_has_its_meeting_interval(tmp_path, monkeypatch):
    tok, tok_paid, kyo, kyo_paid = single_and_two_meeting_picks()
    got = run_v2(monkeypatch, tmp_path, tok + kyo, tok_paid + kyo_paid)
    m = metrics_of(variant(got, shadow_run.PICK_OPTION))
    assert m["n_clusters[meeting]"]["value"] == 2
    ret = m["return_per_100"]
    assert ret["value"] == pytest.approx((500 + 0 + 300 + 0 + 800 + 0 + 0) / 7)
    assert ret["n"] == 7
    assert ret["lo"] is not None and ret["hi"] is not None and ret["lo"] < ret["hi"]


def test_a_day_that_loses_its_whole_money_keeps_its_growth_row(tmp_path, monkeypatch):
    limits = load_limits()
    records, paid = [], []
    for venue in ("TOK", "KYO"):
        for no in range(1, 13):
            r = race(venue, no)
            records += [
                record(r, "1", value_mean=1.40, value_q05=1.20),
                record(r, "5", value_mean=0.60, value_q05=None),
            ]
            if venue == "TOK" and no <= 7:
                records.append(record(r, "2", value_mean=1.30, value_q05=1.10))
            paid.append(paying(r, "5", 1_000))  # every pick loses
    assert sum(1 for r in records if (r["value_q05"] or 0) > 1) == 31  # one past the cap
    got = run_v2(monkeypatch, tmp_path, records, paid)
    rows = {day_text(g["day"]): g for g in got["pick_growth"]}
    assert D28.isoformat() in rows, f"no growth row for the study day: {sorted(rows)}"
    row = rows[D28.isoformat()]
    assert row["status"] == "settled"
    assert row["n_settled"] == limits.day_cap_yen // limits.flat_stake_yen
    assert row["staked_yen"] == limits.day_cap_yen
    assert row["ending_yen"] == 0
    assert row["log_growth"] in (None, "-inf") or row["log_growth"] == -math.inf
    report = json.loads(Path(got["report_path"]).read_text(encoding="utf-8"))
    assert D28.isoformat() in {day_text(g["day"]) for g in report["pick_growth"]}


# ---------------------------------------------------------------- the primary calibration


def calibration_world(venues: tuple[str, ...], seed: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    records, paid = [], []
    for venue in venues:
        for no in range(1, 7):
            r = race(venue, no)
            for h in range(1, 5):
                records.append(record(r, str(h), value_mean=float(rng.uniform(0.6, 1.4))))
            winner = int(rng.integers(1, 5))
            paid.append(paying(r, str(winner), float(rng.integers(20, 150) * 10)))
    return records, paid


def expected_slope(records: list[dict], paid: list[dict]) -> tuple[float, int]:
    won = {(p["race_id"], p["combination"]): p["dividend"] for p in paid}
    x = [r["value_mean"] - 1.0 for r in records if r["tau_min"] == 2]
    y = [
        won.get((r["race_id"], r["combination"]), 0.0) / 100.0 - 1.0
        for r in records
        if r["tau_min"] == 2
    ]
    return float(np.polyfit(x, y, 1)[0]), len(x)


@pytest.mark.parametrize(
    ("venues", "estimable"), [(("TOK", "KYO"), True), (("TOK",), False)], ids=["two", "one"]
)
def test_the_primary_t2_slope_has_its_meeting_interval_or_its_meeting_count(
    tmp_path, monkeypatch, venues, estimable
):
    records, paid = calibration_world(venues, seed=len(venues))
    slope, n = expected_slope(records, paid)
    got = run_v2(monkeypatch, tmp_path, records, paid)
    assert got["family_id"] == "japan-shadow-calibration-v2"
    primary = got["primary"]
    assert primary["slope"] == pytest.approx(slope, abs=1e-9)
    assert primary["n"] == n
    assert primary["n_meetings"] == len(venues)
    if estimable:
        assert primary["lo"] is not None and primary["hi"] is not None
        assert primary["lo"] < primary["hi"]
    else:
        assert primary["lo"] is None and primary["hi"] is None
    records_of = default_registry("japan").records(shadow_run.FAMILY_ID)
    assert records_of, "the run appended no record of the v2 family"
    assert all({"jra", "jravan"} <= set(r.sources) for r in records_of), (
        "a v2 record does not name the JRA-VAN source its nowcast fit rests on"
    )


# ---------------------------------------------------------------- E20 per (tau, lag) pair


def drift_world(venues: tuple[str, ...], seed: int, races: int = 8, runners: int = 8):
    """Records at T-2, T-5, T-16 and T-30 and the final odds. Each runner's board is the same at
    every t (ln O = L), so the fall to the final, y = L - ln final, is the same from every t. The
    T-16 score tracks the odds level and drives y; the T-5 and T-30 scores are noise. Every third
    race's last runner has no T-16 record."""
    rng = np.random.default_rng(seed)
    records, finals = [], {}
    for venue in venues:
        for no in range(1, races + 1):
            r = race(venue, no)
            for h in range(1, runners + 1):
                level = float(rng.uniform(0.5, 3.5))
                score = {tau: float(rng.normal(0.0, 0.3)) for tau in GRID}
                score[16] = -0.25 * (level - 2.0) + float(rng.normal(0.0, 0.1))
                fall = 0.5 * score[16] + 0.3 * (level - 2.0) + float(rng.normal(0.0, 0.05))
                finals[(r, str(h))] = math.exp(level - fall)
                for tau in GRID:
                    if tau == 16 and h == runners and no % 3 == 0:
                        continue
                    records.append(
                        record(
                            r,
                            str(h),
                            tau=tau,
                            value_mean=1.0 + score[tau],
                            board_odds=math.exp(level),
                        )
                    )
    return records, finals


def pair_rows(records: list[dict], finals: dict, tau: int, lag: int):
    """(score at t + lag, fall from t, ln odds at t) of every record at t with a lagged record."""
    by_key = {(r["race_id"], r["combination"], r["tau_min"]): r for r in records}
    x, y, level = [], [], []
    for r in records:
        if r["tau_min"] != tau:
            continue
        lagged = by_key.get((r["race_id"], r["combination"], tau + lag))
        if lagged is None:
            continue
        x.append(lagged["value_mean"] - 1.0)
        y.append(math.log(r["board_odds"]) - math.log(finals[(r["race_id"], r["combination"])]))
        level.append(math.log(r["board_odds"]))
    return np.array(x), np.array(y), np.array(level)


def plain_slope(x, y) -> float:
    return float(np.polyfit(x, y, 1)[0])


def partial_slope(x, y, level) -> float:
    design = np.column_stack([np.ones_like(x), x, level])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def test_selection_drift_scores_one_tau_lag_pair_never_pooling_taus():
    records, finals = drift_world(("TOK", "KYO", "NAK", "HAN"), seed=11)
    frame = pd.DataFrame(records)
    frame["final_odds"] = [
        finals[(r, c)] for r, c in zip(frame["race_id"], frame["combination"], strict=True)
    ]
    slopes = {}
    for tau, lag in ((2, 3), (5, 11), (2, 14), (16, 14)):
        x, y, _ = pair_rows(records, finals, tau, lag)
        got = selection_drift(frame, lag_minutes=lag, tau_minutes=tau, **BOOT)
        assert got.n == len(x), f"({tau}, {lag}): {got.n} rows used, {len(x)} by hand"
        assert got.slope == pytest.approx(plain_slope(x, y), abs=1e-9), f"({tau}, {lag})"
        assert got.n_clusters == 4
        slopes[(tau, lag)] = got.slope
    assert abs(slopes[(2, 14)] - slopes[(16, 14)]) > 0.2, "the fixture separates lag 14's pairs"


def test_selection_drift_partials_out_the_odds_level():
    records, finals = drift_world(("TOK", "KYO", "NAK", "HAN"), seed=12)
    frame = pd.DataFrame(records)
    frame["final_odds"] = [
        finals[(r, c)] for r, c in zip(frame["race_id"], frame["combination"], strict=True)
    ]
    x, y, level = pair_rows(records, finals, 2, 14)
    plain = selection_drift(frame, lag_minutes=14, tau_minutes=2, **BOOT)
    partial = selection_drift(frame, lag_minutes=14, tau_minutes=2, partial_log_odds=True, **BOOT)
    assert plain.slope == pytest.approx(plain_slope(x, y), abs=1e-9)
    assert partial.slope == pytest.approx(partial_slope(x, y, level), abs=1e-9)
    assert abs(partial.slope - plain.slope) > 0.5, "the fixture's score tracks the odds level"
    assert partial.n == len(x) and partial.n_clusters == 4
    assert partial.lo <= partial.slope <= partial.hi


def test_the_run_reports_e20_per_pair_with_and_without_the_odds_level(tmp_path, monkeypatch):
    records, finals = drift_world(("TOK", "KYO"), seed=13)
    got = run_v2(monkeypatch, tmp_path, records, [], finals)
    pairs = [(p["tau_min"], p["lag_min"]) for p in shadow_run.build_spec()["drift"]["pairs"]]
    e20 = [v for v in got["variants"] if v["hypothesis_id"] == "E20"]
    names = [v["variant"] for v in got["variants"]]
    assert len(names) == len(set(names)), "variant names repeat"
    assert all("tau_min" in v["params"] for v in e20), "an E20 variant pools every tau of a lag"
    found = {
        (v["params"]["tau_min"], v["params"]["lag_min"], bool(v["params"]["partial_log_odds"])): v
        for v in e20
    }
    assert len(found) == len(e20) == 2 * len(pairs)
    assert set(found) == {(t, lag, partial) for t, lag in pairs for partial in (False, True)}
    x, y, level = pair_rows(records, finals, 2, 14)
    plain = metrics_of(found[(2, 14, False)])["drift_slope"]
    partial = metrics_of(found[(2, 14, True)])["drift_slope"]
    assert plain["value"] == pytest.approx(plain_slope(x, y), abs=1e-9)
    assert partial["value"] == pytest.approx(partial_slope(x, y, level), abs=1e-9)
    assert plain["n"] == partial["n"] == len(x)


# ---------------------------------------------------------------- the hand-played bets


def test_bets_flag_each_day_over_the_cap_and_keep_holdout_bets_holdout(monkeypatch):
    limits = load_limits()
    assert limits.day_cap_yen == 30_000
    held = first_held_day()
    stakes = {
        date(2026, 11, 21): [10_000, 10_000, 9_000],  # 29,000: under
        date(2026, 11, 22): [10_000, 10_000, 10_000],  # 30,000: at the cap, not over
        D28: [20_000, 10_100],  # 30,100: over by 100
        held: [25_000, 15_000],  # 40,000: over, on a holdout day
    }
    paid = {D28: [paying(race("TOK", 1), "1", 450), paying(race("TOK", 2), "9", 700)]}
    paid[held] = [paying(race("TOK", 1, held), "1", 300)]
    for day, amounts in stakes.items():
        log = BetLog(bet_log_path(day), day=day, limits=limits)
        for no, stake in enumerate(amounts, 1):
            log.record(
                {
                    "race_id": race("TOK", no, day),
                    "pool": "WIN",
                    "combination": "1",
                    "stake_yen": stake,
                    "placed_utc": f"{day.isoformat()}T05:40:00+00:00",
                    "option_id": None,
                    "note": "synthetic",
                }
            )
    monkeypatch.setattr(shadow_run, "day_dividends", lambda day: dividends(paid.get(day, [])))
    out, played = {}, {}
    metrics, note, failed = shadow_run._bets(limits, out, played)()
    assert not failed, note
    expected = [
        {"day": d.isoformat(), "staked_yen": sum(a), "over_cap": sum(a) > limits.day_cap_yen}
        for d, a in sorted(stakes.items())
    ]
    assert out["day_stakes"] == expected
    by_name = {m.metric: m.value for m in metrics}
    assert by_name["n_bets"] == sum(len(a) for a in stakes.values())
    for d, a in stakes.items():
        assert by_name[f"staked_yen[{d.isoformat()}]"] == sum(a)
    settled = played["settled"]
    outcome = dict(zip(settled["race_id"], settled["outcome"], strict=True))
    assert {outcome[race("TOK", no, held)] for no in (1, 2)} == {"holdout"}
    assert outcome[race("TOK", 1)] == "won" and outcome[race("TOK", 2)] == "lost"
