"""The shadow page's bet-type sections, exotic boards and model probabilities (ticket #506).

Driven from node as in test_shadow_page.py (its harness and readers are reused). Contract pinned
here (the coder implements against it), on top of test_shadow_page.py's:

- Sections: for the race drawn, one `data-pool` section per bet type, whatever the plain feed
  holds: WIN, PLACE, QUINELLA, QUINELLA_PLACE (Wide), FORECAST (Exacta), TRIO, TIERCE (Trifecta),
  BRACKET_QUINELLA; each section's heading names it (Win, Place, Quinella, Wide, Exacta, Trio,
  Trifecta, Bracket quinella). With no shadow feed the sections are still drawn.
- Rows: WIN, PLACE, QUINELLA, QUINELLA_PLACE and BRACKET_QUINELLA list every row of the plain
  feed's pool, favourite first as on the board (#498). FORECAST, TRIO and TIERCE list the 15
  combinations with the largest `p_model` of the shadow race (all of them when fewer exist),
  `p_model` descending, whatever the plain feed's order or odds; each row `data-combination` is
  the combination text, its `data-col="odds"` the plain row's odds with one decimal, or `-` when
  the plain feed has no row for it (no capture).
- Model p: every row of every section has one `data-col="p-model"` cell under a header reading
  `Model p`: the shadow race's `p_model[pool][combination]` as a percent with one decimal (e.g.
  `12.3%`), `-` without one.
- Value: WIN, PLACE and Wide keep the shadow entry's values (test_shadow_page.py). QUINELLA,
  BRACKET_QUINELLA, FORECAST, TRIO and TIERCE have no price forecast (no nowcast, #400): their
  `data-col="value"` is p_model x the plain row's current odds with two decimals (gross return
  per 1 yen), `-` without both; and their section says `no price forecast`, which the WIN, PLACE
  and Wide sections do not.
- Capture age: each tier-2 section (FORECAST, TRIO, TIERCE, BRACKET_QUINELLA) has one element
  `data-capture-age` stating the age of the plain feed pool's `capture_utc` at `now` in whole
  minutes (e.g. `7 min`); with no capture, no minutes figure.
- Never coloured or styled by value or probability: within a section the value cells that show a
  figure share one class list, so do the Model p cells, and no row or cell has an inline style.

The plain feed's tier-2 pools are `{investment, capture_utc, rows}` (test_tier2_log_readers.py);
the shadow race's `p_model` is {pool: {combination: probability}} (test_shadow_p_model.py). The
feeds are synthetic objects built here, one test builds both from synthetic odds logs.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta
from itertools import combinations, combinations_with_replacement, permutations
from pathlib import Path

import test_shadow_page as sp

# isort: split
from page_fakes import favourite_order

needs_node = sp.needs_node
ALL_POOLS = sp.ALL_POOLS
EXOTICS = ("FORECAST", "TRIO", "TIERCE")
TIER2 = ("BRACKET_QUINELLA", "FORECAST", "TRIO", "TIERCE")
NO_FORECAST = ("QUINELLA", "BRACKET_QUINELLA", "FORECAST", "TRIO", "TIERCE")
NOWCAST_POOLS = ("WIN", "PLACE", "QUINELLA_PLACE")
LABELS = {
    "WIN": "win",
    "PLACE": "place",
    "QUINELLA": "quinella",
    "QUINELLA_PLACE": "wide",
    "FORECAST": "exacta",
    "TRIO": "trio",
    "TIERCE": "trifecta",
    "BRACKET_QUINELLA": "bracket quinella",
}
TOP = 15
NOW = sp.NOW


# ---------------------------------------------------------------- combinations


def brackets_of(drawn: int) -> dict[int, int]:
    sizes = {4: (1, 1, 1, 1), 6: (1,) * 6, 12: (1, 1, 1, 1, 2, 2, 2, 2), 10: (1,) * 6 + (2, 2)}
    out, horse = {}, 1
    for bracket, size in enumerate(sizes[drawn], start=1):
        for _ in range(size):
            out[horse] = bracket
            horse += 1
    return out


def combos(pool: str, horses: list[int]) -> list[str]:
    def join(parts) -> str:
        return ",".join(str(h) for h in parts)

    if pool in ("WIN", "PLACE"):
        return [str(h) for h in horses]
    if pool in ("QUINELLA", "QUINELLA_PLACE"):
        return [join(c) for c in combinations(horses, 2)]
    if pool == "FORECAST":
        return [join(c) for c in permutations(horses, 2)]
    if pool == "TRIO":
        return [join(c) for c in combinations(horses, 3)]
    if pool == "TIERCE":
        return [join(c) for c in permutations(horses, 3)]
    brackets = brackets_of(len(horses))
    size = {b: list(brackets.values()).count(b) for b in set(brackets.values())}
    return [
        join((a, b))
        for a, b in combinations_with_replacement(sorted(size), 2)
        if a != b or size[a] >= 2
    ]


def top(p_model: dict[str, float], k: int = TOP) -> list[str]:
    return [c for c, _ in sorted(p_model.items(), key=lambda item: -item[1])[:k]]


# ---------------------------------------------------------------- the synthetic feeds


def distinct(rnd: random.Random, n: int, low: float, high: float) -> list[float]:
    """n different odds with one decimal in [low, high)."""
    seen: set[float] = set()
    out = []
    while len(out) < n:
        x = round(low + rnd.random() * (high - low), 1)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_race(no: int, horses: int, post: datetime, *, seed: int, plain_pools, ages, missing):
    """(plain race, shadow race) of TOK `no`: plain rows for `plain_pools` (the tier-2 ones with
    `capture_utc` = NOW - ages[pool], less the combinations `missing[pool]` names by rank in
    p_model), and a shadow race whose p_model covers every pool's combinations."""
    rnd = random.Random(seed)
    field = list(range(1, horses + 1))
    p_model: dict[str, dict[str, float]] = {}
    for pool_code in ALL_POOLS:
        if pool_code == "BRACKET_QUINELLA" and horses < 9:
            continue
        keys = combos(pool_code, field)
        weights = [rnd.random() ** 4 + 1e-6 * i for i, _ in enumerate(keys)]
        scale = {"PLACE": 3.0, "QUINELLA_PLACE": 3.0}.get(pool_code, 1.0) / sum(weights)
        p_model[pool_code] = {k: w * scale for k, w in zip(keys, weights, strict=True)}
    plain_pools_out = {}
    for pool_code in plain_pools:
        keys = combos(pool_code, field)
        drop = {top(p_model[pool_code], 10**6)[i] for i in missing.get(pool_code, ())}
        odds = distinct(rnd, len(keys), 1.5, 400.0 if pool_code in EXOTICS else 90.0)
        rows = []
        for k, o in zip(keys, odds, strict=True):
            if k in drop:
                continue
            hi = round(o * 1.4, 1) if pool_code in ("PLACE", "QUINELLA_PLACE") else None
            rows.append(sp.line(k, o, hi=hi, mm=0.0, t20=0.0, t2=0.0))
        rnd.shuffle(rows)  # the feed's order is neither the crowd's nor the model's
        body = sp.pool(*rows, investment=1_000_000 + no)
        if pool_code in TIER2:
            body["capture_utc"] = sp.utc_text(NOW - ages[pool_code])
        plain_pools_out[pool_code] = body
    plain = sp.race("TOK", no, post, NOW - timedelta(seconds=20), plain_pools_out)
    values = {
        "WIN": {str(h): sp.win(0.7 + 0.01 * h, 0.6, 0.9) for h in field},
        "PLACE": {str(h): sp.rng(0.8, 0.9 + 0.01 * h) for h in field},
        "QUINELLA_PLACE": {c: sp.rng(0.75, 0.95) for c in combos("QUINELLA_PLACE", field)},
    }
    shadow = sp.shadow_race("TOK", no, post, 16, values)
    shadow["p_model"] = p_model
    return plain, shadow


def minutes(m: int, s: int) -> timedelta:
    return timedelta(minutes=m, seconds=s)


AGES_A = {  # seconds under 30: floor and round agree on the minutes
    "BRACKET_QUINELLA": minutes(3, 10),
    "FORECAST": minutes(7, 15),
    "TRIO": minutes(12, 5),
    "TIERCE": minutes(4, 20),
}
AGES_B = {
    "BRACKET_QUINELLA": minutes(6, 25),
    "FORECAST": minutes(5, 0),
    "TRIO": minutes(9, 20),
    "TIERCE": minutes(15, 10),
}


def feeds() -> tuple[dict, dict]:
    """TOK 9 (12 runners, every pool, some top-15 exotics and a few p_model entries missing), TOK
    10 (4 runners: fewer exotic combinations than 15 for Exacta and Trio, no bracket quinella),
    TOK 11 (6 runners, WIN and PLACE only on the plain feed: nothing captured for the rest)."""
    a_plain, a_shadow = build_race(
        9,
        12,
        sp.jst(10, 20),
        seed=506,
        plain_pools=ALL_POOLS,
        ages=AGES_A,
        missing={"FORECAST": (1, 8), "TRIO": (0,), "TIERCE": (4, 14)},
    )
    for gone in list(a_shadow["p_model"]["QUINELLA"])[:2]:  # plain rows without a probability
        del a_shadow["p_model"]["QUINELLA"][gone]
    del a_shadow["p_model"]["BRACKET_QUINELLA"]["5,5"]
    b_plain, b_shadow = build_race(
        10,
        4,
        sp.jst(10, 40),
        seed=4,
        plain_pools=[p for p in ALL_POOLS if p != "BRACKET_QUINELLA"],
        ages=AGES_B,
        missing={},
    )
    c_plain, c_shadow = build_race(
        11, 6, sp.jst(11, 0), seed=6, plain_pools=["WIN", "PLACE"], ages={}, missing={}
    )
    plain = {
        "race_date": sp.DAY,
        "generated_utc": sp.utc_text(NOW),
        "venues": [{"venue": "TOK", "races": [a_plain, b_plain, c_plain]}],
    }
    shadow = {
        "race_date": sp.DAY,
        "generated_utc": sp.utc_text(NOW),
        "models": {"win": "w", "ordering": "o", "nowcast": "n"},
        "races": [c_shadow, a_shadow, b_shadow],
        "win5": None,
    }
    return plain, shadow


PLAIN, SHADOW = feeds()
RACES = [r["race_id"] for r in PLAIN["venues"][0]["races"]]


def plain_race(race_id: str, plain: dict = PLAIN) -> dict:
    return next(r for v in plain["venues"] for r in v["races"] if r["race_id"] == race_id)


def shadow_race(race_id: str, shadow: dict = SHADOW) -> dict:
    return next(r for r in shadow["races"] if r["race_id"] == race_id)


def _check_fixture() -> None:
    """The fixture can tell the model's order from the crowd's, the feed's and the value's."""
    a = RACES[0]
    for pool_code in EXOTICS:
        p = shadow_race(a)["p_model"][pool_code]
        rows = plain_race(a)["pools"][pool_code]["rows"]
        odds = {r["combination"]: r["odds"] for r in rows}
        want = top(p)
        assert want != [r["combination"] for r in rows][:TOP]
        assert want != favourite_order(rows)[:TOP]
        by_value = sorted(odds, key=lambda c: -p[c] * odds[c])[:TOP]
        assert want != by_value
        assert [c for c in want if c not in odds], f"{pool_code}: every top row is captured"
    assert len(shadow_race(RACES[1])["p_model"]["FORECAST"]) == 12 < TOP
    assert len(shadow_race(RACES[1])["p_model"]["TRIO"]) == 4


_check_fixture()


# ---------------------------------------------------------------- reading the page


def render(race_ids, plain=PLAIN, shadow=SHADOW, *, tmp_path: Path, now=NOW) -> dict:
    steps = [
        sp.call("renderShadow", plain, shadow, {"now": sp.ms(now), "venue": "TOK", "race": r})
        for r in race_ids
    ]
    shown = sp.results(steps, tmp_path)
    return {r: sp.soup(html) for r, html in zip(race_ids, shown, strict=True)}


def section(page, race_id: str, pool_code: str):
    return sp.pool_block(sp.race_block(page, race_id), pool_code)


def rows_in(section_el) -> list[str]:
    return [r["data-combination"] for r in section_el.select("[data-combination]")]


def percent_ok(shown: str, p: float | None) -> bool:
    if p is None:
        return shown == "-"
    m = re.fullmatch(r"(\d+\.\d)\s*%", shown)
    return m is not None and abs(float(m.group(1)) - p * 100) <= 0.05 + 1e-9


def figure_ok(shown: str, value: float | None) -> bool:
    if value is None:
        return shown == "-"
    found = re.findall(r"\d+\.\d+", shown)
    return (
        len(found) == 1
        and len(found[0].split(".")[1]) == 2
        and abs(float(found[0]) - value) <= 0.005 + 1e-9
    )


def odds_of(race_id: str, pool_code: str, plain: dict = PLAIN) -> dict[str, float]:
    body = plain_race(race_id, plain)["pools"].get(pool_code) or {"rows": []}
    return {r["combination"]: r["odds"] for r in body["rows"] if r["odds"] is not None}


# ---------------------------------------------------------------- tests


@needs_node
def test_every_race_shows_a_section_per_bet_type(tmp_path):
    pages = render(RACES, tmp_path=tmp_path)
    bare = render(RACES, shadow=None, tmp_path=tmp_path)
    old = [r["race_id"] for v in sp.FEED["venues"] for r in v["races"]]
    old_pages = {}
    for v in sp.FEED["venues"]:
        for r in v["races"]:
            opts = {"now": sp.ms(NOW), "venue": v["venue"], "race": r["race_id"]}
            old_pages[r["race_id"]] = sp.call("renderShadow", sp.FEED, sp.SHADOW, opts)
    shown_old = sp.results(list(old_pages.values()), tmp_path)
    cases = [(pages[r], r) for r in RACES] + [(bare[r], r) for r in RACES]
    cases += [(sp.soup(html), r) for r, html in zip(old, shown_old, strict=True)]
    for page, race_id in cases:
        block = sp.race_block(page, race_id)
        codes = [e["data-pool"] for e in block.select("[data-pool]")]
        assert sorted(codes) == sorted(ALL_POOLS), (race_id, codes)
        for pool_code in ALL_POOLS:
            heading = sp.pool_block(block, pool_code).find(["h2", "h3", "h4"])
            assert heading is not None, (race_id, pool_code)
            assert LABELS[pool_code] in sp.text(heading).lower(), (race_id, pool_code, heading)

    # the page as the browser runs it: plain scripts, both feeds polled
    servers = {
        "plain": {"feed": PLAIN, "etag": '"p"'},
        "shadow": {"feed": SHADOW, "etag": '"s"'},
    }
    out = sp.run_js([{"fire": 5000}], tmp_path, servers=servers, browser=True)
    loaded = sp.soup(out["results"][-1]["shadow"])
    a = RACES[0]  # the next race to post at NOW
    assert [b["data-race"] for b in sp.race_blocks(loaded)] == [a]
    assert rows_in(section(loaded, a, "TIERCE")) == top(shadow_race(a)["p_model"]["TIERCE"])


@needs_node
def test_exotics_list_the_fifteen_most_probable_combinations(tmp_path):
    pages = render(RACES, tmp_path=tmp_path)
    dashes = 0
    for race_id in RACES:
        p_model = shadow_race(race_id)["p_model"]
        for pool_code in EXOTICS:
            el = section(pages[race_id], race_id, pool_code)
            want = top(p_model[pool_code])
            assert len(want) == min(TOP, len(p_model[pool_code]))
            assert rows_in(el) == want, (race_id, pool_code, rows_in(el)[:5], want[:5])
            odds = odds_of(race_id, pool_code)
            for combination in want:
                row_el = sp.row_of(el, combination)
                p, o = p_model[pool_code][combination], odds.get(combination)
                where = (race_id, pool_code, combination)
                shown_odds = sp.text(sp.cell(row_el, "odds"))
                assert shown_odds == ("-" if o is None else f"{o:.1f}"), (where, shown_odds)
                shown_p = sp.text(sp.cell(row_el, "p-model"))
                assert percent_ok(shown_p, p), (where, shown_p, p)
                shown_value = sp.text(sp.cell(row_el, "value"))
                assert figure_ok(shown_value, None if o is None else p * o), (where, shown_value)
                dashes += o is None
    assert dashes >= 5 + 15 * 3  # TOK 9's missing rows, and TOK 11 has no capture at all

    # no shadow feed: nothing ranks the exotics, so no rows are drawn for them
    bare = render(RACES[:1], shadow=None, tmp_path=tmp_path)[RACES[0]]
    for pool_code in EXOTICS:
        assert rows_in(section(bare, RACES[0], pool_code)) == [], pool_code


@needs_node
def test_every_pool_table_has_a_model_p_column(tmp_path):
    pages = render(RACES, tmp_path=tmp_path)
    old = render([sp.rid("TOK", 1)], plain=sp.FEED, shadow=sp.SHADOW, tmp_path=tmp_path)
    checked = {"figure": 0, "dash": 0}
    for race_id in RACES:
        p_model = shadow_race(race_id)["p_model"]
        plain = plain_race(race_id)
        for pool_code in ALL_POOLS:
            el = section(pages[race_id], race_id, pool_code)
            combos_shown = rows_in(el)
            if pool_code not in EXOTICS:  # every plain row, favourite first
                body = plain["pools"].get(pool_code) or {"rows": []}
                assert combos_shown == favourite_order(body["rows"]), (race_id, pool_code)
            if not combos_shown:
                continue
            heads = [sp.text(th) for th in el.select("thead th")]
            first = sp.row_of(el, combos_shown[0])
            positions = first.find_all(["th", "td"], recursive=False)
            assert "Model p" in heads, (race_id, pool_code, heads)
            assert heads.index("Model p") == positions.index(sp.cell(first, "p-model")), heads
            odds = odds_of(race_id, pool_code)
            for combination in combos_shown:
                row_el = sp.row_of(el, combination)
                p = (p_model.get(pool_code) or {}).get(combination)
                shown = sp.text(sp.cell(row_el, "p-model"))
                assert percent_ok(shown, p), (race_id, pool_code, combination, shown, p)
                checked["dash" if p is None else "figure"] += 1
                if pool_code in ("QUINELLA", "BRACKET_QUINELLA"):  # no forecast: p x odds now
                    o = odds.get(combination)
                    want = None if p is None or o is None else p * o
                    value = sp.text(sp.cell(row_el, "value"))
                    assert figure_ok(value, want), (race_id, pool_code, combination, value)
            # never coloured by value or probability
            for col in ("value", "p-model"):
                cells = el.select(f'[data-col="{col}"]')
                with_figures = [c for c in cells if sp.text(c) != "-"]
                assert len({tuple(sp.classes(c)) for c in with_figures}) <= 1, (pool_code, col)
            assert not [e for e in el.select("[style]")], (race_id, pool_code, "inline style")
    assert checked["dash"] >= 3 and checked["figure"] >= 100, checked
    # WIN, PLACE and Wide keep their shadow values beside the new column
    a = RACES[0]
    entry = shadow_race(a)["pools"]["WIN"]["7"]
    sp.check_value(
        sp.text(sp.cell(sp.row_of(section(pages[a], a, "WIN"), "7"), "value")), "WIN", entry, "W7"
    )
    # a shadow feed without p_model: every Model p is '-'
    t1 = sp.rid("TOK", 1)
    for pool_code in ("WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"):
        for combination in rows_in(section(old[t1], t1, pool_code)):
            row_el = sp.row_of(section(old[t1], t1, pool_code), combination)
            assert sp.text(sp.cell(row_el, "p-model")) == "-", (pool_code, combination)


@needs_node
def test_pools_without_a_price_forecast_are_flagged(tmp_path):
    pages = render(RACES, tmp_path=tmp_path)
    for race_id in RACES:
        for pool_code in ALL_POOLS:
            said = sp.text(section(pages[race_id], race_id, pool_code)).lower()
            flagged = "no price forecast" in said
            assert flagged == (pool_code in NO_FORECAST), (race_id, pool_code, flagged)


@needs_node
def test_tier2_sections_show_their_capture_age(tmp_path):
    pages = render(RACES, tmp_path=tmp_path)
    for race_id, ages in ((RACES[0], AGES_A), (RACES[1], AGES_B), (RACES[2], {})):
        for pool_code in TIER2:
            el = section(pages[race_id], race_id, pool_code)
            stamps = el.select("[data-capture-age]")
            captured = pool_code in plain_race(race_id)["pools"]
            age = ages.get(pool_code) if captured else None
            if age is None:  # nothing captured: no minutes figure
                assert all(not re.search(r"\d+\s*min", sp.text(s)) for s in stamps), stamps
                continue
            assert len(stamps) == 1, (race_id, pool_code, stamps)
            figures = re.findall(r"(\d+)\s*min", sp.text(stamps[0]))
            assert figures == [str(age.seconds // 60)], (race_id, pool_code, sp.text(stamps[0]))
    # ten minutes on, every age has grown by ten
    later = render(RACES[:1], tmp_path=tmp_path, now=NOW + timedelta(minutes=10))[RACES[0]]
    for pool_code in TIER2:
        stamp = section(later, RACES[0], pool_code).select("[data-capture-age]")
        assert re.findall(r"(\d+)\s*min", sp.text(stamp[0])) == [
            str(AGES_A[pool_code].seconds // 60 + 10)
        ], pool_code


# ---------------------------------------------------------------- from the logs to the page


def _row(race_id, pool, combination, odds, at, **extra) -> dict:
    local = at.astimezone(sp.JST)
    return {
        "snapshot_utc": sp.utc_text(at),
        "fetched_utc": sp.utc_text(at),
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "odds": odds,
        "status_raw": f"{local.hour}時{local.minute:02d}分現在",
        "page_state": None,
        "source_update_utc": sp.utc_text(at - timedelta(seconds=10)),
        **extra,
    }


@needs_node
def test_a_logged_day_shows_the_exotics_with_their_odds(tmp_path, monkeypatch):
    """Synthetic tier-1 and tier-2 captures of a 10-runner race through build_board, build_shadow
    (FakeWin and a real ordering fit) and the page: the feeds' combination texts join, so every
    top-15 exotic and every bracket pair shows its logged odds, its p_model and p x odds."""
    from horsetrack.core import paths
    from horsetrack.countries.japan import odds_log
    from horsetrack.countries.japan.board import feed as feed_mod
    from horsetrack.countries.japan.board.flags import FlagRules
    from horsetrack.countries.japan.board.shadow_feed import build_shadow
    from horsetrack.countries.japan.models.v0.ordering import OrderingFit
    from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels

    # isort: split
    from shadow_fakes import FakeNowcast, FakeWin

    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    day = datetime.fromisoformat(sp.DAY).date()
    race_id, post = sp.rid("TOK", 5), sp.jst(11, 0)
    t1_at, t2_at, now = sp.jst(10, 50), sp.jst(10, 46), sp.jst(10, 52)
    field = list(range(1, 11))
    win_odds = {
        h: o
        for h, o in zip(field, (6.1, 2.4, 15.8, 4.2, 33.0, 9.7, 61.5, 3.3, 21.0, 12.6), strict=True)
    }
    rows = [_row(race_id, "WIN", str(h), o, t1_at) for h, o in win_odds.items()]
    for h, o in win_odds.items():
        low, high = round(1 + o / 5, 1), round(1.5 + o / 4, 1)
        rows.append(_row(race_id, "PLACE", str(h), low, t1_at, odds_hi=high))
    for a, b in combinations(field, 2):
        q = round(win_odds[a] * win_odds[b] / 1.7, 1)
        rows.append(_row(race_id, "QUINELLA", f"{a},{b}", q, t1_at))
        rows.append(_row(race_id, "QUINELLA_PLACE", f"{a},{b}", round(q / 3, 1), t1_at, odds_hi=q))
    tier2 = [
        _row(race_id, "BRACKET_QUINELLA", c, round(3.0 + 1.7 * i, 1), t2_at)
        for i, c in enumerate(combos("BRACKET_QUINELLA", field))
    ]
    for pool_code, scale in (("FORECAST", 2.1), ("TRIO", 3.3), ("TIERCE", 11.0)):
        for c in combos(pool_code, field):
            parts = [win_odds[int(h)] for h in c.split(",")]
            o = round(
                scale * parts[0] * (parts[1] ** 0.8) * (parts[2] ** 0.6 if len(parts) > 2 else 1), 1
            )
            tier2.append(_row(race_id, pool_code, c, o, t2_at))
    log = paths.odds_snapshot_path("jra", sp.DAY, "TOK")
    odds_log.append_capture(log, tier2)
    odds_log.append_capture(log, rows)
    odds_log.write_post_times(log.with_name("TOK.races.json"), {5: post}, sp.jst(7, 0))
    models = ShadowModels(
        win=FakeWin(),
        ordering=OrderingFit("discounted_harville", (0.85, 0.7), "ord", 100),
        nowcast=FakeNowcast(),
        ids={"win": "w", "ordering": "o", "nowcast": "n"},
    )
    rules = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
    plain = json.loads(json.dumps(feed_mod.build_board(day, now=lambda: now)))
    shadow = json.loads(
        json.dumps(
            build_shadow(day, now=lambda: now, models=models, flag_rules=rules, n_draws=8, seed=1)
        )
    )

    page = render([race_id], plain, shadow, tmp_path=tmp_path, now=now)[race_id]

    p_model = shadow_race(race_id, shadow).get("p_model") or {}
    for pool_code in EXOTICS:
        assert len(p_model.get(pool_code) or {}) == len(combos(pool_code, field)), pool_code
        el = section(page, race_id, pool_code)
        want = top(p_model[pool_code])
        assert rows_in(el) == want, (pool_code, rows_in(el)[:3], want[:3])
        logged = {r["combination"]: r["odds"] for r in tier2 if r["pool"] == pool_code}
        for combination in want:
            row_el = sp.row_of(el, combination)
            p, o = p_model[pool_code][combination], logged[combination]
            assert sp.text(sp.cell(row_el, "odds")) == f"{o:.1f}", (pool_code, combination)
            assert percent_ok(sp.text(sp.cell(row_el, "p-model")), p), (pool_code, combination)
            assert figure_ok(sp.text(sp.cell(row_el, "value")), p * o), (pool_code, combination)
        stamp = el.select("[data-capture-age]")
        assert len(stamp) == 1 and re.findall(r"(\d+)\s*min", sp.text(stamp[0])) == ["6"]
    bracket = section(page, race_id, "BRACKET_QUINELLA")
    assert sorted(rows_in(bracket)) == sorted(combos("BRACKET_QUINELLA", field))
    for combination in rows_in(bracket):
        p = (p_model.get("BRACKET_QUINELLA") or {}).get(combination)
        shown = sp.text(sp.cell(sp.row_of(bracket, combination), "p-model"))
        assert p is not None and percent_ok(shown, p), (combination, shown, p)
