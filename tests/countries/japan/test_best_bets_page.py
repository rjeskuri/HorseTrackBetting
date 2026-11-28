"""Ticket #714: the shadow page's best-bets tab per venue and the top-3 highlights on the race page.

Driven from node as in test_shadow_page.py (its harness and readers are reused). Contract pinned
here (the coder implements against it), on top of test_shadow_page.py's and
test_shadow_page_pools.py's:

- `renderBestBets(best, feed, shadowFeed, {now}) -> string` (exported by shadow.js): the venue's
  best-bets tab for `best`, one `/best-bets?venue=` payload (test_best_bets.py). An element
  `data-best-venue="<venue>"` holds it, with `data-best-race="<race_id>"` naming the payload's race
  (none when `race` is null, and then no bet is drawn). One `data-best-bet="<rank>"` element per
  best bet, in the payload's order, carrying `data-best-pool` and `data-best-combination`, with
  these cells (`data-col`):
  - `shrunk` and `raw`: shrunk_per_100 and return_per_100 (per ¥100; any rounding to 0.5 or finer);
    `ci`: ci_lo and ci_hi; `bets`: n; `meetings`: n_meetings; `variants`: variants_searched;
  - `odds`: the plain feed's current odds of that (race, pool, combination) with one decimal (its
    first figure), exactly `-` when the plain feed has no row for it;
  - `p-model`: the shadow race's `p_model[pool][combination]` as a percent with one decimal
    (`12.3%`), exactly `-` without one (Model % is shown, it never selects);
  - `rule`: the payload's `description` (the rule in plain English), escaped like every feed text.
  Each duplicate is an element `data-duplicate="<option_id>"` inside its bet, in the payload's
  order; each unmatched row an element `data-unmatched="<option_id>"` in the tab.
- `renderShadow(feed, shadowFeed, {now, venue, race, bestBets})`: when `bestBets` is the payload of
  the race drawn (same race_id), each of its first three bets whose (pool, combination) has a row
  on the race page gets one badge inside that row, an element `data-best-rank="<rank>"` whose text
  holds the rank, `data-highlight="strong"` when its shrunk_per_100 is 100 or more and
  `"muted"` below; the strong badges share one class list, the muted ones another, and the two
  differ; no inline style. No other row gets a badge: not rank 4 on, not from a payload of another
  race, not without a payload. A page with badges has an element `data-best-rule` stating the
  selection rule (wording left to the reviewer).
- `poll()` also GETs `best-bets?venue=<the selected venue>` (relative or rooted), and the page
  draws the badges of the race drawn from it.
- No stake words and no inputs in either rendering (test_shadow_page.py's rule).

Every payload and feed is a synthetic object built here.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import test_shadow_page as sp

needs_node = sp.needs_node
NOW = sp.NOW
KYO3 = sp.rid("KYO", 3)
KYO10 = sp.rid("KYO", 10)


# ---------------------------------------------------------------- the synthetic payloads


def bet(rank, pool, combination, shrunk, raw, *, lo, hi, n, meetings, variants, dups=()) -> dict:
    return {
        "rank": rank,
        "pool": pool,
        "combination": combination,
        "option_id": f"japan-713-predicted-final-n8/rule-{rank}-{pool}@T-10",
        "family_id": "japan-713-predicted-final-n8",
        "rule": f"rule-{rank}",
        "description": f"plain rule {rank}: {pool} on {combination} of the projected board",
        "shrunk_per_100": shrunk,
        "return_per_100": raw,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_level": 0.95,
        "n": n,
        "n_unit": "races",
        "n_meetings": meetings,
        "variants_searched": variants,
        "highlight": "strong" if shrunk >= 100 else "muted",
        "duplicates": [
            {"option_id": d, "rule": d.split("/")[-1], "shrunk_per_100": shrunk - 3 - i}
            for i, d in enumerate(dups)
        ],
    }


def payload(race_id: str | None, bets: list[dict], unmatched=(), venue="KYO") -> dict:
    no = None if race_id is None else int(race_id.rsplit(":", 1)[1])
    return {
        "race_date": sp.DAY,
        "generated_utc": sp.utc_text(NOW),
        "venue": venue,
        "race": None
        if race_id is None
        else {"race_id": race_id, "race_no": no, "post_utc": sp.utc_text(sp.jst(10, 10))},
        "checkpoint_min": 5,
        "frozen": False,
        "best_bets": bets,
        "unmatched": [{"option_id": o, "family_id": o.split("/")[0]} for o in unmatched],
    }


BEST_1 = payload(
    KYO3,
    [
        bet(1, "WIN", "1", 131.4, 140.2, lo=101.5, hi=171.0, n=412, meetings=57, variants=176,
            dups=("japan-713-predicted-final-n1/n1:WIN:band:lt2@T-5",
                  "japan-713-predicted-final-x435/x435:offset:normalised:WIN:top1@T-5")),
        bet(2, "QUINELLA_PLACE", "1,2", 100.0, 104.0, lo=71.0, hi=139.5, n=100, meetings=20,
            variants=27),
        bet(3, "PLACE", "10", 99.6, 120.3, lo=88.25, hi=152.75, n=233, meetings=41, variants=53),
        bet(4, "WIN", "2", 97.0, 111.0, lo=60.0, hi=160.0, n=180, meetings=33, variants=53),
        bet(5, "TRIO", "1,2,10", 90.0, 95.5, lo=40.0, hi=150.0, n=150, meetings=25, variants=9),
    ],
    unmatched=("japan-82-family-l/L1:TRIO:threshold_3.0", "japan-713-predicted-final-n8/gone@T-5"),
)  # fmt: skip
BEST_2 = payload(
    KYO3,
    [
        bet(1, "WIN", "5", 118.0, 164.0, lo=90.0, hi=230.0, n=301, meetings=62, variants=176),
        bet(2, "WIN", "2", 99.99, 102.5, lo=70.0, hi=135.0, n=510, meetings=88, variants=176,
            dups=("japan-713-predicted-final-n8/n8:wide:1/2-3@T-5",)),
        bet(3, "QUINELLA_PLACE", "2,10", 64.0, 77.0, lo=30.0, hi=120.0, n=120, meetings=22,
            variants=27),
        bet(4, "PLACE", "1", 50.0, 52.0, lo=40.0, hi=64.0, n=900, meetings=120, variants=53),
    ],
)  # fmt: skip
PAYLOADS = {"one": BEST_1, "two": BEST_2}


def shadow_with_p_model() -> dict:
    shadow = copy.deepcopy(sp.SHADOW)
    race = next(r for r in shadow["races"] if r["race_id"] == KYO3)
    race["p_model"] = {
        "WIN": {"1": 0.412, "2": 0.274, "10": 0.061, "5": 0.023},
        "PLACE": {"1": 0.81, "2": 0.66, "10": 0.287, "5": 0.12},
        "QUINELLA_PLACE": {"1,2": 0.333, "1,10": 0.151, "2,10": 0.0849},
    }  # no TRIO: the tab shows '-'
    return shadow


SHADOW = shadow_with_p_model()


def plain_odds(race_id: str) -> dict[tuple[str, str], float | None]:
    race = next(r for v in sp.FEED["venues"] for r in v["races"] if r["race_id"] == race_id)
    return {
        (code, ln["combination"]): ln["odds"]
        for code, body in race["pools"].items()
        for ln in body["rows"]
    }


def p_model_of(pool: str, combination: str) -> float | None:
    race = next(r for r in SHADOW["races"] if r["race_id"] == KYO3)
    return race.get("p_model", {}).get(pool, {}).get(combination)


def numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def near(text: str, value: float, tol: float = 0.5) -> bool:
    return any(abs(x - value) <= tol + 1e-9 for x in numbers(text))


# ---------------------------------------------------------------- rendering


def render_tab(tmp_path: Path, *payloads: dict) -> list:
    steps = [sp.call("renderBestBets", p, sp.FEED, SHADOW, {"now": sp.ms(NOW)}) for p in payloads]
    return [sp.soup(html) for html in sp.results(steps, tmp_path)]


def render_race(tmp_path: Path, *payloads, race=KYO3) -> list:
    steps = []
    for p in payloads:
        opts = {"now": sp.ms(NOW), "venue": "KYO", "race": race}
        if p is not None:
            opts["bestBets"] = p
        steps.append(sp.call("renderShadow", sp.FEED, SHADOW, opts))
    return [sp.soup(html) for html in sp.results(steps, tmp_path)]


def badges_by_row(page, race_id: str) -> dict[tuple[str, str], list]:
    block = sp.race_block(page, race_id)
    out = {}
    for pool_el in block.select("[data-pool]"):
        for row_el in pool_el.select("[data-combination]"):
            found = row_el.select("[data-best-rank]")
            if found:
                out[(pool_el["data-pool"], row_el["data-combination"])] = found
    return out


# ---------------------------------------------------------------- tests


@needs_node
def test_the_tab_lists_each_best_bet_with_its_figures_and_ticket(tmp_path):
    pages = render_tab(tmp_path, BEST_1, BEST_2)
    odds = plain_odds(KYO3)
    for page, best in zip(pages, (BEST_1, BEST_2), strict=True):
        tabs = page.select('[data-best-venue="KYO"]')
        assert len(tabs) == 1, page
        tab = tabs[0]
        assert {e["data-best-race"] for e in page.select("[data-best-race]")} == {KYO3}
        items = tab.select("[data-best-bet]")
        assert [e["data-best-bet"] for e in items] == [str(b["rank"]) for b in best["best_bets"]]
        for item, b in zip(items, best["best_bets"], strict=True):
            where = f"rank {b['rank']}"
            assert item["data-best-pool"] == b["pool"], where
            assert item["data-best-combination"] == b["combination"], where
            assert near(sp.text(sp.cell(item, "shrunk")), b["shrunk_per_100"]), where
            assert near(sp.text(sp.cell(item, "raw")), b["return_per_100"]), where
            ci = sp.text(sp.cell(item, "ci"))
            assert near(ci, b["ci_lo"]) and near(ci, b["ci_hi"]), (where, ci)
            assert b["n"] in numbers(sp.text(sp.cell(item, "bets"))), where
            assert b["n_meetings"] in numbers(sp.text(sp.cell(item, "meetings"))), where
            assert b["variants_searched"] in numbers(sp.text(sp.cell(item, "variants"))), where
            shown_odds = sp.text(sp.cell(item, "odds"))
            o = odds.get((b["pool"], b["combination"]))
            if o is None:
                assert shown_odds == "-", (where, shown_odds)
            else:
                assert re.findall(r"\d+\.\d+", shown_odds)[:1] == [f"{o:.1f}"], (where, shown_odds)
            shown_p = sp.text(sp.cell(item, "p-model"))
            p = p_model_of(b["pool"], b["combination"])
            if p is None:
                assert shown_p == "-", (where, shown_p)
            else:
                m = re.fullmatch(r"(\d+\.\d)\s*%", shown_p)
                assert m and abs(float(m.group(1)) - p * 100) <= 0.05 + 1e-9, (where, shown_p)
            assert b["description"] in sp.text(sp.cell(item, "rule")), where
            dups = [d["data-duplicate"] for d in item.select("[data-duplicate]")]
            assert dups == [d["option_id"] for d in b["duplicates"]], where
        unmatched = [e["data-unmatched"] for e in tab.select("[data-unmatched]")]
        assert sorted(unmatched) == sorted(u["option_id"] for u in best["unmatched"])
        html = str(page)
        assert not sp.stake_words_in(html), sorted(set(sp.stake_words_in(html)))
        assert not sp.AMOUNT_INPUTS.search(html)


@needs_node
def test_a_tab_without_a_next_race_draws_no_bet_and_feed_text_is_escaped(tmp_path):
    hostile = copy.deepcopy(BEST_2)
    said = '<script>alert("x")</script> & <marquee>loud</marquee>'
    hostile["best_bets"][0]["description"] = said
    empty = payload(None, [], venue="TOK")
    tab, gone = render_tab(tmp_path, hostile, empty)
    assert not tab.select("script") and not tab.select("marquee")
    first = tab.select("[data-best-bet]")[0]
    assert said in sp.text(sp.cell(first, "rule"))
    assert len(gone.select('[data-best-venue="TOK"]')) == 1
    assert not gone.select("[data-best-bet]") and not gone.select("[data-best-race]")


@needs_node
@pytest.mark.parametrize("name", sorted(PAYLOADS))
def test_the_race_page_badges_the_top_three_in_two_colours(tmp_path, name):
    best = PAYLOADS[name]
    page, other_race, no_payload = render_race(
        tmp_path, best, payload(KYO10, best["best_bets"]), None
    )
    badges = badges_by_row(page, KYO3)
    top = best["best_bets"][:3]
    assert set(badges) == {(b["pool"], b["combination"]) for b in top}, badges
    tone_classes: dict[str, set] = {"strong": set(), "muted": set()}
    for b in top:
        found = badges[(b["pool"], b["combination"])]
        assert len(found) == 1, found
        badge = found[0]
        assert badge["data-best-rank"] == str(b["rank"])
        assert str(b["rank"]) in sp.text(badge)
        tone = "strong" if b["shrunk_per_100"] >= 100 else "muted"
        assert badge.get("data-highlight") == tone, (b, badge)
        assert not badge.get("style")
        tone_classes[tone].add(tuple(sorted(sp.classes(badge))))
    for tone, seen in tone_classes.items():
        assert len(seen) <= 1, (tone, seen)
    if tone_classes["strong"] and tone_classes["muted"]:
        assert tone_classes["strong"] != tone_classes["muted"]
    rules = page.select("[data-best-rule]")
    assert rules and sp.text(rules[0]), "the page states the selection rule beside the badges"
    assert not badges_by_row(other_race, KYO3), "a payload of another race badges nothing"
    assert not badges_by_row(no_payload, KYO3)
    html = str(page)
    assert not sp.stake_words_in(html), sorted(set(sp.stake_words_in(html)))


@needs_node
def test_strong_and_muted_badges_look_different(tmp_path):
    """BEST_1 has two strong (131.4, 100.0) and one muted (99.6); BEST_2 one strong, two muted."""
    pages = render_race(tmp_path, BEST_1, BEST_2)
    strong, muted = set(), set()
    for page in pages:
        for found in badges_by_row(page, KYO3).values():
            key = tuple(sorted(sp.classes(found[0])))
            (strong if found[0].get("data-highlight") == "strong" else muted).add(key)
    assert len(strong) == 1 and len(muted) == 1, (strong, muted)
    assert strong != muted


BEST_ROUTE_OLD = r'const key = /(^|\/)shadow-feed(\?|$)/.test(u) ? "shadow"'
BEST_ROUTE_NEW = (
    r'const key = /(^|\/)best-bets(\?|$)/.test(u) ? "best" : '
    r'/(^|\/)shadow-feed(\?|$)/.test(u) ? "shadow"'
)


@needs_node
def test_the_page_polls_the_selected_venues_best_bets_and_badges_its_race(tmp_path, monkeypatch):
    assert BEST_ROUTE_OLD in sp.HARNESS
    monkeypatch.setattr(sp, "HARNESS", sp.HARNESS.replace(BEST_ROUTE_OLD, BEST_ROUTE_NEW))
    servers = {
        "plain": {"feed": sp.FEED, "etag": '"p1"'},
        "shadow": {"feed": SHADOW, "etag": '"s1"'},
        "best": {"feed": BEST_1, "etag": '"b1"'},
    }
    out = sp.run_js([{"fire": 5000}, {"fire": 5000}], tmp_path, servers=servers, browser=True)
    urls = [r["url"] for step in out["results"] for r in step["requests"]]
    asked = [u for u in urls if re.search(r"(^|/)best-bets\?", u)]
    assert asked, urls
    assert any(re.search(r"[?&]venue=KYO(&|$)", u) for u in asked), asked
    page = sp.soup(out["results"][-1]["shadow"])
    badges = badges_by_row(page, KYO3)  # KYO 3 is KYO's next race at NOW, the default draw
    assert set(badges) == {(b["pool"], b["combination"]) for b in BEST_1["best_bets"][:3]}
