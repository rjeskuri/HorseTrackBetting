"""The horse member on the shadow page (ticket #717): a "Horse model %" column and the p/q mark.

Driven from node as in test_shadow_page.py (its harness and readers are reused). Contract pinned
here, on top of test_shadow_page.py's and test_shadow_page_pools.py's:

- A race whose shadow race has a `member` with figures (`status` null; test_shadow_member.py)
  shows, in its Win section, a header cell reading `Horse model %` next to the `Model p` header,
  and on every row a `data-col="horse-model"` cell under it: the shadow race's
  `p_member[horse]` as a percent with one decimal (e.g. `12.3%`), `-` for a horse without one
  (scratched). The row of each horse in `member.marked` (p_member / q at least the pointer's k)
  carries one mark, an element with the attribute `data-member-mark`; no other row has one. The
  rows keep the board's order, favourite first (#498): the member neither sorts nor drops rows.
- A race whose `member.status` is `no JRDB data` says `no JRDB data` in its Win section and shows
  no horse-model percent and no mark.
- Any other race (`member` null, or a shadow race of the older shape without the keys) shows no
  horse-model figure, no `no JRDB data` and no mark (a horse-model cell, if drawn, is empty or `-`).

The feeds are synthetic objects built here.
"""

from __future__ import annotations

import random
import re

import test_shadow_page as sp

# isort: split
from page_fakes import favourite_order

needs_node = sp.needs_node
NO_JRDB = "no JRDB data"
HEAD = "Horse model %"


def odds_of(n: int, seed: int) -> dict[str, float]:
    rnd = random.Random(seed)
    values = rnd.sample([round(1.6 + 0.7 * i + rnd.random() * 0.5, 1) for i in range(40)], n)
    return {str(h): o for h, o in enumerate(values, 1)}


def member_race(no: int, n: int, seed: int, *, segment: str, k: float, scratched=()):
    """(plain race, shadow race, p_member, marked) of TOK `no` with figures: p_member over the
    runners not scratched, marked the horses with p / q >= k, q the plain odds normalised."""
    rnd = random.Random(seed)
    odds = odds_of(n, seed)
    running = [h for h in odds if h not in scratched]
    inv = {h: 1.0 / odds[h] for h in running}
    total = sum(inv.values())
    weights = {h: inv[h] / total * rnd.uniform(0.3, 2.6) for h in running}
    norm = sum(weights.values())
    p_member = {h: w / norm for h, w in weights.items()}
    marked = sorted((h for h in running if p_member[h] / (inv[h] / total) >= k), key=int)
    post = sp.jst(10, 5 + 10 * no)
    win_rows = [sp.line(h, o, scr=h in scratched) for h, o in odds.items()]
    place_rows = [
        sp.line(h, round(1 + (o - 1) / 4, 1), hi=round(o / 2 + 1, 1)) for h, o in odds.items()
    ]
    plain = sp.race(
        "TOK", no, post, sp.NOW, {"WIN": sp.pool(*win_rows), "PLACE": sp.pool(*place_rows)}
    )
    values = {"WIN": {h: sp.win(0.8, 0.7, 0.9) for h in running}}
    shadow = sp.shadow_race("TOK", no, post, 16, values)
    shadow["p_model"] = {"WIN": {h: inv[h] / total for h in running}}
    shadow["p_member"] = p_member
    shadow["member"] = {"segment": segment, "k": k, "status": None, "marked": marked}
    return plain, shadow, p_member, set(marked)


def other_race(no: int, n: int, seed: int, member: dict | None, *, keys: bool = True):
    odds = odds_of(n, seed)
    post = sp.jst(10, 5 + 10 * no)
    rows = [sp.line(h, o) for h, o in odds.items()]
    plain = sp.race("TOK", no, post, sp.NOW, {"WIN": sp.pool(*rows)})
    shadow = sp.shadow_race("TOK", no, post, 30, {"WIN": {h: sp.win(0.8, 0.7, 0.9) for h in odds}})
    total = sum(1 / o for o in odds.values())
    shadow["p_model"] = {"WIN": {h: (1 / o) / total for h, o in odds.items()}}
    if keys:
        shadow["p_member"] = None
        shadow["member"] = member
    return plain, shadow


def feeds():
    a_plain, a_shadow, a_p, a_marked = member_race(
        1, 10, 73, segment="DEBUT", k=1.3, scratched=("4",)
    )
    e_plain, e_shadow, e_p, e_marked = member_race(5, 14, 75, segment="MAIDEN", k=1.15)
    b_plain, b_shadow = other_race(
        2, 12, 72, {"segment": "MAIDEN", "k": 1.5, "status": NO_JRDB, "marked": []}
    )
    c_plain, c_shadow = other_race(3, 9, 73, None)
    d_plain, d_shadow = other_race(4, 8, 74, None, keys=False)
    plain = {
        "race_date": sp.DAY,
        "generated_utc": sp.utc_text(sp.NOW),
        "venues": [{"venue": "TOK", "races": [a_plain, b_plain, c_plain, d_plain, e_plain]}],
    }
    shadow = {
        "race_date": sp.DAY,
        "generated_utc": sp.utc_text(sp.NOW),
        "models": {"win": "w", "ordering": "o", "nowcast": "n"},
        "races": [e_shadow, c_shadow, a_shadow, d_shadow, b_shadow],
        "win5": None,
    }
    figures = {a_shadow["race_id"]: (a_p, a_marked), e_shadow["race_id"]: (e_p, e_marked)}
    return plain, shadow, figures


PLAIN, SHADOW, FIGURES = feeds()
RACES = [r["race_id"] for r in PLAIN["venues"][0]["races"]]
NO_JRDB_RACE, NOTHING = RACES[1], (RACES[2], RACES[3])


def _check_fixture() -> None:
    """Marks are neither none nor all, and not simply the horses the member likes most."""
    for p_member, marked in FIGURES.values():
        assert 0 < len(marked) < len(p_member), marked
        top = sorted(p_member, key=lambda h: -p_member[h])[: len(marked)]
        assert set(top) != marked, (top, marked)


_check_fixture()


def pages(tmp_path) -> dict:
    steps = [
        sp.call("renderShadow", PLAIN, SHADOW, {"now": sp.ms(sp.NOW), "venue": "TOK", "race": r})
        for r in RACES
    ]
    shown = sp.results(steps, tmp_path)
    return {r: sp.soup(html) for r, html in zip(RACES, shown, strict=True)}


def win_section(page, race_id: str):
    return sp.pool_block(sp.race_block(page, race_id), "WIN")


def rows_in(section_el) -> list[str]:
    return [r["data-combination"] for r in section_el.select("[data-combination]")]


def percent_ok(shown: str, p: float | None) -> bool:
    if p is None:
        return shown == "-"
    m = re.fullmatch(r"(\d+\.\d)\s*%", shown)
    return m is not None and abs(float(m.group(1)) - p * 100) <= 0.05 + 1e-9


def plain_rows(race_id: str) -> list[dict]:
    race_ = next(r for r in PLAIN["venues"][0]["races"] if r["race_id"] == race_id)
    return race_["pools"]["WIN"]["rows"]


@needs_node
def test_debut_and_maiden_races_show_the_horse_model_and_its_marks_and_others_none(tmp_path):
    shown = pages(tmp_path)
    for race_id, (p_member, marked) in FIGURES.items():
        el = win_section(shown[race_id], race_id)
        combos = rows_in(el)
        assert combos == favourite_order(plain_rows(race_id)), race_id

        heads = [sp.text(th) for th in el.select("thead th")]
        assert HEAD in heads and "Model p" in heads, (race_id, heads)
        first = sp.row_of(el, combos[0])
        positions = first.find_all(["th", "td"], recursive=False)
        at = positions.index(sp.cell(first, "horse-model"))
        assert heads.index(HEAD) == at, (race_id, heads)
        assert abs(at - positions.index(sp.cell(first, "p-model"))) == 1, (race_id, heads)

        for horse in combos:
            row_el = sp.row_of(el, horse)
            text = sp.text(sp.cell(row_el, "horse-model"))
            assert percent_ok(text, p_member.get(horse)), (race_id, horse, text)
            marks = row_el.select("[data-member-mark]")
            assert len(marks) == (1 if horse in marked else 0), (race_id, horse, marked)
        assert len(el.select("[data-member-mark]")) == len(marked), race_id
        assert NO_JRDB not in sp.text(el), race_id

    # every other race on the same page run: no horse-model figure, no status, no mark
    for race_id in NOTHING:
        block = sp.race_block(shown[race_id], race_id)
        assert NO_JRDB not in sp.text(block), race_id
        assert not block.select("[data-member-mark]"), race_id
        for cell in block.select('[data-col="horse-model"]'):
            assert sp.text(cell) in ("", "-"), (race_id, sp.text(cell))
        el = win_section(shown[race_id], race_id)
        assert rows_in(el) == favourite_order(plain_rows(race_id)), race_id
        # Model p is still shown: the market's column is untouched
        p_model = next(r for r in SHADOW["races"] if r["race_id"] == race_id)["p_model"]["WIN"]
        for horse in rows_in(el):
            text = sp.text(sp.cell(sp.row_of(el, horse), "p-model"))
            assert percent_ok(text, p_model[horse]), (race_id, horse, text)


@needs_node
def test_without_jrdb_rows_the_race_says_so(tmp_path):
    shown = pages(tmp_path)
    el = win_section(shown[NO_JRDB_RACE], NO_JRDB_RACE)
    assert NO_JRDB in sp.text(el)
    for cell in el.select('[data-col="horse-model"]'):
        assert not re.search(r"\d\s*%", sp.text(cell)), sp.text(cell)
    assert not el.select("[data-member-mark]")
    assert rows_in(el) == favourite_order(plain_rows(NO_JRDB_RACE))
