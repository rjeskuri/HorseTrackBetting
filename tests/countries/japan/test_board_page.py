"""The board page (tickets #42, #498): `board.html` and `board.js` in
`src/horsetrack/countries/japan/board/web/`, driven from node under a tiny fake DOM.

Contract pinned here (the coder implements against it):

`board.js` runs in the browser and, under node, sets `module.exports` (when `module` exists) to
at least these four functions:

- `formatMove(pct) -> string`: a feed move (`move_morning_pct` / `move_t20_pct`, percent with one
  decimal, or null) as signed percent with one decimal: `12.5 -> '+12.5%'`, `-3 -> '-3.0%'`;
  `0 -> '0.0%'` (a leading '+' or '±' is allowed for zero, never '-'); null -> `'-'`.
- `isStale(lastCaptureUtc, nowMs) -> boolean`: true when the capture (ISO-8601 UTC text, as in the
  feed) is more than 2 minutes (120 s) older than `nowMs` (milliseconds since the epoch, as
  `Date.now()`); exactly 120 s is not stale.
- `renderBoard(feed, {now, venue, race}) -> string`: the HTML of the board for one venue of a feed
  (docs/specs/board-feed.md shape), `venue` defaulting to the feed's first venue, `now` in ms.
  One race per page (#498): it draws the race whose `race_id` is `race` when that race is of the
  venue, else the venue's default: the race to post next (the earliest post still after `now`),
  else the venue's last race. The HTML carries these hooks (any tags, any layout around them):
  - one venue tab per venue of the feed, an element with `data-venue="<venue>"`;
  - the race strip: one link per race of the venue, an `a` outside the race block whose `href`
    is `#<hash>` and whose text starts with `<no>R`; following it selects that race;
  - the selected race, and only it, as an element with `data-race="<race_id>"` that contains its
    pools (a race strip may also use `data-race`, but only the race block contains `data-pool`);
    the race block's text shows the post time in Japan time as `HH:MM` and the word 'stale'
    exactly when `isStale(last_capture_utc, now)`;
  - every pool of the race as an element with `data-pool="<POOL>"` inside the race block, with one
    element per feed row, `data-combination="<combination>"`, favourite first (#498): priced rows
    by current `odds` ascending (the low end of Place/Wide), ties in combination order ('2' before
    '10', '2,3' before '2,10'); then the rows without odds (unpriced, scratched) in combination
    order. A row whose status is 'SCR' has class `scratched`, others do not;
  - in each row, cells with `data-col="odds"`, `data-col="move-morning"` and `data-col="move-t20"`.
    The odds cell of a priced row shows `odds` with one decimal, or `low-high` (an en dash is
    allowed, spaces around it too) when `odds_hi` is not null. A move cell's text is
    `formatMove(pct)`, and its class list has `up` for a positive move, `down` for a negative one,
    neither for zero or null.
- `poll() -> Promise`: one GET of the feed (`/feed` or `feed`) with `If-None-Match` set to the last
  ETag seen; on 200 it sets the innerHTML of the element with id `board` to the board of the
  selected venue (the feed's first venue by default) and the selected race; on 304 the board
  stays. The page polls every 5 s (`setInterval` or `setTimeout` with 5000 ms).
- In the browser (#498): the URL hash selects the race at load (the strip's `#<hash>`); following
  a strip link draws that race alone and leaves its hash in the URL; a hash change (back) selects
  that race; a poll redraws the selected race and never jumps back to the default; a venue tab
  draws that venue's default race.

No word or field of a probability, fair odds, a bet, EV, a ticket or a stake appears in board.html
or board.js (a pure-Python scan; it runs without node).

The feed fixture is built by `feed.build_board` itself from synthetic odds logs under tmp_path, so
its shape and field names are the real ones; the ordering cases are a hand-built synthetic feed.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from horsetrack.core import paths
from horsetrack.countries.japan import odds_log

# isort: split
from page_fakes import (
    BROWSER_FAKES,
    chain_of,
    click_target,
    combination_key,
    favourite_order,
    kept_in_place,
    moved_to_race,
    strip_links,
)

REPO = Path(__file__).resolve().parents[3]
WEB = REPO / "src" / "horsetrack" / "countries" / "japan" / "board" / "web"
PAGE, SCRIPT = WEB / "board.html", WEB / "board.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
DAY = date(2026, 11, 28)


def jst(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(DAY.year, DAY.month, DAY.day, hh, mm, ss, tzinfo=JST)


NOW = jst(10, 1)


def ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


def instant(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


# ---------------------------------------------------------------- the node harness

HARNESS = r"""
const fs = require("fs");
const [, BOARD_JS, OUT] = process.argv;   // node -e: argv[0] is node itself
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const els = {}, docHandlers = {}, timers = [], requests = [];
function makeEl(id) {
  const e = {
    id, innerHTML: "", textContent: "", className: "", hidden: false, value: "", dataset: {},
    style: {}, children: [], attrs: {}, listeners: {},
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    removeAttribute(k) { delete this.attrs[k]; },
    addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
    removeEventListener() {},
    querySelector(sel) { return el("q:" + sel); }, querySelectorAll() { return []; },
    appendChild(c) { this.children.push(c); return c; }, append() {}, prepend() {},
    replaceChildren() { this.innerHTML = ""; }, closest() { return null; },
    scrollIntoView(arg) { noteIntoView(this.id, arg); }, focus() {},
    matches() { return false; },
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
  };
  return e;
}
function el(id) { return (els[id] = els[id] || makeEl(id)); }
global.window = global;
global.document = {
  readyState: "complete", body: el("body"), documentElement: el("html"), title: "",
  getElementById: (id) => el(id),
  querySelector: (sel) => (sel.startsWith("#") ? el(sel.slice(1)) : el("q:" + sel)),
  querySelectorAll: () => [], createElement: () => makeEl(""),
  addEventListener: (t, fn) => { (docHandlers[t] = docHandlers[t] || []).push(fn); },
  removeEventListener() {}, hidden: false, visibilityState: "visible",
};
global.addEventListener = (t, fn) => { (docHandlers[t] = docHandlers[t] || []).push(fn); };
global.removeEventListener = () => {};
/*BROWSER_FAKES*/
global.setInterval = (fn, delay) => { timers.push(Number(delay)); return timers.length; };
global.setTimeout = (fn, delay) => { timers.push(Number(delay)); return timers.length; };
global.clearInterval = () => {}; global.clearTimeout = () => {};
global.requestAnimationFrame = (cb) => setImmediate(() => cb(Date.now()));
let server = input.server || null;
global.fetch = async (url, init = {}) => {
  const h = init.headers || {};
  let inm = null;
  if (typeof h.get === "function") inm = h.get("If-None-Match");
  else for (const k of Object.keys(h)) if (k.toLowerCase() === "if-none-match") inm = h[k];
  requests.push({ url: String(url && url.url ? url.url : url), if_none_match: inm });
  const empty = (status, headers) => ({
    ok: false, status, headers: new Headers(headers), text: async () => "",
    json: async () => { throw new Error("no body with status " + status); },
  });
  if (!server) return empty(503, {});
  if (inm !== null && inm === server.etag) return empty(304, { ETag: server.etag });
  const body = JSON.stringify(server.feed);
  return {
    ok: true, status: 200, json: async () => JSON.parse(body), text: async () => body,
    headers: new Headers({ ETag: server.etag, "Content-Type": "application/json" }),
  };
};
const tick = () => new Promise((r) => setImmediate(r));
const settle = async () => { for (let i = 0; i < 5; i++) await tick(); };
(async () => {
  const board = require(BOARD_JS);
  for (const t of ["DOMContentLoaded", "load"]) {
    for (const fn of docHandlers[t] || []) fn({ type: t });
  }
  await settle();
  const results = [];
  for (const step of input.steps) {
    if (step.call) {
      results.push({ result: board[step.call](...step.args) });
      continue;
    }
    const before = requests.length;
    resetViewport();
    if (step.poll) {
      if (step.server) server = step.server;
      await board.poll();
    } else if (step.click) {
      clickChain(step.click, step.root || "board");
    } else if (typeof step.hash === "string") {
      location.hash = step.hash;
    }
    await settle();
    results.push({ board: el("board").innerHTML, hash: location.hash,
      requests: requests.slice(before), viewport: viewportState() });
  }
  fs.writeFileSync(OUT, JSON.stringify({ results, timers, requests }));
})().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
""".replace("/*BROWSER_FAKES*/", BROWSER_FAKES)


def run_js(
    steps: list[dict],
    tmp_path: Path,
    server: dict | None = None,
    *,
    now: datetime = NOW,
    hash_: str = "",
) -> dict:
    out = tmp_path / "node-out.json"
    proc = subprocess.run(
        [NODE, "-e", HARNESS, str(SCRIPT), str(out)],
        input=json.dumps({"steps": steps, "server": server, "now": ms(now), "hash": hash_}),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def call(fn: str, *args) -> dict:
    return {"call": fn, "args": list(args)}


def results(steps: list[dict], tmp_path: Path) -> list:
    return [r["result"] for r in run_js(steps, tmp_path)["results"]]


# ---------------------------------------------------------------- the synthetic feed

BASE = [2.4, 4.0, 6.5, 9.8, 15.0]  # WIN odds of runners 1-5 at the morning capture
F_EARLY = [1.1, 1.1, 1.1, 1.1, 1.1]  # before 07:00 JST: never the morning reference
F_MORNING = [1.0, 1.0, 1.0, 1.0, 1.0]
F_T20 = [1.1, 0.9, 1.0, 1.2, 0.95]
F_NOW = [1.25, 0.8, 1.0, 1.5, 1.1]


def rid(venue: str, no: int) -> str:
    return f"jra:{DAY.isoformat()}:{venue}:{no:02d}"


def row(race_id, pool, combination, odds, at, *, odds_hi=None, scratched=False) -> dict:
    out = {
        "snapshot_utc": at.astimezone(UTC).isoformat(),
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "odds": None if scratched else odds,
        "status_raw": "SCR" if scratched else f"{at.hour:02d}時{at.minute:02d}分現在",
        "source_update_utc": at.astimezone(UTC).isoformat(),
    }
    if odds_hi is not None and not scratched:
        out["odds_hi"] = odds_hi
    return out


def capture(race_id, at, factors, *, runners=5, scratch=(), pairs=True) -> list[dict]:
    """One capture of a race: WIN, PLACE (ranged), QUINELLA_PLACE (ranged), QUINELLA, and totals."""
    win = {k + 1: round(BASE[k] * factors[k], 1) for k in range(runners)}
    rows = []
    for no, odds in win.items():
        rows.append(row(race_id, "WIN", str(no), odds, at, scratched=no in scratch))
        low = round(max(1.1, odds * 0.35), 1)
        rows.append(
            row(
                race_id,
                "PLACE",
                str(no),
                low,
                at,
                odds_hi=round(low * 1.6, 1),
                scratched=no in scratch,
            )
        )
    if pairs:
        for a, b in combinations(range(1, 5), 2):
            if a in scratch or b in scratch:
                continue  # the source stops listing a scratched runner's pairs
            wide = round(win[a] * win[b] * 0.5, 1)
            rows.append(
                row(race_id, "QUINELLA_PLACE", f"{a},{b}", wide, at, odds_hi=round(wide * 1.5, 1))
            )
            rows.append(row(race_id, "QUINELLA", f"{a},{b}", round(win[a] * win[b] * 1.4, 1), at))
    for i, pool in enumerate(("WIN", "PLACE", "QUINELLA_PLACE", "QUINELLA")):
        total = row(race_id, pool, odds_log.POOL_TOTAL, None, at)
        total["status_raw"] = None
        total["investment"] = 1_000_000 * (i + 1) + at.minute * 1000
        rows.append(total)
    return rows


POSTS = {
    "TOK": {1: jst(10, 5), 2: jst(10, 35), 11: jst(15, 25)},
    "KYO": {3: jst(10, 10)},  # KYO 10 has no post time
}


def day_rows() -> dict[str, list[dict]]:
    tok1, tok2, tok11 = rid("TOK", 1), rid("TOK", 2), rid("TOK", 11)
    kyo3, kyo10 = rid("KYO", 3), rid("KYO", 10)
    early_only = row(tok2, "WIN", "6", 30.0, jst(6, 30))  # no morning row: a null morning move
    return {
        "TOK": [
            *capture(tok1, jst(6, 30), F_EARLY),
            *capture(tok1, jst(8, 0), F_MORNING),
            *capture(tok1, jst(9, 40), F_T20),
            *capture(tok1, jst(10, 0), F_NOW, scratch=(3,)),
            *capture(tok2, jst(6, 30), F_EARLY),
            early_only,
            *capture(tok2, jst(9, 50), F_NOW),
            *capture(tok11, jst(8, 0), F_MORNING),
            *capture(tok11, jst(9, 59), F_NOW),  # exactly 2 minutes old at NOW: not stale
        ],
        "KYO": [
            *capture(kyo3, jst(8, 0), F_MORNING),
            *capture(kyo3, jst(9, 50), F_T20),
            *capture(kyo3, jst(9, 59, 30), F_NOW),
            *capture(kyo10, jst(8, 0), F_MORNING, runners=4, pairs=False),
        ],
    }


def write_day(rows_by_venue: dict[str, list[dict]]) -> None:
    for venue, rows in rows_by_venue.items():
        path = paths.odds_snapshot_path("jra", DAY.isoformat(), venue)
        captures: dict[str, list[dict]] = {}
        for r in rows:
            captures.setdefault(r["snapshot_utc"], []).append(r)
        for cap in captures.values():
            odds_log.append_capture(path, cap)
        if POSTS.get(venue):
            odds_log.write_post_times(
                path.with_name(f"{venue}.races.json"),
                {no: at.astimezone(UTC) for no, at in POSTS[venue].items()},
                jst(6, 0).astimezone(UTC),
            )


@pytest.fixture(scope="module")
def feed(tmp_path_factory) -> dict:
    from horsetrack.countries.japan.board import feed as feed_mod

    data = tmp_path_factory.mktemp("data")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(paths.DATA_DIR_VAR, str(data))
        write_day(day_rows())
        board = feed_mod.build_board(DAY, now=lambda: NOW)
    board = json.loads(json.dumps(board))  # exactly what /feed serves
    _check_fixture(board)
    return board


def all_rows(board: dict):
    for venue in board["venues"]:
        for race in venue["races"]:
            for pool, body in race["pools"].items():
                for line in body["rows"]:
                    yield venue["venue"], race, pool, line


def _check_fixture(board: dict) -> None:
    """The fixture has what the tests need: every sign of move, nulls, ranges and a scratch."""
    assert [v["venue"] for v in board["venues"]] == ["KYO", "TOK"]
    lines = [line for *_, line in all_rows(board)]
    for key in ("move_morning_pct", "move_t20_pct"):
        moves = [line[key] for line in lines]
        assert None in moves and any(m is not None and m > 0 for m in moves), key
        assert any(m is not None and m < 0 for m in moves), key
    assert 0.0 in [line["move_morning_pct"] for line in lines]
    assert any(line["odds_hi"] is not None for line in lines)
    assert any(line["status"] == "SCR" for line in lines)
    assert any(line["move_morning_pct"] is None and line["odds"] is not None for line in lines)
    # the favourite-first order differs from the feed's own order somewhere, with a tie
    assert any(
        favourite_order(body["rows"]) != [r["combination"] for r in body["rows"]]
        for *_, body in _pools(board)
    )
    assert any(
        len({r["odds"] for r in body["rows"] if r["odds"] is not None})
        < len([r for r in body["rows"] if r["odds"] is not None])
        for *_, body in _pools(board)
    )


def _pools(board: dict):
    for venue in board["venues"]:
        for race in venue["races"]:
            for pool, body in race["pools"].items():
                yield venue["venue"], race, pool, body


def races_of(board: dict, venue: str) -> list[dict]:
    return next(v for v in board["venues"] if v["venue"] == venue)["races"]


# ---------------------------------------------------------------- reading the rendered HTML


def soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def text(node) -> str:
    return " ".join(node.get_text(" ").split())


def race_blocks(page: BeautifulSoup) -> list:
    return [e for e in page.select("[data-race]") if e.select("[data-pool]")]


def race_block(page: BeautifulSoup, race_id: str):
    found = [e for e in race_blocks(page) if e["data-race"] == race_id]
    assert len(found) == 1, f"race {race_id}: {len(found)} race blocks with pools"
    return found[0]


def pool_block(block, pool: str):
    found = block.select(f'[data-pool="{pool}"]')
    assert len(found) == 1, f"pool {pool}: {len(found)} blocks"
    return found[0]


def row_of(pool_el, combination: str):
    found = pool_el.select(f'[data-combination="{combination}"]')
    assert len(found) == 1, f"combination {combination}: {len(found)} rows"
    return found[0]


def cell(row_el, col: str):
    found = row_el.select(f'[data-col="{col}"]')
    assert len(found) == 1, f"{col}: {len(found)} cells in {row_el}"
    return found[0]


def classes(node) -> list[str]:
    return list(node.get("class") or [])


def shown_races(page: BeautifulSoup) -> list[str]:
    return [b["data-race"] for b in race_blocks(page)]


def render_all(feed: dict, tmp_path: Path, now: datetime = NOW) -> dict[str, BeautifulSoup]:
    """race_id -> the page of its venue with that race selected."""
    keys = [(v["venue"], r["race_id"]) for v in feed["venues"] for r in v["races"]]
    steps = [call("renderBoard", feed, {"now": ms(now), "venue": v, "race": r}) for v, r in keys]
    shown = results(steps, tmp_path)
    return {r: soup(html) for (_, r), html in zip(keys, shown, strict=True)}


def move_text_ok(shown: str, pct) -> bool:
    shown = shown.strip()
    if pct is None:
        return shown == "-"
    if pct == 0:
        return shown in {"0.0%", "+0.0%", "±0.0%"}
    return shown == f"{'+' if pct > 0 else ''}{pct:.1f}%"


# ---------------------------------------------------------------- forbidden words

FORBIDDEN = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"probab",
        r"\bprob\b",
        r"\bp_[a-z]",
        r"\bfair",
        r"\bbets?\b",
        r"\bbetting\b",
        r"expected[\s_-]*value",
        r"\bstak(e|es|ing)\b",
        r"\bticket",
        r"\bwager",
        r"\bkelly\b",
        r"確率",
        r"期待値",
        r"馬券",
        r"買い目",
        r"推奨",
    )
] + [re.compile(r"\bEV\b")]


def forbidden_in(source: str) -> list[str]:
    return [m.group(0) for p in FORBIDDEN for m in p.finditer(source)]


# ---------------------------------------------------------------- tests


@needs_node
def test_renders_every_race_and_pool_of_a_feed(feed, tmp_path):
    pages = render_all(feed, tmp_path)
    venues = {v["venue"] for v in feed["venues"]}
    for venue in feed["venues"]:
        for race in venue["races"]:
            page = pages[race["race_id"]]
            tabs = {e["data-venue"] for e in page.select("[data-venue]")}
            assert tabs == venues, f"venue tabs on {race['race_id']}: {tabs}"
            assert not forbidden_in(text(page)), forbidden_in(text(page))
            # one race per page: no other race's block, pools or rows
            assert shown_races(page) == [race["race_id"]], shown_races(page)
            block = race_block(page, race["race_id"])
            shown = {e["data-pool"] for e in block.select("[data-pool]")}
            assert shown == set(race["pools"]), f"{race['race_id']} pools {shown}"
            assert len(page.select("[data-pool]")) == len(race["pools"])
            n_rows = sum(len(body["rows"]) for body in race["pools"].values())
            assert len(page.select("[data-combination]")) == n_rows, race["race_id"]
            if race["post_utc"] is not None:
                post = instant(race["post_utc"])
                assert post.astimezone(JST).strftime("%H:%M") in text(block), race["race_id"]
                assert post.strftime("%H:%M") not in text(block), "post time shown in UTC"
            for pool, body in race["pools"].items():
                pool_el = pool_block(block, pool)
                combos = [e["data-combination"] for e in pool_el.select("[data-combination]")]
                assert combos == favourite_order(body["rows"]), (race["race_id"], pool)
                for line in body["rows"]:
                    row_el = row_of(pool_el, line["combination"])
                    assert ("scratched" in classes(row_el)) == (line["status"] == "SCR")
                    if line["odds"] is not None and line["odds_hi"] is None:
                        assert text(cell(row_el, "odds")) == f"{line['odds']:.1f}", line


# A hand-built feed for the ordering rule: rows in the feed's numeric order, odds not; ties across
# '2'/'10' and '2,3'/'2,10' (numeric and text order disagree); morning odds in another order.
def _hand(combination, odds, hi=None, *, scr=False, morning=None) -> dict:
    return {
        "combination": combination,
        "odds": None if scr else odds,
        "odds_hi": None if scr else hi,
        "status": "SCR" if scr else "10時00分現在",
        "morning_odds": morning,
        "move_morning_pct": None,
        "t20_odds": None,
        "move_t20_pct": None,
        "t2_odds": None,
        "move_t20_t2_pct": None,
    }


ORDER_POOLS = {
    "WIN": [
        _hand("1", 8.5, morning=1.5),
        _hand("2", 3.1, morning=20.0),
        _hand("3", None, scr=True, morning=2.0),
        _hand("4", 3.1, morning=4.0),
        _hand("5", None, morning=3.0),  # listed, not priced
        _hand("6", 1.9, morning=50.0),
        _hand("10", 3.1, morning=6.0),
        _hand("11", 45.0, morning=1.2),
        _hand("12", None, scr=True),
    ],
    "PLACE": [
        _hand("1", 1.5, 5.0),
        _hand("2", 1.6, 1.9),
        _hand("3", None, scr=True),
        _hand("4", 1.1, 1.3),
        _hand("10", 1.5, 2.0),
        _hand("11", 1.1, 1.2),
    ],
    "QUINELLA": [
        _hand("1,2", 15.0),
        _hand("1,4", 6.2),
        _hand("2,4", 6.2),
        _hand("4,10", 2.5),
    ],
    "QUINELLA_PLACE": [
        _hand("1,2", 9.0, 12.0),
        _hand("1,10", None),
        _hand("2,3", 4.4, 6.0),
        _hand("2,4", 30.0, 41.0),
        _hand("2,10", 4.4, 5.0),
    ],
}
ORDER_WANT = {  # derived by hand from the rule
    "WIN": ["6", "2", "4", "10", "1", "11", "3", "5", "12"],
    "PLACE": ["4", "11", "1", "10", "2", "3"],
    "QUINELLA": ["4,10", "1,4", "2,4", "1,2"],
    "QUINELLA_PLACE": ["2,3", "2,10", "1,2", "2,4", "1,10"],
}


def order_feed() -> dict:
    race = {
        "race_no": 5,
        "race_id": rid("TOK", 5),
        "post_utc": jst(11, 0).astimezone(UTC).isoformat(),
        "last_capture_utc": NOW.astimezone(UTC).isoformat(),
        "pools": {p: {"investment": 1_000_000, "rows": rows} for p, rows in ORDER_POOLS.items()},
    }
    return {
        "race_date": DAY.isoformat(),
        "generated_utc": NOW.astimezone(UTC).isoformat(),
        "venues": [{"venue": "TOK", "races": [race]}],
    }


@needs_node
def test_pool_rows_run_favourite_to_longest_with_unpriced_last(tmp_path):
    for pool, rows in ORDER_POOLS.items():
        assert [r["combination"] for r in rows] == sorted(
            (r["combination"] for r in rows), key=combination_key
        ), "the hand feed keeps the feed's numeric order"
        assert favourite_order(rows) == ORDER_WANT[pool], pool
    feed = order_feed()
    shown = results([call("renderBoard", feed, {"now": ms(NOW)})], tmp_path)[0]
    block = race_block(soup(shown), rid("TOK", 5))
    for pool, want in ORDER_WANT.items():
        pool_el = pool_block(block, pool)
        combos = [e["data-combination"] for e in pool_el.select("[data-combination]")]
        assert combos == want, pool
        rows = pool_el.select("[data-combination]")
        assert [("scratched" in classes(r)) for r in rows] == [
            next(x for x in ORDER_POOLS[pool] if x["combination"] == c)["status"] == "SCR"
            for c in want
        ], pool
    # the page sorts; the feed it was given is untouched
    assert feed == order_feed()


@needs_node
def test_one_race_per_page_the_next_to_post_by_default(feed, tmp_path):
    kyo3, kyo10 = rid("KYO", 3), rid("KYO", 10)
    tok1, tok2, tok11 = rid("TOK", 1), rid("TOK", 2), rid("TOK", 11)
    # (venue, now, race option, the race drawn)
    cases = [
        ("TOK", jst(9, 0), None, tok1),
        ("TOK", NOW, None, tok1),
        ("TOK", jst(10, 5), None, tok2),  # at its post a race is off: the next one
        ("TOK", jst(10, 6), None, tok2),
        ("TOK", jst(10, 40), None, tok11),
        ("TOK", jst(16, 0), None, tok11),  # every post passed: the last race
        ("KYO", NOW, None, kyo3),
        ("KYO", jst(10, 20), None, kyo10),  # KYO 3 is off, KYO 10 has no post: the last race
        (None, jst(10, 20), None, kyo10),  # the first venue by default
        ("TOK", NOW, tok11, tok11),
        ("TOK", NOW, tok2, tok2),
        ("TOK", jst(16, 0), tok1, tok1),
        ("KYO", NOW, kyo10, kyo10),
        ("TOK", NOW, kyo3, tok1),  # another venue's race: this venue's default
        ("KYO", jst(10, 20), rid("KYO", 99), kyo10),  # no such race: the default
    ]
    steps = []
    for venue, now, race, _ in cases:
        opts = {"now": ms(now)}
        if venue is not None:
            opts["venue"] = venue
        if race is not None:
            opts["race"] = race
        steps.append(call("renderBoard", feed, opts))
    for (venue, now, race, want), html in zip(cases, results(steps, tmp_path), strict=True):
        page = soup(html)
        where = (venue, now.strftime("%H:%M"), race)
        assert shown_races(page) == [want], (where, shown_races(page))
        venue_races = races_of(feed, venue or feed["venues"][0]["venue"])
        drawn = next(r for r in venue_races if r["race_id"] == want)
        assert len(page.select("[data-pool]")) == len(drawn["pools"]), where
        # the strip still lists every race of the venue, each with its own hash
        links = strip_links(page, race_blocks(page))
        assert set(links) == {r["race_no"] for r in venue_races}, (where, sorted(links))
        assert len({a["href"] for a in links.values()}) == len(links), where


@needs_node
def test_the_hash_and_the_strip_pick_the_race_and_a_poll_keeps_it(feed, tmp_path):
    kyo3, kyo10, tok1 = rid("KYO", 3), rid("KYO", 10), rid("TOK", 1)
    first = soup(results([call("renderBoard", feed, {"now": ms(NOW)})], tmp_path)[0])
    assert shown_races(first) == [kyo3], "KYO 3 is the next race to post at NOW"
    links = strip_links(first, race_blocks(first))
    assert set(links) == {3, 10}
    to3, to10 = links[3]["href"], links[10]["href"]
    tok_tab = first.select_one('[data-venue="TOK"]')
    assert tok_tab is not None

    newer = copy.deepcopy(feed)
    for venue in newer["venues"]:
        for race in venue["races"]:
            for line in race["pools"]["WIN"]["rows"]:
                if line["odds"] is not None:
                    line["odds"] = round(line["odds"] + 0.3, 1)
    a, b = {"feed": feed, "etag": '"a1"'}, {"feed": newer, "etag": '"b2"'}
    steps = [
        {"poll": True},  # loaded with KYO 10's hash
        {"poll": True, "server": b},  # a new feed
        {"click": chain_of(click_target(links[3]), "board")},  # 3R on the strip
        {"poll": True},  # 304
        {"click": chain_of(click_target(links[10]), "board")},  # 10R
        {"poll": True, "server": a},
        {"hash": to3},  # back to KYO 3's hash
        {"poll": True},
        {"click": chain_of(tok_tab, "board")},  # the TOK tab
        {"poll": True},
    ]
    out = run_js(steps, tmp_path, server=a, hash_=to10)
    want = [kyo10, kyo10, kyo3, kyo3, kyo10, kyo10, kyo3, kyo3, tok1, tok1]
    got = [shown_races(soup(r["board"])) for r in out["results"]]
    assert got == [[w] for w in want], got
    hashes = [r["hash"] for r in out["results"]]
    assert hashes[2] == to3 and hashes[4] == to10, hashes  # the selection is in the URL
    # the new feed's odds are drawn for the kept race
    kyo10_win = race_block(soup(out["results"][1]["board"]), kyo10)
    new_rows = races_of(newer, "KYO")[1]["pools"]["WIN"]["rows"]
    for line in new_rows:
        if line["odds"] is not None:
            shown = text(cell(row_of(pool_block(kyo10_win, "WIN"), line["combination"]), "odds"))
            assert shown == f"{line['odds']:.1f}", line


@needs_node
def test_a_newly_selected_race_moves_the_viewport_and_a_poll_leaves_it(feed, tmp_path):
    # The reader is page_fakes.READ_Y px down (the bottom of a race's tables) before every step.
    # Picking another race (strip link, back) brings its race to the top of the viewport; a
    # poll, with or without a new feed, leaves the viewport where the reader put it.
    kyo3, kyo10, tok1 = rid("KYO", 3), rid("KYO", 10), rid("TOK", 1)
    first = soup(results([call("renderBoard", feed, {"now": ms(NOW)})], tmp_path)[0])
    links = strip_links(first, race_blocks(first))
    to10 = links[10]["href"]
    tok_tab = first.select_one('[data-venue="TOK"]')
    newer = copy.deepcopy(feed)
    for venue in newer["venues"]:
        for race in venue["races"]:
            for line in race["pools"]["WIN"]["rows"]:
                if line["odds"] is not None:
                    line["odds"] = round(line["odds"] + 0.2, 1)
    a, b = {"feed": feed, "etag": '"a1"'}, {"feed": newer, "etag": '"b2"'}
    # (step, race drawn after it, the viewport moves to the race)
    table = [
        ({"poll": True}, kyo10, False),  # 304
        ({"poll": True, "server": b}, kyo10, False),  # a new feed, same race
        ({"click": chain_of(click_target(links[3]), "board")}, kyo3, True),  # 3R on the strip
        ({"poll": True, "server": a}, kyo3, False),
        ({"hash": to10}, kyo10, True),  # back
        ({"poll": True}, kyo10, False),
        ({"click": chain_of(tok_tab, "board")}, tok1, True),  # the TOK tab
        ({"poll": True, "server": b}, tok1, False),
    ]
    out = run_js([step for step, _, _ in table], tmp_path, server=a, hash_=to10)
    got = [shown_races(soup(r["board"])) for r in out["results"]]
    assert got == [[race] for _, race, _ in table], got
    for i, ((step, race, moves), result) in enumerate(zip(table, out["results"], strict=True)):
        where = (i, sorted(step)[0], race, result["viewport"])
        if moves:
            assert moved_to_race(result["viewport"]), ("left deep in the new race", where)
        else:
            assert kept_in_place(result["viewport"]), ("a poll moved the reader", where)


@needs_node
def test_renders_without_a_venue_option_and_for_an_empty_day(feed, tmp_path):
    empty = {"race_date": DAY.isoformat(), "generated_utc": NOW.isoformat(), "venues": []}
    default, blank = results(
        [call("renderBoard", feed, {"now": ms(NOW)}), call("renderBoard", empty, {"now": ms(NOW)})],
        tmp_path,
    )
    assert shown_races(soup(default)) == [rid("KYO", 3)]  # the first venue's next race
    assert not soup(blank).select("[data-pool]")


@needs_node
def test_redraws_on_a_new_feed_and_sends_the_etag(feed, tmp_path):
    first = feed["venues"][0]
    second = copy.deepcopy(feed)
    race = copy.deepcopy(first["races"][0])
    race.update(race_no=7, race_id=rid(first["venue"], 7))
    race["pools"] = {"WIN": copy.deepcopy(race["pools"]["WIN"])}
    for i, line in enumerate(race["pools"]["WIN"]["rows"]):
        if line["odds"] is not None:
            line["odds"] = 70.0 + i + 0.7
    second["venues"] = [{"venue": first["venue"], "races": [race]}]
    a, b = {"feed": feed, "etag": '"a1"'}, {"feed": second, "etag": '"b2"'}
    out = run_js(
        [{"poll": True, "server": a}, {"poll": True}, {"poll": True, "server": b}], tmp_path
    )
    shown_a, again, shown_b = out["results"]

    default = first["races"][0]["race_id"]  # KYO 3, the next race to post at NOW
    assert shown_races(soup(shown_a["board"])) == [default]
    assert shown_races(soup(again["board"])) == [default]
    assert again["requests"] and again["requests"][-1]["if_none_match"] == '"a1"'
    page_b = soup(shown_b["board"])
    win = pool_block(race_block(page_b, race["race_id"]), "WIN")
    for line in race["pools"]["WIN"]["rows"]:
        if line["odds"] is not None:
            assert text(cell(row_of(win, line["combination"]), "odds")) == f"{line['odds']:.1f}"
    for r in first["races"]:
        assert f'data-race="{r["race_id"]}"' not in shown_b["board"], "the old feed is still drawn"
    for req in out["requests"]:
        assert re.search(r"(^|/)feed(\?|$)", req["url"]), req["url"]
    assert 5000 in out["timers"], f"no 5 s poll timer: {out['timers']}"


@needs_node
def test_moves_show_sign_and_percent(feed, tmp_path):
    table = [12.5, -3.0, 150.0, -0.4, 7.0, -66.7, 0.1, 0.0, -100.0, 1234.5]
    shown = results([call("formatMove", pct) for pct in table], tmp_path)
    for pct, text_ in zip(table, shown, strict=True):
        assert move_text_ok(text_, pct), f"formatMove({pct}) = {text_!r}"

    pages = render_all(feed, tmp_path)
    for _, race, pool, line in all_rows(feed):
        row_el = row_of(
            pool_block(race_block(pages[race["race_id"]], race["race_id"]), pool),
            line["combination"],
        )
        for col, key in (("move-morning", "move_morning_pct"), ("move-t20", "move_t20_pct")):
            pct, el = line[key], cell(row_el, col)
            assert move_text_ok(text(el), pct), (
                f"{race['race_id']} {pool} {line['combination']} {col}"
            )
            assert ("up" in classes(el)) == (pct is not None and pct > 0), (col, pct)
            assert ("down" in classes(el)) == (pct is not None and pct < 0), (col, pct)


@needs_node
def test_null_moves_show_a_dash(feed, tmp_path):
    assert results([call("formatMove", None)], tmp_path) == ["-"]
    pages = render_all(feed, tmp_path)
    nulls = 0
    for _, race, pool, line in all_rows(feed):
        row_el = row_of(
            pool_block(race_block(pages[race["race_id"]], race["race_id"]), pool),
            line["combination"],
        )
        for col, key in (("move-morning", "move_morning_pct"), ("move-t20", "move_t20_pct")):
            if line[key] is None:
                nulls += 1
                el = cell(row_el, col)
                assert text(el) == "-", f"{race['race_id']} {pool} {line['combination']} {col}"
                assert "up" not in classes(el) and "down" not in classes(el)
    assert nulls > 10


@needs_node
def test_a_capture_older_than_two_minutes_is_flagged_stale(feed, tmp_path):
    capture = jst(9, 30)
    ages = [-30, 0, 60, 119, 120, 121, 180, 3600]
    flags = results(
        [
            call("isStale", capture.astimezone(UTC).isoformat(), ms(capture) + s * 1000)
            for s in ages
        ],
        tmp_path,
    )
    assert flags == [s > 120 for s in ages], dict(zip(ages, flags, strict=True))

    for now in (NOW, jst(10, 2), NOW + timedelta(minutes=10)):
        pages = render_all(feed, tmp_path, now=now)
        for venue in feed["venues"]:
            for race in venue["races"]:
                age = (now - instant(race["last_capture_utc"])).total_seconds()
                block = race_block(pages[race["race_id"]], race["race_id"])
                said = "stale" in text(block).lower()
                assert said == (age > 120), f"{race['race_id']} at age {age:.0f} s, now {now}"


@needs_node
def test_ranged_pools_show_low_and_high(feed, tmp_path):
    pages = render_all(feed, tmp_path)
    ranged = 0
    for _, race, pool, line in all_rows(feed):
        if line["odds"] is None:
            continue
        row_el = row_of(
            pool_block(race_block(pages[race["race_id"]], race["race_id"]), pool),
            line["combination"],
        )
        shown = text(cell(row_el, "odds"))
        if line["odds_hi"] is None:
            assert shown == f"{line['odds']:.1f}", (pool, line)
        else:
            ranged += 1
            low, high = f"{line['odds']:.1f}", f"{line['odds_hi']:.1f}"
            want = rf"{re.escape(low)}\s*[-–]\s*{re.escape(high)}"
            assert re.fullmatch(want, shown), f"{pool} {line['combination']}: {shown!r}"
    assert ranged > 20


def test_page_has_no_bet_or_probability_words():
    bad = ["Bet if 3.2 or more", "p_fundamental", "EV +4%", "Fair odds 2.5", "stake 100",
           "ticket", "Win probability 30%", "期待値", "bets"]  # fmt: skip
    fine = ["Between races", "Place odds 1.2-1.8", "event", "Better than morning", "Wide"]
    assert all(forbidden_in(s) for s in bad), [s for s in bad if not forbidden_in(s)]
    assert not any(forbidden_in(s) for s in fine), [s for s in fine if forbidden_in(s)]

    for path in (PAGE, SCRIPT):
        assert path.is_file(), f"{path.relative_to(REPO)} is missing"
        source = path.read_text(encoding="utf-8")
        assert source.strip(), f"{path.name} is empty"
        assert not forbidden_in(source), f"{path.name}: {sorted(set(forbidden_in(source)))}"
    assert "renderBoard" in SCRIPT.read_text(encoding="utf-8")
