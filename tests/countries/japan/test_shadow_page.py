"""The shadow page (ticket #122): `shadow.html` and `shadow.js` in
`src/horsetrack/countries/japan/board/web/`, the live board (#42's board.js) plus each row's shadow
value, bucket label and avoid-flag chips, driven from node under a tiny fake DOM as in
test_board_page.py.

Contract pinned here (the coder implements against it):

`shadow.js` runs in the browser after `board.js` (shadow.html loads `board.js` first, then
`shadow.js`, as plain `<script src>` tags) and reuses board.js's render functions. Under node it
sets `module.exports` (when `module` exists) to at least:

- `renderShadow(feed, shadowFeed, {now, venue}) -> string`: the HTML for one venue (the plain
  feed's first venue by default; `now` in ms since the epoch). `feed` is the plain feed
  (docs/specs/board-feed.md, each row amended with `t2_odds` and `move_t2_pct`, which may be
  absent); `shadowFeed` is #121's shadow feed or null. Rows join on (race_id, pool, combination).
  The HTML carries every hook of board.js's `renderBoard` (venue tabs `data-venue`, race blocks
  `data-race` containing `data-pool` blocks, rows `data-combination`, cells `data-col="odds"`,
  `"move-morning"`, `"move-t20"`), in the same order as `renderBoard` draws them for the same
  plain feed: no sorting, highlighting or colouring by value. Rows come from the plain feed only:
  a shadow entry without a plain row draws nothing. Each row adds these cells:
  - `data-col="move-t2"`: board.js's `formatMove(move_t2_pct)` with class `up` / `down` as the other
    move cells; '-' when the figure is null or absent.
  - `data-col="value"`: for WIN, `value_mean` then the band `value_q05` and `value_q95`, each with
    two decimals, in that order (e.g. `0.81 (0.74–0.88)`; no other decimal number in the cell).
    For every other pool, `value_at_low` and `value_at_high` with two decimals as `low–high`
    (hyphen or en dash, spaces allowed). A null figure shows '-'; a row with no figure at all (all
    null, no shadow entry for the combination, the pool or the race, or no shadow feed) shows
    exactly `-`. The value cells of one pool that show figures carry the same class list, and no
    value cell or row has an inline style.
  - `data-col="bucket"`: `@T-<tau_bucket_min>` of the shadow race (e.g. `@T-5`); `-` when it is
    null or the race has no shadow entry.
  - `data-col="flags"`: one chip per flag of the shadow entry, an element with
    `data-flag="<flag>"` (the feed's code: floor, tail, tail_member, stale, scratched) whose text
    is the plain-English label (the code with '_' as a space, e.g. 'tail member'); no chip when
    there are no flags or no shadow entry.
- `poll() -> Promise`: one GET of `feed` and one of `shadow-feed` (relative or rooted), each with
  `If-None-Match` set to that URL's last ETag; a 200 replaces that feed, a 304 keeps it; while no
  shadow feed has been served (e.g. 503) the board still draws with every value '-'. It then sets
  the innerHTML of the element with id `shadow` to
  `renderShadow(feed, shadowFeed, {now: Date.now(), venue})`. The page starts polling at load and
  polls again every 5 s (`setTimeout`/`setInterval` with 5000 ms).

In the browser there is no module system: shadow.js must find board.js's functions some other way
(board.js may expose them, e.g. on `window`, as long as test_board_page.py stays green). One test
loads board.js then shadow.js as plain scripts (no `module`, no `require`), fires the 5 s timers
and checks that #shadow fills and both feeds are re-requested with their ETags.

shadow.html carries a fixed banner, the element with id `banner`, saying these are shadow values
of the v0 models, gross return per ¥1 at the final price, no real money, and that nothing on the
page is a bet; and a flag legend, the element with id `flag-legend` (in shadow.html or drawn by
shadow.js), with one `data-flag` element per flag and text saying they are candidate avoid rules,
not adopted. No stake words (stake, Kelly, wager, bet size, bet amount, fair odds, bet if) appear
in shadow.html, shadow.js, board.js or the rendered page, and no `<input>` or `<textarea>` does:
the banner says 'no real money' rather than 'no stakes' for that reason.

Both feeds are synthetic objects built here, of the shapes in the ticket.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parents[3]
WEB = REPO / "src" / "horsetrack" / "countries" / "japan" / "board" / "web"
PAGE, SCRIPT, BOARD_JS = WEB / "shadow.html", WEB / "shadow.js", WEB / "board.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
DAY = "2026-11-28"
FLAGS = ("floor", "tail", "tail_member", "stale", "scratched")


def jst(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(2026, 11, 28, hh, mm, ss, tzinfo=JST)


def utc_text(at: datetime) -> str:
    return at.astimezone(UTC).isoformat()


NOW = jst(10, 1)


def ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


# ---------------------------------------------------------------- the node harness

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const [, BOARD_JS, SHADOW_JS, OUT] = process.argv;   // node -e: argv[0] is node itself
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const els = {}, docHandlers = {}, timers = [], requests = [];
function makeEl(id) {
  return {
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
    scrollIntoView() {}, focus() {}, matches() { return false; },
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
  };
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
global.setInterval = (fn, delay) => { timers.push({ fn, delay: Number(delay), keep: true });
  return timers.length; };
global.setTimeout = (fn, delay) => { timers.push({ fn, delay: Number(delay), keep: false });
  return timers.length; };
global.clearInterval = () => {}; global.clearTimeout = () => {};
global.requestAnimationFrame = (cb) => setImmediate(() => cb(Date.now()));
global.scrollTo = () => {};
let servers = input.servers || {};
global.fetch = async (url, init = {}) => {
  const u = String(url && url.url ? url.url : url);
  const h = init.headers || {};
  let inm = null;
  if (typeof h.get === "function") inm = h.get("If-None-Match");
  else for (const k of Object.keys(h)) if (k.toLowerCase() === "if-none-match") inm = h[k];
  requests.push({ url: u, if_none_match: inm });
  const empty = (status, headers) => ({
    ok: false, status, headers: new Headers(headers), text: async () => "",
    json: async () => { throw new Error("no body with status " + status); },
  });
  const key = /(^|\/)shadow-feed(\?|$)/.test(u) ? "shadow" : /(^|\/)feed(\?|$)/.test(u) ? "plain"
    : null;
  if (key === null) return empty(404, {});
  const s = servers[key];
  if (!s) return empty(503, {});
  if (inm !== null && inm === s.etag) return empty(304, { ETag: s.etag });
  const body = JSON.stringify(s.feed);
  return {
    ok: true, status: 200, json: async () => JSON.parse(body), text: async () => body,
    headers: new Headers({ ETag: s.etag, "Content-Type": "application/json" }),
  };
};
const tick = () => new Promise((r) => setImmediate(r));
const settle = async () => { for (let i = 0; i < 30; i++) await tick(); };
(async () => {
  const mods = {};
  if (input.browser) {
    // As the page loads them: plain scripts in order, no module system.
    global.module = undefined; global.require = undefined; global.exports = undefined;
    for (const f of [BOARD_JS, SHADOW_JS]) {
      vm.runInThisContext(fs.readFileSync(f, "utf8"), { filename: f });
    }
  } else {
    mods.shadow = require(SHADOW_JS);
    mods.board = require(BOARD_JS);
  }
  for (const t of ["DOMContentLoaded", "load"]) {
    for (const fn of docHandlers[t] || []) fn({ type: t });
  }
  await settle();
  const results = [{ shadow: el("shadow").innerHTML, requests: requests.slice() }];
  for (const step of input.steps) {
    if (step.servers) servers = step.servers;
    const before = requests.length;
    if (step.call) {
      results.push({ result: mods[step.mod || "shadow"][step.call](...step.args) });
      continue;
    }
    if (step.poll) {
      await mods.shadow.poll();
    } else if (step.fire) {
      const due = timers.filter((t) => t.delay === step.fire && !t.done);
      for (const t of due) { if (!t.keep) t.done = true; t.fn(); }
    }
    await settle();
    results.push({ shadow: el("shadow").innerHTML, requests: requests.slice(before) });
  }
  fs.writeFileSync(OUT, JSON.stringify({ results, timers: timers.map((t) => t.delay) }));
})().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
"""


def run_js(
    steps: list[dict], tmp_path: Path, *, servers: dict | None = None, browser: bool = False
) -> dict:
    for path in (SCRIPT, BOARD_JS):
        assert path.is_file(), f"{path.relative_to(REPO)} is missing"
    out = tmp_path / "node-out.json"
    proc = subprocess.run(
        [NODE, "-e", HARNESS, str(BOARD_JS), str(SCRIPT), str(out)],
        input=json.dumps({"steps": steps, "servers": servers or {}, "browser": browser}),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def call(fn: str, *args, mod: str = "shadow") -> dict:
    return {"call": fn, "args": list(args), "mod": mod}


def results(steps: list[dict], tmp_path: Path) -> list:
    return [r["result"] for r in run_js(steps, tmp_path)["results"][1:]]


# ---------------------------------------------------------------- the synthetic feeds


def rid(venue: str, no: int) -> str:
    return f"jra:{DAY}:{venue}:{no:02d}"


def line(combo, odds, *, hi=None, mm=None, t20=None, t2=None, scr=False, no_t2=False) -> dict:
    """One plain-feed row (board-feed.md), with the #40 amendment's T-2 fields unless no_t2."""
    out = {
        "combination": combo,
        "odds": None if scr else odds,
        "odds_hi": None if scr else hi,
        "status": "SCR" if scr else "10時00分現在",
        "morning_odds": odds,
        "move_morning_pct": None if scr else mm,
        "t20_odds": odds,
        "move_t20_pct": None if scr else t20,
    }
    if not no_t2:
        out["t2_odds"] = None if scr else odds
        out["move_t2_pct"] = None if scr else t2
    return out


def pool(*rows, investment=1_234_000) -> dict:
    return {"investment": investment, "rows": list(rows)}


def race(venue, no, post, last, pools) -> dict:
    return {
        "race_no": no,
        "race_id": rid(venue, no),
        "post_utc": None if post is None else utc_text(post),
        "last_capture_utc": utc_text(last),
        "pools": dict(sorted(pools.items())),  # the feed sorts pools by name
    }


def plain_feed() -> dict:
    kyo3 = race(
        "KYO",
        3,
        jst(10, 10),
        jst(9, 59, 30),
        {
            # rows deliberately not in numeric, string or value order
            "WIN": pool(
                line("2", 3.1, mm=-4.2, t20=1.5, t2=-2.0),
                line("10", 12.0, mm=8.0, t20=-3.3, t2=4.4),
                line("1", 1.8, mm=0.0, t20=None, t2=0.0),
                line("5", 25.5, mm=None, t20=2.0, t2=None),
            ),
            "PLACE": pool(
                line("2", 1.2, hi=1.6, mm=1.0, t20=-1.0, t2=3.3),
                line("10", 2.8, hi=4.1, mm=-2.5, t20=0.0, t2=-7.1),
                line("1", 1.1, hi=1.3, mm=None, t20=None, t2=12.5),
                line("5", 5.0, hi=7.9, mm=6.0, t20=5.5, t2=-0.4),
            ),
            "QUINELLA_PLACE": pool(
                line("2,10", 9.9, hi=14.0, mm=-10.0, t20=2.2, t2=1.1),
                line("1,2", 2.4, hi=3.0, mm=3.0, t20=-6.6, t2=None),
                line("1,10", 7.5, hi=9.1, mm=None, t20=1.0, t2=-5.0),
            ),
        },
    )
    kyo10 = race(
        "KYO",
        10,
        None,
        jst(8, 0),
        {
            "WIN": pool(
                *(line(str(n), 2.0 + n, mm=0.0, no_t2=True) for n in (1, 2, 3, 4)),
                investment=None,
            )
        },
    )
    tok1 = race(
        "TOK",
        1,
        jst(10, 5),
        jst(10, 0),
        {
            "WIN": pool(
                line("1", 2.9, mm=-12.5, t20=-5.0, t2=-1.2),
                line("2", 3.6, mm=4.0, t20=2.5, t2=6.6),
                line("3", None, scr=True),
                line("4", 18.0, mm=20.0, t20=9.0, t2=150.0),
            ),
            "PLACE": pool(
                line("1", 1.3, hi=1.7, mm=-3.0, t20=-1.0, t2=-2.2),
                line("2", 1.5, hi=2.2, mm=2.0, t20=1.0, t2=0.0),
                line("3", None, scr=True),
                line("4", 3.9, hi=6.0, mm=10.0, t20=None, t2=-66.7),
            ),
            "QUINELLA": pool(
                line("1,2", 8.8, mm=1.0, t20=1.0, t2=1.0),
                line("1,4", 40.1, mm=-2.0, t20=-2.0, t2=-2.0),
                line("2,4", 55.0, mm=3.0, t20=3.0, t2=3.0),
            ),
            "QUINELLA_PLACE": pool(
                line("1,2", 3.3, hi=4.4, mm=2.1, t20=0.5, t2=-3.4),
                line("1,4", 12.2, hi=15.5, mm=-1.9, t20=-0.5, t2=8.8),
                line("2,4", 14.4, hi=19.0, mm=0.0, t20=0.0, t2=None),
                line("1,3", None, scr=True),
            ),
        },
    )
    tok11 = race(
        "TOK",
        11,
        jst(15, 25),
        jst(9, 59),
        {
            "WIN": pool(line("1", 4.2, mm=1.1, t2=None), line("2", 6.3, mm=-1.1, t2=None)),
            "PLACE": pool(
                line("1", 1.6, hi=2.1, mm=0.5, t2=None), line("2", 2.0, hi=2.9, mm=-0.5, t2=None)
            ),
        },
    )
    return {
        "race_date": DAY,
        "generated_utc": utc_text(NOW),
        "venues": [
            {"venue": "KYO", "races": [kyo3, kyo10]},
            {"venue": "TOK", "races": [tok1, tok11]},
        ],
    }


def win(mean, q05, q95, *flags) -> dict:
    return {"value_mean": mean, "value_q05": q05, "value_q95": q95, "flags": list(flags)}


def rng(low, high, *flags) -> dict:
    return {"value_at_low": low, "value_at_high": high, "flags": list(flags)}


def shadow_race(venue, no, post, tau, pools) -> dict:
    return {
        "race_id": rid(venue, no),
        "venue": venue,
        "race_no": no,
        "post_utc": utc_text(post),
        "minutes_to_post": (post - NOW).total_seconds() / 60,
        "tau_bucket_min": tau,
        "board_capture_utc": utc_text(NOW - timedelta(seconds=20)),
        "pools": pools,
    }


def shadow_feed() -> dict:
    """Values per (race, pool, combination); races and keys in an order unlike the board's."""
    return {
        "race_date": DAY,
        "generated_utc": utc_text(NOW),
        "models": {"WIN": "v0-synthetic", "PLACE": "v0-synthetic", "QUINELLA_PLACE": "v0-synth"},
        "races": [
            shadow_race(
                "TOK",
                11,
                jst(15, 25),
                20,
                {
                    "PLACE": {"2": rng(0.6, 0.72), "1": rng(0.75, 0.9)},
                    "WIN": {"2": win(0.83, None, None), "1": win(0.97, 0.9, 1.04)},
                },
            ),
            shadow_race(
                "KYO",
                3,
                jst(10, 10),
                10,
                {
                    "WIN": {
                        "1": win(1.5, 1.2, 1.83, "floor", "stale"),
                        "5": win(None, None, None, "tail"),
                        "2": win(0.92, 0.85, 0.99),
                        "10": win(0.6, 0.41, 0.77, "tail"),
                    },
                    "QUINELLA_PLACE": {
                        "1,10": rng(None, None, "tail_member", "stale"),
                        "1,2": rng(0.48, 0.71),
                        "2,10": rng(0.62, 0.95, "tail_member"),
                    },
                    "PLACE": {
                        "5": rng(0.91, 0.95, "floor"),
                        "1": rng(None, None),
                        "10": rng(0.55, 1.12, "tail"),
                        "2": rng(0.7, 0.86),
                    },
                },
            ),
            shadow_race(
                "TOK",
                1,
                jst(10, 5),
                5,
                {
                    "QUINELLA_PLACE": {
                        "1,3": rng(None, None, "scratched"),
                        "3,9": rng(0.99, 1.01),  # no plain row: draws nothing
                        "1,4": rng(0.41, 0.57, "tail_member"),
                        "1,2": rng(0.88, 1.21),
                    },  # '2,4' has no shadow entry
                    "WIN": {
                        "4": win(0.05, 0.01, 0.12, "tail"),
                        "99": win(1.11, 1.0, 1.2),  # no plain row: draws nothing
                        "3": win(None, None, None, "scratched"),
                        "2": win(0.74, 0.6, 0.88, "floor"),
                        "1": win(1.07, 0.98, 1.16),
                    },
                    "PLACE": {
                        "1": rng(0.8, 0.93),
                        "2": rng(0.66, 0.79, "floor"),
                        "3": rng(None, None, "scratched"),
                        "4": rng(0.33, 1.4, "tail"),
                    },
                },
            ),
        ],  # KYO 10 has no shadow entry; no shadow QUINELLA pool anywhere
    }


def shadow_index(shadow: dict | None) -> dict:
    out = {}
    for r in (shadow or {}).get("races", []):
        for pool_code, entries in r["pools"].items():
            for combo, entry in entries.items():
                out[(r["race_id"], pool_code, combo)] = entry
    return out


def shadow_races(shadow: dict | None) -> dict:
    return {r["race_id"]: r for r in (shadow or {}).get("races", [])}


def all_rows(feed: dict):
    for venue in feed["venues"]:
        for race_ in venue["races"]:
            for pool_code, body in race_["pools"].items():
                for row_ in body["rows"]:
                    yield venue["venue"], race_, pool_code, row_


def _check_fixture(feed: dict, shadow: dict) -> None:
    """The fixture has what the tests need."""
    idx = shadow_index(shadow)
    entries = list(idx.values())
    assert {f for e in entries for f in e["flags"]} == set(FLAGS)
    assert any(e.get("value_mean") is None and "value_q05" in e for e in entries)
    assert any(e.get("value_at_low") is None and "value_at_low" in e for e in entries)
    plain_keys = {(r["race_id"], p, ln["combination"]) for _, r, p, ln in all_rows(feed)}
    assert plain_keys - set(idx), "some rows have no shadow entry"
    assert set(idx) - plain_keys, "some shadow entries have no plain row"
    assert len({r["tau_bucket_min"] for r in shadow["races"]}) >= 3


FEED, SHADOW = plain_feed(), shadow_feed()
_check_fixture(FEED, SHADOW)


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


def pool_block(block, pool_code: str):
    found = block.select(f'[data-pool="{pool_code}"]')
    assert len(found) == 1, f"pool {pool_code}: {len(found)} blocks"
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


def order_of(page: BeautifulSoup) -> list[tuple[str, str, str]]:
    """(race_id, pool, combination) of every row, in document order."""
    return [
        (block["data-race"], pool_el["data-pool"], row_el["data-combination"])
        for block in race_blocks(page)
        for pool_el in block.select("[data-pool]")
        for row_el in pool_el.select("[data-combination]")
    ]


def venues_of(feed: dict) -> list[str]:
    return [v["venue"] for v in feed["venues"]]


def render_all(tmp_path: Path, feed=FEED, shadow=SHADOW) -> dict[str, BeautifulSoup]:
    steps = [
        call("renderShadow", feed, shadow, {"now": ms(NOW), "venue": v}) for v in venues_of(feed)
    ]
    shown = results(steps, tmp_path)
    return {v: soup(html) for v, html in zip(venues_of(feed), shown, strict=True)}


def row_in(pages, venue, race_id, pool_code, combo):
    return row_of(pool_block(race_block(pages[venue], race_id), pool_code), combo)


def decimals(shown: str) -> list[str]:
    return re.findall(r"\d+\.\d+", shown)


def check_value(shown: str, pool_code: str, entry: dict | None, where: str) -> None:
    shown = shown.strip()
    keys = (
        ("value_mean", "value_q05", "value_q95")
        if pool_code == "WIN"
        else (
            "value_at_low",
            "value_at_high",
        )
    )
    figures = [None if entry is None else entry.get(k) for k in keys]
    if all(f is None for f in figures):
        assert shown == "-", f"{where}: {shown!r} for no value"
        return
    want = [f"{f:.2f}" for f in figures if f is not None]
    assert decimals(shown) == want, f"{where}: {shown!r}, want {want}"
    if pool_code != "WIN" and None not in figures:
        pattern = rf"{re.escape(want[0])}\s*[-–]\s*{re.escape(want[1])}"
        assert re.fullmatch(pattern, shown), f"{where}: {shown!r}"
    if None in figures:
        assert "-" in shown, f"{where}: a null figure shows no '-': {shown!r}"


def move_text_ok(shown: str, pct) -> bool:
    shown = shown.strip()
    if pct is None:
        return shown == "-"
    if pct == 0:
        return shown in {"0.0%", "+0.0%", "±0.0%"}
    return shown == f"{'+' if pct > 0 else ''}{pct:.1f}%"


def chips(row_el) -> list:
    return cell(row_el, "flags").select("[data-flag]")


# ---------------------------------------------------------------- forbidden words

STAKE_WORDS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bstak(e|es|ed|ing)\b",
        r"\bkelly\b",
        r"\bwager",
        r"\bbet[\s_-]*siz",
        r"\bbet[\s_-]*amount",
        r"\bbet[\s_-]*if\b",
        r"\bfair[\s_-]*(odds|price)",
    )
]
AMOUNT_INPUTS = re.compile(r"<\s*(input|textarea)\b|contenteditable", re.IGNORECASE)


def stake_words_in(source: str) -> list[str]:
    return [m.group(0) for p in STAKE_WORDS for m in p.finditer(source)]


# ---------------------------------------------------------------- tests


@needs_node
def test_renders_values_for_every_race_and_pool(tmp_path):
    # the page loads board.js, then shadow.js
    assert PAGE.is_file(), f"{PAGE.relative_to(REPO)} is missing"
    srcs = [s.get("src", "") for s in soup(PAGE.read_text(encoding="utf-8")).select("script[src]")]
    names = [Path(s.split("?")[0]).name for s in srcs]
    assert "board.js" in names and "shadow.js" in names, srcs
    assert names.index("board.js") < names.index("shadow.js"), srcs

    idx, taus = shadow_index(SHADOW), shadow_races(SHADOW)
    boards = results(
        [
            call("renderBoard", FEED, {"now": ms(NOW), "venue": v}, mod="board")
            for v in venues_of(FEED)
        ],
        tmp_path,
    )
    pages = render_all(tmp_path)
    for venue, board_html in zip(venues_of(FEED), boards, strict=True):
        page, board = pages[venue], soup(board_html)
        assert {e["data-venue"] for e in page.select("[data-venue]")} == set(venues_of(FEED))
        races = [r for v in FEED["venues"] if v["venue"] == venue for r in v["races"]]
        assert [b["data-race"] for b in race_blocks(page)] == [r["race_id"] for r in races]
        for race_ in races:
            block = race_block(page, race_["race_id"])
            assert {e["data-pool"] for e in block.select("[data-pool]")} == set(race_["pools"])
            tau = (taus.get(race_["race_id"]) or {}).get("tau_bucket_min")
            for pool_code, body in race_["pools"].items():
                for row_ in body["rows"]:
                    key = (race_["race_id"], pool_code, row_["combination"])
                    row_el = row_of(pool_block(block, pool_code), row_["combination"])
                    board_row = row_of(
                        pool_block(race_block(board, race_["race_id"]), pool_code),
                        row_["combination"],
                    )
                    for col in ("odds", "move-morning", "move-t20"):  # the board's own cells
                        assert text(cell(row_el, col)) == text(cell(board_row, col)), (key, col)
                    entry = idx.get(key)
                    check_value(text(cell(row_el, "value")), pool_code, entry, str(key))
                    if entry is not None:
                        want = "-" if tau is None else f"@T-{tau}"
                        assert text(cell(row_el, "bucket")) == want, key
                        assert {c["data-flag"] for c in chips(row_el)} == set(entry["flags"]), key

    # the page as the browser runs it: plain scripts, polling both feeds every 5 s
    newer = copy.deepcopy(SHADOW)
    newer["races"][1]["pools"]["WIN"]["2"] = win(1.37, 1.29, 1.46, "tail")
    plain = {"plain": {"feed": FEED, "etag": '"p1"'}}
    first = {**plain, "shadow": {"feed": SHADOW, "etag": '"s1"'}}
    second = {**plain, "shadow": {"feed": newer, "etag": '"s2"'}}
    out = run_js(
        [{"fire": 5000}, {"fire": 5000, "servers": second}],
        tmp_path,
        servers=first,
        browser=True,
    )
    loaded, again, changed = out["results"]
    urls = [r["url"] for r in loaded["requests"]]
    assert any(re.search(r"(^|/)shadow-feed(\?|$)", u) for u in urls), urls
    assert any(re.search(r"(^|/)feed(\?|$)", u) for u in urls), urls
    shown = soup(loaded["shadow"])
    kyo3 = rid("KYO", 3)
    for b in ("2", "10", "1"):
        entry = idx[(kyo3, "WIN", b)]
        check_value(text(cell(row_in({"KYO": shown}, "KYO", kyo3, "WIN", b), "value")), "WIN",
                    entry, f"loaded page KYO 3 WIN {b}")  # fmt: skip
    sent = {}
    for r in again["requests"]:
        kind = "shadow" if re.search(r"(^|/)shadow-feed(\?|$)", r["url"]) else "plain"
        sent[kind] = r["if_none_match"]
    assert sent == {"plain": '"p1"', "shadow": '"s1"'}, again["requests"]
    row2 = row_in({"KYO": soup(changed["shadow"])}, "KYO", kyo3, "WIN", "2")
    check_value(text(cell(row2, "value")), "WIN", newer["races"][1]["pools"]["WIN"]["2"], "new")
    assert {c["data-flag"] for c in chips(row2)} == {"tail"}


@needs_node
def test_place_and_wide_show_low_and_high_values(tmp_path):
    pages = render_all(tmp_path)
    idx = shadow_index(SHADOW)
    ranged = 0
    for venue, race_, pool_code, row_ in all_rows(FEED):
        if pool_code not in ("PLACE", "QUINELLA_PLACE"):
            continue
        key = (race_["race_id"], pool_code, row_["combination"])
        entry = idx.get(key)
        shown = text(cell(row_in(pages, venue, *key), "value"))
        if entry is None or entry["value_at_low"] is None:
            continue
        ranged += 1
        low, high = f"{entry['value_at_low']:.2f}", f"{entry['value_at_high']:.2f}"
        assert re.fullmatch(rf"{re.escape(low)}\s*[-–]\s*{re.escape(high)}", shown), (key, shown)
    assert ranged >= 10


@needs_node
def test_null_values_show_a_dash(tmp_path):
    no_tau = copy.deepcopy(SHADOW)
    for r in no_tau["races"]:
        if r["race_id"] == rid("TOK", 1):
            r["tau_bucket_min"] = None
    with_nulls, without_shadow, tau_null = results(
        [
            call("renderShadow", FEED, SHADOW, {"now": ms(NOW), "venue": "TOK"}),
            call("renderShadow", FEED, None, {"now": ms(NOW), "venue": "KYO"}),
            call("renderShadow", FEED, no_tau, {"now": ms(NOW), "venue": "TOK"}),
        ],
        tmp_path,
    )
    tok, kyo = {"TOK": soup(with_nulls)}, {"KYO": soup(without_shadow)}
    t1, k3, k10 = rid("TOK", 1), rid("KYO", 3), rid("KYO", 10)
    dash_cells = [
        (t1, "WIN", "3", "value"),  # all-null entry (scratched)
        (t1, "PLACE", "3", "value"),
        (t1, "QUINELLA_PLACE", "1,3", "value"),
        (t1, "QUINELLA_PLACE", "2,4", "value"),  # no shadow entry for the combination
        (t1, "QUINELLA", "1,2", "value"),  # no shadow pool
        (t1, "QUINELLA", "2,4", "value"),
    ]
    for key in dash_cells:
        assert text(cell(row_in(tok, "TOK", *key[:3]), key[3])) == "-", key
    # the cells next to a dash still carry their figures
    assert decimals(text(cell(row_in(tok, "TOK", t1, "WIN", "4"), "value"))) == [
        "0.05",
        "0.01",
        "0.12",
    ]
    # a band that is null: the mean shows, the band is '-'
    t11 = {"TOK": soup(with_nulls)}
    partial = text(cell(row_in(t11, "TOK", rid("TOK", 11), "WIN", "2"), "value"))
    assert decimals(partial) == ["0.83"] and "-" in partial.replace("0.83", ""), partial
    # no shadow feed at all, and a race the shadow feed lacks: every value and bucket is '-'
    kyo_races = [r for v in FEED["venues"] if v["venue"] == "KYO" for r in v["races"]]
    assert [r["race_id"] for r in kyo_races] == [k3, k10]
    for race_ in kyo_races:
        for pool_code, body in race_["pools"].items():
            for row_ in body["rows"]:
                key = (race_["race_id"], pool_code, row_["combination"])
                row_el = row_in(kyo, "KYO", *key)
                assert text(cell(row_el, "value")) == "-", key
                assert text(cell(row_el, "bucket")) == "-", key
                assert not chips(row_el), key
    # KYO 10 is missing from the shadow feed: '-' even when the shadow feed is there
    with_shadow = {"KYO": soup(results(
        [call("renderShadow", FEED, SHADOW, {"now": ms(NOW), "venue": "KYO"})], tmp_path
    )[0])}  # fmt: skip
    for n in ("1", "2", "3", "4"):
        row_el = row_in(with_shadow, "KYO", k10, "WIN", n)
        assert text(cell(row_el, "value")) == "-" and text(cell(row_el, "bucket")) == "-", n
    # a null bucket shows '-' while the values still show
    page = {"TOK": soup(tau_null)}
    for pool_code, combo in (("WIN", "1"), ("PLACE", "4"), ("QUINELLA_PLACE", "1,4")):
        row_el = row_in(page, "TOK", t1, pool_code, combo)
        assert text(cell(row_el, "bucket")) == "-", (pool_code, combo)
        assert decimals(text(cell(row_el, "value"))), (pool_code, combo)

    # the page polls a shadow feed that is not there yet: the board shows, values are '-'
    out = run_js([{"poll": True}], tmp_path, servers={"plain": {"feed": FEED, "etag": '"p"'}})
    polled = {"KYO": soup(out["results"][-1]["shadow"])}
    for race_ in FEED["venues"][0]["races"]:
        for pool_code, body in race_["pools"].items():
            for row_ in body["rows"]:
                row_el = row_in(polled, "KYO", race_["race_id"], pool_code, row_["combination"])
                assert text(cell(row_el, "value")) == "-", (race_["race_id"], pool_code, row_)


@needs_node
def test_flags_render_as_chips_with_the_legend(tmp_path):
    pages = render_all(tmp_path)
    idx = shadow_index(SHADOW)
    seen = set()
    for venue, race_, pool_code, row_ in all_rows(FEED):
        key = (race_["race_id"], pool_code, row_["combination"])
        row_el = row_in(pages, venue, *key)
        found = chips(row_el)
        want = (idx.get(key) or {}).get("flags", [])
        assert sorted(c["data-flag"] for c in found) == sorted(want), key
        for chip in found:
            label = text(chip).lower()
            assert chip["data-flag"].replace("_", " ") in label, (key, label)
            assert "_" not in label, (key, label)
            seen.add(chip["data-flag"])
    assert seen == set(FLAGS)

    assert PAGE.is_file(), f"{PAGE.relative_to(REPO)} is missing"
    legends = soup(PAGE.read_text(encoding="utf-8")).select("#flag-legend") + [
        e for p in pages.values() for e in p.select("#flag-legend")
    ]
    assert legends, "no element with id flag-legend in shadow.html or the rendered page"
    legend = legends[0]
    assert {e["data-flag"] for e in legend.select("[data-flag]")} >= set(FLAGS)
    said = text(legend).lower()
    assert "candidate" in said and "not adopted" in said, said


@needs_node
def test_the_bucket_label_is_shown(tmp_path):
    pages = render_all(tmp_path)
    idx, taus = shadow_index(SHADOW), shadow_races(SHADOW)
    labels = set()
    for venue, race_, pool_code, row_ in all_rows(FEED):
        key = (race_["race_id"], pool_code, row_["combination"])
        if key not in idx:
            continue
        shown = text(cell(row_in(pages, venue, *key), "bucket"))
        assert shown == f"@T-{taus[race_['race_id']]['tau_bucket_min']}", (key, shown)
        labels.add(shown)
    assert labels == {"@T-5", "@T-10", "@T-20"}


@needs_node
def test_the_t2_move_is_shown(tmp_path):
    pages = render_all(tmp_path)
    count = 0
    for venue, race_, pool_code, row_ in all_rows(FEED):
        key = (race_["race_id"], pool_code, row_["combination"])
        el = cell(row_in(pages, venue, *key), "move-t2")
        pct = row_.get("move_t2_pct")  # absent on a feed without the T-2 amendment
        assert move_text_ok(text(el), pct), (key, text(el), pct)
        assert ("up" in classes(el)) == (pct is not None and pct > 0), (key, pct)
        assert ("down" in classes(el)) == (pct is not None and pct < 0), (key, pct)
        count += pct is not None
    assert count >= 20


@needs_node
def test_rows_keep_the_board_order(tmp_path):
    steps = []
    for v in venues_of(FEED):
        steps.append(call("renderBoard", FEED, {"now": ms(NOW), "venue": v}, mod="board"))
        steps.append(call("renderShadow", FEED, SHADOW, {"now": ms(NOW), "venue": v}))
    shown = results(steps, tmp_path)
    for i, venue in enumerate(venues_of(FEED)):
        board, page = soup(shown[2 * i]), soup(shown[2 * i + 1])
        assert order_of(page) == order_of(board), venue
        # independently: feed order of races and rows, the board's pool order
        want = []
        for race_ in next(v for v in FEED["venues"] if v["venue"] == venue)["races"]:
            rank = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"]
            for pool_code in sorted(race_["pools"], key=rank.index):
                want += [
                    (race_["race_id"], pool_code, r["combination"])
                    for r in race_["pools"][pool_code]["rows"]
                ]
        assert order_of(page) == want, venue
        for block in race_blocks(page):
            for pool_el in block.select("[data-pool]"):
                values = pool_el.select('[data-col="value"]')
                figures = [v for v in values if text(v) != "-"]
                assert len({tuple(classes(v)) for v in figures}) <= 1, "value cells styled by value"
                assert not [v for v in values if v.get("style")], "inline style on a value cell"
                assert not [r for r in pool_el.select("[data-combination]") if r.get("style")]
    tok = soup(shown[3])
    assert not tok.select('[data-combination="99"]'), "a shadow-only row was drawn"
    assert not tok.select('[data-combination="3,9"]'), "a shadow-only row was drawn"


def test_page_has_no_stake_words_or_amount_inputs(tmp_path):
    bad = ["Stake 100", "half Kelly", "wager", "Bet size", "bet amount", "bet if 3.2",
           "fair odds 2.5", "staking plan", "<input type=number>", "<TEXTAREA>"]  # fmt: skip
    fine = ["mistake", "Between races", "Wide", "gross return per ¥1", "nothing here is a bet"]
    assert all(stake_words_in(s) or AMOUNT_INPUTS.search(s) for s in bad)
    assert not [s for s in fine if stake_words_in(s) or AMOUNT_INPUTS.search(s)]

    for path in (PAGE, SCRIPT, BOARD_JS):
        assert path.is_file(), f"{path.relative_to(REPO)} is missing"
        source = path.read_text(encoding="utf-8")
        assert source.strip(), f"{path.name} is empty"
        assert not stake_words_in(source), f"{path.name}: {sorted(set(stake_words_in(source)))}"
        assert not AMOUNT_INPUTS.search(source), f"{path.name} has an input"
    assert "renderShadow" in SCRIPT.read_text(encoding="utf-8")
    assert not soup(PAGE.read_text(encoding="utf-8")).select("input, textarea")

    if NODE is None:
        return
    for venue, page in render_all(tmp_path).items():
        html = str(page)
        assert not stake_words_in(html), (venue, sorted(set(stake_words_in(html))))
        assert not AMOUNT_INPUTS.search(html), venue
        assert not page.select("input, textarea"), venue


def test_the_banner_says_no_stakes():
    assert PAGE.is_file(), f"{PAGE.relative_to(REPO)} is missing"
    banners = soup(PAGE.read_text(encoding="utf-8")).select("#banner")
    assert len(banners) == 1, f"{len(banners)} elements with id banner in shadow.html"
    banner = banners[0]
    assert banner.get("hidden") is None, "the banner is hidden"
    said = text(banner)
    low = said.lower()
    assert "shadow" in low, said
    assert "v0" in low, said
    assert "return" in low and "final" in low, said
    assert re.search(r"¥\s*1\b|\b1\s*yen\b", said), said
    assert re.search(r"\bno\s+(real\s+)?(money|amounts?)\b", low), said
    assert re.search(r"\b(not|no|nothing|never|none)\b[^.]{0,80}\bbets?\b", low), said
    assert not stake_words_in(said), said
