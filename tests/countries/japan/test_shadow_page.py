"""The shadow page (tickets #122, #498): `shadow.html` and `shadow.js` in
`src/horsetrack/countries/japan/board/web/`, the live board (#42's board.js) plus each row's shadow
value, bucket label and avoid-flag chips, driven from node under a tiny fake DOM as in
test_board_page.py.

Contract pinned here (the coder implements against it):

`shadow.js` runs in the browser after `board.js` (shadow.html loads `board.js` first, then
`shadow.js`, as plain `<script src>` tags) and reuses board.js's render functions. Under node it
sets `module.exports` (when `module` exists) to at least:

- `renderShadow(feed, shadowFeed, {now, venue, race}) -> string`: the HTML for one race of one
  venue (the plain feed's first venue by default; `now` in ms since the epoch; the race chosen as
  board.js's `renderBoard` chooses it: `race` when it is of the venue, else the next race to post,
  else the venue's last race). `feed` is the plain feed (docs/specs/board-feed.md, each row with
  `t10_odds`, `t5_odds`, `t2_odds` and `move_t20_t2_pct`, which may be absent); `shadowFeed` is
  #121's shadow feed (each entry with #499's `ladder`, which may be absent) or
  null. Rows join on (race_id, pool, combination). The HTML carries every hook of board.js's
  `renderBoard` (venue tabs `data-venue`, the race strip's `#<hash>` links, the race block
  `data-race` containing `data-pool` blocks, rows `data-combination`, cells `data-col="odds"`,
  `"move-morning"`, `"move-t20"`), in the same order as `renderBoard` draws them for the same
  plain feed: favourite first by the crowd's current odds (#498, the owner's display choice), never
  sorted, highlighted or coloured by shadow value (D19). Rows come from the plain feed only:
  a shadow entry without a plain row draws nothing. Each row adds these cells:
  - The odds ladder (#499, replacing #498's single T-20→T-2 cell `data-col="move-t20-t2"`, which
    is gone, as is the 'Since T-2' cell `data-col="move-t2"`): five cells `data-col="ladder-t20"`,
    `"ladder-t10"`, `"ladder-t5"`, `"ladder-t2"`, `"ladder-final"`, in that order, under header
    cells reading `T-20`, `T-10`, `T-5`, `T-2`, `Final` in the same columns. Each shows the shadow
    entry's `ladder` checkpoint ("20", "10", "5", "2", "0"): an observed one as its odds with one
    decimal (e.g. `4.8`), class `up` / `down` when it is above / below the T-20 odds (none for
    the T-20 cell itself, or without a T-20 figure); a projected one as `≈` and its odds with one
    decimal (e.g. `≈4.8`), with class `projected`; a null one as `-`. A row with no ladder (no
    shadow entry, a pool the shadow feed lacks such as Quinella, no shadow feed) shows the plain
    row's `t20_odds`, `t10_odds`, `t5_odds`, `t2_odds` as observed and `-` for the rest (the final
    always). The two feeds refresh apart, so per checkpoint an observed figure wins: where the
    shadow ladder's point is projected or null and the plain row's reference for it is not null,
    the cell shows the plain reference as observed (no `≈`, no `projected`); an observed ladder
    point stays when the plain reference is null. A legend line in shadow.html explains the `≈`.
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
  the innerHTML of the element with id `shadow` to the shadow page of the selected venue and race.
  The page starts polling at load and polls again every 5 s (`setTimeout`/`setInterval` with
  5000 ms). As on the board page (#498): the URL hash selects the race at load, a strip link draws
  that race alone, a poll keeps the selection and a venue tab draws that venue's default race.
  A poll that redraws the same race keeps each pool's horizontal scroll: the `[data-pool]` element
  (shadow.html's `.pool`, `overflow-x: auto`) has the same `scrollLeft` after the redraw as before.

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

Both feeds are synthetic objects built here, of the shapes in the ticket; one test builds the plain
feed with `feed.build_board` from synthetic odds logs under tmp_path.
"""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

# isort: split
from page_fakes import (
    BROWSER_FAKES,
    chain_of,
    click_target,
    favourite_order,
    kept_in_place,
    moved_to_race,
    strip_links,
)
from shadow_fakes import FakeOrdering, FakeWin

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
    scrollIntoView(arg) { noteIntoView(this.id, arg); }, focus() {},
    matches() { return false; },
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
global.removeEventListener = () => {};
/*BROWSER_FAKES*/
global.setInterval = (fn, delay) => { timers.push({ fn, delay: Number(delay), keep: true });
  return timers.length; };
global.setTimeout = (fn, delay) => { timers.push({ fn, delay: Number(delay), keep: false });
  return timers.length; };
global.clearInterval = () => {}; global.clearTimeout = () => {};
global.requestAnimationFrame = (cb) => setImmediate(() => cb(Date.now()));
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
// input.liveDom: #shadow's innerHTML is parsed into fresh elements on every write, as a browser
// replaces them (each with scrollLeft 0); querySelector(All) on #shadow and the document find them.
const VOID_TAGS = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link",
  "meta", "source", "track", "wbr"]);
const unescapeAttr = (s) => s.replace(/&quot;/g, '"').replace(/&#39;/g, "'")
  .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
function parseAttrs(textIn) {
  const attrs = {}, re = /([^\s=\/>"']+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>"']+)))?/g;
  let m;
  while ((m = re.exec(textIn))) {
    const v = m[2] !== undefined ? m[2] : m[3] !== undefined ? m[3] : m[4] !== undefined ? m[4]
      : "";
    attrs[m[1].toLowerCase()] = unescapeAttr(v);
  }
  return attrs;
}
function descendants(node) {
  const out = [];
  const walk = (n) => { for (const c of n.liveChildren || []) { out.push(c); walk(c); } };
  walk(node);
  return out;
}
function queryIn(node, sel) { return descendants(node).filter((e) => selectorMatches(e, sel)); }
function liveNode(tag, attrs, parent) {
  const cls = () => String(attrs.class || "").split(/\s+/).filter(Boolean);
  const n = {
    tagName: tag.toUpperCase(), nodeName: tag.toUpperCase(), nodeType: 1, attrs,
    parentElement: parent, parentNode: parent, liveChildren: [],
    scrollLeft: 0, scrollTop: 0, scrollWidth: 2000, clientWidth: 360, offsetLeft: 0,
    offsetWidth: 0, offsetTop: 0,
    get id() { return attrs.id || ""; },
    get className() { return attrs.class || ""; },
    get children() { return this.liveChildren.slice(); },
    get dataset() {
      const d = {};
      for (const k of Object.keys(attrs)) {
        if (!k.startsWith("data-")) continue;
        d[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = attrs[k];
      }
      return d;
    },
    getAttribute(k) { return k in attrs ? attrs[k] : null; },
    hasAttribute(k) { return k in attrs; },
    setAttribute(k, v) { attrs[k] = String(v); },
    removeAttribute(k) { delete attrs[k]; },
    classList: { contains: (c) => cls().includes(c), add() {}, remove() {}, toggle() {} },
    matches(sel) { return selectorMatches(this, sel); },
    closest(sel) {
      for (let at = this; at; at = at.parentElement) if (selectorMatches(at, sel)) return at;
      return null;
    },
    querySelectorAll(sel) { return queryIn(this, sel); },
    querySelector(sel) { return queryIn(this, sel)[0] || null; },
    getElementsByClassName(c) { return descendants(this).filter((e) => e.classList.contains(c)); },
    getElementsByTagName(t) {
      return descendants(this).filter((e) => t === "*" || e.tagName === t.toUpperCase());
    },
    scrollTo(a, b) {
      if (a && typeof a === "object") {
        if ("left" in a) this.scrollLeft = Number(a.left);
        if ("top" in a) this.scrollTop = Number(a.top);
      } else if (a !== undefined) {
        this.scrollLeft = Number(a);
        if (b !== undefined) this.scrollTop = Number(b);
      }
    },
    addEventListener() {}, removeEventListener() {}, focus() {},
    scrollIntoView(arg) { noteIntoView(this.id, arg); },
  };
  n.scroll = n.scrollTo;
  return n;
}
function parseHtml(root, html) {
  const top = [], stack = [root];
  const re = /<!--[\s\S]*?-->|<(\/?)([a-zA-Z][\w-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/g;
  let m;
  root.liveChildren = top;
  while ((m = re.exec(html))) {
    if (m[2] === undefined) continue;
    const tag = m[2].toLowerCase();
    if (m[1]) {
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].tagName === tag.toUpperCase()) { stack.length = i; break; }
      }
      continue;
    }
    const parent = stack[stack.length - 1];
    const node = liveNode(tag, parseAttrs(m[3]), parent === root ? root : parent);
    parent.liveChildren.push(node);
    if (!VOID_TAGS.has(tag) && !/\/\s*$/.test(m[3])) stack.push(node);
  }
}
if (input.liveDom) {
  const root = el("shadow");
  let html = "";
  root.tagName = root.nodeName = "DIV";
  root.attrs.id = "shadow";
  root.liveChildren = [];
  Object.defineProperty(root, "innerHTML", {
    get: () => html,
    set: (v) => { html = String(v); parseHtml(root, html); },
    configurable: true,
  });
  Object.defineProperty(root, "children", {
    get: () => root.liveChildren.slice(), configurable: true,
  });
  root.querySelectorAll = (sel) => queryIn(root, sel);
  root.querySelector = (sel) => queryIn(root, sel)[0] || null;
  root.getElementsByClassName = (c) => descendants(root).filter((e) => e.classList.contains(c));
  document.querySelectorAll = (sel) => [root, ...descendants(root)]
    .filter((e) => selectorMatches(e, sel));
  document.querySelector = (sel) => document.querySelectorAll(sel)[0] || null;
}
function poolScroll() {
  const out = {};
  for (const e of queryIn(el("shadow"), "[data-pool]")) {
    out[e.getAttribute("data-pool")] = e.scrollLeft;
  }
  return out;
}
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
  const results = [{ shadow: el("shadow").innerHTML, requests: requests.slice(),
    pools: input.liveDom ? poolScroll() : undefined }];
  for (const step of input.steps) {
    if (step.servers) servers = step.servers;
    const before = requests.length;
    resetViewport();
    if (step.call) {
      results.push({ result: mods[step.mod || "shadow"][step.call](...step.args) });
      continue;
    }
    if (step.poll) {
      await mods.shadow.poll();
    } else if (step.fire) {
      const due = timers.filter((t) => t.delay === step.fire && !t.done);
      for (const t of due) { if (!t.keep) t.done = true; t.fn(); }
    } else if (step.click) {
      clickChain(step.click, step.root || "shadow");
    } else if (typeof step.hash === "string") {
      location.hash = step.hash;
    } else if (step.scroll) {  // the reader scrolls pools sideways: {pool code: scrollLeft}
      for (const e of queryIn(el("shadow"), "[data-pool]")) {
        const code = e.getAttribute("data-pool");
        if (code in step.scroll) e.scrollLeft = step.scroll[code];
      }
    }
    await settle();
    results.push({ shadow: el("shadow").innerHTML, hash: location.hash,
      requests: requests.slice(before), viewport: viewportState(),
      pools: input.liveDom ? poolScroll() : undefined });
  }
  fs.writeFileSync(OUT, JSON.stringify({ results, timers: timers.map((t) => t.delay) }));
})().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
""".replace("/*BROWSER_FAKES*/", BROWSER_FAKES)


def run_js(
    steps: list[dict],
    tmp_path: Path,
    *,
    servers: dict | None = None,
    browser: bool = False,
    hash_: str = "",
    live_dom: bool = False,
) -> dict:
    for path in (SCRIPT, BOARD_JS):
        assert path.is_file(), f"{path.relative_to(REPO)} is missing"
    out = tmp_path / "node-out.json"
    payload = {
        "steps": steps,
        "servers": servers or {},
        "browser": browser,
        "now": ms(NOW),
        "hash": hash_,
        "liveDom": live_dom,
    }
    proc = subprocess.run(
        [NODE, "-e", HARNESS, str(BOARD_JS), str(SCRIPT), str(out)],
        input=json.dumps(payload),
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
    """One plain-feed row (board-feed.md), with the T-2 fields (#498) unless no_t2; `t2` is the
    T-20 -> T-2 move, `move_t20_t2_pct`."""
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
        out["move_t20_t2_pct"] = None if scr else t2
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


def race_keys(feed: dict) -> list[tuple[str, str]]:
    return [(v["venue"], r["race_id"]) for v in feed["venues"] for r in v["races"]]


def render_all(tmp_path: Path, feed=FEED, shadow=SHADOW) -> dict[str, BeautifulSoup]:
    """race_id -> the shadow page of its venue with that race selected (one race per page)."""
    keys = race_keys(feed)
    steps = [
        call("renderShadow", feed, shadow, {"now": ms(NOW), "venue": v, "race": r}) for v, r in keys
    ]
    shown = results(steps, tmp_path)
    return {r: soup(html) for (_, r), html in zip(keys, shown, strict=True)}


def row_in(pages, venue, race_id, pool_code, combo):
    """The row in `pages` keyed by race_id (render_all) or, for one page, by venue."""
    page = pages[race_id] if race_id in pages else pages[venue]
    return row_of(pool_block(race_block(page, race_id), pool_code), combo)


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
    keys = race_keys(FEED)
    boards = results(
        [
            call("renderBoard", FEED, {"now": ms(NOW), "venue": v, "race": r}, mod="board")
            for v, r in keys
        ],
        tmp_path,
    )
    pages = render_all(tmp_path)
    for (venue, race_id), board_html in zip(keys, boards, strict=True):
        page, board = pages[race_id], soup(board_html)
        assert {e["data-venue"] for e in page.select("[data-venue]")} == set(venues_of(FEED))
        assert [b["data-race"] for b in race_blocks(page)] == [race_id]  # one race per page
        races = [r for v in FEED["venues"] if v["venue"] == venue for r in v["races"]]
        for race_ in [r for r in races if r["race_id"] == race_id]:
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
    t1, t11, k3, k10 = rid("TOK", 1), rid("TOK", 11), rid("KYO", 3), rid("KYO", 10)
    with_nulls, t11_page, kyo3_bare, kyo10_bare, tau_null = results(
        [
            call("renderShadow", FEED, SHADOW, {"now": ms(NOW), "venue": "TOK", "race": t1}),
            call("renderShadow", FEED, SHADOW, {"now": ms(NOW), "venue": "TOK", "race": t11}),
            call("renderShadow", FEED, None, {"now": ms(NOW), "venue": "KYO", "race": k3}),
            call("renderShadow", FEED, None, {"now": ms(NOW), "venue": "KYO", "race": k10}),
            call("renderShadow", FEED, no_tau, {"now": ms(NOW), "venue": "TOK", "race": t1}),
        ],
        tmp_path,
    )
    tok = {"TOK": soup(with_nulls)}
    kyo = {k3: soup(kyo3_bare), k10: soup(kyo10_bare)}
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
    partial = text(cell(row_in({"TOK": soup(t11_page)}, "TOK", t11, "WIN", "2"), "value"))
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
        [call("renderShadow", FEED, SHADOW, {"now": ms(NOW), "venue": "KYO", "race": k10})],
        tmp_path,
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
    assert [b["data-race"] for b in race_blocks(polled["KYO"])] == [k3]  # the next race to post
    for race_ in FEED["venues"][0]["races"][:1]:
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


# ---------------------------------------------------------------- the odds ladder (#499)

LADDER_COLS = ("ladder-t20", "ladder-t10", "ladder-t5", "ladder-t2", "ladder-final")
LADDER_KEYS = ("20", "10", "5", "2", "0")
LADDER_HEADS = ("T-20", "T-10", "T-5", "T-2", "Final")
PLAIN_REFS = ("t20_odds", "t10_odds", "t5_odds", "t2_odds", None)  # the final has none


def seen(odds):
    return {"odds": odds, "observed": True}


def ahead(odds):
    return {"odds": odds, "observed": False}


def ladder_line(combo, odds, t20, t10, *, hi=None) -> dict:
    """A plain row 9 minutes before the post: T-20 and T-10 passed, T-5 and T-2 not yet."""
    out = line(combo, odds, hi=hi, mm=0.0, t20=None, t2=None)
    out.update(t20_odds=t20, t10_odds=t10, t5_odds=None, t2_odds=None)
    out["move_t20_pct"] = None if t20 is None else round((odds / t20 - 1) * 100, 1)
    return out


LADDER_POST = jst(10, 10)  # NOW is 10:01: nine minutes to post


def ladder_feeds() -> tuple[dict, dict]:
    """TOK 8 at NOW, with a ladder per shadow entry: observed at T-20 and T-10, projected ahead.
    Shadow entries agree with the plain rows' t20_odds / t10_odds."""
    plain_race = race(
        "TOK",
        8,
        LADDER_POST,
        NOW,
        {
            "WIN": pool(
                ladder_line("1", 2.4, 2.8, 2.5),
                ladder_line("2", 5.1, 4.6, 4.9),
                ladder_line("3", 9.3, 9.3, 9.3),
                ladder_line("4", 31.0, None, 28.7),  # first priced after T-20
            ),
            "PLACE": pool(
                ladder_line("1", 1.3, 1.4, 1.3, hi=1.6),
                ladder_line("2", 1.9, 1.7, 1.8, hi=2.7),
                ladder_line("4", 6.2, 6.6, 6.4, hi=9.9),  # its shadow entry has no ladder
            ),
            "QUINELLA_PLACE": pool(
                ladder_line("1,2", 3.3, 3.7, 3.4, hi=4.1),
                ladder_line("1,3", 5.6, 5.2, 5.9, hi=7.2),  # no shadow entry
            ),
            "QUINELLA": pool(  # no shadow pool, and no projection for it
                ladder_line("1,2", 7.7, 8.4, 7.9),
                ladder_line("1,3", 14.6, 13.2, 14.1),
                ladder_line("2,4", 120.5, None, 133.3),
            ),
        },
    )
    plain = {
        "race_date": DAY,
        "generated_utc": utc_text(NOW),
        "venues": [{"venue": "TOK", "races": [plain_race]}],
    }

    def entry(values: dict, ladder: dict | None) -> dict:
        out = dict(values)
        if ladder is not None:
            out["ladder"] = dict(zip(LADDER_KEYS, ladder, strict=True))
        return out

    shadow = {
        "race_date": DAY,
        "generated_utc": utc_text(NOW),
        "models": {"nowcast": "now-c3"},
        "races": [
            shadow_race(
                "TOK",
                8,
                LADDER_POST,
                16,
                {
                    "WIN": {
                        "1": entry(
                            win(0.9, 0.8, 1.0), (seen(2.8), seen(2.5), ahead(2.31), ahead(2.24),
                                                 ahead(2.07))
                        ),
                        "2": entry(
                            win(1.1, 1.0, 1.2), (seen(4.6), seen(4.9), ahead(5.43), ahead(5.71),
                                                 ahead(6.02))
                        ),
                        "3": entry(
                            win(0.7, 0.6, 0.8), (seen(9.3), seen(9.3), ahead(9.66), ahead(10.04),
                                                 None)
                        ),
                        "4": entry(
                            win(0.3, 0.2, 0.4), (None, seen(28.7), ahead(33.8), ahead(36.1),
                                                 ahead(41.27))
                        ),
                    },
                    "PLACE": {
                        "1": entry(
                            rng(0.8, 0.9), (seen(1.4), seen(1.3), ahead(1.21), ahead(1.08),
                                            ahead(1.0))
                        ),
                        "2": entry(
                            rng(0.7, 1.0), (seen(1.7), seen(1.8), ahead(1.93), ahead(2.04),
                                            ahead(2.18))
                        ),
                        "4": entry(rng(0.5, 0.8), None),
                    },
                    "QUINELLA_PLACE": {
                        "1,2": entry(
                            rng(0.6, 0.75), (seen(3.7), seen(3.4), ahead(3.17), ahead(3.02),
                                             ahead(2.88))
                        ),
                    },
                },
            )
        ],
    }  # fmt: skip
    return plain, shadow


def expected_ladder(row_: dict, entry: dict | None) -> list[dict | None]:
    """The five cells of a row: the shadow ladder, else the plain row's references, observed."""
    ladder = (entry or {}).get("ladder")
    if ladder is not None:
        return [ladder[k] for k in LADDER_KEYS]
    return [None if ref is None or row_.get(ref) is None else seen(row_[ref]) for ref in PLAIN_REFS]


def check_ladder_row(row_el, want: list[dict | None], where) -> None:
    cells = [cell(row_el, col) for col in LADDER_COLS]
    positions = [c for c in row_el.find_all(["th", "td"], recursive=False)]
    assert [positions.index(c) for c in cells] == sorted(positions.index(c) for c in cells), where
    t20 = want[0]["odds"] if want[0] is not None else None
    for i, (el, expected) in enumerate(zip(cells, want, strict=True)):
        shown, cls = text(el), classes(el)
        at = (where, LADDER_COLS[i], shown, cls, expected)
        if expected is None:
            assert shown == "-", at
            assert "projected" not in cls, at
            continue
        figure_ = f"{expected['odds']:.1f}"
        if expected["observed"]:
            assert shown == figure_, at
            assert "projected" not in cls, at
            if i == 0 or t20 is None:
                assert "up" not in cls and "down" not in cls, at
            else:
                assert ("up" in cls) == (expected["odds"] > t20), at
                assert ("down" in cls) == (expected["odds"] < t20), at
        else:
            assert re.fullmatch(rf"≈\s*{re.escape(figure_)}", shown), at
            assert "projected" in cls, at


@needs_node
def test_the_ladder_shows_observed_and_projected_odds(tmp_path):
    plain, shadow = ladder_feeds()
    opts = {"now": ms(NOW), "venue": "TOK"}
    with_shadow, without, board = (
        soup(html)
        for html in results(
            [
                call("renderShadow", plain, shadow, opts),
                call("renderShadow", plain, None, opts),
                call("renderBoard", plain, opts, mod="board"),
            ],
            tmp_path,
        )
    )
    # display only: the rows stay in the board's order, favourite first
    assert order_of(with_shadow) == order_of(board) == order_of(without)
    idx = shadow_index(shadow)
    race_id = rid("TOK", 8)
    kinds = {"observed": 0, "projected": 0, "null": 0, "coloured": 0}
    for _, _race, pool_code, row_ in all_rows(plain):
        key = (race_id, pool_code, row_["combination"])
        row_el = row_of(pool_block(race_block(with_shadow, race_id), pool_code), key[2])
        want = expected_ladder(row_, idx.get(key))
        check_ladder_row(row_el, want, key)
        assert not row_el.select('[data-col="move-t20-t2"]'), "the T-20→T-2 cell is back"
        assert not row_el.select('[data-col="move-t2"]'), "the blank 'Since T-2' cell is back"
        for i, w in enumerate(want):
            if w is None:
                kinds["null"] += 1
            elif w["observed"]:
                kinds["observed"] += 1
                kinds["coloured"] += i > 0 and want[0] is not None and w["odds"] != want[0]["odds"]
            else:
                kinds["projected"] += 1
        # without a shadow feed: the plain row's references, '-' ahead
        bare = row_of(pool_block(race_block(without, race_id), pool_code), key[2])
        check_ladder_row(bare, expected_ladder(row_, None), ("no shadow feed", key))
    assert kinds["projected"] >= 15 and kinds["null"] >= 10 and kinds["coloured"] >= 6, kinds

    # Quinella (no projection) and a combination without a shadow entry: observed, then '-'
    block = race_block(with_shadow, race_id)
    for pool_code, combo, t20, t10 in (
        ("QUINELLA", "1,3", "13.2", "14.1"),
        ("QUINELLA_PLACE", "1,3", "5.2", "5.9"),
        ("PLACE", "4", "6.6", "6.4"),
    ):
        row_el = row_of(pool_block(block, pool_code), combo)
        shown = [text(cell(row_el, col)) for col in LADDER_COLS]
        assert shown == [t20, t10, "-", "-", "-"], (pool_code, combo, shown)

    # the header names the five columns, each above its cells
    for pool_el in block.select("[data-pool]"):
        heads = [text(th) for th in pool_el.select("thead th")]
        row_el = pool_el.select("[data-combination]")[0]
        positions = row_el.find_all(["th", "td"], recursive=False)
        for col, head in zip(LADDER_COLS, LADDER_HEADS, strict=True):
            assert head in heads, (pool_el["data-pool"], heads)
            assert heads.index(head) == positions.index(cell(row_el, col)), (col, heads)

    # shadow.html explains the projected mark
    legends = soup(PAGE.read_text(encoding="utf-8")).select(".legend")
    assert any("≈" in text(e) for e in legends), "no legend line explains the ≈ mark"


def ref_line(combo, odds, t20, t10, t5, t2, *, hi=None) -> dict:
    """A plain row with its four references as given."""
    out = line(combo, odds, hi=hi, mm=0.0, t20=None, t2=None)
    out.update(t20_odds=t20, t10_odds=t10, t5_odds=t5, t2_odds=t2)
    out["move_t20_pct"] = None if t20 is None else round((odds / t20 - 1) * 100, 1)
    return out


@needs_node
def test_an_observed_plain_reference_wins_over_a_cached_projection(tmp_path):
    """The plain feed has crossed T-5 (and for some rows T-2, T-10) while the cached shadow feed
    still projects those checkpoints or has them null: the cell shows the plain feed's observed
    figure. Where the plain feed has null, the ladder's observed point stays."""
    post = jst(10, 2, 30)  # NOW 10:01, 1.5 minutes to post on the plain feed
    # combination: (plain row, shadow ladder, the five cells wanted)
    table = {
        "WIN": {
            "1": (ref_line("1", 2.2, 2.8, 2.5, 2.3, None),
                  (seen(2.8), seen(2.5), ahead(2.21), ahead(2.14), ahead(2.07)),
                  (seen(2.8), seen(2.5), seen(2.3), ahead(2.14), ahead(2.07))),
            "2": (ref_line("2", 5.8, 4.6, 4.9, 5.6, 5.9),
                  (seen(4.6), seen(4.9), None, ahead(5.71), ahead(6.02)),
                  (seen(4.6), seen(4.9), seen(5.6), seen(5.9), ahead(6.02))),
            "3": (ref_line("3", 8.6, 9.3, None, 8.8, None),
                  (seen(9.3), seen(9.7), ahead(9.66), None, None),
                  (seen(9.3), seen(9.7), seen(8.8), None, None)),
            "4": (ref_line("4", 31.0, None, 28.7, 33.1, 35.2),
                  (None, seen(28.7), ahead(33.8), ahead(36.1), ahead(41.27)),
                  (None, seen(28.7), seen(33.1), seen(35.2), ahead(41.27))),
        },
        "PLACE": {
            "1": (ref_line("1", 1.2, 1.4, 1.3, 1.24, None, hi=1.6),
                  (seen(1.4), seen(1.3), ahead(1.21), ahead(1.08), ahead(1.0)),
                  (seen(1.4), seen(1.3), seen(1.24), ahead(1.08), ahead(1.0))),
            "2": (ref_line("2", 1.9, 1.7, 1.8, 1.8, None, hi=2.7),
                  (seen(1.7), seen(1.8), seen(1.8), ahead(2.04), ahead(2.18)),
                  (seen(1.7), seen(1.8), seen(1.8), ahead(2.04), ahead(2.18))),
        },
        "QUINELLA_PLACE": {
            "1,2": (ref_line("1,2", 3.3, 3.7, 3.4, 3.2, None, hi=4.1),
                    (seen(3.7), ahead(3.52), ahead(3.17), ahead(3.02), ahead(2.88)),
                    (seen(3.7), seen(3.4), seen(3.2), ahead(3.02), ahead(2.88))),
            "1,3": (ref_line("1,3", 5.6, 5.2, 5.9, None, None, hi=7.2),
                    (seen(5.2), seen(5.9), seen(6.1), ahead(6.3), ahead(6.5)),
                    (seen(5.2), seen(5.9), seen(6.1), ahead(6.3), ahead(6.5))),
        },
    }  # fmt: skip
    plain = {
        "race_date": DAY,
        "generated_utc": utc_text(NOW),
        "venues": [
            {
                "venue": "TOK",
                "races": [
                    race("TOK", 9, post, NOW, {
                        code: pool(*(r for r, _, _ in rows.values()))
                        for code, rows in table.items()
                    })
                ],
            }
        ],
    }  # fmt: skip
    values = {"WIN": win(0.9, 0.8, 1.0)}
    shadow = {
        "race_date": DAY,
        "generated_utc": utc_text(NOW - timedelta(minutes=3)),
        "models": {"nowcast": "now-c3"},
        "races": [
            shadow_race("TOK", 9, post, 5, {
                code: {
                    combo: {**values.get(code, rng(0.6, 0.8)),
                            "ladder": dict(zip(LADDER_KEYS, ladder, strict=True))}
                    for combo, (_, ladder, _) in rows.items()
                }
                for code, rows in table.items()
            })
        ],
    }  # fmt: skip
    (page,) = (soup(h) for h in results([call("renderShadow", plain, shadow, {"now": ms(NOW)})],
                                        tmp_path))  # fmt: skip
    block = race_block(page, rid("TOK", 9))
    replaced = 0
    for code, rows in table.items():
        for combo, (_row, ladder, want) in rows.items():
            row_el = row_of(pool_block(block, code), combo)
            check_ladder_row(row_el, list(want), (code, combo))
            replaced += sum(w != lad for w, lad in zip(want, ladder, strict=True))
    assert replaced >= 7, replaced  # the fixture has stale points on both sides of the cutoff


def _write_logged_day(captures: dict[datetime, dict[str, float]], post: datetime) -> None:
    """One TOK race (5R) of synthetic WIN captures, and its post time, under the data dir."""
    from horsetrack.core import paths
    from horsetrack.countries.japan import odds_log

    log = paths.odds_snapshot_path("jra", DAY, "TOK")
    for at, prices in captures.items():
        stamp = utc_text(at)
        rows = [
            {
                "snapshot_utc": stamp,
                "race_id": rid("TOK", 5),
                "pool": "WIN",
                "combination": combo,
                "odds": odds,
                "status_raw": f"{at.hour:02d}時{at.minute:02d}分現在",
                "source_update_utc": stamp,
            }
            for combo, odds in prices.items()
        ]
        odds_log.append_capture(log, rows)
    odds_log.write_post_times(
        log.with_name("TOK.races.json"), {5: post.astimezone(UTC)}, jst(6, 0).astimezone(UTC)
    )


def _drift_nowcast():
    """A fitted v0 nowcast with mu = 0.02 * tau in every cell (so mu(7) - mu(5) = 0.04)."""
    from horsetrack.countries.japan.board.snapshot import load_decision_grid
    from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0

    grid, bands = load_decision_grid(), (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    pools = ("WIN", "PLACE", "QUINELLA_PLACE")
    sigma0 = {(p, t): 0.1 for p in pools for t in grid.minutes}
    cells = {
        (p, t, b): DriftCell(p, t, b, 30, 0.02 * t, 0.1)
        for p in pools
        for t in grid.minutes
        for b in range(len(bands) + 1)
    }
    return NowcastV0(bands, grid, 50.0, ("2026-11-14",), sigma0, cells, "drift-e2e")


@needs_node
def test_a_logged_day_shows_the_ladder(tmp_path, monkeypatch):
    """Synthetic WIN captures through build_board, build_shadow (a known nowcast) and the page:
    read at 10:45 (T-15) only T-20 is observed; read at 10:53 (T-7) T-20 and T-10 are, and T-5,
    T-2 and the final are projected: O_now * exp(0.02 * (7 - k)), shown as '≈' with one decimal."""
    from horsetrack.core import paths
    from horsetrack.countries.japan.board import feed as feed_mod
    from horsetrack.countries.japan.board.flags import FlagRules
    from horsetrack.countries.japan.board.shadow_feed import build_shadow
    from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels

    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    post = jst(11, 0)  # T-20 10:40, T-10 10:50, T-5 10:55
    captures = {
        jst(10, 30): {"1": 4.0, "2": 2.5, "3": 9.0},
        jst(10, 40): {"1": 5.0, "2": 2.2, "3": 9.0},  # exactly T-20
        jst(10, 45): {"1": 4.8, "2": 2.3, "3": 10.4},
        jst(10, 50): {"1": 4.6, "2": 2.4, "3": 11.0},  # exactly T-10
        jst(10, 53): {"1": 4.2, "2": 2.6, "3": 12.0},
    }
    _write_logged_day(captures, post)
    models = ShadowModels(
        win=FakeWin(), ordering=FakeOrdering(), nowcast=_drift_nowcast(), ids={"nowcast": "e2e"}
    )
    rules = FlagRules(floor_max_odds=1.3, tail_min_odds=50.0, tail_rank=6, stale_s=40.0)
    day = datetime.fromisoformat(DAY).date()
    steps, reads = [], (jst(10, 45), jst(10, 53))
    for at in reads:
        plain = json.loads(json.dumps(feed_mod.build_board(day, now=lambda at=at: at)))
        shadow = json.loads(
            json.dumps(
                build_shadow(
                    day, now=lambda at=at: at, models=models, flag_rules=rules, n_draws=8, seed=1
                )
            )
        )
        steps.append(call("renderShadow", plain, shadow, {"now": ms(at), "venue": "TOK"}))
    early, late = ({"TOK": soup(p)} for p in results(steps, tmp_path))

    def shown(page, combo):
        row_el = row_in(page, "TOK", rid("TOK", 5), "WIN", combo)
        return [(text(cell(row_el, c)), classes(cell(row_el, c))) for c in LADDER_COLS]

    def near(got: str, want: float) -> bool:
        m = re.fullmatch(r"≈\s*(\d+\.\d)", got)
        return m is not None and abs(float(m.group(1)) - want) <= 0.051

    for combo in ("1", "2", "3"):
        t20, t10 = captures[jst(10, 40)][combo], captures[jst(10, 50)][combo]
        # 10:53, tau 7: T-20 and T-10 observed, the T-10 cell coloured by its move since T-20
        cells = shown(late, combo)
        assert cells[0][0] == f"{t20:.1f}" and "projected" not in cells[0][1], (combo, cells)
        assert cells[1][0] == f"{t10:.1f}" and "projected" not in cells[1][1], (combo, cells)
        assert ("up" if t10 > t20 else "down") in cells[1][1], (combo, cells)
        now_odds = captures[jst(10, 53)][combo]
        for (got, cls), k in zip(cells[2:], (5, 2, 0), strict=True):
            want = now_odds * math.exp(0.02 * (7 - k))
            assert near(got, want) and "projected" in cls, (combo, k, got, want)
        # 10:45, tau 15: T-20 observed, the rest projected from the 10:45 odds
        cells = shown(early, combo)
        assert cells[0][0] == f"{t20:.1f}" and "projected" not in cells[0][1], (combo, cells)
        now_odds = captures[jst(10, 45)][combo]
        for (got, cls), k in zip(cells[1:], (10, 5, 2, 0), strict=True):
            want = now_odds * math.exp(0.02 * (15 - k))
            assert near(got, want) and "projected" in cls, (combo, k, got, want)


# A Win pool whose crowd order, feed order and shadow-value order all differ, with a tie.
ORDER_FEED = {
    "race_date": DAY,
    "generated_utc": utc_text(NOW),
    "venues": [
        {
            "venue": "TOK",
            "races": [
                race(
                    "TOK",
                    7,
                    jst(11, 0),
                    NOW,
                    {
                        "WIN": pool(
                            line("1", 12.0, mm=1.0, t20=1.0, t2=1.0),
                            line("2", 3.3, mm=1.0, t20=1.0, t2=1.0),
                            line("3", None, scr=True),
                            line("4", 1.7, mm=1.0, t20=1.0, t2=1.0),
                            line("10", 3.3, mm=1.0, t20=1.0, t2=1.0),
                        ),
                    },
                )
            ],
        }
    ],
}
ORDER_SHADOW = {
    "race_date": DAY,
    "generated_utc": utc_text(NOW),
    "models": {"WIN": "v0-synthetic"},
    "races": [
        shadow_race(
            "TOK",
            7,
            jst(11, 0),
            5,
            {
                "WIN": {
                    "1": win(1.4, 1.3, 1.5),
                    "2": win(0.6, 0.5, 0.7),
                    "3": win(None, None, None, "scratched"),
                    "4": win(0.9, 0.8, 1.0),
                    "10": win(1.1, 1.0, 1.2),
                }
            },
        )
    ],
}


@needs_node
def test_rows_run_favourite_first_as_on_the_board(tmp_path):
    keys = race_keys(FEED)
    steps = []
    for v, r in keys:
        opts = {"now": ms(NOW), "venue": v, "race": r}
        steps.append(call("renderBoard", FEED, opts, mod="board"))
        steps.append(call("renderShadow", FEED, SHADOW, opts))
    opts = {"now": ms(NOW), "venue": "TOK"}
    steps.append(call("renderShadow", ORDER_FEED, ORDER_SHADOW, opts))
    shown = results(steps, tmp_path)
    for i, (_, race_id) in enumerate(keys):
        board, page = soup(shown[2 * i]), soup(shown[2 * i + 1])
        assert order_of(page) == order_of(board), race_id
        # independently: the board's pool order, rows favourite first by the crowd's odds
        race_ = next(r for v in FEED["venues"] for r in v["races"] if r["race_id"] == race_id)
        rank = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"]
        want = [
            (race_id, pool_code, combo)
            for pool_code in sorted(race_["pools"], key=rank.index)
            for combo in favourite_order(race_["pools"][pool_code]["rows"])
        ]
        assert order_of(page) == want, race_id
        for block in race_blocks(page):
            for pool_el in block.select("[data-pool]"):
                values = pool_el.select('[data-col="value"]')
                figures = [v for v in values if text(v) != "-"]
                assert len({tuple(classes(v)) for v in figures}) <= 1, "value cells styled by value"
                assert not [v for v in values if v.get("style")], "inline style on a value cell"
                assert not [r for r in pool_el.select("[data-combination]") if r.get("style")]
    tok1 = soup(shown[2 * keys.index(("TOK", rid("TOK", 1))) + 1])
    assert not tok1.select('[data-combination="99"]'), "a shadow-only row was drawn"
    assert not tok1.select('[data-combination="3,9"]'), "a shadow-only row was drawn"
    # by the crowd's odds, ties by number ('2' before '10'), never by the shadow value
    hand = pool_block(race_block(soup(shown[-1]), rid("TOK", 7)), "WIN")
    combos = [e["data-combination"] for e in hand.select("[data-combination]")]
    assert combos == ["4", "2", "10", "1", "3"], combos


@needs_node
def test_the_hash_and_the_strip_pick_the_race_and_a_poll_keeps_it(tmp_path):
    k3, k10, t1 = rid("KYO", 3), rid("KYO", 10), rid("TOK", 1)
    first = soup(results([call("renderShadow", FEED, SHADOW, {"now": ms(NOW)})], tmp_path)[0])
    assert [b["data-race"] for b in race_blocks(first)] == [k3], "KYO 3 posts next at NOW"
    links = strip_links(first, race_blocks(first))
    assert set(links) == {3, 10}
    tok_tab = first.select_one('[data-venue="TOK"]')
    assert tok_tab is not None

    newer = copy.deepcopy(SHADOW)
    newer["races"][1]["pools"]["WIN"]["2"] = win(1.37, 1.29, 1.46, "tail")
    plain = {"plain": {"feed": FEED, "etag": '"p1"'}}
    one = {**plain, "shadow": {"feed": SHADOW, "etag": '"s1"'}}
    two = {**plain, "shadow": {"feed": newer, "etag": '"s2"'}}
    steps = [
        {"fire": 5000, "servers": two},  # a poll with a new shadow feed
        {"click": chain_of(click_target(links[3]), "shadow")},  # 3R on the strip
        {"fire": 5000},
        {"click": chain_of(tok_tab, "shadow")},  # the TOK tab
        {"fire": 5000},
    ]
    out = run_js(steps, tmp_path, servers=one, browser=True, hash_=links[10]["href"])
    got = [[b["data-race"] for b in race_blocks(soup(r["shadow"]))] for r in out["results"]]
    assert got == [[k10], [k10], [k3], [k3], [t1], [t1]], got  # loaded with KYO 10's hash
    assert out["results"][2]["hash"] == links[3]["href"], out["results"][2]["hash"]


@needs_node
def test_a_newly_selected_race_moves_the_viewport_and_a_poll_leaves_it(tmp_path):
    # The reader is page_fakes.READ_Y px down before every step. Picking another race (strip
    # link, back) brings it to the top of the viewport; a poll, new values or not, leaves it.
    k3, k10, t1 = rid("KYO", 3), rid("KYO", 10), rid("TOK", 1)
    first = soup(results([call("renderShadow", FEED, SHADOW, {"now": ms(NOW)})], tmp_path)[0])
    links = strip_links(first, race_blocks(first))
    to10 = links[10]["href"]
    tok_tab = first.select_one('[data-venue="TOK"]')
    newer = copy.deepcopy(SHADOW)
    newer["races"][1]["pools"]["WIN"]["2"] = win(1.37, 1.29, 1.46, "tail")
    plain = {"plain": {"feed": FEED, "etag": '"p1"'}}
    one = {**plain, "shadow": {"feed": SHADOW, "etag": '"s1"'}}
    two = {**plain, "shadow": {"feed": newer, "etag": '"s2"'}}
    # (step, race drawn after it, the viewport moves to the race)
    table = [
        ({"fire": 5000}, k10, False),  # 304s
        ({"fire": 5000, "servers": two}, k10, False),  # new values, same race
        ({"click": chain_of(click_target(links[3]), "shadow")}, k3, True),  # 3R on the strip
        ({"fire": 5000, "servers": one}, k3, False),
        ({"hash": to10}, k10, True),  # back
        ({"fire": 5000}, k10, False),
        ({"click": chain_of(tok_tab, "shadow")}, t1, True),  # the TOK tab
        ({"fire": 5000, "servers": two}, t1, False),
    ]
    out = run_js([step for step, _, _ in table], tmp_path, servers=one, browser=True, hash_=to10)
    steps_out = out["results"][1:]  # results[0] is the page at load
    got = [[b["data-race"] for b in race_blocks(soup(r["shadow"]))] for r in steps_out]
    assert got == [[race] for _, race, _ in table], got
    for i, ((step, race, moves), result) in enumerate(zip(table, steps_out, strict=True)):
        where = (i, sorted(step)[0], race, result["viewport"])
        if moves:
            assert moved_to_race(result["viewport"]), ("left deep in the new race", where)
        else:
            assert kept_in_place(result["viewport"]), ("a poll moved the reader", where)


@needs_node
def test_a_poll_keeps_each_pools_horizontal_scroll(tmp_path):
    # The reader scrolls each pool's table sideways (the [data-pool] element scrolls, as
    # shadow.html's .pool overflow-x: auto); a poll redrawing the same race, 304s or new values,
    # leaves every pool where the reader put it.
    newer = copy.deepcopy(SHADOW)
    newer["races"][1]["pools"]["WIN"]["2"] = win(1.37, 1.29, 1.46, "tail")
    plain = {"plain": {"feed": FEED, "etag": '"p1"'}}
    one = {**plain, "shadow": {"feed": SHADOW, "etag": '"s1"'}}
    two = {**plain, "shadow": {"feed": newer, "etag": '"s2"'}}
    first = {"WIN": 140, "PLACE": 0, "QUINELLA_PLACE": 75}
    second = {"WIN": 20, "PLACE": 210, "QUINELLA_PLACE": 75}
    steps = [
        {"scroll": first},
        {"fire": 5000},  # 304s: the same race drawn again
        {"scroll": second},
        {"fire": 5000, "servers": two},  # new values
        {"fire": 5000},
    ]
    out = run_js(steps, tmp_path, servers=one, browser=True, live_dom=True)
    loaded, scrolled, poll_304, rescrolled, poll_new, poll_again = out["results"]
    k3 = rid("KYO", 3)
    for r in out["results"]:
        assert [b["data-race"] for b in race_blocks(soup(r["shadow"]))] == [k3]
    assert loaded["pools"] == {"WIN": 0, "PLACE": 0, "QUINELLA_PLACE": 0}, loaded["pools"]
    assert scrolled["pools"] == first and rescrolled["pools"] == second  # the fake holds them
    assert poll_304["pools"] == first, poll_304["pools"]
    assert poll_new["pools"] == second, poll_new["pools"]
    assert poll_again["pools"] == second, poll_again["pools"]
    # the new values were drawn: the poll did redraw
    row2 = row_in({"KYO": soup(poll_new["shadow"])}, "KYO", k3, "WIN", "2")
    check_value(text(cell(row2, "value")), "WIN", newer["races"][1]["pools"]["WIN"]["2"], "new")


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
