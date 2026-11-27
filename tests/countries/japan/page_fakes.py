"""Shared pieces of the board and shadow page tests (ticket #498): a browser-like URL hash and
click for the node harnesses, and the expected favourite-first row order, derived here.

`BROWSER_FAKES` is JavaScript pasted into a harness after its `el(id)` and `docHandlers`; it reads
`input.now` (a fixed clock in ms, for `Date.now()` and `new Date()`) and `input.hash` (the URL
hash at load) and defines:
- `location` (`hash`, `href`; setting a new hash fires `hashchange` asynchronously, as a browser
  does) and `history` (`pushState` / `replaceState` change the hash without an event);
- `fire(type)`: calls the window/document listeners of `type` (and `on<type>`);
- `clickChain(chain, rootId)`: a click on the element described by `chain` (target first, then its
  ancestors, as `chain_of` builds it) delivered to the listeners of `#rootId`, the document and
  the window; when no listener calls `preventDefault`, the default action of a link follows:
  `location.href = <its href>`. The fake elements answer `closest`, `matches`, `getAttribute`,
  `dataset` and `classList.contains` for simple selectors.
- a vertical viewport: `scrollTo` / `scroll` / `scrollBy` (x, y or {top}), `scrollY`,
  `pageYOffset` and the `scrollTop` of `document.documentElement` and `document.body` share one
  offset; an element's `scrollIntoView(arg)` is noted through `noteIntoView(id, arg)`.
  `resetViewport()` puts the reader `READ_Y` px down the page (deep in a race's tables) with no
  note; `viewportState()` returns `{y, intoView: [{id, block}]}`. A harness calls
  `resetViewport()` before each step and must not overwrite `scrollTo` after these fakes.
"""

from __future__ import annotations

import re

READ_Y = 2400  # px down the page: the reader at the bottom of a race's tables


def moved_to_race(viewport: dict) -> bool:
    """The step moved the viewport to the race drawn: the page top, or a scrollIntoView that
    aligns an element (a `block: 'nearest'` nudge, which leaves a visible element alone, does
    not count)."""
    aligned = [n for n in viewport["intoView"] if n["block"] != "nearest"]
    return viewport["y"] == 0 or bool(aligned)


def kept_in_place(viewport: dict) -> bool:
    """The step left the reader where they were: no scroll away from READ_Y, no alignment."""
    aligned = [n for n in viewport["intoView"] if n["block"] != "nearest"]
    return viewport["y"] == READ_Y and not aligned


BROWSER_FAKES = r"""
if (input.now !== null && input.now !== undefined) {
  const RealDate = Date, fixed = Number(input.now);
  global.Date = class extends RealDate {
    constructor(...a) { if (a.length) super(...a); else super(fixed); }
    static now() { return fixed; }
  };
}
const ORIGIN = "http://127.0.0.1:8765";
let urlHash = "";
const normHash = (v) => {
  v = String(v === null || v === undefined ? "" : v);
  if (v === "" || v === "#") return "";
  return v[0] === "#" ? v : "#" + v;
};
function fire(type) {
  for (const fn of (docHandlers[type] || []).slice()) fn({ type, newURL: ORIGIN + "/" + urlHash });
  const prop = global["on" + type];
  if (typeof prop === "function") prop({ type });
}
global.location = {
  get hash() { return urlHash; },
  set hash(v) {
    const n = normHash(v);
    if (n === urlHash) return;
    urlHash = n;
    setImmediate(() => fire("hashchange"));
  },
  get href() { return ORIGIN + "/" + urlHash; },
  set href(v) { const s = String(v), i = s.indexOf("#"); this.hash = i >= 0 ? s.slice(i) : ""; },
  pathname: "/", search: "", origin: ORIGIN, host: "127.0.0.1:8765", protocol: "http:",
  assign(v) { this.href = v; }, replace(v) { this.href = v; }, reload() {},
  toString() { return this.href; },
};
urlHash = normHash(input.hash);
const quietHash = (url) => {
  if (url === undefined || url === null) return;
  const s = String(url), i = s.indexOf("#");
  urlHash = i >= 0 ? normHash(s.slice(i)) : "";
};
global.history = {
  state: null, length: 1,
  pushState(st, t, url) { this.state = st; quietHash(url); },
  replaceState(st, t, url) { this.state = st; quietHash(url); },
  back() {}, forward() {}, go() {},
};
document.location = global.location;

const viewport = { y: 0, intoView: [] };
const setY = (v) => {
  const n = Number(v);
  viewport.y = Number.isFinite(n) ? Math.max(0, n) : 0;
};
const scrollTarget = (a, b, base) => {
  if (a && typeof a === "object") return "top" in a ? base + Number(a.top) : null;
  return b === undefined ? null : base + Number(b);
};
global.scrollTo = global.scroll = (a, b) => {
  const y = scrollTarget(a, b, 0);
  if (y !== null) setY(y);
};
global.scrollBy = (a, b) => {
  const y = scrollTarget(a, b, viewport.y);
  if (y !== null) setY(y);
};
for (const name of ["scrollY", "pageYOffset"]) {
  Object.defineProperty(global, name, { get: () => viewport.y, set: setY, configurable: true });
}
for (const node of [document.documentElement, document.body]) {
  Object.defineProperty(node, "scrollTop", {
    get: () => viewport.y, set: setY, configurable: true,
  });
}
function noteIntoView(id, arg) {
  const block = arg && typeof arg === "object" ? arg.block || "start"
    : arg === false ? "end" : "start";
  viewport.intoView.push({ id: String(id || ""), block });
}
function resetViewport() { setY(/*READ_Y*/); viewport.intoView = []; }
function viewportState() { return { y: viewport.y, intoView: viewport.intoView.slice() }; }

function compoundMatches(e, c) {
  if (!e) return false;
  const re = new RegExp([
    String.raw`:not\(([^)]*)\)`, String.raw`\*`, String.raw`([a-zA-Z][\w-]*)`,
    String.raw`#([\w-]+)`, String.raw`\.([\w-]+)`,
    String.raw`\[\s*([\w-]+)\s*(?:([~^$*|]?=)\s*(?:"([^"]*)"|'([^']*)'|([^\]\s]*)))?\s*\]`,
    String.raw`(:[\w-]+(?:\([^)]*\))?)`,
  ].join("|"), "y");
  let pos = 0;
  while (pos < c.length) {
    re.lastIndex = pos;
    const m = re.exec(c);
    if (!m) return false;
    pos = re.lastIndex;
    if (m[1] !== undefined) { if (compoundMatches(e, m[1].trim())) return false; continue; }
    if (m[0] === "*") continue;
    if (m[2] !== undefined) { if (e.tagName !== m[2].toUpperCase()) return false; continue; }
    if (m[3] !== undefined) { if (e.getAttribute("id") !== m[3]) return false; continue; }
    if (m[4] !== undefined) { if (!e.classList.contains(m[4])) return false; continue; }
    if (m[5] !== undefined) {
      const have = e.getAttribute(m[5]);
      if (have === null) return false;
      if (m[6] === undefined) continue;
      const want = m[7] !== undefined ? m[7] : m[8] !== undefined ? m[8] : m[9];
      const ok = { "=": have === want, "^=": have.startsWith(want), "$=": have.endsWith(want),
        "*=": have.includes(want), "~=": have.split(/\s+/).includes(want),
        "|=": have === want || have.startsWith(want + "-") }[m[6]];
      if (!ok) return false;
      continue;
    }
    return false; // an unsupported pseudo-class
  }
  return true;
}
function selectorMatches(e, sel) {
  return String(sel).split(",").some((part) => {
    const compounds = part.trim().split(/\s*>\s*|\s+/).filter(Boolean);
    if (!compounds.length || !compoundMatches(e, compounds[compounds.length - 1])) return false;
    let at = e.parentElement;
    for (let i = compounds.length - 2; i >= 0; i--) {
      while (at && !compoundMatches(at, compounds[i])) at = at.parentElement;
      if (!at) return false;
      at = at.parentElement;
    }
    return true;
  });
}
function chainElements(chain) {
  const nodes = chain.map((n) => {
    const attrs = Object.assign({}, n.attrs || {});
    const dataset = {};
    for (const k of Object.keys(attrs)) {
      if (k.startsWith("data-")) {
        dataset[k.slice(5).replace(/-([a-z])/g, (_, ch) => ch.toUpperCase())] = attrs[k];
      }
    }
    const cls = () => String(attrs.class || "").split(/\s+/).filter(Boolean);
    const e = {
      tagName: String(n.tag).toUpperCase(), nodeName: String(n.tag).toUpperCase(), nodeType: 1,
      id: attrs.id || "", className: attrs.class || "", textContent: n.text || "",
      innerText: n.text || "", dataset, parentElement: null, parentNode: null,
      getAttribute(k) { return k in attrs ? attrs[k] : null; },
      hasAttribute(k) { return k in attrs; },
      classList: { contains: (c) => cls().includes(c), add() {}, remove() {}, toggle() {} },
      matches(sel) { return selectorMatches(this, sel); },
      closest(sel) {
        for (let at = this; at; at = at.parentElement) if (at.matches(sel)) return at;
        return null;
      },
      contains(other) {
        for (let at = other; at; at = at.parentElement) if (at === this) return true;
        return false;
      },
      addEventListener() {}, removeEventListener() {}, focus() {},
      scrollIntoView(arg) { noteIntoView(this.id, arg); },
    };
    if ("href" in attrs) {
      const raw = attrs.href, i = raw.indexOf("#");
      e.href = raw.startsWith("#") ? ORIGIN + "/" + raw : raw;
      e.hash = i >= 0 ? raw.slice(i) : "";
    }
    return e;
  });
  nodes.forEach((e, i) => { e.parentElement = e.parentNode = nodes[i + 1] || null; });
  return nodes;
}
function clickChain(chain, rootId) {
  const target = chainElements(chain)[0];
  const ev = {
    type: "click", target, currentTarget: null, button: 0, which: 1, detail: 1,
    metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, defaultPrevented: false,
    preventDefault() { this.defaultPrevented = true; }, stopPropagation() {},
    stopImmediatePropagation() {}, composedPath() { return []; },
  };
  const root = el(rootId);
  for (const fn of (root.listeners.click || []).slice()) {
    ev.currentTarget = root;
    fn.call(root, ev);
  }
  for (const fn of (docHandlers.click || []).slice()) { ev.currentTarget = document; fn(ev); }
  if (!ev.defaultPrevented) {
    const link = target.closest("a[href]");
    if (link) location.href = link.getAttribute("href");
  }
}
""".replace("/*READ_Y*/", str(READ_Y))


def chain_of(node, root_id: str) -> list[dict]:
    """The click chain of a BeautifulSoup element: itself and its ancestors in the rendered
    markup, then the page's container (`<div id=root_id>`, whose innerHTML the markup is)."""
    out = []
    for at in [node, *node.parents]:
        if at.name in (None, "[document]"):
            break
        attrs = {
            k: " ".join(v) if isinstance(v, list) else str(v) for k, v in (at.attrs or {}).items()
        }
        out.append({"tag": at.name, "attrs": attrs, "text": " ".join(at.get_text(" ").split())})
    out.append({"tag": "div", "attrs": {"id": root_id}, "text": ""})
    return out


def click_target(node):
    """What a finger lands on: the first child element of a link (its `<b>2R</b>`), else itself."""
    inner = node.find(True)
    return inner if inner is not None else node


def strip_links(page, race_blocks: list) -> dict[int, object]:
    """Race number -> the race strip's link to it: an `a` with an `href` starting '#', outside
    every race block, whose text starts with '<no>R'."""
    out: dict[int, object] = {}
    for link in page.select('a[href^="#"]'):
        if any(parent is block for parent in link.parents for block in race_blocks):
            continue
        m = re.match(r"\s*(\d+)R\b", " ".join(link.get_text(" ").split()))
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = link
    return out


def combination_key(combination) -> tuple:
    """Numeric order of a combination ('2' before '10', '2,3' before '2,10')."""
    try:
        return (0, tuple(int(part) for part in str(combination).split(",")), "")
    except ValueError:
        return (1, (), str(combination))


def favourite_order(rows: list[dict]) -> list[str]:
    """The combinations of a pool's feed rows favourite first: priced rows by current `odds`
    (the low end of a ranged pool) ascending, ties by combination; then the rows without odds
    (unpriced or scratched) by combination."""

    def priced(line: dict) -> bool:
        return line.get("odds") is not None and line.get("status") != "SCR"

    on = sorted(
        (r for r in rows if priced(r)),
        key=lambda r: (float(r["odds"]), combination_key(r["combination"])),
    )
    off = sorted(
        (r for r in rows if not priced(r)), key=lambda r: combination_key(r["combination"])
    )
    return [r["combination"] for r in on + off]
