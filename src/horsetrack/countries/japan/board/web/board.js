/* The board page (#42): renders the live-board feed (docs/specs/board-feed.md) and polls it.
 *
 * Pure render functions first (exported for node), then the browser glue: poll `feed` every 5 s
 * with If-None-Match, redraw #board, switch venue tabs, keep the race strip on the next race.
 * Feed text is untrusted: every feed string goes through esc() before it reaches the markup.
 * The shadow page (#122) reuses renderBoard with `opts.columns` (extra cells per row) and finds
 * these functions on window.HorseBoard in the browser.
 * The page shows prices and their moves only.
 */
(function () {
  "use strict";

  var STALE_MS = 120 * 1000;
  var POLL_MS = 5000;
  var JST_MS = 9 * 3600 * 1000; // Japan keeps no daylight saving time

  var VENUES = {
    SAP: "Sapporo", HAK: "Hakodate", FKS: "Fukushima", NII: "Niigata", TOK: "Tokyo",
    NAK: "Nakayama", CKY: "Chukyo", KYO: "Kyoto", HSN: "Hanshin", KOK: "Kokura",
  };
  var POOLS = [
    ["WIN", "Win"], ["PLACE", "Place"], ["BRACKET_QUINELLA", "Bracket quinella"],
    ["QUINELLA", "Quinella"], ["QUINELLA_PLACE", "Wide"], ["FORECAST", "Exacta"],
    ["EXACTA", "Exacta"], ["TRIO", "Trio"], ["TIERCE", "Trifecta"], ["TRIFECTA", "Trifecta"],
  ];
  var POOL_LABEL = {}, POOL_RANK = {};
  POOLS.forEach(function (p, i) { POOL_LABEL[p[0]] = p[1]; POOL_RANK[p[0]] = i; });

  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function num(value) {
    if (value === null || value === undefined || value === "") return null;
    var n = Number(value);
    return isFinite(n) ? n : null;
  }

  function instantMs(text) {
    if (text === null || text === undefined) return null;
    var t = Date.parse(String(text));
    return isFinite(t) ? t : null;
  }

  /* A move in percent as signed text with one decimal; '-' when there is none. */
  function formatMove(pct) {
    var n = num(pct);
    if (n === null) return "-";
    var shown = Math.abs(n).toFixed(1);
    if (shown === "0.0") return "0.0%";
    return (n > 0 ? "+" : "-") + shown + "%";
  }

  function moveClass(pct) {
    var n = num(pct);
    if (n === null || n === 0) return "";
    return n > 0 ? "up" : "down";
  }

  /* True when the capture is more than 2 minutes older than nowMs (no readable capture: stale). */
  function isStale(lastCaptureUtc, nowMs) {
    var t = instantMs(lastCaptureUtc);
    if (t === null) return true;
    return Number(nowMs) - t > STALE_MS;
  }

  function two(n) { return (n < 10 ? "0" : "") + n; }

  function jstClock(ms) {
    var d = new Date(ms + JST_MS);
    return two(d.getUTCHours()) + ":" + two(d.getUTCMinutes());
  }

  function yen(value) {
    var n = num(value);
    if (n === null) return null;
    return "¥" + String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function untilPost(postMs, nowMs) {
    var diff = postMs - nowMs;
    if (diff > 0) return Math.ceil(diff / 60000) + " min to post";
    var ago = Math.floor(-diff / 60000);
    return ago < 1 ? "at post" : "post " + ago + " min ago";
  }

  function captureAge(lastMs, nowMs) {
    if (lastMs === null) return "no capture";
    var s = Math.round((nowMs - lastMs) / 1000);
    if (s < 5) return "updated just now";
    if (s < 120) return "updated " + s + " s ago";
    return "updated " + Math.floor(s / 60) + " min ago";
  }

  function venueName(code) { return VENUES[code] || code; }

  function poolLabel(code) {
    if (POOL_LABEL[code]) return POOL_LABEL[code];
    var words = String(code).toLowerCase().replace(/_/g, " ");
    return words.charAt(0).toUpperCase() + words.slice(1);
  }

  function poolOrder(a, b) {
    var ra = a in POOL_RANK ? POOL_RANK[a] : POOLS.length;
    var rb = b in POOL_RANK ? POOL_RANK[b] : POOLS.length;
    return ra - rb || (a < b ? -1 : a > b ? 1 : 0);
  }

  function anchor(raceId) { return "race-" + String(raceId).replace(/[^A-Za-z0-9_-]/g, "-"); }

  function raceTitle(venue, race) {
    var no = num(race.race_no);
    return esc(venueName(venue)) + " " + (no === null ? esc(race.race_id) : no + "R");
  }

  /* The race to post next: the first race, in post-time order, whose post is still ahead. */
  function nextRaceId(races, nowMs) {
    var best = null, bestMs = Infinity;
    races.forEach(function (r) {
      var t = instantMs(r.post_utc);
      if (t !== null && t > nowMs && t < bestMs) { best = r.race_id; bestMs = t; }
    });
    return best;
  }

  function oddsText(line) {
    var lo = num(line.odds), hi = num(line.odds_hi);
    if (line.status === "SCR") return "Scratched";
    if (lo === null) return "-";
    return hi === null ? lo.toFixed(1) : lo.toFixed(1) + "–" + hi.toFixed(1);
  }

  function moveCell(col, pct) {
    var cls = moveClass(pct);
    return '<td data-col="' + col + '" class="num mv' + (cls ? " " + cls : "") + '">' +
      esc(formatMove(pct)) + "</td>";
  }

  /* columns (optional): {head: extra <th> markup, cells(raceId, code, line): extra <td> markup}. */
  function renderPool(code, body, raceId, columns) {
    var rows = (body && body.rows) || [];
    var multi = rows.some(function (l) { return String(l.combination).indexOf(",") >= 0; });
    var money = yen(body && body.investment);
    var out = '<section class="pool" data-pool="' + esc(code) + '">' +
      '<div class="pool-head"><h3>' + esc(poolLabel(code)) + "</h3>" +
      '<span class="muted small mono">' + (money ? "Pool " + esc(money) : "Pool money not shown") +
      "</span></div>" +
      '<table><thead><tr><th scope="col">' + (multi ? "Horses" : "Horse") + "</th>" +
      '<th scope="col" class="num">Odds</th><th scope="col" class="num">Since morning</th>' +
      '<th scope="col" class="num">Since T-20</th>' + (columns ? columns.head : "") +
      "</tr></thead><tbody>";
    rows.forEach(function (line) {
      var scr = line.status === "SCR";
      out += "<tr" + (scr ? ' class="scratched"' : "") + ' data-combination="' +
        esc(line.combination) + '"><th scope="row" class="combo mono">' +
        esc(String(line.combination).split(",").join("-")) + "</th>" +
        '<td data-col="odds" class="num">' + esc(oddsText(line)) + "</td>" +
        moveCell("move-morning", line.move_morning_pct) +
        moveCell("move-t20", line.move_t20_pct) +
        (columns ? columns.cells(raceId, code, line) : "") + "</tr>";
    });
    return out + "</tbody></table></section>";
  }

  function renderRace(venue, race, nowMs, nextId, columns) {
    var postMs = instantMs(race.post_utc);
    var lastMs = instantMs(race.last_capture_utc);
    var stale = isStale(race.last_capture_utc, nowMs);
    var pools = Object.keys(race.pools || {}).sort(poolOrder);
    var cls = "race" + (race.race_id === nextId ? " next" : "") +
      (postMs !== null && postMs <= nowMs ? " off" : "");
    var out = '<article class="' + cls + '" id="' + esc(anchor(race.race_id)) +
      '" data-race="' + esc(race.race_id) + '"><header class="race-head">' +
      "<h2>" + raceTitle(venue, race) + "</h2><p class=\"meta mono\">";
    if (postMs === null) out += "<span>No post time</span>";
    else {
      out += "<span>Post " + jstClock(postMs) + " JST</span><span>" +
        esc(untilPost(postMs, nowMs)) + "</span>";
    }
    out += '<span class="chip ' + (stale ? "stale" : "live") + '">' +
      (stale ? "stale · " : "") + esc(captureAge(lastMs, nowMs)) + "</span></p></header>";
    pools.forEach(function (code) {
      out += renderPool(code, race.pools[code], race.race_id, columns);
    });
    if (!pools.length) out += '<p class="muted">No odds captured yet.</p>';
    return out + "</article>";
  }

  /* The HTML of the board for one venue of a feed (the feed's first venue by default). */
  function renderBoard(feed, opts) {
    opts = opts || {};
    var nowMs = num(opts.now);
    if (nowMs === null) nowMs = Date.now();
    var venues = (feed && feed.venues) || [];
    var shown = null;
    venues.forEach(function (v) { if (!shown && v.venue === opts.venue) shown = v; });
    if (!shown) shown = venues[0] || null;
    if (!shown) return '<p class="empty muted">No odds logged for this day yet.</p>';

    var races = shown.races || [];
    var nextId = nextRaceId(races, nowMs);
    var out = '<nav class="tabs" aria-label="Venues">';
    venues.forEach(function (v) {
      out += '<button type="button" class="tab" data-venue="' + esc(v.venue) +
        '" aria-pressed="' + (v === shown ? "true" : "false") + '">' +
        esc(venueName(v.venue)) + "</button>";
    });
    out += '</nav><nav class="strip" aria-label="Races">';
    races.forEach(function (r) {
      var postMs = instantMs(r.post_utc), no = num(r.race_no);
      var cls = "rbtn" + (r.race_id === nextId ? " next" : "") +
        (postMs !== null && postMs <= nowMs ? " off" : "");
      out += '<a class="' + cls + '" href="#' + esc(anchor(r.race_id)) + '"' +
        (r.race_id === nextId ? ' aria-current="true"' : "") + "><b>" +
        (no === null ? esc(r.race_id) : no + "R") + "</b><span>" +
        (postMs === null ? "--:--" : jstClock(postMs)) + "</span><i>" +
        (r.race_id === nextId ? "next" : postMs !== null && postMs <= nowMs ? "off" : "&nbsp;") +
        "</i></a>";
    });
    out += '</nav><div class="races">';
    races.forEach(function (r) {
      out += renderRace(shown.venue, r, nowMs, nextId, opts.columns);
    });
    if (!races.length) out += '<p class="empty muted">No races captured at this venue yet.</p>';
    return out + "</div>";
  }

  /* ------------------------------------------------------------ the browser glue */

  var state = { etag: null, feed: null, venue: null };

  function setStatus(text, cls) {
    if (typeof document === "undefined") return;
    var el = document.getElementById("status");
    if (!el) return;
    el.textContent = text;
    el.className = "chip" + (cls ? " " + cls : "");
  }

  function followNext() {
    var strip = document.querySelector("#board .strip");
    var next = document.querySelector("#board .strip .next");
    if (!strip || !next || typeof next.offsetLeft !== "number") return;
    strip.scrollLeft = Math.max(0, next.offsetLeft - strip.clientWidth / 2 + next.offsetWidth / 2);
  }

  function draw() {
    var board = document.getElementById("board");
    if (!board || !state.feed) return;
    board.innerHTML = renderBoard(state.feed, { now: Date.now(), venue: state.venue });
    followNext();
  }

  /* One GET of the feed with If-None-Match; a new feed redraws, a 304 keeps it (times refresh). */
  function poll() {
    var headers = {};
    if (state.etag) headers["If-None-Match"] = state.etag;
    return fetch("feed", { headers: headers, cache: "no-store" })
      .then(function (res) {
        if (res.status === 304) { draw(); setStatus("live", "live"); return; }
        if (res.status !== 200) {
          setStatus(res.status === 503 ? "waiting for the first capture" : "feed error " + res.status,
            "stale");
          draw();
          return;
        }
        return res.json().then(function (feed) {
          state.feed = feed;
          state.etag = (res.headers && res.headers.get("ETag")) || null;
          draw();
          setStatus("live", "live");
        });
      })
      .catch(function () { setStatus("no connection", "stale"); });
  }

  function onClick(event) {
    var target = event && event.target;
    var tab = target && target.closest ? target.closest("[data-venue]") : null;
    if (!tab) return;
    state.venue = tab.getAttribute("data-venue");
    draw();
    if (typeof window !== "undefined" && window.scrollTo) window.scrollTo(0, 0);
  }

  function start() {
    var board = document.getElementById("board");
    if (board) board.addEventListener("click", onClick);
    (function tick() {
      poll().then(function () { setTimeout(tick, POLL_MS); });
    })();
  }

  var api = {
    formatMove: formatMove, moveClass: moveClass, isStale: isStale, renderBoard: renderBoard,
    poll: poll, esc: esc, num: num,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.HorseBoard = api;
  // Only a page with #board polls here; the shadow page draws #shadow with shadow.js.
  if (typeof document !== "undefined" && typeof fetch === "function" &&
      document.getElementById("board")) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
  }
})();
