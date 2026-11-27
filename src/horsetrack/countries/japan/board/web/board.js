/* The board page (#42): renders the live-board feed (docs/specs/board-feed.md) and polls it.
 *
 * Pure render functions first (exported for node), then the browser glue: poll `feed` every 5 s
 * with If-None-Match, redraw #board, switch venue tabs.
 * One race per page (#498): the race whose hash is in the URL (`#race-<race_id>`, so reload and
 * back keep it), else the venue's next race to post, else its last race. The race strip's links
 * set the hash; a venue tab puts that venue's default race in it; a poll redraws the same race.
 * Pool rows run favourite first (#498): by the crowd's current odds (the low end of Place and
 * Wide), ties in combination order, the rows without odds last; the feed itself is not reordered.
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

  /* The race a venue shows without a choice: the next to post, else the venue's last race. */
  function defaultRaceId(races, nowMs) {
    var next = nextRaceId(races, nowMs);
    if (next !== null || !races.length) return next;
    return races[races.length - 1].race_id;
  }

  /* Numeric combination parts ('2,10' -> [2, 10]), or null when a part is not a whole number. */
  function comboParts(combination) {
    var parts = String(combination).split(",");
    for (var i = 0; i < parts.length; i++) {
      if (!/^\s*\d+\s*$/.test(parts[i])) return null;
      parts[i] = Number(parts[i]);
    }
    return parts;
  }

  /* Combination order: '2' before '10', '2,3' before '2,10'; non-numeric ones after, as text. */
  function comboOrder(a, b) {
    var pa = comboParts(a), pb = comboParts(b);
    if (pa === null || pb === null) {
      if (pa !== pb) return pa === null ? 1 : -1;
      return String(a) < String(b) ? -1 : String(a) > String(b) ? 1 : 0;
    }
    for (var i = 0; i < Math.min(pa.length, pb.length); i++) {
      if (pa[i] !== pb[i]) return pa[i] - pb[i];
    }
    return pa.length - pb.length;
  }

  function rowOdds(line) { return line.status === "SCR" ? null : num(line.odds); }

  /* A copy of a pool's rows, favourite first: priced rows by odds, then the rest. */
  function favouriteFirst(rows) {
    return rows.slice().sort(function (a, b) {
      var oa = rowOdds(a), ob = rowOdds(b);
      if (oa === null || ob === null) {
        if (oa !== ob) return oa === null ? 1 : -1;
      } else if (oa !== ob) return oa - ob;
      return comboOrder(a.combination, b.combination);
    });
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
    favouriteFirst(rows).forEach(function (line) {
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

  function raceLabel(race) {
    var no = num(race.race_no);
    return no === null ? esc(race.race_id) : no + "R";
  }

  /* A previous / next race link beside the title; an empty placeholder at either end. */
  function stepLink(race, label, arrowFirst) {
    if (!race) return '<span class="step" aria-hidden="true"></span>';
    return '<a class="step mono" href="#' + esc(anchor(race.race_id)) + '" aria-label="' + label +
      '">' + (arrowFirst ? "‹ " + raceLabel(race) : raceLabel(race) + " ›") + "</a>";
  }

  function renderRace(venue, race, nowMs, nextId, columns, prev, next) {
    var postMs = instantMs(race.post_utc);
    var lastMs = instantMs(race.last_capture_utc);
    var stale = isStale(race.last_capture_utc, nowMs);
    var pools = Object.keys(race.pools || {}).sort(poolOrder);
    var cls = "race" + (race.race_id === nextId ? " next" : "") +
      (postMs !== null && postMs <= nowMs ? " off" : "");
    var out = '<article class="' + cls + '" id="' + esc(anchor(race.race_id)) +
      '" data-race="' + esc(race.race_id) + '"><header class="race-head">' +
      '<div class="race-title">' + stepLink(prev, "Previous race", true) +
      "<h2>" + raceTitle(venue, race) + "</h2>" + stepLink(next, "Next race", false) +
      '</div><p class="meta mono">';
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

  /* The venue of a feed named `code`, else the feed's first venue (null on an empty day). */
  function venueOf(feed, code) {
    var venues = (feed && feed.venues) || [];
    var shown = null;
    venues.forEach(function (v) { if (!shown && v.venue === code) shown = v; });
    return shown || venues[0] || null;
  }

  /* The HTML of the board for one race of one venue of a feed: opts.venue (the feed's first
   * venue by default) and opts.race when that race is of the venue, else the venue's default. */
  function renderBoard(feed, opts) {
    opts = opts || {};
    var nowMs = num(opts.now);
    if (nowMs === null) nowMs = Date.now();
    var venues = (feed && feed.venues) || [];
    var shown = venueOf(feed, opts.venue);
    if (!shown) return '<p class="empty muted">No odds logged for this day yet.</p>';

    var races = shown.races || [];
    var nextId = nextRaceId(races, nowMs);
    var at = -1;
    races.forEach(function (r, i) { if (at < 0 && r.race_id === opts.race) at = i; });
    if (at < 0) {
      var fallback = defaultRaceId(races, nowMs);
      races.forEach(function (r, i) { if (at < 0 && r.race_id === fallback) at = i; });
    }
    var selId = at < 0 ? null : races[at].race_id;
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
        (r.race_id === selId ? " sel" : "") + (postMs !== null && postMs <= nowMs ? " off" : "");
      out += '<a class="' + cls + '" href="#' + esc(anchor(r.race_id)) + '"' +
        (r.race_id === selId ? ' aria-current="page"' : "") + "><b>" +
        (no === null ? esc(r.race_id) : no + "R") + "</b><span>" +
        (postMs === null ? "--:--" : jstClock(postMs)) + "</span><i>" +
        (r.race_id === nextId ? "next" : postMs !== null && postMs <= nowMs ? "off" : "&nbsp;") +
        "</i></a>";
    });
    out += '</nav><div class="races">';
    if (at >= 0) {
      out += renderRace(shown.venue, races[at], nowMs, nextId, opts.columns,
        races[at - 1] || null, races[at + 1] || null);
    }
    if (!races.length) out += '<p class="empty muted">No races captured at this venue yet.</p>';
    return out + "</div>";
  }

  /* ------------------------------------------------------------ the browser glue */

  var state = { etag: null, feed: null, venue: null };

  /* The race the URL hash names in a feed, {venue, race}, or null. A hash naming another
   * venue's race switches to that venue, so a reload shows the race the page last showed. */
  function hashRace(feed) {
    var hash = typeof location !== "undefined" ? String(location.hash || "") : "";
    if (!hash) return null;
    var found = null;
    ((feed && feed.venues) || []).forEach(function (v) {
      (v.races || []).forEach(function (r) {
        if (!found && "#" + anchor(r.race_id) === hash) found = { venue: v.venue, race: r.race_id };
      });
    });
    return found;
  }

  /* The selection to draw: the hash's race (and its venue), else `venue`'s default race. */
  function selection(feed, venue) {
    return hashRace(feed) || { venue: venue, race: null };
  }

  /* A venue tab: put that venue's default race in the URL (no race: clear the hash). */
  function chooseVenue(feed, venue, nowMs) {
    var shown = venueOf(feed, venue);
    var id = shown ? defaultRaceId(shown.races || [], nowMs) : null;
    if (id !== null) location.hash = "#" + anchor(id);
    else if (typeof history !== "undefined" && history.replaceState) {
      history.replaceState(null, "", location.pathname + location.search);
    }
  }

  function setStatus(text, cls) {
    if (typeof document === "undefined") return;
    var el = document.getElementById("status");
    if (!el) return;
    el.textContent = text;
    el.className = "chip" + (cls ? " " + cls : "");
  }

  /* Scroll the race strip of `root` to the selected race. */
  function followSelected(root) {
    var strip = document.querySelector(root + " .strip");
    var sel = document.querySelector(root + " .strip .sel");
    if (!strip || !sel || typeof sel.offsetLeft !== "number") return;
    strip.scrollLeft = Math.max(0, sel.offsetLeft - strip.clientWidth / 2 + sel.offsetWidth / 2);
  }

  function draw() {
    var board = document.getElementById("board");
    if (!board || !state.feed) return;
    var pick = selection(state.feed, state.venue);
    state.venue = pick.venue;
    board.innerHTML = renderBoard(state.feed,
      { now: Date.now(), venue: pick.venue, race: pick.race });
    followSelected("#board");
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
    chooseVenue(state.feed, state.venue, Date.now());
    draw();
    if (typeof window !== "undefined" && window.scrollTo) window.scrollTo(0, 0);
  }

  function start() {
    var board = document.getElementById("board");
    if (board) board.addEventListener("click", onClick);
    window.addEventListener("hashchange", draw); // a strip link, a step link or back
    (function tick() {
      poll().then(function () { setTimeout(tick, POLL_MS); });
    })();
  }

  var api = {
    formatMove: formatMove, moveClass: moveClass, isStale: isStale, renderBoard: renderBoard,
    poll: poll, esc: esc, num: num, selection: selection, chooseVenue: chooseVenue,
    followSelected: followSelected,
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
