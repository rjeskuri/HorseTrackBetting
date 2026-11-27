/* The shadow page (#122): the live board (board.js, #42) plus each row's shadow value, the bucket
 * the value was read at, and the avoid flags (#117, candidate rules of #82, none adopted).
 *
 * renderShadow draws board.js's renderBoard with extra cells per row: the odds ladder (#499: T-20,
 * T-10, T-5, T-2 and the final from the shadow entry's `ladder`, observed odds as figures coloured
 * by their move since T-20, the nowcast's projection of a checkpoint still ahead as '≈x.x'; without
 * a ladder the plain row's t20/t10/t5/t2_odds and '-' ahead), the value (WIN: mean with its 5-95%
 * band; other pools: the value at the low and the high odds), the bucket label and the flag chips.
 * Rows come from the plain feed only, one race per page, in board.js's order: favourite first by
 * the crowd's current odds (#498, the owner's display choice). The shadow feed (#121) is joined on
 * (race_id, pool, combination) and never sorts, highlights or colours a row by shadow value (a
 * selection rule is the owner's, D19). The race follows the URL hash as on the board page.
 * Values are gross return per 1 yen at the final price: information only, nothing here is a bet.
 * On a WIN5 day (#135) a line above the board, the element with data-win5, shows the shadow feed's
 * `win5`: the whole WIN5 pool's gross return per 1 yen with the carryover (two decimals), the
 * carryover and the tickets sold so far. No such element when `win5` is null.
 * Feed text is untrusted: every feed string goes through esc() before it reaches the markup.
 */
(function () {
  "use strict";

  var board = typeof module !== "undefined" && module.exports && typeof require === "function"
    ? require("./board.js")
    : window.HorseBoard;
  var esc = board.esc, num = board.num;
  var POLL_MS = 5000;
  var DASH = "-";
  var has = Object.prototype.hasOwnProperty;

  function own(obj, key) {
    return obj && typeof obj === "object" && has.call(obj, key) ? obj[key] : null;
  }

  function figure(value) {
    var n = num(value);
    return n === null ? DASH : n.toFixed(2);
  }

  function allNull(values) {
    return values.every(function (v) { return num(v) === null; });
  }

  /* WIN: 'mean (q05–q95)'; other pools: 'low–high'; '-' for a null figure, '-' alone for none. */
  function valueText(code, entry) {
    if (code === "WIN") {
      var w = [own(entry, "value_mean"), own(entry, "value_q05"), own(entry, "value_q95")];
      if (allNull(w)) return DASH;
      return figure(w[0]) + " (" + figure(w[1]) + "–" + figure(w[2]) + ")";
    }
    var r = [own(entry, "value_at_low"), own(entry, "value_at_high")];
    if (allNull(r)) return DASH;
    return figure(r[0]) + "–" + figure(r[1]);
  }

  function flagLabel(code) { return String(code).replace(/_/g, " "); }

  function flagChips(entry) {
    var flags = own(entry, "flags");
    if (!Array.isArray(flags)) return "";
    return flags.map(function (f) {
      return '<span class="chip flag" data-flag="' + esc(f) + '">' + esc(flagLabel(f)) + "</span>";
    }).join(" ");
  }

  /* race_id -> shadow race, from the shadow feed (or none). */
  function shadowRaces(shadowFeed) {
    var out = {};
    var races = (shadowFeed && Array.isArray(shadowFeed.races)) ? shadowFeed.races : [];
    races.forEach(function (r) {
      if (r && r.race_id !== null && r.race_id !== undefined) out[String(r.race_id)] = r;
    });
    return out;
  }

  function bucketText(race) {
    var tau = num(own(race, "tau_bucket_min"));
    return tau === null ? DASH : "@T-" + tau;
  }

  /* 1234567 -> '¥1,234,567'; '-' for a null figure. */
  function yen(value) {
    var n = num(value);
    if (n === null) return DASH;
    return "¥" + String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  /* The WIN5 line of the shadow feed, or '' when the feed has none. */
  function win5Line(shadowFeed) {
    var w = own(shadowFeed, "win5");
    if (!w || typeof w !== "object") return "";
    return '<p class="win5 small" data-win5>WIN5 pool, gross return per ¥1 with the carryover: ' +
      '<b class="mono">' + esc(figure(own(w, "pool_ev"))) + "</b> · carryover " +
      '<span class="mono">' + esc(yen(own(w, "carryover"))) + "</span> · sold so far " +
      '<span class="mono">' + esc(yen(own(w, "investment"))) + "</span></p>";
  }

  /* The odds ladder (#499): [column, shadow ladder key, plain-feed reference, header]. */
  var LADDER = [
    ["ladder-t20", "20", "t20_odds", "T-20"],
    ["ladder-t10", "10", "t10_odds", "T-10"],
    ["ladder-t5", "5", "t5_odds", "T-5"],
    ["ladder-t2", "2", "t2_odds", "T-2"],
    ["ladder-final", "0", null, "Final"],
  ];

  /* The five checkpoints of a row, each {odds, observed} or null: the shadow entry's ladder, else
   * the plain row's references as observed and null ahead (the final always). */
  function ladderOf(entry, line) {
    var ladder = own(entry, "ladder");
    return LADDER.map(function (c) {
      if (ladder && typeof ladder === "object") {
        var point = own(ladder, c[1]);
        if (!point || typeof point !== "object" || num(own(point, "odds")) === null) return null;
        return { odds: num(point.odds), observed: point.observed === true };
      }
      var ref = c[2] === null ? null : num(own(line, c[2]));
      return ref === null ? null : { odds: ref, observed: true };
    });
  }

  /* Observed odds with one decimal, coloured up / down against an observed T-20; projected ones
   * as '≈' and the odds, muted; '-' for none. */
  function ladderCells(entry, line) {
    var points = ladderOf(entry, line);
    var t20 = points[0] && points[0].observed ? points[0].odds : null;
    return points.map(function (p, i) {
      var cls = "num lad", shown = DASH;
      if (p !== null && p.observed) {
        shown = p.odds.toFixed(1);
        if (i > 0 && t20 !== null && p.odds !== t20) {
          cls += " mv " + (p.odds > t20 ? "up" : "down");
        }
      } else if (p !== null) {
        shown = "≈" + p.odds.toFixed(1);
        cls += " projected";
      }
      return '<td data-col="' + LADDER[i][0] + '" class="' + cls + '">' + esc(shown) + "</td>";
    }).join("");
  }

  var HEAD = LADDER.map(function (c) { return '<th scope="col" class="num">' + c[3] + "</th>"; })
    .join("") +
    '<th scope="col" class="num">Value</th><th scope="col">Read at</th>' +
    '<th scope="col">Flags</th>';

  /* The HTML of the shadow page for one race of one venue: the board plus ladder, value, bucket
   * and flag cells. */
  function renderShadow(feed, shadowFeed, opts) {
    opts = opts || {};
    var races = shadowRaces(shadowFeed);
    var columns = {
      head: HEAD,
      cells: function (raceId, code, line) {
        var race = has.call(races, String(raceId)) ? races[String(raceId)] : null;
        var entry = own(own(own(race, "pools"), code), String(line.combination));
        return ladderCells(entry, line) +
          '<td data-col="value" class="num val">' + esc(valueText(code, entry)) + "</td>" +
          '<td data-col="bucket" class="mono small">' + esc(race ? bucketText(race) : DASH) +
          "</td>" + '<td data-col="flags" class="flags">' + flagChips(entry) + "</td>";
      },
    };
    return win5Line(shadowFeed) +
      board.renderBoard(feed,
        { now: opts.now, venue: opts.venue, race: opts.race, columns: columns });
  }

  /* ------------------------------------------------------------ the browser glue */

  var state = { feed: null, feedTag: null, shadow: null, shadowTag: null, venue: null };

  function setStatus(text, cls) {
    if (typeof document === "undefined") return;
    var el = document.getElementById("status");
    if (!el) return;
    el.textContent = text;
    el.className = "chip" + (cls ? " " + cls : "");
  }

  function draw() {
    var root = document.getElementById("shadow");
    if (!root || !state.feed) return;
    var pick = board.selection(state.feed, state.venue);
    state.venue = pick.venue;
    root.innerHTML = renderShadow(state.feed, state.shadow,
      { now: Date.now(), venue: pick.venue, race: pick.race });
    board.followSelected("#shadow");
  }

  /* One GET with If-None-Match: resolves to the HTTP status (200 stores the body and its ETag). */
  function fetchFeed(url, tagKey, bodyKey) {
    var headers = {};
    if (state[tagKey]) headers["If-None-Match"] = state[tagKey];
    return fetch(url, { headers: headers, cache: "no-store" })
      .then(function (res) {
        if (res.status !== 200) return res.status;
        return res.json().then(function (body) {
          state[bodyKey] = body;
          state[tagKey] = (res.headers && res.headers.get("ETag")) || null;
          return 200;
        });
      })
      .catch(function () { return 0; });
  }

  /* One GET of each feed; a 200 replaces that feed, a 304 or an error keeps the last one. */
  function poll() {
    return Promise.all([
      fetchFeed("feed", "feedTag", "feed"),
      fetchFeed("shadow-feed", "shadowTag", "shadow"),
    ]).then(function (codes) {
      draw();
      var plain = codes[0], shadow = codes[1];
      if (plain !== 200 && plain !== 304) {
        setStatus(plain === 503 ? "waiting for the first capture"
          : plain === 0 ? "no connection" : "feed error " + plain, "stale");
      } else if (shadow !== 200 && shadow !== 304) {
        setStatus(state.shadow ? "values not refreshed" : "no shadow values yet", "stale");
      } else {
        setStatus("live", "live");
      }
    });
  }

  function onClick(event) {
    var target = event && event.target;
    var tab = target && target.closest ? target.closest("[data-venue]") : null;
    if (!tab) return;
    state.venue = tab.getAttribute("data-venue");
    board.chooseVenue(state.feed, state.venue, Date.now());
    draw();
    board.toTop();
  }

  /* A strip link, a step link or back: draw the race the hash names, from its top. */
  function onHashChange() {
    draw();
    board.toTop();
  }

  function start() {
    var root = document.getElementById("shadow");
    if (root) root.addEventListener("click", onClick);
    window.addEventListener("hashchange", onHashChange);
    (function tick() {
      poll().then(function () { setTimeout(tick, POLL_MS); });
    })();
  }

  var api = { renderShadow: renderShadow, poll: poll };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document !== "undefined" && typeof fetch === "function" &&
      document.getElementById("shadow")) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
  }
})();
