/* The shadow page (#122): the live board (board.js, #42) plus each row's shadow value, the bucket
 * the value was read at, and the avoid flags (#117, candidate rules of #82, none adopted).
 *
 * renderShadow draws board.js's renderBoard with four extra cells per row: the T-20→T-2 move
 * (`move_t20_t2_pct`, '-' until the T-2 capture exists), the value (WIN: mean with its 5-95% band;
 * other pools: the value at the low and the high odds), the bucket label and the flag chips.
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

  var HEAD = '<th scope="col" class="num">T-20→T-2</th>' +
    '<th scope="col" class="num">Value</th><th scope="col">Read at</th>' +
    '<th scope="col">Flags</th>';

  /* The HTML of the shadow page for one race of one venue: the board plus move, value, bucket
   * and flag cells. */
  function renderShadow(feed, shadowFeed, opts) {
    opts = opts || {};
    var races = shadowRaces(shadowFeed);
    var columns = {
      head: HEAD,
      cells: function (raceId, code, line) {
        var race = has.call(races, String(raceId)) ? races[String(raceId)] : null;
        var entry = own(own(own(race, "pools"), code), String(line.combination));
        var mv = board.moveClass(line.move_t20_t2_pct);
        return '<td data-col="move-t20-t2" class="num mv' + (mv ? " " + mv : "") + '">' +
          esc(board.formatMove(line.move_t20_t2_pct)) + "</td>" +
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
    if (typeof window !== "undefined" && window.scrollTo) window.scrollTo(0, 0);
  }

  function start() {
    var root = document.getElementById("shadow");
    if (root) root.addEventListener("click", onClick);
    window.addEventListener("hashchange", draw); // a strip link, a step link or back
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
