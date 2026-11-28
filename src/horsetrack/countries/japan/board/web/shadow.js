/* The shadow page (#122): the live board (board.js, #42) plus each row's shadow value, the bucket
 * the value was read at, and the avoid flags (#117, candidate rules of #82, none adopted).
 *
 * renderShadow draws board.js's renderBoard with extra cells per row: the odds ladder (#499: T-20,
 * T-10, T-5, T-2 and the final from the shadow entry's `ladder`, observed odds as figures coloured
 * by their move since T-20 and, when the point has one (#627), the projection made one rung earlier
 * muted beside them as '(≈x.x)', the nowcast's projection of a checkpoint still ahead as '≈x.x'; without
 * a ladder, or when the shadow race's post time is not /feed's (the schedule authority), the plain
 * row's t20/t10/t5/t2_odds and '-' ahead), the value (WIN: mean with its 5-95%
 * band; other pools: the value at the low and the high odds), the bucket label and the flag chips.
 * One race per page, one section per bet type whatever the plain feed holds (#506, SECTIONS):
 * Win, Place, Quinella, Wide, Exacta, Trio, Trifecta, Bracket quinella. Their rows come from the
 * plain feed in board.js's order, favourite first by the crowd's current odds (#498, the owner's
 * display choice), except Exacta, Trio and Trifecta: the shadow race's 15 most likely
 * combinations by `p_model`, most likely first, each joined to the plain row of its odds ('-'
 * without one), leaving out first every combination /feed shows scratched (a runner SCR on its
 * WIN or PLACE row, or the combination's own row SCR). Every row has a Model p cell, the shadow
 * race's `p_model` as a percent. In a DEBUT or MAIDEN race whose shadow race has a `member`
 * (#717, display only), the Win section adds a Horse model % column beside Model p: the shadow
 * race's `p_member` as a percent ('-' without one), a mark (data-member-mark, its title the k) on
 * the rows of `member.marked` (p_member / q at least the leaderboard's k, the one exception to
 * the rule below), and a note in the section head; with a `member.status` ("no JRDB data") the
 * note says so and the column shows no figure and no mark. The pools
 * without a price forecast (NO_FORECAST: no nowcast, #400) show value = Model p x the odds now
 * and say so; the tier-2 sections show their capture's age. The shadow feed (#121) is joined on
 * (race_id, pool, combination) and never highlights or colours a row by value or probability
 * (a selection rule is the owner's, D19). The race follows the URL hash as on the board page.
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
  var SECTIONS = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO", "TIERCE",
    "BRACKET_QUINELLA"];
  var TOP = 15;  // rows of a ranked section
  var RANKED = { FORECAST: true, TRIO: true, TIERCE: true };
  var NO_FORECAST = { QUINELLA: true, BRACKET_QUINELLA: true, FORECAST: true, TRIO: true,
    TIERCE: true };
  var LATEST_ONLY = { BRACKET_QUINELLA: true, FORECAST: true, TRIO: true, TIERCE: true };
  var NO_FORECAST_NOTE = '<span class="muted small">no price forecast: value is Model p × the ' +
    "odds now</span>";

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

  /* A pool without a price forecast: Model p x the plain row's odds now, two decimals; '-'
   * without both (a scratched row has no odds). */
  function productText(p, line) {
    var odds = line.status === "SCR" ? null : num(line.odds);
    return p === null || odds === null ? DASH : (p * odds).toFixed(2);
  }

  /* 0.1234 -> '12.3%'; '-' for none. */
  function percentText(p) {
    return p === null ? DASH : (p * 100).toFixed(1) + "%";
  }

  /* The shadow race's p_model of one pool, {combination: probability}, or null. */
  function pModelOf(race, code) {
    var p = own(own(race, "p_model"), code);
    return p && typeof p === "object" ? p : null;
  }

  /* The runners the plain feed marks scratched, {horse number text: true}: a current SCR row on
   * WIN or on PLACE, either is enough. */
  function scratchedRunners(pools) {
    var out = Object.create(null);
    ["WIN", "PLACE"].forEach(function (code) {
      var body = own(pools, code);
      (body && Array.isArray(body.rows) ? body.rows : []).forEach(function (l) {
        if (l && typeof l === "object" && l.status === "SCR") out[String(l.combination)] = true;
      });
    });
    return out;
  }

  /* A ranked section's rows: the TOP combinations of `p` by probability, most likely first (ties
   * in the feed's order), each the plain row of the same combination or a row without odds. The
   * two feeds are polled apart, so `p` can predate a scratch /feed already shows: a combination
   * with a runner in `scratched`, or whose own plain row is SCR, is left out before the TOP. */
  function topRows(p, body, scratched) {
    if (!p) return [];
    var lines = Object.create(null);
    (body && Array.isArray(body.rows) ? body.rows : []).forEach(function (l) {
      if (l && typeof l === "object") lines[String(l.combination)] = l;
    });
    var probs = Object.create(null);
    var keys = Object.keys(p).filter(function (c) {
      probs[c] = num(p[c]);
      if (probs[c] === null || (c in lines && lines[c].status === "SCR")) return false;
      return !c.split(",").some(function (h) { return scratched[h] === true; });
    });
    keys.sort(function (a, b) { return probs[b] - probs[a]; });
    return keys.slice(0, TOP).map(function (c) {
      return c in lines ? lines[c] : { combination: c, odds: null, odds_hi: null, status: null };
    });
  }

  /* The age of a tier-2 pool's capture at nowMs, in whole minutes. */
  function captureAgeNote(body, nowMs) {
    var at = board.instantMs(own(body, "capture_utc"));
    var said = at === null ? "not captured yet"
      : "captured " + Math.max(0, Math.floor((nowMs - at) / 60000)) + " min ago";
    return '<span class="muted small mono" data-capture-age>' + esc(said) + "</span>";
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

  /* The five checkpoints of a row, each {odds, observed} or null. /feed is the schedule
   * authority: the shadow entry's ladder is used only when the shadow race has /feed's post time
   * (`sameSchedule`); otherwise its timed checkpoints and a projected Final are for another
   * schedule and are ignored, but an observed Final is the price that paid and always stays.
   * With a ladder of the same schedule an observed ladder point stays, else the plain row's
   * reference replaces a projected or null point (the feeds refresh apart). Without one, the
   * plain row's references as observed and null ahead (the final unless observed). */
  function ladderOf(entry, line, sameSchedule) {
    var ladder = own(entry, "ladder");
    var hasLadder = ladder && typeof ladder === "object";
    return LADDER.map(function (c) {
      var point = hasLadder ? own(ladder, c[1]) : null;
      if (!point || typeof point !== "object" || num(own(point, "odds")) === null) point = null;
      else {
        var earlier = own(point, "projected");
        point = { odds: num(point.odds), observed: point.observed === true };
        if (point.observed && num(earlier) !== null) point.projected = num(earlier);
      }
      if (point !== null && !sameSchedule && !(c[1] === "0" && point.observed)) point = null;
      /* the T-2 projection of an observed Final belongs to the shadow race's own schedule */
      if (point !== null && !sameSchedule) delete point.projected;
      if (point !== null && point.observed) return point;
      var ref = c[2] === null ? null : num(own(line, c[2]));
      return ref !== null ? { odds: ref, observed: true } : point;
    });
  }

  /* Observed odds with one decimal, coloured up / down against an observed T-20, and when the
   * point has the projection made one rung earlier (#627) a muted '(≈x.x)' after it in the same
   * cell; projected ones as '≈' and the odds, muted; '-' for none. */
  function ladderCells(entry, line, sameSchedule) {
    var points = ladderOf(entry, line, sameSchedule);
    var t20 = points[0] && points[0].observed ? points[0].odds : null;
    return points.map(function (p, i) {
      var cls = "num lad", shown = esc(DASH);
      if (p !== null && p.observed) {
        shown = esc(p.odds.toFixed(1));
        if (i > 0 && t20 !== null && p.odds !== t20) {
          cls += " mv " + (p.odds > t20 ? "up" : "down");
        }
        if (p.projected !== undefined) {
          shown += ' <span class="projected">(≈' + esc(p.projected.toFixed(1)) + ")</span>";
        }
      } else if (p !== null) {
        shown = esc("≈" + p.odds.toFixed(1));
        cls += " projected";
      }
      return '<td data-col="' + LADDER[i][0] + '" class="' + cls + '">' + shown + "</td>";
    }).join("");
  }

  var HEAD_P = LADDER.map(function (c) { return '<th scope="col" class="num">' + c[3] + "</th>"; })
    .join("") + '<th scope="col" class="num">Model p</th>';
  var HEAD_MEMBER = '<th scope="col" class="num">Horse model %</th>';
  var HEAD_REST = '<th scope="col" class="num">Value</th>' +
    '<th scope="col">Read at</th><th scope="col">Flags</th>';

  /* The shadow race's horse member (#717), {segment, k, status, marked}, or null. */
  function memberOf(race) {
    var m = own(race, "member");
    return m && typeof m === "object" ? m : null;
  }

  /* Whether a section shows the Horse model % column: the Win section of a member race. */
  function memberShown(race, code) {
    return code === "WIN" && memberOf(race) !== null;
  }

  /* The horse-model cell: p_member as a percent ('-' without one), and the p/q mark of a horse
   * in `marked` (an empty element drawn by CSS, its title naming the k). */
  function memberCell(race, line) {
    var m = memberOf(race), horse = String(line.combination);
    var figures = m.status === null || m.status === undefined;
    var p = figures ? num(own(own(race, "p_member"), horse)) : null;
    var marked = figures && Array.isArray(m.marked) && m.marked.map(String).indexOf(horse) >= 0;
    var k = num(m.k);
    var mark = marked ? '<span class="member-mark" data-member-mark title="' +
      esc("p/q ≥ " + (k === null ? "k" : k.toFixed(2))) + '"></span>' : "";
    return '<td data-col="horse-model" class="num">' + mark + esc(percentText(p)) + "</td>";
  }

  /* The Win section's note on the member: its status ("no JRDB data") when it has no figures,
   * else what the column is and the k of the mark. */
  function memberNote(race) {
    var m = memberOf(race);
    if (m === null) return "";
    if (m.status !== null && m.status !== undefined) {
      return '<span class="muted small" data-member-status>' + esc(String(m.status)) + "</span>";
    }
    var k = num(m.k);
    return '<span class="muted small">Horse model %: ' + esc(String(m.segment)) +
      " member" + (k === null ? "" : ", ▲ at p/q ≥ " + esc(k.toFixed(2))) + "</span>";
  }

  /* race_id -> whether the shadow race has the plain feed's post time (the same instant, or a
   * null post time on both feeds: the same unknown schedule). */
  function sameSchedules(feed, races) {
    var same = {};
    var venues = feed && Array.isArray(feed.venues) ? feed.venues : [];
    venues.forEach(function (v) {
      (v && Array.isArray(v.races) ? v.races : []).forEach(function (r) {
        var id = r ? String(r.race_id) : null;
        if (id === null || !has.call(races, id)) return;
        if (r.post_utc == null && races[id].post_utc == null) {
          same[id] = true;
          return;
        }
        var plainMs = board.instantMs(r.post_utc);
        same[id] = plainMs !== null && plainMs === board.instantMs(races[id].post_utc);
      });
    });
    return same;
  }

  /* The HTML of the shadow page for one race of one venue: the board's header and one section per
   * bet type, each row with ladder, Model p, value, bucket and flag cells. /feed is the schedule
   * authority: its post times set the header, the countdown, the strip and the next-race choice. */
  function renderShadow(feed, shadowFeed, opts) {
    opts = opts || {};
    var races = shadowRaces(shadowFeed);
    var same = sameSchedules(feed, races);
    function shadowRace(raceId) {
      return has.call(races, String(raceId)) ? races[String(raceId)] : null;
    }
    var columns = {
      head: function (raceId, code) {
        return HEAD_P + (memberShown(shadowRace(raceId), code) ? HEAD_MEMBER : "") + HEAD_REST;
      },
      cells: function (raceId, code, line) {
        var race = shadowRace(raceId);
        var entry = own(own(own(race, "pools"), code), String(line.combination));
        var p = num(own(pModelOf(race, code), String(line.combination)));
        var value = NO_FORECAST[code] ? productText(p, line) : valueText(code, entry);
        return ladderCells(entry, line, same[String(raceId)] === true) +
          '<td data-col="p-model" class="num">' + esc(percentText(p)) + "</td>" +
          (memberShown(race, code) ? memberCell(race, line) : "") +
          '<td data-col="value" class="num val">' + esc(value) + "</td>" +
          '<td data-col="bucket" class="mono small">' + esc(race ? bucketText(race) : DASH) +
          "</td>" + '<td data-col="flags" class="flags">' + flagChips(entry) + "</td>";
      },
    };
    function sections(race, nowMs) {
      var scratched = scratchedRunners(race.pools);
      return SECTIONS.map(function (code) {
        var body = own(race.pools, code);
        var section = { code: code, body: body,
          note: (NO_FORECAST[code] ? NO_FORECAST_NOTE : "") +
            (LATEST_ONLY[code] ? captureAgeNote(body, nowMs) : "") +
            (memberShown(shadowRace(race.race_id), code) ? memberNote(shadowRace(race.race_id))
              : "") };
        if (RANKED[code]) {
          section.rows = topRows(pModelOf(shadowRace(race.race_id), code), body, scratched);
        }
        return section;
      });
    }
    return win5Line(shadowFeed) +
      board.renderBoard(feed, { now: opts.now, venue: opts.venue, race: opts.race,
        columns: columns, sections: sections });
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

  /* The race drawn in #shadow, or null. */
  function drawnRace(root) {
    var block = root.querySelector ? root.querySelector("[data-race]") : null;
    return block && block.getAttribute ? block.getAttribute("data-race") : null;
  }

  /* Each pool's horizontal scroll ({pool code: scrollLeft}); the .pool sections scroll sideways
   * on a narrow screen and innerHTML replaces them. */
  function poolScroll(root) {
    var out = {};
    var pools = root.querySelectorAll ? root.querySelectorAll("[data-pool]") : [];
    Array.prototype.forEach.call(pools, function (e) {
      out[e.getAttribute("data-pool")] = e.scrollLeft;
    });
    return out;
  }

  function draw() {
    var root = document.getElementById("shadow");
    if (!root || !state.feed) return;
    var pick = board.selection(state.feed, state.venue);
    state.venue = pick.venue;
    var race = drawnRace(root), scroll = poolScroll(root);
    root.innerHTML = renderShadow(state.feed, state.shadow,
      { now: Date.now(), venue: pick.venue, race: pick.race });
    if (race !== null && drawnRace(root) === race) {  // the same race redrawn: keep each pool
      var pools = root.querySelectorAll ? root.querySelectorAll("[data-pool]") : [];
      Array.prototype.forEach.call(pools, function (e) {
        var code = e.getAttribute("data-pool");
        if (has.call(scroll, code) && scroll[code]) e.scrollLeft = scroll[code];
      });
    }
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
