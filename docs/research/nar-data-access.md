# NAR data access for live odds (research, #36)

Date of all findings: 2026-11-24. Epic #5. Every claim below has a row in [docs/ledger.md](../ledger.md)
under "Japan (NAR): data access". Sample paths are relative to `$HORSETRACK_DATA_DIR`.

## Verdict

**Blocked on owner decision.** The official site's robots.txt shuts every crawler except Googlebot out of
the pages a logger needs (today's race lists and odds, and the results archive), so this document does not
recommend a NAR logger. The question for the owner is under [Question for the owner](#question-for-the-owner).
The [Logger spec](#logger-spec-for-nar-draft-not-a-recommendation) below is a draft for the parsing ticket,
should the owner choose a source that allows it; it is not a recommendation to build one.

## The official source: keiba.go.jp

The site is the 地方競馬情報サイト, run by the local racing organisers and the National Association of Racing
(地方競馬主催者および地方競馬全国協会, "主催者等" in its terms). It is the only official source covering every
NAR venue.

### robots.txt (fetched, `raw/nar/policy/2026/robots.html.gz`)

- `User-agent: Googlebot` with an empty `Disallow`: Google may crawl everything.
- `User-agent: *`: `Crawl-delay: 10`, and `Disallow` for:
  `/KeibaWeb/TodayRaceInfo/`, `/KeibaWeb/DataRoom/`, `/KeibaWeb/DataDownload/`,
  `/KeibaWeb/MonthlyConveneInfo/`, `/KeibaWeb_IPAT/`, `/KeibaWebSP/TodayRaceInfo/`,
  `/KeibaWebSP/DataRoom/`, `/KeibaWebSP/MonthlyConveneInfo/`, `/wp-json/`, `/xmlrpc.php`.
- The file's own comment gives the reason (gist): these are dynamic PHP/DB pages with heavy load and a wide
  parameter space; bots are to be steered to public static content.

`TodayRaceInfo` is the home page's "レース情報" entry (`/KeibaWeb/TodayRaceInfo/TodayRaceInfoTop`), so the
race lists, cards and odds of the day sit under a disallowed path. `MonthlyConveneInfo` (the meeting
calendar) and `DataRoom` (results, history) are disallowed as well.

### Terms of use (fetched, `raw/nar/policy/2026/terms.html.gz`, https://www.keiba.go.jp/terms.html)

No clause names robots, scraping or programs. The clauses that bear on automated access, by heading:

- 利用規約について: using the site means accepting the terms.
- 著作権について: rights in the site's images and text belong in principle to the organisers; reproduction or
  copying (転載、複製) without prior permission is prohibited.
- 禁止事項について, items 7 and 8: acts that break, or may break, laws and regulations; and any other act the
  organisers judge inappropriate.
- 不正なアクセスへの対応について: an IP address with an abnormal number of accesses, or unauthorised access, may
  be blocked.
- 免責事項について: no warranty of completeness, accuracy or currency of the content.

Read together: the terms alone do not forbid a polite private reader, but the robots.txt is the site's
explicit statement about automated access, item 8 lets the organisers treat ignoring it as inappropriate,
and storing pages is copying (複製) under the copyright clause. That is a restriction on automated access to
the pages a logger needs, so the ticket's rule applies: stop and ask.

### What was not fetched, and why

No page under a disallowed path was requested, through the Fetcher or otherwise: no race list, odds page,
card, calendar or result. The page structure below is therefore unverified (every such row in the ledger is
an assumption). Reading the home page and terms with WebFetch, and four robots.txt files plus the terms
through the Fetcher, are the only accesses made.

### What the site is believed to offer (unverified)

- Today's races: `TodayRaceInfoTop` lists the day's meetings; each meeting has a race list, a card per race,
  odds pages per pool and a dividends page. Query parameters are believed to be `k_raceDate` (YYYY/MM/DD),
  `k_babaCode` (venue number) and `k_raceNo`, e.g. `/KeibaWeb/TodayRaceInfo/RaceList`,
  `/KeibaWeb/TodayRaceInfo/OddsTanFuku`, `/KeibaWeb/TodayRaceInfo/RefundMoneyList`.
- Pools: 単勝, 複勝, 枠複, 枠単, 馬複, 馬単, ワイド, 3連複, 3連単 (NAR says 馬複/枠複 where JRA says 馬連/枠連, and
  sells 枠単, which JRA does not). Some organisers also sell multi-race bets through the betting sites only.
  Takeout varies by organiser and pool; do not carry JRA's rates over.
- History: results and dividends under `DataRoom`; the terms mention notation changes from 2001, so the
  archive is believed to reach back at least that far.
- Encoding: the static pages fetched are UTF-8 (`<meta charset="UTF-8">` on the terms page). The dynamic pages
  are unverified.
- Odds refresh rate: unknown. JRA's is about 30-40 s with `max-age=20` (ledger, JRA site rows); nothing is
  known for NAR.

## Other sources of NAR live odds

| Site | Status | robots.txt (2026-11-24) | Terms that matter |
|---|---|---|---|
| keiba.go.jp | official (organisers and NAR) | `*` disallowed from today's races, calendar and results; `Crawl-delay: 10` | above |
| SPAT4 (www.spat4.jp) | betting site run for the Minami-Kanto organisers; members bet | HTTP 403 to a generic User-Agent; not read | not read (403) |
| Oddspark (www.oddspark.com) | betting site for many NAR organisers | HTTP 403 to a generic User-Agent; site policy page 403 to WebFetch as well | not read (403) |
| Rakuten Keiba (keiba.rakuten.co.jp) | betting site for NAR; odds pages public | `*` `Crawl-Delay: 60`; disallows only `/app/` and a prediction path, not odds | 楽天競馬規約: 利用規約 Art. 17 (prohibitions on 会員: (7) commercial use, (9) heavy load on systems or servers), Art. 18 (IP rights to Rakuten or the provider), 規約 Art. 32 (no reposting or reuse). Written for members; silent on non-member readers. Odds URL: `/odds/tanfuku/RACEID/<18 digits>` |
| netkeiba (nar.netkeiba.com) | private media site | no robots.txt (HTTP 404) | netkeiba 利用規約 Art. 14 (no copying beyond private use; making data available to many people is beyond private use), Art. 15 (no commercial use), Art. 16 (12) unauthorised access, (15) obstructing the service |

The two betting sites that answered 403 refuse a plain client; getting past that would be evasion and is
out of bounds. Rakuten Keiba and netkeiba do not disallow odds pages to crawlers; both are third-party
sources whose terms allow private use at most. This project already reads netkeiba for JRA history on a
private-use-only basis (`countries/japan/config/jra.yaml`); whether that stance extends to NAR live odds,
or to Rakuten Keiba, is the owner's call, not a ruling.

## Request cost

The Fetcher keeps one gap per host, and every NAR venue is on one host (www.keiba.go.jp), so all meetings of
a day share one request budget. At the robots.txt `Crawl-delay: 10` a host allows 360 requests an hour.

- One capture of one meeting, assuming one race-list page plus one page per pool per race (unverified page
  layout): 1 + P x R requests. With R = 12 races and P = 3 pages (単勝/複勝 together, 馬複, ワイド):
  37 requests, about 6 minutes at 10 s. The JRA capture is 2 + 3R = 38 requests, about 38 s at 1 a second.
- A day: weekday NAR cards run about 4-7 meetings (assumption). One sweep of 6 meetings x 12 races x 3 pages
  is 222 requests, about 37 minutes, so the JRA pattern (sweep the whole meeting every few minutes) cannot
  run at that delay.
- A per-race pattern fits: k captures per race near its off (say T-30, T-10, T-5, T-2 and the final board
  after the off) is 5 x 3 = 15 requests a race, 1,080 for 72 races, 3 hours of request time spread over a
  racing day of about 11 hours (10:00-21:00 JST). Races at different venues are staggered by minutes, so the
  T-2 captures of two venues can collide; the scheduler must order by post time and accept a late capture
  rather than shorten the gap.
- Dividends: one dividends page per meeting after its last race: about 6 requests a day.

At Rakuten Keiba's `Crawl-Delay: 60` the same per-race pattern costs 18 hours a day: not feasible beyond
one or two meetings.

## Night racing: logger cadence and stop rule

- NAR racing spans most days of the week, from late morning to about 21:00 JST: Ohi (TCK Twinkle), Kawasaki,
  Funabashi, Mombetsu, Kochi and Sonoda run night or twilight cards in season; Obihiro (ban'ei) runs into the
  evening. All assumption until a calendar is read.
- 21:00 JST is 12:00 UTC on the same date, so no meeting crosses a UTC date; the race date in a race id stays
  the JST date and storage timestamps stay UTC, as for JRA.
- Cadence: schedule by race (post time), not by meeting sweep, as above. The logger reads post times once
  per meeting at the start of the day and again after any race-list change (NAR delays posts more often than
  JRA; assumption).
- Stop rule: per meeting, not per day. A meeting stops after its last race's final board and the dividends
  page; the day ends after the last meeting, which may be after 20:30 JST. The JRA rule (stop after three
  consecutive WrongMeeting or NotListed captures, #23) carries over per meeting, and a meeting cancelled
  mid-card (weather is commoner at NAR's dirt-only tracks; assumption) must stop that meeting alone.

## Logger spec for NAR (draft, not a recommendation)

For the parsing ticket, if the owner approves a source. Every item marked (unverified) must be checked
against a captured page first; the first capture is a fixture task, not a logger task.

- Source key `nar`; race id `nar:<YYYY-MM-DD>:<VENUE>:<NN>` (JST date), rows in the format of
  `docs/specs/odds-log.md` at `odds_snapshots/nar/<YYYY-MM-DD>/<VENUE>.jsonl.gz`, post-time sidecar beside it.
- Entry URL (keiba.go.jp, unverified): `/KeibaWeb/TodayRaceInfo/TodayRaceInfoTop` for the day's meetings;
  then `RaceList?k_raceDate=<YYYY/MM/DD>&k_babaCode=<n>` per meeting; then odds pages per race and pool with
  `&k_raceNo=<n>`; `RefundMoneyList` for dividends. Like the JRA capture, follow links read off a fetched page
  rather than computing them, and check every page's date and venue (WrongMeeting).
- Selectors (unverified): none are known; the parsing ticket takes them from the first fixture. Expect
  table-based layouts; key columns by their header labels (馬番, 単勝, 複勝, 人気), not by position.
- Pool labels to pool keys: 単勝 WIN, 複勝 PLACE, 枠複 BRACKET_QUINELLA, 枠単 BRACKET_EXACTA (new key), 馬複
  QUINELLA, 馬単 FORECAST, ワイド QUINELLA_PLACE, 3連複 TRIO, 3連単 TIERCE. Win/Place/Wide first (D3).
  Places paid and takeout per organiser go in a `nar.yaml` with a source per value, not copied from `jra.yaml`.
- Venue codes and names (`k_babaCode` unverified; short codes proposed here):

  | short | name | k_babaCode |
  |---|---|---|
  | OBI | 帯広 (ばんえい) | 3 |
  | MON | 門別 | 36 |
  | MOR | 盛岡 | 10 |
  | MIZ | 水沢 | 11 |
  | URA | 浦和 | 18 |
  | FUN | 船橋 | 19 |
  | OOI | 大井 | 20 |
  | KAW | 川崎 | 21 |
  | KNZ | 金沢 | 22 |
  | KSM | 笠松 | 23 |
  | NGY | 名古屋 | 24 |
  | SON | 園田 | 27 |
  | HIM | 姫路 | 28 |
  | KOC | 高知 | 31 |
  | SAG | 佐賀 | 32 |

  Ban'ei (OBI) is a different sport for modelling purposes; log it only if the owner says so.
- Synthetic fixture shape (`tests/fixtures/synthetic/nar/`): a meetings page listing two venues on one date;
  a race list of three races with post times, one already off; a 単勝/複勝 page of eight runners with one
  scratched (取消) and a ranged 複勝 price; a dividends page for one race. Invented names and numbers only,
  in the encoding the real page uses, built by a builder function like the JRA test double
  (`tests/countries/japan/jradb_double.py`).
- Rate: the site's `Crawl-delay: 10`, i.e. `www.keiba.go.jp: {requests_per_second: 0.1, jitter: 0.1}`; one
  host for all venues; the circuit breaker stops on 403 and 429 as for JRA. The terms' IP-blocking clause makes
  a block costly (it would also cut off manual reading), so no retry storm: `max_retries` at most 2.

## Question for the owner

The official site's robots.txt disallows all crawlers but Googlebot from `/KeibaWeb/TodayRaceInfo/` (today's
race lists and odds), `/KeibaWeb/DataRoom/` (results) and `/KeibaWeb/MonthlyConveneInfo/` (the calendar), and
its terms prohibit copying without permission and any act the organisers judge inappropriate. Options:

1. Do not log NAR (close the epic's NAR goal as infeasible).
2. Ask the organisers (the site's feedback form) for permission to read these pages automatically at the
   crawl delay, for private research.
3. Use a third-party source that does not disallow odds pages (Rakuten Keiba at a 60 s crawl delay, or
   nar.netkeiba.com under the same private-use stance as JRA history), accepting its terms.
4. Read the official pages anyway at 10 s a request. Not advised: it goes against the site's stated robots.txt
   and risks the IP block in its terms.
