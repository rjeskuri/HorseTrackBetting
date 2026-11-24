# NAR data access for live odds (research, #36)

Epic #5. Findings are dated: 2026-11-24 (the official site and the other candidates) and 2026-11-25 (the
chosen source, nar.netkeiba.com). Every claim below has a row in [docs/ledger.md](../ledger.md) under
"Japan (NAR): data access". Sample paths are relative to `$HORSETRACK_DATA_DIR`; no fetched page is in git.

## Verdict

**Feasible from nar.netkeiba.com, under the owner's conditions, with one check left for the logger ticket.**
Race lists, cards, odds and results are plain server-rendered UTF-8 HTML on one host, the race id encodes the
date and venue, one page carries both 単勝 and 複勝, and a day of NAR racing fits the 10 s rate when each race is captured at the
owner's decision grid (D2, `countries/japan/config/decision.yaml`) plus the final board: about 560 requests,
93 minutes of request time, on the 2026-11-25 card. Not yet seen: an odds page
while sales are open. At 02:45 JST no race was on sale, and the odds page showed netkeiba's own AI forecast
(予想オッズ) instead. The parsing ticket's first capture must be a live-sales page; until then the live-odds
markup is an assumption (that it is the finished-race markup, which was captured).

## Owner decision (2026-11-24)

The official site restricts automated access (below), so the ticket went to needs-input. The owner chose
option (3), **nar.netkeiba.com, polite**, on the same footing as the JRA history source:

- private use only; at least 10 s between requests;
- nothing scraped in the public repo: pages and rows stay under `$HORSETRACK_DATA_DIR`, tests use synthetic
  fixtures, and PRs show aggregates only;
- block policy: on HTTP 403 or 429, pause 15 minutes, resume once at half rate, and on a second block stop
  for the day and alert;
- keiba.go.jp's disallowed paths are not read. #37 and #38 build against nar.netkeiba.com.

## The official source: keiba.go.jp (2026-11-24, not used)

The 地方競馬情報サイト is run by the local organisers and the National Association of Racing.

- **robots.txt** (`raw/nar/policy/2026/robots.html.gz`): only Googlebot may crawl everything. For `*`:
  `Crawl-delay: 10` and `Disallow` for `/KeibaWeb/TodayRaceInfo/` (the day's races and odds),
  `/KeibaWeb/DataRoom/` (results), `/KeibaWeb/DataDownload/`, `/KeibaWeb/MonthlyConveneInfo/` (calendar),
  `/KeibaWeb_IPAT/` and the `KeibaWebSP` twins. The file's comment: these are heavy dynamic PHP/DB pages,
  and bots are to keep to static content.
- **Terms of use** (`raw/nar/policy/2026/terms.html.gz`, https://www.keiba.go.jp/terms.html): no clause names
  robots or scraping. 著作権について prohibits reproduction or copying (転載、複製) without prior permission;
  禁止事項について item 8 covers any act the organisers judge inappropriate; 不正なアクセスへの対応について allows
  blocking an IP with abnormal access. Read with the robots.txt, this restricts automated access to the
  pages a logger needs, which is why the ticket stopped for the owner.
- No page under a disallowed path was ever requested.

Other candidates (2026-11-24): SPAT4 and Oddspark answer 403 to a plain client (not pursued: getting past it
would be evasion); Rakuten Keiba sets `Crawl-Delay: 60`, too slow for more than one or two meetings.

## The chosen source: nar.netkeiba.com (2026-11-25)

- **Access**: ten requests, 2026-11-24 17:44-17:49 UTC (02:44-02:49 JST on Wed 2026-11-25), all through
  `horsetrack.core.ingest.Fetcher` at one request per 10.5 s or slower, with the project's User-Agent: six
  HTML pages on nar.netkeiba.com (race list shell and sub page for 2026-11-25, one card, the `b1` odds page
  of a race not yet on sale and of a finished 2026-11-24 race, that race's result), its robots.txt, two odds
  scripts on cdnv2.netkeiba.com, and the terms on www.netkeiba.com. Nine answered 200;
  `nar.netkeiba.com/robots.txt` answered 404, as on 2026-11-24. No 403 or 429. Stored under
  `raw/nar/netkeiba/2026/` (the six pages and two scripts: eight `<key>.headers.json` sidecars, six of them
  HTML) and `raw/nar/policy/2026/terms_netkeiba.html.gz` (with its own sidecar); a sidecar holds the status,
  URL and response headers.
- **Terms** (`raw/nar/policy/2026/terms_netkeiba.html.gz`, https://www.netkeiba.com/info/kiyaku.html, last
  revised 2022-04-01): Art. 14 forbids use beyond the private use copyright law allows, treats all data from the
  service as copyrighted, and counts putting data where unspecified or many people can see it as beyond
  private use; Art. 15 forbids commercial use; Art. 16 (12) forbids unauthorised access to, or obstructing,
  the service's equipment. No clause names robots or scraping, and there is no robots.txt. The owner's
  conditions (private, 10 s, nothing published but aggregates) are written to stay inside these articles.
- **Encoding and headers**: every HTML page is UTF-8 (`Content-Type: text/html; charset=UTF-8` and
  `<meta charset="UTF-8">`), served by Apache behind CloudFront (`x-cache: Miss from cloudfront` on every
  HTML response). No `Cache-Control`, `Expires`, `ETag` or `Last-Modified` on HTML, and the odds page prints no
  as-of time, so a snapshot's time is the fetch time. Odds refresh rate: unknown.
- **Race id**: 12 digits `YYYY VV MMDD RR` (year, venue code, month and day, race number), e.g.
  `2026 42 1124 12` is 浦和 12R on 2026-11-24 (the page title says so). A meeting id (`kaisai_id`) is the
  first 10 digits. Unlike JRA's netkeiba id (`YYYY VV KK DD RR`, meeting and day numbers), the date is in the
  id, so ids can be computed; the logger still reads them off the race list. NAR venue codes are 30 and up; JRA's are 01-10.
- **The 2026-11-25 card** (single day): 3 meetings, 36 races, 12 each: 浦和 (12:20-18:20), 名古屋
  (nighter, 14:40-20:50) and 園田 (10:40-16:45). Two venues have a race at the same minute (13:20).

## Request cost (10 s a request, one host)

All venues share one host and so one budget: 360 requests an hour.

- **One capture of one meeting** (12 races): the race list (one page for all meetings) plus, per race, the
  `b1` page (単勝 and 複勝 together) and the `b5` page (ワイド): 1 + 2 x 12 = 25 requests, about 4 minutes.
  WIN/PLACE only: 13 requests, about 2 minutes.
- **One sweep of a whole day** (2026-11-25: 3 meetings, 36 races): 1 + 72 = 73 requests, about 12 minutes,
  so the JRA pattern of sweeping every meeting every few minutes does not fit.
- **A day by race**: the capture times are the owner's decision grid (D2), read from
  `src/horsetrack/countries/japan/config/decision.yaml` (`grid_min: [60, 30, 16, 5, 2, 1]`, primary T-2);
  this doc sets no NAR schedule, and #38 owns the logger's cadence. Six grid captures plus the final board
  after the off = 7 captures x 2 pages (`b1`, `b5`) = 14 requests a race, 504 for 36 races; plus one result
  page per race (36) and a race-list refresh every 30 minutes (about 20): about 560 requests, about 93
  minutes of request time across 10:40-20:50 JST. WIN/PLACE only (`b1`): 7 a race, about 310 requests, 52
  minutes. A day of 5-6 meetings (assumption) about doubles it, still well under the host budget.
- **Close posts**: on 2026-11-25, 浦和 3R and 園田 6R both post at 13:20. Each grid time then asks for 4
  requests (two races x two pages), 40 s; T-2 and T-1 are 60 s apart, so the two captures do not overlap,
  and fetching both `b1` pages before either `b5` keeps each race's WIN/PLACE within 10 s of the first
  request, the last `b5` within 30 s. A simulation of the whole card (one queue, 10 s a request, WIN/PLACE
  first near the off, final board 2-5 minutes after post: an assumption) puts every T-2 and T-1 `b1` within
  10 s of its grid time and every T-2 and T-1 `b5` within 30 s; the worst lag on the day is 50-70 s, on T-60
  and T-5 `b5` pages where one race's grid time meets another race's captures. `board_at` uses only rows
  available at or before t (`docs/specs/board-at-t.md`), so a capture that starts at its grid time lands
  after it; how far ahead of each grid time to start is #38's call.
- Bytes per page (single-run): race list about 40 KB, odds about 55 KB, result about 78 KB, card about
  200 KB (the card is not needed by the logger).

## Night racing: cadence and stop rule

- NAR races most days of the week, morning to about 21:00 JST; the 2026-11-25 card runs 10:40-20:50, with the
  nighter meeting marked on the race list (`small.icon-nighter`). 21:00 JST is 12:00 UTC on the same date,
  so no meeting crosses a UTC date; race ids keep the JST date, storage timestamps stay UTC.
- **Cadence** (#38 owns it): schedule by race post time at the `decision.yaml` grid, not by meeting sweep. Read the race list at the start of the day
  and every 30 minutes (NAR posts move; assumption). When two races share a post minute, order by post time,
  take WIN/PLACE before ワイド near the off, and accept a late capture rather than shorten the 10 s gap.
- **Stop rule**: per meeting, then per day. A meeting stops after its last race's final board and result
  page; the day stops after the last meeting (as late as 21:00 JST). The JRA rule (stop after three
  consecutive WrongMeeting or NotListed captures, #23) applies per meeting, and a meeting cancelled mid-card
  stops alone. The block policy above overrides both: a second block stops the whole day.

## Logger spec for NAR

For the parsing ticket (#37) and the logger (#38). "Verified" means seen in the captured pages listed above;
everything else is marked unverified.

- **Source and storage**: source key `nar`; race id `nar:<YYYY-MM-DD>:<VENUE>:<NN>` (JST date), the
  12-digit netkeiba id kept as a field; rows in the format of `docs/specs/odds-log.md` at
  `odds_snapshots/nar/<YYYY-MM-DD>/<VENUE>.jsonl.gz`; raw pages via the Fetcher at
  `raw/nar/<page_type>/<YYYY>/<key>.html.gz` (page types `race_list`, `odds_b1`, `odds_b5`, `result`).
- **Entry URLs and navigation** (verified unless marked):
  - Day's meetings and races: `https://nar.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD`. The
    page `top/race_list.html?kaisai_date=` is a shell that loads `race_list_get_date_list.html` and then
    `race_list_sub.html` by JavaScript; fetch the sub page directly.
  - Odds: `https://nar.netkeiba.com/odds/?race_id=<id>&type=<t>`, with the race id read off the race list
    (the race list links only to cards; the card's odds links have this form). Types: `b0` 上位人気一覧, `b1` 単勝・複勝, `b3` 枠連, `b9` 枠単, `b4` 馬連,
    `b5` ワイド, `b6` 馬単, `b7` 3連複, `b8` 3連単. Only `b1` was captured.
  - Result and dividends: `https://nar.netkeiba.com/race/result.html?race_id=<id>`. A per-meeting
    `top/payback_list.html?kaisai_date=YYYYMMDD&kaisai_id=<10 digits>` is linked (unverified: not fetched).
  - The odds are in the HTML: neither the odds page nor `odds_common.js`/`odds_tanfuku.js` references an odds
    API (the only `api/` calls are favourites and cart). Unverified for a race on sale.
  - Check every page: the race id in the URL and in the page's links matches the race asked for
    (WrongMeeting, as for JRA).
- **Selectors** (verified on the pages captured):
  - Race list: venue tabs `ul.RaceList_ProvinceSelect li a` (text: venue name; `href` has `kaisai_id`); one
    `dl.RaceList_DataList` per meeting; meeting title `p.RaceList_DataTitle` (round, venue, day; nighter flag
    `small.icon-nighter`); races `li.RaceList_DataItem > a[href*="shutuba.html?race_id="]`, number in
    `.Race_Num`, post time the first `span` of `.RaceData` (`HH:MM`), runner count as text `N頭` in `.RaceData`.
  - Odds `b1`, on sale or after: `#odds_tan_block` (h2 単勝) and `#odds_fuku_block` (h2 複勝), each a
    `table.RaceOdds_HorseList_Table` with headers 枠, 馬番, 印, 馬名, オッズ; per row, 馬番 is the second `td`
    (no class) and the price `td.Odds span.Odds`; 複勝 is a range `lo - hi` (to `odds` and `odds_hi`). Key
    columns by header label, not position. Verified on a finished race only.
  - Odds `b1`, before sales: one `table.RaceOdds_HorseList_Table.Ninki` with headers 人気, 馬番, 印, 馬名,
    予想オッズ, and a notice that the forecast switches to real odds when sales open. This is netkeiba's
    model, not the board: the parser must reject it (a `NotOnSale` result), never log it. The same goes for
    the card's 予想オッズ column (`span.Odds_Ninki`).
  - Result: `#All_Result_Table` (headers 着順, 枠, 馬番, 馬名, ..., 人気, 単勝オッズ); dividends in two
    `table.Payout_Detail_Table`, one `tr` per pool with classes `Tansho`, `Fukusho`, `Wakuren`, `Umaren`,
    `Wide`, `Wakutan`, `Umatan`, `Fuku3`, `Tan3`; cells `td.Result` (numbers in `span`s or `ul li span`),
    `td.Payout span` (`N,NNN円`, several winners separated by `<br />`) and `td.Ninki`.
  - Unverified: the scratched-runner (取消/除外) markup, the `b5` matrix, and any sales-closed marker.
  - Open for #37, not decided here: how the final board is recognised. The stored finished-race `b1` page
    has no 最終 or 確定 marker (nor 締切 or 発売), while `final_board` in `docs/specs/board-at-t.md` keys on
    `最終オッズ` in `page_state` or `status_raw`. Options: treat the first `b1` capture after the off as
    final, or take the result page's 単勝オッズ column.
- **Pool labels to pool keys** (netkeiba prints JRA-style names, though NAR's own are 枠複/馬複; list both):
  単勝 WIN, 複勝 PLACE, 枠連/枠複 BRACKET_QUINELLA, 馬連/馬複 QUINELLA, ワイド QUINELLA_PLACE, 枠単
  BRACKET_EXACTA (new key: JRA does not sell it), 馬単 FORECAST, 3連複 TRIO, 3連単 TIERCE. All nine were paid
  on the one result captured. Win/Place/Wide first (D3). Places paid and each organiser's takeout go in a
  `countries/japan/config/nar.yaml` with a source per value, not copied from `jra.yaml` (unverified).
- **Venue codes and names** (characters 5-6 of the race id, `jyo_cd` on netkeiba; short codes proposed):

  | short | name | code | |
  |---|---|---|---|
  | OBI | 帯広 (ばんえい) | 65 | unverified |
  | MON | 門別 | 30 | unverified |
  | MOR | 盛岡 | 35 | unverified |
  | MIZ | 水沢 | 36 | unverified |
  | URA | 浦和 | 42 | verified |
  | FUN | 船橋 | 43 | unverified |
  | OOI | 大井 | 44 | unverified |
  | KAW | 川崎 | 45 | unverified |
  | KNZ | 金沢 | 46 | unverified |
  | KSM | 笠松 | 47 | unverified |
  | NGY | 名古屋 | 48 | verified |
  | SON | 園田 | 50 | verified |
  | HIM | 姫路 | 51 | unverified |
  | KOC | 高知 | 54 | unverified |
  | SAG | 佐賀 | 55 | unverified |

  The logger maps by the race list's tab text and code, and fails on a code it does not know. Ban'ei (OBI) is
  a different sport for modelling; log it only if the owner says so.
- **Synthetic fixtures** (`tests/fixtures/synthetic/nar/`, UTF-8, invented names and numbers, built by a
  builder function like `tests/countries/japan/jradb_double.py`): a `race_list_sub` page with two meetings on
  one date (one with `icon-nighter`) and three races each with post times; a `b1` page on sale with eight
  runners, one scratched (markup a guess until seen) and ranged 複勝 prices; a `b1` page before sales (the
  予想オッズ table, which must parse to `NotOnSale`); a result page with the nine payout rows, ranged rows with
  three winners.
- **Rate and block policy**: in `nar.yaml`,

  ```yaml
  http:
    hosts:
      default: {requests_per_second: 0.1, jitter: 0.0}
      nar.netkeiba.com: {requests_per_second: 0.1, jitter: 0.0}
    max_retries: 2
    circuit_breaker: {stop_on_status: [403, 429], max_consecutive_failures: 5}
  ```

  Jitter stays 0: the Fetcher's jitter is symmetric, so any positive value allows gaps under 10 s. The
  Fetcher spaces retries and redirect hops by the same gap. Its breaker opens on the first 403/429 and stays
  open; the logger implements the rest of the block policy on top of `CircuitOpen`: wait 15 minutes, start a
  fresh Fetcher at `requests_per_second: 0.05`, and on a second block stop for the day and alert. Never
  change the User-Agent or otherwise work around a block.
