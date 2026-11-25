# Weather: JMA AMeDAS observations per JRA venue

Ticket #89 (epic #72). Code: `src/horsetrack/countries/japan/jma/` (`parse.py`, `fetch.py`,
`stations.yaml`), `countries/japan/config/weather.yaml` and the pre-registered terms in
`countries/japan/config/weather_terms.yaml`. Tests: `tests/countries/japan/test_jma.py` over the
synthetic `tests/fixtures/synthetic/jma_amedas_hourly.csv`.

Status: the owner's answers on #89 (2026-11-25) set the publication lag (120 min), approved the
ingest under PDL1.0 (the 2014-2026 backfill, one request per 5 s, a stop on 403 or 429) and the two
terms as drafted. `horsetrack japan jma-backfill` fills the `weather` table; the terms are
pre-registered below, with three details still open (asked on #89), and the family is not frozen
yet.

## The table

`weather` (`core.store.schemas`), one row per station and hour, sorted by `station_id, obs_utc`:

| column | kind | meaning |
| --- | --- | --- |
| `jurisdiction` | str | the jurisdiction whose venues the station serves (`jra`) |
| `station_id` | str | JMA's block_no (地点番号): `47412`, `1133`, `0598` (`stations.yaml`) |
| `obs_utc` | utc | the observation time: JST clock in UTC (see Time) |
| `temp_c` | float | 気温 (℃), at the hour |
| `precip_mm` | float | 降水量 (mm) in the hour ending at `obs_utc` |
| `wind_ms` | float | 風速 (m/s), the 10-minute mean ending at `obs_utc` |
| `wind_dir_deg` | float | 風向, degrees clockwise from north the wind blows from (16 points) |
| `sunshine_h` | float | 日照時間 (hours) in the hour ending at `obs_utc` |
| `available_utc` | utc | when the value counts as published: `obs_utc` + the configured lag |
| `ingested_utc` | utc | when this copy was fetched |

The page build (`core.store.build`) never writes it; `horsetrack japan jma-backfill` does (Fetch,
below), as `processed/jra/weather.parquet` through the store's writer.

## Parse rules (`parse.parse_amedas_csv(text, *, station_id)`)

Input: one station's hourly download from 過去の気象データ・ダウンロード, decoded from Shift_JIS
(cp932) by the caller, LF or CRLF.

- **Columns by header, never by position.** The 年月日時 row names each item (気温(℃), 降水量(mm),
  日照時間(時間), 風速(m/s)), matched without its unit; a 風向 sub-row marks the wind direction
  columns under 風速; the last header row labels each item's 品質情報, 現象なし情報 and 均質番号.
  Element order may vary, an AMeDAS-only station has no 現象なし情報 columns, and an element the
  parser does not read (相対湿度) is ignored. An element the file does not hold is NaN on every row.
- **Refused (ValueError)**: no 年月日時 row; several stations in one file (one station per
  download); an element without its 品質情報 column; a body line that is not `Y/M/D H:MM[:SS]`.
- **Time.** The clock is JST (日本中央標準時, no daylight saving), converted to UTC (−9 h). JMA's
  hours run 1 to 24; with dates as literals it stores 24時 as 0:00 of the next day, and a
  `24:00:00` label means the same instant. Amounts (precipitation, sunshine) are JMA's 前1時間,
  the hour before the label.
- **Quality (品質情報).** Kept as printed: 8 正常値 and 5 準正常値 (some data missing, within
  tolerance). Missing (NaN): 4 資料不足値, 2 疑問値, 1 欠測, 0 (not an observed item), blank
  (JMA leaves it blank when its own fetch failed), `///` and anything else. JMA's help pages
  disagree on whether 欠測 is 1 or 0; both are missing here.
- **Missing is NaN, never 0.** A value that is not a number (blank, `///`, `×`, `#`, a value with
  `)` or `]` attached) is NaN.
- **Genuine zeros.** A printed 0 stays 0.0 (for precipitation, JMA prints 0.0 when rain fell but
  less than the gauge's step). A blank amount whose 現象なし情報 is 1 (no rain, no sun) is 0.0, and
  so is an amount printed `--`, JMA's 該当現象がない in symbol mode. At an AMeDAS-only station,
  which has no 現象なし情報, a blank amount is NaN.
- **Wind direction.** The 16 compass names in degrees clockwise from north: 北 0, 北北東 22.5,
  ... 北北西 337.5. 静穏 (calm: mean speed 0.2 m/s or less) has no direction, NaN, and keeps its
  speed. Speed and direction carry their own quality columns: a flagged direction leaves the speed.
- **Sunshine.** Measured at official stations; at AMeDAS stations JMA has given a satellite
  estimate (推計気象分布) since 2021-03-02 and cut the series there; airport stations have none
  (NaN). The CSV does not mark estimates; the 均質番号 change is the only trace, and the parser does
  not read it (it restarts from 1 in every download).

`parse.with_publication_time(frame, lag)` returns a copy with `available_utc = obs_utc + lag`; a
negative lag is a ValueError. `parse.publication_lag()` reads `publication_lag_min` (minutes) from
`config/weather.yaml`.

## Stations (`stations.yaml`, `parse.station_for_venue`)

One station per JRA venue, each distinct. The rule: the active station nearest the racecourse
that observes temperature, precipitation and wind; where two are within 1 km of each other (the
coordinates' error), the one that also has sunshine. Distances from the racecourses' Wikipedia
coordinates to JMA's station positions, read 2026-11-25 from the etrn station pages and the
station master (地域気象観測所一覧).

| venue | station | station_id (block_no) | kind | prec_no | km | sunshine |
| --- | --- | --- | --- | --- | --- | --- |
| SAP | 札幌 | 47412 | official | 14 | 2.0 | measured |
| HAK | 函館 | 47430 | official | 23 | 4.2 | measured |
| FKS | 福島 | 47595 | official | 36 | 1.3 | measured |
| NII | 松浜 (新潟空港) | 1469 | AMeDAS | 54 | 6.7 | none |
| TOK | 府中 | 1133 | AMeDAS | 44 | 2.4 | estimated since 2021-03-02 |
| NAK | 船橋 | 1236 | AMeDAS | 45 | 7.5 | estimated since 2021-03-02 |
| CKY | 大府 | 1638 | AMeDAS | 51 | 9.0 | estimated since 2021-03-02 |
| KYO | 京田辺 | 0598 | AMeDAS | 61 | 9.1 | estimated since 2021-03-02 |
| HSN | 豊中 (大阪空港) | 0602 | AMeDAS | 62 | 6.9 | none |
| KOK | 八幡 | 0780 | AMeDAS | 82 | 12.2 | estimated since 2021-03-02 |

`stations.yaml` names the stations passed over (rain gauges, an airport station without sunshine
at a tie) and the nearest station with sunshine where the choice has none (NII: 新潟, 16.2 km;
HSN: 神戸, 16.7 km). Those two are `sunshine_station` entries: T1 borrows their sunshine (the
owner's answer (c) on #89), so the backfill fetches them too and their rows sit in the table under
their own ids (`47604`, `47770`). 大府 has observed since 2012-10-17. Any other venue code is a
KeyError.

## Availability

Japan's availability config (`countries/japan/config/availability.yaml`, `docs/specs/as-of.md`)
holds `weather: {kind: column, column: available_utc}`, with no fallback: a view at t holds
exactly the rows whose `available_utc` is at or before t, whatever their `obs_utc` or
`ingested_utc`, and a row without `available_utc` is never in a view. `asof_join` on
`station_id` takes the latest such row per key.

The lag is an owner decision: `publication_lag_min: 120` (the owner's answer (a) on #89,
2026-11-25), the live pages' hourly update padded for their undocumented latency, so every row's
`available_utc` is `obs_utc` + 2 h and a view at 15:40 JST holds the 13:00 JST row at the latest.
The stored values come from the next-day download and may carry corrections the live page did not
show yet; #343 logs the live AMeDAS values with their fetch times to measure an exact rule. The
backfill recomputes `available_utc` on every row whenever it writes, so a changed lag reaches the
whole table on the next run. What JMA says about publication (read 2026-11-25):

- 過去の気象データ・ダウンロード: 「前日分のデータまで表示、ダウンロードが可能です。当日のデータは
  利用できません。」 and 「データは、その後の品質管理等により、値が修正になる場合があります。」
  (https://www.data.jma.go.jp/risk/obsdl/top/attention.html). Its periods and update times
  follow 過去の気象データ検索 (obsdl FAQ).
- 過去の気象データ検索: 「…各データは、概ね下表の時刻までに更新します。」, 10分ごと～日ごとの値:
  「毎日1時頃」 (https://www.data.jma.go.jp/stats/data/mdrr/man/update_k.html).
- 最新の気象データ: 「降水の状況　10分ごとに更新」, 「風の状況／気温の状況…毎時50分頃に更新」,
  「これらは速報値ですので、修正する場合があります。」
  (https://www.data.jma.go.jp/stats/data/mdrr/man/update_n.html).

So a downloaded value for an hour of day D was first downloadable at about 01:00 JST of D+1; the
live pages showed a preliminary value within the hour, which a later correction may have changed.

## Fetch (`fetch.py`)

Through `horsetrack.core.ingest.fetcher.Fetcher` (cache-first, per-host rate, circuit breaker),
with `config/weather.yaml`'s http block: one request every 5 s to `www.data.jma.go.jp`, a stop on
403 or 429, three failures in a row open the circuit. `month_task(entry, year, month)` posts the
download form for one station-month (`entry` a `stations.yaml` entry; hourly 気温, 降水量, 日照時間,
風向・風速; quality columns; dates as literals). The form's names come from the page's script, not
from documentation; the first wired fetch confirmed them (2026-11-25: 府中, 新潟 and 神戸 for
2024-06, one POST each, no session or cookie, HTTP 200, 720 hourly rows each). `fetch_month`
quarantines a body without the CSV header and raises `FetchError`; `parse_page` parses a fetched
page and adds `ingested_utc`, the download time JMA prints on the file's first line (JST, to the
second), so a rerun from the cache writes the same value.

Caching: a month not yet ended in JST is refetched on every run (`mutable`). An ended month is
fetched once, unless the cached copy stops short of the month's end (its last hourly line is not
the month's 24時, because it was downloaded while the month ran): then it is fetched again, once.

`horsetrack japan jma-backfill [--start YYYY-MM] [--end YYYY-MM] [--venues TOK,NII]`
(`fetch.backfill`) fetches every month of the span, station by station: each venue's station, then
the sunshine station it borrows (default: all ten venues, twelve stations). The default span runs
from the price era's month (2014-06, D10: no race before 2014-06-07 has a price; the jra store's
races start in 2010) to the current JST month. It prints one line per month sent to JMA and per
failure, then a summary. It stops early on an open circuit (403, 429, or failed requests in a
row), after three failed months in a row (a body that is not the CSV, a parse error) or on Ctrl-C,
and exits 1 with "rerun the same command to resume". Whatever it fetched, from the network or the
cache, is merged into the table before it exits: a row of the same `(station_id, obs_utc)` is
replaced, every other row kept, `available_utc` recomputed from the configured lag on every row,
sorted by `station_id, obs_utc`, written through the store's writer (`core.store.build`, a tmp
file moved into place). A rerun over cached months sends no request and writes identical bytes.

JMA caps the data volume of one download (help1: no download above 100% of its gauge). By the
page's own gauge one station-month of the four elements is about 1.5% and a station-year about
17.5%. The default backfill is 12 stations × 150 months (2014-06 to 2026-11) = 1,800 requests,
about 2.5 hours at one request every 5 s; 300 of them are the two sunshine stations. A later rerun
costs one request per station for the current month (and one more per station for a month that
ended since the last run). Nothing runs on a race-day command.

## Pre-registration (E27: two terms)

`config/weather_terms.yaml` holds the family as the owner approved it (answer (c) on #89,
2026-11-25: both terms as drafted), written 2026-11-25 before any weather row was joined to a race.
That file is the definition; this section restates it.

- **Decision time and rows.** t = the race's `post_utc`; both terms condition on the final Win
  price. A row is visible at t when its `available_utc` (obs_utc + 120 min) is at or before t. h0
  is the latest whole JST hour with obs_utc <= t - 120 min (a post at 15:40 JST reads h0 =
  13:00 JST). "The 3 hourly rows before t" are the rows at h0, h0 - 1 h and h0 - 2 h exactly, and
  "the latest" temperature, sunshine and wind are the row at h0; an absent row is missing, never
  replaced by an older one. Sunshine at NII and HSN comes from 新潟 and 神戸 at the same hour.
- **Form.** The C4 offset form (`models/meta.py`, #87): u_i = ln q_i + Σ coef·x_i, a conditional
  logit on the winner, q_i the normalised final Win price, on D9's development folds over #87's
  development races; the holdout is never opened. Each term is fitted on its own.
- **T1, good-mood weather × ln q.** G_r = 1 when no rain fell in the three rows (precip_mm = 0.0 in
  each), 10.0 <= temp_c(h0) <= 25.0 °C and sunshine_h(h0) >= 0.5 h; else 0. Member G_r · ln q_i
  (coefficient γ: the market's slope is 1 + γ on good-mood races), with a going-band × ln q_i
  nuisance term (`races.going`, in the views from post_utc - 60 min). Reported as the draft has
  it, ΔLL against the market alone, with the market + nuisance fit beside it (descriptive). A null
  value among the five, an absent row or a null going leaves the race out.
- **T2, home-straight tailwind × run style.** T_r = wind_ms(h0) · cos(radians(wind_dir_deg(h0) +
  180 − b_v)) m/s, where JMA's direction is the one the wind blows from (clockwise from true
  north), so +180 is the direction it blows to, and b_v is the venue's running bearing (below):
  T_r > 0 is a tailwind toward the finish. 静穏 (no direction, wind_ms <= 0.2) is T_r = 0; a null
  speed, or a null direction above 0.2 m/s, leaves the race out. Runner score c_i = style_code −
  1.5 from `handicap/factors.py`'s `style` (front 0, stalker 1, midpack 2, closer 3: front −1.5 …
  closer +1.5). Member T_r · c_i (coefficient δ), tested two-sided.
- **Open (asked on #89), null in the file:** T1's going bands and reference band, T1's test
  (sidedness; the draft fixed T2's only), and how T2's null-style flag enters the fit (c_i = 0
  "plus a flag").
- **Freeze.** Family `japan-weather-e27-final-win`, size 2. The weather study embeds
  `weather_terms.yaml` verbatim in the spec it freezes with `core.eval.registry.freeze_family`
  (with the fold scheme and the scoring settings it uses, as #77 and #87 do), before it reads any
  race outcome or joins any weather row to a race, and refuses to freeze while a key in the file is
  null. Not frozen yet: the open details above.
- **Caveats, as drafted.** AMeDAS sunshine at TOK, NAK, CKY, KYO and KOK is a satellite estimate
  from 2021-03-02 (a series break, not adjusted). AMeDAS sunshine is blank at night (NaN), so a race
  whose h0 falls before sunrise at an AMeDAS station is left out by the missing rule. Because of the
  lag, h0 is 2 to 3 hours before the off.
- **Sources.** Weather from JMA; the final price, the going and the run style from the jra store
  (netkeiba results; style from #128's PP lines and #129's factors). No JRDB or JRA-VAN input
  (`WITHHELD_SOURCES`).

### Home-straight bearings (T2's b_v)

The direction runners travel along the straight in front of the stands that ends at the finish,
degrees clockwise from true north, to within about ±3°. One value per venue: turf and dirt are
concentric with parallel home straights, inner and outer turf courses share theirs, jump races
finish on it and Niigata's straight 1000 m runs along it (NAK's turf and dirt, measured apart,
agree within 0.5°). Measured 2026-11-25:

- **Axis**: GSI aerial photos (地理院タイル 全国最新写真（シームレス）, zoom 16; zoom 17 for NAK and KYO):
  the angle at which the rails and track edges along the straight project sharpest, then the line
  drawn over the photo to check it follows them. Where OpenStreetMap maps the track, its straight
  agrees within 0.5°.
- **Direction**: JRA's course plans (コース紹介, コース平面図, `www.jra.go.jp/facilities/race/<venue>/course/`,
  fetched through `core.ingest`) give the handedness and draw the stands, the finish post and the
  direction of running; the photos place the stands.

| venue | bearing | hand | stands | photo axis | OpenStreetMap |
| --- | --- | --- | --- | --- | --- |
| SAP | 359.0 | right | west | 178.8–180.0 | no track mapped |
| HAK | 123.0 | right | northeast | 122.5–123.4 | no straight mapped |
| FKS | 22.0 | right | west | 20.4–22.5 | no straight mapped |
| NII | 241.5 | left | northwest | 61.5–63.5 | way 332008108: 241.4 |
| TOK | 267.5 | left | north | 86.8–87.7 | ways 155529613, 566666116, 566666123: 267.3–267.6 |
| NAK | 15.0 | right | west | 14.9–15.3 | not fetched (server busy) |
| CKY | 234.0 | left | northwest | 53.5–54.4 | no track mapped |
| KYO | 52.7 | right | northwest | 51.4–52.9 | way 526656541 runs at 42°, off the rails: not used |
| HSN | 287.5 | right | south | 106.8–108.5 | no track mapped |
| KOK | 156.5 | right | east | 156.0–157.5 | way 137541144: 156.5 |

Sources: 出典：国土地理院（地理院タイル、全国最新写真（シームレス））を加工して作成 (bearings measured
from the photos); © OpenStreetMap contributors (ODbL); JRA コース紹介 (handedness and the plans).

## Attribution and licence

JMA's terms of use (https://www.jma.go.jp/jma/kishou/info/coment.html, read 2026-11-25):

- 「気象庁ホームページで公開している情報（以下「コンテンツ」といいます。）は、権利表記の記載がない限り
  「公共データ利用規約（第1.0版）」に準拠した利用条件の下で、利用することができます。」
- 「コンテンツを利用する際は出典を記載してください。」 The forms given: 「出典：気象庁ホームページ
  （当該ページのURL）」, 「出典：○○気象台ホームページ（当該ページのURL）」.
- 「コンテンツを編集・加工等して利用する場合は、上記出典とは別に、編集・加工等を行ったことを記載して
  ください。また編集・加工した情報を、あたかも国（又は府省等）が作成したかのような態様で公表・利用
  してはいけません。」 Example: 「気象庁「図・写真等の名称」（当該ページのURL）を加工して作成」.
- 「一部のコンテンツには、個別法令により利用に制約がある場合があります。」 The page names 気象業務法
  第十七条第一項 (forecasting) and 第二十三条 (warnings).

The 公共データ利用規約（第1.0版） (PDL1.0, set 2024-07-05,
https://www.digital.go.jp/resources/open_data/public_data_license_v1.0): 「本利用ルールは、
クリエイティブ・コモンズ・ライセンスの表示4.0 国際ライセンスに規定される著作権利用許諾条件
（以下「CC BY」といいます。）と互換性があります。」 and 「なお、数値データ、簡単な表・グラフ等は
著作権による保護の対象ではありませんので、これらについては本利用ルールの適用はなく、自由に利用
できます。」 Use begun under the earlier 政府標準利用規約 keeps its conditions (the same page).

What the repository does:

- No downloaded JMA file is committed: raw downloads stay under `$HORSETRACK_DATA_DIR/raw/jma/`,
  and the committed fixture is synthetic. The station ids, names and positions in
  `stations.yaml` are JMA data, cited there.
- Every report, table or figure built on these observations carries the attribution, and says
  the data were processed:

  > 出典：気象庁ホームページ（https://www.data.jma.go.jp/risk/obsdl/index.php）。
  > 気象庁「過去の気象データ・ダウンロード」の時別値を加工して作成（JST を UTC に換算、
  > 品質情報による欠測処理、風向を度に変換）。
  >
  > Source: Japan Meteorological Agency website (過去の気象データ・ダウンロード), hourly values,
  > processed by this project (JST to UTC, missing values by JMA's quality flags, wind direction
  > in degrees); not published by JMA.
