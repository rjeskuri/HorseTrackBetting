# Weather: JMA AMeDAS observations per JRA venue

Ticket #89 (epic #72). Code: `src/horsetrack/countries/japan/jma/` (`parse.py`, `fetch.py`,
`stations.yaml`) and `countries/japan/config/weather.yaml`. Tests:
`tests/countries/japan/test_jma.py` over the synthetic `tests/fixtures/synthetic/jma_amedas_hourly.csv`.

Status: the table, the parser, the station map and the fetch adapter exist; no request has been
sent to JMA and no weather row is stored. Two owner decisions gate the ingest (asked on #89):
whether to ingest JMA at all, and the publication lag. Until the lag is set,
`parse.publication_lag()` raises, so no row can be stamped with an `available_utc`.

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

The page build (`core.store.build`) never writes it. No code writes it yet; the writer comes with
the go-ahead.

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
HSN: 神戸, 16.7 km). 大府 has observed since 2012-10-17. Any other venue code is a KeyError.

## Availability

Japan's availability config (`countries/japan/config/availability.yaml`, `docs/specs/as-of.md`)
holds `weather: {kind: column, column: available_utc}`, with no fallback: a view at t holds
exactly the rows whose `available_utc` is at or before t, whatever their `obs_utc` or
`ingested_utc`, and a row without `available_utc` is never in a view. `asof_join` on
`station_id` takes the latest such row per key.

The lag is an owner decision and is unset. What JMA says about publication (read 2026-11-25):

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
403 or 429, three failures in a row open the circuit. `month_task(venue, year, month)` posts the
download form for one station-month (hourly 気温, 降水量, 日照時間, 風向・風速; quality columns;
dates as literals). The form's names come from the page's script (2026-11-25), not from
documentation, and hold until the first wired fetch confirms them. A month not yet ended in JST is
refetched; an ended one is cached for good. `fetch_month` quarantines a body without the CSV
header and raises `FetchError`; `parse_page` parses a fetched page and adds `ingested_utc`.

JMA caps the data volume of one download (help1: no download above 100% of its gauge). By the
page's own gauge one station-month of the four elements is about 1.5% and a station-year about
17.5%. A 2014-2026 backfill of the ten stations is 10 × 156 = 1,560 monthly requests, about
2.2 hours at one request every 5 s. Nothing runs on a race-day command, and no CLI exists.

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
