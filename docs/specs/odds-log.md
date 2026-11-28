# Odds snapshot log and its sidecars

Code: `src/horsetrack/countries/japan/odds_log.py`. The format is HKTrack's, unchanged: logs copied
from HKTrack read exactly like logs written here.

## The log

Path: `$HORSETRACK_DATA_DIR/odds_snapshots/<source>/<YYYY-MM-DD>/<VENUE>.jsonl.gz`
(`core.paths.odds_snapshot_path(source, race_date, venue)`): `jra` for JRA, `nar` for NAR
(ticket #38, `log-day --sources nar`, venues of `nar.yaml`). `log-day` appends one health line per
venue to `logs/japan/health.log`, ending ` source=<jra|nar>` (`logger/health.py`).

- One gzip member per capture, appended in one write and fsynced before `append_capture` returns.
  A capture with no rows writes nothing.
- Inside a member: one JSON object per line, compact separators, ASCII-escaped.
- Reading keeps whole, CRC-checked members only. A torn or damaged member costs that capture alone:
  reading resumes at the next member header (`1f 8b 08`).
- Rows are returned as logged: no keys added or dropped, no values coerced.

### Row keys

| key | type | notes |
| --- | --- | --- |
| `snapshot_utc` | ISO UTC text | when the capture was taken; orders the log for `latest_*` |
| `fetched_utc` | ISO UTC text or null | optional; when the page was fetched (absent in HKTrack logs) |
| `race_id` | text | `jra:<YYYY-MM-DD>:<VENUE>:<NN>`; NAR: `nar:<YYYY-MM-DD>:<VENUE>:<NN>` (venues of `nar.yaml`) |
| `source_race_key` | text | optional; NAR rows: the 12-digit netkeiba race id (`nar/pages.py`) |
| `pool` | text | `WIN`, `PLACE`, `QUINELLA`, ... |
| `combination` | text | runner or combination; `__pool__` on a pool-total row |
| `odds` | number or null | null for a scratched runner, and usually on a pool-total row |
| `odds_hi` | number or null | optional; the upper end of a ranged price (PLACE, QUINELLA_PLACE) |
| `status_raw` | text or null | the page's own text for the price (time text such as `14時00分現在`, `SCR`) |
| `page_state` | text or null | optional; the page's selling state (absent in HKTrack logs) |
| `source_update_utc` | ISO UTC text or null | the source's own "as of" time |
| `investment` | number or null | pool-total rows only: the money in the pool |
| `carryover` | number or null | optional; WIN5 rows only (race_id `jra:<YYYY-MM-DD>:WIN5`): the carryover in yen |
| `legs` | list of text | optional; WIN5 rows only: the five designated races' race_ids, in leg order |

Tier-2 pools (ticket #130): `BRACKET_QUINELLA`, `FORECAST`, `TRIO`, `TIERCE` combinations are
numbers joined by `,`, FORECAST and TIERCE in finishing order, the others sorted ascending
(`1,1` is a same-bracket pair).

Tier-2 and WIN5 rows share the venue's log with tier 1, so a day's log is far larger than its
tier-1 rows (an 18-runner 3連単 capture alone is 4,896 rows). Readers stream it with
`iter_members` (one decompressed member at a time) rather than `read_rows`:
- the live board (`board/feed.py`, docs/specs/board-feed.md) keeps tier-1 rows only, row by row;
- `logger compact` keeps every row, tier 2 and WIN5 included, one Parquet row per logged row, and
  holds the day column-wise (one Arrow table per member) rather than as one Python dict per row.

Optional keys are left out of a row, not written as null, when the source does not give them.

### Values on write

- Datetimes are written in UTC with `isoformat()` (`+00:00`), whatever zone they carry; a naive
  datetime is taken as UTC.
- Dates are written as ISO dates.
- Scalars with `.item()` (numpy) are written as their Python value; anything else as `str()`.

### Latest views

- `latest_prices(rows)`: `{(race_id, pool, combination): row}`, pool-total rows left out.
- `latest_totals(rows)`: `{(race_id, pool): row}`, pool-total rows with an `investment` only.
- Newest by `snapshot_utc` compared as instants; at the same instant the later row in the log wins.
  Rows without a readable `snapshot_utc` are never the latest.

## Post-time sidecar

Path: `<VENUE>.races.json` beside the log, one per meeting.

```json
{
  "updated_utc": "2026-11-28T00:30:00+00:00",
  "post_times": {
    "1": "2026-11-28T01:05:00+00:00",
    "11": "2026-11-28T06:40:00+00:00"
  }
}
```

- `updated_utc`: when the post times were last taken from the source, ISO UTC.
- `post_times`: race number (as a JSON string key) to post time, ISO UTC.
- `write_post_times(path, post_times, updated_utc)` replaces the file as a whole: a temporary file,
  fsynced, then `os.replace`. A failed write leaves the previous sidecar and no temporary file.
- `read_post_times(path)` returns `{race_number: aware UTC datetime}`, empty when the file is missing.

## Results sidecar (ticket #225)

Path: `<VENUE>.results.json` beside the log, one per meeting, written by `log-day` when logger.yaml
has a `results` block.

```json
{
  "published_utc": {
    "1": "2026-11-28T01:14:00+00:00",
    "2": "2026-11-28T01:47:20+00:00"
  },
  "unpublished_utc": {
    "2": "2026-11-28T01:46:20+00:00"
  }
}
```

- `published_utc`: race number (as a JSON string key) to the first poll instant, ISO UTC, at which
  the meeting's レース結果 race list linked the race's result page.
- `unpublished_utc`: the race's lower bound, the instant the last successful poll of the same run
  before that one returned (it did not show the race), ISO UTC. Only bounded races appear: a race
  shown on a run's first successful poll (a restart, a late start, a wake from sleep, or a result
  out before the first post) is a late sighting and has no bound, so its `published_utc` says when
  it was seen, not when it came out. A failed poll shows nothing and is never a bound. The
  published time lies in `(unpublished_utc, published_utc]`; a bracket wider than the poll
  interval spans failed polls.
- `record_published(path, races, seen_utc, unpublished_utc=None)` adds each race not yet there at
  `seen_utc` with the bound `unpublished_utc` (None: a late sighting); a race already there keeps
  its first instant and its bound, whatever a later call or run says. Written as the post-time
  sidecar is (temporary file, fsync, `os.replace`), and only when a race is new. Returns the
  `published_utc` mapping after the write.
- `read_published(path)` returns `{race_number: aware UTC datetime}`, empty when the file is missing.
- `read_unpublished(path)` returns `{race_number: aware UTC datetime or None}`, one entry per race
  of `read_published`; a sidecar written before the bound (no `unpublished_utc` key) reads every
  race as None, and a missing file reads as empty.

## Card log (ticket #426)

Path: `<VENUE>.cards.jsonl.gz` beside the log, one per meeting, written by `log-day` (JRA only)
when logger.yaml has a `cards` block: each race's 出馬表 (JRADB pw01dde) once per grid point of
`cards.grid_min` (minutes before post, stretched as tier 2's), from what the odds leave of each
round's budget (`logger/schedule.py`).

- Append-only, in the log's format: one gzip member per capture (`odds_log.append_capture`), read
  back with `iter_members`. A later capture never rewrites an earlier one, so every capture of a
  race stays, each with its own fetch time; the odds log is never written.
- A capture (`jra.capture.capture_cards`) reads the 出馬表 menu, the card race list, then the card
  page each asked race's row links (2 + races requests). A card page of another meeting or race is
  refused (WrongMeeting) and the capture writes nothing.
- One row per race and one per runner of each card read: `official_card.parse`'s `races` and
  `racecard_runners` rows (docs/specs/store.md), plus:

| key | type | notes |
| --- | --- | --- |
| `table` | text | `races` or `racecard_runners`: the store table the row's columns are |
| `fetched_utc` | ISO UTC text | when the row's card page was fetched (RawPage.fetched_utc) |
| `snapshot_utc` | ISO UTC text | when the capture started, shared by every row of the member |

- A runner row's `card_scraped_utc` equals its `fetched_utc`; `is_scratched` and the race row's
  `field_size` (runners not scratched) are as the page showed them at that fetch.
- Card pages themselves are not kept under `raw/` (`raw_pages` covers the odds pages only); the
  rows keep every field the parser reads.
- No as-of view reads the card log yet: its availability rule is the owner's question (#426).
