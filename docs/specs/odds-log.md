# Odds snapshot log and post-time sidecar

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
