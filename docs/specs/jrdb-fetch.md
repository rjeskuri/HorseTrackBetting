# The JRDB fetch

Code: `src/horsetrack/countries/japan/jrdb/fetch.py` (#718). It downloads one race day's JRDB
member files with the owner's login into `$HORSETRACK_DATA_DIR/raw/jrdb/`, where
`horsetrack japan jrdb build` (`jrdb/tables.py`) reads them. JRDB is licensed for private use
(D21): the files, and anything built on them, stay under the data dir. This spec holds rules
only, no JRDB value or count.

## Command

```
horsetrack japan jrdb fetch [--date YYYY-MM-DD] [--families KYI,CYB] [--dry-run]
                            [--retry-missing] [--print-launchd]
```

| option | meaning |
|---|---|
| `--date` | the race day; default tomorrow in Asia/Tokyo (a run at 20:00 JST fetches the next day's files) |
| `--families` | comma-separated family codes; default the eve set below. An unknown code or a bad date: exit 2, nothing sent or written |
| `--dry-run` | resolves the login, prints the URL path of every file a real run now would request (held files and missing files waiting to be rechecked are listed apart, without a URL), sends and writes nothing |
| `--retry-missing` | asks again for files recorded missing, however recently |
| `--print-launchd` | prints the nightly LaunchAgent plist; needs no login, sends and writes nothing |

## Files

A zip family's day file is `https://jrdb.com/member/datazip/<Fam>/<yyyy>/<FAM><yymmdd>.zip`
(`<Fam>` is the code capitalised, `Kyi`); an lzh-only family's is
`/member/data/<Fam>/<FAM><yymmdd>.lzh`. Every publication time below is an assumption, from public
knowledge of JRDB's member data, until measured: each manifest entry records `published_utc`
(the response's Last-Modified) and `fetched_utc` (#603).

| family | what it is | when JRDB publishes it (assumed) | fetched |
|---|---|---|---|
| BAC | 番組データ: the race programme (races, conditions, distances) | by the eve | by default |
| KAB | 開催データ: the meeting (weather, going) | the eve; may be updated on race day | by default |
| KYI | 競走馬データ: per runner, JRDB's ratings and indices | the eve | by default |
| CYB | 調教分析データ: training analysis per runner | the eve | by default |
| CHA | 調教本追切データ: the main workout per runner | the eve | by default |
| KKA | 競走馬拡張データ: per-runner extension of KYI | the eve | by default |
| UKC | 馬基本データ: the master record of each horse running | the eve | by default |
| TYB | 直前情報データ: race-day information (paddock, odds) | race day, updated through it | `--families` only; a held file is never refreshed |
| SED, SKB | 成績データ and its extension: results | after the races | `--families` only |
| SEC | results served only as .lzh | after the races | `--families` only; stored as served, not unpacked (no decoder) |

JOA is not fetched: it has no zip day file, and an .lzh alone in `raw/jrdb/JOA/` would stop the
build.

## What is written

- `raw/jrdb/<FAM>/<FAM><yymmdd>.zip` as served, and every `.txt` member extracted beside it under
  its base name (`raw/jrdb/KYI/KYI261205.txt`). A member whose name is absolute or climbs out
  (`..`) stays in the zip, not extracted; nothing is written outside the family folder. The
  build reads each delivery once (`tables.collect`).
- `raw/jrdb/manifest.json`, one JSON object, rewritten after each stored file or missing answer.
  Each stored archive appends to `files` an entry `{file (relative to raw/jrdb), url_path, size,
  sha256, fetched_utc, published_utc}`, times ISO 8601 UTC (`+00:00`), `published_utc` null
  without Last-Modified. A 404 or 410 goes to `missing` as `{url_path, status, checked_utc}`,
  one entry per url_path, updated in place. Every other key and entry is kept as it was read.
- Every file (an archive, a member, the manifest) is written to a temp file in its folder and
  moved into place with `os.replace`; the members and the archive are on disk before their entry.
  Nothing else is written in the data dir (no `core.ingest` cache copy).

A manifest that does not read (empty, not JSON, not an object, `files` or `missing` not a list)
stops the run before anything is sent and is left byte for byte.

## Held, missing, recheck

A file is held, and never requested, when a `files` entry names its url_path or its file, or
its archive is on disk. A missing file is asked for again only once `MIN_RECHECK` (1 hour) has
passed since its `checked_utc`, or with `--retry-missing`; an entry without `checked_utc` (the
backfill's shape) can be asked for at once. A missing file then fetched leaves `missing`.

## Stop rule

Every request goes through `core.ingest.fetcher.Fetcher.request`: one at a time, starts at least
2 s apart, the `HORSETRACK_USER_AGENT` User-Agent, no retry. A 401, 403, 429, any 5xx, a
transport error (refused, reset, cut short), any other non-200 answer, a redirect off
`https://jrdb.com` (another host, or http), or a 200 whose body is not its format (zip: starts
`PK\x03\x04` and opens with every CRC good; lzh: `-lh?-` at offset 2) stops the run after that
one request: an `ERROR` line that never shows the response body, exit 1, nothing of that file
stored. Files stored before the stop stay stored and listed. 404 and 410 are not stops: however
many files are missing, each is requested once, recorded, and the run exits 0.

## Closed hours

JRDB closes downloads on Saturday and Sunday from 08:00 to 19:00 JST (carried over from HKTrack's
downloader). A run that starts then sends nothing, says so, leaves the manifest alone and exits
0; a run that reaches the window stops before the next request. The check is made before each
request, the Fetcher's 2 s spacing aside.

## Credentials

The owner's decision on #718. The account is `HORSETRACK_JRDB_ACCOUNT`, from the process
environment, else the repo-root `.env` (`core.env.get_setting`), stripped; blank is missing. The
password lives only in the macOS keychain, as the generic-password item of service `jrdb` and
that account, added once with

```
security add-generic-password -s jrdb -a <account> -w
```

which prompts for the password (never on the command line). The fetch reads it with
`/usr/bin/security find-generic-password -s jrdb -a <account> -w`: stdin closed, both output
streams captured, only the one trailing newline dropped and the rest used verbatim. No password
is read from the environment or the `.env` (`HORSETRACK_JRDB_USER` and
`HORSETRACK_JRDB_PASSWORD`, read before the decision, are ignored, even when right), and no other
keychain item is used. No account: exit 1, a message naming `HORSETRACK_JRDB_ACCOUNT`, nothing
sent. An account but no password (no item, no `security` tool, a locked keychain): exit 1, a
message naming the keychain item and the command above with a placeholder account, nothing sent.
The password is on no argv; the account is on the `security` argv only and, being personal data,
in no output. The login goes as HTTP Basic auth only to `https://jrdb.com`
(`core.ingest.fetcher.OriginAuth`: stripped from any hop to another origin) and never appears in
output, logs, exception text, the manifest or any file.

## The nightly LaunchAgent

`--print-launchd` prints a plist, label `horsetrack.japan.jrdb-fetch`, that runs at 20:00 JST
on Friday, Saturday and Sunday (the eves of Saturday, Sunday and Monday racing; another race day
is fetched by hand), converted to the Mac's local zone when printed:

```
/bin/sh -c '<venv>/bin/horsetrack japan jrdb fetch && <venv>/bin/horsetrack japan jrdb build'
```

with absolute paths to this install's entry point (launchd's environment has no PATH to it) and
the repo root as working directory, so the data dir and the user agent come from its `.env`. The
build runs only when the fetch exits 0 (missing files and the closed window are 0); after a stop
it does not, the stored files wait for the next good run, and the error is in the log. Install
it from the main checkout:

```
mkdir -p "$HORSETRACK_DATA_DIR/logs/japan"
horsetrack japan jrdb fetch --print-launchd > ~/Library/LaunchAgents/horsetrack.japan.jrdb-fetch.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/horsetrack.japan.jrdb-fetch.plist
```

Print and reinstall it after a change of the Mac's time zone or a daylight-saving change (the
hour is fixed at print time). The unattended run needs the login keychain unlocked (the Mac's
user logged in) and `security` allowed to read the item: if macOS asks on the first run, choose
"Always Allow", or the run finds no password and stops with the keychain message. Holdout days are fetched and built like any other; no study reads them (D14).
