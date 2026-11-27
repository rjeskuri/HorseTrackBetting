# Experiment report: board-fitted against final-fitted terms on the JRDB board (Japan, #103)

Ticket #103 (epic #73; 08 §4 E2, E45, R13). The study fits two sets of terms on each of two prices of the same
races and scores every fit on both prices: JRDB's fixed-time TYB board and the final Win price. E2 fits the Win
price's own power and quadratic terms; E45 fits #99's quadratic_context ordering coefficients. **Nothing is adopted**
(D16): no holdout is opened, and every fit x score cell is a member row of the private leaderboard.

**Every figure is withheld (D21).** The board is JRDB data, so every coefficient, ΔLL, bound, count, span, sign and
registry run id of this study is JRDB-derived (a run id hashes the whole private record, its metrics included). This
report says only what was run, where its outputs live and which registry family and variant hold each result. It
gives no number, sign, bound or run id from them; the run ids are in the private reports.

## Runs

Two families, each frozen (`--freeze`) before any outcome was read, each appended to
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` with the sources netkeiba and jrdb and ticket 103. A run is named here
by its family and variant.

- **`japan-board-vs-final-e2-e45-v2`, the corrected run.** Spec sha256
  `f5a708e53d4af2c96c8709cc968da8caeafe1198c35060d2a814cac2b7000516`, run at `396f8eb` (2026-11-28) on development
  data, every record status scored. E45's Place places and Wide sale come from each race's paid combinations (see
  Data and rules); E2's cells are recomputed under the same spec. Private outputs under
  `$HORSETRACK_DATA_DIR/reports/`, written through `write_report` with the sources netkeiba and jrdb: the report
  `board-vs-final-japan-japan-board-vs-final-e2-e45-v2.md` (both cross-application tables, the per-fold
  coefficients, the board's lead to the off per fold and E45's races per pool by paid-record status), the cross
  tables as JSON (`…-tables.json`) and the private leaderboard (`…-leaderboard.md`, `…-leaderboard.json`). The
  committed summary `docs/experiments/registry/japan/japan-board-vs-final-e2-e45-v2.md` lists the records with every
  value withheld.
- **`japan-board-vs-final-e2-e45-v1`, superseded and kept.** Spec sha256
  `f6c2a4952950770928b546973fd7622020c7adc831a151f224f33bfa5032eb1f`, run at `63b5298` (2026-11-28). Its rule took
  E45's Place places and Wide sale from the field of starters (jra.yaml's places for the starters), so a race with
  a late scratch from eight to seven scored third place as a loss although third was paid. Its records stay in the
  registry as recorded and its private outputs keep their v1 file names; committed summary
  `docs/experiments/registry/japan/japan-board-vs-final-e2-e45-v1.md`.

Code: `src/horsetrack/countries/japan/studies/board_vs_final.py` (tested in
`tests/countries/japan/test_board_vs_final.py`); the run choices each spec freezes are read from
`src/horsetrack/countries/japan/config/board_vs_final.yaml`. To reproduce, run
`uv run python -m horsetrack.countries.japan.studies.board_vs_final --run`. `--out DIR` must lie under
`$HORSETRACK_DATA_DIR`, and every writer of the outputs refuses one that does not. A run in which a table cannot be
produced (a store file it fingerprints missing included) appends its failed records, writes only an attempt-named
report (`…-attempt-<UTC time>.md`, with a `-<n>` suffix when that name is taken, so no attempt replaces another),
leaves the last complete outputs as they are and exits non-zero.

## Hypothesis

- **E2.** The quadratic/power terms fitted on the board are partly a price-forecast correction: |γ_final| <
  |γ_board|. The fit is the conditional logit u = β ln q + γ (ln q)² on the price fitted on (#87's stage-2 market
  terms, unbounded per R13, ridge 0, a dead heat at 1/k), refit on each of D9's walk-forward folds. Each fit is
  scored on both prices as the per-race winner ΔLL against the scored price itself. The claim is shown wrong if the
  board-fitted terms gain as much on the final as the final's own terms do, or if γ fitted on the final is no
  smaller in size than γ fitted on the board.
- **E45.** The ordering coefficients depend on the horizon of the Win probabilities fed to them. #99's
  quadratic_context ordering is refit per fold on each price and scored on both prices against Harville on the
  scored price, as Place, Wide and Trio log-likelihood per race. The claim is shown wrong if a fit carries over to
  the other price with no loss against that price's own fit.
- Both run on the races priced both ways on every starter only, so every fit and every score sees the same races and
  seasons (the season-mismatch confound of 08 §4). R13: a stage-2 weight measured on the board says how stale the
  board is, not how informative a model is; the coefficients are reported with their snapshot, per fold, beside the
  board's lead to the off in the same folds.

## Data and rules

- The board: `tyb_win_odds` at `tyb_odds_time` (HHMM on the Tokyo clock of the race date), read through
  `core.store.asof.read_asof` under a column rule on its release time, the odds time plus `release_lag_min`
  (`config/board_vs_final.yaml`). The lag is an assumption: the store keeps no JRDB release log (ledger row
  added). The board is read at each race's own off (`races.post_utc`, availability.yaml's derived post); a board
  released after its race's off is not used. A missing or malformed odds time, or missing TYB odds, gives no board;
  nothing is filled from the final or KYI odds. availability.yaml's `jrdb_pre_race` rule (every `tyb_` column a
  label) is unchanged: the board is a separate view with its own rule.
- The final price: `runners.win_odds` from the netkeiba store, cleaned. Each price is 1/odds normalised over the
  race's starters (not scratched, with a final price; a race needs a winner).
- Development days only, from 2014-06-07 (D10): the store reads drop holdout days in the scan, then `split_off`;
  the dividends rows are read for the development races only, filtered in the scan. The holdout is never read or
  opened (D11-D15).
- Folds: D9's walk-forward quarters (`config/folds.yaml`: expanding, 1-week embargo, 22 quarters of training before
  the first scored quarter), computed over each hypothesis's matched races, the same for both prices.
- Bounds: D7's meeting (venue x day) clusters, 95% one-sided, a paired bootstrap of the per-race ΔLL
  (`config/scoring.yaml`: 10,000 replicates; seed 103 from `config/board_vs_final.yaml`).
- E45's races: matched races with one runner each in first, second and third and no dead heat among them. Trio is
  ln P of the first three as a set. Place is the Bernoulli sum over starters of placing in the first p, where p is
  the number of horses the race's Place pool paid (the dividends table's non-refund PLACE rows). Wide is the
  Bernoulli sum over pairs of both finishing in the first three. These are v2's rules. A race scores in Place only
  when p is a count of places jra.yaml pays and the paid horses are exactly the first p finishers. It scores in
  Wide only when the paid QUINELLA_PLACE pairs are exactly the pairs of the first three. A pool does not score a
  race that has no paid Win (no paid record), no paid row of that pool (the pool was not sold, or its record is
  missing) or paid combinations that differ from the result. The private report counts the races of each status
  per pool. The dividends table matches the jravan store's payouts one to one (ledger, "Payouts per pool
  2010-2026"). v1 instead took p, and whether Wide was sold, from jra.yaml's places for the field of starters.
- The board's lead to the off (R13): per fold, the median minutes from each matched race's latest board release to
  its off, over the fold's training races and its test races, for E2's and E45's rows (private report).

## Variants tried

Each variant is one registry record per family, scored on D9's walk-forward folds of development races. Each one
carries its ΔLL on both prices with its one-sided bounds, or its per-fold coefficients. Figures withheld (D21).

| E | Variant | What |
| --- | --- | --- |
| E2 | `WIN:fit=q_final`, `WIN:fit=q_board` | β, γ fitted on each price, scored on both against the scored price |
| E45 | `PLACE:fit=q_final`, `PLACE:fit=q_board` | quadratic_context fitted on each price, Place scored on both against Harville |
| E45 | `QUINELLA_PLACE:fit=q_final`, `QUINELLA_PLACE:fit=q_board` | the same fits, Wide scored on both |
| E45 | `TRIO:fit=q_final`, `TRIO:fit=q_board` | the same fits, Trio scored on both |
| E2 | `coefficients:fit=q_final`, `coefficients:fit=q_board` | β, γ per fold with meeting-clustered bounds |
| E45 | `coefficients:fit=q_final`, `coefficients:fit=q_board` | the eight quadratic_context coefficients per fold |

Both families hold these twelve variants. In `…-v1`, the Place and Wide records were scored under the superseded
place rule (places by the field of starters); they stay as recorded and are not compared here. In `…-v2`, every
variant is scored under the corrected rule (places and Wide's sale from the paid combinations). No variant failed
in either run.

## Alternatives rejected and why

- **In-sample fits on all matched races.** The diagonal cells (fit and score on the same price) would carry the
  fit's own optimism. Every fit is refit per fold and scored out of fold instead.
- **`models.meta.MetaSpec` for E2.** It needs at least one member; E2 has none. The fit uses `models.condlogit`
  directly with MetaSpec's market terms (ln q, (ln q)²), the same design.
- **Discounted Harville (#115's v0 family) for E45.** #99 named quadratic_context as the candidate ordering; the
  ticket's "ordering coefficients" are its.
- **Places by the field of starters (v1's rule).** The places a pool pays follow the field when sales opened
  (jra.yaml), not the starters: a late scratch from eight to seven still pays three places. v1 used the starters
  and is superseded by v2; its records are kept.
- **The field when sales opened.** #99's `models.exotic` settles by it (`field_at_sale`), but the store does not
  hold it. `races.field_size` counts the starters, and the runner rows also hold horses scratched before sales
  opened. The paid combinations record what each pool paid, so v2 reads those.
- **`asof_join` per runner at its race's off.** The ticket names `read_asof`. The board is read once at the latest
  off of the races asked for, and each race keeps only the rows released by its own off.
- **`tyb_post_time` as the off.** It is sparse in the store; the netkeiba post time is not.
- **Several release lags as counted variants.** One lag was frozen to keep the run to one fit per price and fold; a
  second lag is follow-up #602.

## What this rules out

Withheld (D21). The cross-application tables, their one-sided bounds, the per-fold coefficients and the board's
leads are in v2's private report, named above. v1's are in its own report, under the superseded place rule. No
reading of them is given here, because a sign or a bound would be a JRDB-derived figure.

## Leaderboard rows

Sixteen member rows per family, one per fit x score cell. All are withheld (D21) and unranked: a cell is a ΔLL per
race, not a return per ¥100, so it carries no return. v2's rows are in its private leaderboard. v1's rows stay in
v1's private leaderboard file, superseded. Each option id is `<family>:<E>:<pool>:fit=<price>:score=<price>`:

- `E2:WIN`, `E45:PLACE`, `E45:QUINELLA_PLACE` and `E45:TRIO`, each with `fit=q_final:score=q_final`,
  `fit=q_final:score=q_board`, `fit=q_board:score=q_final` and `fit=q_board:score=q_board`.

## Follow-ups filed

- #602: the release lag as a counted variant (the board at its odds time, lag 0, beside the frozen lag).
- #603: a JRDB release log (the fetch time of each TYB file), so the lag can be measured rather than assumed.
