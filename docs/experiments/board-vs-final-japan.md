# Experiment report: board-fitted against final-fitted terms on the JRDB board (Japan, #103)

Ticket #103 (epic #73; 08 §4 E2, E45, R13). The study fits two sets of terms on each of two prices of the same
races and scores every fit on both prices: JRDB's fixed-time TYB board and the final Win price. E2 fits the Win
price's own power and quadratic terms; E45 fits #99's quadratic_context ordering coefficients. **Nothing is adopted**
(D16): no holdout is opened, and every fit x score cell is a member row of the private leaderboard.

**Every figure is withheld (D21).** The board is JRDB data, so every coefficient, ΔLL, bound, count, span and sign
of this run is JRDB-derived. This report says only what was run, where its outputs live and which registry records
hold them. It gives no number, sign or bound from them.

- Registry family `japan-board-vs-final-e2-e45-v1`, spec sha256
  `f6c2a4952950770928b546973fd7622020c7adc831a151f224f33bfa5032eb1f`, frozen (`--freeze`) before any outcome was
  read. The spec holds the hypotheses, the board's release-time rule and decision time, the race and price rules,
  the folds, both fits and scores, the pools' log-likelihoods, the bootstrap, the cluster unit, the record layout
  and the leaderboard treatment. The run choices it freezes are read from
  `src/horsetrack/countries/japan/config/board_vs_final.yaml`.
- Code at `63b5298`: `src/horsetrack/countries/japan/studies/board_vs_final.py` (tested in
  `tests/countries/japan/test_board_vs_final.py`). To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.board_vs_final --run`.
- Private outputs, written through `write_report` with the sources netkeiba and jrdb, under
  `$HORSETRACK_DATA_DIR/reports/`: the report `board-vs-final-japan-japan-board-vs-final-e2-e45-v1.md` (both
  cross-application tables, E2 and E45, and the per-fold coefficients), the cross tables as JSON
  (`…-tables.json`) and the private leaderboard (`…-leaderboard.md`, `…-leaderboard.json`).
- Registry records: twelve, appended to `$HORSETRACK_DATA_DIR/registry/japan.jsonl` by the run at `63b5298`
  (2026-11-28), each with the sources netkeiba and jrdb, status scored, ticket 103. The committed summary
  `docs/experiments/registry/japan/japan-board-vs-final-e2-e45-v1.md` lists them with every value withheld.

| E | Variant | Registry run id |
| --- | --- | --- |
| E2 | `WIN:fit=q_final` | `c4e2c1a081f32b109347f9e01d8b41bbc22d7ffc1c6f921ed308bf14a9fd548e` |
| E2 | `WIN:fit=q_board` | `27b0901baafb988122c45058e431e8aaeaa05b647ef42eb86db7d6282a5c4deb` |
| E45 | `PLACE:fit=q_final` | `9cbffcfa03d8faf6738d87e6dd01e16257c33d43f42eb2e735b89fc5e653a9df` |
| E45 | `PLACE:fit=q_board` | `db2e4c6eaf6df1da783c1d68c877c41f2a543258761a211aac67a3e504bfd7d7` |
| E45 | `QUINELLA_PLACE:fit=q_final` | `1634a86881f535e9f8305c19b51b189fc8823240b6ce1538e9f05955675e19f5` |
| E45 | `QUINELLA_PLACE:fit=q_board` | `e3ee70ee3f9519c2c2ee651a4607b815051668bde5a11cbe96dc69a30a2d31c5` |
| E45 | `TRIO:fit=q_final` | `6168b0bb869f95d683d1312fdfd4fe0fc596944f51708193da1ca5f337ba961f` |
| E45 | `TRIO:fit=q_board` | `64b8fcc91fae62e657e664c7a4471cfd2d3c618e63b9e4237c45c28afabbe53d` |
| E2 | `coefficients:fit=q_final` | `925836ebfc7da097193df0e5c9141d6235d26af456af82516947a88ef1334948` |
| E2 | `coefficients:fit=q_board` | `980e83a0d81fdb5340b683c9d24fff13d9a7774fc8709164aa27b1c364a16af1` |
| E45 | `coefficients:fit=q_final` | `5230213036de0949c8d1f0cf3eaec1eecba0a3f5e0149827ab8bdabf693cb5d6` |
| E45 | `coefficients:fit=q_board` | `5dabd363c207e4fcfcfc724d86d2441b8678bb7f029f5542472ddd5aeb9c6145` |

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
  board is, not how informative a model is; the coefficients are reported with their snapshot, per fold.

## Data and rules

- The board: `tyb_win_odds` at `tyb_odds_time` (HHMM on the Tokyo clock of the race date), read through
  `core.store.asof.read_asof` under a column rule on its release time, the odds time plus `release_lag_min`
  (`config/board_vs_final.yaml`). The lag is an assumption: the store keeps no JRDB release log (ledger row
  added). The board is read at each race's off (`races.post_utc`, availability.yaml's derived post); a board
  released after its race's off is not used. A missing or malformed odds time, or missing TYB odds, gives no board;
  nothing is filled from the final or KYI odds. availability.yaml's `jrdb_pre_race` rule (every `tyb_` column a
  label) is unchanged: the board is a separate view with its own rule.
- The final price: `runners.win_odds` from the netkeiba store, cleaned. Each price is 1/odds normalised over the
  race's starters (not scratched, with a final price; a race needs a winner).
- Development days only, from 2014-06-07 (D10): the store reads drop holdout days in the scan, then `split_off`;
  the holdout is never read or opened (D11-D15).
- Folds: D9's walk-forward quarters (`config/folds.yaml`: expanding, 1-week embargo, 22 quarters of training before
  the first scored quarter), computed over each hypothesis's matched races, the same for both prices.
- Bounds: D7's meeting (venue x day) clusters, 95% one-sided, a paired bootstrap of the per-race ΔLL
  (`config/scoring.yaml`: 10,000 replicates; seed 103 from `config/board_vs_final.yaml`).
- E45's races: matched races with one runner each in first, second and third and no dead heat among them. Place is
  the Bernoulli sum over starters of placing in the places the field of starters is paid (jra.yaml), Wide the
  Bernoulli sum over pairs of both finishing in the first three, Trio ln P of the first three as a set; a race
  where a pool is not sold to its field is not scored in that pool.

## Variants tried

Each variant is one registry record, scored on D9's walk-forward folds of development races; each carries its ΔLL
on both prices with its one-sided bounds, or its per-fold coefficients. Figures withheld (D21).

| E | Variant | What |
| --- | --- | --- |
| E2 | `WIN:fit=q_final`, `WIN:fit=q_board` | β, γ fitted on each price, scored on both against the scored price |
| E45 | `PLACE:fit=q_final`, `PLACE:fit=q_board` | quadratic_context fitted on each price, Place scored on both against Harville |
| E45 | `QUINELLA_PLACE:fit=q_final`, `QUINELLA_PLACE:fit=q_board` | the same fits, Wide scored on both |
| E45 | `TRIO:fit=q_final`, `TRIO:fit=q_board` | the same fits, Trio scored on both |
| E2 | `coefficients:fit=q_final`, `coefficients:fit=q_board` | β, γ per fold with meeting-clustered bounds |
| E45 | `coefficients:fit=q_final`, `coefficients:fit=q_board` | the eight quadratic_context coefficients per fold |

No variant failed. The registry ids of each variant are listed above.

## Alternatives rejected and why

- **In-sample fits on all matched races.** The diagonal cells (fit and score on the same price) would carry the
  fit's own optimism. Every fit is refit per fold and scored out of fold instead.
- **`models.meta.MetaSpec` for E2.** It needs at least one member; E2 has none. The fit uses `models.condlogit`
  directly with MetaSpec's market terms (ln q, (ln q)²), the same design.
- **Discounted Harville (#115's v0 family) for E45.** #99 named quadratic_context as the candidate ordering; the
  ticket's "ordering coefficients" are its.
- **Paid combinations from the dividends table for Place and Wide.** The places a pool pays depend on the field when
  sales opened, which the store does not hold; the field of starters is used for both the model and the outcome.
- **`asof_join` per runner at its race's off.** The ticket names `read_asof`; the board is read once at the latest
  off of the races asked for and each race keeps the rows released by its own off.
- **`tyb_post_time` as the off.** It is sparse in the store; the netkeiba post time is not.
- **Several release lags as counted variants.** One lag was frozen to keep the run to one fit per price and fold; a
  second lag is a follow-up.

## What this rules out

Withheld (D21). The cross-application tables, their one-sided bounds and the per-fold coefficients are in the
private report named above. No reading of them is given here: a sign or a bound would be a JRDB-derived figure.

## Leaderboard rows

Sixteen member rows, one per fit x score cell, all withheld (D21) and unranked: a cell is a ΔLL per race, not a
return per ¥100, so it carries no return. Option ids (family `japan-board-vs-final-e2-e45-v1`), each
`japan-board-vs-final-e2-e45-v1:<E>:<pool>:fit=<price>:score=<price>`:

- `E2:WIN`, `E45:PLACE`, `E45:QUINELLA_PLACE` and `E45:TRIO`, each with `fit=q_final:score=q_final`,
  `fit=q_final:score=q_board`, `fit=q_board:score=q_final` and `fit=q_board:score=q_board`.

## Follow-ups filed

None filed by this run. Proposed: the release lag as a counted variant (the board at its odds time, lag 0, beside
the frozen lag), and a JRDB release log so the lag can be measured rather than assumed.
