# Known-vs-assumed ledger

Every claim the project relies on, with the label its evidence supports. Seeded on 2026-11-24 from HKTrack, the
read-only predecessor project. A claim moves to a stronger label only when new evidence is cited in its row.

## Labels

- **measured**: reproduced from a stored artifact (a saved page, fixture, report or config with its primary source)
  or from several runs, with the source named.
- **single-run**: seen once: one day, one venue or one run (one backtest, one model's test seasons, one card).
- **assumption**: untested. A claim carried over from HKTrack without a test is an assumption until a ticket tests
  it.

## Conventions

- Sources are paths relative to the HKTrack repository (`HKTrack docs/DECISIONS.md #13`) or report names. Section
  numbers use §, decision numbers use #.
- Figures derived from the licensed data source are never written here (owner decision D21 on #14). Such rows
  cite the document and section only and say "figures withheld (licensed source)".
- "Retest by" names the check that would confirm, upgrade or drop the claim. The six HKTrack design assumptions
  say "retest or drop": each needs a ticket that tests it before HorseTrackBetting relies on it.

## Hong Kong (HKJC)

| Claim | Label | Evidence (source, n, date) | Retest by |
|---|---|---|---|
| HKJC returns HTTP 200 with body text "No information." for a missing page (no meeting that day, a past or undeclared race card), so pages are classified by content, not status. | measured | HKTrack `docs/research/02-hkjc-historical-data-sources.md` §2.2-2.3 (results route, three empty dates, 2026-09-20); HKTrack `docs/research/03-hkjc-racecard-and-live-odds.md` §2.1 (race card, three dates on 2026-09-20, re-tested by a verifier on 2026-09-21); stored sample HKTrack `docs/research/samples/racecard_not_available_message.txt`. | Parser test on the stored sample when the HKJC adapter is ported. |
| Horse profile pages show current values (current rating, season stakes, current trainer), not values at the time of each run, so they are not features; rating at run comes from the run-by-run table. | measured | HKTrack `docs/research/02-hkjc-historical-data-sources.md` §5 (profile block fields, 2026-09-20); stored sample HKTrack `docs/research/samples/horse_HK_2023_J087_option1_all_seasons.html` (field labelled "Current Rating"); rule in HKTrack `docs/research/06-happy-valley-domain-and-features.md` §3.16 and HKTrack `CLAUDE.md` Gotchas. | Port of the HKJC profile parser. |
| Brand numbers are recycled, so horses are keyed on horse id, never on the displayed brand. | measured | HKTrack `docs/research/02-hkjc-historical-data-sources.md` §2.6: the 2010 page prints `(CK277)` for `HK_2008_K277`, the 2015 page mixes old and new brand forms (2026-09-20); stored sample HKTrack `docs/research/samples/otherhorse_HK_2008_K277_retired.html`. | Port of the HKJC adapter: count brands mapped to more than one horse id. |
| Horse names are recycled. | assumption | Stated in HKTrack `CLAUDE.md` Gotchas and `docs/research/02-hkjc-historical-data-sources.md` §2.6 without an observed case. Keying on horse id is safe either way. | Count names mapped to more than one horse id in the ported HK store. |
| HK weight carried = card weight (Wt.) minus the apprentice claim plus overweight. | single-run | HKTrack `docs/DECISIONS.md` #13 (2026-09-21: "verified on the live 2026-09-23 card"); HKTrack `docs/research/08-completeness-critique.md` §3 R2 (one card, Happy Valley R1 of 2026-09-23, 12 runners, read 2026-09-21). Historical cards are not served (research 03 §2.1), so no multi-meeting check exists. | When HK resumes: card-derived weight against results "Act. Wt." over several meetings. |
| The HK live-odds GraphQL API silently returns another meeting when the requested date and venue are not active, so every response's date and venue are checked. | measured | HKTrack `docs/research/03-hkjc-racecard-and-live-odds.md` §3.2: on 2026-09-20 two requests (HV 2026-09-23, HV 2026-09-16) returned the simulcast S1 meeting; on 2026-09-21 a verifier re-confirmed it with the roles reversed (a past date returned the upcoming meeting); HKTrack `CLAUDE.md`. Two days, two queries. | Port of the HKJC odds adapter: assert date and venue on a live meeting. |
| The stage-2 weight on the form model (alpha) is 0.00 in HK. | measured | HKTrack `docs/DECISIONS.md` #16 (2026-09-22, walk-forward, 6 seasons, 865 test races: alpha 0.00, beta 0.98) and #21 (2026-09-22, 18 seasons, six test seasons, 4,214 races: alpha 0.000, max over seasons 0.021). Zero is the lower bound of `COEF_BOUNDS = (0.0, 5.0)` in HKTrack `src/hktrack/models/stage2.py`, so the unconstrained value is zero or negative and was never estimated. | When HK model work resumes: an unconstrained fit with a CI. |

## Japan (JRA): modelling and betting results

| Claim | Label | Evidence (source, n, date) | Retest by |
|---|---|---|---|
| Research-24 scorecard: the form model alone trails the crowd at the final odds; the form model blended with the final odds improves log loss per race by a tiny pseudo-R2 gain whose CI only just excludes zero. | single-run | HKTrack `docs/research/24-process-review-2026-11-22.md` §2 (one live model's manifest, its test seasons, 2026-11-22). Figures withheld (licensed source). One model; its test seasons were also used for model selection. | The evaluation harness on development folds with a paired bootstrap CI. |
| The shipped JRA Win rule claimed a clearly positive EV that was negative with the same probabilities at the actual final odds. | single-run | HKTrack `docs/DECISIONS.md` #27 (2026-11-20; one backtest of one rule, 2024-26; report `outputs/reports/stage2-market-input.json`); cited in HKTrack `docs/research/25-horsetrackbetting-brief.md` "Why". Figures withheld (licensed source). | Any value claim is scored at the final dividends by the evaluation harness. |
| Every tested betting rule loses money. | single-run | HKTrack `docs/research/24-process-review-2026-11-22.md` §1 and §4 (the rules compared in the review, one backtest, 2026-11-22; figures withheld, licensed source). HKTrack `docs/DECISIONS.md` #26 and #27 report some Win rules with a positive point return whose interval spans zero, so the defensible claim is "no rule showed a profit that survived its uncertainty". Hong Kong agrees: #21 flat Win stakes on the favourite, the model's top pick and the combined top pick lose 15.7-16.1% (4,214 races, 2026-09-22). | Each betting rule re-tested by a pre-registered spec on development folds. |
| The form model blended with the early board (10-16 minutes out) trails the final odds. | single-run | HKTrack `docs/research/24-process-review-2026-11-22.md` §1 and §2 (one analysis, 2026-11-22). Figures withheld (licensed source; the board is a licensed tote snapshot). HKTrack `docs/DECISIONS.md` #27 finds the same direction on a separate fit (2026-11-20). | The evaluation harness on logged boards from development race days. |
| A JRA race's official result is published within 30 minutes of its post time, so a race posted more than 32 minutes before a target's decision time (post - 2 min) is known at that decision. | assumption | Proxy used by #138's development report (`track_bias_report.PUBLISH_LAG`, `DECISION_LEAD`); the store keeps no publication timestamp, and no published-result time has been logged. | Log the results page's first appearance per race on live days and compare with post time. |
| Late price moves: flagged Place prices fell and flagged Wide prices fell further between T-5 and the off. | single-run | HKTrack `docs/research/24-process-review-2026-11-22.md` §4 item 1 (one Saturday, 2026-11-21, Kyoto, tickets flagged by the model only, so selection-conditional). Sizes withheld: the tickets were selected by a model with licensed inputs. | Price-move study on every logged venue and day, unconditional and per pool. |

## Japan (JRA): site, rules and pricing

| Claim | Label | Evidence (source, n, date) | Retest by |
|---|---|---|---|
| JRA odds pages send `Cache-Control: max-age=20` and refresh about every 30-40 s, so 20 s is the polite polling floor. | single-run | HKTrack `config/scrape.yaml` `live_odds` comment ("research 2026-11-21"), one race day. HKTrack `docs/research/results/jra-2026-11-18/B-betting-rules.md` §10 still listed the refresh rate as unverified on 2026-11-18. | The ported odds logger records response headers and update times over several race days. |
| 1 request a second to www.jra.go.jp is polite. | assumption | HKTrack `config/scrape.yaml` `jurisdictions.jra.rate_limit` 1.0 a second, set by the owner on 2026-11-20 (raised from 0.6); no test against site pushback is recorded. | The ported ingest logs 403/429 and latency at that rate. |
| The card race list prints each race's post time. | single-run | HKTrack `tests/fixtures/jra/official_card_racelist_2026-11-15_KYO.html`: a 発走時刻 column, 12 races, one day (2026-11-15), one venue (Kyoto). | A second fixture from another venue and day. |
| The odds race list prints 発走済 for a race after its off. | single-run | HKTrack `tests/fixtures/jra/official_odds_racelist_2026-11-15_KYO.html`, captured after racing: every race marked 発走済; one day (2026-11-15), one venue (Kyoto). | A capture during racing, with races before and after their off. |
| What the odds race list shows for a race before its off. | assumption | Unverified: no fixture captured before the off. | Capture the page before the first race of a logged day. |
| JRA takeout since 2014-06-07: Win and Place 20%; bracket quinella, Quinella and Wide 22.5%. Before that date the rate varied per race. | measured | HKTrack `docs/research/results/jra-2026-11-18/B-betting-rules.md` §1.1 (the official JRA payout-rate notice quoted, 2026-11-18; the pre-2014 note is from a secondary source); HKTrack `config/jurisdictions/jra.yaml` `pools` and `eras.payout_rate_reform`. | On any JRA rate notice; premium-race overrides need a per-race lookup. |
| Dividends per 100 yen are rounded down to 10-yen steps, with a 110-yen minimum (JRA Plus 10) that falls back to 100 yen when paying it would exceed the pool. Final odds are therefore floored to 0.1. | measured | HKTrack `docs/research/results/jra-2026-11-18/B-betting-rules.md` §1.5 (競馬法 Art. 10 and the JRA Plus 10 rule quoted, 2026-11-18); HKTrack `config/jurisdictions/jra.yaml` `dividend_step: 10`, `min_dividend: 110`. | On any change to the JRA rules. |
| Odds displayed before the off are floored to 0.1 (not rounded), like the final odds. | assumption | Stated in HKTrack `config/jurisdictions/jra.yaml` `odds_display` and HKTrack `docs/research/results/jra-2026-11-18/B-betting-rules.md` §8 without a quoted primary source or a check against pool totals. | Compare displayed odds with odds recomputed from logged pool totals. |
| Adding back 0.05 (half a step) to displayed odds removes the flooring bias. | assumption | HKTrack `config/jurisdictions/jra.yaml` `odds_display.detruncate_below` and `detruncate_at_or_above` 0.05; assumes the floored remainder is uniform over the step. | Same comparison as the row above, by odds level. |

## HKTrack design assumptions (retest or drop)

| Claim | Label | Evidence (source, n, date) | Retest by |
|---|---|---|---|
| Benter two-stage architecture: a fundamental within-race model, then a conditional-logit combiner with the market log-probability. | assumption | HKTrack `docs/DECISIONS.md` #3 (2026-09-21), chosen from the literature, not tested against alternatives. HK alpha 0.00 (row above) means stage 1 added nothing there. | retest or drop |
| Overnight as-of contract: features use history strictly before the race date plus the race card, no same-day results. | assumption | HKTrack `docs/DECISIONS.md` #1 (2026-09-21); the pre-race alternative was deferred, not tested. | retest or drop |
| Pool selection: bet only a chosen subset of pools (HK: the 17.5% pools Win, Place, Quinella, Quinella Place, with Quinella display-only; JRA: Win, Place, Quinella and Wide enabled). | assumption | HKTrack `docs/DECISIONS.md` #2, #8 and #17 (2026-09-21 to 2026-09-22); HKTrack `config/jurisdictions/jra.yaml` `enabled` flags. Chosen by takeout and backtestability, not by a test of value per pool. | retest or drop |
| Unit-bucket staking (1-3 units by expected value), Kelly kept as an option. | assumption | HKTrack `docs/DECISIONS.md` #9 (2026-09-21), chosen for a small bankroll and transparency, not tested. | retest or drop |
| Closing-line value (log of final odds over odds at decision, and the share of bets whose final odds stayed at or above `min_odds`) as the primary live metric. | assumption | HKTrack `docs/DECISIONS.md` #12 (2026-09-21); owner decision D8 on #14 makes a measure of this kind a secondary diagnostic, never the primary metric. | retest or drop |
| A jurisdiction-agnostic core: only the site adapters and config name a jurisdiction; everything else works on normalised tables. | assumption | HKTrack `CLAUDE.md` "Architecture boundary"; HKTrack `docs/research/25-horsetrackbetting-brief.md` "Decided" 2 replaces it with country folders and a minimal shared core. | retest or drop |
