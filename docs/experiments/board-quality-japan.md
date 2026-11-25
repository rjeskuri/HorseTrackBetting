# Experiment report: board quality by horizon and the price-forecast EV bias (Japan, #86)

Ticket #86 (epic #73; 08 §4 E10, E18, E21, R5). This is a diagnostic run, and **nothing is adopted**. Every figure
is an aggregate. The boards come from JRA's own odds pages, as logged, and the winners from the netkeiba results
store. No JRDB or JRA-VAN data is used, and no runner or race row appears here.

- Registry family `japan-board-quality-by-horizon`, spec sha256 `8ea613a37508...`. The family is not frozen
  (diagnostic). Six records, one per variant, sit under git `415a14b`. An identical run made before that commit sits
  under `5cc6f6c`. The earlier runs under `a7147f5` and `a3c8dfe` used the in-sample β and had no results.
- Code: `countries/japan/studies/board_quality.py`. Reproduce with
  `uv run python -m horsetrack.countries.japan.studies.board_quality --run`, after compacting each logged day
  (`horsetrack japan compact --date D`).
- Data: odds logs for 2026-11-21 and 2026-11-22 at Kyoto. That is 24 races, two meetings, and every race has a final
  board by the final-state rule and a stored result. Both days are development days. Post times come from HKTrack's
  stored card. The results of both days were built into the store with the existing tooling: `horsetrack scrape plan`
  and `run` for the November calendar, race lists and results pages, then `horsetrack store build`. The rows from
  before 2026-11-16 are unchanged. The full report is under the data directory at
  `reports/board-quality-japan-8ea613a37508.md`.

## Owner decisions used

D2 sets the grid: T-60, T-30, T-16, T-5, T-2 and T-1 (`config/decision.yaml`). D7 sets the bootstrap: meeting
clusters, 95% one-sided, 10,000 replicates (`config/scoring.yaml`). Under D6 and #69, a dead heat counts each of its
k winners at 1/k, a void race is dropped and counted, and unpriced runners are removed and the rest renormalised.
D9 sets the fold used to fit the power β. D10 keeps every price read from 2014-06-07 on. D11 to D15 lock the holdout:
holdout days are never read and the holdout is not opened.

**The de-vig is still the owner's to choose.** Normalised, power-β and Shin run side by side. The power β is
estimated out of sample, and that estimate is a diagnostic candidate that waits on the owner's pick of de-vig. It
uses #77's `fit_power_beta` on the training rows of the D9 fold whose test window holds the earliest logged day,
which is the 2026-Q4 fold, the most recent one #77 fits. That gives β = 1.0206 (SE 0.0052) over 42,578 development
races from 2014-06-07 to 2026-09-26 (train_end 2026-09-26). The same β applies to every board, and none of the
scored boards enters the fit.

## Hypothesis

- E18: the de-vigged board at T-τ has a higher winner log loss than the de-vigged final board, and the gap shrinks as
  τ falls. 08 §4 E18 expects the gap at T-17 (08 R14) to roughly halve by T-5. The hypothesis is wrong if the gap at
  T-1 or T-2 is not above 0, or if it does not shrink with τ.
- E10: the SD of ln(O_final / O_τ) falls with τ, and the late card moves less than earlier races.
- E21: at the measured SDs, multiplying p̂ by Ô_F overstates the EV by rate · (e^{σ²} − 1). A Monte Carlo of 200,000
  draws agrees with that formula within 3 SE.

## Variants tried

There are six variants, each a registry record. Five were scored and one failed. All are in-sample on the two logged
development days. The only parameter is the power β, which is fitted out of sample as described above.

**E18, scored under each de-vig.** The table gives the mean gap in nats per race, board minus final, in the Win pool.
lo and hi are the one-sided 95% bounds from the meeting bootstrap. Races 1 to 3 on 2026-11-21 have no board at any τ,
because the log started after they ran. Race 4 has none at T-30 or T-60. That leaves 21 races (20 at T-30 and T-60).
No race was left out for a missing final, a void race or an unpriced winner.

| τ | normalised (lo, hi) | power-β (lo, hi) | Shin (lo, hi) | races |
| --- | --- | --- | --- | --- |
| T-60 | 0.187 (0.147, 0.246) | 0.187 (0.148, 0.247) | 0.201 (0.163, 0.258) | 20 |
| T-30 | 0.168 (0.116, 0.245) | 0.168 (0.117, 0.246) | 0.172 (0.114, 0.259) | 20 |
| T-16 | 0.145 (0.126, 0.170) | 0.145 (0.127, 0.170) | 0.154 (0.136, 0.177) | 21 |
| T-5 | 0.122 (0.105, 0.135) | 0.123 (0.105, 0.137) | 0.148 (0.109, 0.177) | 21 |
| T-2 | 0.104 (0.094, 0.112) | 0.106 (0.095, 0.114) | 0.133 (0.099, 0.159) | 21 |
| T-1 | 0.087 (0.069, 0.101) | 0.089 (0.069, 0.104) | 0.114 (0.072, 0.145) | 21 |

**These bounds come from two clusters.** With two meetings, a resample is AA, AB or BB, so lo and hi are close to the
two meetings' own means. They show that both meetings agree in sign, and they are not a credible interval. Under the
normalised de-vig, the gap at each τ is within about 0.001 of the winners' mean ln(O_τ / O_final), because the overround
hardly moves. Between the board at T-1 and the final board, the winners' odds shorten by about 9% on average. The final
board matches the store's final Win odds exactly.

**E10 and E21, scored.** The table gives the SD of ln(O_final / O_τ) over the runners priced on both boards, with
n = 262 to 274 runners. PLACE uses the low end of the range. Spurious EV is computed at the payout rate of 0.8 in both
pools. At every row, the Monte Carlo agrees with the formula within 1.5 SE.

| τ | WIN sd | WIN spurious EV | PLACE sd | PLACE spurious EV |
| --- | --- | --- | --- | --- |
| T-60 | 0.466 | 0.194 | 0.410 | 0.146 |
| T-30 | 0.425 | 0.159 | 0.367 | 0.115 |
| T-16 | 0.358 | 0.109 | 0.298 | 0.075 |
| T-5 | 0.240 | 0.047 | 0.185 | 0.028 |
| T-2 | 0.204 | 0.034 | 0.152 | 0.019 |
| T-1 | 0.179 | 0.026 | 0.128 | 0.013 |

**E10 by segment.** For late card, scored: races 10 to 12 (88 runners) and earlier races (174 to 186 runners) have
similar SDs from T-5 on, and the WIN SD at T-2 is 0.204 in both. These SDs are point estimates with no interval.

**E10 by low turnover, failed.** The compacted logs do hold WIN pool totals, but every one is stamped after the post
(the latest is 7 minutes after it). So the segment's rule, the latest total before the post, finds none. Per-race
Win sales before the post are #278.

## Alternatives rejected and why

- **A power β fitted on the scored races' own final boards.** The first version of this study did this. An in-sample
  β lowers the final's loss by about 1/(2n) nats per race, which is the size of the late-τ gaps, and so it biases the
  gap upward. It was replaced by the out-of-sample fit above.
- **A β refitted at each τ.** That would turn E18 into a comparison of fits, not of boards. One β serves every board.
- **Days or races as the bootstrap cluster.** D7 sets the meeting as the cluster, and it is the owner's choice.
  Two meetings are too few for that unit, so the fix is more logged days, not a smaller unit.
- **Pool totals from after the post as the turnover measure.** These are final sales, which are known only after the
  decision time, so using them would leak. The segment waits on #278.

## What this rules out

- That the board at T-2 or T-1 is as good as the final, on these two meetings. Under every de-vig the gap is about
  0.09 to 0.13 nats per race, and both meetings are above 0. Inference beyond these days is not possible with two
  clusters. A price-based model scored at T-2 therefore starts with a handicap of roughly this size against the
  final-price baseline.
- That the gap halves from T-16 to T-5, as 08's anchor expects. It falls only from 0.145 to 0.122 (normalised). Two
  meetings cannot settle this.
- That multiplying p̂ by Ô_F is a usable EV gate (08 R5). At T-2 it overstates EV by about 3.4 points in Win and 1.9
  in Place, which is larger than a usual edge.
- Not ruled out: the choice of de-vig. Normalised and power-β differ by less than 0.002 nats at every τ. Shin's gap is
  larger, but its bounds overlap the others'. Nothing here estimates L6's skip threshold, which also needs the spread
  of σ from race to race.

## Follow-ups filed

- #278: per-race Win sales before the post, for the low-turnover segment.
- Not yet filed: re-run E18, E10 and E21 as the logged days grow, so the meeting bootstrap has enough clusters to
  read, and build each logged day's results into the store as part of that run.
