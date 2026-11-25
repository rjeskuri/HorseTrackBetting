# Experiment report: board quality by horizon and the price-forecast EV bias (Japan, #86)

Ticket #86 (epic #73; 08 §4 E10, E18, E21, R5). This is a diagnostic run, and **nothing is adopted**. It measures
three things. E18 is how far the de-vigged board at each D2 horizon trails the final board in winner log loss. E10
is the SD of ln(O_final / O_tau) by pool, horizon and segment. E21 is the spurious EV, rate · (e^{σ²} − 1), that
comes from multiplying p̂ by Ô_F, checked by Monte Carlo. Every figure is an aggregate. The board figures come from
JRA's own odds pages, as logged, and the winners would come from the netkeiba results store. No JRDB or JRA-VAN
data is used, and no runner or race row appears here.

- Registry family `japan-board-quality-by-horizon`, spec sha256 `0359f73aa077...`. The family is not frozen
  (diagnostic). The records are under git `a7147f5`, and an earlier identical run sits under `a3c8dfe`, before the
  commit.
- Code: `countries/japan/studies/board_quality.py`. Reproduce with
  `uv run python -m horsetrack.countries.japan.studies.board_quality --run`, after compacting each logged day
  (`horsetrack japan compact --date D`).
- Data: odds logs for 2026-11-21 and 2026-11-22 (Kyoto, 24 races, all with a final board by the final-state rule).
  Both are development days. Post times come from HKTrack's stored card. The full report is under the data
  directory at `reports/board-quality-japan-0359f73aa077.md`.

## Owner decisions used

D2 sets the grid: T-60, T-30, T-16, T-5, T-2 and T-1 (`config/decision.yaml`). D7 sets the bootstrap: meeting
clusters, 95% one-sided, 10,000 replicates (`config/scoring.yaml`). Under D6 and #69, a dead heat counts each of its
k winners at 1/k, a void race is dropped and counted, and unpriced runners are removed and the rest renormalised.
D11 to D15 lock the holdout: holdout days are never read and the holdout is not opened. The de-vig is left open:
normalised, power-β and Shin run side by side.

## Results

**E18 was not measured.** The results store holds no race after 2026-11-15, so none of the 24 logged races has a
winner. The three E18 variants are failed registry records: "no logged race with a final board has a stored result".
The board-quality table needs the results of 2026-11-21 and 2026-11-22 to be built into the store.

**E10 and E21, measured on a single run.** The table gives the SD of ln(O_final / O_tau) over the runners priced on
both boards, with n = 262 to 274 runners. The 2026-11-21 log started after races 1 to 3 had gone off, so those
races have no board at any τ. Race 4's log starts at T-22, so it has no board at T-30 or T-60. PLACE uses the low
end of the range. Spurious EV is computed at the payout rate of 0.8 in both pools. At every
row the Monte Carlo agrees with the formula within 1.5 SE (200,000 draws).

| τ | WIN sd | WIN spurious EV | PLACE sd | PLACE spurious EV |
| --- | --- | --- | --- | --- |
| T-60 | 0.466 | 0.194 | 0.410 | 0.146 |
| T-30 | 0.425 | 0.159 | 0.367 | 0.115 |
| T-16 | 0.358 | 0.109 | 0.298 | 0.075 |
| T-5 | 0.240 | 0.047 | 0.185 | 0.028 |
| T-2 | 0.204 | 0.034 | 0.152 | 0.019 |
| T-1 | 0.179 | 0.026 | 0.128 | 0.013 |

Late card (races 10 to 12, 88 runners) and earlier races (174 to 186 runners) have similar SDs from T-5 on. The WIN
SD at T-2 is 0.204 for both. The low-turnover segment failed because these logs carry no pool totals.

## What this rules out, and what it does not

At the primary decision time, T-2, multiplying a price forecast by a probability conditioned on the board overstates
EV by about 3.4 points in Win and 1.9 in Place. That is larger than the usual size of an edge, so a gate on p̂ · Ô_F
is not usable (08 R5). These figures come from two days at one venue and ignore the race-level spread of σ, so they
are not an estimate of L6's skip threshold. 08 §4's anchors for σ were 0.15, 0.22 and 0.42. The T-5 to T-1 SDs here
lie between the first two.

## Follow-ups

- Build the results of the logged days into the store, then re-run E18.
- Log the pool totals so the low-turnover segment can be measured.
- Re-run on every logged day as the log grows.
