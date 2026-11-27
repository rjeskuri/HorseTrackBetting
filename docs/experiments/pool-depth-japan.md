# Pool depth and stake capacity, Japan (N3, #432)

Family `japan-pool-depth-v1`, spec sha256 `1545d788a9ac6997365900f48e1abc48177a24b9451d985f4d8b754b1857f34c` (frozen before the run; docs/specs/pool-depth-prereg.md). Race days from 2014-06-07 to the run's last day (a span of the store: withheld (D21)), development days only, confirmed vote and odds records only. The JRA-VAN store is licensed: every figure computed from it is withheld (D21), in the private report `pool-depth-japan-f103a3b667bed45d.md` under `$HORSETRACK_DATA_DIR/reports/`. A run's private report is named in its line below where the report folder holds it: `<stem>-<first 16 hex digits of the run id>.md`, or for the first invocation's runs `<stem>-<first 12 hex digits of the spec hash>.md`.

## Hypothesis

Null: in every pool x class group x odds band, newcomer and maiden races included, the stake at which our own money eats half of a 5, 10 or 20% edge is below the flat stake, so betting big is not possible. A cell whose capacity quantiles sit above the flat stake shows it wrong for that cell. Nothing here tests whether an edge exists: the capacity is a property of the pool, given an edge.

## What ran

From the jravan store (`pool_votes`, `final_odds`, `races`, and `dividends` for the co-placer variant), streamed a year at a time: per priced combination, the money on it (W, its final votes x 100), the pool (V, the `__pool__` votes net of refunds x 100), its class group and its final-odds band. Per cell, the quantiles of W, of V (each race once) and of the capacity stake at each edge, the share of combinations whose capacity reaches each flat stake, the same by pool-size tertile and odds band, and the Win, Place and Quinella pool totals two minutes before the post against the final. Figures: withheld (D21).

## The identity and worked arithmetic

A single-winner pool pays rate x V / W per yen. Our stake s goes into both, so a bet worth 1 + e is worth (1 + e)(V + s)W / (V(W + s)) once it is in. The capacity stake solves that for 1 + e(1 - share_lost): s = (e - k)VW / ((1 + k)V - (1 + e)W), k = e(1 - share_lost); at share_lost 1/2, s = kVW / ((1 + k)V - (1 + e)W) with k = e/2. No breakage and no floor.

- Invented pool of 1,000,000 yen with 10,000 on the combination: the capacity is 246.43 yen at e = 0.05, 481.23 at e = 0.10 and 919.12 at e = 0.20.
- The same 10,000 yen in a pool of 1,000,000,000 yen: 476.20 at e = 0.10, close to the deep-pool limit W k / (1 + k) = 10,000 x 0.05 / 1.05. A shallower pool allows more, because the rest of the pool pays for part of our impact.
- The dividend itself (`models.decision.own_stake_dividend`): floor(100 x rate x (V + s) / (W + s)) to 10 yen, at least 110. Pool 1,000,000, W 10,000, rate 0.725: 7,250 per 100 yen with no stake, 6,590 with a stake of 1,000.
- Place and Wide (co-placer variant): the pre-registered rule pays 1 + (rV - sum W) / (P W_i) per yen, the sum over the P paying combinations. With A = rV - sum W and c = (1 + k) / (1 + e) the capacity is s = (1 - c)(A + P W_i) / ((c d0 - 1) P + 1 - r), d0 the dividend with no stake; at P = 1 it is the single-winner identity. The run checks the rule against every eligible paid dividend, Place and Wide apart, a paid combination without its votes or final price counting as not reproduced, and is recorded failed when either pool's share is below the spec's (95%); the shares themselves are withheld (D21).
- Place and Wide, re-registered (family `japan-pool-depth-coplacer-v2`): the rule of `core.pricing.stake_back_share_dividend` pays r(1 + B / (P W_i)) per yen, B = V - sum W the losers' money. Our stake adds to V and to sum W alike, so B stays and the capacity is s = (1 - c)(B + P W_i) / (P(c - 1) + c B / W_i), whatever r is; at P = 1 it is the single-winner identity again. The same gate applies.
- The exacta (#350): 100 yen on one combination moves its dividend by (V + s) / V x W / (W + s) - 1, with W the combination's final money.

## Variants tried

- `depth` at `f21e0aa`: scored, registry run `6573127e490b925a6a73d3e61fb306029643773691312459e705c53555980669`, private report `pool-depth-japan-1545d788a9ac.md`; every figure withheld (D21).
- `co-placer` at `f21e0aa`: failed, registry run `fc33a936a7a8684355e2335f0698490dab964f39aea6711ce372c8bad8e72ba2`, private report `pool-depth-coplacer-japan-1545d788a9ac.md`; every figure withheld (D21).
- `exacta-own-impact` at `f21e0aa`: scored, registry run `e2dccdba5ce4d77ef771a6771b5341bdac2d3704afb27aa8e76c0279e7b9a6f2`, private report `exacta-impact-history-japan-1545d788a9ac.md`; every figure withheld (D21).
- `depth` at `3ff70c5`: scored, registry run `caa088dd3cacaf0fb30ab1ae2375ffda1009dd17dd6b1c438ab916b249b695ff`, private report `pool-depth-japan-caa088dd3cacaf0f.md`; every figure withheld (D21).
- `co-placer` at `3ff70c5`: failed, registry run `c0bea021fdd0cca4f3c70e756907ff406002527b186159f347591239a67e1761`, private report `pool-depth-coplacer-japan-c0bea021fdd0cca4.md`; every figure withheld (D21).
- `exacta-own-impact` at `3ff70c5`: scored, registry run `9d968b30dd66c843122b752f0ca8aa463fd5f8dd8b4f277a4b4050dcee62cea7`, private report `exacta-impact-history-japan-9d968b30dd66c843.md`; every figure withheld (D21).
- `co-placer` (family `japan-pool-depth-coplacer-v2`, spec sha256 `82a471b219bfee2e65f53c5c12e2f66514757169d82184b6be45b35b2454e89a`) at `42b46cc`: scored, registry run `19e524668afd7572f5b7cb455b12f5a170f0865e148bfc59e0c25d2faa617fc5`, private report `pool-depth-coplacer-japan-19e524668afd7572.md`; every figure withheld (D21).
- `depth` at `3de5687`: scored, registry run `f103a3b667bed45d0c0f3aff18b31dce16e853cf5da4b09e8bc240fc46c0c546`, private report `pool-depth-japan-f103a3b667bed45d.md`; every figure withheld (D21).
- `co-placer` at `3de5687`: failed, registry run `420c46b4106171f09368a82ac8dd0e24c0bfef3992c0763512f0a26389bd92f8`, private report `pool-depth-coplacer-japan-420c46b4106171f0.md`; every figure withheld (D21).
- `exacta-own-impact` at `3de5687`: scored, registry run `77f70ef00b49e5c3c6a7c9e5b71c4410121d421265246ad3198f7867a6ea9050`, private report `exacta-impact-history-japan-77f70ef00b49e5c3.md`; every figure withheld (D21).
- `co-placer` (family `japan-pool-depth-coplacer-v2`, spec sha256 `82a471b219bfee2e65f53c5c12e2f66514757169d82184b6be45b35b2454e89a`) at `3de5687`: scored, registry run `aaeaaccd5dc4f9cfed503d5ba7d0167ddef5d9f87e3f401400941e5ac2f1d371`, private report `pool-depth-coplacer-japan-aaeaaccd5dc4f9cf.md`; every figure withheld (D21).

## Alternatives rejected and why

- TIERCE and BRACKET_QUINELLA: not in the cells. The trifecta has several times the trio's combinations, too many for exact quantiles in memory on this machine; the bracket quinella needs the horse-to-bracket map.
- Quantile sketches: rejected for exact quantiles over the six pools.
- Capacity with breakage and the 110-yen floor: rejected; breakage makes the stake a step function of the price, and the continuous stake is the stated quantity.
- The gross pool for V: rejected; refunded votes do not pay.
- The co-placer rule frozen in `japan-pool-depth-v1`, 1 + (rV - sum W) / (P W_i): rejected by its own rule check (its runs are recorded failed and kept); it leaves the paying combinations' own money untaxed, while JRA takes the payout rate on the whole net pool. The rule of `core.pricing.stake_back_share_dividend`, the stake-back-share rule as `config/jra.yaml` states it, was v1's diagnostic and is `japan-pool-depth-coplacer-v2`'s registered rule (withheld (D21)).

## What this rules out

Which cells the flat stake fits, and so where betting big is possible: withheld (D21). By the identity alone, whatever the pool, a stake below W k / (1 + k), k = e / 2, never eats half of the edge: about 2.4% of the money already on the combination at e = 0.05, 4.8% at 0.10 and 9.1% at 0.20.

## Follow-ups filed

Filed as proposed tickets:

- #489, the co-placer re-registration: done here by family `japan-pool-depth-coplacer-v2`, whose co-placer run is scored; the ticket can close.
- #490: the trifecta (TIERCE) and bracket-quinella depth, with a streaming quantile.
- #491: the capacity beside each leaderboard row.

Publishable figures need the logged public days (#130).
