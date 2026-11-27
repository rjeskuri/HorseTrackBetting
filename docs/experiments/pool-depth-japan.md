# Pool depth and stake capacity, Japan (N3, #432)

Family `japan-pool-depth-v1`, spec sha256 `1545d788a9ac6997365900f48e1abc48177a24b9451d985f4d8b754b1857f34c` (frozen before the run; docs/specs/pool-depth-prereg.md). Race days from 2014-06-07 to the run's last day (a span of the store: withheld (D21)), development days only. The JRA-VAN store is licensed: every figure computed from it is withheld (D21), in the private report `pool-depth-japan-1545d788a9ac.md` under `$HORSETRACK_DATA_DIR/reports/`.

## Hypothesis

Null: in every pool x class group x odds band, newcomer and maiden races included, the stake at which our own money eats half of a 5, 10 or 20% edge is below the flat stake, so betting big is not possible. A cell whose capacity quantiles sit above the flat stake shows it wrong for that cell. Nothing here tests whether an edge exists: the capacity is a property of the pool, given an edge.

## What ran

From the jravan store (`pool_votes`, `final_odds`, `races`, and `dividends` for the co-placer variant), streamed a year at a time: per priced combination, the money on it (W, its final votes x 100), the pool (V, the `__pool__` votes net of refunds x 100), its class group and its final-odds band. Per cell, the quantiles of W, of V (each race once) and of the capacity stake at each edge, the share of combinations whose capacity reaches each flat stake, the same by pool-size tertile and odds band, and the Win, Place and Quinella pool totals two minutes before the post against the final. Figures: withheld (D21).

## The identity and worked arithmetic

A single-winner pool pays rate x V / W per yen. Our stake s goes into both, so a bet worth 1 + e is worth (1 + e)(V + s)W / (V(W + s)) once it is in. The capacity stake solves that for 1 + e(1 - share_lost): s = (e - k)VW / ((1 + k)V - (1 + e)W), k = e(1 - share_lost); at share_lost 1/2, s = kVW / ((1 + k)V - (1 + e)W) with k = e/2. No breakage and no floor.

- Invented pool of 1,000,000 yen with 10,000 on the combination: the capacity is 246.43 yen at e = 0.05, 481.23 at e = 0.10 and 919.12 at e = 0.20.
- The same 10,000 yen in a pool of 1,000,000,000 yen: 476.20 at e = 0.10, close to the deep-pool limit W k / (1 + k) = 10,000 x 0.05 / 1.05. A shallower pool allows more, because the rest of the pool pays for part of our impact.
- The dividend itself (`models.decision.own_stake_dividend`): floor(100 x rate x (V + s) / (W + s)) to 10 yen, at least 110. Pool 1,000,000, W 10,000, rate 0.725: 7,250 per 100 yen with no stake, 6,590 with a stake of 1,000.
- Place and Wide (co-placer variant): the pre-registered rule pays 1 + (rV - sum W) / (P W_i) per yen, the sum over the P paying combinations. With A = rV - sum W and c = (1 + k) / (1 + e) the capacity is s = (1 - c)(A + P W_i) / ((c d0 - 1) P + 1 - r), d0 the dividend with no stake; at P = 1 it is the single-winner identity. The run checks the rule against the paid dividends and is recorded failed when it reproduces fewer than the spec's share of them (95%); the share itself is withheld (D21).
- The exacta (#350): 100 yen on one combination moves its dividend by (V + s) / V x W / (W + s) - 1, with W the combination's final money.

## Variants tried

- `depth`: scored, registry run `6573127e490b925a6a73d3e61fb306029643773691312459e705c53555980669`; every figure withheld (D21).
- `co-placer`: failed, registry run `fc33a936a7a8684355e2335f0698490dab964f39aea6711ce372c8bad8e72ba2`; every figure withheld (D21).
- `exacta-own-impact`: scored, registry run `e2dccdba5ce4d77ef771a6771b5341bdac2d3704afb27aa8e76c0279e7b9a6f2`; every figure withheld (D21).

## Alternatives rejected and why

- TIERCE and BRACKET_QUINELLA: not in the cells. The trifecta has several times the trio's combinations, too many for exact quantiles in memory on this machine; the bracket quinella needs the horse-to-bracket map.
- Quantile sketches: rejected for exact quantiles over the six pools.
- Capacity with breakage and the 110-yen floor: rejected; breakage makes the stake a step function of the price, and the continuous stake is the stated quantity.
- The gross pool for V: rejected; refunded votes do not pay.
- The rule of `core.pricing.stake_back_share_dividend`, r(1 + (V - sum W) / (P W_i)), the stake-back-share rule as `config/jra.yaml` states it: not the pre-registered co-placer rule. Its agreement with the paid dividends is a diagnostic in the private report (withheld (D21)).

## What this rules out

Which cells the flat stake fits, and so where betting big is possible: withheld (D21). By the identity alone, whatever the pool, a stake below W k / (1 + k), k = e / 2, never eats half of the edge: about 2.4% of the money already on the combination at e = 0.05, 4.8% at 0.10 and 9.1% at 0.20.

## Follow-ups filed

Proposed, not filed by this run: the co-placer variant re-registered as a new family on the rule of `core.pricing.stake_back_share_dividend`; the trifecta's depth by a streaming exact quantile; the capacity column beside the leaderboard rows; publishable figures from the logged public days (#130).
