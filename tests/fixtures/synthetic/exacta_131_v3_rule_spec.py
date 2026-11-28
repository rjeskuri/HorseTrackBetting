"""The v3 #131 E2 family spec, japan/131-exacta-rule-v3, as frozen on 2026-11-27.

Synthetic test data, not a scraped or licensed file. Plain JSON types only, so its canonical JSON
(``json.dumps(spec, sort_keys=True, separators=(",", ":"))``) hashes to the v3 E2 entry of the spec
table in docs/specs/exacta-overload-prereg.md (V3_RULE_HASH in
tests/countries/japan/research/test_exacta_study.py, which checks it). Generated mechanically from
``exacta_study.family_specs()`` at git 83684eb; do not edit by hand.
"""

V3_RULE_ID = "japan/131-exacta-rule-v3"

V3_RULE_SPEC = {
    "benchmark": "no bet (loss 0)",
    "block_size": 5,
    "category": "fav_first, fav_second or no_fav of combination c against the favourite on the "
    "Win board at T-2 (combo_category)",
    "cluster": "meeting",
    "cuts": [0.6, 0.7, 0.8, 0.9],
    "data": "development race days only (is_holdout_day False, filtered before any outcome, odds "
    "or dividend is read); races from 2014-06-07 that have #114 out-of-fold Win "
    "probabilities, i.e. its walk-forward test quarters (from 2020-Q1)",
    "estimator": "rule",
    "exclusions": [
        "boost_may_be_withheld",
        "fewer than 5 starters",
        "more than one paying exacta (dead heat for 1st or 2nd)",
        "a refunded exacta",
        "no exacta row",
    ],
    "family_size": 16,
    "favourite": "lowest Win odds on the Win board at T-2 (the board at t), ties to the lowest "
    "number",
    "hypothesis_id": "E43",
    "judged_at": "final dividends, own impact from the payout identity with the race's logged "
    "pool totals",
    "lambda2": "full_development",
    "level": 0.95,
    "members": [
        {"category": "fav_second", "cut": 0.6, "scope": "all"},
        {"category": "fav_second", "cut": 0.6, "scope": "headline"},
        {"category": "fav_second", "cut": 0.7, "scope": "all"},
        {"category": "fav_second", "cut": 0.7, "scope": "headline"},
        {"category": "fav_second", "cut": 0.8, "scope": "all"},
        {"category": "fav_second", "cut": 0.8, "scope": "headline"},
        {"category": "fav_second", "cut": 0.9, "scope": "all"},
        {"category": "fav_second", "cut": 0.9, "scope": "headline"},
        {"category": "no_fav", "cut": 0.6, "scope": "all"},
        {"category": "no_fav", "cut": 0.6, "scope": "headline"},
        {"category": "no_fav", "cut": 0.7, "scope": "all"},
        {"category": "no_fav", "cut": 0.7, "scope": "headline"},
        {"category": "no_fav", "cut": 0.8, "scope": "all"},
        {"category": "no_fav", "cut": 0.8, "scope": "headline"},
        {"category": "no_fav", "cut": 0.9, "scope": "all"},
        {"category": "no_fav", "cut": 0.9, "scope": "headline"},
    ],
    "min_meetings": 100,
    "n_boot": 10000,
    "ordering": "discounted Harville; full_development = #115's lambda2, one fit on the "
    "development out-of-fold races, the scored races among them, so partly in-sample "
    "(ordering_v0.yaml; the fit id is recorded with each run); 1.0 = plain Harville",
    "owner_signoff": "2026-11-25, #131 (all nine points as proposed)",
    "pool": "FORECAST",
    "primary_metric": "expected return per 100 yen staked at the final dividend, takeout and "
    "breakage in, with its CI (owner reframe 2026-11-27 on #131: rank every "
    "option by expected return at the final price, CI beside it)",
    "rate": "takeout.race_takeout(race_id, 'FORECAST', race_class_raw=...).payout_rate",
    "relabel": "owner 2026-11-26 on #131: #115's single full-development lambda2 fit is labelled "
    "full_development (the v1 label said in-fold); the design is v1's unchanged",
    "revision": "v3 (review of 0c39a8d, 2026-11-27): rule-v2 took the favourite from the lowest "
    "final Win odds; v3 takes the favourite and the category from the board at T-2, "
    "as the signed spec's sections 5 and 6 say, and changes nothing else",
    "rule": "at T-2 buy exacta c of category X when q_c(t) / p_c(t) < k; q_c(t) from the exacta "
    "board at t, p_c(t) Harville on the Win board at t",
    "scopes": {
        "all": "every scored race",
        "headline": "favourite's Win odds at T-2 >= 3.0 and less popular; on #130 days, "
        "spec section 6: 'the real exacta pool total exists (the odds log's "
        "investment row). There, the lower tertile within each logged day "
        'defines "less popular"\'',
    },
    "scored_on": "#130-logged development meetings only, once there are min_meetings",
    "seed": 131,
    "selection_time": "T-2",
    "sources": ["jra"],
    "spec": "docs/specs/exacta-overload-prereg.md",
    "stake_yen": 100,
    "supersedes": "japan/131-exacta-rule-v2",
    "ticket": 131,
    "win_probabilities": "#114 out-of-fold calibrated p_cal, renormalised over the starters",
}
