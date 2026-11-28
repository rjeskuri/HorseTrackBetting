"""The v2 #131 E2 family spec, japan/131-exacta-rule-v2, as frozen on 2026-11-27.

Synthetic test data, not a scraped or licensed file. Plain JSON types only, so its canonical JSON
(``json.dumps(spec, sort_keys=True, separators=(",", ":"))``) hashes to the v2 E2 entry of the spec
table in docs/specs/exacta-overload-prereg.md (V2_RULE_HASH in
tests/countries/japan/research/test_exacta_study.py, which checks it). Generated mechanically from
``exacta_study.family_specs()`` at git 0c39a8d; do not edit by hand.
"""

V2_RULE_ID = "japan/131-exacta-rule-v2"

V2_RULE_SPEC = {
    "benchmark": "no bet (loss 0)",
    "block_size": 5,
    "cluster": "meeting",
    "cuts": [0.6, 0.7, 0.8, 0.9],
    "data": "development race days only (is_holdout_day False, filtered before any "
    "outcome, odds or dividend is read); races from 2014-06-07 that have #114 "
    "out-of-fold Win probabilities, i.e. its walk-forward test quarters (from "
    "2020-Q1)",
    "estimator": "rule",
    "exclusions": [
        "boost_may_be_withheld",
        "fewer than 5 starters",
        "more than one paying exacta (dead heat for 1st or 2nd)",
        "a refunded exacta",
        "no exacta row",
    ],
    "family_size": 16,
    "favourite": "lowest final Win odds, ties to the lowest number",
    "hypothesis_id": "E43",
    "judged_at": "final dividends, own impact from the payout identity with the "
    "race's logged pool totals",
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
    "ordering": "discounted Harville; full_development = #115's lambda2, one fit on "
    "the development out-of-fold races, the scored races among them, so "
    "partly in-sample (ordering_v0.yaml; the fit id is recorded with each "
    "run); 1.0 = plain Harville",
    "owner_signoff": "2026-11-25, #131 (all nine points as proposed)",
    "pool": "FORECAST",
    "rate": "takeout.race_takeout(race_id, 'FORECAST', race_class_raw=...).payout_rate",
    "relabel": "owner 2026-11-26 on #131: #115's single full-development lambda2 fit "
    "is labelled full_development (the v1 label said in-fold); the design "
    "is v1's unchanged",
    "rule": "at T-2 buy exacta c of category X when q_c(t) / p_c(t) < k; q_c(t) from "
    "the exacta board at t, p_c(t) Harville on the Win board at t",
    "scored_on": "#130-logged development meetings only, once there are min_meetings",
    "seed": 131,
    "sources": ["jra"],
    "spec": "docs/specs/exacta-overload-prereg.md",
    "stake_yen": 100,
    "supersedes": "japan/131-exacta-rule-v1",
    "ticket": 131,
    "win_probabilities": "#114 out-of-fold calibrated p_cal, renormalised over the starters",
}
