"""The v4 #131 E2 family spec, japan-131-exacta-rule-v4, as frozen on 2026-11-27.

Synthetic test data, not a scraped or licensed file. Plain JSON types only, so its canonical JSON
(``json.dumps(spec, sort_keys=True, separators=(",", ":"))``) hashes to the v4 E2 entry of the spec
table in docs/specs/exacta-overload-prereg.md (V4_RULE_HASH in
tests/countries/japan/research/test_exacta_study.py, which checks it). Generated mechanically from
``exacta_study.family_specs()`` at git 2c73ca0; do not edit by hand.
"""

V4_RULE_ID = "japan-131-exacta-rule-v4"

V4_RULE_SPEC = {
    "benchmark": "no bet (loss 0)",
    "block_size": 5,
    "category": "fav_first, fav_second or no_fav of combination c against the favourite on the "
    "Win board at T-2 (combo_category)",
    "cluster": "meeting",
    "cuts": [0.6, 0.7, 0.8, 0.9],
    "data": "development race days only (is_holdout_day False, filtered before any outcome, "
    "odds or dividend is read); races from 2014-06-07 that have #114 out-of-fold Win "
    "probabilities, i.e. its walk-forward test quarters (from 2020-Q1)",
    "diagnostics": {
        "less_popular_agreement": "the lower tertile of exacta pool totals within "
        "each logged day (the odds log's investment row), "
        "reported as its agreement with the race-card "
        "proxy, not a selection variable: a whole-day "
        "tertile uses later races' totals"
    },
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
    "n_boot": 10000,
    "ordering": "discounted Harville; full_development = #115's lambda2, one fit on the "
    "development out-of-fold races, the scored races among them, so partly "
    "in-sample (ordering_v0.yaml; the fit id is recorded with each run); 1.0 = "
    "plain Harville",
    "owner_signoff": "2026-11-25, #131 (all nine points as proposed)",
    "pool": "FORECAST",
    "primary_metric": "expected return per 100 yen staked at the final dividend, takeout and "
    "breakage in, with its CI (owner reframe 2026-11-27 on #131: rank every "
    "option by expected return at the final price, CI beside it)",
    "rate": "takeout.race_takeout(race_id, 'FORECAST', race_class_raw=...).payout_rate",
    "relabel": "owner 2026-11-26 on #131: #115's single full-development lambda2 fit is "
    "labelled full_development (the v1 label said in-fold); the design is v1's "
    "unchanged",
    "revision": 'v4 (review of 83684eb, 2026-11-27): rule-v3\'s headline took "less popular" '
    "from the lower tertile of each logged day's pool totals, which later races "
    "set; v4 takes it from the race card (race_no 1-4 or DEBUT/MAIDEN, E1's "
    "popularity segment), reports the tertile as an agreement diagnostic only and "
    "scores the logged development days available at each run with no meeting gate, "
    "and changes nothing else. v3 (review of 0c39a8d): the favourite and the "
    "category from the board at T-2, not the final Win odds",
    "rule": "at T-2 buy exacta c of category X when q_c(t) / p_c(t) < k; q_c(t) from the exacta "
    "board at t, p_c(t) Harville on the Win board at t",
    "scopes": {
        "all": "every scored race",
        "headline": {
            "fav_odds_min": 3.0,
            "information_cutoff": {
                "fav_win_odds": "T-2",
                "race_class": "race card",
                "race_no": "race card",
            },
            "less_popular": {"race_classes": ["DEBUT", "MAIDEN"], "race_no_max": 4},
            "rule": "favourite's Win odds at T-2 >= 3.0 and less popular; less "
            "popular = race_no 1-4 or a DEBUT/MAIDEN race, from the "
            "race card (E1's popularity segment)",
        },
    },
    "scored_on": "#130-logged development days available at each run, with n and the CI, "
    "re-scored as more days are logged",
    "seed": 131,
    "selection_time": "T-2",
    "sources": ["jra"],
    "spec": "docs/specs/exacta-overload-prereg.md",
    "stake_yen": 100,
    "supersedes": "japan/131-exacta-rule-v3",
    "ticket": 131,
    "win_probabilities": "#114 out-of-fold calibrated p_cal, renormalised over the starters",
}
