"""The SPEC acceptance criteria for factor evaluation, on the synthetic panel."""

import numpy as np
import pytest

from factorpipe.evaluation import information_coefficient, screen_factors
from factorpipe.stats import newey_west_mean
from factorpipe.synthetic import PlantedFactor, SyntheticConfig, make_panel

EXACT = {"delist_prob": 0.0, "late_listing_share": 0.0}


def test_planted_factors_are_detected_and_noise_is_not():
    synthetic = make_panel(SyntheticConfig(seed=1, **EXACT))
    table = screen_factors(synthetic.panel, names=synthetic.truth.planted + synthetic.truth.noise)
    assert table.loc[synthetic.truth.planted, "significant_holm"].all()
    assert table.loc[synthetic.truth.planted, "clears_t3"].all()
    assert not table.loc[synthetic.truth.noise, "significant_holm"].any()


def test_planted_factors_are_detected_with_listings_and_delistings():
    synthetic = make_panel(SyntheticConfig(seed=1))
    table = screen_factors(synthetic.panel)
    assert table.loc[synthetic.truth.planted, "significant_holm"].all()
    # The reported version of the strong factor is stale but still informative.
    assert table.loc["planted_strong_reported", "significant_holm"]
    assert table.loc["planted_strong_reported", "rank_ic"] < table.loc["planted_strong", "rank_ic"]


@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
def test_the_confidence_interval_contains_the_true_ic(horizon):
    synthetic = make_panel(SyntheticConfig(seed=1, **EXACT))
    for name in synthetic.truth.planted:
        series = information_coefficient(synthetic.panel, name, horizon, method="pearson")
        test = newey_west_mean(series, horizon=horizon)
        lo, hi = test.confidence_interval(z=2.5758)  # 99%
        assert lo < synthetic.truth.cumulative_ic_at(name, horizon) < hi


def test_confidence_intervals_cover_the_truth_at_about_the_stated_rate():
    covered, total = 0, 0
    for seed in range(60):
        cfg = SyntheticConfig(
            n_assets=150,
            n_months=120,
            planted=(PlantedFactor("p", ic=0.04, persistence=0.9),),
            n_noise=0,
            reported_factor=None,
            seed=seed,
            **EXACT,
        )
        synthetic = make_panel(cfg)
        test = newey_west_mean(information_coefficient(synthetic.panel, "p", method="pearson"))
        lo, hi = test.confidence_interval()
        covered += lo < 0.04 < hi
        total += 1
    # A 95% interval should cover about 57 of 60. Below 50 would mean the standard
    # error is too small.
    assert covered >= 50, f"covered {covered} of {total}"


def test_noise_factors_are_flagged_at_no_more_than_the_nominal_rate():
    raw, holm_any, by_flags, tests, seeds = 0, 0, 0, 0, 40
    for seed in range(seeds):
        cfg = SyntheticConfig(
            n_assets=150,
            n_months=120,
            planted=(PlantedFactor("p", ic=0.04, persistence=0.9),),
            n_noise=25,
            reported_factor=None,
            seed=1000 + seed,
            **EXACT,
        )
        synthetic = make_panel(cfg)
        table = screen_factors(synthetic.panel, names=synthetic.truth.noise)
        raw += int(table["significant_raw"].sum())
        holm_any += int(table["significant_holm"].any())
        by_flags += int(table["significant_by"].sum())
        tests += len(table)

    # Unadjusted: at least 5% of 1000 noise tests are "significant". That is the problem
    # the adjustment exists for.
    assert 0.025 < raw / tests < 0.085
    # Holm: the share of panels with any false positive. Nominal is 5%. A larger run
    # (400 panels, see DECISIONS.md) measures about 9%, because Newey-West standard
    # errors are a little too small in samples this short. The bound below is loose
    # enough for 40 panels and tight enough to catch an adjustment that does nothing,
    # which would give about 70%.
    assert holm_any / seeds <= 0.125
    # Benjamini-Yekutieli on pure noise is at least as strict.
    assert by_flags <= holm_any * 2 + 1
    assert raw > 5 * max(holm_any, 1)


def test_dropping_failed_factors_from_the_list_makes_a_noise_factor_look_significant():
    cfg = SyntheticConfig(
        n_assets=150,
        n_months=120,
        planted=(),
        n_noise=60,
        reported_factor=None,
        seed=3,
        **EXACT,
    )
    synthetic = make_panel(cfg)
    honest = screen_factors(synthetic.panel)
    best = honest["p_value"].idxmin()
    assert not honest.loc[best, "significant_holm"]
    cherry_picked = screen_factors(synthetic.panel, names=[best])
    assert cherry_picked.loc[best, "significant_holm"]
    assert np.isclose(cherry_picked.loc[best, "p_holm"], honest.loc[best, "p_value"])
