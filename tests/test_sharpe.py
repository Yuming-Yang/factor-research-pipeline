import math

import numpy as np
import pytest
from scipy import stats

from factorpipe.sharpe import (
    EULER_GAMMA,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_standard_error,
)


def normal_returns(n: int, mean: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, 0.04, size=n)


def test_sharpe_ratio_by_hand():
    r = [0.01, 0.03, -0.01, 0.05]
    expected = np.mean(r) / np.std(r, ddof=1)
    assert sharpe_ratio(r, periods_per_year=1) == pytest.approx(expected)
    assert sharpe_ratio(r) == pytest.approx(expected * math.sqrt(12))
    assert math.isnan(sharpe_ratio([0.01, 0.01, 0.01]))
    assert math.isnan(sharpe_ratio([0.01]))
    assert sharpe_ratio([0.01, np.nan, 0.03, -0.01, 0.05], 1) == pytest.approx(expected)


def test_standard_error_reduces_to_the_normal_formula():
    r = normal_returns(100_000, 0.01, seed=0)
    sr = sharpe_ratio(r, 1)
    assert sharpe_standard_error(r) == pytest.approx(
        math.sqrt((1 + sr**2 / 2) / (len(r) - 1)), rel=0.02
    )


def test_standard_error_matches_the_spread_of_simulated_sharpe_ratios():
    estimates = [sharpe_ratio(normal_returns(240, 0.01, seed=s), 1) for s in range(2000)]
    one_sample = sharpe_standard_error(normal_returns(240, 0.01, seed=1))
    assert one_sample == pytest.approx(np.std(estimates), rel=0.08)


def test_negative_skew_and_fat_tails_widen_the_standard_error():
    rng = np.random.default_rng(3)
    base = rng.normal(0.01, 0.04, size=5000)
    crashes = base.copy()
    crashes[rng.random(5000) < 0.02] -= 0.25  # rare large losses
    # Give both series the same mean and standard deviation, so only the shape differs.
    crashes = (crashes - crashes.mean()) / crashes.std() * base.std() + base.mean()
    assert sharpe_ratio(crashes, 1) == pytest.approx(sharpe_ratio(base, 1))
    assert sharpe_standard_error(crashes) > 1.05 * sharpe_standard_error(base)


def test_psr_is_a_normal_probability_of_the_standardized_gap():
    r = normal_returns(240, 0.008, seed=5)
    sr, se = sharpe_ratio(r, 1), sharpe_standard_error(r)
    assert probabilistic_sharpe_ratio(r) == pytest.approx(stats.norm.cdf(sr / se))
    assert probabilistic_sharpe_ratio(r, benchmark=sr) == pytest.approx(0.5)
    assert probabilistic_sharpe_ratio(r, 0.1) < probabilistic_sharpe_ratio(r, 0.0)


def test_expected_max_sharpe_formula_and_limits():
    n, v = 50, 0.01
    by_hand = math.sqrt(v) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * math.e))
    )
    assert expected_max_sharpe(n, v) == pytest.approx(by_hand)
    assert expected_max_sharpe(1, v) == 0.0
    assert expected_max_sharpe(50, 0.0) == 0.0
    assert expected_max_sharpe(10, v) < expected_max_sharpe(100, v) < expected_max_sharpe(1000, v)
    with pytest.raises(ValueError, match="at least 1"):
        expected_max_sharpe(0, v)


@pytest.mark.parametrize("n_trials", [5, 20, 100])
def test_expected_max_sharpe_matches_simulation(n_trials):
    rng = np.random.default_rng(7)
    sigma = 0.07
    simulated = rng.normal(0.0, sigma, size=(20_000, n_trials)).max(axis=1).mean()
    assert expected_max_sharpe(n_trials, sigma**2) == pytest.approx(simulated, rel=0.05)


def test_more_trials_deflate_the_same_returns_further():
    r = normal_returns(240, 0.01, seed=9)
    single = deflated_sharpe_ratio(r, n_trials=1, trial_variance=0.005)
    few = deflated_sharpe_ratio(r, n_trials=10, trial_variance=0.005)
    many = deflated_sharpe_ratio(r, n_trials=500, trial_variance=0.005)
    assert single.dsr == pytest.approx(single.psr)
    assert single.hurdle == 0.0
    assert single.dsr > few.dsr > many.dsr
    assert few.sharpe == many.sharpe == pytest.approx(sharpe_ratio(r, 1))


def test_the_best_of_many_skill_free_strategies_is_not_significant_after_deflation():
    rng = np.random.default_rng(11)
    trials = rng.normal(0.0, 0.04, size=(200, 240))  # 200 strategies, no skill
    sharpes = np.array([sharpe_ratio(t, 1) for t in trials])
    best = trials[sharpes.argmax()]
    assert probabilistic_sharpe_ratio(best) > 0.99  # looks excellent on its own
    deflated = deflated_sharpe_ratio(best, len(trials), float(sharpes.var(ddof=1)))
    assert deflated.dsr < 0.9
    assert deflated.hurdle == pytest.approx(sharpes.max(), rel=0.35)
