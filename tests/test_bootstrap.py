import math

import numpy as np
import pytest

from factorpipe.bootstrap import (
    default_block,
    sharpe_difference_interval,
    sharpe_interval,
    stationary_indices,
)


def test_blocks_are_runs_of_consecutive_months_with_the_stated_mean_length():
    rng = np.random.default_rng(0)
    index = stationary_indices(n_obs=500, n_draws=200, mean_block=6.0, rng=rng)
    assert index.shape == (200, 500)
    assert index.min() >= 0 and index.max() <= 499
    consecutive = (np.diff(index, axis=1) % 500) == 1
    # A block continues with probability 1 - 1/6 (a restart can land on the next month
    # by chance, which adds 1/500 of the restarts).
    assert consecutive.mean() == pytest.approx(1 - 1 / 6, abs=0.01)
    # Every month is equally likely.
    counts = np.bincount(index.ravel(), minlength=500)
    assert counts.std() / counts.mean() < 0.1


def test_a_block_length_of_one_is_the_ordinary_bootstrap():
    index = stationary_indices(300, 100, 1.0, np.random.default_rng(1))
    consecutive = (np.diff(index, axis=1) % 300) == 1
    assert consecutive.mean() < 0.01


def test_the_interval_width_matches_the_analytic_standard_error_for_iid_returns():
    r = np.random.default_rng(2).normal(0.01, 0.04, size=600)
    result = sharpe_interval(r, n_draws=4000, mean_block=1.0)
    sr = r.mean() / r.std(ddof=1)
    analytic = math.sqrt((1 + sr**2 / 2) / len(r))
    assert result.estimate == pytest.approx(sr)
    assert (result.upper - result.lower) / (2 * 1.96) == pytest.approx(analytic, rel=0.08)
    assert result.lower < result.estimate < result.upper


def test_the_interval_covers_the_true_sharpe_ratio_at_about_the_stated_rate():
    true = 0.25
    covered = 0
    for seed in range(150):
        r = np.random.default_rng(100 + seed).normal(true * 0.04, 0.04, size=240)
        result = sharpe_interval(r, n_draws=400, seed=seed)
        covered += result.lower < true < result.upper
    assert 130 <= covered <= 149  # nominal 95% of 150 is about 142


def test_positive_autocorrelation_needs_blocks():
    rng = np.random.default_rng(3)
    shocks = rng.normal(0, 0.04, size=1200)
    r = np.empty(1200)
    r[0] = shocks[0]
    for t in range(1, 1200):
        r[t] = 0.5 * r[t - 1] + shocks[t]
    r += 0.01
    single = sharpe_interval(r, mean_block=1.0, n_draws=2000)
    blocked = sharpe_interval(r, mean_block=12.0, n_draws=2000)
    assert (blocked.upper - blocked.lower) > 1.3 * (single.upper - single.lower)


def test_a_paired_comparison_is_tighter_than_two_separate_ones():
    rng = np.random.default_rng(4)
    common = rng.normal(0.0, 0.04, size=480)
    a = common + rng.normal(0.012, 0.01, size=480)
    b = common + rng.normal(0.004, 0.01, size=480)
    diff = sharpe_difference_interval(a, b, n_draws=3000)
    assert diff.lower > 0  # the difference is clear because the common part cancels
    assert diff.share_not_positive < 0.01
    separate = sharpe_interval(a, n_draws=3000)
    assert (diff.upper - diff.lower) < 0.6 * (separate.upper - separate.lower)
    same = sharpe_difference_interval(a, a)
    assert same.estimate == 0.0 and same.lower == same.upper == 0.0


def test_results_are_reproducible_and_inputs_are_checked():
    r = np.random.default_rng(5).normal(0.01, 0.04, size=120)
    assert sharpe_interval(r, seed=7) == sharpe_interval(r, seed=7)
    assert sharpe_interval(r, seed=7) != sharpe_interval(r, seed=8)
    assert default_block(240) == 6 and default_block(1) == 1
    with pytest.raises(ValueError, match="three periods"):
        sharpe_interval([0.01, 0.02])
    with pytest.raises(ValueError, match="level"):
        sharpe_interval(r, level=1.5)
    with pytest.raises(ValueError, match="mean_block"):
        sharpe_interval(r, mean_block=0.5)
