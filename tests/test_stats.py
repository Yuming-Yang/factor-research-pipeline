import math

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from factorpipe.stats import (
    benjamini_yekutieli,
    default_lags,
    holm,
    newey_west_mean,
    two_sided_p,
)


def ar1(n: int, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.standard_normal()
    return x + 0.3


@pytest.mark.parametrize("lags", [0, 1, 5, 11])
def test_newey_west_matches_statsmodels(lags):
    x = ar1(300, 0.6, seed=0)
    ours = newey_west_mean(x, lags=lags)
    fit = sm.OLS(x, np.ones_like(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )
    assert ours.mean == pytest.approx(fit.params[0])
    assert ours.std_error == pytest.approx(fit.bse[0], rel=1e-10)
    assert ours.t_stat == pytest.approx(fit.tvalues[0], rel=1e-10)
    assert ours.p_value == pytest.approx(fit.pvalues[0], rel=1e-8)


def test_zero_lags_is_the_ordinary_standard_error():
    x = ar1(200, 0.0, seed=1)
    ours = newey_west_mean(x, lags=0)
    assert ours.std_error == pytest.approx(x.std(ddof=0) / math.sqrt(len(x)))


def test_positive_autocorrelation_widens_the_standard_error():
    x = ar1(500, 0.7, seed=2)
    assert newey_west_mean(x, lags=12).std_error > 1.5 * newey_west_mean(x, lags=0).std_error


def test_hand_worked_example():
    # x = [1, 2, 4], mean 7/3. Deviations d = [-4/3, -1/3, 5/3].
    # g0 = (16 + 1 + 25) / 27 = 42/27.  g1 = (d1*d0 + d2*d1) / 3 = (4/9 - 5/9) / 3 = -1/27.
    # L = 1: long-run variance = g0 + 2 * (1/2) * g1 = 41/27.  se = sqrt(41/81).
    result = newey_west_mean([1.0, 2.0, 4.0], lags=1)
    assert result.mean == pytest.approx(7 / 3)
    assert result.std_error == pytest.approx(math.sqrt(41 / 81))


def test_missing_values_are_dropped():
    x = ar1(100, 0.3, seed=3)
    with_gaps = np.insert(x, [10, 50], np.nan)
    assert newey_west_mean(with_gaps, lags=3) == newey_west_mean(x, lags=3)


def test_degenerate_inputs_do_not_raise():
    assert math.isnan(newey_west_mean([1.0]).t_stat)
    assert math.isnan(newey_west_mean([2.0, 2.0, 2.0]).t_stat)


def test_default_lags_cover_the_overlap():
    assert default_lags(240, horizon=1) == 4
    assert default_lags(240, horizon=12) == 11
    assert default_lags(60, horizon=1) == 3


def test_confidence_interval_is_symmetric_around_the_mean():
    result = newey_west_mean(ar1(100, 0.2, seed=4), lags=2)
    lo, hi = result.confidence_interval()
    assert (lo + hi) / 2 == pytest.approx(result.mean)
    assert hi - lo == pytest.approx(2 * 1.959963984540054 * result.std_error)


def test_two_sided_p_known_values():
    assert two_sided_p(0.0) == pytest.approx(1.0)
    assert two_sided_p(1.959963984540054) == pytest.approx(0.05)
    assert two_sided_p(-3.0) == pytest.approx(0.0026997960632601866)


@pytest.mark.parametrize("seed", range(5))
def test_adjustments_match_statsmodels(seed):
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(size=30), rng.uniform(0, 0.002, size=5)])
    rng.shuffle(p)
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])
    np.testing.assert_allclose(benjamini_yekutieli(p), multipletests(p, method="fdr_by")[1])


def test_holm_hand_worked_example():
    # Sorted: 0.01, 0.02, 0.04 -> times 3, 2, 1 -> 0.03, 0.04, 0.04.
    np.testing.assert_allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])


def test_adjusted_p_values_are_never_below_the_raw_ones():
    p = np.random.default_rng(9).uniform(size=50)
    assert (holm(p) >= p).all()
    assert (benjamini_yekutieli(p) >= p).all()
    assert (benjamini_yekutieli(p) <= holm(p) * np.sum(1 / np.arange(1, 51)) + 1e-12).all()


def test_missing_p_values_stay_missing_and_do_not_count():
    adjusted = holm([0.01, np.nan, 0.02])
    assert np.isnan(adjusted[1])
    np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.02])
    assert np.isnan(holm([np.nan, np.nan])).all()
