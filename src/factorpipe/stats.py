"""Inference for a time series of monthly statistics, and multiple-testing adjustments.

Everything here is written out from the formulas so that it can be derived by hand.
statsmodels is used in the tests as an independent check, not at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def default_lags(n_obs: int, horizon: int = 1) -> int:
    """Lags for a Newey-West estimate on a series built from ``horizon``-month returns.

    Returns over ``horizon`` months overlap for ``horizon - 1`` months, which induces
    autocorrelation of at least that order, so that is the minimum. The rule of thumb
    ``floor(4 * (n / 100) ** (2 / 9))`` from Newey and West (1994) covers autocorrelation
    that does not come from overlap.
    """
    rule_of_thumb = math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))
    return max(horizon - 1, rule_of_thumb)


@dataclass(frozen=True)
class MeanTest:
    """A test of whether the mean of a series is zero."""

    mean: float
    std_error: float
    t_stat: float
    p_value: float
    n_obs: int
    lags: int

    def confidence_interval(self, z: float = 1.959963984540054) -> tuple[float, float]:
        return self.mean - z * self.std_error, self.mean + z * self.std_error


def newey_west_mean(values, lags: int | None = None, horizon: int = 1) -> MeanTest:
    """Test the mean of a series with a Newey-West (Bartlett kernel) standard error.

    With autocovariances ``g_j = (1/n) * sum_t (x_t - m)(x_{t-j} - m)``, the long-run
    variance is ``g_0 + 2 * sum_{j=1..L} (1 - j/(L+1)) * g_j`` and the standard error of
    the mean is ``sqrt(long_run_variance / n)``. With ``L = 0`` this is the ordinary
    standard error with a 1/n variance. The Bartlett weights keep the estimate
    non-negative.

    Missing values are dropped first, and the remaining observations are treated as
    consecutive. The p-value is two-sided and uses the normal distribution.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return MeanTest(math.nan, math.nan, math.nan, math.nan, n, 0)
    if lags is None:
        lags = default_lags(n, horizon)
    lags = min(lags, n - 1)

    mean = float(x.mean())
    d = x - mean
    long_run = float(d @ d) / n
    for j in range(1, lags + 1):
        weight = 1.0 - j / (lags + 1.0)
        long_run += 2.0 * weight * float(d[j:] @ d[:-j]) / n
    std_error = math.sqrt(long_run / n)
    if std_error == 0.0:
        return MeanTest(mean, 0.0, math.nan, math.nan, n, lags)
    t_stat = mean / std_error
    return MeanTest(mean, std_error, t_stat, two_sided_p(t_stat), n, lags)


def two_sided_p(t_stat: float) -> float:
    """Two-sided p-value of a standard normal statistic."""
    return math.erfc(abs(t_stat) / math.sqrt(2.0))


def holm(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values. Controls the family-wise error rate.

    Sort the m p-values in increasing order. The k-th smallest is multiplied by
    ``m - k + 1``, and the results are made non-decreasing so that a larger raw p-value
    never gets a smaller adjusted one. Valid under any dependence between the tests.
    Missing p-values stay missing and do not count toward m.
    """
    return _adjust(p_values, lambda p_sorted, m: _step_down(p_sorted * (m - np.arange(m))))


def benjamini_yekutieli(p_values) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values. Controls the false discovery rate.

    The k-th smallest p-value is multiplied by ``m * c(m) / k`` with
    ``c(m) = 1 + 1/2 + ... + 1/m``, and the results are made non-decreasing from the
    largest down. The ``c(m)`` term is what makes it valid under any dependence; without
    it this is Benjamini-Hochberg, which needs independence or positive dependence.
    """

    def by(p_sorted: np.ndarray, m: int) -> np.ndarray:
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        scaled = p_sorted * m * c_m / np.arange(1, m + 1)
        return np.minimum.accumulate(scaled[::-1])[::-1]

    return _adjust(p_values, by)


def _step_down(scaled: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(scaled)


def _adjust(p_values, rule) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    present = ~np.isnan(p)
    m = int(present.sum())
    if m == 0:
        return out
    order = np.argsort(p[present], kind="stable")
    adjusted_sorted = np.clip(rule(p[present][order], m), 0.0, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    out[present] = adjusted
    return out


def hac_regression(y, x: pd.DataFrame, lags: int | None = None) -> pd.DataFrame:
    """Regress ``y`` on a constant and the columns of ``x``, with Newey-West errors.

    Used to ask how much of a strategy's return is explained by known factor returns:
    the intercept (``alpha``) is the part that is not. Coefficients are ordinary least
    squares, ``b = (X'X)^-1 X'y``. Their covariance is the sandwich
    ``(X'X)^-1 S (X'X)^-1`` with ``S = sum_j w_j * (G_j + G_j')`` over lags ``j``,
    ``G_j = sum_t e_t e_{t-j} x_t x_{t-j}'``, Bartlett weights ``w_j = 1 - j/(L+1)`` and
    ``w_0 = 1/2``. With no regressors this is ``newey_west_mean``.

    Returns one row per coefficient with its standard error, t-statistic and p-value.
    The R-squared is in ``attrs["r_squared"]`` and the row count in ``attrs["n_obs"]``.
    """
    frame = pd.concat([pd.Series(np.asarray(y, dtype=float), index=x.index, name="_y"), x], axis=1)
    frame = frame.dropna()
    n = len(frame)
    design = np.column_stack([np.ones(n), frame[x.columns].to_numpy(dtype=float)])
    if n <= design.shape[1]:
        raise ValueError("not enough observations for this many regressors")
    target = frame["_y"].to_numpy(dtype=float)
    if lags is None:
        lags = default_lags(n)
    lags = min(lags, n - 1)

    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ target
    residual = target - design @ beta
    scores = design * residual[:, None]
    meat = scores.T @ scores
    for j in range(1, lags + 1):
        weight = 1.0 - j / (lags + 1.0)
        cross = scores[j:].T @ scores[:-j]
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    std_error = np.sqrt(np.diag(covariance))
    t_stat = beta / std_error
    table = pd.DataFrame(
        {
            "coefficient": beta,
            "std_error": std_error,
            "t_stat": t_stat,
            "p_value": [two_sided_p(t) for t in t_stat],
        },
        index=["alpha", *x.columns],
    )
    total = float(((target - target.mean()) ** 2).sum())
    table.attrs["r_squared"] = 1.0 - float(residual @ residual) / total if total > 0 else math.nan
    table.attrs["n_obs"] = n
    table.attrs["lags"] = lags
    return table
