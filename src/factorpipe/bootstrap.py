"""Stationary bootstrap confidence intervals for Sharpe ratios.

Politis and Romano (1994). Resampling single months destroys any serial dependence in
the returns. The stationary bootstrap resamples blocks instead: start at a random
month, keep taking the next month with probability ``1 - 1/b`` and jump to a new random
month with probability ``1/b``, wrapping around the end of the sample. Block lengths are
then geometric with mean ``b``, and the resampled series is stationary.

To compare two strategies, their returns are resampled with the same months, so the
dependence between them is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    #: Share of bootstrap draws at or below zero.
    share_not_positive: float
    n_draws: int
    mean_block: float


def default_block(n_obs: int) -> float:
    """Mean block length ``n ** (1/3)``, a common rule of thumb, never below 1."""
    return max(1.0, round(n_obs ** (1.0 / 3.0)))


def stationary_indices(
    n_obs: int, n_draws: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Row indices for ``n_draws`` resamples, shape (n_draws, n_obs)."""
    if mean_block < 1:
        raise ValueError("mean_block must be at least 1")
    restart = rng.random((n_draws, n_obs)) < 1.0 / mean_block
    fresh = rng.integers(0, n_obs, size=(n_draws, n_obs))
    index = np.empty((n_draws, n_obs), dtype=np.int64)
    index[:, 0] = fresh[:, 0]
    for t in range(1, n_obs):
        index[:, t] = np.where(restart[:, t], fresh[:, t], (index[:, t - 1] + 1) % n_obs)
    return index


def sharpe_interval(
    returns,
    *,
    level: float = 0.95,
    n_draws: int = 2000,
    mean_block: float | None = None,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile interval for the per-period Sharpe ratio of one return series."""
    r = _clean(pd.DataFrame({"r": np.asarray(returns, dtype=float)}))[:, 0]
    return _interval(r[:, None], lambda s: s[..., 0], level, n_draws, mean_block, seed)


def sharpe_difference_interval(
    first,
    second,
    *,
    level: float = 0.95,
    n_draws: int = 2000,
    mean_block: float | None = None,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile interval for ``Sharpe(first) - Sharpe(second)``, resampled jointly.

    The two series must cover the same periods. Periods missing in either are dropped.
    """
    frame = pd.DataFrame(
        {"a": pd.Series(first).to_numpy(float), "b": pd.Series(second).to_numpy(float)}
    )
    return _interval(
        _clean(frame), lambda s: s[..., 0] - s[..., 1], level, n_draws, mean_block, seed
    )


def _interval(values, statistic, level, n_draws, mean_block, seed) -> BootstrapInterval:
    n_obs = values.shape[0]
    if n_obs < 3:
        raise ValueError("need at least three periods")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    block = default_block(n_obs) if mean_block is None else mean_block
    rng = np.random.default_rng(seed)
    resampled = values[stationary_indices(n_obs, n_draws, block, rng)]
    draws = statistic(_sharpe(resampled, axis=1))
    draws = draws[np.isfinite(draws)]
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return BootstrapInterval(
        estimate=float(statistic(_sharpe(values, axis=0))),
        lower=float(lower),
        upper=float(upper),
        share_not_positive=float((draws <= 0).mean()),
        n_draws=len(draws),
        mean_block=float(block),
    )


def _sharpe(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return values.mean(axis=axis) / values.std(axis=axis, ddof=1)


def _clean(frame: pd.DataFrame) -> np.ndarray:
    return frame.dropna().to_numpy(dtype=float)
