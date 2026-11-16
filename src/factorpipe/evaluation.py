"""Single-factor evaluation: information coefficient, decay, quantile returns, turnover.

All functions pair a characteristic at ``t`` with returns from ``t+1`` on, through
``Panel.forward_returns``. Only assets that are investable at ``t`` take part.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from factorpipe.panel import Panel
from factorpipe.stats import benjamini_yekutieli, holm, newey_west_mean

#: Smallest cross-section for which a monthly statistic is computed.
MIN_ASSETS = 20

#: The t-statistic hurdle proposed for new factors by Harvey, Liu and Zhu (2016).
HLZ_T_HURDLE = 3.0


def information_coefficient(
    panel: Panel,
    name: str,
    horizon: int = 1,
    method: str = "spearman",
    min_assets: int = MIN_ASSETS,
) -> pd.Series:
    """Monthly cross-sectional correlation between a factor and the forward return.

    The forward return is compounded over ``horizon`` months, so consecutive values
    overlap when ``horizon > 1``. ``method`` is ``"spearman"`` (both sides ranked) or
    ``"pearson"``.
    """
    factor = panel.characteristics[name]
    return _rowwise_correlation(factor, panel.forward_returns(horizon), method, min_assets)


def decay_curve(
    panel: Panel,
    name: str,
    max_lag: int = 12,
    method: str = "spearman",
    min_assets: int = MIN_ASSETS,
) -> pd.Series:
    """Mean IC against the single-month return ``lag`` months ahead, for each lag.

    This shows how fast the information in today's factor value fades. It uses assets
    that are still listed ``lag`` months later, so unlike ``information_coefficient`` it
    conditions on survival. Use it as a diagnostic, not as a performance number.
    """
    factor = panel.characteristics[name].where(panel.investable)
    means = {}
    for lag in range(1, max_lag + 1):
        ahead = panel.returns.shift(-lag)
        means[lag] = _rowwise_correlation(factor, ahead, method, min_assets).mean()
    return pd.Series(means, name=name).rename_axis("lag")


def quantile_assignments(
    panel: Panel, name: str, n_quantiles: int = 5, min_assets: int = MIN_ASSETS
) -> pd.DataFrame:
    """Quantile (1 = lowest factor value) of each investable asset on each date.

    Assets are ranked within the date and split into groups of equal size. Ties are
    broken by column order so that group sizes stay equal.
    """
    factor = panel.characteristics[name].where(panel.investable)
    count = factor.notna().sum(axis=1)
    rank = factor.rank(axis=1, method="first")
    quantile = np.ceil(rank.mul(n_quantiles).div(count, axis=0))
    return quantile.where(count >= max(min_assets, n_quantiles), np.nan)


def quantile_returns(
    panel: Panel,
    name: str,
    n_quantiles: int = 5,
    horizon: int = 1,
    min_assets: int = MIN_ASSETS,
) -> pd.DataFrame:
    """Equal-weighted forward return of each factor quantile, by date."""
    quantile = quantile_assignments(panel, name, n_quantiles, min_assets)
    forward = panel.forward_returns(horizon)
    columns = {
        q: forward.where(quantile == q).mean(axis=1, skipna=True) for q in range(1, n_quantiles + 1)
    }
    return pd.DataFrame(columns).rename_axis(columns="quantile")


def long_short(by_quantile: pd.DataFrame) -> pd.Series:
    """Top quantile minus bottom quantile."""
    return (by_quantile.iloc[:, -1] - by_quantile.iloc[:, 0]).rename("long_short")


def quantile_turnover(
    panel: Panel,
    name: str,
    n_quantiles: int = 5,
    quantile: int | None = None,
    min_assets: int = MIN_ASSETS,
) -> pd.Series:
    """Share of the names in a quantile that were not in it the month before.

    Defaults to the top quantile. A value of 0 means the same names; 1 means all new.
    """
    target = n_quantiles if quantile is None else quantile
    member = quantile_assignments(panel, name, n_quantiles, min_assets) == target
    size = member.sum(axis=1)
    stayed = (member & member.shift(1, fill_value=False)).sum(axis=1)
    turnover = 1.0 - stayed / size.where(size > 0)
    turnover.iloc[0] = np.nan
    return turnover.where(size.shift(1) > 0).rename("turnover")


def rank_autocorrelation(
    panel: Panel, name: str, lag: int = 1, min_assets: int = MIN_ASSETS
) -> pd.Series:
    """Cross-sectional rank correlation between the factor now and ``lag`` months ago.

    A slow-moving factor is close to 1. This is a turnover measure that does not depend
    on the number of quantiles.
    """
    factor = panel.characteristics[name]
    return _rowwise_correlation(factor, factor.shift(lag), "spearman", min_assets)


def evaluate_factor(
    panel: Panel,
    name: str,
    horizons: Sequence[int] = (1, 3, 6, 12),
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """One row per horizon: IC, rank IC and long-short spread, each with a Newey-West t.

    Standard errors use at least ``horizon - 1`` lags because of the overlap.
    """
    rows = []
    for horizon in horizons:
        ic = newey_west_mean(
            information_coefficient(panel, name, horizon, "pearson"), None, horizon
        )
        rank_ic = newey_west_mean(information_coefficient(panel, name, horizon), None, horizon)
        spread = newey_west_mean(
            long_short(quantile_returns(panel, name, n_quantiles, horizon)), None, horizon
        )
        rows.append(
            {
                "horizon": horizon,
                "ic": ic.mean,
                "ic_t": ic.t_stat,
                "rank_ic": rank_ic.mean,
                "rank_ic_se": rank_ic.std_error,
                "rank_ic_t": rank_ic.t_stat,
                "spread": spread.mean,
                "spread_t": spread.t_stat,
                "n_months": rank_ic.n_obs,
                "nw_lags": rank_ic.lags,
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def screen_factors(
    panel: Panel,
    names: Iterable[str] | None = None,
    horizon: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test every factor and adjust for the number of factors tested.

    The test statistic is the Newey-West t of the mean rank IC. ``p_holm`` controls the
    chance of any false positive across the whole list; ``p_by`` controls the expected
    share of false positives among the factors flagged. ``names`` must be the full list
    of factors that were looked at: leaving out the ones that failed makes the
    adjustment meaningless.
    """
    names = list(panel.characteristics) if names is None else list(names)
    rows = []
    for name in names:
        test = newey_west_mean(information_coefficient(panel, name, horizon), None, horizon)
        rows.append(
            {
                "factor": name,
                "rank_ic": test.mean,
                "t_stat": test.t_stat,
                "p_value": test.p_value,
                "n_months": test.n_obs,
            }
        )
    table = pd.DataFrame(rows).set_index("factor")
    table["p_holm"] = holm(table["p_value"].to_numpy())
    table["p_by"] = benjamini_yekutieli(table["p_value"].to_numpy())
    table["significant_raw"] = table["p_value"] < alpha
    table["significant_holm"] = table["p_holm"] < alpha
    table["significant_by"] = table["p_by"] < alpha
    table["clears_t3"] = table["t_stat"].abs() > HLZ_T_HURDLE
    return table


def _rowwise_correlation(
    x: pd.DataFrame, y: pd.DataFrame, method: str, min_assets: int
) -> pd.Series:
    """Correlation across columns for each row, over the columns present in both."""
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    both = x.notna() & y.notna()
    x, y = x.where(both), y.where(both)
    if method == "spearman":
        x, y = x.rank(axis=1), y.rank(axis=1)
    a = x.to_numpy(dtype=float)
    b = y.to_numpy(dtype=float)
    n = both.to_numpy().sum(axis=1)
    safe_n = np.where(n > 0, n, 1)
    a = a - (np.nansum(a, axis=1) / safe_n)[:, None]
    b = b - (np.nansum(b, axis=1) / safe_n)[:, None]
    cov = np.nansum(a * b, axis=1)
    scale = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / scale
    corr[(n < min_assets) | (scale == 0.0)] = np.nan
    return pd.Series(corr, index=x.index)
