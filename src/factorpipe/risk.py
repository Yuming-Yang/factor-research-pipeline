"""A structural risk model: market, beta, industries and style factors.

Returns are explained month by month by a cross-sectional regression

    r[i, s+1] = sum_k X[i, k, s] * f[k, s+1] + e[i, s+1]

on exposures known at the end of month ``s``. The fitted ``f`` are factor returns and the
``e`` are asset-specific returns. The risk model at a decision date uses the covariance
of the factor returns realized so far and the variance of each asset's specific returns:

    Sigma = X F X' + diag(D)

Exposure columns:

- one dummy per industry if the panel has industry labels, otherwise a single column of
  ones. Either way the columns add up to one for every asset, so they carry the common
  market move;
- market beta, standardized across assets. Standardizing makes it a tilt around the
  average asset, which keeps it from duplicating the market column;
- each style characteristic, standardized across assets.

Dollar neutrality removes the market column's contribution exactly. A book that is long
high-beta and short low-beta assets is still exposed to the market through the beta
column, and the optimizer now sees that as risk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from robust_mvo.covariance import ledoit_wolf
from robust_mvo.risk_model import FactorRiskModel

from factorpipe.combine import standardize
from factorpipe.panel import Panel

#: Fewest months of factor returns before a factor covariance is estimated.
MIN_FACTOR_HISTORY = 24
#: Fewest months of specific returns before an asset gets its own specific variance.
MIN_SPECIFIC_HISTORY = 12
#: A specific variance is never set below this share of the cross-sectional median.
SPECIFIC_FLOOR = 0.25


def trailing_beta(panel: Panel, window: int = 36, min_periods: int = 12) -> pd.DataFrame:
    """Slope of each asset's return on the equal-weighted market return, trailing.

    Row ``t`` uses returns through month ``t``. Assets with too little history are
    missing, which the risk model reads as an average beta.
    """
    market = panel.returns.mean(axis=1)
    rolling = panel.returns.rolling(window, min_periods=min_periods)
    covariance = rolling.cov(market)
    variance = market.rolling(window, min_periods=min_periods).var()
    return covariance.div(variance, axis=0).where(panel.listed)


@dataclass(frozen=True)
class RiskHistory:
    """Exposures and the factor and specific returns estimated from them.

    ``factor_returns[s]`` and ``specific_returns[s]`` are realized in month ``s + 1`` from
    exposures at ``s``, so at a decision date ``t`` only rows before ``t`` are known.
    """

    names: list[str]
    #: Shape (dates, assets, factors). Missing where the asset is not investable.
    exposures: np.ndarray
    #: Shape (dates, factors).
    factor_returns: np.ndarray
    #: Shape (dates, assets).
    specific_returns: np.ndarray

    def model_at(
        self, bar: int, use: np.ndarray, factor_window: int = 60, specific_window: int = 36
    ) -> FactorRiskModel | None:
        """The risk model for the assets in ``use`` at decision date ``bar``.

        Returns ``None`` until there are ``MIN_FACTOR_HISTORY`` months of factor returns.
        """
        factors = self.factor_returns[max(0, bar - factor_window) : bar]
        factors = factors[~np.isnan(factors).any(axis=1)]
        if len(factors) < MIN_FACTOR_HISTORY:
            return None
        factor_cov = shrunk_covariance(factors)

        specific = self.specific_returns[max(0, bar - specific_window) : bar][:, use]
        present = ~np.isnan(specific)
        count = present.sum(axis=0)
        enough = count >= MIN_SPECIFIC_HISTORY
        filled = np.where(present, specific, 0.0)
        mean = filled.sum(axis=0) / np.maximum(count, 1)
        squares = (np.where(present, specific - mean, 0.0) ** 2).sum(axis=0)
        variance = np.where(enough, squares / np.maximum(count - 1, 1), np.nan)
        median = np.nanmedian(variance) if np.isfinite(variance).any() else np.nan
        if not np.isfinite(median):
            return None
        variance = np.where(np.isnan(variance), median, variance)
        variance = np.maximum(variance, SPECIFIC_FLOOR * median)
        return FactorRiskModel(self.exposures[bar][use], factor_cov, variance)


def shrunk_covariance(factor_returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage applied to correlations, then scaled back to covariances.

    The factors differ in volatility by orders of magnitude: an industry moves with the
    market, a style factor by a fraction of a percent. Shrinking the covariance matrix
    toward a multiple of the identity would pull every variance toward their average and
    inflate the small ones many times over. Dividing each factor by its own volatility
    first means only the correlations are shrunk, toward zero.
    """
    scale = factor_returns.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    correlation, _ = ledoit_wolf(factor_returns / scale)
    return correlation * np.outer(scale, scale)


def build_risk_history(panel: Panel, styles: Sequence[str], beta_window: int = 36) -> RiskHistory:
    """Estimate factor and specific returns for every month of the panel."""
    investable = panel.investable.to_numpy()
    n_dates, n_assets = investable.shape

    if panel.industry is None:
        market_columns = np.ones((n_assets, 1))
        market_names = ["market"]
    else:
        labels = panel.industry.reindex(panel.assets)
        market_names = sorted(labels.unique())
        market_columns = np.column_stack(
            [(labels == name).to_numpy(float) for name in market_names]
        )

    with_beta = panel.with_characteristic("_beta", trailing_beta(panel, beta_window))
    tilts = standardize(with_beta, ["_beta", *styles])  # (dates, 1 + styles, assets)
    names = [*market_names, "beta", *styles]

    exposures = np.full((n_dates, n_assets, len(names)), np.nan)
    exposures[:, :, : len(market_names)] = market_columns[None, :, :]
    exposures[:, :, len(market_names) :] = np.transpose(tilts, (0, 2, 1))
    exposures[~investable] = np.nan

    forward = panel.forward_returns(1).to_numpy()
    factor_returns = np.full((n_dates, len(names)), np.nan)
    specific = np.full((n_dates, n_assets), np.nan)
    for s in range(n_dates):
        rows = ~np.isnan(forward[s])
        if rows.sum() <= len(names):
            continue
        x, y = exposures[s][rows], forward[s][rows]
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        factor_returns[s] = coefficients
        specific[s, rows] = y - x @ coefficients
    return RiskHistory(names, exposures, factor_returns, specific)
