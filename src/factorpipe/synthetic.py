"""A synthetic monthly panel in which the true signal is known.

Research code is hard to test on real data because nobody knows the right answer. Here
the answer is set by construction: some characteristics are *planted factors* with a
chosen information coefficient, the rest are noise with an information coefficient of
exactly zero.

Model. Each characteristic ``z_k`` follows a stationary AR(1) per asset with unit
variance and persistence ``rho_k``. Returns are

    r[i, t+1] = market_mean + m[t+1] + idio_vol * ( sum_k ic_k * z_k[i, t] + c * eps[i, t+1] )

with ``c = sqrt(1 - sum_k ic_k**2)``, so the bracket has unit variance. The market term is
the same for every asset and drops out of any cross-sectional statistic. It follows that
the cross-sectional correlation between ``z_k`` at ``t`` and the return of month ``t+h`` is
``ic_k * rho_k**(h-1)``. That identity is exact when ``delist_prob`` is zero; delisting
returns add return variance that is unrelated to the signal and shrink the correlation
slightly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factorpipe.panel import Panel
from factorpipe.pit import lag_to_availability


@dataclass(frozen=True)
class PlantedFactor:
    """A characteristic that predicts next month's return with a known ``ic``."""

    name: str
    ic: float
    persistence: float = 0.9


@dataclass(frozen=True)
class SyntheticConfig:
    n_assets: int = 500
    n_months: int = 240
    start: str = "2000-01-31"
    planted: tuple[PlantedFactor, ...] = (
        PlantedFactor("planted_strong", ic=0.05, persistence=0.95),
        PlantedFactor("planted_weak", ic=0.02, persistence=0.80),
    )
    n_noise: int = 10
    noise_persistence: float = 0.9
    market_mean: float = 0.006
    market_vol: float = 0.045
    idio_vol: float = 0.10
    # Listing and delisting.
    late_listing_share: float = 0.2
    delist_prob: float = 0.004
    delist_return_bad: float = -0.30
    delist_return_good: float = 0.05
    delist_bad_share: float = 0.7
    # Quarterly reports of one planted factor, published with a lag.
    reported_factor: str | None = "planted_strong"
    report_lag_months: tuple[int, int] = (1, 3)
    restatement_prob: float = 0.05
    restatement_delay_months: int = 6
    restatement_error_vol: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        names = [f.name for f in self.planted]
        if len(set(names)) != len(names):
            raise ValueError("planted factor names must be unique")
        if sum(f.ic**2 for f in self.planted) >= 1.0:
            raise ValueError("the squared ICs of the planted factors must sum to less than 1")
        for f in self.planted:
            if not 0.0 <= f.persistence < 1.0:
                raise ValueError("persistence must be in [0, 1)")
        if not 0.0 <= self.noise_persistence < 1.0:
            raise ValueError("persistence must be in [0, 1)")
        if self.reported_factor is not None and self.reported_factor not in names:
            raise ValueError("reported_factor must be one of the planted factors")
        if self.n_months < 24:
            raise ValueError("n_months must be at least 24")


@dataclass(frozen=True)
class Truth:
    """What the generator planted. Noise factors have an IC of zero."""

    ic: dict[str, float]
    persistence: dict[str, float]

    def ic_at(self, name: str, horizon: int = 1) -> float:
        """Population IC of ``name`` against the single-month return ``horizon`` months ahead."""
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        return self.ic[name] * self.persistence[name] ** (horizon - 1)

    def cumulative_ic_at(self, name: str, horizon: int = 1) -> float:
        """Population IC of ``name`` against the return summed over the next ``horizon`` months.

        The numerator adds the single-month ICs. The denominator is the standard deviation
        of the summed standardized return, which exceeds ``sqrt(horizon)`` because the
        persistent signal makes monthly returns positively correlated: the covariance
        between months ``j`` apart is ``sum_k ic_k**2 * rho_k**j``. This is exact for
        summed returns and an approximation for compounded ones.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        numerator = sum(self.ic_at(name, h) for h in range(1, horizon + 1))
        variance = float(horizon)
        for gap in range(1, horizon):
            covariance = sum(self.ic[k] ** 2 * self.persistence[k] ** gap for k in self.ic)
            variance += 2.0 * (horizon - gap) * covariance
        return numerator / float(np.sqrt(variance))

    def rank_ic_at(self, name: str, horizon: int = 1) -> float:
        """Population Spearman IC. For a bivariate normal pair with correlation ``rho``,
        Spearman's correlation is ``(6 / pi) * asin(rho / 2)``."""
        return float(6.0 / np.pi * np.arcsin(self.ic_at(name, horizon) / 2.0))

    @property
    def planted(self) -> list[str]:
        return [name for name, ic in self.ic.items() if ic != 0.0]

    @property
    def noise(self) -> list[str]:
        return [name for name, ic in self.ic.items() if ic == 0.0]


@dataclass(frozen=True)
class SyntheticPanel:
    panel: Panel
    truth: Truth
    #: Long table of quarterly reports: asset, period_end, publish_date, value.
    fundamentals: pd.DataFrame
    #: Delisting return per delisted asset, already compounded into its last monthly return.
    delisting_returns: pd.Series = field(repr=False)


def make_panel(config: SyntheticConfig | None = None) -> SyntheticPanel:
    """Generate a panel. The same config always gives the same panel."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    n_dates, n_assets = cfg.n_months, cfg.n_assets
    dates = pd.date_range(cfg.start, periods=n_dates, freq="ME")
    assets = pd.Index([f"A{i:04d}" for i in range(n_assets)], name="asset")

    names = [f.name for f in cfg.planted] + [f"noise_{j:02d}" for j in range(cfg.n_noise)]
    ic = np.array([f.ic for f in cfg.planted] + [0.0] * cfg.n_noise)
    rho = np.array([f.persistence for f in cfg.planted] + [cfg.noise_persistence] * cfg.n_noise)

    # Characteristics: stationary AR(1) with unit variance, shape (dates, factors, assets).
    z = np.empty((n_dates, len(names), n_assets))
    z[0] = rng.standard_normal((len(names), n_assets))
    innovation_scale = np.sqrt(1.0 - rho**2)[:, None]
    for t in range(1, n_dates):
        z[t] = rho[:, None] * z[t - 1] + innovation_scale * rng.standard_normal(z[0].shape)

    # Returns: month t+1 is driven by the characteristics known at the end of month t.
    market = cfg.market_vol * rng.standard_normal(n_dates)
    eps = rng.standard_normal((n_dates, n_assets))
    residual_scale = np.sqrt(1.0 - np.sum(ic**2))
    standardized = np.empty((n_dates, n_assets))
    standardized[0] = eps[0]
    standardized[1:] = np.einsum("k,tkn->tn", ic, z[:-1]) + residual_scale * eps[1:]
    returns = cfg.market_mean + market[:, None] + cfg.idio_vol * standardized
    returns = np.maximum(returns, -0.99)

    # Listing spells. Most assets list at the start; the rest list later.
    first = np.zeros(n_assets, dtype=int)
    late = rng.random(n_assets) < cfg.late_listing_share
    first[late] = rng.integers(1, n_dates - 12, size=int(late.sum()))
    if cfg.delist_prob > 0:
        life = rng.geometric(cfg.delist_prob, size=n_assets)
    else:
        life = np.full(n_assets, n_dates)
    last = first + life  # index of the last listed month
    delisted = last < n_dates - 1
    last = np.minimum(last, n_dates - 1)

    bad = rng.random(n_assets) < cfg.delist_bad_share
    delist_return = np.where(bad, cfg.delist_return_bad, cfg.delist_return_good)
    idx = np.flatnonzero(delisted)
    returns[last[idx], idx] = (1.0 + returns[last[idx], idx]) * (1.0 + delist_return[idx]) - 1.0

    month = np.arange(n_dates)[:, None]
    listed = (month >= first[None, :]) & (month <= last[None, :])
    returns_frame = pd.DataFrame(np.where(listed, returns, np.nan), index=dates, columns=assets)
    characteristics = {
        name: pd.DataFrame(np.where(listed, z[:, k, :], np.nan), index=dates, columns=assets)
        for k, name in enumerate(names)
    }

    fundamentals = _make_reports(cfg, rng, dates, assets, z, names, listed)
    if cfg.reported_factor is not None:
        reported = lag_to_availability(fundamentals, dates, assets)
        characteristics[f"{cfg.reported_factor}_reported"] = reported.where(listed)

    truth = Truth(
        ic=dict(zip(names, ic.tolist(), strict=True)),
        persistence=dict(zip(names, rho.tolist(), strict=True)),
    )
    return SyntheticPanel(
        panel=Panel(returns_frame, characteristics),
        truth=truth,
        fundamentals=fundamentals,
        delisting_returns=pd.Series(delist_return[idx], index=assets[idx], name="delisting_return"),
    )


def _make_reports(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    assets: pd.Index,
    z: np.ndarray,
    names: list[str],
    listed: np.ndarray,
) -> pd.DataFrame:
    """Quarterly reports of one factor: its value at the quarter end, published later.

    A share of first reports carry an error and are corrected by a restatement.
    """
    columns = ["asset", "period_end", "publish_date", "value"]
    if cfg.reported_factor is None:
        return pd.DataFrame(columns=columns)
    k = names.index(cfg.reported_factor)
    quarter_ends = np.flatnonzero(dates.month % 3 == 0)
    t_idx, a_idx = np.nonzero(listed[quarter_ends])
    t_idx = quarter_ends[t_idx]
    n = len(t_idx)

    true_value = z[t_idx, k, a_idx]
    lo, hi = cfg.report_lag_months
    lag = rng.integers(lo, hi + 1, size=n)
    period_end = dates[t_idx]
    publish = _add_months(period_end, lag)

    restated = rng.random(n) < cfg.restatement_prob
    first_value = np.where(
        restated, true_value + cfg.restatement_error_vol * rng.standard_normal(n), true_value
    )
    original = pd.DataFrame(
        {
            "asset": assets[a_idx],
            "period_end": period_end,
            "publish_date": publish,
            "value": first_value,
        }
    )
    correction = pd.DataFrame(
        {
            "asset": assets[a_idx][restated],
            "period_end": period_end[restated],
            "publish_date": _add_months(publish[restated], cfg.restatement_delay_months),
            "value": true_value[restated],
        }
    )
    table = pd.concat([original, correction], ignore_index=True)
    return table.sort_values(["asset", "publish_date", "period_end"], ignore_index=True)


def _add_months(dates: pd.DatetimeIndex, months: np.ndarray | int) -> pd.DatetimeIndex:
    """Shift month-end dates by whole months, staying on month ends."""
    periods = dates.to_period("M") + months
    return periods.to_timestamp(how="end").normalize()
