"""From scores to positions, and through the backtest engine.

Two ways to turn a composite score into a dollar-neutral book:

- ``quantile``: equal weights long the top group and short the bottom group. No
  estimates beyond the ranking, so it is the baseline.
- ``mvo``: scores become expected returns, and ``robust-mvo`` trades them off against
  risk, turnover and costs under position and leverage caps.

Both run inside the engine's decision callback, so the optimizer starts from the weights
the book has actually drifted to, and every input is sliced at the decision date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from robust_mvo.covariance import ledoit_wolf
from robust_mvo.optimizer import Constraints, CostModel, Inputs
from robust_mvo.risk_model import FactorRiskModel
from robust_mvo.robust import robust_optimize

from factorpipe.backtest import BacktestRun, Snapshot, market_from_panel, run_backtest
from factorpipe.combine import Combination, factor_returns, standardize
from factorpipe.evaluation import MIN_ASSETS, information_coefficient
from factorpipe.ledger import TrialLedger, fingerprint
from factorpipe.panel import Panel
from factorpipe.sharpe import sharpe_ratio

PORTFOLIOS = ("quantile", "mvo")

DEFAULTS: dict[str, dict[str, Any]] = {
    "quantile": {"n_quantiles": 5, "gross": 2.0},
    "mvo": {
        "gamma": 5.0,
        "turnover_penalty": 0.0,
        "max_position": 0.02,
        "max_gross": 2.0,
        "ic_window": 36,
        "vol_window": 36,
        "risk_window": 60,
    },
}

#: Fewest months of history before a trailing estimate is used.
MIN_HISTORY = 12
#: Fewest months of factor returns before a factor covariance is estimated.
MIN_RISK_HISTORY = 24
#: Idiosyncratic variance is never set below this share of total variance.
IDIO_FLOOR = 0.25


def trailing_volatility(panel: Panel, window: int = 36) -> pd.DataFrame:
    """Standard deviation of each asset's monthly returns over the last ``window`` months.

    Row ``t`` uses returns through month ``t``. A listed asset with fewer than
    ``MIN_HISTORY`` months gets the cross-sectional median of that date.
    """
    vol = panel.returns.rolling(window, min_periods=MIN_HISTORY).std()
    median = vol.median(axis=1)
    filled = vol.T.fillna(median).T
    return filled.where(panel.listed)


def trailing_ic(
    panel: Panel, scores: pd.DataFrame, horizon: int = 1, window: int = 36, embargo: int = 0
) -> pd.Series:
    """Mean rank IC of ``scores`` over the last ``window`` values known at each date.

    The IC formed at ``s`` is known at ``s + horizon + embargo``.
    """
    scored = panel.with_characteristic("_score", scores)
    known = information_coefficient(scored, "_score", horizon).shift(horizon + embargo)
    return known.rolling(window, min_periods=MIN_HISTORY).mean()


def grinold_alpha(scores: np.ndarray, ic: float, volatility: np.ndarray) -> np.ndarray:
    """Expected return per asset: ``ic * volatility * z``, with ``z`` the standardized score.

    This is the forecasting rule of Grinold (1994). A score has no units; this gives it
    the units of a return. If the score and the return are jointly normal, the expected
    return given the score is ``corr * sd(return) * z``, and the IC is that correlation.
    """
    centred = scores - scores.mean()
    spread = scores.std()
    z = centred / spread if spread > 0 else np.zeros_like(scores)
    return ic * volatility * z


@dataclass
class QuantilePortfolio:
    """Equal weights long the top quantile and short the bottom one."""

    scores: pd.DataFrame
    n_quantiles: int = 5
    gross: float = 2.0

    def __call__(self, t: pd.Timestamp, snapshot: Snapshot) -> pd.Series | None:
        row = self.scores.iloc[snapshot.bar].dropna()
        if len(row) < max(MIN_ASSETS, self.n_quantiles):
            return None
        group = np.ceil(row.rank(method="first") * self.n_quantiles / len(row))
        top, bottom = group == self.n_quantiles, group == 1
        weights = pd.Series(0.0, index=row.index)
        weights[top] = self.gross / 2.0 / top.sum()
        weights[bottom] = -self.gross / 2.0 / bottom.sum()
        return weights


@dataclass
class MvoPortfolio:
    """Mean-variance positions from ``robust-mvo``, rebuilt at every decision date.

    Risk model at ``t``: exposures are the standardized factors at ``t``; the factor
    covariance is the Ledoit-Wolf estimate on factor returns realized by ``t``; the
    idiosyncratic variance is each asset's trailing variance minus its factor part,
    floored at a quarter of the total.
    """

    panel: Panel
    scores: pd.DataFrame
    factors: list[str]
    horizon: int = 1
    embargo: int = 0
    cost_rate: float = 0.0
    gamma: float = 5.0
    turnover_penalty: float = 0.0
    max_position: float = 0.02
    max_gross: float = 2.0
    ic_window: int = 36
    vol_window: int = 36
    risk_window: int = 60
    #: Degradation level of each solve, by decision date (0 = full problem solved).
    levels: dict[pd.Timestamp, int] = field(default_factory=dict)
    #: Target weights returned at each decision date, before any drift.
    targets: dict[pd.Timestamp, pd.Series] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._z = standardize(self.panel, self.factors)
        self._factor_returns = factor_returns(self.panel, self._z, self.factors).to_numpy()
        self._vol = trailing_volatility(self.panel, self.vol_window).to_numpy()
        self._ic = trailing_ic(
            self.panel, self.scores, self.horizon, self.ic_window, self.embargo
        ).to_numpy()
        self._scores = self.scores.to_numpy(dtype=float)

    def __call__(self, t: pd.Timestamp, snapshot: Snapshot) -> pd.Series | None:
        i = snapshot.bar
        use = ~np.isnan(self._scores[i]) & ~np.isnan(self._vol[i])
        ic = self._ic[i]
        # Factor returns formed before t are realized by t; the one formed at t is not.
        history = self._factor_returns[max(0, i - self.risk_window) : i]
        history = history[~np.isnan(history).any(axis=1)]
        if use.sum() < MIN_ASSETS or np.isnan(ic) or len(history) < MIN_RISK_HISTORY:
            return None

        exposures = self._z[i][:, use].T
        factor_cov, _ = ledoit_wolf(history)
        vol = self._vol[i][use]
        systematic = np.einsum("nk,kl,nl->n", exposures, factor_cov, exposures)
        idio_var = np.maximum(vol**2 - systematic, IDIO_FLOOR * vol**2)
        inputs = Inputs(
            alpha=grinold_alpha(self._scores[i][use], max(ic, 0.0), vol),
            risk=FactorRiskModel(exposures, factor_cov, idio_var),
            w0=snapshot.weights.to_numpy()[use],
            gamma=self.gamma,
            turnover_penalty=self.turnover_penalty,
            costs=CostModel(linear=self.cost_rate),
            constraints=Constraints(
                dollar_neutral=True, max_position=self.max_position, max_gross=self.max_gross
            ),
        )
        weights, report = robust_optimize(inputs)
        self.levels[t] = report.level
        self.targets[t] = pd.Series(weights, index=self.panel.assets[use])
        return self.targets[t]


@dataclass(frozen=True)
class StrategyRun:
    portfolio: str
    params: dict[str, Any]
    run: BacktestRun
    #: Degradation level per decision date for ``mvo``; empty for ``quantile``.
    levels: pd.Series
    trial_id: str
    #: Target weights by decision date for ``mvo``; empty for ``quantile``.
    targets: dict[pd.Timestamp, pd.Series] = field(default_factory=dict, repr=False)


def backtest_strategy(
    panel: Panel,
    combination: Combination,
    *,
    ledger: TrialLedger,
    portfolio: str = "mvo",
    cost_rate: float = 0.0,
    borrow_rate: float = 0.0,
    **params: Any,
) -> StrategyRun:
    """Build positions from a combination's scores and run them through the engine.

    The portfolio settings are a trial of their own: the ledger records them together
    with the identifier of the combination they were applied to.
    """
    if portfolio not in PORTFOLIOS:
        raise ValueError(f"unknown portfolio {portfolio!r}; choose from {PORTFOLIOS}")
    unknown = set(params) - set(DEFAULTS[portfolio])
    if unknown:
        raise ValueError(f"unknown parameters for {portfolio!r}: {sorted(unknown)}")
    settings = {**DEFAULTS[portfolio], **params}
    recorded = {
        "combination": combination.trial_id,
        "cost_rate": cost_rate,
        "borrow_rate": borrow_rate,
        **settings,
    }
    trial_id = ledger.record(f"portfolio:{portfolio}", recorded, fingerprint(panel))

    market = market_from_panel(panel, cost_rate, borrow_rate)
    if portfolio == "quantile":
        decide = QuantilePortfolio(combination.scores, **settings)
        levels = pd.Series(dtype=int)
        run = run_backtest(market, decide)
    else:
        decide = MvoPortfolio(
            panel,
            combination.scores,
            list(combination.params["factors"]),
            horizon=combination.params["horizon"],
            embargo=combination.params["embargo"],
            cost_rate=cost_rate,
            **settings,
        )
        run = run_backtest(market, decide)
        levels = pd.Series(decide.levels, dtype=int)
    targets = getattr(decide, "targets", {})
    active = run.active_returns
    if len(active) > 1 and active.std(ddof=1) > 0:
        ledger.record_result(
            trial_id, sharpe=float(sharpe_ratio(active, periods_per_year=1)), n_obs=len(active)
        )
    return StrategyRun(portfolio, recorded, run, levels, trial_id, targets)
