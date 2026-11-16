"""Experiments on the synthetic panel that the README and the tests both use."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from factorpipe.bootstrap import BootstrapInterval, sharpe_interval
from factorpipe.combine import combine
from factorpipe.ledger import TrialLedger, fingerprint
from factorpipe.panel import Panel
from factorpipe.pbo import probability_of_backtest_overfitting
from factorpipe.portfolio import backtest_strategy
from factorpipe.sharpe import DeflatedSharpe, deflated_sharpe_from_ledger, sharpe_ratio


@dataclass(frozen=True)
class SelectionResult:
    """What happens when the best of many single-factor strategies is picked."""

    best: str
    #: Annualized Sharpe ratio of every factor's long-short portfolio.
    sharpes: pd.Series
    deflated: DeflatedSharpe
    pbo: float
    interval: BootstrapInterval

    @property
    def best_annual_sharpe(self) -> float:
        return float(self.sharpes[self.best])


def select_best_factor(panel: Panel, ledger: TrialLedger, n_blocks: int = 10) -> SelectionResult:
    """Backtest a long-short quantile portfolio on every factor, then judge the winner.

    This is the search a researcher runs when looking for a factor that works. Every
    backtest goes into the ledger, the winner's Sharpe ratio is deflated by the number of
    backtests, and the probability of backtest overfitting is computed over all of them.
    """
    returns = {}
    for name in panel.characteristics:
        combination = combine(panel, "equal", ledger=ledger, factors=[name])
        result = backtest_strategy(panel, combination, ledger=ledger, portfolio="quantile")
        returns[name] = result.run.active_returns
    table = pd.DataFrame(returns).dropna()
    sharpes = table.apply(sharpe_ratio)
    best = str(sharpes.idxmax())
    return SelectionResult(
        best=best,
        sharpes=sharpes,
        deflated=deflated_sharpe_from_ledger(table[best], ledger, fingerprint(panel)),
        pbo=probability_of_backtest_overfitting(table, n_blocks).pbo,
        interval=sharpe_interval(table[best]),
    )


def annualize(sharpe: float, periods_per_year: int = 12) -> float:
    return sharpe * math.sqrt(periods_per_year)
