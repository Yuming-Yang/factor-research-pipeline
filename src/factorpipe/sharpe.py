"""The Sharpe ratio, its standard error, and what is left of it after selection.

All ratios in this module are per period (per month for monthly returns) unless
``periods_per_year`` says otherwise. Inference is done per period, because the
standard-error formula is stated for the per-period ratio.

References: Lo (2002) and Mertens (2002) for the standard error; Bailey and Lopez de
Prado (2012) for the probabilistic Sharpe ratio; Bailey and Lopez de Prado (2014) for the
deflated Sharpe ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from factorpipe.ledger import TrialLedger

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def sharpe_ratio(returns, periods_per_year: int = 12) -> float:
    """Mean over standard deviation (n - 1 in the variance), scaled by sqrt(periods)."""
    r = _clean(returns)
    if r.size < 2:
        return math.nan
    spread = r.std(ddof=1)
    if spread == 0.0:
        return math.nan
    return float(r.mean() / spread * math.sqrt(periods_per_year))


def sharpe_standard_error(returns) -> float:
    """Standard error of the per-period Sharpe ratio, allowing for non-normal returns.

    ``sqrt((1 - skew * SR + (kurt - 1) / 4 * SR**2) / (n - 1))``, with ``kurt`` the plain
    (not excess) kurtosis. For normal returns this is ``sqrt((1 + SR**2 / 2) / (n - 1))``.
    Negative skew and fat tails both widen it. It assumes returns are not autocorrelated.
    """
    r = _clean(returns)
    n = r.size
    sr = sharpe_ratio(r, periods_per_year=1)
    if n < 3 or math.isnan(sr):
        return math.nan
    centred = r - r.mean()
    scale = r.std(ddof=0)
    skew = float(np.mean(centred**3) / scale**3)
    kurt = float(np.mean(centred**4) / scale**4)
    variance = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1)
    return math.sqrt(max(variance, 0.0))


def probabilistic_sharpe_ratio(returns, benchmark: float = 0.0) -> float:
    """Probability that the true per-period Sharpe ratio exceeds ``benchmark``.

    ``Phi((SR - benchmark) / standard_error)``. It is a one-sided confidence level, so
    0.95 corresponds to rejecting ``SR <= benchmark`` at 5%.
    """
    sr = sharpe_ratio(returns, periods_per_year=1)
    se = sharpe_standard_error(returns)
    if math.isnan(sr) or math.isnan(se) or se == 0.0:
        return math.nan
    return _NORMAL.cdf((sr - benchmark) / se)


def expected_max_sharpe(n_trials: int, trial_variance: float) -> float:
    """Sharpe ratio the best of ``n_trials`` strategies with no skill is expected to show.

    If the trials' Sharpe ratios are independent draws with mean zero and variance
    ``trial_variance``, the expected maximum is approximately

        sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))

    with ``g`` the Euler-Mascheroni constant. With one trial there is no selection and
    the hurdle is zero.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1 or trial_variance <= 0.0:
        return 0.0
    a = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    b = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trial_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


@dataclass(frozen=True)
class DeflatedSharpe:
    #: Per-period Sharpe ratio of the strategy being judged.
    sharpe: float
    #: Probability that the true Sharpe ratio is above zero, ignoring selection.
    psr: float
    #: Number of trials the strategy was selected from.
    n_trials: int
    #: Variance of the per-period Sharpe ratios across those trials.
    trial_variance: float
    #: The Sharpe ratio expected from the best of that many trials with no skill.
    hurdle: float
    #: Probability that the true Sharpe ratio is above the hurdle.
    dsr: float


def deflated_sharpe_ratio(returns, n_trials: int, trial_variance: float) -> DeflatedSharpe:
    """Probabilistic Sharpe ratio measured against the expected maximum of ``n_trials``."""
    hurdle = expected_max_sharpe(n_trials, trial_variance)
    return DeflatedSharpe(
        sharpe=sharpe_ratio(returns, periods_per_year=1),
        psr=probabilistic_sharpe_ratio(returns, 0.0),
        n_trials=n_trials,
        trial_variance=trial_variance,
        hurdle=hurdle,
        dsr=probabilistic_sharpe_ratio(returns, hurdle),
    )


def deflated_sharpe_from_ledger(returns, ledger: TrialLedger, data: str) -> DeflatedSharpe:
    """Deflate a strategy's Sharpe ratio by every backtest the ledger holds for ``data``.

    The trial count and the variance of trial Sharpe ratios are read from the ledger, not
    passed in, so they cannot be understated by leaving out the runs that did not work.
    A trial counts if it produced a backtest: its method starts with ``portfolio`` and it
    has a recorded Sharpe ratio. ``data`` is the fingerprint of the panel.
    """
    results = ledger.results()
    sharpes = [
        results[t["id"]]["sharpe"]
        for t in ledger.trials()
        if t["data"] == data and t["method"].startswith("portfolio") and t["id"] in results
    ]
    if not sharpes:
        raise ValueError("the ledger holds no backtest results for this data")
    variance = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    return deflated_sharpe_ratio(returns, len(sharpes), variance)


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[~np.isnan(r)]
