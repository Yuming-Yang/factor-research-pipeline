"""Combining factors into one score, walk-forward.

Every method answers the same question at the end of month ``t``: given what was known
at ``t``, what score does each asset get? "Known at ``t``" is enforced in one place for
each kind of history:

- An IC with horizon ``h`` formed at ``s`` uses returns through ``s + h``. It is usable at
  ``t`` only if ``s + h + embargo <= t``.
- A one-month factor return formed at ``s`` is usable at ``t`` if ``s + 1 + embargo <= t``.
- A regression row formed at ``s`` with an ``h``-month label is usable if
  ``s + h + embargo <= t``.

Dropping rows whose label window reaches past ``t`` is the *purge*. The *embargo* is an
extra gap on top of it, for dependence that outlasts the label window. In a walk-forward
design the training data always precedes the evaluation date, so the embargo sits before
each evaluation date; there is no training data after it to protect.

Every call goes through ``combine``, which records the trial in a ledger first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from robust_mvo.covariance import ledoit_wolf
from sklearn.ensemble import HistGradientBoostingRegressor

from factorpipe.evaluation import information_coefficient
from factorpipe.ledger import TrialLedger, fingerprint
from factorpipe.panel import Panel

METHODS = ("equal", "icir", "momentum", "mvo", "ridge", "trees")

DEFAULTS: dict[str, dict[str, Any]] = {
    "equal": {},
    "icir": {"window": 36},
    "momentum": {"window": 12},
    "mvo": {"window": 60},
    "ridge": {"min_train": 36, "window": None, "alphas": (0.0, 0.1, 1.0, 10.0)},
    "trees": {
        "min_train": 36,
        "window": 120,
        "refit_every": 12,
        "max_iter": 60,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_samples_leaf": 50,
    },
}


@dataclass(frozen=True)
class Combination:
    method: str
    params: dict[str, Any]
    #: Composite score by date and asset. Missing until the method has enough history.
    scores: pd.DataFrame
    #: Weight on each factor by date, for the linear methods. Absolute weights sum to 1.
    weights: pd.DataFrame | None
    trial_id: str


def combine(
    panel: Panel,
    method: str,
    *,
    ledger: TrialLedger,
    factors: Sequence[str] | None = None,
    horizon: int = 1,
    embargo: int = 0,
    **params: Any,
) -> Combination:
    """Combine ``factors`` into one score with ``method``, using only past data.

    ``ledger`` is required. The trial is written to it before anything is computed, so a
    configuration cannot be evaluated without being counted.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    unknown = set(params) - set(DEFAULTS[method])
    if unknown:
        raise ValueError(f"unknown parameters for {method!r}: {sorted(unknown)}")
    if horizon < 1 or embargo < 0:
        raise ValueError("horizon must be at least 1 and embargo cannot be negative")
    names = list(panel.characteristics) if factors is None else list(factors)
    settings = {**DEFAULTS[method], **params}
    recorded = {"factors": names, "horizon": horizon, "embargo": embargo, **settings}
    trial_id = ledger.record(method, recorded, fingerprint(panel))

    z = standardize(panel, names)
    if method == "trees":
        scores = _tree_scores(panel, z, horizon, embargo, settings)
        weights = None
    else:
        weights = _linear_weights(panel, names, z, method, horizon, embargo, settings)
        scores = np.einsum("tk,tkn->tn", weights.to_numpy(), z)
    frame = pd.DataFrame(scores, index=panel.dates, columns=panel.assets)
    frame = frame.where(panel.investable)
    return Combination(method, recorded, frame, weights, trial_id)


def standardize(panel: Panel, names: Sequence[str], clip: float = 3.0) -> np.ndarray:
    """Cross-sectional z-scores, shape (dates, factors, assets).

    Each factor is centred and scaled across the assets that are investable on that
    date, then clipped at ``clip`` standard deviations. An investable asset with a
    missing factor value gets zero, the cross-sectional mean. Assets that are not
    investable are missing.
    """
    investable = panel.investable.to_numpy()
    out = np.full((len(panel.dates), len(names), len(panel.assets)), np.nan)
    for k, name in enumerate(names):
        x = np.where(investable, panel.characteristics[name].to_numpy(dtype=float), np.nan)
        count = (~np.isnan(x)).sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nansum(x, axis=1, keepdims=True) / count
            centred = x - mean
            std = np.sqrt(np.nansum(centred**2, axis=1, keepdims=True) / count)
            z = np.clip(centred / std, -clip, clip)
        z = np.where(np.isnan(z), 0.0, z)
        out[:, k, :] = np.where(investable, z, np.nan)
    return out


def factor_returns(panel: Panel, z: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """One-month return of each factor, indexed by formation date.

    The return formed at ``s`` is the average of ``z * (r - mean r)`` over assets, with
    ``r`` the return of month ``s + 1``. That is the slope of a cross-sectional regression
    of next month's return on the standardized factor, and also the return of a
    dollar-neutral portfolio with weights ``z / N``.
    """
    forward = panel.forward_returns(1).to_numpy()
    valid = ~np.isnan(forward)
    count = valid.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        demeaned = forward - np.nansum(forward, axis=1, keepdims=True) / count
        product = np.where(valid[:, None, :], z * demeaned[:, None, :], 0.0)
        values = product.sum(axis=2) / count
    values[count[:, 0] == 0] = np.nan
    return pd.DataFrame(values, index=panel.dates, columns=list(names))


def _linear_weights(
    panel: Panel,
    names: list[str],
    z: np.ndarray,
    method: str,
    horizon: int,
    embargo: int,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if method == "equal":
        raw = pd.DataFrame(1.0, index=panel.dates, columns=names)
    elif method == "icir":
        ic = pd.DataFrame({n: information_coefficient(panel, n, horizon) for n in names})
        known = ic.shift(horizon + embargo)
        rolling = known.rolling(settings["window"], min_periods=settings["window"])
        raw = rolling.mean() / rolling.std(ddof=1)
    elif method == "momentum":
        known = factor_returns(panel, z, names).shift(1 + embargo)
        raw = known.rolling(settings["window"], min_periods=settings["window"]).mean()
    elif method == "mvo":
        known = factor_returns(panel, z, names).shift(1 + embargo)
        raw = _mvo_weights(known, settings["window"])
    else:
        raw = _ridge_weights(panel, names, z, horizon, embargo, settings)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    total = raw.abs().sum(axis=1, skipna=False)
    return raw.div(total.where(total > 0), axis=0)


def _mvo_weights(known: pd.DataFrame, window: int) -> pd.DataFrame:
    """Mean-variance weights on factor returns: Sigma^-1 mu on a trailing window.

    Sigma is the Ledoit-Wolf estimate from robust-mvo. With a dozen factors and five
    years of months the sample covariance is usable but noisy, and its inverse amplifies
    that noise.
    """
    values = known.to_numpy()
    out = np.full(values.shape, np.nan)
    for t in range(window - 1, len(values)):
        block = values[t - window + 1 : t + 1]
        if np.isnan(block).any():
            continue
        sigma, _ = ledoit_wolf(block)
        out[t] = np.linalg.solve(sigma, block.mean(axis=0))
    return pd.DataFrame(out, index=known.index, columns=known.columns)


def _labels(panel: Panel, horizon: int) -> np.ndarray:
    """Forward returns with the cross-sectional mean removed, so the market drops out."""
    forward = panel.forward_returns(horizon).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        count = (~np.isnan(forward)).sum(axis=1, keepdims=True)
        return forward - np.nansum(forward, axis=1, keepdims=True) / count


def _ridge_weights(
    panel: Panel,
    names: list[str],
    z: np.ndarray,
    horizon: int,
    embargo: int,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Pooled ridge regression of forward returns on factors, refitted every month.

    Each formation date contributes its Gram matrix ``X'X``, its cross-product ``X'y``
    and ``y'y``. A fit over any set of dates is then a sum of small matrices, which makes
    a monthly refit cheap. The penalty is ``alpha`` times the number of training rows, so
    ``alpha`` has the same meaning at every date. It is chosen on the most recent quarter
    of the training dates, which are themselves purged from the dates used to fit.
    """
    y = _labels(panel, horizon)
    n_dates, n_factors = z.shape[0], z.shape[1]
    gram = np.zeros((n_dates, n_factors, n_factors))
    cross = np.zeros((n_dates, n_factors))
    yy = np.zeros(n_dates)
    rows = np.zeros(n_dates)
    for s in range(n_dates):
        use = ~np.isnan(y[s])
        if not use.any():
            continue
        x = z[s][:, use].T
        gram[s], cross[s] = x.T @ x, x.T @ y[s][use]
        yy[s], rows[s] = float(y[s][use] @ y[s][use]), use.sum()

    def fit(dates: np.ndarray, alpha: float) -> np.ndarray:
        g, c, n = gram[dates].sum(axis=0), cross[dates].sum(axis=0), rows[dates].sum()
        return np.linalg.solve(g + alpha * n * np.eye(n_factors) + 1e-12 * np.eye(n_factors), c)

    def squared_error(dates: np.ndarray, beta: np.ndarray) -> float:
        g, c = gram[dates].sum(axis=0), cross[dates].sum(axis=0)
        return float(yy[dates].sum() - 2.0 * beta @ c + beta @ g @ beta)

    out = np.full((n_dates, n_factors), np.nan)
    has_rows = rows > 0
    for t in range(n_dates):
        last = t - horizon - embargo
        first = 0 if settings["window"] is None else max(0, last - settings["window"] + 1)
        train = np.arange(first, last + 1)
        train = train[has_rows[train]] if last >= first else train[:0]
        if len(train) < settings["min_train"]:
            continue
        n_validate = max(1, len(train) // 4)
        validate = train[-n_validate:]
        fit_dates = train[train + horizon <= validate[0]]
        alpha = settings["alphas"][0]
        if len(settings["alphas"]) > 1 and len(fit_dates) > 0:
            errors = [squared_error(validate, fit(fit_dates, a)) for a in settings["alphas"]]
            alpha = settings["alphas"][int(np.argmin(errors))]
        out[t] = fit(train, alpha)
    return pd.DataFrame(out, index=panel.dates, columns=names)


def _tree_scores(
    panel: Panel, z: np.ndarray, horizon: int, embargo: int, settings: dict[str, Any]
) -> np.ndarray:
    """Gradient-boosted trees on pooled rows, refitted every ``refit_every`` months.

    The model fitted at ``t`` scores dates ``t`` through ``t + refit_every - 1``. It is
    small on purpose: with an IC of a few percent there is little signal per row, and a
    large model fits noise.
    """
    y = _labels(panel, horizon)
    n_dates = z.shape[0]
    scores = np.full((n_dates, z.shape[2]), np.nan)
    model = None
    since_fit = 0
    for t in range(n_dates):
        last = t - horizon - embargo
        first = 0 if settings["window"] is None else max(0, last - settings["window"] + 1)
        if model is None or since_fit >= settings["refit_every"]:
            train = [s for s in range(first, last + 1) if not np.isnan(y[s]).all()]
            if len(train) < settings["min_train"]:
                continue
            x_rows = [z[s][:, ~np.isnan(y[s])].T for s in train]
            y_rows = [y[s][~np.isnan(y[s])] for s in train]
            model = HistGradientBoostingRegressor(
                max_iter=settings["max_iter"],
                max_depth=settings["max_depth"],
                learning_rate=settings["learning_rate"],
                min_samples_leaf=settings["min_samples_leaf"],
                early_stopping=False,
                random_state=0,
            ).fit(np.vstack(x_rows), np.concatenate(y_rows))
            since_fit = 0
        present = ~np.isnan(z[t]).any(axis=0)
        if present.any():
            scores[t, present] = model.predict(z[t][:, present].T)
        since_fit += 1
    return scores
