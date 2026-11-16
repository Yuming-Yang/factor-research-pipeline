import numpy as np
import pandas as pd
import pytest
from robust_mvo.covariance import ledoit_wolf
from robust_mvo.risk_model import FactorRiskModel

from factorpipe.combine import combine, factor_returns, standardize
from factorpipe.ledger import TrialLedger
from factorpipe.panel import Panel, PanelError
from factorpipe.portfolio import backtest_strategy, trailing_volatility
from factorpipe.risk import (
    MIN_FACTOR_HISTORY,
    build_risk_history,
    shrunk_covariance,
    trailing_beta,
)
from factorpipe.synthetic import SyntheticConfig, make_panel

STRUCTURED = SyntheticConfig(
    n_assets=300,
    n_months=150,
    n_noise=2,
    reported_factor=None,
    beta_dispersion=0.4,
    n_industries=8,
    seed=3,
)


@pytest.fixture(scope="module")
def synthetic():
    return make_panel(STRUCTURED)


@pytest.fixture(scope="module")
def history(synthetic):
    return build_risk_history(synthetic.panel, list(synthetic.panel.characteristics))


def perturb_after(panel: Panel, cutoff: int) -> Panel:
    rng = np.random.default_rng(0)

    def scramble(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
        values = frame.to_numpy(dtype=float).copy()
        noise = scale * rng.standard_normal(values[cutoff + 1 :].shape)
        values[cutoff + 1 :] = np.where(np.isnan(values[cutoff + 1 :]), np.nan, noise)
        return pd.DataFrame(values, index=frame.index, columns=frame.columns)

    return Panel(
        scramble(panel.returns, 0.1),
        {name: scramble(frame, 1.0) for name, frame in panel.characteristics.items()},
        panel.industry,
    )


# --- the generator's risk structure ------------------------------------------------------


def test_risk_structure_is_off_by_default_and_changes_only_returns_when_on():
    base = make_panel(SyntheticConfig(n_assets=100, n_months=60, seed=9))
    assert base.panel.industry is None
    assert (base.betas == 1.0).all()
    tilted = make_panel(
        SyntheticConfig(n_assets=100, n_months=60, seed=9, beta_dispersion=0.3, n_industries=4)
    )
    assert tilted.panel.industry.nunique() == 4
    assert tilted.betas.std() == pytest.approx(0.3, abs=0.06)
    pd.testing.assert_frame_equal(
        base.panel.characteristics["planted_strong"], tilted.panel.characteristics["planted_strong"]
    )
    pd.testing.assert_series_equal(base.market_returns, tilted.market_returns)
    assert not base.panel.returns.equals(tilted.panel.returns)


def test_a_panel_needs_an_industry_label_for_every_asset():
    panel = make_panel(SyntheticConfig(n_assets=50, n_months=30, seed=1)).panel
    with pytest.raises(PanelError, match="industry"):
        Panel(panel.returns, industry=pd.Series({"A0000": "x"}))


# --- exposures and factor returns --------------------------------------------------------


def test_trailing_beta_tracks_the_true_betas_and_uses_only_the_past(synthetic):
    beta = trailing_beta(synthetic.panel, window=60)
    last = beta.iloc[-1].dropna()
    assert np.corrcoef(last, synthetic.betas[last.index])[0, 1] > 0.6
    assert last.mean() == pytest.approx(1.0, abs=0.1)
    changed = trailing_beta(perturb_after(synthetic.panel, 100), window=60)
    pd.testing.assert_frame_equal(beta.iloc[:101], changed.iloc[:101])


def test_exposure_columns(synthetic, history):
    assert history.names[:8] == [f"I{g:02d}" for g in range(8)]
    assert history.names[8] == "beta"
    assert history.names[9:] == list(synthetic.panel.characteristics)
    investable = synthetic.panel.investable.to_numpy()
    x = history.exposures[80][investable[80]]
    np.testing.assert_allclose(x[:, :8].sum(axis=1), 1.0)  # industries carry the market
    assert np.abs(x[:, 8:].mean(axis=0)).max() < 0.03  # tilts are centred
    assert np.isnan(history.exposures[80][~investable[80]]).all()


def test_without_industries_there_is_one_market_column():
    panel = make_panel(SyntheticConfig(n_assets=120, n_months=60, n_noise=1, seed=2)).panel
    history = build_risk_history(panel, ["planted_strong"])
    assert history.names == ["market", "beta", "planted_strong"]
    assert np.nanmin(history.exposures[30][:, 0]) == 1.0


def test_factor_and_specific_returns_add_up_to_the_asset_returns(synthetic, history):
    forward = synthetic.panel.forward_returns(1).to_numpy()
    for s in (30, 90, 140):
        rows = ~np.isnan(forward[s])
        rebuilt = history.exposures[s][rows] @ history.factor_returns[s]
        np.testing.assert_allclose(
            rebuilt + history.specific_returns[s][rows], forward[s][rows], atol=1e-12
        )
    assert np.isnan(history.factor_returns[-1]).all()  # nothing is realized after the last date


def test_the_industry_factors_recover_the_market_return(synthetic, history):
    realized = synthetic.market_returns.shift(-1).to_numpy()
    ok = ~np.isnan(history.factor_returns).any(axis=1) & ~np.isnan(realized)
    estimated = history.factor_returns[:, :8].mean(axis=1)
    assert ok.sum() > 140
    assert np.corrcoef(estimated[ok], realized[ok])[0, 1] > 0.9
    # The beta factor pays off when the market is up.
    beta_factor = history.factor_returns[:, 8]
    assert np.corrcoef(beta_factor[ok], realized[ok])[0, 1] > 0.5


# --- covariance and the model at a date --------------------------------------------------


def test_shrinkage_keeps_variances_and_pulls_correlations_toward_zero():
    rng = np.random.default_rng(0)
    common = rng.standard_normal((40, 1))
    scales = np.array([0.05, 0.05, 0.002, 0.0005, 0.03, 0.001])
    factors = (0.6 * common + rng.standard_normal((40, 6))) * scales
    shrunk = shrunk_covariance(factors)
    sample = np.cov(factors, rowvar=False, ddof=0)
    np.testing.assert_allclose(np.diag(shrunk), np.diag(sample), rtol=1e-10)
    off = ~np.eye(6, dtype=bool)
    assert (np.abs(shrunk[off]) <= np.abs(sample[off]) + 1e-15).all()
    assert np.abs(shrunk[off]).sum() < 0.95 * np.abs(sample[off]).sum()
    # Shrinking covariances toward one common variance would inflate the smallest factor.
    plain, _ = ledoit_wolf(factors)
    assert plain[3, 3] > 20 * sample[3, 3]


def test_no_model_before_there_is_enough_history(synthetic, history):
    use = synthetic.panel.investable.to_numpy()[MIN_FACTOR_HISTORY - 1]
    assert history.model_at(MIN_FACTOR_HISTORY - 1, use) is None
    use = synthetic.panel.investable.to_numpy()[60]
    model = history.model_at(60, use)
    assert model.exposures.shape == (use.sum(), len(history.names))
    assert (model.idio_var > 0).all()
    assert model.idio_var.min() >= 0.25 * np.median(model.idio_var) - 1e-15


def test_the_model_at_a_date_does_not_depend_on_later_data(synthetic):
    styles = list(synthetic.panel.characteristics)
    cutoff = 100
    use = synthetic.panel.investable.to_numpy()[cutoff]
    before = build_risk_history(synthetic.panel, styles).model_at(cutoff, use)
    after = build_risk_history(perturb_after(synthetic.panel, cutoff), styles).model_at(cutoff, use)
    np.testing.assert_allclose(before.exposures, after.exposures, atol=1e-12)
    np.testing.assert_allclose(before.factor_cov, after.factor_cov, atol=1e-15)
    np.testing.assert_allclose(before.idio_var, after.idio_var, atol=1e-15)


def test_predicted_risk_matches_realized_risk_where_the_old_model_fails(synthetic, history):
    """Bias statistic: the standard deviation of realized return over predicted volatility,
    across months. A calibrated model gives 1; above 1 means risk was understated."""
    panel = synthetic.panel
    styles = list(panel.characteristics)
    investable = panel.investable.to_numpy()
    forward = panel.forward_returns(1).to_numpy()
    z = standardize(panel, styles)
    style_returns = factor_returns(panel, z, styles).to_numpy()
    vol = trailing_volatility(panel).to_numpy()
    beta = trailing_beta(panel).to_numpy()
    in_first_industry = (panel.industry == "I00").to_numpy()
    rng = np.random.default_rng(0)

    ratios: dict[str, list[float]] = {}
    for i in range(70, 148):
        use = investable[i] & ~np.isnan(vol[i]) & ~np.isnan(beta[i])
        structural = history.model_at(i, use)
        f, _ = ledoit_wolf(style_returns[i - 60 : i])
        b = z[i][:, use].T
        v = vol[i][use]
        d = np.maximum(v**2 - np.einsum("nk,kl,nl->n", b, f, b), 0.25 * v**2)
        styles_only = FactorRiskModel(b, f, d)
        tilts = {
            "random": rng.standard_normal(use.sum()),
            "beta": beta[i][use],
            "industry": in_first_industry[use].astype(float),
        }
        for kind, w in tilts.items():
            w = w - w.mean()
            w = w / np.abs(w).sum()
            realized = float(w @ forward[i][use])
            for label, model in (("structural", structural), ("styles_only", styles_only)):
                predicted = np.sqrt(w @ model.covariance() @ w)
                ratios.setdefault(f"{label}:{kind}", []).append(realized / predicted)
    bias = {key: float(np.std(values)) for key, values in ratios.items()}

    for kind in ("random", "beta", "industry"):
        assert 0.6 < bias[f"structural:{kind}"] < 1.2
    # A book tilted on beta or on one industry carries risk the style factors cannot see.
    assert bias["styles_only:beta"] > 1.25
    assert bias["styles_only:industry"] > 1.25


# --- through the optimizer ---------------------------------------------------------------


def test_both_risk_models_run_through_the_optimizer_and_are_separate_trials(synthetic, tmp_path):
    panel = synthetic.panel
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    combination = combine(panel, "icir", ledger=ledger)
    beta = trailing_beta(panel)

    def run(risk_model: str) -> tuple[float, float]:
        result = backtest_strategy(panel, combination, ledger=ledger, risk_model=risk_model)
        assert (result.levels == 0).all() and len(result.targets) > 80
        exposure = [
            abs(float((target * beta.loc[date, target.index].fillna(1.0)).sum()))
            for date, target in result.targets.items()
        ]
        active = result.run.active_returns
        return float(np.mean(exposure)), float(active.mean() / active.std())

    structural_beta, structural_sharpe = run("structural")
    styles_beta, styles_sharpe = run("alpha_factors")
    # Both books carry little net beta, and the structural model does not add any. With
    # correctly scaled factor variances, beta risk is small next to the expected return
    # at the default risk aversion, so the two books differ less than the risk forecasts.
    assert structural_beta < 0.1 and styles_beta < 0.1
    assert structural_beta < 1.1 * styles_beta
    assert structural_sharpe > 0.3 and styles_sharpe > 0.3
    assert ledger.n_trials == 3  # one combination, two portfolios
    with pytest.raises(ValueError, match="risk_model"):
        backtest_strategy(panel, combination, ledger=ledger, risk_model="barra")
    assert ledger.n_trials == 3
