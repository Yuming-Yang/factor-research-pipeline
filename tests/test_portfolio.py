import numpy as np
import pandas as pd
import pytest

from factorpipe.combine import combine
from factorpipe.ledger import TrialLedger
from factorpipe.panel import Panel
from factorpipe.portfolio import (
    MIN_HISTORY,
    backtest_strategy,
    grinold_alpha,
    trailing_ic,
    trailing_volatility,
)
from factorpipe.synthetic import PlantedFactor, SyntheticConfig, make_panel

CONFIG = SyntheticConfig(
    n_assets=120,
    n_months=120,
    planted=(PlantedFactor("planted_strong", 0.10, 0.9), PlantedFactor("planted_weak", 0.05, 0.8)),
    n_noise=2,
    reported_factor=None,
    seed=11,
)


@pytest.fixture
def ledger(tmp_path):
    return TrialLedger(tmp_path / "trials.jsonl")


@pytest.fixture(scope="module")
def panel():
    return make_panel(CONFIG).panel


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
    )


def test_grinold_alpha_by_hand():
    alpha = grinold_alpha(np.array([1.0, 2.0, 3.0]), ic=0.1, volatility=np.array([0.2, 0.2, 0.4]))
    z = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2 / 3)
    np.testing.assert_allclose(alpha, 0.1 * np.array([0.2, 0.2, 0.4]) * z)
    np.testing.assert_array_equal(grinold_alpha(np.ones(3), 0.1, np.ones(3)), np.zeros(3))


def test_trailing_volatility_uses_only_the_past(panel):
    vol = trailing_volatility(panel)
    assert vol.iloc[: MIN_HISTORY - 1].isna().all().all()
    assert (
        vol.where(panel.listed)
        .iloc[MIN_HISTORY:]
        .notna()
        .to_numpy()[panel.listed.iloc[MIN_HISTORY:].to_numpy()]
        .all()
    )
    assert vol.iloc[60].median() == pytest.approx(0.11, abs=0.02)
    changed = trailing_volatility(perturb_after(panel, 80))
    pd.testing.assert_frame_equal(vol.iloc[:81], changed.iloc[:81])


@pytest.mark.parametrize("horizon,embargo", [(1, 0), (3, 2)])
def test_trailing_ic_waits_for_the_label_and_the_embargo(panel, horizon, embargo):
    scores = panel.characteristics["planted_strong"].where(panel.investable)
    ic = trailing_ic(panel, scores, horizon=horizon, embargo=embargo)
    first = int(np.flatnonzero(ic.notna().to_numpy())[0])
    assert first == MIN_HISTORY - 1 + horizon + embargo
    assert ic.iloc[-1] > 0.05


def test_mvo_targets_respect_the_constraints(panel, ledger):
    combination = combine(panel, "icir", ledger=ledger)
    result = backtest_strategy(panel, combination, ledger=ledger, max_position=0.03, max_gross=1.5)
    assert len(result.targets) > 50
    assert (result.levels == 0).all()  # clean data: the full problem solves every time
    for target in result.targets.values():
        assert abs(target.sum()) < 1e-6
        assert target.abs().max() <= 0.03 + 1e-6
        assert target.abs().sum() <= 1.5 + 1e-6
    # The book is actually used: leverage is near its cap at some point.
    assert max(t.abs().sum() for t in result.targets.values()) > 1.0


def test_both_portfolios_earn_the_planted_signal(panel, ledger):
    combination = combine(panel, "icir", ledger=ledger)
    for portfolio in ("quantile", "mvo"):
        result = backtest_strategy(panel, combination, ledger=ledger, portfolio=portfolio)
        traded = result.run.returns[result.run.turnover.cumsum() > 0]
        assert len(traded) > 50
        t_stat = traded.mean() / traded.std() * np.sqrt(len(traded))
        assert t_stat > 3


@pytest.mark.parametrize("portfolio", ["quantile", "mvo"])
def test_positions_do_not_depend_on_later_data(panel, ledger, portfolio):
    cutoff = 95
    original = backtest_strategy(
        panel, combine(panel, "icir", ledger=ledger), ledger=ledger, portfolio=portfolio
    )
    scrambled = perturb_after(panel, cutoff)
    changed = backtest_strategy(
        scrambled, combine(scrambled, "icir", ledger=ledger), ledger=ledger, portfolio=portfolio
    )
    assert original.run.weights.iloc[: cutoff + 1].abs().sum().sum() > 0
    pd.testing.assert_series_equal(
        original.run.nav.iloc[: cutoff + 1], changed.run.nav.iloc[: cutoff + 1], rtol=1e-9
    )
    pd.testing.assert_frame_equal(
        original.run.weights.iloc[: cutoff + 1], changed.run.weights.iloc[: cutoff + 1], atol=1e-9
    )
    assert not np.allclose(original.run.nav.iloc[cutoff + 2 :], changed.run.nav.iloc[cutoff + 2 :])


def test_costs_and_a_turnover_penalty_reduce_trading(panel, ledger):
    combination = combine(panel, "icir", ledger=ledger)

    def turnover(**kwargs) -> float:
        run = backtest_strategy(panel, combination, ledger=ledger, **kwargs).run
        return float(run.turnover[run.turnover > 0].iloc[1:].mean())

    free = turnover()
    costly = turnover(cost_rate=0.003)
    penalized = turnover(cost_rate=0.003, turnover_penalty=0.003)
    assert free > 1.5 * costly > 1.5 * penalized > 0


def test_costs_lower_the_result_of_the_same_positions(panel, ledger):
    combination = combine(panel, "icir", ledger=ledger)
    free = backtest_strategy(panel, combination, ledger=ledger, portfolio="quantile")
    costly = backtest_strategy(
        panel, combination, ledger=ledger, portfolio="quantile", cost_rate=0.002, borrow_rate=0.001
    )
    assert costly.run.nav.iloc[-1] < free.run.nav.iloc[-1]
    assert costly.run.costs.sum() > 0 and free.run.costs.sum() == 0


def test_portfolio_settings_are_trials(panel, ledger):
    combination = combine(panel, "equal", ledger=ledger)
    backtest_strategy(panel, combination, ledger=ledger, portfolio="quantile")
    backtest_strategy(panel, combination, ledger=ledger, portfolio="quantile")
    assert ledger.n_trials == 2  # one combination, one portfolio
    backtest_strategy(panel, combination, ledger=ledger, portfolio="quantile", cost_rate=0.001)
    backtest_strategy(panel, combination, ledger=ledger, portfolio="quantile", n_quantiles=10)
    assert ledger.n_trials == 4
    methods = [t["method"] for t in ledger.trials()]
    assert methods.count("portfolio:quantile") == 3
    assert all(
        t["params"]["combination"] == combination.trial_id
        for t in ledger.trials()
        if t["method"].startswith("portfolio")
    )


def test_bad_portfolio_calls_are_rejected_and_not_recorded(panel, ledger):
    combination = combine(panel, "equal", ledger=ledger)
    with pytest.raises(ValueError, match="unknown portfolio"):
        backtest_strategy(panel, combination, ledger=ledger, portfolio="risk_parity")
    with pytest.raises(ValueError, match="unknown parameters"):
        backtest_strategy(panel, combination, ledger=ledger, portfolio="quantile", gamma=3.0)
    with pytest.raises(TypeError, match="ledger"):
        backtest_strategy(panel, combination)
    assert ledger.n_trials == 1
