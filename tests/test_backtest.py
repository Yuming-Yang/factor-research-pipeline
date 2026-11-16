import numpy as np
import pandas as pd
import pytest

from factorpipe.backtest import Market, market_from_panel, run_backtest
from factorpipe.evaluation import long_short, quantile_returns
from factorpipe.panel import Panel
from factorpipe.portfolio import QuantilePortfolio
from factorpipe.synthetic import SyntheticConfig, make_panel

DATES = pd.date_range("2020-01-31", periods=5, freq="ME")


def panel_of(columns: dict[str, list[float]]) -> Panel:
    return Panel(pd.DataFrame(columns, index=DATES))


def hold(weights: dict[str, float]):
    return lambda t, snapshot: pd.Series(weights)


def test_prices_reproduce_the_panel_returns():
    panel = panel_of(
        {"a": [0.10, -0.20, 0.05, 0.0, 0.30], "b": [np.nan, 0.02, 0.03, np.nan, np.nan]}
    )
    market = market_from_panel(panel)
    assert market.close["a"].pct_change().iloc[1:].to_numpy() == pytest.approx(
        [-0.20, 0.05, 0.0, 0.30]
    )
    # The open of a month is the previous month-end price.
    assert market.open["a"].iloc[1:].to_numpy() == pytest.approx(market.close["a"].iloc[:-1])
    # No open in the first listed month: the asset cannot be bought before it has a price.
    assert np.isnan(market.open["a"].iloc[0]) and np.isnan(market.open["b"].iloc[1])
    assert market.last_bar.to_dict() == {"b": DATES[2]}
    assert np.isnan(market.close["b"].iloc[3])


def test_holding_one_asset_earns_its_returns_from_the_next_month():
    panel = panel_of({"a": [0.10, -0.20, 0.05, 0.0, 0.30], "b": [0.0] * 5})
    run = run_backtest(market_from_panel(panel), hold({"a": 1.0}))
    # Decided at the end of month 0, so month 0's own return is not earned.
    assert run.returns.to_numpy() == pytest.approx([0.0, -0.20, 0.05, 0.0, 0.30])
    assert run.nav.iloc[-1] == pytest.approx(0.8 * 1.05 * 1.3)
    assert run.turnover.iloc[1] == pytest.approx(1.0)
    assert run.turnover.iloc[2:].sum() == pytest.approx(0.0)


def test_a_delisting_turns_the_position_into_cash():
    # "a" delists in month 2; its return there already includes the delisting return.
    panel = panel_of({"a": [0.0, 0.10, -0.40, np.nan, np.nan], "b": [0.0] * 5})
    run = run_backtest(market_from_panel(panel), hold({"a": 0.5}))
    # Rebalanced to 50% at each month-end: +5% in month 1, then -20% in month 2.
    assert run.nav.iloc[2] == pytest.approx(1.05 * 0.80)
    assert run.weights["a"].iloc[2] == 0.0
    assert run.nav.iloc[2:].nunique() == 1  # the later orders in "a" are cancelled


def test_an_order_in_an_asset_without_a_price_is_cancelled():
    panel = panel_of({"a": [np.nan, np.nan, 0.05, 0.05, 0.05], "b": [0.0] * 5})
    run = run_backtest(market_from_panel(panel), hold({"a": 1.0}))
    first_fill = pd.to_datetime(run.trades["date"]).min()
    assert first_fill == DATES[3]  # listed in month 2, first month-end price then
    assert run.returns.to_numpy() == pytest.approx([0.0, 0.0, 0.0, 0.05, 0.05])


def test_costs_come_out_of_cash():
    panel = panel_of({"a": [0.0] * 5, "b": [0.0] * 5})
    flip = lambda t, snapshot: pd.Series({"a": 0.5 if snapshot.bar % 2 == 0 else 0.0})  # noqa: E731
    run = run_backtest(market_from_panel(panel, cost_rate=0.01), flip)
    assert run.nav.iloc[1] == pytest.approx(1.0 - 0.01 * 0.5)
    assert run.costs.iloc[1] == pytest.approx(0.005)
    assert run.nav.is_monotonic_decreasing


def test_no_decision_means_no_order_and_the_last_decision_never_fills():
    panel = panel_of({"a": [0.0, 0.10, 0.10, 0.10, 0.10], "b": [0.0] * 5})
    calls = []

    def decide(t, snapshot):
        calls.append(t)
        return pd.Series({"a": 1.0}) if snapshot.bar == 4 else None

    run = run_backtest(market_from_panel(panel), decide)
    assert len(calls) == 5
    assert run.trades.empty and (run.nav == 1.0).all()


def test_the_snapshot_shows_nothing_after_its_date():
    panel = panel_of({"a": [0.01] * 5, "b": [0.0] * 5})
    seen = {}

    def decide(t, snapshot):
        seen[t] = (snapshot.close.index.max(), snapshot.open.index.max(), len(snapshot.close))
        return None

    run_backtest(market_from_panel(panel), decide)
    for i, t in enumerate(DATES):
        assert seen[t] == (t, t, i + 1)


def test_bad_targets_are_rejected():
    market = market_from_panel(panel_of({"a": [0.0] * 5, "b": [0.0] * 5}))
    with pytest.raises(ValueError, match="unknown assets"):
        run_backtest(market, hold({"zzz": 1.0}))
    with pytest.raises(ValueError, match="finite"):
        run_backtest(market, hold({"a": np.inf}))


def test_invalid_markets_are_rejected():
    prices = pd.DataFrame({"a": [1.0, 1.1]}, index=DATES[:2])
    with pytest.raises(ValueError, match="fill_at"):
        Market(prices, prices, fill_at="vwap")
    with pytest.raises(ValueError, match="positive"):
        Market(prices, prices * 0.0)
    with pytest.raises(ValueError, match="share dates"):
        Market(prices, prices.iloc[:1])


def test_a_total_loss_does_not_break_the_price_index():
    panel = panel_of({"a": [0.0, -1.0, np.nan, np.nan, np.nan], "b": [0.0] * 5})
    run = run_backtest(market_from_panel(panel), hold({"a": 0.5}))
    assert np.isfinite(run.nav).all()
    assert run.nav.iloc[1] == pytest.approx(0.5, abs=1e-6)


def test_the_engine_reproduces_the_long_short_spread_from_the_evaluation_module():
    """Two code paths, one answer: quantile spreads computed directly from forward
    returns, and the same portfolio traded through the engine with delistings and
    late listings."""
    synthetic = make_panel(SyntheticConfig(n_assets=200, n_months=90, n_noise=1, seed=3))
    panel = synthetic.panel
    assert len(synthetic.delisting_returns) > 20
    scores = panel.characteristics["planted_strong"].where(panel.investable)
    run = run_backtest(market_from_panel(panel), QuantilePortfolio(scores))
    spread = long_short(quantile_returns(panel, "planted_strong"))
    traded = run.returns.shift(-1)  # the return earned on the portfolio formed at t
    both = pd.concat([spread, traded], axis=1).dropna()
    assert len(both) == 89
    np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-12)
