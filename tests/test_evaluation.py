import numpy as np
import pandas as pd
import pytest

from factorpipe.evaluation import (
    decay_curve,
    evaluate_factor,
    information_coefficient,
    long_short,
    quantile_assignments,
    quantile_returns,
    quantile_turnover,
    rank_autocorrelation,
    screen_factors,
)
from factorpipe.panel import Panel
from factorpipe.synthetic import SyntheticConfig, make_panel

DATES = pd.date_range("2020-01-31", periods=4, freq="ME")
ASSETS = list("abcdef")


def small_panel(factor_rows, return_rows) -> Panel:
    returns = pd.DataFrame(return_rows, index=DATES, columns=ASSETS, dtype=float)
    factor = pd.DataFrame(factor_rows, index=DATES, columns=ASSETS, dtype=float)
    return Panel(returns, {"f": factor})


def test_ic_is_one_when_the_factor_orders_next_months_returns():
    next_month = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    panel = small_panel(
        [[1, 2, 3, 4, 5, 6]] * 4,
        [[0.0] * 6, next_month, next_month[::-1], [0.0] * 6],
    )
    ic = information_coefficient(panel, "f", min_assets=2)
    assert ic.iloc[0] == pytest.approx(1.0)
    assert ic.iloc[1] == pytest.approx(-1.0)
    # Month 3 returns are all equal, and the last row has no forward return.
    assert ic.iloc[2:].isna().all()


def test_spearman_ranks_both_sides():
    # Factor and return are monotone but not linear: Spearman is 1, Pearson is below 1.
    panel = small_panel(
        [[1, 2, 3, 4, 5, 100]] * 4,
        [[0.0] * 6, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], [0.0] * 6, [0.0] * 6],
    )
    assert information_coefficient(panel, "f", method="spearman", min_assets=2).iloc[
        0
    ] == pytest.approx(1.0)
    pearson = information_coefficient(panel, "f", method="pearson", min_assets=2).iloc[0]
    assert 0.5 < pearson < 0.99


def test_ic_ignores_the_current_months_return():
    rows = [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]] * 4
    base = small_panel([[1, 2, 3, 4, 5, 6]] * 4, rows)
    changed = [list(r) for r in rows]
    changed[0] = [0.9, -0.5, 0.3, 0.0, 0.1, -0.2]
    bumped = small_panel([[1, 2, 3, 4, 5, 6]] * 4, changed)
    assert information_coefficient(base, "f", min_assets=2).iloc[0] == pytest.approx(
        information_coefficient(bumped, "f", min_assets=2).iloc[0]
    )


def test_small_cross_sections_are_skipped():
    panel = small_panel([[1, 2, 3, 4, 5, 6]] * 4, [[0.01 * i for i in range(6)]] * 4)
    assert information_coefficient(panel, "f").isna().all()  # default needs 20 assets


def test_unknown_method_is_rejected():
    panel = small_panel([[1, 2, 3, 4, 5, 6]] * 4, [[0.0] * 6] * 4)
    with pytest.raises(ValueError, match="pearson"):
        information_coefficient(panel, "f", method="kendall")


def test_quantiles_split_the_cross_section_evenly():
    panel = small_panel([[6, 5, 4, 3, 2, 1]] * 4, [[0.0] * 6] * 4)
    q = quantile_assignments(panel, "f", n_quantiles=3, min_assets=2)
    assert list(q.iloc[0]) == [3, 3, 2, 2, 1, 1]
    assert q.iloc[-1].isna().all()  # nothing is investable on the last date


def test_quantile_returns_and_spread_by_hand():
    panel = small_panel(
        [[1, 2, 3, 4, 5, 6]] * 4,
        [[0.0] * 6, [-0.02, 0.00, 0.01, 0.03, 0.04, 0.06], [0.0] * 6, [0.0] * 6],
    )
    by_q = quantile_returns(panel, "f", n_quantiles=3, min_assets=2)
    assert list(by_q.iloc[0]) == pytest.approx([-0.01, 0.02, 0.05])
    assert long_short(by_q).iloc[0] == pytest.approx(0.06)


def test_assets_that_cannot_be_bought_are_left_out_of_quantiles():
    returns = [[0.0] * 6, [0.0] * 6, [0.0] * 5 + [np.nan], [0.0] * 5 + [np.nan]]
    panel = small_panel([[1, 2, 3, 4, 5, 6]] * 4, returns)
    q = quantile_assignments(panel, "f", n_quantiles=5, min_assets=2)
    assert q.iloc[0]["f"] == 5
    assert np.isnan(q.iloc[1]["f"])  # delists in month 2, so not investable at its end
    assert q.iloc[1].notna().sum() == 5


def test_turnover_counts_new_names_in_the_top_quantile():
    panel = small_panel(
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [6, 2, 3, 4, 5, 1], [1, 2, 3, 4, 5, 6]],
        [[0.0] * 6] * 4,
    )
    turnover = quantile_turnover(panel, "f", n_quantiles=3, min_assets=2)
    assert np.isnan(turnover.iloc[0])
    assert turnover.iloc[1] == pytest.approx(0.0)
    assert turnover.iloc[2] == pytest.approx(0.5)  # top pair was {e, f}, now {a, e}


def test_rank_autocorrelation_matches_the_persistence_of_the_factor():
    synthetic = make_panel(SyntheticConfig(delist_prob=0.0, late_listing_share=0.0, seed=1))
    for name in ("planted_strong", "planted_weak"):
        rho = synthetic.truth.persistence[name]
        expected = 6 / np.pi * np.arcsin(rho / 2)
        observed = rank_autocorrelation(synthetic.panel, name).mean()
        assert observed == pytest.approx(expected, abs=0.005)


def test_a_slow_factor_turns_over_less_than_a_fast_one():
    synthetic = make_panel(SyntheticConfig(delist_prob=0.0, late_listing_share=0.0, seed=1))
    slow = quantile_turnover(synthetic.panel, "planted_strong").mean()  # persistence 0.95
    fast = quantile_turnover(synthetic.panel, "planted_weak").mean()  # persistence 0.80
    assert 0.05 < slow < fast < 0.8


def test_decay_curve_follows_the_planted_decay():
    cfg = SyntheticConfig(delist_prob=0.0, late_listing_share=0.0, seed=1)
    synthetic = make_panel(cfg)
    curve = decay_curve(synthetic.panel, "planted_weak", max_lag=6, method="pearson")
    assert list(curve.index) == [1, 2, 3, 4, 5, 6]
    tolerance = 4 / np.sqrt(cfg.n_assets * cfg.n_months)
    for lag, value in curve.items():
        assert value == pytest.approx(synthetic.truth.ic_at("planted_weak", lag), abs=tolerance)


def test_evaluate_factor_reports_every_horizon_with_enough_lags():
    synthetic = make_panel(SyntheticConfig(seed=2))
    report = evaluate_factor(synthetic.panel, "planted_strong")
    assert list(report.index) == [1, 3, 6, 12]
    assert (report["nw_lags"] >= report.index - 1).all()
    assert (report["n_months"] == [239, 237, 234, 228]).all()
    assert (report["rank_ic"] > 0).all() and (report["spread"] > 0).all()
    # A longer window collects more of a persistent signal.
    assert report["ic"].is_monotonic_increasing


def test_screen_reports_one_row_per_factor_and_consistent_flags():
    synthetic = make_panel(SyntheticConfig(seed=2))
    table = screen_factors(synthetic.panel)
    assert set(table.index) == set(synthetic.panel.characteristics)
    assert (table["p_holm"] >= table["p_value"]).all()
    assert (table["p_by"] >= table["p_value"]).all()
    assert not (table["significant_holm"] & ~table["significant_raw"]).any()
    subset = screen_factors(synthetic.panel, names=["noise_00", "noise_01"])
    assert list(subset.index) == ["noise_00", "noise_01"]
