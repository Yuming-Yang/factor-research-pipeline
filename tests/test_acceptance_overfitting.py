"""Deflated Sharpe ratio, PBO and the ledger, working together on the synthetic panel."""

import numpy as np
import pytest

from factorpipe.combine import combine
from factorpipe.experiments import select_best_factor
from factorpipe.ledger import TrialLedger, fingerprint
from factorpipe.portfolio import backtest_strategy
from factorpipe.sharpe import deflated_sharpe_from_ledger, sharpe_ratio
from factorpipe.synthetic import PlantedFactor, SyntheticConfig, make_panel

SHARED = {"n_assets": 150, "n_months": 144, "reported_factor": None, "seed": 5}
NOISE = SyntheticConfig(planted=(), n_noise=24, **SHARED)
SIGNAL = SyntheticConfig(
    planted=(PlantedFactor("planted_strong", ic=0.06, persistence=0.9),), n_noise=23, **SHARED
)


@pytest.fixture
def ledger(tmp_path):
    return TrialLedger(tmp_path / "trials.jsonl")


def test_the_best_of_many_noise_factors_does_not_survive(ledger):
    result = select_best_factor(make_panel(NOISE).panel, ledger)
    assert result.deflated.n_trials == 24
    assert result.best_annual_sharpe > 0.3  # picking the best of 24 always finds something
    assert result.deflated.psr > 0.85  # and on its own it looks convincing
    assert result.deflated.dsr < 0.75  # but not after counting the search
    assert 0.25 < result.pbo < 0.85  # the in-sample winner is a coin flip out of sample
    assert result.deflated.hurdle > 0.5 * result.deflated.sharpe


def test_a_real_factor_survives_the_same_search(ledger):
    result = select_best_factor(make_panel(SIGNAL).panel, ledger)
    assert result.best == "planted_strong"
    assert result.deflated.n_trials == 24
    assert result.deflated.dsr > 0.95
    assert result.pbo < 0.05
    assert result.interval.lower > 0


def test_the_deflation_reads_its_trial_count_from_the_ledger(ledger):
    panel = make_panel(SIGNAL).panel
    data = fingerprint(panel)
    names = list(panel.characteristics)

    def run(name: str):
        combination = combine(panel, "equal", ledger=ledger, factors=[name])
        return backtest_strategy(panel, combination, ledger=ledger, portfolio="quantile")

    chosen = run("planted_strong").run.active_returns
    alone = deflated_sharpe_from_ledger(chosen, ledger, data)
    assert alone.n_trials == 1 and alone.hurdle == 0.0 and alone.dsr == pytest.approx(alone.psr)

    for name in names[1:6]:
        run(name)
    after_five = deflated_sharpe_from_ledger(chosen, ledger, data)
    for name in names[6:]:
        run(name)
    after_all = deflated_sharpe_from_ledger(chosen, ledger, data)

    assert (after_five.n_trials, after_all.n_trials) == (6, 24)
    assert after_all.hurdle > 0.0 and after_five.hurdle > 0.0
    assert after_five.dsr <= alone.dsr
    # Re-running a backtest does not add a trial.
    run(names[3])
    assert deflated_sharpe_from_ledger(chosen, ledger, data).n_trials == 24


def test_the_ledger_stores_each_backtests_sharpe_ratio(ledger):
    panel = make_panel(SIGNAL).panel
    combination = combine(panel, "equal", ledger=ledger, factors=["planted_strong"])
    result = backtest_strategy(panel, combination, ledger=ledger, portfolio="quantile")
    stored = ledger.results()[result.trial_id]
    assert stored["sharpe"] == pytest.approx(sharpe_ratio(result.run.active_returns, 1))
    assert stored["n_obs"] == len(result.run.active_returns)
    assert combination.trial_id not in ledger.results()  # a combination alone has no Sharpe
    assert ledger.n_trials == 2
    with pytest.raises(KeyError, match="no trial"):
        ledger.record_result("missing", sharpe=1.0)


def test_other_data_and_trials_without_a_backtest_are_not_counted(ledger):
    panel = make_panel(SIGNAL).panel
    other = make_panel(NOISE).panel
    for p, name in ((panel, "planted_strong"), (other, "noise_00"), (other, "noise_01")):
        combination = combine(p, "equal", ledger=ledger, factors=[name])
        backtest_strategy(p, combination, ledger=ledger, portfolio="quantile")
    combine(panel, "icir", ledger=ledger)  # scored, never backtested
    returns = np.random.default_rng(0).normal(0.01, 0.03, size=100)
    assert deflated_sharpe_from_ledger(returns, ledger, fingerprint(panel)).n_trials == 1
    assert deflated_sharpe_from_ledger(returns, ledger, fingerprint(other)).n_trials == 2
    with pytest.raises(ValueError, match="no backtest results"):
        deflated_sharpe_from_ledger(returns, ledger, "unknown-data")
