import numpy as np
import pandas as pd
import pytest

from factorpipe.pbo import probability_of_backtest_overfitting as pbo


def noise(n_periods: int, n_configs: int, seed: int) -> pd.DataFrame:
    return pd.DataFrame(np.random.default_rng(seed).normal(0, 0.04, size=(n_periods, n_configs)))


def test_number_of_splits_is_the_number_of_ways_to_choose_half_the_blocks():
    result = pbo(noise(240, 10, seed=0), n_blocks=8)
    assert result.n_splits == 70  # 8 choose 4
    assert result.n_configurations == 10
    assert len(result.logits) == len(result.in_sample) == len(result.out_of_sample) == 70


def test_configurations_without_skill_give_a_pbo_near_one_half():
    values = [pbo(noise(240, 20, seed=s), n_blocks=10).pbo for s in range(30)]
    assert np.mean(values) == pytest.approx(0.5, abs=0.08)


def test_one_configuration_with_real_skill_gives_a_pbo_near_zero():
    returns = noise(240, 20, seed=1)
    returns[7] += 0.02  # a per-period Sharpe ratio of about 0.5
    result = pbo(returns, n_blocks=10)
    assert result.pbo < 0.02
    assert (result.out_of_sample > 0).mean() > 0.98


def test_the_in_sample_winner_looks_better_in_sample_than_out_of_sample():
    result = pbo(noise(240, 50, seed=2), n_blocks=10)
    assert result.in_sample.mean() > 0.1
    assert abs(result.out_of_sample.mean()) < 0.1
    assert result.in_sample.mean() > result.out_of_sample.mean() + 0.1


def test_hand_worked_two_blocks():
    # Two blocks, two configurations. A wins block 1 and loses block 2; B is the reverse.
    # Whichever block is in sample, its winner is the worse one out of sample: rank 1 of 2,
    # omega = 1/3, logit = ln(1/2) < 0. So PBO = 1.
    returns = pd.DataFrame(
        {
            "a": [0.03, 0.05, 0.04, -0.03, -0.05, -0.04],
            "b": [-0.03, -0.05, -0.04, 0.03, 0.05, 0.04],
        }
    )
    result = pbo(returns, n_blocks=2)
    assert result.n_splits == 2
    assert result.pbo == 1.0
    np.testing.assert_allclose(result.logits, np.log(0.5))


def test_the_oldest_rows_are_dropped_to_fit_the_blocks():
    returns = noise(245, 5, seed=3)
    pd.testing.assert_series_equal(
        pd.Series(pbo(returns, n_blocks=8).logits),
        pd.Series(pbo(returns.iloc[5:], n_blocks=8).logits),
    )


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="even"):
        pbo(noise(100, 5, seed=0), n_blocks=5)
    with pytest.raises(ValueError, match="two configurations"):
        pbo(noise(100, 1, seed=0))
    with pytest.raises(ValueError, match="not enough periods"):
        pbo(noise(20, 5, seed=0), n_blocks=16)
