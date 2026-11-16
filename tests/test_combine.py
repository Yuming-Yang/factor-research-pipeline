import numpy as np
import pandas as pd
import pytest

from factorpipe.combine import METHODS, combine, factor_returns, standardize
from factorpipe.evaluation import information_coefficient
from factorpipe.ledger import TrialLedger
from factorpipe.panel import Panel
from factorpipe.synthetic import SyntheticConfig, make_panel

EXACT = {"delist_prob": 0.0, "late_listing_share": 0.0, "reported_factor": None}
BURN_IN = 60


@pytest.fixture
def ledger(tmp_path):
    return TrialLedger(tmp_path / "trials.jsonl")


@pytest.fixture(scope="module")
def big():
    return make_panel(SyntheticConfig(seed=1, **EXACT))


def composite_ic(panel: Panel, scores: pd.DataFrame) -> float:
    series = information_coefficient(
        panel.with_characteristic("composite", scores), "composite", method="pearson"
    )
    return float(series.iloc[BURN_IN:].mean())


# --- building blocks -------------------------------------------------------------------


def test_standardize_gives_unit_scores_over_investable_assets():
    panel = make_panel(SyntheticConfig(n_assets=200, n_months=48, n_noise=1, seed=3)).panel
    names = ["planted_strong", "noise_00"]
    z = standardize(panel, names)
    investable = panel.investable.to_numpy()
    assert z.shape == (48, 2, 200)
    assert np.isnan(z[:, 0, :][~investable]).all()
    assert not np.isnan(z[:, 0, :][investable]).any()
    row = z[10, 0, :][investable[10]]
    assert abs(row.mean()) < 0.02  # clipping moves the mean slightly
    assert row.std() == pytest.approx(1.0, abs=0.02)
    assert np.nanmax(np.abs(z)) <= 3.0


def test_a_missing_factor_value_becomes_the_cross_sectional_mean():
    synthetic = make_panel(SyntheticConfig(n_assets=50, n_months=30, n_noise=0, seed=3, **EXACT))
    factor = synthetic.panel.characteristics["planted_strong"].copy()
    factor.iloc[5, 7] = np.nan
    panel = synthetic.panel.with_characteristic("planted_strong", factor)
    assert standardize(panel, ["planted_strong"])[5, 0, 7] == 0.0


def test_factor_return_by_hand():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    returns = pd.DataFrame(
        [[0.0] * 4, [0.04, 0.02, -0.02, -0.04], [0.0] * 4], index=dates, columns=list("abcd")
    )
    factor = pd.DataFrame([[2.0, 1.0, -1.0, -2.0]] * 3, index=dates, columns=list("abcd"))
    panel = Panel(returns, {"f": factor})
    z = standardize(panel, ["f"])
    # z = f / sqrt(2.5); mean of z * r = (0.08 + 0.02 + 0.02 + 0.08) / 4 / sqrt(2.5).
    expected = 0.05 / np.sqrt(2.5)
    assert factor_returns(panel, z, ["f"]).iloc[0, 0] == pytest.approx(expected)


# --- no information from the future ----------------------------------------------------


def perturb_after(panel: Panel, cutoff: int, seed: int = 0) -> Panel:
    """Replace every value after ``cutoff`` with noise, keeping which cells are missing."""
    rng = np.random.default_rng(seed)

    def scramble(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
        values = frame.to_numpy(dtype=float).copy()
        noise = scale * rng.standard_normal(values[cutoff + 1 :].shape)
        values[cutoff + 1 :] = np.where(np.isnan(values[cutoff + 1 :]), np.nan, noise)
        return pd.DataFrame(values, index=frame.index, columns=frame.columns)

    return Panel(
        scramble(panel.returns, 0.2),
        {name: scramble(frame, 1.0) for name, frame in panel.characteristics.items()},
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("horizon,embargo", [(1, 0), (3, 1)])
def test_scores_do_not_depend_on_later_data(method, horizon, embargo, ledger):
    panel = make_panel(SyntheticConfig(n_assets=80, n_months=110, n_noise=3, seed=4)).panel
    cutoff = 85
    params = {"window": 24} if method in ("icir", "mvo") else {}
    original = combine(panel, method, ledger=ledger, horizon=horizon, embargo=embargo, **params)
    changed = combine(
        perturb_after(panel, cutoff),
        method,
        ledger=ledger,
        horizon=horizon,
        embargo=embargo,
        **params,
    )
    assert original.scores.iloc[: cutoff + 1].notna().any().any()
    pd.testing.assert_frame_equal(
        original.scores.iloc[: cutoff + 1], changed.scores.iloc[: cutoff + 1]
    )
    if original.weights is not None:
        pd.testing.assert_frame_equal(
            original.weights.iloc[: cutoff + 1], changed.weights.iloc[: cutoff + 1]
        )
    # The test has teeth only if later scores do change.
    assert not original.scores.iloc[cutoff + 1 :].equals(changed.scores.iloc[cutoff + 1 :])


def test_the_lookahead_check_catches_a_leaky_weighting():
    # Same ICIR idea, but using each month's IC in the month it is formed, before the
    # return it is measured against has happened.
    panel = make_panel(SyntheticConfig(n_assets=80, n_months=110, n_noise=3, seed=4)).panel
    cutoff = 85

    def leaky_weights(p: Panel) -> pd.DataFrame:
        ic = pd.DataFrame({n: information_coefficient(p, n) for n in p.characteristics})
        return ic.rolling(24, min_periods=24).mean()

    before = leaky_weights(panel).iloc[: cutoff + 1]
    after = leaky_weights(perturb_after(panel, cutoff)).iloc[: cutoff + 1]
    assert not before.equals(after)


@pytest.mark.parametrize("horizon,embargo", [(1, 0), (6, 0), (3, 2)])
def test_icir_waits_for_the_window_the_label_and_the_embargo(horizon, embargo, ledger):
    cfg = SyntheticConfig(n_assets=80, n_months=110, n_noise=1, reported_factor=None, seed=4)
    panel = make_panel(cfg).panel
    window = 24
    result = combine(panel, "icir", ledger=ledger, horizon=horizon, embargo=embargo, window=window)
    first_valid = int(np.flatnonzero(result.weights.notna().all(axis=1).to_numpy())[0])
    # The IC formed at s is known at s + horizon + embargo; the window needs `window` of them.
    assert first_valid == window - 1 + horizon + embargo


# --- against the planted truth ---------------------------------------------------------


def test_equal_weight_scores_from_the_first_date(big, ledger):
    result = combine(big.panel, "equal", ledger=ledger)
    assert result.scores.iloc[0].notna().any()
    assert np.allclose(result.weights.to_numpy(), 1.0 / result.weights.shape[1])
    expected = big.truth.equal_weight_ic(list(big.panel.characteristics))
    assert composite_ic(big.panel, result.scores) == pytest.approx(expected, abs=0.012)


@pytest.mark.parametrize("method", ["icir", "mvo", "ridge"])
def test_learned_weights_beat_equal_weight_and_do_not_beat_the_truth(method, big, ledger):
    equal = composite_ic(big.panel, combine(big.panel, "equal", ledger=ledger).scores)
    result = combine(big.panel, method, ledger=ledger)
    learned = composite_ic(big.panel, result.scores)
    tolerance = 4 / np.sqrt(500 * (240 - BURN_IN))
    assert learned > 2 * equal
    assert learned == pytest.approx(big.truth.best_ic, abs=tolerance)
    late = result.weights.iloc[-24:].mean()
    assert late.idxmax() == "planted_strong"
    assert late["planted_strong"] > late["planted_weak"] > late[big.truth.noise].abs().max()
    assert np.allclose(result.weights.dropna().abs().sum(axis=1), 1.0)


def test_momentum_and_trees_find_the_signal(big, ledger):
    equal = composite_ic(big.panel, combine(big.panel, "equal", ledger=ledger).scores)
    for method in ("momentum", "trees"):
        learned = composite_ic(big.panel, combine(big.panel, method, ledger=ledger).scores)
        assert 2 * equal < learned < big.truth.best_ic + 0.012


def test_a_flipped_factor_gets_a_negative_weight(big, ledger):
    flipped = big.panel.with_characteristic(
        "planted_strong", -big.panel.characteristics["planted_strong"]
    )
    for method in ("icir", "momentum", "mvo", "ridge"):
        weights = combine(flipped, method, ledger=ledger).weights
        assert weights["planted_strong"].iloc[-24:].mean() < -0.15


def test_scores_exist_only_for_investable_assets(ledger):
    panel = make_panel(SyntheticConfig(n_assets=120, n_months=90, n_noise=2, seed=6)).panel
    for method in ("equal", "ridge", "trees"):
        scores = combine(panel, method, ledger=ledger).scores
        assert scores.notna().any().any()
        assert np.isnan(scores.to_numpy()[~panel.investable.to_numpy()]).all()


def test_invalid_horizon_or_embargo_is_rejected(big, ledger):
    with pytest.raises(ValueError, match="horizon"):
        combine(big.panel, "equal", ledger=ledger, horizon=0)
    with pytest.raises(ValueError, match="embargo"):
        combine(big.panel, "equal", ledger=ledger, embargo=-1)
