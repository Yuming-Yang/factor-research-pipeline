import numpy as np
import pandas as pd
import pytest

from factorpipe.pit import lag_to_availability
from factorpipe.synthetic import PlantedFactor, SyntheticConfig, make_panel

NO_DELISTING = SyntheticConfig(delist_prob=0.0, late_listing_share=0.0, seed=1)


def mean_ic(characteristic: pd.DataFrame, returns: pd.DataFrame, horizon: int = 1) -> float:
    """Average cross-sectional correlation with the single-month return `horizon` ahead."""
    ahead = returns.shift(-horizon)
    return float(characteristic.corrwith(ahead, axis=1).mean())


def mean_rank_ic(characteristic: pd.DataFrame, returns: pd.DataFrame) -> float:
    # Spearman is the Pearson correlation of ranks. Both sides are ranked.
    ranked_returns = returns.shift(-1).rank(axis=1)
    return float(characteristic.rank(axis=1).corrwith(ranked_returns, axis=1).mean())


def tolerance(cfg: SyntheticConfig, n_sigma: float = 4.0) -> float:
    # One month's cross-sectional correlation has a standard error near 1/sqrt(N).
    return n_sigma / np.sqrt(cfg.n_assets * cfg.n_months)


def test_the_same_config_gives_the_same_panel():
    a, b = make_panel(SyntheticConfig(seed=7)), make_panel(SyntheticConfig(seed=7))
    pd.testing.assert_frame_equal(a.panel.returns, b.panel.returns)
    pd.testing.assert_frame_equal(a.fundamentals, b.fundamentals)
    c = make_panel(SyntheticConfig(seed=8))
    assert not a.panel.returns.equals(c.panel.returns)


def test_planted_factors_have_their_stated_ic():
    synthetic = make_panel(NO_DELISTING)
    panel, truth = synthetic.panel, synthetic.truth
    assert truth.planted == ["planted_strong", "planted_weak"]
    for name in truth.planted:
        observed = mean_ic(panel.characteristics[name], panel.returns)
        assert observed == pytest.approx(truth.ic_at(name), abs=tolerance(NO_DELISTING))


def test_noise_factors_have_no_ic():
    synthetic = make_panel(NO_DELISTING)
    panel, truth = synthetic.panel, synthetic.truth
    assert len(truth.noise) == NO_DELISTING.n_noise
    for name in truth.noise:
        observed = mean_ic(panel.characteristics[name], panel.returns)
        assert abs(observed) < tolerance(NO_DELISTING)


@pytest.mark.parametrize("horizon", [2, 6, 12])
def test_ic_decays_with_the_persistence_of_the_factor(horizon):
    synthetic = make_panel(NO_DELISTING)
    panel, truth = synthetic.panel, synthetic.truth
    for name in truth.planted:
        observed = mean_ic(panel.characteristics[name], panel.returns, horizon)
        assert observed == pytest.approx(truth.ic_at(name, horizon), abs=tolerance(NO_DELISTING))
    assert truth.ic_at("planted_strong", 12) == pytest.approx(0.05 * 0.95**11)


def test_rank_ic_matches_the_bivariate_normal_formula():
    cfg = SyntheticConfig(
        planted=(PlantedFactor("big", ic=0.30, persistence=0.5),),
        n_noise=0,
        reported_factor=None,
        delist_prob=0.0,
        late_listing_share=0.0,
        seed=2,
    )
    synthetic = make_panel(cfg)
    observed = mean_rank_ic(synthetic.panel.characteristics["big"], synthetic.panel.returns)
    expected = synthetic.truth.rank_ic_at("big")
    assert expected == pytest.approx(6 / np.pi * np.arcsin(0.15))
    assert expected < 0.30
    assert observed == pytest.approx(expected, abs=tolerance(cfg))


def test_a_characteristic_does_not_predict_its_own_month_beyond_persistence():
    # The return of month t is driven by z at t-1, so the same-month correlation is
    # ic * rho, not ic. A generator that leaked would show the full ic or more.
    synthetic = make_panel(NO_DELISTING)
    panel, truth = synthetic.panel, synthetic.truth
    observed = mean_ic(panel.characteristics["planted_weak"], panel.returns, horizon=0)
    expected = truth.ic["planted_weak"] * truth.persistence["planted_weak"]
    assert observed == pytest.approx(expected, abs=tolerance(NO_DELISTING))


def test_the_default_panel_has_listings_and_delistings_and_stays_close_to_the_truth():
    cfg = SyntheticConfig()
    synthetic = make_panel(cfg)
    panel = synthetic.panel
    listed = panel.listed
    assert (~listed.iloc[0]).sum() > 0  # some assets list late
    assert len(synthetic.delisting_returns) > 0
    assert set(synthetic.delisting_returns.unique()) == {
        cfg.delist_return_bad,
        cfg.delist_return_good,
    }
    for asset in synthetic.delisting_returns.index:
        assert not listed[asset].iloc[-1]
    observed = mean_ic(panel.characteristics["planted_strong"], panel.returns)
    assert observed == pytest.approx(0.05, abs=2 * tolerance(cfg))


def test_the_delisting_return_is_inside_the_last_monthly_return():
    cfg = SyntheticConfig(delist_return_bad=-0.9, delist_bad_share=1.0, delist_prob=0.01, seed=3)
    synthetic = make_panel(cfg)
    returns = synthetic.panel.returns
    last_returns = pd.Series(
        {a: returns[a].dropna().iloc[-1] for a in synthetic.delisting_returns.index}
    )
    # A -90% delisting return dominates any ordinary month.
    assert (last_returns < -0.75).all()
    assert returns.min().min() >= -1.0


def test_reported_values_follow_publication_dates():
    synthetic = make_panel(SyntheticConfig(seed=4))
    panel, reports = synthetic.panel, synthetic.fundamentals
    assert (reports["publish_date"] > reports["period_end"]).all()
    assert reports.duplicated(["asset", "period_end"]).any()  # restatements exist

    reported = panel.characteristics["planted_strong_reported"]
    rebuilt = lag_to_availability(reports, panel.dates, panel.assets).where(panel.listed)
    pd.testing.assert_frame_equal(reported, rebuilt)

    # Before the first publication nothing is known.
    first_publication = reports["publish_date"].min()
    assert reported.loc[reported.index < first_publication].isna().all().all()


def test_dating_reports_by_period_end_inflates_the_ic():
    cfg = SyntheticConfig(
        planted=(PlantedFactor("planted_strong", ic=0.08, persistence=0.7),),
        n_noise=0,
        delist_prob=0.0,
        late_listing_share=0.0,
        restatement_prob=0.0,
        seed=5,
    )
    synthetic = make_panel(cfg)
    panel, reports = synthetic.panel, synthetic.fundamentals
    honest = panel.characteristics["planted_strong_reported"]
    peeking = lag_to_availability(
        reports.assign(publish_date=reports["period_end"]), panel.dates, panel.assets
    )
    honest_ic = mean_ic(honest, panel.returns)
    peeking_ic = mean_ic(peeking, panel.returns)
    assert 0 < honest_ic < peeking_ic < 0.08
    assert peeking_ic - honest_ic > 2 * tolerance(cfg)


def test_invalid_configs_are_rejected():
    with pytest.raises(ValueError, match="sum to less than 1"):
        SyntheticConfig(planted=(PlantedFactor("a", 0.8), PlantedFactor("b", 0.7)))
    with pytest.raises(ValueError, match="unique"):
        SyntheticConfig(planted=(PlantedFactor("a", 0.1), PlantedFactor("a", 0.1)))
    with pytest.raises(ValueError, match="one of the planted"):
        SyntheticConfig(reported_factor="missing")
    with pytest.raises(ValueError, match="persistence"):
        SyntheticConfig(planted=(PlantedFactor("a", 0.1, persistence=1.0),), reported_factor=None)
