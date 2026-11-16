"""Cross-check IC and quantile returns against alphalens-reloaded.

alphalens is a development dependency only. It builds forward returns from prices and
bins by factor value, so agreement here is a check by independent code.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from factorpipe.evaluation import information_coefficient, quantile_returns
from factorpipe.synthetic import SyntheticConfig, make_panel

PERIODS = (1, 3)


@pytest.fixture(scope="module")
def both():
    cfg = SyntheticConfig(
        n_assets=100,
        n_months=60,
        n_noise=1,
        reported_factor=None,
        delist_prob=0.0,
        late_listing_share=0.0,
        seed=1,
    )
    panel = make_panel(cfg).panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import alphalens

        prices = (1.0 + panel.returns).cumprod()
        factor = panel.characteristics["planted_strong"].stack()
        factor.index.names = ["date", "asset"]
        data = alphalens.utils.get_clean_factor_and_forward_returns(
            factor, prices, quantiles=5, periods=PERIODS, max_loss=1.0
        )
        ic = alphalens.performance.factor_information_coefficient(data)
        by_quantile, _ = alphalens.performance.mean_return_by_quantile(
            data, by_date=True, demeaned=False
        )
    return panel, ic, by_quantile


def test_rank_ic_agrees_with_alphalens(both):
    panel, theirs, _ = both
    assert len(theirs) >= 50
    for column, horizon in zip(theirs.columns, PERIODS, strict=True):
        ours = information_coefficient(panel, "planted_strong", horizon).reindex(theirs.index)
        np.testing.assert_allclose(ours.to_numpy(), theirs[column].to_numpy(), atol=1e-12)


def test_quantile_returns_agree_with_alphalens(both):
    panel, _, theirs = both
    for column, horizon in zip(theirs.columns, PERIODS, strict=True):
        expected = theirs[column].unstack("factor_quantile")
        ours = quantile_returns(panel, "planted_strong", 5, horizon).reindex(expected.index)
        np.testing.assert_allclose(ours.to_numpy(), expected.to_numpy(), atol=1e-12)
        assert isinstance(expected.index, pd.DatetimeIndex)
