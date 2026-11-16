import numpy as np
import pandas as pd
import pytest

from factorpipe.panel import Panel, PanelError

DATES = pd.date_range("2020-01-31", periods=5, freq="ME")


def frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=DATES)


def test_forward_return_compounds_over_the_horizon():
    panel = Panel(frame({"a": [0.0, 0.10, 0.20, -0.50, 0.0]}))
    assert panel.forward_returns(1)["a"].iloc[0] == pytest.approx(0.10)
    assert panel.forward_returns(2)["a"].iloc[0] == pytest.approx(1.10 * 1.20 - 1)
    assert panel.forward_returns(3)["a"].iloc[0] == pytest.approx(1.10 * 1.20 * 0.50 - 1)


def test_forward_return_never_uses_the_current_month():
    base = Panel(frame({"a": [0.0, 0.10, 0.20, 0.0, 0.0]}))
    bumped = Panel(frame({"a": [0.9, 0.10, 0.20, 0.0, 0.0]}))
    assert base.forward_returns(2)["a"].iloc[0] == bumped.forward_returns(2)["a"].iloc[0]


def test_partial_windows_at_the_end_are_missing():
    panel = Panel(frame({"a": [0.01] * 5}))
    fwd = panel.forward_returns(2)["a"]
    assert fwd.iloc[:3].notna().all()
    assert fwd.iloc[3:].isna().all()


def test_delisting_inside_the_window_leaves_cash():
    # Listed for months 0..2; the return in month 2 already includes the delisting return.
    panel = Panel(frame({"a": [0.0, 0.10, -0.30, np.nan, np.nan], "b": [0.0] * 5}))
    fwd = panel.forward_returns(3)["a"]
    assert fwd.iloc[0] == pytest.approx(1.10 * 0.70 - 1)


def test_an_asset_in_its_last_month_is_not_investable():
    panel = Panel(frame({"a": [0.0, 0.10, -0.30, np.nan, np.nan], "b": [0.0] * 5}))
    assert list(panel.investable["a"]) == [True, True, False, False, False]
    assert np.isnan(panel.forward_returns(1)["a"].iloc[2])
    # The last row has no later month, so nothing is investable there.
    assert not panel.investable.iloc[-1].any()


def test_an_asset_is_not_investable_before_it_lists():
    panel = Panel(frame({"a": [np.nan, np.nan, 0.05, 0.05, 0.05], "b": [0.0] * 5}))
    fwd = panel.forward_returns(1)["a"]
    assert fwd.iloc[:2].isna().all()
    assert fwd.iloc[2] == pytest.approx(0.05)


def test_a_gap_inside_a_listed_spell_is_rejected():
    with pytest.raises(PanelError, match="gap"):
        Panel(frame({"a": [0.01, np.nan, 0.01, 0.01, 0.01]}))


def test_a_return_below_minus_one_is_rejected():
    with pytest.raises(PanelError, match="-100%"):
        Panel(frame({"a": [0.0, -1.5, 0.0, 0.0, 0.0]}))


def test_unsorted_dates_are_rejected():
    with pytest.raises(PanelError, match="increasing"):
        Panel(frame({"a": [0.0] * 5}).iloc[::-1])


def test_a_characteristic_on_another_grid_is_rejected():
    returns = frame({"a": [0.0] * 5, "b": [0.0] * 5})
    with pytest.raises(PanelError, match="same grid"):
        Panel(returns, {"size": returns[["a"]]})


def test_horizon_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        Panel(frame({"a": [0.0] * 5})).forward_returns(0)
