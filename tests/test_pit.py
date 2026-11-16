import numpy as np
import pandas as pd
import pytest

from factorpipe.pit import lag_to_availability

DATES = pd.date_range("2020-01-31", periods=12, freq="ME")
ASSETS = pd.Index(["a", "b"])


def reports(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=["asset", "period_end", "publish_date", "value"])
    table["period_end"] = pd.to_datetime(table["period_end"])
    table["publish_date"] = pd.to_datetime(table["publish_date"])
    return table


def test_a_value_appears_on_its_publication_date_not_its_period_end():
    table = reports([("a", "2020-03-31", "2020-05-15", 1.0)])
    out = lag_to_availability(table, DATES, ASSETS)["a"]
    assert out.loc[:"2020-04-30"].isna().all()
    assert (out.loc["2020-05-31":] == 1.0).all()


def test_a_value_published_on_the_date_itself_is_visible():
    table = reports([("a", "2020-03-31", "2020-04-30", 1.0)])
    out = lag_to_availability(table, DATES, ASSETS)["a"]
    assert np.isnan(out.loc["2020-03-31"])
    assert out.loc["2020-04-30"] == 1.0


def test_a_newer_period_replaces_the_older_one():
    table = reports(
        [("a", "2020-03-31", "2020-05-15", 1.0), ("a", "2020-06-30", "2020-08-10", 2.0)]
    )
    out = lag_to_availability(table, DATES, ASSETS)["a"]
    assert out.loc["2020-07-31"] == 1.0
    assert out.loc["2020-08-31"] == 2.0


def test_a_restatement_is_visible_only_from_its_own_publication_date():
    table = reports(
        [("a", "2020-03-31", "2020-05-15", 1.0), ("a", "2020-03-31", "2020-07-20", 1.5)]
    )
    out = lag_to_availability(table, DATES, ASSETS)["a"]
    assert out.loc["2020-05-31"] == 1.0
    assert out.loc["2020-06-30"] == 1.0
    assert out.loc["2020-07-31"] == 1.5


def test_a_late_restatement_of_an_old_period_does_not_hide_a_newer_period():
    table = reports(
        [
            ("a", "2020-03-31", "2020-05-15", 1.0),
            ("a", "2020-06-30", "2020-08-10", 2.0),
            ("a", "2020-03-31", "2020-09-05", 9.0),
        ]
    )
    out = lag_to_availability(table, DATES, ASSETS)["a"]
    assert out.loc["2020-09-30"] == 2.0


def test_values_expire_after_max_age():
    table = reports([("a", "2020-03-31", "2020-05-15", 1.0)])
    out = lag_to_availability(table, DATES, ASSETS, max_age_months=6)["a"]
    assert out.loc["2020-09-30"] == 1.0
    assert np.isnan(out.loc["2020-10-31"])


def test_assets_are_kept_apart_and_unknown_assets_are_ignored():
    table = reports(
        [
            ("a", "2020-03-31", "2020-05-15", 1.0),
            ("b", "2020-03-31", "2020-04-10", 7.0),
            ("zzz", "2020-03-31", "2020-04-10", 5.0),
        ]
    )
    out = lag_to_availability(table, DATES, ASSETS)
    assert out.loc["2020-04-30", "b"] == 7.0
    assert np.isnan(out.loc["2020-04-30", "a"])
    assert list(out.columns) == ["a", "b"]


def test_row_order_of_the_input_does_not_matter():
    table = reports(
        [
            ("a", "2020-03-31", "2020-05-15", 1.0),
            ("a", "2020-06-30", "2020-08-10", 2.0),
            ("a", "2020-03-31", "2020-07-20", 1.5),
            ("b", "2020-03-31", "2020-04-10", 7.0),
        ]
    )
    shuffled = table.sample(frac=1.0, random_state=3)
    pd.testing.assert_frame_equal(
        lag_to_availability(table, DATES, ASSETS), lag_to_availability(shuffled, DATES, ASSETS)
    )


def test_reports_published_later_never_change_earlier_dates():
    rng = np.random.default_rng(0)
    rows = []
    for asset in ASSETS:
        for period in pd.date_range("2019-12-31", periods=6, freq="QE"):
            publish = period + pd.Timedelta(days=int(rng.integers(20, 120)))
            rows.append((asset, period, publish, float(rng.normal())))
            if rng.random() < 0.4:
                later = publish + pd.Timedelta(days=int(rng.integers(30, 200)))
                rows.append((asset, period, later, float(rng.normal())))
    table = pd.DataFrame(rows, columns=["asset", "period_end", "publish_date", "value"])
    full = lag_to_availability(table, DATES, ASSETS)
    for cutoff in DATES:
        truncated = lag_to_availability(table[table["publish_date"] <= cutoff], DATES, ASSETS)
        pd.testing.assert_frame_equal(full.loc[:cutoff], truncated.loc[:cutoff])


def test_publication_before_period_end_is_rejected():
    table = reports([("a", "2020-03-31", "2020-03-01", 1.0)])
    with pytest.raises(ValueError, match="before its period ends"):
        lag_to_availability(table, DATES, ASSETS)


def test_missing_columns_are_reported():
    with pytest.raises(ValueError, match="missing columns"):
        lag_to_availability(pd.DataFrame({"asset": ["a"]}), DATES, ASSETS)
