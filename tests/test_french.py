import io
import zipfile

import pandas as pd
import pytest

from factorpipe.french import (
    BASE_URL,
    FIVE_FACTORS,
    MOMENTUM,
    load_french_factors,
    parse_french_csv,
)
from factorpipe.stats import hac_regression

FIVE = """This file was created using the 202609 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RMW,CMA,RF
196307,   -0.39,   -0.41,   -0.97,    0.68,   -1.18,    0.27
196308,    5.07,   -0.80,    1.80,    0.36,   -0.35,    0.25
196309,   -1.57,   -0.52,    0.13,   -0.71,    0.29,    0.27

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
  1964,   12.54,   -0.39,    9.79,   -1.20,    6.62,    3.54
"""

MOM_HEADER = ",Mom" + "   "  # the real file pads the column name
MOM = f"""This file was created using the 202609 CRSP database.
Missing data are indicated by -99.99 or -999.

{MOM_HEADER}
196307,    0.90
196308,  -99.99
196309,    0.19

Annual Factors:
{MOM_HEADER}
  1964,    0.13
"""


def zipped(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def test_the_monthly_section_is_read_as_decimals_and_the_annual_one_is_ignored():
    frame = parse_french_csv(FIVE)
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert list(frame.index) == list(pd.to_datetime(["1963-07-31", "1963-08-31", "1963-09-30"]))
    assert frame.loc["1963-08-31", "Mkt-RF"] == pytest.approx(0.0507)
    assert frame.loc["1963-07-31", "RF"] == pytest.approx(0.0027)


def test_padded_names_and_missing_codes():
    frame = parse_french_csv(MOM)
    assert list(frame.columns) == ["Mom"]
    assert frame["Mom"].iloc[0] == pytest.approx(0.009)
    assert pd.isna(frame["Mom"].iloc[1])


def test_files_without_the_expected_layout_are_rejected():
    with pytest.raises(ValueError, match="no header"):
        parse_french_csv("nothing here\n1,2,3\n")
    with pytest.raises(ValueError, match="no monthly rows"):
        parse_french_csv(",A,B\n\n")


def test_loading_downloads_once_and_then_reads_the_cache(tmp_path):
    payloads = {
        BASE_URL + FIVE_FACTORS: zipped("F-F_Research_Data_5_Factors_2x3.csv", FIVE),
        BASE_URL + MOMENTUM: zipped("F-F_Momentum_Factor.csv", MOM),
    }
    calls = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return payloads[url]

    frame = load_french_factors(tmp_path, fetch)
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    assert len(frame) == 3 and len(calls) == 2

    def no_network(url: str) -> bytes:
        raise AssertionError("the cache should have been used")

    pd.testing.assert_frame_equal(load_french_factors(tmp_path, no_network), frame)


def test_the_factors_plug_into_the_attribution_regression(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    factors = pd.DataFrame(
        rng.normal(0, 0.03, size=(120, 6)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    )
    strategy = 0.004 + 0.3 * factors["Mom"] + rng.normal(0, 0.01, size=120)
    table = hac_regression(strategy, factors)
    assert table.loc["Mom", "coefficient"] == pytest.approx(0.3, abs=0.08)
    assert table.loc["alpha", "t_stat"] > 3
