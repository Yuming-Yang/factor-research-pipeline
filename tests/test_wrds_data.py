import numpy as np
import pandas as pd
import pytest

from factorpipe.evaluation import screen_factors
from factorpipe.panel import Panel, PanelError
from factorpipe.wrds_data import (
    FACTORS,
    RawData,
    annual_variables,
    build_panel,
    crsp_panel,
    download,
    link_compustat,
    load_raw,
    month_end,
    point_in_time,
    save_raw,
    sic_division,
    size_universe,
)

MONTHS = pd.date_range("2018-01-31", periods=8, freq="ME")


def crsp_rows(permno, returns, exchange="N", sic=3571, cap=1000.0, start=0):
    rows = []
    for i, r in enumerate(returns):
        if r is None:
            continue
        # CRSP dates are the last trading day, not the calendar month end.
        day = MONTHS[start + i] - pd.Timedelta(days=2)
        rows.append((permno, day, r, cap * (1 + i), exchange, sic))
    return pd.DataFrame(
        rows, columns=["permno", "mthcaldt", "mthret", "mthcap", "primaryexch", "siccd"]
    )


def funda_row(gvkey, datadate, **values):
    base = dict.fromkeys(
        ["at", "lt", "seq", "ceq", "pstk", "pstkl", "pstkrv", "txditc", "ib", "revt", "cogs"]
        + ["act", "lct", "che", "dlc", "txp", "dp", "csho", "ajex"],
        np.nan,
    )
    return {"gvkey": gvkey, "datadate": pd.Timestamp(datadate), **base, **values}


# --- CRSP --------------------------------------------------------------------------------


def test_dates_are_moved_to_the_calendar_month_end():
    moved = month_end(pd.Series(pd.to_datetime(["2018-01-29", "2018-02-28", "2018-06-01"])))
    assert list(moved) == list(pd.to_datetime(["2018-01-31", "2018-02-28", "2018-06-30"]))


def test_the_monthly_file_becomes_aligned_frames():
    table = pd.concat(
        [
            crsp_rows(10001, [0.01] * 8),
            crsp_rows(10002, [0.02, 0.03, -0.5], exchange="Q", sic=6020, start=2),
        ]
    )
    panel = crsp_panel(table)
    assert list(panel.returns.index) == list(MONTHS)
    assert list(panel.returns.columns) == [10001, 10002]
    assert panel.returns[10002].notna().sum() == 3
    assert panel.returns.loc[MONTHS[4], 10002] == -0.5
    assert panel.market_cap.loc[MONTHS[2], 10002] == 1000.0
    assert panel.exchange.loc[MONTHS[3], 10002] == "Q"
    assert panel.industry.to_dict() == {10001: "manufacturing", 10002: "finance"}
    assert panel.months_dropped == 0
    Panel(panel.returns)  # a valid panel


def test_a_security_with_a_gap_keeps_its_longest_spell():
    table = pd.concat(
        [
            crsp_rows(10001, [0.01, 0.01, None, 0.02, 0.02, 0.02, None, 0.03]),
            crsp_rows(10002, [0.0] * 8),
        ]
    )
    panel = crsp_panel(table)
    kept = panel.returns[10001]
    assert list(kept.notna()) == [False, False, False, True, True, True, False, False]
    assert panel.months_dropped == 3
    assert panel.market_cap[10001].notna().sum() == 3
    Panel(panel.returns)


def test_a_month_missing_for_every_security_is_a_gap_not_a_shorter_calendar():
    table = pd.concat([crsp_rows(1, [0.01, 0.01, None, 0.02, 0.02, 0.02, 0.02, 0.02])])
    panel = crsp_panel(table)
    assert len(panel.returns) == 8
    assert panel.returns[1].notna().sum() == 5


def test_bad_crsp_tables_are_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        crsp_panel(pd.DataFrame({"permno": [1]}))
    doubled = pd.concat([crsp_rows(1, [0.01, 0.02]), crsp_rows(1, [0.03])])
    with pytest.raises(ValueError, match="more than one row"):
        crsp_panel(doubled)


def test_sic_divisions():
    assert sic_division(1311) == "mining"
    assert sic_division(2834) == "manufacturing"
    assert sic_division(4911) == "transport_utilities"
    assert sic_division(5411) == "retail"
    assert sic_division(6798) == "finance"
    assert sic_division(7372) == "services"
    assert sic_division(None) == "other" and sic_division(9999) == "public"
    assert sic_division(1800) == "other"


def test_the_size_cut_uses_nyse_breakpoints_only():
    dates = MONTHS[:1]
    cap = pd.DataFrame([[100.0, 200.0, 300.0, 400.0, 500.0, 5.0, 150.0, np.nan]], index=dates)
    exchange = pd.DataFrame([["N", "N", "N", "N", "N", "Q", "Q", "N"]], index=dates)
    universe = size_universe(cap, exchange, percentile=0.2)
    # The NYSE 20th percentile of 100..500 is 180. Tiny NASDAQ stocks do not move it.
    assert list(universe.iloc[0]) == [False, True, True, True, True, False, False, False]


# --- Compustat ---------------------------------------------------------------------------


def test_links_apply_only_inside_their_dates():
    funda = pd.DataFrame(
        [funda_row("001", "2015-12-31", at=1.0), funda_row("001", "2017-12-31", at=2.0)]
        + [funda_row("002", "2016-12-31", at=3.0), funda_row("003", "2016-12-31", at=4.0)]
    )
    links = pd.DataFrame(
        {
            "gvkey": ["001", "001", "002"],
            "permno": [10001.0, 10009.0, 10002.0],
            "linkdt": pd.to_datetime(["2010-01-01", "2017-01-01", "2012-01-01"]),
            "linkenddt": pd.to_datetime(["2016-12-31", None, None]),
        }
    )
    linked = link_compustat(funda, links)
    assert list(zip(linked["gvkey"], linked["permno"], strict=True)) == [
        ("001", 10001),
        ("002", 10002),
        ("001", 10009),
    ]
    assert "003" not in set(linked["gvkey"])  # no link, dropped


def test_annual_variables_by_hand():
    rows = [
        funda_row(
            "001",
            "2016-12-31",
            at=100.0,
            lt=60.0,
            seq=40.0,
            pstk=5.0,
            txditc=2.0,
            ib=8.0,
            revt=90.0,
            cogs=50.0,
            act=50.0,
            lct=30.0,
            che=10.0,
            dlc=5.0,
            txp=1.0,
            dp=4.0,
            csho=10.0,
            ajex=1.0,
        ),
        funda_row(
            "001",
            "2017-12-31",
            at=120.0,
            lt=70.0,
            seq=50.0,
            pstk=5.0,
            txditc=2.0,
            ib=9.0,
            revt=110.0,
            cogs=60.0,
            act=62.0,
            lct=36.0,
            che=14.0,
            dlc=7.0,
            txp=2.0,
            dp=5.0,
            csho=5.5,
            ajex=2.0,
        ),
    ]
    linked = pd.DataFrame(rows).assign(permno=10001)
    out = annual_variables(linked).set_index("datadate")
    first, second = out.iloc[0], out.iloc[1]
    assert first["book_equity"] == pytest.approx(40.0 + 2.0 - 5.0)
    assert first["gross_profitability"] == pytest.approx(0.4)
    assert first[["asset_growth", "accruals", "share_issuance"]].isna().all()
    assert second["asset_growth"] == pytest.approx(0.2)
    # Non-cash current assets +8 (12 - 4); current liabilities excluding debt and tax +3
    # (6 - 2 - 1); depreciation 5; average assets 110.
    assert second["accruals"] == pytest.approx((8.0 - 3.0 - 5.0) / 110.0)
    # 10 shares become 5.5 * 2 = 11 after adjusting for the split.
    assert second["share_issuance"] == pytest.approx(np.log(1.1))
    assert second["earnings"] == 9.0


def test_book_equity_fallbacks_and_stale_previous_years():
    rows = [
        funda_row("001", "2014-12-31", at=100.0, lt=70.0, pstkl=3.0),  # equity = at - lt
        funda_row("001", "2017-12-31", at=150.0, ceq=20.0, pstk=4.0, pstkrv=6.0),
        funda_row("002", "2017-12-31", at=50.0, lt=60.0),  # negative book equity
    ]
    linked = pd.DataFrame(rows).assign(permno=[1, 1, 2])
    out = annual_variables(linked)
    assert out["book_equity"].iloc[0] == pytest.approx(30.0 - 3.0)
    assert out["book_equity"].iloc[1] == pytest.approx(20.0 + 4.0 - 6.0)
    assert np.isnan(out["book_equity"].iloc[2])
    assert np.isnan(out["asset_growth"].iloc[1])  # the previous year is three years old


def test_annual_accounts_become_public_six_months_after_the_year_end():
    annual = pd.DataFrame(
        {
            "permno": [1, 1],
            "datadate": pd.to_datetime(["2016-12-31", "2017-12-31"]),
            "book_equity": [10.0, 12.0],
        }
    )
    dates = pd.date_range("2017-01-31", periods=36, freq="ME")
    out = point_in_time(annual, "book_equity", dates, pd.Index([1, 2]))[1]
    assert np.isnan(out.loc["2017-05-31"])
    assert out.loc["2017-06-30"] == 10.0
    assert out.loc["2018-05-31"] == 10.0
    assert out.loc["2018-06-30"] == 12.0
    assert out.loc["2019-06-30"] == 12.0  # 18 months after the fiscal year end
    assert np.isnan(out.loc["2019-07-31"])


# --- end to end on a made-up market ------------------------------------------------------


def made_up_market(n_assets: int = 60, n_months: int = 72, seed: int = 0) -> RawData:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-31", periods=n_months, freq="ME")
    crsp, funda, links = [], [], []
    for a in range(n_assets):
        permno, gvkey = 10000 + a, f"{a:06d}"
        cap = float(np.exp(rng.normal(6, 1.5)))
        for date in dates:
            r = float(rng.normal(0.01, 0.08))
            cap *= 1 + r
            crsp.append((permno, date, r, cap, "N" if a % 2 else "Q", 2000 + 100 * (a % 60)))
        links.append((gvkey, float(permno), pd.Timestamp("2000-01-01"), pd.NaT))
        assets = float(np.exp(rng.normal(6, 1)))
        for year in range(2010, 2018):
            assets *= float(np.exp(rng.normal(0.05, 0.1)))
            funda.append(
                funda_row(
                    gvkey,
                    f"{year}-12-31",
                    at=assets,
                    lt=0.6 * assets,
                    seq=0.4 * assets,
                    ib=0.05 * assets * rng.normal(1, 0.5),
                    revt=assets,
                    cogs=0.6 * assets,
                    act=0.5 * assets,
                    lct=0.3 * assets,
                    che=0.1 * assets,
                    dlc=0.05 * assets,
                    txp=0.01 * assets,
                    dp=0.04 * assets,
                    csho=100.0 * (1 + 0.02 * (year - 2010)),
                    ajex=1.0,
                )
            )
    return RawData(
        crsp=pd.DataFrame(
            crsp, columns=["permno", "mthcaldt", "mthret", "mthcap", "primaryexch", "siccd"]
        ),
        compustat=pd.DataFrame(funda),
        links=pd.DataFrame(links, columns=["gvkey", "permno", "linkdt", "linkenddt"]),
    )


def test_a_panel_is_built_with_all_eleven_factors():
    raw = made_up_market()
    panel = build_panel(raw)
    assert set(panel.characteristics) == set(FACTORS) and len(FACTORS) == 11
    assert panel.industry.notna().all()
    assert 0.3 < panel.universe.iloc[-1].mean() < 0.95
    for name in FACTORS:
        assert panel.characteristics[name].iloc[-2].notna().mean() > 0.9, name
    # The pipeline runs on it. The data is noise, so nothing should be strongly significant.
    table = screen_factors(panel)
    assert len(table) == 11
    assert table["t_stat"].abs().max() < 4.5


def test_factor_definitions_on_single_cells():
    raw = made_up_market()
    panel = build_panel(raw, size_percentile=None)
    assert panel.universe is None
    returns = panel.returns
    asset, t = returns.columns[3], 40
    window = returns[asset].iloc[t - 11 : t]  # months t-11 .. t-1
    assert panel.characteristics["momentum"][asset].iloc[t] == pytest.approx(
        (1 + window).prod() - 1
    )
    assert panel.characteristics["reversal"][asset].iloc[t] == -returns[asset].iloc[t]
    cap = crsp_panel(raw.crsp).market_cap
    assert panel.characteristics["size"][asset].iloc[t] == pytest.approx(
        -np.log(cap[asset].iloc[t])
    )
    # Book-to-market in July 2014 uses the fiscal year 2013 book equity over that month's cap.
    july = pd.Timestamp("2014-07-31")
    accounts = raw.compustat
    year_2013 = (accounts["gvkey"] == "000003") & (accounts["datadate"] == "2013-12-31")
    book = accounts.loc[year_2013, "seq"].iloc[0]
    assert panel.characteristics["book_to_market"].loc[july, asset] == pytest.approx(
        book / cap.loc[july, asset]
    )


def test_assets_outside_the_universe_are_not_investable_but_keep_their_returns():
    panel = build_panel(made_up_market())
    outside = ~panel.universe & panel.listed
    assert outside.iloc[:-1].any().any()
    assert not (panel.investable & outside).any().any()
    assert panel.forward_returns(1).where(outside).isna().all().all()
    assert panel.returns.where(outside).notna().any().any()
    with pytest.raises(PanelError, match="universe"):
        Panel(panel.returns, universe=panel.universe.iloc[:-1])


# --- download and cache ------------------------------------------------------------------


class FakeConnection:
    def __init__(self, raw: RawData):
        self.raw, self.calls = raw, []

    def raw_sql(self, sql, params=None, date_cols=None):
        self.calls.append((sql, params, date_cols))
        if "crsp.msf_v2" in sql:
            return self.raw.crsp
        return self.raw.compustat if "comp.funda" in sql else self.raw.links


def test_download_runs_three_queries_with_the_date_range(tmp_path):
    raw = made_up_market(n_assets=5, n_months=24)
    connection = FakeConnection(raw)
    fetched = download(connection, "2012-01-01", "2013-12-31")
    assert len(connection.calls) == 3
    assert connection.calls[0][1] == {"start": "2012-01-01", "end": "2013-12-31"}
    assert connection.calls[0][2] == ["mthcaldt"]
    assert "sharetype = 'NS'" in connection.calls[0][0]
    assert "linktype in ('LU', 'LC')" in connection.calls[2][0]
    save_raw(fetched, tmp_path / "cache")
    again = load_raw(tmp_path / "cache")
    # CSV keeps what the pipeline needs: dates as dates, gvkey as text with leading zeros.
    assert again.compustat["gvkey"].iloc[0] == "000000"
    assert pd.api.types.is_datetime64_any_dtype(again.crsp["mthcaldt"])
    assert pd.api.types.is_datetime64_any_dtype(again.links["linkenddt"])
    rebuilt, original = build_panel(again), build_panel(raw)
    pd.testing.assert_frame_equal(rebuilt.returns, original.returns, check_freq=False)
    pd.testing.assert_frame_equal(
        rebuilt.characteristics["book_to_market"],
        original.characteristics["book_to_market"],
        check_freq=False,
    )


# --- holdout -----------------------------------------------------------------------------


def test_loading_with_an_end_date_never_reads_later_data(tmp_path):
    from factorpipe.wrds_data import truncate

    raw = made_up_market(n_assets=20, n_months=72)
    save_raw(raw, tmp_path)
    cutoff = "2015-12-31"
    early = load_raw(tmp_path, end=cutoff)
    assert early.crsp["mthcaldt"].max() == pd.Timestamp(cutoff)
    assert early.compustat["datadate"].max() <= pd.Timestamp(cutoff)
    assert len(early.crsp) < len(raw.crsp) and len(early.compustat) < len(raw.compustat)
    pd.testing.assert_frame_equal(early.crsp, truncate(raw, cutoff).crsp)

    # The panel built without the later data is the same, month for month, as the early
    # part of the panel built with it: nothing before the cutoff depends on what follows.
    short, full = build_panel(early), build_panel(load_raw(tmp_path))
    assert short.returns.index.max() == pd.Timestamp(cutoff)
    upto = full.returns.index <= pd.Timestamp(cutoff)
    pd.testing.assert_frame_equal(short.returns, full.returns.loc[upto], check_freq=False)
    for name in FACTORS:
        pd.testing.assert_frame_equal(
            short.characteristics[name],
            full.characteristics[name].loc[upto],
            check_freq=False,
            obj=name,
        )
    pd.testing.assert_frame_equal(short.universe, full.universe.loc[upto], check_freq=False)


def test_a_start_date_trims_the_panel_after_factors_are_computed():
    raw = made_up_market(n_assets=20, n_months=72)
    full = build_panel(raw)
    trimmed = build_panel(raw, start="2014-01-31")
    assert trimmed.returns.index.min() == pd.Timestamp("2014-01-31")
    # Momentum is already available in the first month, because it was computed on the
    # earlier history before trimming.
    assert trimmed.characteristics["momentum"].iloc[0].notna().mean() > 0.9
    pd.testing.assert_frame_equal(
        trimmed.characteristics["momentum"],
        full.characteristics["momentum"].loc["2014-01-31":],
        check_freq=False,
    )
    Panel(trimmed.returns, trimmed.characteristics, trimmed.industry, trimmed.universe)


def test_dropping_securities_never_in_the_universe_changes_nothing_investable():
    from factorpipe.wrds_data import restrict_to_universe

    panel = build_panel(made_up_market())
    never = ~panel.universe.any()
    forced = panel.universe.copy()
    forced.iloc[:, :5] = False  # make sure some securities are never in the universe
    panel = Panel(panel.returns, panel.characteristics, panel.industry, forced)
    small = restrict_to_universe(panel)
    assert small.returns.shape[1] == panel.returns.shape[1] - 5 - int(never.iloc[5:].sum())
    kept = small.assets
    pd.testing.assert_frame_equal(small.investable, panel.investable[kept])
    pd.testing.assert_frame_equal(small.forward_returns(3), panel.forward_returns(3)[kept])
    assert list(small.industry.index) == list(kept)
    plain = Panel(panel.returns)
    assert restrict_to_universe(plain) is plain
