"""CRSP and Compustat through WRDS: queries, and the steps from raw tables to a panel.

Status. The table and column names below follow the WRDS documentation for the CRSP
"CIZ" format (the only one published since 2025) and for Compustat annual fundamentals.
They have not been run against a live account from this repository. Run
``scripts/wrds_download.py --smoke`` first: it executes each query on a short window and
prints what comes back. Everything after the download works on plain data frames and is tested
with small hand-made tables.

Nothing from WRDS is stored in the repository. ``save_raw`` writes to a local folder
that is git-ignored.

The ``wrds`` package pins an older pandas than this project uses, so it is not a
dependency. The download script runs in its own environment and hands over CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorpipe.panel import Panel
from factorpipe.pit import lag_to_availability

# Common stocks of U.S. companies on NYSE, NYSE American and NASDAQ. In the CIZ format
# these flags replace the old share codes 10 and 11 and exchange codes 1, 2 and 3.
CRSP_MONTHLY_SQL = """
    select permno, mthcaldt, mthret, mthcap, primaryexch, siccd
    from crsp.msf_v2
    where mthcaldt between %(start)s and %(end)s
      and sharetype = 'NS'
      and securitytype = 'EQTY'
      and securitysubtype = 'COM'
      and usincflg = 'Y'
      and issuertype in ('ACOR', 'CORP')
      and primaryexch in ('N', 'A', 'Q')
      and conditionaltype = 'RW'
      and tradingstatusflg = 'A'
"""

COMPUSTAT_ANNUAL_SQL = """
    select gvkey, datadate, at, lt, seq, ceq, pstk, pstkl, pstkrv, txditc,
           ib, revt, cogs, act, lct, che, dlc, txp, dp, csho, ajex
    from comp.funda
    where datadate between %(start)s and %(end)s
      and indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'
"""

CCM_LINK_SQL = """
    select gvkey, lpermno as permno, linkdt, linkenddt
    from crsp.ccmxpf_lnkhist
    where linktype in ('LU', 'LC') and linkprim in ('P', 'C')
"""

CRSP_COLUMNS = ("permno", "mthcaldt", "mthret", "mthcap", "primaryexch", "siccd")
LINK_COLUMNS = ("gvkey", "permno", "linkdt", "linkenddt")

#: Months between a fiscal year end and the date its accounts are treated as public.
REPORTING_LAG_MONTHS = 6
#: Annual accounts are dropped once the fiscal year end is this many months old.
MAX_AGE_MONTHS = 18


@dataclass(frozen=True)
class RawData:
    crsp: pd.DataFrame
    compustat: pd.DataFrame
    links: pd.DataFrame


def download(connection: Any, start: str, end: str) -> RawData:
    """Run the three queries. ``connection`` is a ``wrds.Connection``."""
    params = {"start": start, "end": end}
    return RawData(
        crsp=connection.raw_sql(CRSP_MONTHLY_SQL, params=params, date_cols=["mthcaldt"]),
        compustat=connection.raw_sql(COMPUSTAT_ANNUAL_SQL, params=params, date_cols=["datadate"]),
        links=connection.raw_sql(CCM_LINK_SQL, date_cols=["linkdt", "linkenddt"]),
    )


_DATE_COLUMNS = {"crsp": ["mthcaldt"], "compustat": ["datadate"], "links": ["linkdt", "linkenddt"]}


def save_raw(raw: RawData, folder: str | Path) -> None:
    """Write the three tables as CSV. CSV survives a change of pandas version, which
    matters because the download runs in a separate environment (see the script)."""
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    for name in _DATE_COLUMNS:
        getattr(raw, name).to_csv(target / f"{name}.csv", index=False)


def load_raw(folder: str | Path) -> RawData:
    source = Path(folder)
    tables = [
        pd.read_csv(source / f"{name}.csv", parse_dates=dates, dtype={"gvkey": str})
        for name, dates in _DATE_COLUMNS.items()
    ]
    return RawData(*tables)


# --- CRSP: returns, market value, industry, universe -------------------------------------


@dataclass(frozen=True)
class CrspPanel:
    returns: pd.DataFrame
    market_cap: pd.DataFrame
    exchange: pd.DataFrame
    industry: pd.Series
    #: Months that were dropped because they lay outside an asset's longest unbroken spell.
    months_dropped: int


def month_end(dates: pd.Series) -> pd.Series:
    """Move dates to the calendar month end, so tables with different day conventions align."""
    return pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()


def crsp_panel(crsp: pd.DataFrame) -> CrspPanel:
    """Pivot the monthly file into dates-by-assets frames.

    The panel requires one unbroken listed spell per asset. A security with a gap in its
    monthly returns keeps its longest spell and loses the rest; the number of months lost
    is reported. In the CIZ format the monthly return already includes any delisting
    return, so nothing is merged in here.
    """
    missing = [c for c in CRSP_COLUMNS if c not in crsp.columns]
    if missing:
        raise ValueError(f"the CRSP table is missing columns {missing}")
    table = crsp.assign(date=month_end(crsp["mthcaldt"]))
    if table.duplicated(["permno", "date"]).any():
        raise ValueError("more than one row for a security in one month")
    returns = table.pivot(index="date", columns="permno", values="mthret").astype(float)
    returns = returns.sort_index().asfreq("ME")  # a month with no rows at all becomes a gap
    kept = _longest_spell(returns.notna())
    dropped = int((returns.notna() & ~kept).sum().sum())
    returns = returns.where(kept).clip(lower=-1.0)
    returns = returns.loc[:, returns.notna().any()]

    def aligned(column: str) -> pd.DataFrame:
        frame = table.pivot(index="date", columns="permno", values=column)
        return frame.reindex(index=returns.index, columns=returns.columns)

    last_sic = table.sort_values("date").groupby("permno")["siccd"].last()
    industry = last_sic.reindex(returns.columns).map(sic_division).rename("industry")
    return CrspPanel(
        returns=returns,
        market_cap=aligned("mthcap").astype(float).where(returns.notna()),
        exchange=aligned("primaryexch"),
        industry=industry,
        months_dropped=dropped,
    )


def _longest_spell(present: pd.DataFrame) -> pd.DataFrame:
    """Keep, for each column, the longest run of consecutive True values (the first if tied)."""
    values = present.to_numpy()
    kept = np.zeros_like(values)
    for j in range(values.shape[1]):
        column = values[:, j]
        best_start, best_length, start = 0, 0, None
        for i, flag in enumerate(np.append(column, False)):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start > best_length:
                    best_start, best_length = start, i - start
                start = None
        kept[best_start : best_start + best_length, j] = True
    return pd.DataFrame(kept, index=present.index, columns=present.columns)


_SIC_DIVISIONS = (
    (100, 999, "agriculture"),
    (1000, 1499, "mining"),
    (1500, 1799, "construction"),
    (2000, 3999, "manufacturing"),
    (4000, 4999, "transport_utilities"),
    (5000, 5199, "wholesale"),
    (5200, 5999, "retail"),
    (6000, 6799, "finance"),
    (7000, 8999, "services"),
    (9100, 9999, "public"),
)


def sic_division(code: Any) -> str:
    """Division of the Standard Industrial Classification for a four-digit code."""
    try:
        value = int(code)
    except (TypeError, ValueError):
        return "other"
    for low, high, name in _SIC_DIVISIONS:
        if low <= value <= high:
            return name
    return "other"


def size_universe(
    market_cap: pd.DataFrame, exchange: pd.DataFrame, percentile: float = 0.2
) -> pd.DataFrame:
    """True where market value is at or above the NYSE ``percentile`` breakpoint that month.

    Breakpoints come from NYSE stocks only, as in Fama and French, so that the many small
    NASDAQ stocks do not decide where "small" starts.
    """
    nyse = market_cap.where(exchange == "N")
    breakpoint_ = nyse.quantile(percentile, axis=1)
    return market_cap.ge(breakpoint_, axis=0) & market_cap.notna()


# --- Compustat: link, annual variables, point-in-time ------------------------------------


def link_compustat(compustat: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Attach the CRSP ``permno`` that was linked to each ``gvkey`` on the fiscal year end.

    A link is valid from ``linkdt`` to ``linkenddt``; a missing end date means it is still
    open. Fiscal years with no valid link are dropped.
    """
    missing = [c for c in LINK_COLUMNS if c not in links.columns]
    if missing:
        raise ValueError(f"the link table is missing columns {missing}")
    merged = compustat.merge(links, on="gvkey", how="inner")
    date = pd.to_datetime(merged["datadate"])
    start = pd.to_datetime(merged["linkdt"])
    end = pd.to_datetime(merged["linkenddt"])
    valid = (date >= start) & (end.isna() | (date <= end))
    linked = merged[valid].drop(columns=["linkdt", "linkenddt"])
    linked = linked.sort_values(["permno", "datadate"]).drop_duplicates(["permno", "datadate"])
    return linked.assign(permno=linked["permno"].astype(int)).reset_index(drop=True)


ANNUAL_VARIABLES = (
    "book_equity",
    "earnings",
    "gross_profitability",
    "asset_growth",
    "accruals",
    "share_issuance",
)


def annual_variables(linked: pd.DataFrame) -> pd.DataFrame:
    """Accounting variables per security and fiscal year, from the cited definitions.

    - ``book_equity`` (Fama and French 1993): stockholders' equity, plus deferred taxes
      and investment tax credit, minus preferred stock. Stockholders' equity is ``seq``,
      else ``ceq + pstk``, else ``at - lt``. Preferred stock is ``pstkrv``, else ``pstkl``,
      else ``pstk``. Non-positive values are dropped.
    - ``earnings``: income before extraordinary items, ``ib``.
    - ``gross_profitability`` (Novy-Marx 2013): ``(revt - cogs) / at``.
    - ``asset_growth`` (Cooper, Gulen and Schill 2008): ``at / at[-1] - 1``.
    - ``accruals`` (Sloan 1996): change in non-cash current assets, minus change in
      current liabilities excluding debt and taxes payable, minus depreciation, over
      average total assets.
    - ``share_issuance`` (Pontiff and Woodgate 2008, annual version): log growth of
      split-adjusted shares, ``csho * ajex``.

    Year-over-year changes use the previous fiscal year of the same security and are
    missing if that year is more than 18 months earlier.
    """
    f = linked.sort_values(["permno", "datadate"]).reset_index(drop=True)
    numeric = [c for c in f.columns if c not in ("gvkey", "permno", "datadate")]
    f[numeric] = f[numeric].astype(float)
    previous = f.groupby("permno")[numeric + ["datadate"]].shift(1)
    gap = (pd.to_datetime(f["datadate"]) - pd.to_datetime(previous["datadate"])).dt.days
    has_previous = gap.between(1, 550)

    def change(column: str) -> pd.Series:
        return (f[column].fillna(0.0) - previous[column].fillna(0.0)).where(has_previous)

    equity = f["seq"].fillna(f["ceq"] + f["pstk"]).fillna(f["at"] - f["lt"])
    preferred = f["pstkrv"].fillna(f["pstkl"]).fillna(f["pstk"]).fillna(0.0)
    book_equity = equity + f["txditc"].fillna(0.0) - preferred

    average_assets = ((f["at"] + previous["at"]) / 2.0).where(has_previous)
    working_capital = (change("act") - change("che")) - (
        change("lct") - change("dlc") - change("txp")
    )
    shares = f["csho"] * f["ajex"]
    previous_shares = previous["csho"] * previous["ajex"]

    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame(
            {
                "permno": f["permno"],
                "datadate": pd.to_datetime(f["datadate"]),
                "book_equity": book_equity.where(book_equity > 0),
                "earnings": f["ib"],
                "gross_profitability": (f["revt"] - f["cogs"]) / f["at"].where(f["at"] > 0),
                "asset_growth": (f["at"] / previous["at"].where(previous["at"] > 0) - 1.0).where(
                    has_previous
                ),
                "accruals": (working_capital - f["dp"].fillna(0.0))
                / average_assets.where(average_assets > 0),
                "share_issuance": np.log(
                    shares.where(shares > 0) / previous_shares.where(previous_shares > 0)
                ).where(has_previous),
            }
        )
    return out.replace([np.inf, -np.inf], np.nan)


def point_in_time(
    annual: pd.DataFrame, variable: str, dates: pd.DatetimeIndex, assets: pd.Index
) -> pd.DataFrame:
    """Monthly frame of ``variable`` as it was public at each month end.

    Compustat's annual file has no publication date. Each fiscal year is treated as
    public ``REPORTING_LAG_MONTHS`` after its end, the convention behind the June
    rebalance in Fama and French (1992), and is dropped after ``MAX_AGE_MONTHS``.
    """
    rows = annual[["permno", "datadate", variable]].dropna()
    period_end = month_end(rows["datadate"])
    publish = (period_end.dt.to_period("M") + REPORTING_LAG_MONTHS).dt.to_timestamp(how="end")
    table = pd.DataFrame(
        {
            "asset": rows["permno"].to_numpy(),
            "period_end": period_end.to_numpy(),
            "publish_date": publish.dt.normalize().to_numpy(),
            "value": rows[variable].to_numpy(dtype=float),
        }
    )
    return lag_to_availability(table, dates, assets, max_age_months=MAX_AGE_MONTHS)


# --- the eleven factors ------------------------------------------------------------------

FACTORS = (
    "size",
    "book_to_market",
    "earnings_yield",
    "momentum",
    "reversal",
    "gross_profitability",
    "asset_growth",
    "accruals",
    "share_issuance",
    "volatility",
    "beta",
)


def build_panel(raw: RawData, size_percentile: float | None = 0.2) -> Panel:
    """From raw WRDS tables to a panel with the eleven factors.

    Every factor is signed so that a higher value goes with a higher expected return in
    the paper it comes from: size, reversal, asset growth, accruals, share issuance,
    volatility and beta enter with a minus sign.
    """
    from factorpipe.risk import trailing_beta

    crsp = crsp_panel(raw.crsp)
    returns, cap = crsp.returns, crsp.market_cap
    dates, assets = returns.index, returns.columns
    annual = annual_variables(link_compustat(raw.compustat, raw.links))
    pit = {name: point_in_time(annual, name, dates, assets) for name in ANNUAL_VARIABLES}
    listed = returns.notna()
    positive_cap = cap.where(cap > 0)

    # Return over months t-11 .. t-1: skip the most recent month, as in Jegadeesh and Titman.
    log_growth = np.log((1.0 + returns).clip(lower=1e-9)).where(listed)
    momentum = np.exp(log_growth.shift(1).rolling(11, min_periods=11).sum()) - 1.0
    base = Panel(returns)
    characteristics = {
        "size": -np.log(positive_cap),
        "book_to_market": pit["book_equity"] / positive_cap,
        "earnings_yield": pit["earnings"] / positive_cap,
        "momentum": momentum,
        "reversal": -returns,
        "gross_profitability": pit["gross_profitability"],
        "asset_growth": -pit["asset_growth"],
        "accruals": -pit["accruals"],
        "share_issuance": -pit["share_issuance"],
        "volatility": -returns.rolling(36, min_periods=24).std(),
        "beta": -trailing_beta(base, window=60, min_periods=24),
    }
    characteristics = {
        name: frame.where(listed).replace([np.inf, -np.inf], np.nan)
        for name, frame in characteristics.items()
    }
    universe = None
    if size_percentile is not None:
        universe = size_universe(cap, crsp.exchange, size_percentile)
    return Panel(returns, characteristics, crsp.industry, universe)
