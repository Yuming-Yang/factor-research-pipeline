"""A bar-by-bar backtest engine.

The engine knows nothing about factors. It takes a market of raw prices and a function
that returns target weights at each close, and it keeps the books: shares, cash, costs,
corporate actions, delistings and borrow fees. Bars can be days or months.

Rules, in the order they apply within a bar ``t``:

1. Corporate actions with ex-date ``t`` take effect at the open: a dividend pays cash on
   the shares held before any split, then a split multiplies the share count.
2. Orders decided at the previous close are filled, at this bar's open or close
   depending on ``fill_at``. Every order is sized from the same net asset value, taken
   just before the first fill, with assets that have no fill price marked at their last
   close: ``shares = weight * nav / price``. Shares are fractional.
3. Each fill pays ``cost_rate * |notional|`` out of cash.
4. An order in an asset with no price on the fill bar is cancelled, not carried.
5. An order may not open or enlarge a short in an asset listed in ``no_short``.
6. A position in an asset whose last bar is ``t`` becomes cash at the close:
   ``shares * close * (1 + delisting_return)``.
7. Shorts pay ``borrow_rate * |short market value|`` at each close.
8. The decision function is then called with what is known at the close of ``t``.

A decision made on the last bar is never filled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factorpipe.panel import Panel

#: Trades smaller than this fraction of net asset value are floating-point noise.
MIN_TRADE = 1e-12


@dataclass(frozen=True)
class Market:
    """Raw prices and the rules for trading them.

    ``open`` and ``close`` are bars-by-assets frames of prices as traded; a missing value
    means no trading in that bar. ``splits`` (new shares per old share) and ``dividends``
    (cash per share) are on the same grid, missing where nothing happens. ``last_bar``
    gives the final trading date of each asset that delists, and ``delisting_return`` the
    return from that close to the cash finally received.
    """

    open: pd.DataFrame
    close: pd.DataFrame
    last_bar: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns]"))
    delisting_return: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    splits: pd.DataFrame | None = None
    dividends: pd.DataFrame | None = None
    cost_rate: float = 0.0
    borrow_rate: float = 0.0
    no_short: frozenset = frozenset()
    fill_at: str = "open"
    initial_cash: float = 1.0

    def __post_init__(self) -> None:
        if not self.open.index.equals(self.close.index) or not self.open.columns.equals(
            self.close.columns
        ):
            raise ValueError("open and close must share dates and assets")
        if self.fill_at not in ("open", "close"):
            raise ValueError("fill_at must be 'open' or 'close'")
        if self.cost_rate < 0 or self.borrow_rate < 0 or self.initial_cash <= 0:
            raise ValueError("rates cannot be negative and initial_cash must be positive")
        for frame in (self.open, self.close):
            values = frame.to_numpy(dtype=float)
            if (values[~np.isnan(values)] <= 0).any():
                raise ValueError("prices must be positive where present")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def assets(self) -> pd.Index:
        return self.close.columns


@dataclass(frozen=True)
class Snapshot:
    """What the engine exposes at the close of ``as_of``. Nothing in it is dated later."""

    as_of: pd.Timestamp
    bar: int
    nav: float
    #: Holdings as fractions of net asset value at this close, before any new trade.
    weights: pd.Series
    _market: Market = field(repr=False)

    @property
    def open(self) -> pd.DataFrame:
        return self._market.open.iloc[: self.bar + 1]

    @property
    def close(self) -> pd.DataFrame:
        return self._market.close.iloc[: self.bar + 1]


@dataclass(frozen=True)
class BacktestRun:
    #: Net asset value after each close.
    nav: pd.Series
    #: Holdings as fractions of net asset value after each close.
    weights: pd.DataFrame
    #: One row per fill: date, asset, quantity (signed shares), price, cost (cash).
    trades: pd.DataFrame
    #: Traded notional over net asset value at the fill, per bar.
    turnover: pd.Series
    #: Transaction costs and borrow fees paid per bar, in cash.
    costs: pd.Series
    initial_cash: float

    @property
    def returns(self) -> pd.Series:
        previous = self.nav.shift(1, fill_value=self.initial_cash)
        return (self.nav / previous - 1.0).rename("return")

    @property
    def active_returns(self) -> pd.Series:
        """Returns from the bar of the first fill on, leaving out the time spent in cash
        before the strategy had enough history to trade."""
        started = self.turnover.gt(0).cumsum().gt(0)
        return self.returns[started]


Decide = Callable[[pd.Timestamp, Snapshot], "pd.Series | None"]


def run_backtest(market: Market, decide: Decide) -> BacktestRun:
    """Run ``decide`` through ``market``.

    ``decide(t, snapshot)`` returns target weights as fractions of net asset value,
    indexed by asset. Assets left out are targeted at zero; ``None`` means no order.
    """
    dates, assets = market.dates, market.assets
    n_assets = len(assets)
    opens = market.open.to_numpy(dtype=float)
    closes = market.close.to_numpy(dtype=float)
    splits = None if market.splits is None else _on_grid(market.splits, market, 1.0)
    dividends = None if market.dividends is None else _on_grid(market.dividends, market, 0.0)

    last_bar = np.full(n_assets, -1)
    final_return = np.zeros(n_assets)
    for asset, day in market.last_bar.dropna().items():
        last_bar[assets.get_loc(asset)] = dates.get_loc(pd.Timestamp(day))
        final_return[assets.get_loc(asset)] = float(market.delisting_return.get(asset, 0.0))
    no_short = np.array([a in market.no_short for a in assets])

    shares = np.zeros(n_assets)
    cash = float(market.initial_cash)
    last_close = np.full(n_assets, np.nan)
    pending: np.ndarray | None = None
    navs, held, turnover, costs = [], [], [], []
    fills: list[pd.DataFrame] = []

    for i, t in enumerate(dates):
        if dividends is not None:
            cash += float(shares @ dividends[i])
        if splits is not None:
            shares = shares * splits[i]

        traded, paid = 0.0, 0.0
        if pending is not None:
            price = opens[i] if market.fill_at == "open" else closes[i]
            tradable = ~np.isnan(price)
            mark = np.where(tradable, price, last_close)
            nav_fill = cash + _value(shares, mark).sum()
            safe_price = np.where(tradable, price, 1.0)
            wanted = np.where(tradable, pending * nav_fill / safe_price, shares)
            wanted = np.where(no_short, np.maximum(wanted, np.minimum(shares, 0.0)), wanted)
            delta = wanted - shares
            notional = delta * safe_price
            fill = tradable & (np.abs(notional) > MIN_TRADE * abs(nav_fill))
            if fill.any():
                cost = market.cost_rate * np.abs(notional[fill])
                shares = np.where(fill, wanted, shares)
                cash -= float(notional[fill].sum() + cost.sum())
                traded = float(np.abs(notional[fill]).sum()) / abs(nav_fill)
                paid += float(cost.sum())
                fills.append(
                    pd.DataFrame(
                        {
                            "date": t,
                            "asset": assets[fill],
                            "quantity": delta[fill],
                            "price": price[fill],
                            "cost": cost,
                        }
                    )
                )
            pending = None

        last_close = np.where(np.isnan(closes[i]), last_close, closes[i])
        gone = last_bar == i
        if gone.any():
            cash += float((_value(shares, last_close) * (1.0 + final_return))[gone].sum())
            shares = np.where(gone, 0.0, shares)

        value = _value(shares, last_close)
        fee = market.borrow_rate * float(-value[value < 0].sum())
        cash -= fee
        nav = cash + float(value.sum())
        weights = value / nav
        navs.append(nav)
        held.append(weights)
        turnover.append(traded)
        costs.append(paid + fee)

        snapshot = Snapshot(t, i, nav, pd.Series(weights, index=assets), market)
        pending = _target(decide(t, snapshot), assets)

    columns = ["date", "asset", "quantity", "price", "cost"]
    trades = pd.concat(fills, ignore_index=True) if fills else pd.DataFrame(columns=columns)
    return BacktestRun(
        nav=pd.Series(navs, index=dates, name="nav"),
        weights=pd.DataFrame(np.array(held), index=dates, columns=assets),
        trades=trades[columns],
        turnover=pd.Series(turnover, index=dates, name="turnover"),
        costs=pd.Series(costs, index=dates, name="costs"),
        initial_cash=float(market.initial_cash),
    )


def market_from_panel(panel: Panel, cost_rate: float = 0.0, borrow_rate: float = 0.0) -> Market:
    """Turn a panel of monthly total returns into a market the engine can trade.

    Each asset gets a total-return price index. The open of month ``t`` equals the close
    of ``t-1``, so an order decided at the end of ``t-1`` is filled at that month-end price
    and earns exactly the return of month ``t``. An asset has no open in its first listed
    month, so it cannot be bought before it has a month-end price. Delisting returns are
    already inside the panel's last monthly return, so the engine's own delisting return
    is zero and the position is cashed at the last close.
    """
    returns = panel.returns
    growth = (1.0 + returns).clip(lower=1e-9)
    close = growth.where(returns.notna(), 1.0).cumprod().where(returns.notna())
    open_ = close.shift(1).where(returns.notna())
    listed = returns.notna()
    ends = listed & ~listed.shift(-1, fill_value=True)
    last_bar = ends.apply(lambda col: col.index[col][-1] if col.any() else pd.NaT)
    return Market(
        open=open_,
        close=close,
        last_bar=last_bar.dropna(),
        delisting_return=pd.Series(0.0, index=last_bar.dropna().index),
        cost_rate=cost_rate,
        borrow_rate=borrow_rate,
    )


def _value(shares: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Market value per asset; an asset that is not held is worth zero even without a price."""
    return np.where(shares != 0.0, shares * np.where(np.isnan(price), 0.0, price), 0.0)


def _on_grid(frame: pd.DataFrame, market: Market, neutral: float) -> np.ndarray:
    aligned = frame.reindex(index=market.dates, columns=market.assets)
    return aligned.fillna(neutral).to_numpy(dtype=float)


def _target(target: pd.Series | None, assets: pd.Index) -> np.ndarray | None:
    if target is None:
        return None
    unknown = set(target.index) - set(assets)
    if unknown:
        raise ValueError(f"targets refer to unknown assets {sorted(unknown)[:5]}")
    values = target.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("target weights must be finite")
    return values
