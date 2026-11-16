"""The panel: monthly returns and characteristics on one date-by-asset grid.

Timing convention used by the whole pipeline:

- Row ``t`` of ``returns`` is the simple total return earned over month ``t``.
  If an asset delists in month ``t``, that row already includes the delisting return.
- Row ``t`` of a characteristic is what was publicly known at the end of month ``t``.

So a characteristic at ``t`` may be used to form a portfolio that earns the returns of
``t+1`` onward, and never the return of ``t`` itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class PanelError(ValueError):
    """The inputs do not form a valid panel."""


@dataclass(frozen=True)
class Panel:
    """Monthly returns and point-in-time characteristics.

    ``returns`` is a dates-by-assets frame. A missing value means the asset is not listed
    in that month. Each asset must be listed for one unbroken spell: a gap inside the
    spell would be ambiguous between a delisting and missing data, so it is rejected.
    """

    returns: pd.DataFrame
    characteristics: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_grid(self.returns)
        values = self.returns.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise PanelError("returns contain infinite values")
        if (values[~np.isnan(values)] < -1.0).any():
            raise PanelError("a simple return cannot be below -100%")
        _check_one_spell(self.returns)
        for name, frame in self.characteristics.items():
            if not frame.index.equals(self.returns.index) or not frame.columns.equals(
                self.returns.columns
            ):
                raise PanelError(f"characteristic {name!r} is not on the same grid as returns")
            if np.isinf(frame.to_numpy(dtype=float)).any():
                raise PanelError(f"characteristic {name!r} contains infinite values")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.returns.index

    @property
    def assets(self) -> pd.Index:
        return self.returns.columns

    @property
    def listed(self) -> pd.DataFrame:
        """True where the asset has a return in that month."""
        return self.returns.notna()

    @property
    def investable(self) -> pd.DataFrame:
        """True where a position can be opened at the end of the month.

        An asset is investable at the end of ``t`` if it is listed in ``t`` and still
        listed in ``t+1``. An asset whose last listed month is ``t`` delisted during
        ``t``, so there is nothing left to buy. The last row is never investable,
        because there is no later month to confirm it.
        """
        listed = self.listed
        return listed & listed.shift(-1, fill_value=False)

    def forward_returns(self, horizon: int = 1) -> pd.DataFrame:
        """Compounded return over months ``t+1 .. t+horizon`` of a position opened at ``t``.

        If the asset delists inside the window, the proceeds (which include the delisting
        return) sit in cash at zero return for the rest of the window. The value is
        missing where the asset is not investable at ``t`` or the window runs past the
        end of the sample; partial windows are never returned.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        growth = (1.0 + self.returns).where(self.listed, 1.0)
        total = pd.DataFrame(1.0, index=self.dates, columns=self.assets)
        for step in range(1, horizon + 1):
            total = total * growth.shift(-step)
        return (total - 1.0).where(self.investable)


def _check_grid(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise PanelError("the row index must be a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise PanelError("dates must be strictly increasing")
    if not frame.columns.is_unique:
        raise PanelError("asset identifiers must be unique")


def _check_one_spell(returns: pd.DataFrame) -> None:
    listed = returns.notna().to_numpy()
    n_dates = listed.shape[0]
    count = listed.sum(axis=0)
    first = listed.argmax(axis=0)
    last = n_dates - 1 - listed[::-1].argmax(axis=0)
    broken = (count > 0) & (last - first + 1 != count)
    if broken.any():
        names = list(returns.columns[broken][:5])
        raise PanelError(f"assets with a gap inside their listed spell: {names}")
