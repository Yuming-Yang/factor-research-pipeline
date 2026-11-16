"""Point-in-time alignment of fundamentals.

A fundamental value describes a fiscal period but becomes public later. Dating it by the
period end lets a backtest trade on numbers nobody had yet. This module dates every value
by its publication date, and shows a restated value only from the day it was restated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("asset", "period_end", "publish_date", "value")


def lag_to_availability(
    fundamentals: pd.DataFrame,
    dates: pd.DatetimeIndex,
    assets: pd.Index,
    *,
    max_age_months: int | None = None,
) -> pd.DataFrame:
    """Return, for each date and asset, the latest value that was public on that date.

    ``fundamentals`` is a long table with one row per report: ``asset``, ``period_end``,
    ``publish_date`` and ``value``. A period may appear more than once; later rows for the
    same period are restatements.

    At date ``t`` the result is the value for the most recent period among the rows with
    ``publish_date <= t``. If that period was reported more than once by ``t``, the most
    recently published row wins. A late restatement of an older period does not replace
    a newer period that is already public.

    With ``max_age_months`` set, a value is dropped once its period end is more than that
    many months before ``t``, so a company that stops reporting does not keep a stale
    number forever.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in fundamentals.columns]
    if missing:
        raise ValueError(f"fundamentals is missing columns: {missing}")
    if (fundamentals["publish_date"] < fundamentals["period_end"]).any():
        raise ValueError("a value cannot be published before its period ends")

    out = pd.DataFrame(np.nan, index=dates, columns=assets)
    query = dates.to_numpy(dtype="datetime64[ns]")
    if max_age_months is not None:
        oldest_allowed = (dates - pd.DateOffset(months=max_age_months)).to_numpy(
            dtype="datetime64[ns]"
        )

    ordered = fundamentals.sort_values(["asset", "publish_date", "period_end"], kind="stable")
    for asset, rows in ordered.groupby("asset", sort=False):
        if asset not in out.columns:
            continue
        period = rows["period_end"].to_numpy(dtype="datetime64[ns]")
        # A row becomes the visible one only if no newer period was already public.
        current = period >= np.maximum.accumulate(period)
        period = period[current]
        published = rows["publish_date"].to_numpy(dtype="datetime64[ns]")[current]
        value = rows["value"].to_numpy(dtype=float)[current]

        position = np.searchsorted(published, query, side="right") - 1
        known = position >= 0
        picked = np.where(known, value[np.clip(position, 0, None)], np.nan)
        if max_age_months is not None:
            stale = known & (period[np.clip(position, 0, None)] < oldest_allowed)
            picked = np.where(stale, np.nan, picked)
        out[asset] = picked
    return out
