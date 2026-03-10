"""
data/data_loader.py – Price Data Loader
========================================
Handles downloading historical daily price data from Yahoo Finance via
the yfinance library.  Returns a clean pandas DataFrame indexed by date.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """Download daily OHLCV data for *ticker* between *start_date* and *end_date*.

    Parameters
    ----------
    ticker:
        A valid Yahoo Finance ticker symbol, e.g. ``"AAPL"`` or ``"SPY"``.
    start_date:
        ISO-8601 date string for the start of the period (inclusive), e.g.
        ``"2020-01-01"``.
    end_date:
        ISO-8601 date string for the end of the period (inclusive), e.g.
        ``"2024-12-31"``.

    Returns
    -------
    pd.DataFrame | None
        A DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume`` indexed by ``Date``, or ``None`` if the download fails.
    """
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
    except Exception:  # noqa: BLE001
        return None

    if raw.empty:
        return None

    # Flatten multi-level columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index.name = "Date"
    return raw
