"""
data/data_loader.py – Price Data Loader
========================================
Handles downloading historical daily price data from Yahoo Finance via
the yfinance library.  Returns a clean pandas DataFrame indexed by date.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

# Canonical column order; missing columns are silently omitted.
_COLUMN_ORDER = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


@st.cache_data(show_spinner=False)
def get_price_data(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Download and clean daily OHLCV data for *ticker*.

    Data is cached by Streamlit so repeated calls with the same arguments
    avoid redundant network requests.

    Parameters
    ----------
    ticker:
        A valid Yahoo Finance ticker symbol, e.g. ``"SPY"`` or ``"BTC-USD"``.
    start:
        ISO-8601 date string for the start of the period (inclusive), e.g.
        ``"2020-01-01"``.
    end:
        ISO-8601 date string for the end of the period (inclusive), e.g.
        ``"2024-12-31"``.

    Returns
    -------
    pd.DataFrame | None
        A DataFrame indexed by ``Date`` with columns
        ``Open``, ``High``, ``Low``, ``Close``, ``Adj Close`` (when available),
        ``Volume``, and ``Return`` (daily percentage change of ``Adj Close``
        when available, otherwise ``Close``).
        Returns ``None`` when the ticker is invalid or no data is available
        for the requested period.
    """
    try:
        # auto_adjust=False retains Adj Close alongside the raw OHLCV columns.
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
    except Exception:  # noqa: BLE001
        return None

    if raw is None or raw.empty:
        return None

    # Flatten multi-level columns that yfinance returns for single tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Standardize to canonical column names (keep only those present in data).
    available = [c for c in _COLUMN_ORDER if c in raw.columns]
    df = raw[available].copy()

    # Drop rows where the primary price signal is unavailable.
    df.dropna(subset=["Close"], inplace=True)

    if df.empty:
        return None

    # Daily percentage return based on adjusted close when present, else Close.
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["Return"] = df[price_col].pct_change()

    df.index.name = "Date"
    return df


def load_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """Thin wrapper around :func:`get_price_data` for backward compatibility.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol.
    start_date:
        ISO-8601 start date string.
    end_date:
        ISO-8601 end date string.

    Returns
    -------
    pd.DataFrame | None
        See :func:`get_price_data`.
    """
    return get_price_data(ticker, start_date, end_date)
