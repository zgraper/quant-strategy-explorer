"""
strategies/ma_crossover.py – Moving Average Crossover Strategy
===============================================================
Generates long/flat signals based on a short vs. long simple moving
average (SMA) crossover.  A buy signal (1) is produced when the short SMA
is above the long SMA; a sell/flat signal (0) when it falls below.
"""

from __future__ import annotations

import pandas as pd


def generate_ma_signals(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """Compute Moving Average Crossover signals from a price DataFrame.

    Signals are determined at the *close* of each bar.  Any signal produced
    today should be acted on at the *next* bar's open in a live or backtest
    context (no lookahead bias is introduced here).

    Parameters
    ----------
    df:
        DataFrame containing at least a ``Close`` column, indexed by date.
    short_window:
        Lookback period (in trading days) for the short-term SMA.
    long_window:
        Lookback period (in trading days) for the long-term SMA.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with four additional columns:

        - ``short_ma``: rolling mean of ``Close`` over *short_window* bars.
        - ``long_ma``: rolling mean of ``Close`` over *long_window* bars.
        - ``signal``: ``1`` when ``short_ma > long_ma`` (long), ``0`` otherwise
          (cash / flat).  NaN values in the warm-up period are filled with
          ``0``.
        - ``position_change``: first-difference of ``signal``; ``+1`` marks an
          entry, ``-1`` an exit, ``0`` no change.
    """
    out = df.copy()

    # Compute simple moving averages
    out["short_ma"] = out["Close"].rolling(short_window).mean()
    out["long_ma"] = out["Close"].rolling(long_window).mean()

    # Long when short MA is above long MA, otherwise flat
    out["signal"] = (out["short_ma"] > out["long_ma"]).astype(int)
    out["signal"] = out["signal"].fillna(0)

    # Detect entries (+1) and exits (-1)
    out["position_change"] = out["signal"].diff().fillna(0).astype(int)

    return out


def ma_crossover_signals(close: pd.Series, params: dict) -> pd.Series:
    """Compute long/flat signals for the Moving Average Crossover strategy.

    Thin wrapper around :func:`generate_ma_signals` that accepts the
    app-level ``params`` dictionary and returns a plain ``pd.Series`` signal.

    Parameters
    ----------
    close:
        Daily closing prices as a pandas Series indexed by date.
    params:
        Dictionary of strategy parameters.  Expected keys:

        - ``"short_window"`` (int): lookback period for the short SMA.
        - ``"long_window"`` (int): lookback period for the long SMA.

    Returns
    -------
    pd.Series
        Integer signal series aligned to *close*: ``1`` = long, ``0`` = flat.
        NaN values at the start (warm-up period) are filled with ``0``.
    """
    short_window: int = params.get("short_window", 20)
    long_window: int = params.get("long_window", 50)

    df_input = pd.DataFrame({"Close": close})
    result = generate_ma_signals(df_input, short_window, long_window)
    return result["signal"]
