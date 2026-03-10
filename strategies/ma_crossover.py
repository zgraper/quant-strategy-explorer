"""
strategies/ma_crossover.py – Moving Average Crossover Strategy
===============================================================
Generates long/flat signals based on a fast vs. slow simple moving
average (SMA) crossover.  A buy signal (1) is produced when the fast SMA
crosses above the slow SMA; a sell/flat signal (0) when it crosses below.
"""

from __future__ import annotations

import pandas as pd


def ma_crossover_signals(close: pd.Series, params: dict) -> pd.Series:
    """Compute long/flat signals for the Moving Average Crossover strategy.

    Parameters
    ----------
    close:
        Daily closing prices as a pandas Series indexed by date.
    params:
        Dictionary of strategy parameters.  Expected keys:

        - ``"fast_window"`` (int): lookback period for the fast SMA.
        - ``"slow_window"`` (int): lookback period for the slow SMA.

    Returns
    -------
    pd.Series
        Integer signal series aligned to *close*: ``1`` = long, ``0`` = flat.
        NaN values at the start (warm-up period) are filled with ``0``.
    """
    fast_window: int = params.get("fast_window", 20)
    slow_window: int = params.get("slow_window", 50)

    fast_ma = close.rolling(fast_window).mean()
    slow_ma = close.rolling(slow_window).mean()

    # Long when fast MA is above slow MA
    signal = (fast_ma > slow_ma).astype(int)
    signal = signal.fillna(0)
    return signal
