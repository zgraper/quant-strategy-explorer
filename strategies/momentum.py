"""
strategies/momentum.py – Momentum Strategy
==========================================
Generates long/flat signals based on trailing price returns.  The asset
is held long when its return over the lookback window is positive,
signalling positive momentum.
"""

from __future__ import annotations

import pandas as pd


def momentum_signals(close: pd.Series, params: dict) -> pd.Series:
    """Compute long/flat signals for the Momentum strategy.

    The strategy goes long when the *lookback_window*-day return is positive
    and exits (flat) otherwise.

    Parameters
    ----------
    close:
        Daily closing prices as a pandas Series indexed by date.
    params:
        Dictionary of strategy parameters.  Expected keys:

        - ``"lookback_window"`` (int): lookback window for return calculation.

    Returns
    -------
    pd.Series
        Integer signal series aligned to *close*: ``1`` = long, ``0`` = flat.
    """
    window: int = params.get("lookback_window", 20)

    trailing_return = close.pct_change(window)
    signal = (trailing_return > 0).astype(int)
    signal = signal.fillna(0)
    return signal
