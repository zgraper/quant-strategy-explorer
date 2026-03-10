"""
strategies/momentum.py – Momentum Strategy
==========================================
Generates long/flat signals based on trailing price returns.  The asset
is held long when its return over the lookback window exceeds a
configurable threshold, signalling positive momentum.
"""

from __future__ import annotations

import pandas as pd


def momentum_signals(close: pd.Series, params: dict) -> pd.Series:
    """Compute long/flat signals for the Momentum strategy.

    The strategy goes long when the *lookback*-day return is greater than
    or equal to ``min_return``, and exits (flat) when it falls below.

    Parameters
    ----------
    close:
        Daily closing prices as a pandas Series indexed by date.
    params:
        Dictionary of strategy parameters.  Expected keys:

        - ``"mom_window"`` (int): lookback window for return calculation.
        - ``"min_return"`` (float): minimum fractional return to trigger a
          long signal (e.g. ``0.02`` for 2 %).

    Returns
    -------
    pd.Series
        Integer signal series aligned to *close*: ``1`` = long, ``0`` = flat.
    """
    window: int = params.get("mom_window", 20)
    min_return: float = params.get("min_return", 0.02)

    trailing_return = close.pct_change(window)
    signal = (trailing_return >= min_return).astype(int)
    signal = signal.fillna(0)
    return signal
