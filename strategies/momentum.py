"""
strategies/momentum.py – Momentum Strategy
==========================================
Generates long/flat signals based on trailing price returns.  The asset
is held long when its return over the lookback window is positive,
signalling positive momentum.
"""

from __future__ import annotations

import pandas as pd


def generate_momentum_signals(
    df: pd.DataFrame,
    lookback_window: int,
) -> pd.DataFrame:
    """Compute Momentum signals from a price DataFrame.

    Momentum is measured as the percent change in ``Close`` over
    *lookback_window* trading days.  The strategy is long when momentum
    is positive and flat otherwise.

    Parameters
    ----------
    df:
        DataFrame containing at least a ``Close`` column, indexed by date.
    lookback_window:
        Lookback period (in trading days) for the momentum calculation.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with three additional columns:

        - ``momentum``: percent change in ``Close`` over *lookback_window*
          bars (NaN for the first *lookback_window* rows).
        - ``signal``: ``1`` when ``momentum > 0``, ``0`` otherwise.  NaN
          rows in the warm-up period are filled with ``0``.
        - ``position_change``: first-difference of ``signal``; ``+1`` =
          entry, ``-1`` = exit, ``0`` = no change.
    """
    out = df.copy()

    out["momentum"] = out["Close"].pct_change(lookback_window)
    out["signal"] = (out["momentum"] > 0).astype(int)
    out["signal"] = out["signal"].fillna(0)
    out["position_change"] = out["signal"].diff().fillna(0).astype(int)

    return out


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
