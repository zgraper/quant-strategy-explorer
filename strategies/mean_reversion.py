"""
strategies/mean_reversion.py – Mean Reversion Strategy
=======================================================
Generates long/flat signals using a z-score of the closing price
relative to a rolling mean and standard deviation.  A buy signal is
issued when the price dips below the mean by a configurable threshold
(oversold), and exits when it reverts back above.
"""

from __future__ import annotations

import pandas as pd


def mean_reversion_signals(close: pd.Series, params: dict) -> pd.Series:
    """Compute long/flat signals for the Mean Reversion strategy.

    The strategy buys when the rolling z-score falls below
    ``-entry_z`` (price is "cheap") and exits once the z-score climbs
    back above ``-exit_z``.

    Parameters
    ----------
    close:
        Daily closing prices as a pandas Series indexed by date.
    params:
        Dictionary of strategy parameters.  Expected keys:

        - ``"mr_window"`` (int): rolling lookback window for mean/std.
        - ``"entry_z"`` (float): z-score threshold to enter (buy).
        - ``"exit_z"`` (float): z-score threshold to exit (sell/flat).

    Returns
    -------
    pd.Series
        Integer signal series aligned to *close*: ``1`` = long, ``0`` = flat.
    """
    window: int = params.get("lookback_window", 20)
    entry_z: float = params.get("zscore_threshold", -1.5)
    exit_z: float = 0.0  # exit when z-score reverts to mean

    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()
    z_score = (close - rolling_mean) / rolling_std

    signal = pd.Series(0, index=close.index)
    in_position = False

    for i in range(len(z_score)):
        z = z_score.iloc[i]
        if pd.isna(z):
            continue
        if not in_position and z <= entry_z:
            in_position = True
        elif in_position and z >= exit_z:
            in_position = False
        signal.iloc[i] = int(in_position)

    return signal
