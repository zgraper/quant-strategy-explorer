"""
strategies/mean_reversion.py – Mean Reversion Strategy
=======================================================
Generates long/flat signals using a z-score of the closing price
relative to a rolling mean and standard deviation.  A buy signal is
issued when the price dips below the mean by a configurable threshold
(oversold), and exits when it reverts back above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_mean_reversion_signals(
    df: pd.DataFrame,
    lookback_window: int,
    zscore_threshold: float,
) -> pd.DataFrame:
    """Compute Mean Reversion signals from a price DataFrame.

    Parameters
    ----------
    df:
        DataFrame containing at least a ``Close`` column, indexed by date.
    lookback_window:
        Rolling window (in trading days) used to compute the mean and
        standard deviation.
    zscore_threshold:
        Positive threshold value.  A long position is entered when the
        z-score falls *below* ``-zscore_threshold`` (oversold) and exited
        when the z-score climbs back to ``>= 0`` (reverted to mean).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with five additional columns:

        - ``rolling_mean``: rolling mean of ``Close`` over *lookback_window*.
        - ``rolling_std``: rolling std of ``Close`` over *lookback_window*.
        - ``zscore``: ``(Close - rolling_mean) / rolling_std``; NaN rows in
          the warm-up period and rows where std equals zero are left as NaN.
        - ``signal``: ``1`` = long, ``0`` = flat (cash).
        - ``position_change``: first-difference of ``signal``; ``+1`` = entry,
          ``-1`` = exit, ``0`` = no change.
    """
    out = df.copy()

    out["rolling_mean"] = out["Close"].rolling(lookback_window).mean()
    out["rolling_std"] = out["Close"].rolling(lookback_window).std()

    # Avoid division by zero: where std == 0 the z-score is undefined (NaN)
    out["zscore"] = (out["Close"] - out["rolling_mean"]) / out["rolling_std"].replace(
        0, float("nan")
    )

    # State-machine signal generation: iterates over the z-score array using
    # raw numpy values to avoid per-row pandas overhead.
    zscore_arr = out["zscore"].to_numpy()
    signal_arr = np.zeros(len(zscore_arr), dtype=int)
    in_position = False

    for i, z in enumerate(zscore_arr):
        if np.isnan(z):
            continue
        if not in_position and z < -zscore_threshold:
            in_position = True
        elif in_position and z >= 0:
            in_position = False
        signal_arr[i] = int(in_position)

    signal = pd.Series(signal_arr, index=out.index)

    out["signal"] = signal
    out["position_change"] = out["signal"].diff().fillna(0).astype(int)

    return out


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
