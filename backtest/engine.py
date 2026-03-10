"""
backtest/engine.py – Long-Only Backtest Engine
===============================================
Runs a simple daily long-only backtest given a price series and a binary
signal series (1 = long, 0 = flat).  Returns a DataFrame containing
daily portfolio values and related columns for further analysis.

Assumptions
-----------
- Daily close prices; signals act on the *next* day's close.
- Long-only: no short positions.
- No transaction costs or slippage.
- Starting capital is 1.0 (all calculations are in normalized units).
"""

from __future__ import annotations

import pandas as pd


def run_backtest(close: pd.Series, signal: pd.Series) -> pd.DataFrame:
    """Execute a long-only backtest and return a results DataFrame.

    The position is entered/exited at the closing price of the bar *after*
    the signal is generated (i.e. ``signal.shift(1)``).

    Parameters
    ----------
    close:
        Daily closing prices as a pandas Series indexed by date.
    signal:
        Binary signal series aligned to *close*: ``1`` = long, ``0`` = flat.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns:

        - ``close``: original closing prices.
        - ``signal``: raw signal (same-day).
        - ``position``: lagged signal applied to returns.
        - ``daily_return``: daily percentage return of the underlying.
        - ``strategy_return``: daily return of the strategy.
        - ``equity``: cumulative strategy equity (starts at 1.0).
        - ``drawdown``: rolling drawdown (negative values, fraction).
    """
    results = pd.DataFrame(index=close.index)
    results["close"] = close
    results["signal"] = signal.reindex(close.index).fillna(0).astype(int)

    # Position is entered the day after the signal (avoid lookahead bias)
    results["position"] = results["signal"].shift(1).fillna(0).astype(int)

    results["daily_return"] = close.pct_change().fillna(0.0)
    results["strategy_return"] = results["position"] * results["daily_return"]

    results["equity"] = (1.0 + results["strategy_return"]).cumprod()

    # Drawdown relative to running peak
    rolling_peak = results["equity"].cummax()
    results["drawdown"] = (results["equity"] - rolling_peak) / rolling_peak

    return results
