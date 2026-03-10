"""
backtest/engine.py – Long-Only Backtest Engine
===============================================
Runs a simple daily long-only backtest given a DataFrame containing
Close prices and a binary signal (1 = invested, 0 = cash).  Returns the
DataFrame with additional columns for daily returns, cumulative returns,
equity curve, and drawdown.

Assumptions
-----------
- Daily close prices; signal from day t-1 is applied to day t's return.
- Long-only: no short positions.
- Transaction cost is subtracted whenever the position changes.
- Starting equity equals ``initial_capital``.
- 252 trading days per year.
"""

from __future__ import annotations

import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Execute a long-only backtest and return a results DataFrame.

    The position on day t is determined by the signal on day t-1 to avoid
    lookahead bias.

    Parameters
    ----------
    df:
        DataFrame containing at least:

        - ``Close``: daily closing prices.
        - ``signal``: binary signal (1 = invested, 0 = cash).
    initial_capital:
        Starting portfolio value in dollars (default 10,000).
    transaction_cost:
        Fractional one-way transaction cost, e.g. 0.001 = 0.1 %.
        Applied whenever the position changes.

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with columns:

        - ``market_return``: daily percentage return of the underlying.
        - ``position_change``: day-over-day change in the shifted signal.
        - ``strategy_return``: daily return of the strategy after costs.
        - ``cumulative_market``: cumulative buy-and-hold return (starts at 1).
        - ``cumulative_strategy``: cumulative strategy return (starts at 1).
        - ``equity_curve``: strategy equity in dollar terms.
        - ``drawdown``: rolling drawdown as a fraction (negative values).
    """
    out = df.copy()

    # Market return from Close percentage change
    out["market_return"] = out["Close"].pct_change().fillna(0.0)

    # Shift signal by 1 day to avoid lookahead bias
    shifted_signal = out["signal"].shift(1).fillna(0).astype(int)

    # Position change for transaction cost calculation
    out["position_change"] = shifted_signal.diff().fillna(0)

    # Strategy return = shifted signal * market return
    out["strategy_return"] = shifted_signal * out["market_return"]

    # Subtract transaction cost whenever position changes
    if transaction_cost > 0.0:
        out["strategy_return"] -= out["position_change"].abs() * transaction_cost

    # Cumulative returns (normalized, starting at 1)
    out["cumulative_market"] = (1.0 + out["market_return"]).cumprod()
    out["cumulative_strategy"] = (1.0 + out["strategy_return"]).cumprod()

    # Equity curve in dollar terms
    out["equity_curve"] = out["cumulative_strategy"] * initial_capital

    # Drawdown relative to running peak of cumulative strategy
    rolling_peak = out["cumulative_strategy"].cummax()
    out["drawdown"] = (out["cumulative_strategy"] - rolling_peak) / rolling_peak

    return out
