"""
backtest/metrics.py – Performance Metrics
==========================================
Computes summary statistics from the results DataFrame produced by
``backtest.engine.run_backtest``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate key performance metrics from a backtest results DataFrame.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`backtest.engine.run_backtest`.  Must
        contain columns ``strategy_return``, ``drawdown``, and
        ``position_change``.

    Returns
    -------
    dict
        Dictionary with the following keys (values rounded for display):

        - ``total_return`` (float): overall fractional return.
        - ``annualized_return`` (float): CAGR over the full period.
        - ``annualized_volatility`` (float): annualized std of daily returns.
        - ``sharpe_ratio`` (float): annualized Sharpe (risk-free rate = 0).
        - ``max_drawdown`` (float): worst peak-to-trough drawdown (negative).
        - ``win_rate`` (float): fraction of non-zero-return days that are positive.
        - ``number_of_trades`` (int): number of position changes.
    """
    if df is None or df.empty:
        return _empty_metrics()

    try:
        strategy_return = df["strategy_return"]
        drawdown = df["drawdown"]
        position_change = (
            df["position_change"]
            if "position_change" in df.columns
            else pd.Series([], dtype=float)
        )
    except (KeyError, AttributeError):
        return _empty_metrics()

    if strategy_return.empty:
        return _empty_metrics()

    # --- Total return -------------------------------------------------------
    total_return = round(float((1.0 + strategy_return).prod() - 1.0), 4)

    # --- Annualized return (CAGR) --------------------------------------------
    n_days = len(strategy_return)
    n_years = n_days / TRADING_DAYS_PER_YEAR
    final_value = 1.0 + total_return
    if n_years > 0 and final_value > 0:
        annualized_return = round(float(final_value ** (1.0 / n_years) - 1.0), 4)
    else:
        annualized_return = 0.0

    # --- Annualized volatility -----------------------------------------------
    daily_std = strategy_return.std()
    annualized_volatility = round(
        float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if daily_std > 0 else 0.0,
        4,
    )

    # --- Sharpe ratio -------------------------------------------------------
    if daily_std > 0:
        sharpe_ratio = round(
            float(strategy_return.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)),
            2,
        )
    else:
        sharpe_ratio = 0.0

    # --- Max drawdown -------------------------------------------------------
    max_drawdown = round(float(drawdown.min()), 4) if not drawdown.empty else 0.0

    # --- Win rate (days with nonzero strategy_return) -----------------------
    active_days = strategy_return[strategy_return != 0.0]
    win_rate = (
        round(float((active_days > 0).sum() / len(active_days)), 4)
        if len(active_days) > 0
        else 0.0
    )

    # --- Number of trades (position changes) --------------------------------
    number_of_trades = (
        int((position_change != 0).sum()) if not position_change.empty else 0
    )

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "number_of_trades": number_of_trades,
    }


def _empty_metrics() -> dict:
    """Return a zeroed metrics dictionary for invalid or empty inputs."""
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "number_of_trades": 0,
    }
