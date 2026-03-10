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


def compute_metrics(results: pd.DataFrame) -> dict:
    """Calculate key performance metrics from a backtest results DataFrame.

    Parameters
    ----------
    results:
        DataFrame returned by :func:`backtest.engine.run_backtest`.  Must
        contain columns ``equity``, ``strategy_return``, ``position``,
        and ``drawdown``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``total_return`` (float): overall fractional return.
        - ``annualized_return`` (float): CAGR over the full period.
        - ``sharpe_ratio`` (float): annualized Sharpe (risk-free rate = 0).
        - ``max_drawdown`` (float): worst peak-to-trough drawdown (negative).
        - ``win_rate`` (float): fraction of trades that were profitable.
        - ``num_trades`` (int): total number of completed round-trip trades.
    """
    equity = results["equity"]
    strategy_return = results["strategy_return"]
    position = results["position"]

    # --- Total return -------------------------------------------------------
    total_return = float(equity.iloc[-1] - 1.0)

    # --- Annualized return (CAGR) --------------------------------------------
    n_days = len(results)
    n_years = n_days / TRADING_DAYS_PER_YEAR
    if n_years > 0 and equity.iloc[-1] > 0:
        annualized_return = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)
    else:
        annualized_return = 0.0

    # --- Sharpe ratio -------------------------------------------------------
    daily_std = strategy_return.std()
    if daily_std > 0:
        sharpe_ratio = float(
            strategy_return.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe_ratio = 0.0

    # --- Max drawdown -------------------------------------------------------
    max_drawdown = float(results["drawdown"].min())

    # --- Trade statistics ---------------------------------------------------
    num_trades, win_rate = _trade_stats(position, strategy_return)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "num_trades": num_trades,
    }


def _trade_stats(position: pd.Series, strategy_return: pd.Series) -> tuple[int, float]:
    """Count completed trades and compute win rate.

    A trade starts when ``position`` flips from 0 → 1 and ends when it
    flips from 1 → 0.  The trade P&L is the cumulative return during
    the holding period.

    Parameters
    ----------
    position:
        Daily position series (0 or 1).
    strategy_return:
        Daily strategy return series aligned to *position*.

    Returns
    -------
    tuple[int, float]
        ``(num_trades, win_rate)`` where *win_rate* is 0.0 when there are
        no completed trades.
    """
    trades: list[float] = []
    in_trade = False
    trade_returns: list[float] = []

    for pos, ret in zip(position, strategy_return):
        if not in_trade and pos == 1:
            in_trade = True
            trade_returns = []
        if in_trade:
            trade_returns.append(ret)
        if in_trade and pos == 0:
            in_trade = False
            # Cumulative trade return
            cum = float((1 + pd.Series(trade_returns)).prod() - 1)
            trades.append(cum)

    num_trades = len(trades)
    win_rate = float(sum(t > 0 for t in trades) / num_trades) if num_trades else 0.0
    return num_trades, win_rate
