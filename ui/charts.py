"""
ui/charts.py – Plotly Chart Builders
======================================
Functions that build and return interactive Plotly figures for the
Quant Strategy Explorer app.  Each function is self-contained and
returns a ``plotly.graph_objects.Figure``.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_price_with_signals(df: pd.DataFrame, signal: pd.Series) -> go.Figure:
    """Build a candlestick / line chart overlaid with buy & sell markers.

    Parameters
    ----------
    df:
        OHLCV DataFrame with at least a ``Close`` column, indexed by date.
    signal:
        Binary signal series (1 = long, 0 = flat) aligned to *df*.

    Returns
    -------
    go.Figure
        Plotly figure with price line and buy/sell entry/exit markers.
    """
    close = df["Close"]

    # Detect entry (0→1) and exit (1→0) transitions
    prev_signal = signal.shift(1).fillna(0)
    entries = close[((signal == 1) & (prev_signal == 0))]
    exits = close[((signal == 0) & (prev_signal == 1))]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close,
            mode="lines",
            name="Close",
            line=dict(color="#1f77b4", width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=entries.index,
            y=entries,
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", color="green", size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=exits.index,
            y=exits,
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", color="red", size=10),
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified",
    )
    return fig


def plot_equity_curve(results: pd.DataFrame) -> go.Figure:
    """Build an equity curve chart comparing strategy vs. buy-and-hold.

    Parameters
    ----------
    results:
        DataFrame returned by :func:`backtest.engine.run_backtest`.
        Must contain columns ``equity``, ``daily_return``.

    Returns
    -------
    go.Figure
        Plotly figure with two lines: strategy equity and buy-and-hold.
    """
    bah_equity = (1 + results["daily_return"]).cumprod()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["equity"],
            mode="lines",
            name="Strategy",
            line=dict(color="#2ca02c", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=bah_equity,
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#aec7e8", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value (normalized)",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified",
    )
    return fig


def plot_drawdown(results: pd.DataFrame) -> go.Figure:
    """Build a filled area chart showing the strategy drawdown over time.

    Parameters
    ----------
    results:
        DataFrame returned by :func:`backtest.engine.run_backtest`.
        Must contain column ``drawdown``.

    Returns
    -------
    go.Figure
        Plotly figure with a shaded drawdown area.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["drawdown"] * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#d62728", width=1),
            fillcolor="rgba(214, 39, 40, 0.3)",
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".1f",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified",
    )
    return fig
