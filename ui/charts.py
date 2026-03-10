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


def plot_price_with_signals(df: pd.DataFrame) -> go.Figure:
    """Build a price chart overlaid with buy and sell markers.

    Buy markers appear where ``position_change > 0``.
    Sell markers appear where ``position_change < 0``.

    Parameters
    ----------
    df:
        DataFrame containing at least ``Close`` and ``position_change``
        columns, indexed by date.

    Returns
    -------
    go.Figure
        Plotly figure with price line and buy/sell entry/exit markers.
    """
    close = df["Close"]
    position_change = df.get("position_change", pd.Series(dtype=float))

    buys = close[position_change > 0] if not position_change.empty else pd.Series(dtype=float)
    sells = close[position_change < 0] if not position_change.empty else pd.Series(dtype=float)

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
            x=buys.index,
            y=buys,
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", color="green", size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells,
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


def plot_equity_curve(df: pd.DataFrame) -> go.Figure:
    """Build an equity curve chart comparing strategy vs. buy-and-hold.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`backtest.engine.run_backtest`.
        Must contain columns ``cumulative_strategy`` and ``cumulative_market``.

    Returns
    -------
    go.Figure
        Plotly figure with two lines: strategy equity and buy-and-hold.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["cumulative_strategy"],
            mode="lines",
            name="Strategy",
            line=dict(color="#2ca02c", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["cumulative_market"],
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


def plot_drawdown(df: pd.DataFrame) -> go.Figure:
    """Build a filled area chart showing the strategy drawdown over time.

    Parameters
    ----------
    df:
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
            x=df.index,
            y=df["drawdown"] * 100,
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
