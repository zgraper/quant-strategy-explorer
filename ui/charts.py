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

# Shared palette aligned with the app's dark-blue minimalist theme
_DARK_BLUE = "#1B3A6B"
_LIGHT_BLUE = "#4A90D9"
_GRID_COLOR = "#E8ECF0"
_BG_COLOR = "#FFFFFF"
_FONT_COLOR = "#2D3748"

_BASE_LAYOUT = dict(
    paper_bgcolor=_BG_COLOR,
    plot_bgcolor=_BG_COLOR,
    font=dict(family="Inter, sans-serif", color=_FONT_COLOR, size=12),
    margin=dict(l=48, r=24, t=32, b=48),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False, linecolor=_GRID_COLOR, zeroline=False),
    yaxis=dict(
        gridcolor=_GRID_COLOR,
        linecolor=_GRID_COLOR,
        zeroline=False,
        gridwidth=1,
    ),
)


def _apply_base(fig: go.Figure, **extra) -> go.Figure:
    layout = {**_BASE_LAYOUT, **extra}
    fig.update_layout(**layout)
    return fig


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
            line=dict(color=_DARK_BLUE, width=1.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys,
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", color="#27AE60", size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells,
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", color="#E74C3C", size=10),
        )
    )

    return _apply_base(fig, xaxis_title="Date", yaxis_title="Price")


def plot_equity_curve(df: pd.DataFrame, show_benchmark: bool = True) -> go.Figure:
    """Build an equity curve chart comparing strategy vs. buy-and-hold.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`backtest.engine.run_backtest`.
        Must contain columns ``cumulative_strategy`` and ``cumulative_market``.
    show_benchmark:
        When ``True`` (default) the buy-and-hold equity curve is overlaid.

    Returns
    -------
    go.Figure
        Plotly figure with strategy equity and, optionally, buy-and-hold.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["cumulative_strategy"],
            mode="lines",
            name="Strategy",
            line=dict(color=_DARK_BLUE, width=2),
        )
    )

    if show_benchmark:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["cumulative_market"],
                mode="lines",
                name="Buy & Hold",
                line=dict(color=_LIGHT_BLUE, width=1.5, dash="dash"),
            )
        )

    return _apply_base(
        fig,
        xaxis_title="Date",
        yaxis_title="Portfolio Value (normalized)",
    )


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
            line=dict(color="#E74C3C", width=1),
            fillcolor="rgba(231, 76, 60, 0.15)",
        )
    )

    return _apply_base(
        fig,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".1f",
    )
