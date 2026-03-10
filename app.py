"""
app.py – Quant Strategy Explorer
=================================
Main Streamlit entry point. Orchestrates the sidebar, data loading,
strategy execution, backtesting, and chart rendering.

Flow
----
1. Render sidebar and collect user parameters.
2. Gate all computation behind the "Run Backtest" button.
3. Validate inputs (date range, strategy-specific constraints).
4. Load price data via yfinance.
5. Generate strategy signals.
6. Run the long-only backtest and compute performance metrics.
7. Display metrics, interactive Plotly charts, and an expandable results table.
"""

import streamlit as st

from data.data_loader import load_price_data
from ui.sidebar import render_sidebar
from ui.charts import (
    plot_price_with_signals,
    plot_equity_curve,
    plot_drawdown,
)
from backtest.engine import run_backtest
from backtest.metrics import calculate_metrics
from strategies.ma_crossover import ma_crossover_signals
from strategies.mean_reversion import mean_reversion_signals
from strategies.momentum import momentum_signals
from strategies.hmm_regime import hmm_regime_signals
from utils.helpers import validate_date_range

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Quant Strategy Explorer",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Title and project description
# ---------------------------------------------------------------------------

st.title("📈 Quant Strategy Explorer")
st.markdown(
    """
    Explore and backtest quantitative trading strategies on historical daily
    price data.  Select a ticker, date range, and strategy from the sidebar,
    then click **▶ Run Backtest** to view performance metrics, interactive
    charts, and a detailed results table.
    """
)

# ---------------------------------------------------------------------------
# Sidebar – returns all user-selected parameters as a dict
# ---------------------------------------------------------------------------

params = render_sidebar()

# ---------------------------------------------------------------------------
# Assumptions & limitations (always visible)
# ---------------------------------------------------------------------------

with st.expander("ℹ️ Assumptions & Limitations", expanded=False):
    st.markdown(
        """
        - **Daily data only** – signals and returns are computed on end-of-day
          close prices; intraday granularity is not supported in v1.
        - **Simple long / cash strategy** – the portfolio is either fully
          invested (signal = 1) or in cash (signal = 0).  No short selling
          or leverage is applied.
        - **No slippage (v1)** – trades are assumed to execute at the closing
          price with zero market-impact adjustment.
        - **Simplified transaction cost handling** – a flat fractional cost
          is subtracted whenever the position changes; defaults to zero in
          this demo.  Real-world costs (spread, market impact) are not
          modelled.
        - **HMM is a demonstration model** – the Hidden Markov Model is fit
          in-sample on a trailing window.  It is illustrative and has not
          been validated for live trading.
        - **No risk management** – position sizing is all-in; no stop-losses,
          position limits, or portfolio-level risk controls are applied.
        """
    )

# ---------------------------------------------------------------------------
# Guard: only execute when the user presses the button
# ---------------------------------------------------------------------------

if not params["run_backtest"]:
    st.info(
        "Configure your parameters in the sidebar and click "
        "**▶ Run Backtest** to begin."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

if not validate_date_range(params["start_date"], params["end_date"]):
    st.error(
        "⚠️ **Invalid date range** – start date must be strictly before end "
        "date.  Please adjust the date range in the sidebar."
    )
    st.stop()

# Moving Average Crossover: short window must be smaller than long window.
if params["strategy"] == "Moving Average Crossover":
    if params.get("short_window", 20) >= params.get("long_window", 50):
        st.error(
            "⚠️ **Invalid MA windows** – the Short MA window must be smaller "
            "than the Long MA window.  Please adjust the sliders."
        )
        st.stop()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

with st.spinner(f"Downloading price data for **{params['ticker']}**…"):
    df = load_price_data(params["ticker"], params["start_date"], params["end_date"])

if df is None or df.empty:
    st.error(
        f"⚠️ No data returned for **{params['ticker']}** between "
        f"**{params['start_date']}** and **{params['end_date']}**.  "
        "Please verify the ticker symbol and date range."
    )
    st.stop()

if len(df) < 30:
    st.warning(
        f"⚠️ Only **{len(df)}** trading days found.  Results may be "
        "unreliable with a very short history – consider extending the "
        "date range."
    )

st.success(
    f"Loaded **{len(df):,}** trading days for **{params['ticker']}** "
    f"({params['start_date']} → {params['end_date']})."
)

# ---------------------------------------------------------------------------
# Strategy signal generation
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "Moving Average Crossover": ma_crossover_signals,
    "Mean Reversion": mean_reversion_signals,
    "Momentum": momentum_signals,
    "HMM Regime": hmm_regime_signals,
}

strategy_fn = STRATEGY_MAP[params["strategy"]]

with st.spinner(f"Generating **{params['strategy']}** signals…"):
    signals = strategy_fn(df["Close"], params)

# Sanitise signal: replace NaN (warm-up period) with 0 (cash) and cast to int.
signals = signals.fillna(0).astype(int)

if signals.sum() == 0:
    st.warning(
        "⚠️ The strategy produced **no buy signals** for the selected "
        "parameters and date range.  All performance metrics will reflect "
        "a cash position.  Try adjusting the strategy parameters or "
        "extending the date range."
    )

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

df_with_signal = df.copy()
df_with_signal["signal"] = signals

results = run_backtest(df_with_signal)
metrics = calculate_metrics(results)

# ---------------------------------------------------------------------------
# Metrics display – shown near the top before the charts
# ---------------------------------------------------------------------------

st.subheader("📊 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
col5, col6, col7 = st.columns(3)

col1.metric("Total Return", f"{metrics['total_return']:.1%}")
col2.metric("Ann. Return", f"{metrics['annualized_return']:.1%}")
col3.metric("Ann. Volatility", f"{metrics['annualized_volatility']:.1%}")
col4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
col5.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
col6.metric("Win Rate", f"{metrics['win_rate']:.1%}")
col7.metric("# Trades", f"{metrics['number_of_trades']:,}")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.subheader("📈 Price & Signals")
st.plotly_chart(plot_price_with_signals(results), use_container_width=True)

st.subheader("💹 Equity Curve")
st.plotly_chart(plot_equity_curve(results), use_container_width=True)

st.subheader("📉 Drawdown")
st.plotly_chart(plot_drawdown(results), use_container_width=True)

# ---------------------------------------------------------------------------
# Expandable results dataframe
# ---------------------------------------------------------------------------

with st.expander("🗃️ View Full Results Data", expanded=False):
    # Select the columns most useful for manual inspection.
    display_cols = [
        col
        for col in [
            "Close",
            "signal",
            "market_return",
            "strategy_return",
            "cumulative_market",
            "cumulative_strategy",
            "drawdown",
        ]
        if col in results.columns
    ]
    display_df = results[display_cols].copy()

    # Format percentage columns for readability.
    for pct_col in ("market_return", "strategy_return", "drawdown"):
        if pct_col in display_df.columns:
            display_df[pct_col] = display_df[pct_col].map("{:.2%}".format)

    # Format cumulative-return columns to 4 decimal places.
    for cum_col in ("cumulative_market", "cumulative_strategy"):
        if cum_col in display_df.columns:
            display_df[cum_col] = display_df[cum_col].map("{:.4f}".format)

    st.dataframe(display_df, use_container_width=True)
    st.caption(
        f"Showing {len(display_df):,} rows · signal: 1 = long, 0 = cash"
    )
