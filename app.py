"""
app.py – Quant Strategy Explorer
=================================
Main Streamlit entry point. Orchestrates the sidebar, data loading,
strategy execution, backtesting, and chart rendering.
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Quant Strategy Explorer",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar – returns all user-selected parameters as a dict
# ---------------------------------------------------------------------------

params = render_sidebar()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

st.title("📈 Quant Strategy Explorer")

with st.spinner("Downloading price data…"):
    df = load_price_data(params["ticker"], params["start_date"], params["end_date"])

if df is None or df.empty:
    st.error("No data returned. Please check the ticker symbol and date range.")
    st.stop()

st.success(f"Loaded {len(df)} trading days for **{params['ticker']}**.")

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
signals = strategy_fn(df["Close"], params)

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

df_with_signal = df.copy()
df_with_signal["signal"] = signals

results = run_backtest(df_with_signal)
metrics = calculate_metrics(results)

# ---------------------------------------------------------------------------
# Metrics display
# ---------------------------------------------------------------------------

st.subheader("Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
col5, col6, col7 = st.columns(3)

col1.metric("Total Return", f"{metrics['total_return']:.1%}")
col2.metric("Ann. Return", f"{metrics['annualized_return']:.1%}")
col3.metric("Ann. Volatility", f"{metrics['annualized_volatility']:.1%}")
col4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
col5.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
col6.metric("Win Rate", f"{metrics['win_rate']:.1%}")
col7.metric("# Trades", str(metrics["number_of_trades"]))

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.subheader("Price & Signals")
st.plotly_chart(plot_price_with_signals(results), use_container_width=True)

st.subheader("Equity Curve")
st.plotly_chart(plot_equity_curve(results), use_container_width=True)

st.subheader("Drawdown")
st.plotly_chart(plot_drawdown(results), use_container_width=True)
