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

import io
import os
import sys

# Ensure the repository root is in the Python path so that local packages
# (data, strategies, backtest, ui, utils) are importable on all platforms,
# including Streamlit Community Cloud.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
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
# Custom CSS – white minimalist theme with dark-blue accents
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F7F9FC;
        color: #2D3748;
        font-family: "Inter", sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* ── Card tile helper ────────────────────────────────────── */
    .card {
        background: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(27,58,107,0.08);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Section headings ────────────────────────────────────── */
    .section-header {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #1B3A6B;
        border-bottom: 2px solid #1B3A6B;
        padding-bottom: 0.35rem;
        margin-bottom: 1rem;
    }

    /* ── Metric tiles ────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(27,58,107,0.07);
        padding: 0.75rem 1rem !important;
    }
    [data-testid="metric-container"] label {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
        color: #718096 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1B3A6B !important;
    }

    /* ── Plotly chart container ──────────────────────────────── */
    [data-testid="stPlotlyChart"] {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(27,58,107,0.07);
        overflow: hidden;
        background: #FFFFFF;
    }

    /* ── Disclaimer banner ───────────────────────────────────── */
    .disclaimer {
        background: #EBF0F8;
        border-left: 4px solid #1B3A6B;
        border-radius: 6px;
        padding: 0.65rem 1rem;
        font-size: 0.8rem;
        color: #4A5568;
        margin-top: 1.5rem;
    }

    /* ── Strategy summary box ────────────────────────────────── */
    .strategy-summary {
        background: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(27,58,107,0.07);
        padding: 1rem 1.25rem;
        font-size: 0.88rem;
        color: #4A5568;
        line-height: 1.6;
    }

    /* ── Download button ─────────────────────────────────────── */
    [data-testid="stDownloadButton"] > button {
        background-color: #1B3A6B;
        color: #FFFFFF;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 0.45rem 1.1rem;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background-color: #14305A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Strategy plain-English descriptions
# ---------------------------------------------------------------------------

STRATEGY_DESCRIPTIONS = {
    "Moving Average Crossover": (
        "Goes **long** when the short-term moving average rises above the long-term "
        "moving average, and moves to **cash** when it falls below. Designed to capture "
        "sustained directional trends while avoiding choppy, sideways markets."
    ),
    "Mean Reversion": (
        "Buys when the price drops significantly below its recent average — as measured "
        "by a negative z-score — then returns to **cash** once the price normalises. "
        "Built on the statistical tendency of assets to revert toward their historical mean."
    ),
    "Momentum": (
        "Goes **long** when the asset's total return over the lookback window is positive "
        "(recent trend is up), and moves to **cash** when it turns negative. Exploits the "
        "empirical observation that recent winners tend to continue outperforming."
    ),
    "HMM Regime": (
        "Uses a Hidden Markov Model to classify the market into hidden regimes "
        "(e.g. bull / bear) based on recent returns. Goes **long** during the regime "
        "associated with higher expected returns and moves to **cash** otherwise. "
        "Illustrative — fit in-sample on a trailing window only."
    ),
}

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.markdown("<h1 style='color:#1B3A6B;font-weight:800;'>📈 Quant Strategy Explorer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#718096;font-size:0.95rem;margin-top:-0.5rem;'>"
    "Select a ticker, date range, and strategy in the sidebar, then click "
    "<strong>▶ Run Backtest</strong> to analyse performance.</p>",
    unsafe_allow_html=True,
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
          close prices; intraday granularity is not supported.
        - **Simple long / cash strategy** – the portfolio is either fully
          invested (signal = 1) or in cash (signal = 0).  No short selling
          or leverage is applied.
        - **Transaction cost** – a flat fractional cost is subtracted whenever
          the position changes.  Real-world costs (spread, market impact) are
          not modelled.
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

if params["strategy"] == "Moving Average Crossover":
    if params.get("short_window", 20) >= params.get("long_window", 50):
        st.error(
            "⚠️ **Invalid MA windows** – the Short MA window must be smaller "
            "than the Long MA window.  Please adjust the sliders."
        )
        st.stop()

# ---------------------------------------------------------------------------
# ── SECTION: Data ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>Data</div>", unsafe_allow_html=True)

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
        "unreliable with a very short history – consider extending the date range."
    )

st.success(
    f"Loaded **{len(df):,}** trading days for **{params['ticker']}** "
    f"({params['start_date']} → {params['end_date']})."
)

# ---------------------------------------------------------------------------
# ── SECTION: Strategy ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>Strategy Signals</div>", unsafe_allow_html=True)

# Strategy summary box
st.markdown(
    f"<div class='strategy-summary'><strong>{params['strategy']}</strong> — "
    f"{STRATEGY_DESCRIPTIONS[params['strategy']]}</div>",
    unsafe_allow_html=True,
)

STRATEGY_MAP = {
    "Moving Average Crossover": ma_crossover_signals,
    "Mean Reversion": mean_reversion_signals,
    "Momentum": momentum_signals,
    "HMM Regime": hmm_regime_signals,
}

strategy_fn = STRATEGY_MAP[params["strategy"]]

with st.spinner(f"Generating **{params['strategy']}** signals…"):
    signals = strategy_fn(df["Close"], params)

signals = signals.fillna(0).astype(int)

if signals.sum() == 0:
    st.warning(
        "⚠️ The strategy produced **no buy signals** for the selected "
        "parameters and date range.  All metrics will reflect a cash position.  "
        "Try adjusting parameters or extending the date range."
    )

df_with_signal = df.copy()
df_with_signal["signal"] = signals

st.plotly_chart(plot_price_with_signals(df_with_signal), use_container_width=True)

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

results = run_backtest(
    df_with_signal,
    transaction_cost=params.get("transaction_cost", 0.0),
)
metrics = calculate_metrics(results)

# ---------------------------------------------------------------------------
# ── SECTION: Performance ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>Performance</div>", unsafe_allow_html=True)

# — Metrics row —
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Total Return",      f"{metrics['total_return']:.1%}")
col2.metric("Ann. Return",       f"{metrics['annualized_return']:.1%}")
col3.metric("Ann. Volatility",   f"{metrics['annualized_volatility']:.1%}")
col4.metric("Sharpe Ratio",      f"{metrics['sharpe_ratio']:.2f}")
col5.metric("Max Drawdown",      f"{metrics['max_drawdown']:.1%}")
col6.metric("Win Rate",          f"{metrics['win_rate']:.1%}")
col7.metric("# Trades",          f"{metrics['number_of_trades']:,}")

# — Equity curve —
st.plotly_chart(
    plot_equity_curve(results, show_benchmark=params.get("show_benchmark", True)),
    use_container_width=True,
)

# — Drawdown —
st.plotly_chart(plot_drawdown(results), use_container_width=True)

# — Parameter summary —
with st.expander("⚙️ Backtest Parameter Summary", expanded=False):
    summary_rows = {
        "Ticker": params["ticker"],
        "Start Date": params["start_date"],
        "End Date": params["end_date"],
        "Strategy": params["strategy"],
        "Transaction Cost (one-way)": f"{params.get('transaction_cost', 0.0):.3%}",
        "Benchmark Shown": str(params.get("show_benchmark", True)),
    }
    # Strategy-specific parameters
    for key in ("short_window", "long_window", "lookback_window", "zscore_threshold", "n_states"):
        if key in params:
            label = key.replace("_", " ").title()
            summary_rows[label] = str(params[key])

    summary_df = pd.DataFrame(
        list(summary_rows.items()), columns=["Parameter", "Value"]
    ).set_index("Parameter")
    st.dataframe(summary_df, use_container_width=True)

# ---------------------------------------------------------------------------
# ── SECTION: Raw Results ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

st.markdown("<div class='section-header'>Raw Results</div>", unsafe_allow_html=True)

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

for pct_col in ("market_return", "strategy_return", "drawdown"):
    if pct_col in display_df.columns:
        display_df[pct_col] = display_df[pct_col].map("{:.2%}".format)

for cum_col in ("cumulative_market", "cumulative_strategy"):
    if cum_col in display_df.columns:
        display_df[cum_col] = display_df[cum_col].map("{:.4f}".format)

with st.expander("🗃️ View Full Results Table", expanded=False):
    st.dataframe(display_df, use_container_width=True)
    st.caption(f"Showing {len(display_df):,} rows · signal: 1 = long, 0 = cash")

# Downloadable CSV
csv_buffer = io.StringIO()
results[display_cols].to_csv(csv_buffer)
st.download_button(
    label="⬇ Download Results CSV",
    data=csv_buffer.getvalue(),
    file_name=f"{params['ticker']}_{params['strategy'].replace(' ', '_')}_backtest.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='disclaimer'>"
    "⚠️ <strong>Research & Educational Use Only.</strong> "
    "This tool is intended for learning and exploratory analysis. "
    "Nothing presented here constitutes investment advice, and past performance "
    "is not indicative of future results. Always consult a qualified financial "
    "professional before making investment decisions."
    "</div>",
    unsafe_allow_html=True,
)

