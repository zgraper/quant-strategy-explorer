"""
ui/sidebar.py – Streamlit Sidebar Controls
============================================
Renders all interactive controls in the Streamlit sidebar and returns a
dictionary of user-selected parameters used by the rest of the app.
"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGIES = [
    "Moving Average Crossover",
    "Mean Reversion",
    "Momentum",
    "HMM Regime Detection",
]

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "BTC-USD", "ETH-USD"]


def render_sidebar() -> dict:
    """Render sidebar widgets and return a parameter dictionary.

    Returns
    -------
    dict
        Keys vary by selected strategy but always include:

        - ``"ticker"`` (str)
        - ``"start_date"`` (str, ISO-8601)
        - ``"end_date"`` (str, ISO-8601)
        - ``"strategy"`` (str)

        Additional keys are strategy-specific (see each strategy module).
    """
    st.sidebar.header("⚙️ Configuration")

    # ---- Asset selection ---------------------------------------------------
    st.sidebar.subheader("Asset")
    ticker_input = st.sidebar.text_input(
        "Ticker symbol",
        value="AAPL",
        help="Enter any valid Yahoo Finance ticker, e.g. AAPL, SPY, BTC-USD.",
    )
    ticker = ticker_input.strip().upper() or "AAPL"

    # ---- Date range --------------------------------------------------------
    st.sidebar.subheader("Date Range")
    end_default = date.today()
    start_default = end_default - timedelta(days=5 * 365)

    start_date = st.sidebar.date_input("Start date", value=start_default)
    end_date = st.sidebar.date_input("End date", value=end_default)

    # ---- Strategy selection ------------------------------------------------
    st.sidebar.subheader("Strategy")
    strategy = st.sidebar.selectbox("Strategy", STRATEGIES)

    params: dict = {
        "ticker": ticker,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "strategy": strategy,
    }

    # ---- Strategy-specific parameters -------------------------------------
    st.sidebar.subheader("Strategy Parameters")

    if strategy == "Moving Average Crossover":
        params["fast_window"] = st.sidebar.slider(
            "Fast MA window (days)", min_value=5, max_value=50, value=20, step=1
        )
        params["slow_window"] = st.sidebar.slider(
            "Slow MA window (days)", min_value=20, max_value=200, value=50, step=5
        )

    elif strategy == "Mean Reversion":
        params["mr_window"] = st.sidebar.slider(
            "Lookback window (days)", min_value=5, max_value=60, value=20, step=1
        )
        params["entry_z"] = st.sidebar.slider(
            "Entry z-score threshold", min_value=-3.0, max_value=-0.5, value=-1.5, step=0.1
        )
        params["exit_z"] = st.sidebar.slider(
            "Exit z-score threshold", min_value=-1.0, max_value=1.0, value=-0.5, step=0.1
        )

    elif strategy == "Momentum":
        params["mom_window"] = st.sidebar.slider(
            "Momentum window (days)", min_value=5, max_value=252, value=20, step=1
        )
        params["min_return"] = st.sidebar.slider(
            "Min return to go long (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5
        ) / 100.0

    elif strategy == "HMM Regime Detection":
        params["n_regimes"] = st.sidebar.slider(
            "Number of regimes", min_value=2, max_value=4, value=2, step=1
        )
        params["hmm_iter"] = st.sidebar.slider(
            "EM iterations", min_value=50, max_value=500, value=100, step=50
        )

    return params
