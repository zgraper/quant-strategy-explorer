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

TICKERS = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD"]

STRATEGIES = [
    "Moving Average Crossover",
    "Mean Reversion",
    "Momentum",
    "HMM Regime",
]


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
        - ``"run_backtest"`` (bool) – ``True`` when the user pressed the button

        Additional keys are strategy-specific:

        *Moving Average Crossover*: ``short_window``, ``long_window``

        *Mean Reversion*: ``lookback_window``, ``zscore_threshold``

        *Momentum*: ``lookback_window``

        *HMM Regime*: ``n_states``, ``lookback_window``
    """
    st.sidebar.header("⚙️ Configuration")

    # ---- Asset selection ---------------------------------------------------
    st.sidebar.subheader("Asset")
    ticker = st.sidebar.selectbox(
        "Ticker",
        options=TICKERS,
        index=2,  # default: SPY
        help="Select a ticker from the list.",
    )

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
        params["short_window"] = st.sidebar.slider(
            "Short MA window (days)", min_value=5, max_value=50, value=20, step=1
        )
        params["long_window"] = st.sidebar.slider(
            "Long MA window (days)", min_value=20, max_value=200, value=50, step=5
        )

    elif strategy == "Mean Reversion":
        params["lookback_window"] = st.sidebar.slider(
            "Lookback window (days)", min_value=5, max_value=60, value=20, step=1
        )
        params["zscore_threshold"] = st.sidebar.slider(
            "Z-score entry threshold", min_value=-3.0, max_value=-0.5, value=-1.5, step=0.1
        )

    elif strategy == "Momentum":
        params["lookback_window"] = st.sidebar.slider(
            "Lookback window (days)", min_value=5, max_value=252, value=20, step=1
        )

    elif strategy == "HMM Regime":
        params["n_states"] = st.sidebar.slider(
            "Number of states", min_value=2, max_value=4, value=2, step=1
        )
        params["lookback_window"] = st.sidebar.slider(
            "Lookback window (days)", min_value=50, max_value=500, value=100, step=50
        )

    # ---- Run button --------------------------------------------------------
    params["run_backtest"] = st.sidebar.button("▶ Run Backtest", use_container_width=True)

    return params
