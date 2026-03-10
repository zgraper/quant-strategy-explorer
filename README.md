# 📈 Quant Strategy Explorer

A Streamlit web application for backtesting quantitative trading strategies on
historical price data.  Built as a portfolio project to demonstrate practical
skills in algorithmic trading research, data engineering, and interactive
data-science application development.

---

## Project Overview

Quant Strategy Explorer lets you select any Yahoo Finance–supported ticker,
choose a date range, pick a strategy, tune its parameters, and immediately
visualise the backtest results — all inside a clean, interactive web UI.

The project is intentionally modular: data loading, strategy logic, backtest
engine, metrics calculation, and UI components each live in their own file,
making it straightforward to swap strategies, extend the framework, or add
new data sources.

---

## Key Features

| Feature | Details |
|---|---|
| 🔍 Asset selection | Any ticker supported by yfinance (equities, ETFs, crypto) |
| 📅 Flexible date range | Custom start and end date via sidebar |
| 🧮 Four strategies | MA Crossover · Mean Reversion · Momentum · HMM Regime |
| ⚙️ Tunable parameters | Per-strategy sliders exposed in the sidebar |
| 🛡️ Input validation | Date-range and parameter checks before any computation |
| 📊 Performance metrics | Total return, CAGR, volatility, Sharpe ratio, max drawdown, win rate, trade count |
| 📉 Interactive charts | Price + signals, equity curve vs buy-and-hold, drawdown — all Plotly |
| 🗃️ Expandable data table | Full results DataFrame available for manual inspection |
| ⚠️ Graceful error handling | User-friendly messages for bad tickers, empty data, and NaN signals |

---

## Supported Strategies

### Moving Average Crossover
Generates a long signal when a short-period simple moving average (SMA) crosses
above a long-period SMA, and moves to cash when it crosses back below.  A
classic trend-following approach that works well in sustained trending markets.

### Mean Reversion
Computes a rolling z-score of the closing price.  Goes long when the price dips
a configurable number of standard deviations below its rolling mean (oversold),
and exits once the z-score reverts back to zero.  Suited for range-bound assets.

### Momentum
Holds a long position when the trailing return over a lookback window is
positive, and moves to cash otherwise.  Captures persistent directional moves
and is one of the most robust empirical anomalies in finance.

### HMM Regime Detection
Fits a Gaussian Hidden Markov Model to recent daily log-returns to identify
latent market regimes (e.g., bull vs. bear).  The strategy goes long when the
current decoded state matches the historically higher-return regime.  This is a
demonstration model and has not been validated for live trading.

---

## Tech Stack

| Layer | Library |
|---|---|
| Web UI | [Streamlit](https://streamlit.io) |
| Data | [yfinance](https://github.com/ranaroussi/yfinance) · [pandas](https://pandas.pydata.org) |
| Numerics | [NumPy](https://numpy.org) · [SciPy](https://scipy.org) |
| Machine learning | [scikit-learn](https://scikit-learn.org) · [hmmlearn](https://hmmlearn.readthedocs.io) |
| Charts | [Plotly](https://plotly.com/python/) |

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/zgraper/quant-strategy-explorer.git
cd quant-strategy-explorer

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run Locally

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`), configure
the sidebar controls, and click **▶ Run Backtest**.

---

## Project Structure

```
quant-strategy-explorer/
│
├── app.py                    # Streamlit entry point
├── requirements.txt          # Python package dependencies
├── README.md
├── .gitignore
│
├── data/
│   └── data_loader.py        # yfinance download + caching helpers
│
├── strategies/
│   ├── ma_crossover.py       # Moving Average Crossover signals
│   ├── mean_reversion.py     # Mean Reversion (z-score) signals
│   ├── momentum.py           # Momentum / trend-following signals
│   └── hmm_regime.py         # HMM Regime Detection signals
│
├── backtest/
│   ├── engine.py             # Long-only backtest engine
│   └── metrics.py            # Performance metric calculations
│
├── ui/
│   ├── sidebar.py            # Streamlit sidebar controls
│   └── charts.py             # Plotly chart builders
│
└── utils/
    └── helpers.py            # Shared utility functions
```

---

## Limitations & Future Improvements

**Current limitations (v1)**
- Daily close prices only — no intraday granularity
- Long/cash only — no short selling or leverage
- No slippage or realistic transaction cost modelling
- HMM strategy uses in-sample regime identification (mild look-ahead bias)
- No walk-forward validation or out-of-sample testing framework

**Planned improvements**
- [ ] Portfolio-level backtesting across multiple assets
- [ ] Walk-forward / rolling-window validation
- [ ] Additional strategies (RSI, Bollinger Bands, MACD)
- [ ] Realistic transaction cost and slippage models
- [ ] Parameter optimisation with overfitting guardrails
- [ ] CSV export of results

---

## Resume-Ready Highlights

- **Built a modular, production-quality Streamlit application** that downloads
  live market data, runs configurable quantitative trading strategy backtests,
  and presents results through interactive Plotly visualisations — demonstrating
  end-to-end ownership of a data-science product from ingestion to deployment.

- **Implemented four trading strategies** (Moving Average Crossover, Mean
  Reversion, Momentum, and HMM Regime Detection) with clean, tested, and
  well-documented Python modules, showcasing applied knowledge of time-series
  analysis, statistical signal processing, and unsupervised machine learning.

- **Engineered a robust long-only backtest engine** with lookahead-bias
  prevention, configurable transaction costs, full performance metrics (CAGR,
  Sharpe ratio, max drawdown, win rate), and graceful handling of edge cases
  such as sparse signals and insufficient data — reflecting software-engineering
  best practices in a quantitative finance context.

---

## License

MIT
