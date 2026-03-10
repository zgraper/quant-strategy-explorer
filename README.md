# Quant Strategy Explorer

A Streamlit portfolio app for exploring quantitative trading strategies on historical price data.

## Features

- **Asset selection** – pick any ticker supported by yfinance
- **Four strategies** – Moving Average Crossover, Mean Reversion, Momentum, HMM Regime Detection
- **Interactive sidebar** – tune strategy parameters in real time
- **Simple long-only backtest** – daily close prices, no leverage
- **Performance metrics** – total return, annualized return, Sharpe ratio, max drawdown, win rate, number of trades
- **Interactive Plotly charts** – price + signals, equity curve, drawdown

## Project Structure

```
quant_strategy_explorer/
│
├── app.py                    # Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── data_loader.py        # yfinance download helpers
│
├── strategies/
│   ├── ma_crossover.py       # Moving average crossover signals
│   ├── mean_reversion.py     # Mean reversion (z-score) signals
│   ├── momentum.py           # Momentum / trend-following signals
│   └── hmm_regime.py         # HMM regime detection signals
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

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Assumptions

- Daily close prices only (v1)
- Long-only positions (no shorting)
- No transaction costs or slippage
- Signals are generated end-of-day; positions open next-day open (approximated by next close)
