"""
strategies/hmm_regime.py – HMM Regime Detection Strategy
=========================================================
Uses a Gaussian Hidden Markov Model (HMM) to identify latent market
regimes (e.g. bull vs. bear) from daily log-returns.  The strategy goes
long when the most recently decoded regime matches the historically
"higher-return" regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def generate_hmm_signals(
    df: pd.DataFrame,
    n_states: int,
    lookback_window: int,
) -> pd.DataFrame:
    """Compute HMM Regime Detection signals from a price DataFrame.

    A Gaussian HMM is fit to the most recent *lookback_window* daily
    log-returns.  Each bar is then assigned to a hidden state (regime),
    and the strategy goes long in the state with the highest average return.

    Assumptions and limitations
    ---------------------------
    - Only daily log-returns are used as the observation feature.  More
      sophisticated models might add rolling volatility or volume.
    - The model is re-fit on a fixed trailing window rather than a true
      rolling walk-forward, which means regimes near the start of the
      series may not reflect the current market dynamics.
    - HMM inference is non-deterministic; results can vary between runs
      unless ``random_state`` is fixed (it is, set to 42 here).
    - This implementation is intentionally lightweight and educational,
      not institution-grade.

    Parameters
    ----------
    df:
        DataFrame containing at least a ``Close`` column, indexed by date.
    n_states:
        Number of hidden states (regimes) for the HMM.
    lookback_window:
        Number of the most recent trading days used to *fit* the model.
        At least ``n_states * 10`` rows are required; if fewer observations
        are available, the function falls back to an all-flat signal.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with three additional columns:

        - ``regime``: integer label (0 … n_states-1) of the inferred hidden
          state for each bar.  NaN for bars before the first log-return.
          Falls back to NaN everywhere on model failure.
        - ``signal``: ``1`` = long (bull regime), ``0`` = flat.
        - ``position_change``: first-difference of ``signal``; ``+1`` =
          entry, ``-1`` = exit, ``0`` = no change.
    """
    out = df.copy()

    log_returns = np.log(out["Close"] / out["Close"].shift(1)).dropna()

    # Initialise output columns with safe defaults
    out["regime"] = float("nan")
    out["signal"] = 0
    out["position_change"] = 0

    # Need enough data to fit the model (heuristic: at least n_states * 10)
    min_required = max(n_states * 10, n_states + 1)
    if len(log_returns) < min_required:
        # Not enough data – return all-flat signal
        return out

    # Fit on the most recent lookback_window observations only.
    # Using a trailing window avoids refitting the entire history each time
    # and keeps the model focused on the current market environment.
    fit_data = log_returns.iloc[-lookback_window:].values.reshape(-1, 1)

    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(fit_data)

        # Decode regimes for the full available return series
        all_returns = log_returns.values.reshape(-1, 1)
        hidden_states = model.predict(all_returns)

        # Identify the "bull" regime as the state with the highest mean return.
        # States with no observations receive -inf so they are never selected.
        # Note: this is an in-sample identification – the bull state is chosen
        # based on historical data, which introduces a mild look-ahead bias for
        # the earliest observations in the series.
        state_means = [
            float(all_returns[hidden_states == s].mean())
            if (hidden_states == s).any()
            else float("-inf")  # never pick an empty state as the bull regime
            for s in range(n_states)
        ]
        bull_state = int(np.argmax(state_means))

        # Align decoded states back to the original DataFrame index.
        # log_returns drops the first row (NaN from shift), so we write into
        # the rows that correspond to those dates.
        out.loc[log_returns.index, "regime"] = hidden_states.astype(float)
        out.loc[log_returns.index, "signal"] = (hidden_states == bull_state).astype(int)

    except Exception:  # noqa: BLE001
        # Return all-flat signal on any model failure (e.g. convergence error)
        pass

    out["position_change"] = out["signal"].diff().fillna(0).astype(int)

    return out


def hmm_regime_signals(close: pd.Series, params: dict) -> pd.Series:
    """Compute long/flat signals via HMM regime detection.

    A Gaussian HMM is fit to the most recent *lookback_window* log-returns.
    The decoded state sequence is mapped to a long signal by selecting the
    regime whose in-sample mean return is higher.

    Parameters
    ----------
    close:
        Daily closing prices as a pandas Series indexed by date.
    params:
        Dictionary of strategy parameters.  Expected keys:

        - ``"n_states"`` (int): number of hidden states (default 2).
        - ``"lookback_window"`` (int): number of recent trading days used to
          fit the HMM (default 100).

    Returns
    -------
    pd.Series
        Integer signal series aligned to *close*: ``1`` = long, ``0`` = flat.
        Returns all-zero signal if the model fails to converge.
    """
    n_states: int = params.get("n_states", 2)
    lookback_window: int = params.get("lookback_window", 100)

    log_returns = np.log(close / close.shift(1)).dropna()

    # Fit on the most recent lookback_window observations only
    fit_data = log_returns.iloc[-lookback_window:].values.reshape(-1, 1)

    signal = pd.Series(0, index=close.index)

    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(fit_data)

        # Predict regimes for the full series
        all_returns = log_returns.values.reshape(-1, 1)
        hidden_states = model.predict(all_returns)

        # Identify the "bull" regime as the state with the highest mean return
        state_means = [
            all_returns[hidden_states == s].mean() for s in range(n_states)
        ]
        bull_state = int(np.argmax(state_means))

        # Align with original index (log_returns drops first row)
        signal.iloc[1:] = (hidden_states == bull_state).astype(int)

    except Exception:  # noqa: BLE001
        # Return all-flat signal on any model failure
        pass

    return signal
