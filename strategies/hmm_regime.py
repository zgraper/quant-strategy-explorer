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
