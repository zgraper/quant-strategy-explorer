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

    A 2-state Gaussian HMM is fit to rolling log-returns.  The decoded
    state sequence is mapped to a long signal by selecting the regime
    whose in-sample mean return is higher.

    Parameters
    ----------
    close:
        Daily closing prices as a pandas Series indexed by date.
    params:
        Dictionary of strategy parameters.  Expected keys:

        - ``"n_regimes"`` (int): number of hidden states (default 2).
        - ``"hmm_iter"`` (int): EM iterations for model fitting (default 100).

    Returns
    -------
    pd.Series
        Integer signal series aligned to *close*: ``1`` = long, ``0`` = flat.
        Returns all-zero signal if the model fails to converge.
    """
    n_regimes: int = params.get("n_regimes", 2)
    hmm_iter: int = params.get("hmm_iter", 100)

    log_returns = np.log(close / close.shift(1)).dropna().values.reshape(-1, 1)

    signal = pd.Series(0, index=close.index)

    try:
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=hmm_iter,
            random_state=42,
        )
        model.fit(log_returns)
        hidden_states = model.predict(log_returns)

        # Identify the "bull" regime as the state with the highest mean return
        state_means = [
            log_returns[hidden_states == s].mean() for s in range(n_regimes)
        ]
        bull_state = int(np.argmax(state_means))

        # Align with original index (log_returns drops first row)
        signal.iloc[1:] = (hidden_states == bull_state).astype(int)

    except Exception:  # noqa: BLE001
        # Return all-flat signal on any model failure
        pass

    return signal
