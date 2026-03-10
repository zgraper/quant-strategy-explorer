"""
utils/helpers.py – Shared Utility Functions
=============================================
Small, reusable helper functions used across the Quant Strategy Explorer
project.  Keep this module lightweight; strategy- or backtest-specific
logic belongs in its own module.
"""

from __future__ import annotations

import pandas as pd


def pct_format(value: float, decimals: int = 1) -> str:
    """Format a fractional value as a percentage string.

    Parameters
    ----------
    value:
        Fractional value, e.g. ``0.1234``.
    decimals:
        Number of decimal places to show (default 1).

    Returns
    -------
    str
        E.g. ``"12.3 %"``.

    Examples
    --------
    >>> pct_format(0.1234)
    '12.3 %'
    >>> pct_format(-0.056, decimals=2)
    '-5.60 %'
    """
    return f"{value * 100:.{decimals}f} %"


def validate_date_range(start_date: str, end_date: str) -> bool:
    """Return True if *start_date* is strictly before *end_date*.

    Parameters
    ----------
    start_date:
        ISO-8601 date string, e.g. ``"2020-01-01"``.
    end_date:
        ISO-8601 date string, e.g. ``"2024-12-31"``.

    Returns
    -------
    bool
    """
    return pd.Timestamp(start_date) < pd.Timestamp(end_date)


def align_series(s1: pd.Series, s2: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align two Series to their common index, dropping mismatched rows.

    Parameters
    ----------
    s1, s2:
        pandas Series to align.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Both series reindexed to their intersection.
    """
    common = s1.index.intersection(s2.index)
    return s1.loc[common], s2.loc[common]
