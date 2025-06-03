# fuelfinance.py

import numpy as np

def sharpe_ratio(returns, rf=0.0, period="daily"):
    """
    Calculate the Sharpe ratio of a return series.
    - returns: array-like of returns (e.g., daily returns)
    - rf: risk-free rate per period (if returns are daily, rf should be the daily risk-free rate)
    - period: string indicating frequency ("daily", "monthly", etc.)—included for future flexibility
    """
    excess_returns = returns - rf
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=0)
    if std_excess == 0:
        return np.nan
    return mean_excess / std_excess

def volatility(returns, period="daily"):
    """
    Calculate the volatility (standard deviation) of a return series.
    - returns: array-like of returns (e.g., daily returns)
    - period: string indicating frequency ("daily", "monthly", etc.)
    """
    return np.std(returns, ddof=0)

def max_drawdown(returns):
    """
    Calculate the maximum drawdown of a return series.
    - returns: array-like of returns (must be period returns, e.g. daily returns)
    Returns the maximum drawdown as a decimal (e.g., -0.2 for a 20% drawdown).
    """
    # Compute cumulative returns curve
    cumulative = np.cumprod(1 + returns)
    # Compute running maximum
    running_max = np.maximum.accumulate(cumulative)
    # Compute drawdown series
    drawdowns = (cumulative - running_max) / running_max
    # Return the minimum (most negative) drawdown
    return np.min(drawdowns)
