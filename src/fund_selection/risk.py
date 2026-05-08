"""Métricas de riesgo para screening de ETFs y fondos.

Se priorizan métricas descriptivas y fáciles de defender: volatilidad,
downside deviation, Sharpe, Sortino, drawdown, VaR histórico y CVaR. No son
pronósticos; resumen el riesgo observado en la ventana analizada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .performance import TRADING_DAYS_PER_YEAR, _as_numeric_frame


def annualized_volatility(
    returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calcula la volatilidad anualizada de la muestra."""

    clean_returns = _as_numeric_frame(returns)
    return clean_returns.std(skipna=True, ddof=1) * np.sqrt(periods_per_year)


def downside_deviation(
    returns: pd.DataFrame,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calcula downside deviation anualizada frente a un retorno objetivo."""

    clean_returns = _as_numeric_frame(returns)
    daily_target = minimum_acceptable_return / periods_per_year
    downside = np.minimum(clean_returns - daily_target, 0.0)
    return np.sqrt((downside.pow(2)).mean(skipna=True)) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calcula Sharpe anualizado usando retornos diarios en exceso."""

    clean_returns = _as_numeric_frame(returns)
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = clean_returns - daily_rf
    annualized_excess_return = excess_returns.mean(skipna=True) * periods_per_year
    volatility = annualized_volatility(clean_returns, periods_per_year)
    return annualized_excess_return / volatility.replace(0.0, np.nan)


def sortino_ratio(
    returns: pd.DataFrame,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calcula el ratio Sortino anualizado."""

    clean_returns = _as_numeric_frame(returns)
    daily_mar = minimum_acceptable_return / periods_per_year
    annualized_excess_return = (clean_returns - daily_mar).mean(skipna=True) * periods_per_year
    downside_risk = downside_deviation(
        clean_returns,
        minimum_acceptable_return=minimum_acceptable_return,
        periods_per_year=periods_per_year,
    )
    return annualized_excess_return / downside_risk.replace(0.0, np.nan)


def drawdown_series(returns: pd.DataFrame) -> pd.DataFrame:
    """Calcula la serie de drawdown desde retornos diarios."""

    clean_returns = _as_numeric_frame(returns)
    wealth_index = (1.0 + clean_returns.fillna(0.0)).cumprod()
    running_peak = wealth_index.cummax()
    return wealth_index / running_peak - 1.0


def max_drawdown(returns: pd.DataFrame) -> pd.Series:
    """Calcula el máximo drawdown por ticker."""

    return drawdown_series(returns).min(skipna=True)


def historical_var(
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Calcula VaR histórico como pérdida positiva."""

    clean_returns = _as_numeric_frame(returns)
    tail_quantile = clean_returns.quantile(1.0 - confidence_level, interpolation="linear")
    return -tail_quantile


def historical_cvar(
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.Series:
    """Calcula CVaR histórico como pérdida promedio más allá del VaR."""

    clean_returns = _as_numeric_frame(returns)
    var_threshold = clean_returns.quantile(1.0 - confidence_level, interpolation="linear")
    cvar_values = {}
    for ticker in clean_returns.columns:
        tail = clean_returns.loc[clean_returns[ticker] <= var_threshold[ticker], ticker]
        cvar_values[ticker] = -tail.mean() if not tail.empty else np.nan
    return pd.Series(cvar_values, dtype=float)


def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Calcula volatilidad anualizada móvil."""

    clean_returns = _as_numeric_frame(returns)
    return clean_returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(
        periods_per_year
    )


def rolling_sharpe(
    returns: pd.DataFrame,
    window: int = 63,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Calcula Sharpe anualizado móvil."""

    clean_returns = _as_numeric_frame(returns)
    daily_rf = risk_free_rate / periods_per_year
    rolling_excess = (clean_returns - daily_rf).rolling(window=window, min_periods=window)
    annualized_excess = rolling_excess.mean() * periods_per_year
    annualized_risk = clean_returns.rolling(window=window, min_periods=window).std(
        ddof=1
    ) * np.sqrt(periods_per_year)
    return annualized_excess / annualized_risk.replace(0.0, np.nan)


def risk_summary(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Construye una tabla de riesgo por ticker."""

    clean_returns = _as_numeric_frame(returns)
    summary = pd.DataFrame(index=clean_returns.columns)
    summary["volatility"] = annualized_volatility(clean_returns, periods_per_year)
    summary["downside_deviation"] = downside_deviation(clean_returns, periods_per_year=periods_per_year)
    summary["sharpe_ratio"] = sharpe_ratio(clean_returns, risk_free_rate, periods_per_year)
    summary["sortino_ratio"] = sortino_ratio(clean_returns, risk_free_rate, periods_per_year)
    summary["max_drawdown"] = max_drawdown(clean_returns)
    summary["historical_var_95"] = historical_var(clean_returns, confidence_level)
    summary["historical_cvar_95"] = historical_cvar(clean_returns, confidence_level)
    return summary.reset_index(names="ticker")
