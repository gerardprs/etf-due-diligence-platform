"""Métricas de performance para revisión de ETFs y fondos.

Las funciones usan retornos simples diarios como base porque son fáciles de
leer en reportes de performance y comparación contra benchmark. Las métricas
anualizadas asumen 252 ruedas bursátiles salvo que se indique otro valor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _as_numeric_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame numérico ordenado por fecha."""

    frame = data.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    return frame.apply(pd.to_numeric, errors="coerce")


def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos acumulados desde retornos simples diarios."""

    clean_returns = _as_numeric_frame(returns)
    return (1.0 + clean_returns.fillna(0.0)).cumprod() - 1.0


def total_return(returns: pd.DataFrame) -> pd.Series:
    """Calcula el retorno compuesto del periodo por ticker."""

    clean_returns = _as_numeric_frame(returns)
    return (1.0 + clean_returns).prod(skipna=True) - 1.0


def cagr(
    returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calcula el crecimiento anual compuesto por ticker.

    El CAGR anualiza la trayectoria compuesta realizada. Es útil para comparar
    fondos porque refleja el crecimiento de capital de inicio a fin del periodo.
    """

    clean_returns = _as_numeric_frame(returns)
    growth = (1.0 + clean_returns).prod(skipna=True)
    years = clean_returns.count() / periods_per_year

    result = pd.Series(np.nan, index=clean_returns.columns, dtype=float)
    valid = (years > 0) & (growth > 0)
    result.loc[valid] = growth.loc[valid].pow(1.0 / years.loc[valid]) - 1.0
    return result


def annualized_arithmetic_return(
    returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calcula el retorno anualizado aritmético desde retornos diarios."""

    clean_returns = _as_numeric_frame(returns)
    return clean_returns.mean(skipna=True) * periods_per_year


def monthly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compone retornos diarios en retornos mensuales calendario."""

    clean_returns = _as_numeric_frame(returns)
    return (1.0 + clean_returns).resample("ME").prod(min_count=1) - 1.0


def performance_summary(
    returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Construye una tabla de performance por ticker."""

    clean_returns = _as_numeric_frame(returns)
    monthly = monthly_returns(clean_returns)

    summary = pd.DataFrame(index=clean_returns.columns)
    summary["observations"] = clean_returns.count()
    summary["total_return"] = total_return(clean_returns)
    summary["cagr"] = cagr(clean_returns, periods_per_year=periods_per_year)
    summary["annualized_arithmetic_return"] = annualized_arithmetic_return(
        clean_returns,
        periods_per_year=periods_per_year,
    )
    summary["best_month"] = monthly.max(skipna=True)
    summary["worst_month"] = monthly.min(skipna=True)
    summary["positive_months_ratio"] = monthly.gt(0).sum() / monthly.count()

    return summary.reset_index(names="ticker")
