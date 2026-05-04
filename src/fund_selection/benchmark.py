"""Benchmark-fit analytics for ETF/fund selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .performance import TRADING_DAYS_PER_YEAR, _as_numeric_frame


def _safe_beta(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate beta with guardrails for short or zero-variance samples."""

    aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 30:
        return np.nan
    benchmark_variance = aligned.iloc[:, 1].var(ddof=1)
    if benchmark_variance == 0 or pd.isna(benchmark_variance):
        return np.nan
    covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    return covariance / benchmark_variance


def benchmark_summary(
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Calculate benchmark comparison metrics for every fund in the universe."""

    clean_returns = _as_numeric_frame(returns)
    rows: list[dict[str, float | str]] = []

    for record in universe.to_dict("records"):
        ticker = str(record["ticker"]).upper()
        benchmark_ticker = str(record["benchmark_ticker"]).upper()

        if ticker not in clean_returns.columns or benchmark_ticker not in clean_returns.columns:
            rows.append(
                {
                    "ticker": ticker,
                    "benchmark_ticker": benchmark_ticker,
                    "beta": np.nan,
                    "alpha": np.nan,
                    "tracking_error": np.nan,
                    "information_ratio": np.nan,
                    "correlation": np.nan,
                    "r_squared": np.nan,
                    "excess_return": np.nan,
                }
            )
            continue

        fund = clean_returns[ticker]
        benchmark = clean_returns[benchmark_ticker]
        aligned = pd.concat([fund, benchmark], axis=1, keys=["fund", "benchmark"]).dropna()

        if aligned.shape[0] < 30:
            metrics = {
                "beta": np.nan,
                "alpha": np.nan,
                "tracking_error": np.nan,
                "information_ratio": np.nan,
                "correlation": np.nan,
                "r_squared": np.nan,
                "excess_return": np.nan,
            }
        else:
            beta = _safe_beta(aligned["fund"], aligned["benchmark"])
            active_returns = aligned["fund"] - aligned["benchmark"]
            tracking_error = active_returns.std(ddof=1) * np.sqrt(periods_per_year)
            excess_return = active_returns.mean() * periods_per_year
            daily_rf = risk_free_rate / periods_per_year
            fund_excess = (aligned["fund"] - daily_rf).mean() * periods_per_year
            benchmark_excess = (aligned["benchmark"] - daily_rf).mean() * periods_per_year
            alpha = fund_excess - beta * benchmark_excess if pd.notna(beta) else np.nan
            correlation = aligned["fund"].corr(aligned["benchmark"])

            metrics = {
                "beta": beta,
                "alpha": alpha,
                "tracking_error": tracking_error,
                "information_ratio": excess_return / tracking_error if tracking_error else np.nan,
                "correlation": correlation,
                "r_squared": correlation**2 if pd.notna(correlation) else np.nan,
                "excess_return": excess_return,
            }

        rows.append(
            {
                "ticker": ticker,
                "benchmark_ticker": benchmark_ticker,
                **metrics,
            }
        )

    return pd.DataFrame(rows)
