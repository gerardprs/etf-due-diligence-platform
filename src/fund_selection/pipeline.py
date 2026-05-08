"""Pipeline de análisis para la plataforma de selección de ETFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .benchmark import benchmark_summary
from .liquidity_cost import liquidity_cost_summary
from .memo import generate_all_memos
from .performance import cumulative_returns, performance_summary
from .red_flags import generate_red_flags
from .risk import drawdown_series, risk_summary
from .scoring import score_funds


@dataclass(frozen=True)
class SelectionAnalysisBundle:
    """Agrupa las salidas principales del análisis."""

    analysis: pd.DataFrame
    red_flags: pd.DataFrame
    cumulative_returns: pd.DataFrame
    drawdowns: pd.DataFrame
    memos: dict[str, str]


def load_processed_data(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga universo validado, retornos y metadata desde `data/processed`."""

    directory = Path(processed_dir)
    universe = pd.read_csv(directory / "fund_universe_validated.csv")
    returns = pd.read_csv(directory / "daily_returns.csv", parse_dates=["date"], index_col="date")
    metadata = pd.read_csv(directory / "fund_metadata.csv")

    universe["ticker"] = universe["ticker"].astype(str).str.upper()
    universe["benchmark_ticker"] = universe["benchmark_ticker"].astype(str).str.upper()
    returns.columns = [str(column).upper() for column in returns.columns]
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()

    return universe, returns, metadata


def build_selection_analysis(
    universe: pd.DataFrame,
    returns: pd.DataFrame,
    metadata: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> SelectionAnalysisBundle:
    """Ejecuta el flujo completo de análisis de selección."""

    fund_tickers = universe["ticker"].astype(str).str.upper().tolist()
    available_funds = [ticker for ticker in fund_tickers if ticker in returns.columns]
    fund_returns = returns[available_funds]

    perf = performance_summary(fund_returns)
    risk = risk_summary(fund_returns, risk_free_rate=risk_free_rate)
    benchmark = benchmark_summary(returns, universe, risk_free_rate=risk_free_rate)
    liquidity = liquidity_cost_summary(metadata)

    analysis = (
        universe.merge(perf, on="ticker", how="left")
        .merge(risk, on="ticker", how="left")
        .merge(benchmark, on=["ticker", "benchmark_ticker"], how="left")
        .merge(liquidity, on="ticker", how="left", suffixes=("", "_metadata"))
    )

    if "name_metadata" in analysis.columns:
        analysis["name"] = analysis["name"].fillna(analysis["name_metadata"])
        analysis = analysis.drop(columns=["name_metadata"])

    red_flags = generate_red_flags(analysis)
    scored = score_funds(analysis, red_flags)
    cumulative = cumulative_returns(returns)
    drawdowns = drawdown_series(fund_returns)
    memos = generate_all_memos(scored, red_flags)

    return SelectionAnalysisBundle(
        analysis=scored,
        red_flags=red_flags,
        cumulative_returns=cumulative,
        drawdowns=drawdowns,
        memos=memos,
    )


def build_selection_analysis_from_processed(
    processed_dir: str | Path,
    risk_free_rate: float = 0.0,
) -> SelectionAnalysisBundle:
    """Carga archivos procesados y ejecuta el análisis completo."""

    universe, returns, metadata = load_processed_data(processed_dir)
    return build_selection_analysis(
        universe=universe,
        returns=returns,
        metadata=metadata,
        risk_free_rate=risk_free_rate,
    )
