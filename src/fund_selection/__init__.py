"""Paquete de análisis para selección de ETFs y fondos."""

from .data_loader import (
    DataQualityReport,
    FundDataBundle,
    build_data_snapshot,
    calculate_daily_returns,
    fetch_fund_metadata,
    load_fund_universe,
    normalize_price_frame,
)
from .pipeline import (
    SelectionAnalysisBundle,
    build_selection_analysis,
    build_selection_analysis_from_processed,
)

__all__ = [
    "DataQualityReport",
    "FundDataBundle",
    "SelectionAnalysisBundle",
    "build_data_snapshot",
    "build_selection_analysis",
    "build_selection_analysis_from_processed",
    "calculate_daily_returns",
    "fetch_fund_metadata",
    "load_fund_universe",
    "normalize_price_frame",
]
