"""Red-flag detection for preliminary fund due diligence."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = {
    "min_observations": 756,
    "max_expense_ratio_pct": 0.40,
    "min_total_assets": 500_000_000,
    "min_average_dollar_volume": 5_000_000,
    "max_tracking_error": 0.05,
    "max_drawdown_floor": -0.35,
}


FLAG_PENALTIES = {
    "Low": 3.0,
    "Medium": 6.0,
    "High": 10.0,
}


def _add_flag(
    rows: list[dict[str, object]],
    ticker: str,
    severity: str,
    flag: str,
    rationale: str,
) -> None:
    rows.append(
        {
            "ticker": ticker,
            "severity": severity,
            "flag": flag,
            "rationale": rationale,
            "penalty": FLAG_PENALTIES[severity],
        }
    )


def generate_red_flags(
    analysis: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Detect business-relevant red flags for ETF/fund screening."""

    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rows: list[dict[str, object]] = []

    for record in analysis.to_dict("records"):
        ticker = str(record["ticker"]).upper()

        observations = pd.to_numeric(record.get("observations"), errors="coerce")
        if pd.notna(observations) and observations < limits["min_observations"]:
            _add_flag(
                rows,
                ticker,
                "Medium",
                "Historial corto",
                "El historial de retornos disponible es menor a tres años bursátiles.",
            )

        expense_ratio = pd.to_numeric(record.get("expense_ratio_pct"), errors="coerce")
        if pd.isna(expense_ratio):
            _add_flag(
                rows,
                ticker,
                "Medium",
                "Expense ratio no disponible",
                "La información de costos no está disponible y debe verificarse antes de recomendar.",
            )
        elif expense_ratio > limits["max_expense_ratio_pct"]:
            _add_flag(
                rows,
                ticker,
                "Medium",
                "Expense ratio alto",
                "El costo del fondo es alto frente a un filtro sensible a comisiones.",
            )

        total_assets = pd.to_numeric(record.get("total_assets"), errors="coerce")
        if pd.isna(total_assets):
            _add_flag(
                rows,
                ticker,
                "Low",
                "AUM no disponible",
                "La información de activos bajo gestión no está disponible y debe verificarse manualmente.",
            )
        elif total_assets < limits["min_total_assets"]:
            _add_flag(
                rows,
                ticker,
                "Medium",
                "AUM bajo",
                "Una base de activos menor puede elevar riesgos de cierre, spreads y ejecución.",
            )

        dollar_volume = pd.to_numeric(record.get("average_dollar_volume"), errors="coerce")
        if pd.isna(dollar_volume):
            _add_flag(
                rows,
                ticker,
                "Low",
                "Liquidez no disponible",
                "La actividad promedio de negociación no está disponible desde el proveedor de datos.",
            )
        elif dollar_volume < limits["min_average_dollar_volume"]:
            _add_flag(
                rows,
                ticker,
                "Medium",
                "Baja liquidez de negociación",
                "El volumen promedio en dólares está por debajo del umbral institucional definido.",
            )

        tracking_error = pd.to_numeric(record.get("tracking_error"), errors="coerce")
        if pd.notna(tracking_error) and tracking_error > limits["max_tracking_error"]:
            _add_flag(
                rows,
                ticker,
                "High",
                "Tracking error alto",
                "El fondo se desvía de forma relevante frente a su benchmark asignado.",
            )

        max_drawdown = pd.to_numeric(record.get("max_drawdown"), errors="coerce")
        if pd.notna(max_drawdown) and max_drawdown < limits["max_drawdown_floor"]:
            _add_flag(
                rows,
                ticker,
                "Medium",
                "Drawdown elevado",
                "La pérdida histórica de pico a valle es elevada para un filtro preliminar.",
            )

        cagr = pd.to_numeric(record.get("cagr"), errors="coerce")
        if pd.notna(cagr) and cagr < 0:
            _add_flag(
                rows,
                ticker,
                "Medium",
                "CAGR negativo",
                "El fondo registró crecimiento compuesto negativo durante la ventana analizada.",
            )

    if not rows:
        return pd.DataFrame(columns=["ticker", "severity", "flag", "rationale", "penalty"])

    flags = pd.DataFrame(rows)
    flags["penalty"] = pd.to_numeric(flags["penalty"], errors="coerce").replace(np.nan, 0.0)
    flags["severity"] = flags["severity"].map(
        {"Low": "Baja", "Medium": "Media", "High": "Alta"}
    ).fillna(flags["severity"])
    return flags.sort_values(["ticker", "penalty"], ascending=[True, False]).reset_index(drop=True)
