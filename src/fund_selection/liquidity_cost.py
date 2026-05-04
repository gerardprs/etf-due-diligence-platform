"""Liquidity and cost checks for ETF/fund screening."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_expense_ratio(raw_value: object) -> float:
    """Normalize vendor expense ratio into percentage points.

    Yahoo may return ETF expense ratios either as percentage points (0.03 means
    0.03%) or as decimals in some contexts (0.0003 means 0.03%). This guardrail
    keeps the dashboard interpretable without assuming all vendors are clean.
    """

    value = pd.to_numeric(raw_value, errors="coerce")
    if pd.isna(value) or value < 0:
        return np.nan
    if value < 0.01:
        return float(value * 100.0)
    return float(value)


def liquidity_cost_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    """Create fund-level liquidity and fee metrics from vendor metadata."""

    if metadata.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "total_assets",
                "expense_ratio_pct",
                "average_volume",
                "last_price",
                "average_dollar_volume",
                "metadata_available",
            ]
        )

    data = metadata.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper()
    data["total_assets"] = pd.to_numeric(data.get("total_assets"), errors="coerce")
    data["average_volume"] = pd.to_numeric(data.get("average_volume"), errors="coerce")
    data["last_price"] = pd.to_numeric(data.get("last_price"), errors="coerce")
    data["expense_ratio_pct"] = data.get("expense_ratio").map(normalize_expense_ratio)
    data["average_dollar_volume"] = data["average_volume"] * data["last_price"]
    data["metadata_available"] = data.get("metadata_available", False).fillna(False).astype(bool)

    columns = [
        "ticker",
        "total_assets",
        "expense_ratio_pct",
        "average_volume",
        "last_price",
        "average_dollar_volume",
        "metadata_available",
    ]
    optional_columns = [
        column
        for column in ["name", "quote_type", "category", "fund_family", "currency", "exchange"]
        if column in data.columns
    ]
    return data[[*columns, *optional_columns]]
