"""Revisión de liquidez y costo para el screening de ETFs y fondos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_expense_ratio(raw_value: object) -> float:
    """Normaliza el expense ratio del proveedor a puntos porcentuales.

    Yahoo puede devolver el costo como porcentaje (0.03 significa 0.03%) o como
    decimal (0.0003 significa 0.03%). Esta normalización evita comparar costos
    con escalas mezcladas.
    """

    value = pd.to_numeric(raw_value, errors="coerce")
    if pd.isna(value) or value < 0:
        return np.nan
    if value < 0.01:
        return float(value * 100.0)
    return float(value)


def liquidity_cost_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    """Crea métricas de liquidez y costo a partir de la metadata disponible."""

    if metadata.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "total_assets",
                "expense_ratio_pct",
                "average_volume",
                "last_price",
                "average_dollar_volume",
                "trailing_pe",
                "forward_pe",
                "price_to_book",
                "portfolio_pe",
                "portfolio_price_to_book",
                "valuation_pe",
                "valuation_price_to_book",
                "return_on_equity",
                "dividend_yield",
                "top_10_concentration",
                "top_10_holdings",
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
    data["trailing_pe"] = pd.to_numeric(data.get("trailing_pe"), errors="coerce")
    data["forward_pe"] = pd.to_numeric(data.get("forward_pe"), errors="coerce")
    data["price_to_book"] = pd.to_numeric(data.get("price_to_book"), errors="coerce")
    data["portfolio_pe"] = pd.to_numeric(data.get("portfolio_pe"), errors="coerce")
    data["portfolio_price_to_book"] = pd.to_numeric(
        data.get("portfolio_price_to_book"),
        errors="coerce",
    )
    data["valuation_pe"] = data["trailing_pe"].fillna(data["portfolio_pe"]).fillna(data["forward_pe"])
    data["valuation_price_to_book"] = data["price_to_book"].fillna(
        data["portfolio_price_to_book"]
    )
    data["return_on_equity"] = pd.to_numeric(data.get("return_on_equity"), errors="coerce")
    data["dividend_yield"] = pd.to_numeric(data.get("dividend_yield"), errors="coerce")
    data["top_10_concentration"] = pd.to_numeric(
        data.get("top_10_concentration"),
        errors="coerce",
    )
    data["top_10_holdings"] = data.get("top_10_holdings", pd.NA)
    data["metadata_available"] = data.get("metadata_available", False).fillna(False).astype(bool)

    columns = [
        "ticker",
        "total_assets",
        "expense_ratio_pct",
        "average_volume",
        "last_price",
        "average_dollar_volume",
        "trailing_pe",
        "forward_pe",
        "price_to_book",
        "portfolio_pe",
        "portfolio_price_to_book",
        "valuation_pe",
        "valuation_price_to_book",
        "return_on_equity",
        "dividend_yield",
        "top_10_concentration",
        "top_10_holdings",
        "metadata_available",
    ]
    optional_columns = [
        column
        for column in ["name", "quote_type", "category", "fund_family", "currency", "exchange"]
        if column in data.columns
    ]
    return data[[*columns, *optional_columns]]
