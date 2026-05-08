"""Respaldo opcional de perfiles ETF de Alpha Vantage con caché local."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .settings import get_alpha_vantage_api_key


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def _cache_path(project_root: str | Path, ticker: str) -> Path:
    cache_dir = Path(project_root) / "data" / "processed" / "alpha_vantage_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker.upper()}.json"


def _numeric(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan


def _normalize_weight(value: Any) -> float:
    """Convierte el peso de un holding a formato decimal."""

    numeric = _numeric(value)
    if pd.isna(numeric):
        return np.nan
    return numeric / 100.0 if numeric > 1 else numeric


def fetch_etf_profile(
    ticker: str,
    project_root: str | Path,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    """Obtiene el perfil de un ETF desde Alpha Vantage o desde caché local.

    La caché evita llamadas innecesarias a la API, especialmente cuando se usa
    una llave gratuita con límite diario bajo.
    """

    clean_ticker = ticker.upper()
    cache_file = _cache_path(project_root, clean_ticker)

    if cache_file.exists() and not force_refresh:
        with cache_file.open("r", encoding="utf-8") as file:
            return json.load(file)

    api_key = get_alpha_vantage_api_key(project_root)
    if not api_key:
        return None

    response = requests.get(
        ALPHA_VANTAGE_URL,
        params={
            "function": "ETF_PROFILE",
            "symbol": clean_ticker,
            "apikey": api_key,
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    if "Information" in data or "Note" in data or "Error Message" in data:
        return None

    with cache_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return data


def parse_etf_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Extrae campos útiles para el dashboard de revisión."""

    holdings = profile.get("holdings") or profile.get("Holdings") or []
    top_10_weights: list[float] = []
    top_10_holdings: list[dict[str, Any]] = []

    if isinstance(holdings, list):
        for holding in holdings[:10]:
            if not isinstance(holding, dict):
                continue
            raw_weight = (
                holding.get("weight")
                or holding.get("weight_pct")
                or holding.get("Weight")
                or holding.get("portfolio_percentage")
            )
            weight = _normalize_weight(raw_weight)
            top_10_weights.append(weight)
            top_10_holdings.append(
                {
                    "symbol": holding.get("symbol") or holding.get("ticker") or "",
                    "name": holding.get("description") or holding.get("name") or "",
                    "weight": weight,
                }
            )

    top_10_concentration = (
        float(np.nansum(top_10_weights)) if top_10_weights else np.nan
    )

    return {
        "total_assets": _numeric(profile.get("net_assets") or profile.get("netAssets")),
        "expense_ratio": _numeric(
            profile.get("net_expense_ratio")
            or profile.get("expense_ratio")
            or profile.get("expenseRatio")
        ),
        "dividend_yield": _numeric(profile.get("dividend_yield")),
        "top_10_concentration": top_10_concentration,
        "top_10_holdings": json.dumps(top_10_holdings, ensure_ascii=False)
        if top_10_holdings
        else np.nan,
    }


def enrich_metadata_with_alpha_vantage(
    metadata: pd.DataFrame,
    project_root: str | Path,
    max_api_requests: int = 3,
) -> pd.DataFrame:
    """Completa metadata faltante de ETFs usando Alpha Vantage.

    Solo los tickers sin caché consumen el límite de `max_api_requests`.
    Los perfiles ya guardados se reutilizan sin nuevas solicitudes.
    """

    if metadata.empty:
        return metadata

    enriched = metadata.copy()
    api_requests = 0
    request_trigger_fields = [
        "total_assets",
        "expense_ratio",
        "dividend_yield",
        "top_10_concentration",
    ]
    fill_fields = [
        *request_trigger_fields,
        "top_10_holdings",
    ]

    for field in fill_fields:
        if field not in enriched.columns:
            enriched[field] = np.nan
    if "top_10_holdings" in enriched.columns:
        enriched["top_10_holdings"] = enriched["top_10_holdings"].astype("object")

    for idx, row in enriched.iterrows():
        ticker = str(row["ticker"]).upper()
        cache_file = _cache_path(project_root, ticker)
        was_cached = cache_file.exists()
        missing_useful_fields = any(
            pd.isna(row.get(field)) for field in request_trigger_fields
        )

        if not missing_useful_fields and not was_cached:
            continue
        if not was_cached and api_requests >= max_api_requests:
            continue

        profile = fetch_etf_profile(ticker, project_root)
        if not was_cached:
            api_requests += 1

        if profile is None:
            continue

        parsed = parse_etf_profile(profile)
        for field, value in parsed.items():
            if field in enriched.columns and pd.isna(enriched.at[idx, field]) and pd.notna(value):
                enriched.at[idx, field] = value

    return enriched
