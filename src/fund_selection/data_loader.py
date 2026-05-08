"""Carga y limpieza de datos para screening de ETFs y fondos.

La capa de datos es defensiva porque las APIs públicas pueden devolver series
parciales, metadata incompleta o fallas por ticker. El resto del análisis recibe
matrices limpias y un reporte explícito de calidad de datos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - handled at runtime for user clarity.
    yf = None


REQUIRED_UNIVERSE_COLUMNS = {
    "ticker",
    "name",
    "asset_class",
    "benchmark_ticker",
}


@dataclass(frozen=True)
class DataQualityReport:
    """Resumen de calidad del set descargado.

    Campos:
        requested_tickers: fondos y benchmarks solicitados.
        available_tickers: tickers con al menos un precio válido.
        missing_tickers: tickers solicitados sin precios utilizables.
        start_date: primera fecha disponible después de limpieza.
        end_date: última fecha disponible después de limpieza.
        observations: cantidad de filas en la matriz de precios.
        missing_ratio_by_ticker: porcentaje de datos faltantes por ticker.
    """

    requested_tickers: list[str]
    available_tickers: list[str]
    missing_tickers: list[str]
    start_date: str | None
    end_date: str | None
    observations: int
    missing_ratio_by_ticker: dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        """Devuelve el reporte en formato de dos columnas."""

        rows = {
            "requested_tickers": ", ".join(self.requested_tickers),
            "available_tickers": ", ".join(self.available_tickers),
            "missing_tickers": ", ".join(self.missing_tickers) or "None",
            "start_date": self.start_date,
            "end_date": self.end_date,
            "observations": self.observations,
        }
        return pd.DataFrame(
            {"metric": rows.keys(), "value": rows.values()}
        ).reset_index(drop=True)


@dataclass(frozen=True)
class FundDataBundle:
    """Universo validado y datos de mercado para el análisis."""

    universe: pd.DataFrame
    prices: pd.DataFrame
    returns: pd.DataFrame
    metadata: pd.DataFrame
    quality_report: DataQualityReport


def _clean_ticker(value: object) -> str:
    """Normaliza tickers conservando sufijos de Yahoo como `.L`."""

    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _unique_ordered(values: Iterable[str]) -> list[str]:
    """Devuelve valores únicos no vacíos preservando el orden original."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = _clean_ticker(value)
        if cleaned and cleaned not in seen:
            ordered.append(cleaned)
            seen.add(cleaned)
    return ordered


def load_fund_universe(path: str | Path) -> pd.DataFrame:
    """Carga y valida el universo de fondos usado por la plataforma.

    Columnas requeridas:
        ticker, name, asset_class, benchmark_ticker

    Las columnas opcionales se preservan para reportes, peer groups,
    liquidez, costos y generación de memo.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fund universe file not found: {csv_path}")

    universe = pd.read_csv(csv_path)
    universe.columns = [str(col).strip().lower() for col in universe.columns]

    missing_columns = REQUIRED_UNIVERSE_COLUMNS.difference(universe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fund universe is missing required columns: {missing}")

    universe["ticker"] = universe["ticker"].map(_clean_ticker)
    universe["benchmark_ticker"] = universe["benchmark_ticker"].map(_clean_ticker)

    if universe["ticker"].eq("").any():
        raise ValueError("Fund universe contains blank tickers.")

    if universe["benchmark_ticker"].eq("").any():
        raise ValueError("Fund universe contains blank benchmark tickers.")

    duplicated = universe.loc[universe["ticker"].duplicated(), "ticker"].tolist()
    if duplicated:
        duplicated_text = ", ".join(sorted(set(duplicated)))
        raise ValueError(f"Fund universe contains duplicated tickers: {duplicated_text}")

    return universe.sort_values("ticker").reset_index(drop=True)


def download_price_history(
    tickers: Iterable[str],
    start: str = "2021-01-01",
    end: str | None = None,
    price_field: str = "Close",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Descarga y normaliza precios históricos desde Yahoo Finance.

    `auto_adjust=True` ajusta precios por splits y dividendos. Para comparar
    ETFs, esta base es más razonable que precios sin ajustar, aunque sigue
    siendo data pública y debe validarse contra fuentes oficiales.
    """

    if yf is None:
        raise ImportError(
            "yfinance is not installed. Install project dependencies with "
            "`pip install -r requirements.txt`."
        )

    clean_tickers = _unique_ordered(tickers)
    if not clean_tickers:
        raise ValueError("At least one ticker is required to download prices.")

    raw = yf.download(
        tickers=clean_tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    prices = normalize_price_frame(raw, clean_tickers, price_field=price_field)

    missing_tickers = [ticker for ticker in clean_tickers if ticker not in prices.columns]
    for ticker in missing_tickers:
        try:
            single_raw = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                group_by="column",
                threads=False,
            )
            single_prices = normalize_price_frame(
                single_raw,
                [ticker],
                price_field=price_field,
            )
        except Exception:
            single_prices = pd.DataFrame()

        if ticker in single_prices.columns:
            prices = prices.join(single_prices[[ticker]], how="outer")

    available_order = [ticker for ticker in clean_tickers if ticker in prices.columns]
    return prices[available_order].sort_index()


def normalize_price_frame(
    raw_prices: pd.DataFrame,
    requested_tickers: Iterable[str],
    price_field: str = "Close",
) -> pd.DataFrame:
    """Convierte una respuesta de Yahoo en matriz fecha por ticker."""

    tickers = _unique_ordered(requested_tickers)
    if raw_prices.empty:
        return pd.DataFrame(columns=tickers, dtype=float)

    if isinstance(raw_prices.columns, pd.MultiIndex):
        level_zero = raw_prices.columns.get_level_values(0)
        if price_field in level_zero:
            prices = raw_prices[price_field].copy()
        elif "Adj Close" in level_zero:
            prices = raw_prices["Adj Close"].copy()
        else:
            available = ", ".join(sorted(set(map(str, level_zero))))
            raise ValueError(f"Price field not found. Available fields: {available}")
    else:
        if price_field not in raw_prices.columns:
            fallback = "Adj Close"
            if fallback not in raw_prices.columns:
                available = ", ".join(map(str, raw_prices.columns))
                raise ValueError(f"Price field not found. Available fields: {available}")
            price_field = fallback
        prices = raw_prices[[price_field]].copy()
        prices.columns = tickers[:1]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]

    for ticker in tickers:
        if ticker not in prices.columns:
            prices[ticker] = np.nan

    prices = prices[tickers]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all")
    prices = prices.dropna(axis=0, how="all")

    return prices


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos simples diarios desde una matriz limpia de precios.

    Se usan retornos simples porque son más directos para reportes de
    performance y comparación. Los retornos logarítmicos pueden agregarse si
    luego se necesita modelamiento específico.
    """

    if prices.empty:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    returns = prices.pct_change(fill_method=None)
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def build_quality_report(
    requested_tickers: Iterable[str],
    prices: pd.DataFrame,
) -> DataQualityReport:
    """Crea un reporte de calidad de datos para diagnóstico."""

    requested = _unique_ordered(requested_tickers)
    available = [ticker for ticker in requested if ticker in prices.columns]
    missing = [ticker for ticker in requested if ticker not in prices.columns]

    if prices.empty:
        start_date = None
        end_date = None
        observations = 0
        missing_ratio: dict[str, float] = {}
    else:
        start_date = prices.index.min().date().isoformat()
        end_date = prices.index.max().date().isoformat()
        observations = int(len(prices))
        missing_ratio = prices.isna().mean().round(4).to_dict()

    return DataQualityReport(
        requested_tickers=requested,
        available_tickers=available,
        missing_tickers=missing,
        start_date=start_date,
        end_date=end_date,
        observations=observations,
        missing_ratio_by_ticker=missing_ratio,
    )


def fetch_fund_metadata(tickers: Iterable[str]) -> pd.DataFrame:
    """Obtiene metadata descriptiva disponible desde Yahoo Finance.

    La metadata de Yahoo no siempre es consistente. Por eso cada campo se trata
    como opcional y se devuelve nulo cuando falta, en vez de detener todo el
    pipeline.
    """

    if yf is None:
        raise ImportError(
            "yfinance is not installed. Install project dependencies with "
            "`pip install -r requirements.txt`."
        )

    records: list[dict[str, object]] = []
    for ticker in _unique_ordered(tickers):
        info: dict[str, object] = {}
        fast_info: dict[str, object] = {}
        top_10_concentration = np.nan
        top_10_holdings = None
        portfolio_pe = np.nan
        portfolio_pb = np.nan

        try:
            yahoo_ticker = yf.Ticker(ticker)
            info = yahoo_ticker.info or {}
        except Exception:
            info = {}

        try:
            yahoo_ticker = yf.Ticker(ticker)
            fast = yahoo_ticker.fast_info
            fast_info = dict(fast) if fast is not None else {}
        except Exception:
            fast_info = {}

        try:
            funds_data = yf.Ticker(ticker).funds_data
            top_holdings = funds_data.top_holdings
            if isinstance(top_holdings, pd.DataFrame) and "Holding Percent" in top_holdings.columns:
                top_10_concentration = pd.to_numeric(
                    top_holdings["Holding Percent"].head(10),
                    errors="coerce",
                ).sum(min_count=1)
                top_10_holdings = serialize_top_holdings(top_holdings)

            equity_holdings = funds_data.equity_holdings
            if isinstance(equity_holdings, pd.DataFrame) and ticker in equity_holdings.columns:
                values = pd.to_numeric(equity_holdings[ticker], errors="coerce")
                raw_pe = values.get("Price/Earnings", np.nan)
                raw_pb = values.get("Price/Book", np.nan)
                portfolio_pe = _normalize_yahoo_valuation_ratio(raw_pe)
                portfolio_pb = _normalize_yahoo_valuation_ratio(raw_pb)
        except Exception:
            pass

        records.append(
            {
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName"),
                "quote_type": info.get("quoteType"),
                "category": info.get("category"),
                "fund_family": info.get("fundFamily"),
                "total_assets": info.get("totalAssets"),
                "expense_ratio": (
                    info.get("annualReportExpenseRatio")
                    or info.get("netExpenseRatio")
                    or info.get("expenseRatio")
                ),
                "currency": info.get("currency") or fast_info.get("currency"),
                "exchange": info.get("exchange"),
                "average_volume": info.get("averageVolume")
                or info.get("averageDailyVolume10Day")
                or fast_info.get("tenDayAverageVolume"),
                "last_price": fast_info.get("lastPrice") or info.get("regularMarketPrice"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "portfolio_pe": portfolio_pe,
                "portfolio_price_to_book": portfolio_pb,
                "return_on_equity": info.get("returnOnEquity"),
                "dividend_yield": info.get("dividendYield") or info.get("yield"),
                "top_10_concentration": top_10_concentration,
                "top_10_holdings": top_10_holdings,
                "metadata_available": bool(info or fast_info),
            }
        )

    return pd.DataFrame.from_records(records)


def serialize_top_holdings(top_holdings: pd.DataFrame, limit: int = 10) -> str | None:
    """Serializa holdings principales en JSON para snapshots locales."""

    if not isinstance(top_holdings, pd.DataFrame) or top_holdings.empty:
        return None

    records: list[dict[str, object]] = []
    for symbol, row in top_holdings.head(limit).iterrows():
        weight = pd.to_numeric(row.get("Holding Percent"), errors="coerce")
        records.append(
            {
                "symbol": str(symbol).upper() if pd.notna(symbol) else "",
                "name": row.get("Name"),
                "weight": float(weight) if pd.notna(weight) else np.nan,
            }
        )

    if not records:
        return None
    return json.dumps(records, ensure_ascii=False)


def fetch_top_holdings(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Obtiene una tabla de Top Holdings lista para mostrar.

    Se usa como respaldo cuando el snapshot local tiene concentración Top 10
    pero no incluye los nombres de las posiciones.
    """

    columns = ["symbol", "name", "weight"]
    if yf is None:
        return pd.DataFrame(columns=columns)

    try:
        funds_data = yf.Ticker(str(ticker).upper()).funds_data
        top_holdings = funds_data.top_holdings
    except Exception:
        return pd.DataFrame(columns=columns)

    if not isinstance(top_holdings, pd.DataFrame) or top_holdings.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, object]] = []
    for symbol, row in top_holdings.head(limit).iterrows():
        weight = pd.to_numeric(row.get("Holding Percent"), errors="coerce")
        records.append(
            {
                "symbol": str(symbol).upper() if pd.notna(symbol) else "",
                "name": row.get("Name"),
                "weight": float(weight) if pd.notna(weight) else np.nan,
            }
        )

    return pd.DataFrame.from_records(records, columns=columns)


def _normalize_yahoo_valuation_ratio(raw_value: object) -> float:
    """Normaliza métricas de valorización de Yahoo a múltiplos comparables.

    Algunas respuestas parecen venir como yield decimal. Si el valor es menor
    a 1, se invierte para aproximar un múltiplo; si ya parece múltiplo, se
    mantiene.
    """

    value = pd.to_numeric(raw_value, errors="coerce")
    if pd.isna(value) or value <= 0:
        return np.nan
    if value < 1:
        return float(1.0 / value)
    return float(value)


def build_data_snapshot(
    universe_path: str | Path,
    start: str = "2021-01-01",
    end: str | None = None,
    include_metadata: bool = True,
) -> FundDataBundle:
    """Construye el objeto base reutilizable para el análisis."""

    universe = load_fund_universe(universe_path)
    fund_tickers = universe["ticker"].tolist()
    benchmark_tickers = universe["benchmark_ticker"].tolist()
    requested_tickers = _unique_ordered([*fund_tickers, *benchmark_tickers])

    prices = download_price_history(requested_tickers, start=start, end=end)
    returns = calculate_daily_returns(prices)
    metadata = (
        fetch_fund_metadata(fund_tickers) if include_metadata else pd.DataFrame()
    )
    quality_report = build_quality_report(requested_tickers, prices)

    return FundDataBundle(
        universe=universe,
        prices=prices,
        returns=returns,
        metadata=metadata,
        quality_report=quality_report,
    )
