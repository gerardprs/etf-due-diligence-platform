"""Dashboard ejecutivo en Streamlit para selección de ETFs/fondos."""

from __future__ import annotations

import json
from html import escape
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fund_selection.data_loader import (  # noqa: E402
    build_data_snapshot,
    calculate_daily_returns,
    download_price_history,
    fetch_fund_metadata,
    fetch_top_holdings,
)
from fund_selection.alpha_vantage import enrich_metadata_with_alpha_vantage  # noqa: E402
from fund_selection.memo import generate_due_diligence_memo  # noqa: E402
from fund_selection.pipeline import (  # noqa: E402
    build_selection_analysis,
    build_selection_analysis_from_processed,
)


PROCESSED_DIR = ROOT / "data" / "processed"
RAW_UNIVERSE_PATH = ROOT / "data" / "raw" / "fund_universe.csv"
CHART_COLORWAY = [
    "#0b6f69",
    "#1f3a4a",
    "#a66f00",
    "#9f2f2f",
    "#356b9a",
    "#6b5b95",
    "#2f7d59",
    "#b36b00",
]
CHART_BG = "#ffffff"
CHART_GRID = "#e7ebef"
CHART_FONT = "#20242a"

MANDATE_PRESETS = {
    "US Large Cap Core": {
        "asset_class": "US Equity",
        "category": "US Large Cap Core",
        "description": "Exposición core a renta variable large cap de EE.UU.; el benchmark proxy es S&P 500 vía SPY.",
        "tickers": ["VOO", "IVV", "SPLG", "SCHX", "IWB", "OEF"],
        "benchmarks": {
            "VOO": "SPY",
            "IVV": "SPY",
            "SPLG": "SPY",
            "SCHX": "SPY",
            "IWB": "SPY",
            "OEF": "SPY",
        },
    },
    "US Growth / Technology": {
        "asset_class": "Growth Equity",
        "category": "US Growth / Technology",
        "description": "ETFs orientados a growth, Nasdaq 100 o tecnología; cada ETF se evalúa contra un proxy de estilo más cercano.",
        "tickers": ["QQQM", "SCHG", "VUG", "IWF", "XLK", "VGT", "FTEC", "IYW"],
        "benchmarks": {
            "QQQM": "QQQ",
            "SCHG": "VUG",
            "VUG": "IWF",
            "IWF": "VUG",
            "XLK": "VGT",
            "VGT": "XLK",
            "FTEC": "VGT",
            "IYW": "VGT",
        },
    },
    "Dividend Equity": {
        "asset_class": "Dividend Equity",
        "category": "Dividend Equity",
        "description": "ETFs de dividendos, calidad e income equity; se comparan contra proxies de dividend growth o dividend yield.",
        "tickers": ["SCHD", "VIG", "DGRO", "VYM", "DVY", "SDY", "NOBL", "HDV"],
        "benchmarks": {
            "SCHD": "VIG",
            "VIG": "DGRO",
            "DGRO": "VIG",
            "VYM": "DVY",
            "DVY": "VYM",
            "SDY": "NOBL",
            "NOBL": "SDY",
            "HDV": "VYM",
        },
    },
    "Core Fixed Income": {
        "asset_class": "Fixed Income",
        "category": "Core Fixed Income",
        "description": "Bloque core de renta fija agregado; útil para evaluar fondos amplios de bonos investment grade.",
        "tickers": ["BND", "AGG", "IUSB", "SCHZ", "BIV", "GOVT"],
        "benchmarks": {
            "BND": "AGG",
            "AGG": "BND",
            "IUSB": "AGG",
            "SCHZ": "AGG",
            "BIV": "AGG",
            "GOVT": "IEF",
        },
    },
    "Treasury Duration": {
        "asset_class": "Fixed Income",
        "category": "Treasury Duration",
        "description": "ETFs de Treasuries por tramo de duración; el benchmark proxy cambia según sensibilidad a tasas.",
        "tickers": ["SHY", "IEI", "IEF", "TLT", "VGSH", "VGIT", "VGLT"],
        "benchmarks": {
            "SHY": "IEI",
            "IEI": "SHY",
            "IEF": "IEI",
            "TLT": "VGLT",
            "VGSH": "SHY",
            "VGIT": "IEI",
            "VGLT": "TLT",
        },
    },
    "Investment Grade Credit": {
        "asset_class": "Fixed Income",
        "category": "Investment Grade Credit",
        "description": "Crédito corporativo investment grade; foco en costo, liquidez, duración y comportamiento relativo.",
        "tickers": ["LQD", "VCIT", "VCSH", "IGSB", "USIG"],
        "benchmarks": {
            "LQD": "VCIT",
            "VCIT": "LQD",
            "VCSH": "IGSB",
            "IGSB": "VCSH",
            "USIG": "LQD",
        },
    },
    "High Yield Credit": {
        "asset_class": "Fixed Income",
        "category": "High Yield Credit",
        "description": "ETFs high yield; el análisis debe tratar drawdown, liquidez y spread risk con mayor cuidado.",
        "tickers": ["HYG", "JNK", "SHYG", "SJNK", "HYLB"],
        "benchmarks": {
            "HYG": "JNK",
            "JNK": "HYG",
            "SHYG": "HYG",
            "SJNK": "JNK",
            "HYLB": "HYG",
        },
    },
}

ISSUER_LINKS = {
    "VOO": "https://investor.vanguard.com/investment-products/etfs/profile/voo",
    "IVV": "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf",
    "SPLG": "https://www.ssga.com/us/en/intermediary/etfs/spdr-portfolio-sp-500-etf-splg",
    "VTI": "https://investor.vanguard.com/investment-products/etfs/profile/vti",
    "ITOT": "https://www.ishares.com/us/products/239724/ishares-core-sp-total-us-stock-market-etf",
    "SCHX": "https://www.schwabassetmanagement.com/products/schx",
    "SCHG": "https://www.schwabassetmanagement.com/products/schg",
    "QQQM": "https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&ticker=QQQM",
    "SCHD": "https://www.schwabassetmanagement.com/products/schd",
    "VIG": "https://investor.vanguard.com/investment-products/etfs/profile/vig",
    "BND": "https://investor.vanguard.com/investment-products/etfs/profile/bnd",
    "AGG": "https://www.ishares.com/us/products/239458/ishares-core-total-us-bond-market-etf",
    "IUSB": "https://www.ishares.com/us/products/264615/ishares-core-total-usd-bond-market-etf",
    "LQD": "https://www.ishares.com/us/products/239566/ishares-iboxx-investment-grade-corporate-bond-etf",
    "VCIT": "https://investor.vanguard.com/investment-products/etfs/profile/vcit",
    "SHY": "https://www.ishares.com/us/products/239452/ishares-1-3-year-treasury-bond-etf",
    "IEI": "https://www.ishares.com/us/products/239455/ishares-3-7-year-treasury-bond-etf",
    "TLT": "https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf",
    "HYG": "https://www.ishares.com/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf",
    "BNDX": "https://investor.vanguard.com/investment-products/etfs/profile/bndx",
}


def _format_pct(value: object, decimals: int = 1) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric * 100:.{decimals}f}%"


def _format_pct_points(value: object, decimals: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}%"


def _format_money(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    if numeric >= 1_000_000_000:
        return f"${numeric / 1_000_000_000:,.1f}B"
    if numeric >= 1_000_000:
        return f"${numeric / 1_000_000:,.1f}M"
    return f"${numeric:,.0f}"


def _format_multiple(value: object, decimals: int = 1) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}x"


def _ensure_processed_files() -> None:
    required_files = [
        PROCESSED_DIR / "fund_universe_validated.csv",
        PROCESSED_DIR / "daily_returns.csv",
        PROCESSED_DIR / "fund_metadata.csv",
    ]
    if all(path.exists() for path in required_files):
        return

    snapshot = build_data_snapshot(
        universe_path=RAW_UNIVERSE_PATH,
        start="2021-01-01",
        include_metadata=True,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.universe.to_csv(PROCESSED_DIR / "fund_universe_validated.csv", index=False)
    snapshot.prices.to_csv(PROCESSED_DIR / "prices.csv", index_label="date")
    snapshot.returns.to_csv(PROCESSED_DIR / "daily_returns.csv", index_label="date")
    snapshot.metadata.to_csv(PROCESSED_DIR / "fund_metadata.csv", index=False)
    snapshot.quality_report.to_frame().to_csv(
        PROCESSED_DIR / "data_quality_summary.csv",
        index=False,
    )


@st.cache_data(ttl=3600)
def load_dashboard_data(risk_free_rate: float):
    _ensure_processed_files()
    return build_selection_analysis_from_processed(
        PROCESSED_DIR,
        risk_free_rate=risk_free_rate,
    )


def parse_tickers(ticker_text: str, max_tickers: int = 10) -> list[str]:
    """Convierte una lista de tickers separada por comas o espacios."""

    normalized = ticker_text.replace("\n", ",").replace(";", ",").replace(" ", ",")
    tickers: list[str] = []
    seen: set[str] = set()
    for raw_value in normalized.split(","):
        ticker = raw_value.strip().upper()
        if not ticker or ticker in seen:
            continue
        tickers.append(ticker)
        seen.add(ticker)
        if len(tickers) >= max_tickers:
            break
    return tickers


def build_from_master_snapshot(
    universe: pd.DataFrame,
    required_tickers: list[str],
    risk_free_rate: float,
):
    """Usa el universo curado predescargado cuando la data está disponible."""

    returns_path = PROCESSED_DIR / "daily_returns_master.csv"
    metadata_path = PROCESSED_DIR / "fund_metadata_master.csv"
    if not returns_path.exists() or not metadata_path.exists():
        return None

    returns = pd.read_csv(returns_path, parse_dates=["date"], index_col="date")
    returns.columns = [str(column).upper() for column in returns.columns]

    missing_columns = [ticker for ticker in required_tickers if ticker not in returns.columns]
    if missing_columns:
        return None

    metadata = pd.read_csv(metadata_path)
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    selected_funds = universe["ticker"].astype(str).str.upper().tolist()
    metadata = metadata.loc[metadata["ticker"].isin(selected_funds)].copy()
    metadata = enrich_metadata_with_alpha_vantage(
        metadata,
        project_root=ROOT,
        max_api_requests=3,
    )

    return build_selection_analysis(
        universe=universe,
        returns=returns[required_tickers],
        metadata=metadata,
        risk_free_rate=risk_free_rate,
    )


@st.cache_data(ttl=1800)
def load_custom_dashboard_data(
    ticker_text: str,
    mandate_label: str,
    risk_free_rate: float,
    start_date: str,
):
    """Construye el análisis completo para el mandato seleccionado."""

    preset = MANDATE_PRESETS[mandate_label]
    tickers = parse_tickers(ticker_text, max_tickers=10)
    benchmark_map = preset["benchmarks"]

    if not tickers:
        raise ValueError("Ingresa al menos un ETF válido.")

    benchmark_tickers = [benchmark_map[ticker] for ticker in tickers if ticker in benchmark_map]
    missing_benchmarks = [ticker for ticker in tickers if ticker not in benchmark_map]
    if missing_benchmarks:
        missing_text = ", ".join(missing_benchmarks)
        raise ValueError(f"No hay benchmark asignado para: {missing_text}")

    universe = pd.DataFrame(
        {
            "ticker": tickers,
            "name": pd.NA,
            "asset_class": preset["asset_class"],
            "benchmark_ticker": [benchmark_map[ticker] for ticker in tickers],
            "category": preset["category"],
            "role": f"Mandato - {mandate_label}",
        }
    )

    requested_tickers = [*tickers, *benchmark_tickers]
    unique_requested_tickers = list(dict.fromkeys(requested_tickers))
    master_bundle = build_from_master_snapshot(
        universe=universe,
        required_tickers=unique_requested_tickers,
        risk_free_rate=risk_free_rate,
    )
    if master_bundle is not None:
        return master_bundle

    prices = download_price_history(requested_tickers, start=start_date)
    available_funds = [ticker for ticker in tickers if ticker in prices.columns]
    if not available_funds:
        raise ValueError("No se pudo descargar precios para los ETFs ingresados.")

    missing_downloaded_benchmarks = [
        benchmark
        for benchmark in set(benchmark_tickers)
        if benchmark not in prices.columns
    ]
    if missing_downloaded_benchmarks:
        missing_text = ", ".join(sorted(missing_downloaded_benchmarks))
        raise ValueError(f"No se pudo descargar benchmark(s): {missing_text}.")

    universe = universe.loc[universe["ticker"].isin(available_funds)].reset_index(drop=True)
    returns = calculate_daily_returns(prices)
    metadata = fetch_fund_metadata(available_funds)
    metadata = enrich_metadata_with_alpha_vantage(
        metadata,
        project_root=ROOT,
        max_api_requests=3,
    )

    return build_selection_analysis(
        universe=universe,
        returns=returns,
        metadata=metadata,
        risk_free_rate=risk_free_rate,
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f7f9;
            --surface: #ffffff;
            --surface-2: #f3f5f7;
            --ink: #20242a;
            --muted: #626a73;
            --line: #d8dde3;
            --line-strong: #c5ccd4;
            --teal: #0b6f69;
            --teal-soft: #e7f3f1;
            --navy: #18313f;
            --gold: #a66f00;
            --gold-soft: #f6efe0;
            --red: #9f2f2f;
            --red-soft: #f6e7e7;
            --shadow: 0 10px 26px rgba(24, 49, 63, 0.08);
        }
        .stApp {
            background: var(--bg);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        h1 {
            font-size: 2.05rem;
            font-weight: 720;
            margin-bottom: 0.15rem;
        }
        h2 {
            font-size: 1.25rem;
            margin-top: 1rem;
        }
        h3 {
            font-size: 1rem;
        }
        .section-heading {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 1.25rem 0 0.45rem 0;
            font-size: 1.05rem;
            font-weight: 760;
            color: var(--ink);
        }
        .section-heading::before {
            content: "";
            width: 4px;
            height: 1.15rem;
            border-radius: 8px;
            background: var(--teal);
        }
        .hero-panel {
            background: linear-gradient(135deg, #173544 0%, #15313d 100%);
            border: 1px solid #244d5d;
            border-radius: 8px;
            padding: 1.35rem 1.45rem;
            box-shadow: var(--shadow);
            margin: 0.6rem 0 1rem 0;
        }
        .hero-kicker {
            color: #a9c4c1;
            font-size: 0.76rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }
        .hero-kicker a {
            color: #dbe6e8;
            text-decoration: none;
            border-bottom: 1px solid rgba(219,230,232,0.45);
        }
        .hero-kicker a:hover {
            color: #ffffff;
            border-bottom-color: #ffffff;
        }
        .hero-title {
            color: #ffffff;
            font-size: 1.45rem;
            font-weight: 780;
            line-height: 1.25;
            margin-bottom: 0.45rem;
        }
        .hero-copy {
            color: #dbe6e8;
            font-size: 0.96rem;
            line-height: 1.45;
            max-width: 980px;
            margin-bottom: 0.8rem;
        }
        .pill-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .pill {
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(255,255,255,0.08);
            color: #f4faf9;
            border-radius: 999px;
            padding: 0.32rem 0.62rem;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .workflow-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-top: 3px solid var(--teal);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            min-height: 118px;
            box-shadow: 0 6px 18px rgba(24, 49, 63, 0.05);
            margin-bottom: 0.75rem;
        }
        .workflow-card.gold {
            border-top-color: var(--gold);
        }
        .workflow-card.red {
            border-top-color: var(--red);
        }
        .workflow-number {
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 760;
            text-transform: uppercase;
        }
        .workflow-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 760;
            margin-top: 0.2rem;
        }
        .workflow-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.35;
            margin-top: 0.28rem;
        }
        .kpi-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 22px rgba(24, 49, 63, 0.06);
            min-height: 100px;
            margin-bottom: 0.55rem;
        }
        .kpi-card.gold {
            border-left-color: var(--gold);
        }
        .kpi-card.red {
            border-left-color: var(--red);
        }
        .kpi-card.slate {
            border-left-color: var(--navy);
        }
        .kpi-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 720;
            text-transform: uppercase;
        }
        .kpi-value {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 780;
            line-height: 1.2;
            margin-top: 0.3rem;
        }
        .kpi-helper {
            color: var(--muted);
            font-size: 0.8rem;
            margin-top: 0.28rem;
        }
        .analysis-strip {
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 6px 18px rgba(24, 49, 63, 0.05);
            margin: 0.55rem 0 0.95rem 0;
        }
        .analysis-strip-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 760;
        }
        .analysis-strip-copy {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.25rem;
        }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 8px 22px rgba(24, 49, 63, 0.06);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.85rem;
        }
        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.35rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 6px 18px rgba(24, 49, 63, 0.04);
        }
        .executive-ranking {
            display: grid;
            gap: 0.7rem;
            margin: 0.45rem 0 0.9rem 0;
        }
        .ranking-row {
            display: grid;
            grid-template-columns: 42px minmax(330px, 1.65fr) minmax(125px, 0.62fr) minmax(155px, 0.78fr) minmax(130px, 0.72fr);
            gap: 0.8rem;
            align-items: stretch;
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 0.82rem 0.9rem;
            box-shadow: 0 8px 22px rgba(24, 49, 63, 0.05);
        }
        .ranking-row.preferred { border-left-color: var(--teal); }
        .ranking-row.approved { border-left-color: var(--navy); }
        .ranking-row.watch { border-left-color: var(--gold); }
        .ranking-row.reject { border-left-color: var(--red); }
        .ranking-rank {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: var(--surface-2);
            border: 1px solid var(--line);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--ink);
            font-size: 0.78rem;
            font-weight: 780;
            align-self: center;
        }
        .ranking-ticker {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 780;
            line-height: 1.2;
        }
        .ranking-name {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.32;
            margin-top: 0.12rem;
        }
        .ranking-dd-snapshot {
            display: grid;
            gap: 0.48rem;
        }
        .ranking-dd-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.35rem;
        }
        .ranking-dd-cell {
            background: var(--surface-2);
            border: 1px solid var(--line);
            border-radius: 7px;
            padding: 0.34rem 0.42rem;
            min-width: 0;
        }
        .ranking-dd-cell .ranking-label {
            font-size: 0.62rem;
        }
        .ranking-dd-cell .ranking-value {
            font-size: 0.76rem;
            margin-top: 0.08rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .ranking-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 760;
            text-transform: uppercase;
        }
        .ranking-value {
            color: var(--ink);
            font-size: 0.86rem;
            font-weight: 680;
            line-height: 1.28;
            margin-top: 0.16rem;
        }
        .ranking-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
            background: var(--teal-soft);
            color: var(--teal);
            font-size: 0.76rem;
            font-weight: 760;
        }
        .ranking-pill.preferred { background: var(--teal-soft); color: var(--teal); }
        .ranking-pill.approved { background: #e8eef3; color: var(--navy); }
        .ranking-pill.watch { background: var(--gold-soft); color: var(--gold); }
        .ranking-pill.reject { background: var(--red-soft); color: var(--red); }
        .score-track {
            width: 100%;
            height: 8px;
            background: var(--surface-2);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 0.34rem;
        }
        .score-fill {
            height: 100%;
            background: var(--teal);
            border-radius: 999px;
        }
        .ranking-row.preferred .score-fill { background: var(--teal); }
        .ranking-row.approved .score-fill { background: var(--navy); }
        .ranking-row.watch .score-fill { background: var(--gold); }
        .ranking-row.reject .score-fill { background: var(--red); }
        .ranking-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.35rem;
        }
        .ranking-detail-card {
            background: #fbfcfd;
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            margin: 0.25rem 0 0.65rem 0;
        }
        .ranking-detail-card.approved { border-left-color: var(--navy); }
        .ranking-detail-card.watch { border-left-color: var(--gold); }
        .ranking-detail-card.reject { border-left-color: var(--red); }
        .ranking-detail-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 780;
            margin-bottom: 0.65rem;
        }
        .ranking-detail-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
        }
        .ranking-detail-cell {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.62rem 0.68rem;
        }
        @media (max-width: 1100px) {
            .ranking-row {
                grid-template-columns: 38px minmax(160px, 1fr);
            }
            .ranking-dd-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .ranking-metrics,
            .ranking-dd-snapshot,
            .ranking-status,
            .ranking-committee,
            .ranking-score {
                grid-column: 2 / -1;
            }
            .ranking-detail-grid {
                grid-template-columns: 1fr;
            }
        }
        section[data-testid="stSidebar"] {
            background: #eef1f4;
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] * {
            color: var(--ink);
        }
        div[data-testid="stTabs"] button {
            font-weight: 650;
            color: var(--muted);
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--teal);
            border-bottom-color: var(--teal);
        }
        div.stButton > button {
            background: var(--surface) !important;
            border: 1px solid var(--line-strong) !important;
            border-radius: 8px !important;
            color: var(--ink) !important;
            font-weight: 680;
        }
        div.stButton > button p {
            color: var(--ink) !important;
        }
        div.stButton > button[kind="primary"] {
            background: var(--teal) !important;
            border: 1px solid var(--teal) !important;
            border-radius: 8px !important;
            font-weight: 720;
        }
        div.stButton > button[kind="primary"] p {
            color: #ffffff !important;
        }
        div.stButton > button[kind="secondary"] {
            border-radius: 8px;
            border-color: var(--line-strong);
        }
        div[data-baseweb="select"] > div {
            background: var(--surface-2);
            border-color: var(--line);
            color: var(--ink);
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {
            color: var(--ink);
        }
        .stAlert {
            border-radius: 8px;
        }
        .source-link {
            display: block;
            width: 100%;
            padding: 0.82rem 0.9rem;
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            background: var(--surface);
            color: var(--teal) !important;
            text-align: center;
            text-decoration: none;
            font-weight: 720;
            box-shadow: 0 8px 22px rgba(24, 49, 63, 0.06);
        }
        .source-link:hover {
            border-color: var(--teal);
            background: var(--teal-soft);
            color: var(--teal) !important;
        }
        .spacer-md {
            height: 0.85rem;
        }
        .etf-description {
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            color: var(--muted);
            margin: 0.35rem 0 0.95rem 0;
            box-shadow: 0 8px 22px rgba(24, 49, 63, 0.06);
        }
        .etf-description strong {
            color: var(--ink);
        }
        .rail-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: var(--shadow);
            margin: 0.6rem 0 0.85rem 0;
        }
        .rail-kicker {
            color: var(--teal);
            font-size: 0.74rem;
            font-weight: 760;
            text-transform: uppercase;
        }
        .rail-title {
            color: var(--ink);
            font-size: 1.16rem;
            font-weight: 780;
            line-height: 1.25;
            margin-top: 0.2rem;
        }
        .rail-copy {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.38;
            margin-top: 0.45rem;
        }
        .rail-divider {
            height: 1px;
            background: var(--line);
            margin: 0.85rem 0;
        }
        .rail-metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.75rem;
        }
        .rail-metric {
            background: var(--surface-2);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.65rem 0.7rem;
        }
        .rail-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 720;
            text-transform: uppercase;
        }
        .rail-value {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 780;
            margin-top: 0.18rem;
        }
        .rail-note {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 0.55rem;
        }
        .rail-takeaway {
            background: #fbfcfd;
            border: 1px solid var(--line);
            border-left: 3px solid var(--teal);
            border-radius: 8px;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.36;
            margin-top: 0.65rem;
            padding: 0.68rem 0.72rem;
        }
        .rail-takeaway strong {
            color: var(--ink);
        }
        div[data-testid="column"]:has(.rail-panel) {
            position: sticky;
            top: 1.1rem;
            align-self: flex-start;
            max-height: calc(100vh - 1.6rem);
            overflow-y: auto;
            padding-bottom: 0.6rem;
            z-index: 20;
        }
        div[data-testid="column"]:has(.rail-panel) > div[data-testid="stVerticalBlock"] {
            gap: 0.72rem;
        }
        .result-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            box-shadow: var(--shadow);
            margin: 0.6rem 0 1rem 0;
        }
        .result-title {
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 780;
            line-height: 1.25;
        }
        .result-copy {
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.42;
            margin-top: 0.35rem;
        }
        .quick-read-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.85rem 0 0.2rem 0;
        }
        .quick-read-card {
            background: var(--surface-2);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.72rem 0.78rem;
            min-height: 76px;
        }
        .quick-read-label {
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 720;
            text-transform: uppercase;
        }
        .quick-read-value {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 760;
            margin-top: 0.2rem;
        }
        .ic-brief-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.8rem;
        }
        .ic-brief-card {
            background: #fbfcfd;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.72rem 0.78rem;
            min-height: 92px;
        }
        .ic-brief-label {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 760;
            text-transform: uppercase;
        }
        .ic-brief-value {
            color: var(--ink);
            font-size: 0.92rem;
            font-weight: 740;
            line-height: 1.25;
            margin-top: 0.22rem;
        }
        .ic-brief-note {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.3;
            margin-top: 0.3rem;
        }
        .pm-checklist {
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 4px solid var(--navy);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.65rem 0 1rem 0;
            box-shadow: 0 6px 18px rgba(24, 49, 63, 0.04);
        }
        .pm-checklist-title {
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 780;
            margin-bottom: 0.4rem;
        }
        .pm-checklist ul {
            margin: 0.2rem 0 0 1.05rem;
            padding: 0;
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.38;
        }
        .memo-decision-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.35rem 0 0.95rem 0;
            box-shadow: var(--shadow);
        }
        .memo-decision-kicker {
            color: var(--teal);
            font-size: 0.74rem;
            font-weight: 760;
            text-transform: uppercase;
        }
        .memo-decision-title {
            color: var(--ink);
            font-size: 1.18rem;
            font-weight: 780;
            line-height: 1.25;
            margin-top: 0.25rem;
        }
        .memo-decision-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.42;
            margin-top: 0.42rem;
        }
        .memo-decision-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.85rem;
        }
        .memo-decision-stat {
            background: var(--surface-2);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.68rem 0.72rem;
            min-height: 78px;
        }
        .memo-decision-label {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 760;
            text-transform: uppercase;
        }
        .memo-decision-value {
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 780;
            line-height: 1.25;
            margin-top: 0.22rem;
        }
        .memo-question-list {
            background: #fbfcfd;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.2rem 0 1rem 0;
        }
        .memo-question-title {
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 780;
            margin-bottom: 0.4rem;
        }
        .memo-question-list ul {
            margin: 0.2rem 0 0 1.05rem;
            padding: 0;
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.38;
        }
        @media (max-width: 900px) {
            div[data-testid="column"]:has(.rail-panel) {
                position: static;
                max-height: none;
                overflow-y: visible;
            }
            div[data-testid="column"]:has(.rail-panel) > div[data-testid="stVerticalBlock"] {
                gap: 0.7rem;
            }
            .quick-read-grid,
            .ic-brief-grid,
            .memo-decision-grid,
            .rail-metric-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def score_bar_chart(analysis: pd.DataFrame) -> go.Figure:
    colors = {
        "Preferido": "#0b6f69",
        "Aprobado": "#356b9a",
        "En observación": "#a66f00",
        "No prioritario": "#9f2f2f",
        "Requiere revisión": "#626a73",
    }
    ordered = analysis.sort_values("fund_selection_score", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ordered["fund_selection_score"],
            y=ordered["ticker"],
            orientation="h",
            marker_color=[colors.get(value, "#626a73") for value in ordered["recommendation"]],
            text=ordered["recommendation"],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
        )
    )
    chart_height = max(360, min(560, 92 * len(ordered) + 90))
    fig.update_layout(
        height=chart_height,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(range=[0, 100], title="Score de priorización"),
        yaxis=dict(title=""),
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_FONT),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=CHART_GRID)
    return fig


def cumulative_chart(
    cumulative: pd.DataFrame,
    selected_tickers: list[str],
    benchmark_map: dict[str, str],
) -> go.Figure:
    columns = resolve_comparison_columns(
        selected_tickers,
        benchmark_map,
        cumulative.columns,
    )

    fig = go.Figure()
    for index, column in enumerate(columns):
        dash = "dash" if column not in selected_tickers else "solid"
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative[column] * 100.0,
                mode="lines",
                name=column,
                line=dict(
                    width=2.6 if dash == "solid" else 1.8,
                    dash=dash,
                    color=CHART_COLORWAY[index % len(CHART_COLORWAY)],
                ),
                hovertemplate=f"<b>{column}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=10, b=25),
        yaxis_title="Retorno Acumulado",
        xaxis_title="",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=CHART_GRID, zeroline=False)
    return fig


def resolve_comparison_columns(
    selected_tickers: list[str],
    benchmark_map: dict[str, str],
    available_columns: pd.Index,
) -> list[str]:
    """Devuelve ETFs seleccionados y benchmarks, preservando el orden."""

    columns: list[str] = []
    for ticker in selected_tickers:
        columns.append(ticker)
        benchmark = benchmark_map.get(ticker)
        if benchmark:
            columns.append(benchmark)

    available = set(available_columns)
    return [
        column
        for index, column in enumerate(columns)
        if column in available and column not in columns[:index]
    ]


def date_window_from_preset(
    preset: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convierte un preset de fechas en una ventana válida."""

    if preset == "YTD":
        start = pd.Timestamp(year=max_date.year, month=1, day=1)
    elif preset in {"Último año", "Ultimo ano"}:
        start = max_date - pd.DateOffset(years=1)
    elif preset in {"Últimos 3 años", "Ultimos 3 anos"}:
        start = max_date - pd.DateOffset(years=3)
    else:
        start = min_date

    return max(start, min_date), max_date


def rebase_cumulative_window(
    cumulative: pd.DataFrame,
    columns: list[str],
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """Filtra retornos acumulados y rebasa cada serie a 0% al inicio.

    En un gráfico por periodo seleccionado, la comparación debe empezar desde
    cero. Si solo se recorta la serie original, se arrastra performance previa.
    """

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    window = cumulative.loc[
        (cumulative.index >= start) & (cumulative.index <= end),
        columns,
    ].copy()

    if window.empty:
        return window

    rebased = pd.DataFrame(index=window.index)
    for column in window.columns:
        series = window[column].dropna()
        if series.empty:
            rebased[column] = window[column]
            continue
        base_value = 1.0 + series.iloc[0]
        rebased[column] = (1.0 + window[column]) / base_value - 1.0

    return rebased


def drawdown_chart(drawdowns: pd.DataFrame, selected_ticker: str) -> go.Figure:
    fig = go.Figure()
    if selected_ticker in drawdowns.columns:
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns[selected_ticker] * 100.0,
                mode="lines",
                name=selected_ticker,
                fill="tozeroy",
                line=dict(color="#9f2f2f", width=1.8),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=10, b=25),
        yaxis_title="Drawdown",
        xaxis_title="",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_FONT),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=CHART_GRID, zeroline=False)
    return fig


def top_10_concentration_chart(
    row: pd.Series,
    holdings_table: pd.DataFrame | None = None,
) -> go.Figure:
    """Crea una dona con los cinco holdings principales."""

    concentration = pd.to_numeric(row.get("top_10_concentration"), errors="coerce")
    if holdings_table is not None and not holdings_table.empty:
        top_holdings = holdings_table.head(5).copy()
        top_holdings["Peso"] = pd.to_numeric(top_holdings["Peso"], errors="coerce")
        top_holdings = top_holdings.dropna(subset=["Peso"])
    else:
        top_holdings = pd.DataFrame(columns=["Ticker", "Peso"])

    if pd.isna(concentration) and not top_holdings.empty:
        concentration = top_holdings["Peso"].sum() / 100.0

    if pd.isna(concentration):
        labels = ["Dato no disponible"]
        values = [1.0]
        colors = ["#d8dde3"]
        center_text = "n/a"
    else:
        concentration = min(max(float(concentration), 0.0), 1.0)
        top_values = (top_holdings["Peso"] / 100.0).clip(lower=0.0).tolist()
        top_labels = top_holdings["Ticker"].astype(str).tolist()
        top_5_total = float(sum(top_values))
        other_top_10 = max(concentration - top_5_total, 0.0)
        rest_of_etf = max(1.0 - concentration, 0.0)

        labels = [*top_labels]
        values = [*top_values]
        if other_top_10 > 0.001:
            labels.append("Otros Top 10")
            values.append(other_top_10)
        labels.append("Resto del ETF")
        values.append(rest_of_etf)
        colors = [
            "#0b6f69",
            "#356b9a",
            "#a66f00",
            "#9f2f2f",
            "#6b5b95",
            "#8aa39b",
            "#d8dde3",
        ][: len(labels)]
        center_text = f"Top 10<br>{concentration * 100:.1f}%"

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(colors=colors),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        height=370,
        margin=dict(l=20, r=20, t=10, b=72),
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_FONT),
        annotations=[
            dict(
                text=center_text,
                x=0.5,
                y=0.5,
                font=dict(size=18, color=CHART_FONT),
                showarrow=False,
            )
        ],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=CHART_FONT),
        ),
    )
    return fig


def parse_serialized_holdings(value: object) -> pd.DataFrame:
    """Convierte el payload guardado de Top 10 holdings en tabla."""

    columns = ["symbol", "name", "weight"]
    if value is None:
        return pd.DataFrame(columns=columns)

    text_value = str(value).strip()
    if not text_value or text_value.lower() in {"nan", "<na>", "none"}:
        return pd.DataFrame(columns=columns)

    try:
        records = json.loads(text_value)
    except (TypeError, json.JSONDecodeError):
        return pd.DataFrame(columns=columns)

    if not isinstance(records, list):
        return pd.DataFrame(columns=columns)

    table = pd.DataFrame.from_records(records)
    if table.empty:
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in table.columns:
            table[column] = pd.NA

    table["symbol"] = table["symbol"].astype(str).str.upper()
    table["weight"] = pd.to_numeric(table["weight"], errors="coerce")
    return table[columns].head(10)


@st.cache_data(ttl=86400, show_spinner=False)
def load_top_holdings_table(ticker: str, serialized_holdings: object) -> pd.DataFrame:
    """Carga holdings desde payload local y usa Yahoo como respaldo puntual."""

    table = parse_serialized_holdings(serialized_holdings)
    if table.empty:
        table = fetch_top_holdings(ticker, limit=10)

    if table.empty:
        return pd.DataFrame(columns=["#", "Ticker", "Nombre", "Peso"])

    output = table.copy()
    output.insert(0, "#", range(1, len(output) + 1))
    output["Peso"] = pd.to_numeric(output["weight"], errors="coerce") * 100.0
    output = output.rename(columns={"symbol": "Ticker", "name": "Nombre"})
    return output[["#", "Ticker", "Nombre", "Peso"]]


def metric_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("CAGR", _format_pct(row.get("cagr")), "Crecimiento compuesto anualizado del ETF."),
            ("Alpha", _format_pct(row.get("alpha")), "Retorno ajustado por beta frente al benchmark."),
            ("Sortino Ratio", f"{pd.to_numeric(row.get('sortino_ratio'), errors='coerce'):.2f}", "Retorno ajustado solo por volatilidad negativa."),
            ("Max Drawdown", _format_pct(row.get("max_drawdown")), "Mayor caída histórica desde pico a valle."),
            ("Tracking Error", _format_pct(row.get("tracking_error")), "Desviación del ETF frente a su benchmark."),
            ("P/E", _format_multiple(row.get("valuation_pe")), "Valorización aproximada del portafolio del ETF."),
            ("ROE", _format_pct(row.get("return_on_equity")), "Rentabilidad sobre patrimonio de las compañías subyacentes, si está disponible."),
            ("Top 10 Concentration", _format_pct(row.get("top_10_concentration")), "Peso de las diez mayores posiciones del ETF."),
            ("TER", _format_pct_points(row.get("expense_ratio_pct")), "Total Expense Ratio anual del ETF según metadata disponible."),
            ("AUM", _format_money(row.get("total_assets")), "Activos bajo gestión; proxy de escala y estabilidad."),
            ("Volumen Promedio USD", _format_money(row.get("average_dollar_volume")), "Proxy de liquidez operativa diaria."),
            ("Siguiente paso", str(row.get("committee_status", "n/a")), "Lectura operativa para priorizar revisión, watchlist o descarte preliminar."),
        ],
        columns=["Métrica", "Valor", "Descripción"],
    )


def enrich_with_source_links(analysis: pd.DataFrame) -> pd.DataFrame:
    """Agrega URLs de fuente para verificación rápida."""

    enriched = analysis.copy()
    enriched["yahoo_finance_url"] = enriched["ticker"].map(
        lambda ticker: f"https://finance.yahoo.com/quote/{ticker}"
    )
    enriched["issuer_url"] = enriched["ticker"].map(
        lambda ticker: ISSUER_LINKS.get(str(ticker).upper(), "")
    )
    return enriched


def ensure_dashboard_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Asegura columnas opcionales aunque el caché sea antiguo."""

    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output


def ensure_committee_status(frame: pd.DataFrame) -> pd.DataFrame:
    """Completa el siguiente paso cuando Streamlit usa un análisis cacheado."""

    output = frame.copy()
    if "committee_status" not in output.columns:
        output["committee_status"] = pd.NA
    missing = output["committee_status"].isna() | (output["committee_status"].astype(str).str.strip() == "")
    if missing.any():
        output.loc[missing, "committee_status"] = output.loc[missing].apply(
            local_committee_status_from_row,
            axis=1,
        )
    return output


def local_committee_status_from_row(row: pd.Series) -> str:
    """Respaldo local para calcular el siguiente paso."""

    score = pd.to_numeric(row.get("fund_selection_score"), errors="coerce")
    red_flag_count = pd.to_numeric(row.get("red_flag_count"), errors="coerce")
    penalty = pd.to_numeric(row.get("red_flag_penalty"), errors="coerce")
    red_flag_count = 0 if pd.isna(red_flag_count) else int(red_flag_count)
    penalty = 0.0 if pd.isna(penalty) else float(penalty)

    if pd.isna(score):
        return "Revisión manual requerida"
    if score >= 80 and red_flag_count == 0:
        return "Priorizar revisión"
    if score >= 70 and penalty <= 6:
        return "Apto con validación cualitativa"
    if score >= 50:
        return "Watchlist / revisar supuestos"
    return "No priorizar salvo razón cualitativa"


def scroll_to_top_once(flag_name: str) -> None:
    """Reinicia el scroll una vez después de cambiar de etapa."""

    if st.session_state.get(flag_name):
        return
    st.html(
        """
        <script>
        const parentWindow = window.parent;
        const doc = parentWindow.document;
        const main = doc.querySelector('section.main');
        if (main) {
          main.scrollTo({ top: 0, left: 0, behavior: 'instant' });
        }
        parentWindow.scrollTo({ top: 0, left: 0, behavior: 'instant' });
        </script>
        """,
        unsafe_allow_javascript=True,
    )
    st.session_state[flag_name] = True


def render_section_heading(title: str) -> None:
    """Muestra un título compacto de sección."""

    st.markdown(
        f'<div class="section-heading">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: object,
    helper: str = "",
    accent: str = "teal",
) -> None:
    """Muestra una tarjeta KPI con color de acento controlado."""

    allowed_accents = {"teal", "gold", "red", "slate"}
    accent_class = accent if accent in allowed_accents else "teal"
    helper_html = (
        f'<div class="kpi-helper">{escape(str(helper))}</div>' if helper else ""
    )
    st.markdown(
        f"""
        <div class="kpi-card {accent_class}">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{escape(str(value))}</div>
            {helper_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_accent(value: object) -> str:
    """Convierte la recomendación en color de acento."""

    recommendation = str(value)
    if recommendation == "Preferido":
        return "teal"
    if recommendation == "Aprobado":
        return "slate"
    if recommendation == "En observación":
        return "gold"
    return "red"


def _numeric_value(value: object) -> float:
    """Convierte valores del dashboard a float manteniendo NaN."""

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def mandate_fit_label(row: pd.Series) -> tuple[str, str]:
    """Resume si el ETF cumple la exposición buscada."""

    tracking_error = _numeric_value(row.get("tracking_error"))
    r_squared = _numeric_value(row.get("r_squared"))
    if pd.notna(tracking_error) and tracking_error <= 0.015 and pd.notna(r_squared) and r_squared >= 0.95:
        return "Fit alto", "Tracking bajo y alta relación con benchmark."
    if pd.notna(tracking_error) and tracking_error <= 0.035:
        return "Fit razonable", "Cumple mandato, pero revisar tracking relativo."
    return "Revisar fit", "Validar benchmark, estrategia e índice subyacente."


def implementation_label(row: pd.Series) -> tuple[str, str]:
    """Resume calidad de implementación usando escala, liquidez y costo."""

    assets = _numeric_value(row.get("total_assets"))
    dollar_volume = _numeric_value(row.get("average_dollar_volume"))
    expense = _numeric_value(row.get("expense_ratio_pct"))
    liquid = pd.notna(assets) and assets >= 1_000_000_000 and pd.notna(dollar_volume) and dollar_volume >= 10_000_000
    cheap = pd.notna(expense) and expense <= 0.20
    if liquid and cheap:
        return "Implementación eficiente", "Escala, liquidez y TER compatibles con uso operativo."
    if liquid:
        return "Implementable", "Buena escala/liquidez; revisar costo vs peer group."
    return "Revisar ejecución", "Validar AUM, volumen, spreads y riesgo de cierre."


def concentration_label(row: pd.Series) -> tuple[str, str]:
    """Resume riesgo de concentración del ETF seleccionado."""

    top_10 = _numeric_value(row.get("top_10_concentration"))
    if pd.isna(top_10):
        return "Holdings por validar", "Confirmar Top 10 y metodología en factsheet."
    if top_10 <= 0.35:
        return "Concentración controlada", f"Top 10: {_format_pct(top_10)}."
    if top_10 <= 0.50:
        return "Concentración relevante", f"Top 10: {_format_pct(top_10)}; revisar nombres dominantes."
    return "Concentración alta", f"Top 10: {_format_pct(top_10)}; evaluar shocks idiosincráticos."


def render_ic_brief(row: pd.Series) -> None:
    """Muestra una lectura compacta antes del detalle."""

    mandate_label, mandate_note = mandate_fit_label(row)
    implementation, implementation_note = implementation_label(row)
    concentration, concentration_note = concentration_label(row)
    committee_status = str(row.get("committee_status", "Revisión manual requerida"))
    red_flag_value = pd.to_numeric(row.get("red_flag_count"), errors="coerce")
    red_flags = 0 if pd.isna(red_flag_value) else int(red_flag_value)
    status_note = "Sin alertas materiales." if red_flags == 0 else f"{red_flags} alerta(s) para validar."

    st.markdown(
        f"""
        <div class="ic-brief-grid">
            <div class="ic-brief-card">
                <div class="ic-brief-label">Mandato</div>
                <div class="ic-brief-value">{escape(mandate_label)}</div>
                <div class="ic-brief-note">{escape(mandate_note)}</div>
            </div>
            <div class="ic-brief-card">
                <div class="ic-brief-label">Implementación</div>
                <div class="ic-brief-value">{escape(implementation)}</div>
                <div class="ic-brief-note">{escape(implementation_note)}</div>
            </div>
            <div class="ic-brief-card">
                <div class="ic-brief-label">Concentración</div>
                <div class="ic-brief-value">{escape(concentration)}</div>
                <div class="ic-brief-note">{escape(concentration_note)}</div>
            </div>
            <div class="ic-brief-card">
                <div class="ic-brief-label">Siguiente paso</div>
                <div class="ic-brief-value">{escape(committee_status)}</div>
                <div class="ic-brief-note">{escape(status_note)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pm_checklist(row: pd.Series) -> None:
    """Muestra checks cualitativos pendientes después del screening."""

    ticker = str(row.get("ticker", "")).upper()
    benchmark = str(row.get("benchmark_ticker", "benchmark asignado"))
    st.markdown(
        f"""
        <div class="pm-checklist">
            <div class="pm-checklist-title">Checks pendientes antes de avanzar con {escape(ticker)}</div>
            <ul>
                <li>¿El índice y la metodología replican exactamente la exposición buscada frente a {escape(benchmark)}?</li>
                <li>¿El TER, AUM y volumen justifican usar este vehículo frente a peers más baratos o líquidos?</li>
                <li>¿La concentración Top 10 y el overlap con cartera existente crean exposición duplicada?</li>
                <li>¿Hay consideraciones de impuestos, UCITS/offshore, spreads o securities lending que cambien la implementación?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_memo_decision_summary(row: pd.Series) -> None:
    """Muestra la lectura preliminar antes del texto completo del memo."""

    ticker = str(row.get("ticker", "")).upper()
    score = _score_text(row.get("fund_selection_score"))
    recommendation = str(row.get("recommendation", "n/a"))
    committee_status = str(row.get("committee_status", "Revisión manual requerida"))
    benchmark = str(row.get("benchmark_ticker", "benchmark asignado"))
    alpha = _format_pct(row.get("alpha"))
    drawdown = _format_pct(row.get("max_drawdown"))
    ter = _format_pct_points(row.get("expense_ratio_pct"))
    top_10 = _format_pct(row.get("top_10_concentration"))
    mandate_label, mandate_note = mandate_fit_label(row)
    implementation, implementation_note = implementation_label(row)
    concentration, concentration_note = concentration_label(row)

    st.markdown(
        f"""
        <div class="memo-decision-panel">
            <div class="memo-decision-kicker">Lectura preliminar</div>
            <div class="memo-decision-title">{escape(ticker)}: {escape(committee_status)}</div>
            <div class="memo-decision-copy">
                Este borrador no es una recomendacion de compra. Resume si el ETF merece revision
                adicional frente a {escape(benchmark)}, considerando retorno ajustado por riesgo,
                benchmark fit, costo, liquidez, concentracion y alertas.
            </div>
            <div class="memo-decision-grid">
                <div class="memo-decision-stat">
                    <div class="memo-decision-label">Score</div>
                    <div class="memo-decision-value">{escape(score)} / 100</div>
                </div>
                <div class="memo-decision-stat">
                    <div class="memo-decision-label">Lectura</div>
                    <div class="memo-decision-value">{escape(recommendation)}</div>
                </div>
                <div class="memo-decision-stat">
                    <div class="memo-decision-label">Alpha / DD</div>
                    <div class="memo-decision-value">{escape(alpha)} / {escape(drawdown)}</div>
                </div>
                <div class="memo-decision-stat">
                    <div class="memo-decision-label">TER / Top 10</div>
                    <div class="memo-decision-value">{escape(ter)} / {escape(top_10)}</div>
                </div>
            </div>
        </div>
        <div class="memo-question-list">
            <div class="memo-question-title">Checks cualitativos para completar la revision</div>
            <ul>
                <li><strong>Mandato:</strong> {escape(mandate_label)}. {escape(mandate_note)}</li>
                <li><strong>ImplementaciÃ³n:</strong> {escape(implementation)}. {escape(implementation_note)}</li>
                <li><strong>ConcentraciÃ³n:</strong> {escape(concentration)}. {escape(concentration_note)}</li>
                <li><strong>Fuente oficial:</strong> confirmar TER, holdings, indice, spread y tratamiento tributario contra factsheet.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_external_link(label: str, url: object) -> None:
    """Muestra un enlace externo compatible con navegadores embebidos."""

    if pd.isna(url) or not str(url).strip():
        st.caption("Fuente oficial no mapeada para este ETF.")
        return

    safe_url = escape(str(url), quote=True)
    safe_label = escape(label)
    st.markdown(
        f"""
        <a href="{safe_url}" target="_blank" class="source-link">{safe_label}</a>
        """,
        unsafe_allow_html=True,
    )


def render_score_explanation() -> None:
    """Explica el score de priorización de forma breve."""

    st.subheader("Cómo se interpreta el score de priorización")
    st.markdown(
        """
        El score no intenta decir “compra este ETF”. Es un **ranking preliminar de due diligence** para priorizar qué fondos merecen revisión más profunda.

        Primero, cada ETF se evalúa contra su benchmark asignado. Luego, los ETFs seleccionados se comparan entre sí dentro del mandato elegido.
        """
    )

    weights = pd.DataFrame(
        [
            ("Performance", "25%", "CAGR y Sharpe: crecimiento y retorno ajustado por riesgo."),
            ("Riesgo", "25%", "Volatilidad, Sortino, max drawdown y CVaR."),
            ("Benchmark fit", "20%", "Tracking error, R² e information ratio vs benchmark."),
            ("Liquidez", "15%", "AUM y volumen promedio en dólares."),
            ("Costo", "15%", "Expense ratio relativo dentro del peer group."),
            ("Penalizaciones", "Hasta -30 pts", "Alertas por drawdown, tracking, costo, liquidez, P/E o concentración."),
        ],
        columns=["Componente", "Peso", "Qué captura"],
    )
    st.dataframe(weights, hide_index=True, width="stretch")

    st.markdown("**Fórmulas por componente:**")

    with st.expander("+ Performance Score"):
        st.markdown(
            """
            Evalúa crecimiento y retorno ajustado por riesgo.

            ```text
            Performance Score =
              65% * Score(CAGR)
            + 35% * Score(Sharpe Ratio)
            ```

            Donde mayor CAGR y mayor Sharpe reciben mejor puntaje relativo dentro del peer group.
            """
        )

    with st.expander("+ Risk Score"):
        st.markdown(
            """
            Evalúa control de riesgo total y riesgo de cola.

            ```text
            Risk Score =
              35% * Score(menor Volatilidad)
            + 25% * Score(mejor Max Drawdown)
            + 20% * Score(Sortino Ratio)
            + 20% * Score(menor CVaR 95%)
            ```

            Menor volatilidad, menor pérdida extrema y mejor retorno sobre downside risk aumentan el score.
            """
        )

    with st.expander("+ Benchmark Fit Score"):
        st.markdown(
            """
            Evalúa si el ETF cumple la exposición que promete frente a su benchmark asignado.

            ```text
            Benchmark Fit Score =
              45% * Score(menor Tracking Error)
            + 30% * Score(R²)
            + 25% * Score(Information Ratio)
            ```

            Un ETF core debería tener tracking error bajo y alta relación con su benchmark. El beta se calcula, pero no se premia automáticamente porque algunos mandatos smart beta o defensivos pueden buscar beta distinto de 1.
            """
        )

    with st.expander("+ Liquidity Score"):
        st.markdown(
            """
            Evalúa facilidad de implementación.

            ```text
            Liquidity Score =
              55% * Score(log10(AUM))
            + 45% * Score(log10(Avg. Dollar Volume))
            ```

            Se usa escala logarítmica para que ETFs muy grandes no distorsionen toda la comparación.
            """
        )

    with st.expander("+ Cost Score"):
        st.markdown(
            """
            Evalúa eficiencia de costos.

            ```text
            Cost Score = Score(menor Expense Ratio)
            ```

            Dentro de ETFs similares, menor expense ratio mejora el ranking, especialmente en exposiciones core.
            """
        )

    with st.expander("+ Penalizaciones por red flags"):
        st.markdown(
            """
            Las alertas restan puntos al score final.

            ```text
            Penalización Baja   = -3 puntos
            Penalización Media  = -6 puntos
            Penalización Alta   = -10 puntos

            Penalización máxima = -30 puntos
            ```

            Ejemplos: tracking error alto, drawdown elevado, CAGR negativo, baja liquidez, AUM bajo, costo alto, P/E elevado, concentración Top 10 o metadata faltante.
            """
        )

    with st.expander("+ Fórmula final"):
        st.markdown(
            """
            ```text
            Score de priorización =
              25% * Performance Score
            + 25% * Risk Score
            + 20% * Benchmark Fit Score
            + 15% * Liquidity Score
            + 15% * Cost Score
            - Red Flag Penalties
            ```

            El resultado prioriza fondos para revisión cualitativa. No debe interpretarse como recomendación automática de inversión.
            """
        )

    st.markdown(
        """
        **Interpretación:**

        - **Preferido:** candidato fuerte para revisión cualitativa.
        - **Aprobado:** cumple razonablemente el filtro cuantitativo.
        - **En observación:** requiere revisión adicional antes de avanzar.
        - **No prioritario:** no destaca dentro del peer group seleccionado.
        """
        )


def render_score_methodology_summary() -> None:
    """Muestra un resumen corto de la metodología del score."""

    st.markdown(
        """
        El score es un **ranking preliminar**, no una recomendación. Primero compara cada ETF contra su benchmark asignado usando retornos históricos; luego ordena los ETFs del mismo mandato.
        """
    )
    methodology = pd.DataFrame(
        [
            ("Performance", "25%", "CAGR y Sharpe"),
            ("Riesgo", "25%", "Volatilidad, Sortino, drawdown y CVaR"),
            ("Benchmark fit", "20%", "Tracking error, R² e information ratio"),
            ("Liquidez", "15%", "AUM y volumen promedio en dólares"),
            ("Costo", "15%", "Expense ratio"),
            ("Penalizaciones", "Hasta -30 pts", "Red flags de riesgo, liquidez, costo, P/E o concentración"),
        ],
        columns=["Bloque", "Peso", "Qué revisa"],
    )
    st.dataframe(methodology, hide_index=True, width="stretch")
    st.caption(
        "Fórmula resumida: score ponderado de los cinco bloques menos penalizaciones por red flags."
    )


def score_audit_table(row: pd.Series) -> pd.DataFrame:
    """Construye una tabla de contribución al score para un ETF."""

    rows = [
        ("Performance", 0.25, row.get("performance_score")),
        ("Riesgo", 0.25, row.get("risk_score")),
        ("Benchmark fit", 0.20, row.get("benchmark_fit_score")),
        ("Liquidez", 0.15, row.get("liquidity_score")),
        ("Costo", 0.15, row.get("cost_score")),
    ]
    audit = pd.DataFrame(rows, columns=["Componente", "Peso", "Score componente"])
    audit["Score componente"] = pd.to_numeric(audit["Score componente"], errors="coerce")
    audit["Contribución"] = audit["Peso"] * audit["Score componente"]
    penalty = pd.to_numeric(row.get("red_flag_penalty"), errors="coerce")
    penalty = 0.0 if pd.isna(penalty) else float(penalty)
    audit.loc[len(audit)] = ["Penalización red flags", np.nan, -penalty, -penalty]
    return audit


def render_score_audit(row: pd.Series) -> None:
    """Muestra la auditoría del cálculo del score."""

    with st.expander("Ver auditoría del cálculo del score"):
        st.markdown(
            f"""
            **ETF evaluado:** `{row.get("ticker")}` | **Score final:** `{pd.to_numeric(row.get("fund_selection_score"), errors="coerce"):.1f}`

            El score final sale de ponderar cinco bloques cuantitativos y restar penalizaciones por red flags.
            """
        )
        audit = score_audit_table(row)
        st.dataframe(
            audit,
            hide_index=True,
            width="stretch",
            column_config={
                "Componente": st.column_config.TextColumn("Componente"),
                "Peso": st.column_config.NumberColumn("Peso", format="%.0%"),
                "Score componente": st.column_config.NumberColumn("Score componente", format="%.1f"),
                "Contribución": st.column_config.NumberColumn("Contribución", format="%.1f"),
            },
        )
        st.caption(
            "Los scores por componente se normalizan de forma relativa dentro del peer group seleccionado. "
            "Esto permite auditar por qué un ETF lidera o queda rezagado."
        )


def render_etf_description(row: pd.Series) -> None:
    """Muestra una ficha corta del ETF seleccionado."""

    ticker = str(row.get("ticker", "")).upper()
    name = row.get("name")
    benchmark = row.get("benchmark_ticker")
    asset_class = row.get("asset_class")
    category = row.get("category")
    name_text = str(name) if pd.notna(name) and str(name).strip() else ticker
    detail_bits = [
        f"Benchmark asignado: {benchmark}" if pd.notna(benchmark) else "",
        f"Clase: {asset_class}" if pd.notna(asset_class) else "",
        f"Mandato: {category}" if pd.notna(category) else "",
    ]
    detail_text = " | ".join(bit for bit in detail_bits if bit)
    st.markdown(
        f"""
        <div class="etf-description">
            <strong>{escape(ticker)} - {escape(name_text)}</strong><br>
            {escape(detail_text)}. Esta vista resume performance, riesgo relativo, costo, liquidez y concentración del ETF seleccionado.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticker_risk_status(ticker: str, red_flags: pd.DataFrame) -> None:
    """Muestra el estado de riesgo de un ETF."""

    ticker_flags = red_flags.loc[red_flags["ticker"] == ticker] if not red_flags.empty else red_flags
    if ticker_flags.empty:
        st.success("Riesgo preliminar: sin alertas cuantitativas materiales detectadas.")
        return

    max_penalty = pd.to_numeric(ticker_flags["penalty"], errors="coerce").max()
    if max_penalty >= 10:
        st.error("Riesgo preliminar: alto. Revisar alertas antes de avanzar.")
    elif max_penalty >= 6:
        st.warning("Riesgo preliminar: medio. Requiere revisión adicional.")
    else:
        st.info("Riesgo preliminar: bajo, con observaciones menores.")

    st.dataframe(
        ticker_flags[["severity", "flag", "rationale"]],
        hide_index=True,
        width="stretch",
        column_config={
            "severity": st.column_config.TextColumn("Severidad"),
            "flag": st.column_config.TextColumn("Alerta"),
            "rationale": st.column_config.TextColumn("Racional"),
        },
    )


def vertical_ranking_detail_table(row: pd.Series) -> pd.DataFrame:
    """Construye una tabla vertical de detalle para un ETF."""

    return pd.DataFrame(
        [
            ("Ticker", str(row.get("ticker", "n/a"))),
            ("Nombre", str(row.get("name", "n/a"))),
            ("Clase de activo", str(row.get("asset_class", "n/a"))),
            ("Benchmark asignado", str(row.get("benchmark_ticker", "n/a"))),
            ("Recomendación", str(row.get("recommendation", "n/a"))),
            ("Siguiente paso", str(row.get("committee_status", "n/a"))),
            ("Score", _score_text(row.get("fund_selection_score"))),
            ("CAGR", _format_pct(row.get("cagr"))),
            ("Volatilidad", _format_pct(row.get("volatility"))),
            ("Sharpe", f"{pd.to_numeric(row.get('sharpe_ratio'), errors='coerce'):.2f}"),
            ("Max Drawdown", _format_pct(row.get("max_drawdown"))),
            ("Tracking Error", _format_pct(row.get("tracking_error"))),
            ("Alpha", _format_pct(row.get("alpha"))),
            ("P/E", _format_multiple(row.get("valuation_pe"))),
            ("Top 10 Concentration", _format_pct(row.get("top_10_concentration"))),
            ("TER", _format_pct_points(row.get("expense_ratio_pct"))),
            ("AUM", _format_money(row.get("total_assets"))),
            ("Volumen USD promedio", _format_money(row.get("average_dollar_volume"))),
            ("Alertas", str(row.get("red_flag_count", "n/a"))),
            ("Yahoo Finance", str(row.get("yahoo_finance_url", ""))),
            ("Emisor / factsheet", str(row.get("issuer_url", ""))),
        ],
        columns=["Campo", "Valor"],
    )


def recommendation_style_class(recommendation: object, score: object | None = None) -> str:
    """Convierte la recomendación en una clase visual."""

    recommendation_text = str(recommendation)
    score_value = pd.to_numeric(score, errors="coerce")
    if recommendation_text == "Preferido":
        return "preferred"
    if recommendation_text == "Aprobado":
        return "approved"
    if recommendation_text == "En observación":
        return "watch"
    if recommendation_text == "No prioritario":
        return "reject"
    if pd.notna(score_value) and float(score_value) < 50:
        return "reject"
    if pd.notna(score_value) and float(score_value) < 70:
        return "watch"
    return "approved"


def _ranking_snapshot_cells(row: pd.Series) -> str:
    """Muestra datos clave dentro de una fila del ranking."""

    facts = [
        ("Benchmark", row.get("benchmark_ticker", "n/a")),
        ("CAGR", _format_pct(row.get("cagr"))),
        ("Max DD", _format_pct(row.get("max_drawdown"))),
        ("TER", _format_pct_points(row.get("expense_ratio_pct"))),
        ("Vol. USD", _format_money(row.get("average_dollar_volume"))),
        ("Alertas", str(row.get("red_flag_count", "n/a"))),
    ]
    cells = []
    for label, value in facts:
        cells.append(
            f"""
            <div class="ranking-dd-cell">
                <div class="ranking-label">{escape(str(label))}</div>
                <div class="ranking-value">{escape(str(value))}</div>
            </div>
            """
        )
    return "".join(cells)


def render_vertical_ranking_details(ranking: pd.DataFrame) -> None:
    """Muestra detalles por ETF en formato vertical."""

    st.caption(
        "Ficha complementaria por ETF para revisar benchmark, liquidez, costo y fuentes sin scroll lateral."
    )
    for _, row in ranking.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        score = _score_text(row.get("fund_selection_score"))
        recommendation = str(row.get("recommendation", "n/a"))
        style_class = recommendation_style_class(
            recommendation,
            row.get("fund_selection_score"),
        )
        with st.expander(f"{ticker} | {recommendation} | Score {score}", expanded=False):
            detail_cells = []
            for field, value in vertical_ranking_detail_table(row).itertuples(index=False):
                detail_cells.append(
                    f"""
                    <div class="ranking-detail-cell">
                        <div class="ranking-label">{escape(str(field))}</div>
                        <div class="ranking-value">{escape(str(value))}</div>
                    </div>
                    """
                )
            st.html(
                f"""
                <div class="ranking-detail-card {style_class}">
                    <div class="ranking-detail-title">{escape(ticker)} - detalle de due diligence</div>
                    <div class="ranking-detail-grid">{''.join(detail_cells)}</div>
                </div>
                """
            )


def render_executive_ranking(ranking: pd.DataFrame) -> None:
    """Muestra el ranking ejecutivo sin scroll horizontal."""

    rows: list[str] = []
    ordered = ranking.reset_index(drop=True)
    for index, row in ordered.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        name = str(row.get("name", ticker))
        recommendation = str(row.get("recommendation", "n/a"))
        committee = str(row.get("committee_status", "n/a"))
        score = pd.to_numeric(row.get("fund_selection_score"), errors="coerce")
        score_value = 0.0 if pd.isna(score) else float(score)
        score_width = min(max(score_value, 0.0), 100.0)
        row_class = recommendation_style_class(recommendation, score_value)
        rows.append(
            f"""
            <div class="ranking-row {row_class}">
                <div class="ranking-rank">{index + 1}</div>
                <div class="ranking-dd-snapshot">
                    <div>
                        <div class="ranking-ticker">{escape(ticker)}</div>
                        <div class="ranking-name">{escape(name)}</div>
                    </div>
                    <div class="ranking-dd-grid">
                        {_ranking_snapshot_cells(row)}
                    </div>
                </div>
                <div class="ranking-status">
                    <div class="ranking-label">Lectura</div>
                    <div class="ranking-value">
                        <span class="ranking-pill {row_class}">{escape(recommendation)}</span>
                    </div>
                </div>
                <div class="ranking-committee">
                    <div class="ranking-label">Siguiente paso</div>
                    <div class="ranking-value">{escape(committee)}</div>
                </div>
                <div class="ranking-score">
                    <div class="ranking-label">Score</div>
                    <div class="ranking-value">{score_value:.1f} / 100</div>
                    <div class="score-track"><div class="score-fill" style="width: {score_width:.1f}%"></div></div>
                </div>
            </div>
            """
        )

    st.html(f'<div class="executive-ranking">{"".join(rows)}</div>')


def render_intro(compact: bool = False) -> None:
    """Muestra la introducción antes del flujo de análisis."""

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">
                Hecho por <a href="https://www.linkedin.com/in/gerardosparedesromero25" target="_blank">Gerardo Paredes Romero</a>
                | Data Analytics · Python Automation · Investment Analytics
            </div>
            <div class="hero-title">Revisión y Selección de ETFs</div>
            <div class="hero-copy">
                Herramienta de investment analytics en Python para automatizar el primer filtro de ETFs:
                mandato, peer group, benchmark por fondo, score de priorización, alertas y memo preliminar.
                Proyecto educativo, no recomendación de inversión.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    workflow_cards = [
        """
        <div class="workflow-card">
            <div class="workflow-number">Paso 1</div>
            <div class="workflow-title">Mandato</div>
            <div class="workflow-copy">Parte de la pregunta real de cartera: qué exposición necesita cubrir el mandato.</div>
        </div>
        """,
        """
        <div class="workflow-card gold">
            <div class="workflow-number">Paso 2</div>
            <div class="workflow-title">Benchmark Fit</div>
            <div class="workflow-copy">Compara retornos diarios vs benchmark para medir tracking, alpha y riesgo relativo.</div>
        </div>
        """,
        """
        <div class="workflow-card red">
            <div class="workflow-number">Paso 3</div>
            <div class="workflow-title">Output del screening</div>
            <div class="workflow-copy">Entrega ranking, alertas y borrador de memo para priorizar revisión del analista.</div>
        </div>
        """,
    ]
    if compact:
        for card in workflow_cards:
            st.markdown(card, unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns(3)
        for column, card in zip([c1, c2, c3], workflow_cards):
            column.markdown(card, unsafe_allow_html=True)

    with st.expander("Ver workflow automatizado"):
        st.markdown(
            """
            1. Selecciona mandato y peer group.
            2. Asigna benchmark por ETF.
            3. Calcula riesgo, performance, costo y liquidez.
            4. Prioriza ranking, alertas y memo preliminar.

            **Valor operativo:** convierte un screening repetitivo de Excel en un flujo reproducible y auditable.
            """
        )

    with st.expander("Ver metodología resumida del score"):
        render_score_methodology_summary()

    with st.expander("Ver fuente de datos y disclaimer"):
        st.markdown(
            """
            La app usa fuentes públicas y una capa local de estabilidad:

            - **Yahoo Finance vía `yfinance`:** precios históricos, retornos, volumen, AUM y metadata disponible.
            - **Alpha Vantage ETF Profile:** respaldo para holdings y metadata faltante.
            - **Snapshot local:** evita fallas por límites temporales de APIs públicas.

            En un flujo real, el analista validaría TER, AUM, benchmark y holdings contra el factsheet oficial.
            **No constituye recomendación de inversión.**
            """
        )


def render_screening_controls() -> tuple[str, list[str]]:
    """Muestra controles guiados para seleccionar el universo."""

    render_section_heading("Configura una demo rápida")
    st.markdown(
        """
        <div class="analysis-strip">
            <div class="analysis-strip-title">Selección guiada de universo comparable</div>
            <div class="analysis-strip-copy">
                Puedes dejar la selección por defecto y presionar Continuar al análisis. La app usará el mandato
                para asignar benchmarks y construir el peer group.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mandate = st.selectbox(
        "1. Selecciona el mandato de inversión",
        options=list(MANDATE_PRESETS.keys()),
        index=0,
        help="Un fund selector parte del mandato: qué exposición quiere cubrir en cartera.",
    )
    preset = MANDATE_PRESETS[mandate]
    available_tickers = preset["tickers"]

    col1, col2 = st.columns([0.75, 1.25])
    with col1:
        compare_count = st.slider(
            "2. Cantidad de ETFs a comparar",
            min_value=3,
            max_value=len(available_tickers),
            value=min(5, len(available_tickers)),
            help="Para una demo rápida, 3 a 5 ETFs suelen ser suficientes. Puedes ampliar hasta 10.",
        )
        st.info(
            "La app asigna un benchmark por ETF. No se usa un benchmark genérico para todo el mercado."
        )

    with col2:
        selected_tickers = st.multiselect(
            "3. Selecciona los ETFs del universo curado",
            options=available_tickers,
            default=available_tickers[:compare_count],
            key=f"selected_etfs_{mandate}_{compare_count}",
            max_selections=10,
            help="La app ejecutará todo el análisis solo sobre los ETFs seleccionados.",
        )
        st.caption(preset["description"])

    if len(selected_tickers) < 3:
        st.warning("Selecciona al menos 3 ETFs para que la comparación sea útil.")
        st.stop()

    if len(selected_tickers) != compare_count:
        st.info(
            f"Cantidad objetivo: {compare_count}. ETFs seleccionados actualmente: {len(selected_tickers)}."
        )

    benchmark_preview = pd.DataFrame(
        {
            "ETF": selected_tickers,
            "Benchmark asignado": [preset["benchmarks"][ticker] for ticker in selected_tickers],
            "Mandato": preset["category"],
        }
    )
    st.dataframe(benchmark_preview, hide_index=True, width="stretch")

    return mandate, selected_tickers


def setup_step(risk_free_rate: float, start_date: str) -> None:
    """Muestra la configuración y guarda el universo seleccionado."""

    mandate_label, selected_tickers = render_screening_controls()

    st.markdown("---")
    c1, c2 = st.columns([0.32, 0.68])
    with c1:
        continue_clicked = st.button(
            "Continuar al análisis",
            type="primary",
            width="stretch",
        )
    with c2:
        st.caption(
            "Al continuar, la app descargará datos, calculará métricas, ranking, alertas y memo."
        )

    if continue_clicked:
        st.session_state["analysis_config"] = {
            "risk_free_rate": risk_free_rate,
            "start_date": start_date,
            "mandate_label": mandate_label,
            "selected_tickers": selected_tickers,
        }
        st.session_state["analysis_ready"] = True
        st.session_state["analysis_scroll_reset"] = False
        st.rerun()


def render_analysis_header(config: dict[str, object]) -> None:
    """Muestra un resumen compacto del universo seleccionado."""

    left, right = st.columns([0.72, 0.28])
    with left:
        st.markdown(
            f"""
            <div class="analysis-strip">
                <div class="analysis-strip-title">Análisis generado</div>
                <div class="analysis-strip-copy">
                    Mandato: {escape(str(config['mandate_label']))} | Benchmark: asignado por ETF | ETFs:
                    {escape(", ".join(config['selected_tickers']))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Retroceder y ajustar selección", width="stretch"):
            st.session_state["analysis_ready"] = False
            st.session_state["analysis_scroll_reset"] = False
            st.rerun()


def _score_text(value: object) -> str:
    """Formatea un score para la interfaz."""

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.1f}"


def build_leader_takeaway(row: pd.Series, leader_red_flags: int) -> str:
    """Construye una lectura breve del ETF líder."""

    ticker = str(row.get("ticker", "")).upper()
    recommendation = str(row.get("recommendation", ""))
    cagr = _format_pct(row.get("cagr"))
    max_drawdown = _format_pct(row.get("max_drawdown"))
    ter = _format_pct_points(row.get("expense_ratio_pct"))
    alpha = _format_pct(row.get("alpha"))
    committee_status = str(row.get("committee_status", "revisión manual"))
    alert_text = (
        "sin alertas cuantitativas materiales"
        if leader_red_flags == 0
        else f"con {leader_red_flags} alerta(s) para revisar"
    )
    return (
        f"{ticker} queda como {recommendation.lower()} por score relativo, "
        f"CAGR {cagr}, alpha {alpha}, max drawdown {max_drawdown}, TER {ter} y {alert_text}. "
        f"Siguiente paso: {committee_status}."
    )


def render_result_headline(
    config: dict[str, object],
    top: pd.Series,
    scope_count: int,
    leader_red_flags: int,
) -> None:
    """Muestra la primera lectura del análisis."""

    takeaway = build_leader_takeaway(top, leader_red_flags)
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-title">Resultado del screening</div>
            <div class="result-copy">
                Lectura rápida del universo: ETF líder, score, alertas y siguiente paso de revisión.
            </div>
            <div class="result-copy"><strong>30-second read:</strong> {escape(takeaway)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_rail(
    config: dict[str, object],
    top: pd.Series,
    leader_red_flags: int,
    scope_tickers: list[str],
) -> str:
    """Muestra el panel lateral con contexto y controles."""

    ticker = str(top.get("ticker", "")).upper()
    score = _score_text(top.get("fund_selection_score"))
    recommendation = str(top.get("recommendation", "n/a"))
    selected = ", ".join(scope_tickers)
    takeaway = build_leader_takeaway(top, leader_red_flags)
    st.markdown(
        f"""
        <div class="rail-panel">
            <div class="rail-kicker">Automatización de revisión de ETFs</div>
            <div class="rail-title">Vista rápida</div>
            <div class="rail-copy">
                Primer filtro automatizado de ETFs para ranking, alertas y memo preliminar.
            </div>
            <div class="rail-divider"></div>
            <div class="rail-copy">
                <strong>Mandato:</strong> {escape(str(config["mandate_label"]))}<br>
                <strong>ETFs:</strong> {escape(selected)}
            </div>
            <div class="rail-metric-grid">
                <div class="rail-metric">
                    <div class="rail-label">Líder</div>
                    <div class="rail-value">{escape(ticker)}</div>
                </div>
                <div class="rail-metric">
                    <div class="rail-label">Score</div>
                    <div class="rail-value">{escape(score)}</div>
                </div>
                <div class="rail-metric">
                    <div class="rail-label">Status</div>
                    <div class="rail-value">{escape(recommendation)}</div>
                </div>
                <div class="rail-metric">
                    <div class="rail-label">Alertas</div>
                    <div class="rail-value">{leader_red_flags}</div>
                </div>
            </div>
            <div class="rail-takeaway">
                <strong>30-second read:</strong> {escape(takeaway)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Retroceder y ajustar selección", width="stretch", key="rail_adjust_selection"):
        st.session_state["analysis_ready"] = False
        st.session_state["analysis_scroll_reset"] = False
        st.rerun()

    render_score_audit(top)

    return ticker


def main() -> None:
    st.set_page_config(
        page_title="Plataforma de Selección de ETFs",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_theme()
    st.session_state.setdefault("analysis_ready", False)
    st.session_state.setdefault("analysis_config", None)
    with st.sidebar:
        st.header("Parámetros")
        risk_free_rate = st.number_input(
            "Tasa libre de riesgo",
            min_value=0.0,
            max_value=0.15,
            value=0.0,
            step=0.005,
            format="%.3f",
        )
        min_score = st.slider("Score mínimo para tabla", 0, 100, 0, 5)
        start_date = st.text_input("Fecha inicial", value="2021-01-01")

    if not st.session_state["analysis_ready"]:
        st.session_state["analysis_scroll_reset"] = False
        intro_col, setup_col = st.columns([0.42, 0.58], gap="large")
        with intro_col:
            render_intro(compact=True)
        with setup_col:
            setup_step(risk_free_rate=risk_free_rate, start_date=start_date)
        return

    config = st.session_state["analysis_config"]
    if not config or "mandate_label" not in config:
        st.session_state["analysis_ready"] = False
        st.session_state["analysis_config"] = None
        st.session_state["analysis_scroll_reset"] = False
        st.rerun()

    scroll_to_top_once("analysis_scroll_reset")

    selected_tickers = config["selected_tickers"]
    try:
        with st.spinner("Descargando datos y ejecutando due diligence..."):
            bundle = load_custom_dashboard_data(
                ticker_text=", ".join(config["selected_tickers"]),
                mandate_label=config["mandate_label"],
                risk_free_rate=config["risk_free_rate"],
                start_date=config["start_date"],
            )
    except Exception as error:
        st.error(f"No se pudo construir el análisis: {error}")
        st.stop()

    analysis = bundle.analysis.copy()
    analysis = enrich_with_source_links(analysis)
    dashboard_optional_columns = [
        "valuation_pe",
        "return_on_equity",
        "top_10_concentration",
        "top_10_holdings",
        "alpha",
        "committee_status",
    ]
    analysis = ensure_dashboard_columns(analysis, dashboard_optional_columns)
    analysis = ensure_committee_status(analysis)
    red_flags = bundle.red_flags.copy()
    cumulative = bundle.cumulative_returns.copy()
    drawdowns = bundle.drawdowns.copy()

    analysis_scope = analysis.copy()
    if analysis_scope.empty:
        st.warning("No hay ETFs disponibles para el universo seleccionado.")
        st.stop()

    scope_tickers = analysis_scope["ticker"].tolist()
    red_flags = red_flags.loc[red_flags["ticker"].isin(scope_tickers)].copy()
    benchmark_map = dict(zip(analysis["ticker"], analysis["benchmark_ticker"]))
    filtered = analysis_scope.loc[analysis_scope["fund_selection_score"] >= min_score].copy()

    top = analysis_scope.iloc[0]
    leader_red_flags = int(top.get("red_flag_count", 0))

    rail_col, output_col = st.columns([0.30, 0.70], gap="large")
    with rail_col:
        memo_ticker = render_analysis_rail(config, top, leader_red_flags, scope_tickers)

    with output_col:
        tab_overview, tab_detail, tab_memo = st.tabs(
            ["Resumen Ejecutivo", "Análisis por ETF", "Memo preliminar"]
        )

        with tab_overview:
            render_section_heading("Ranking de Selección")
            st.plotly_chart(score_bar_chart(filtered), width="stretch")

            render_section_heading("Comparación de Performance")
            comparison_columns = resolve_comparison_columns(
                selected_tickers,
                benchmark_map,
                cumulative.columns,
            )
            comparison_base = cumulative[comparison_columns].dropna(how="all")

            if comparison_base.empty:
                st.warning("No hay datos suficientes para graficar la comparación.")
            else:
                min_chart_date = pd.Timestamp(comparison_base.index.min())
                max_chart_date = pd.Timestamp(comparison_base.index.max())
                default_range = (min_chart_date.date(), max_chart_date.date())
                range_key = f"overview_date_range_{config['mandate_label']}"
                chart_start, chart_end = st.session_state.get(range_key, default_range)

                rebased_cumulative = rebase_cumulative_window(
                    cumulative,
                    comparison_columns,
                    chart_start,
                    chart_end,
                )
                st.plotly_chart(
                    cumulative_chart(rebased_cumulative, selected_tickers, benchmark_map),
                    width="stretch",
                )
                st.caption(
                    "Las curvas se rebajan a 0% al inicio del periodo seleccionado. "
                    "En mandatos muy similares, como US Large Cap Core, es normal que las líneas se vean casi iguales."
                )
                st.slider(
                    "Rango de fechas del gráfico",
                    min_value=min_chart_date.date(),
                    max_value=max_chart_date.date(),
                    value=(chart_start, chart_end),
                    format="YYYY-MM-DD",
                    key=range_key,
                )

            ranking_columns = [
                "ticker",
                "name",
                "asset_class",
                "benchmark_ticker",
                "recommendation",
                "committee_status",
                "fund_selection_score",
                "cagr",
                "volatility",
                "sharpe_ratio",
                "max_drawdown",
                "tracking_error",
                "alpha",
                "valuation_pe",
                "top_10_concentration",
                "expense_ratio_pct",
                "total_assets",
                "average_dollar_volume",
                "red_flag_count",
                "yahoo_finance_url",
                "issuer_url",
            ]
            render_section_heading("Tabla Ejecutiva de Ranking")
            render_executive_ranking(filtered[ranking_columns])

            if not red_flags.empty:
                render_section_heading("Alertas")
                st.dataframe(
                    red_flags,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "ticker": st.column_config.TextColumn("Ticker"),
                        "severity": st.column_config.TextColumn("Severidad"),
                        "flag": st.column_config.TextColumn("Alerta"),
                        "rationale": st.column_config.TextColumn("Racional"),
                        "penalty": st.column_config.NumberColumn("Penalización", format="%.1f"),
                    },
                )

        with tab_detail:
            selected_detail = st.selectbox(
                "ETF a analizar",
                options=scope_tickers,
                index=scope_tickers.index(memo_ticker) if memo_ticker in scope_tickers else 0,
            )
            detail_row = analysis.loc[analysis["ticker"] == selected_detail].iloc[0]
            render_etf_description(detail_row)

            c1, c2, c3 = st.columns(3)
            with c1:
                render_kpi_card("Score", f"{detail_row['fund_selection_score']:.1f}", "Ranking relativo", "teal")
            with c2:
                render_kpi_card("CAGR", _format_pct(detail_row.get("cagr")), "Retorno anualizado", "slate")
            with c3:
                render_kpi_card(
                    "Sharpe",
                    f"{pd.to_numeric(detail_row.get('sharpe_ratio'), errors='coerce'):.2f}",
                    "Retorno/riesgo",
                    "gold",
                )
            c4, c5 = st.columns(2)
            with c4:
                render_kpi_card("Costo", _format_pct_points(detail_row.get("expense_ratio_pct")), "TER anual", "teal")
            with c5:
                render_kpi_card("AUM", _format_money(detail_row.get("total_assets")), "Escala del fondo", "slate")

            st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
            source_cols = st.columns(2)
            with source_cols[0]:
                render_external_link("Abrir Yahoo Finance", detail_row["yahoo_finance_url"])
            with source_cols[1]:
                render_external_link(
                    "Abrir página del emisor / factsheet",
                    detail_row["issuer_url"],
                )

            render_section_heading("ETF vs Benchmark")
            detail_columns = resolve_comparison_columns(
                [selected_detail],
                benchmark_map,
                cumulative.columns,
            )
            detail_base = cumulative[detail_columns].dropna(how="all")
            if detail_base.empty:
                st.warning("No hay datos suficientes para graficar el ETF contra su benchmark.")
            else:
                detail_min_date = pd.Timestamp(detail_base.index.min()).date()
                detail_max_date = pd.Timestamp(detail_base.index.max()).date()
                detail_start, detail_end = st.slider(
                    "Rango de fechas del gráfico",
                    min_value=detail_min_date,
                    max_value=detail_max_date,
                    value=(detail_min_date, detail_max_date),
                    format="YYYY-MM-DD",
                    key=f"detail_date_range_{selected_detail}",
                )
                detail_cumulative = rebase_cumulative_window(
                    cumulative,
                    detail_columns,
                    detail_start,
                    detail_end,
                )
                st.plotly_chart(
                    cumulative_chart(detail_cumulative, [selected_detail], benchmark_map),
                    width="stretch",
                )
                st.caption(
                    "El gráfico muestra retorno acumulado rebajado a 0% en la fecha inicial seleccionada. "
                    "La línea punteada corresponde al benchmark asignado."
                )

            render_section_heading("Resumen de Métricas")
            st.dataframe(metric_table(detail_row), hide_index=True, width="stretch")

            render_section_heading("Riesgos Detectados")
            render_ticker_risk_status(selected_detail, red_flags)

            risk_left, risk_right = st.columns([1.05, 0.95])
            with risk_left:
                render_section_heading("Drawdown")
                st.plotly_chart(drawdown_chart(drawdowns, selected_detail), width="stretch")
                st.caption(
                    "El drawdown mide la caída desde el último máximo histórico. "
                    "Ayuda a dimensionar pérdida potencial y tiempo de recuperación."
                )
            with risk_right:
                render_section_heading("Concentración Top 10")
                holdings_table = load_top_holdings_table(
                    selected_detail,
                    detail_row.get("top_10_holdings"),
                )
                st.plotly_chart(
                    top_10_concentration_chart(detail_row, holdings_table),
                    width="stretch",
                )
                if holdings_table.empty:
                    st.info(
                        "La fuente pública no entregó el desglose de holdings para este ETF."
                    )
                else:
                    st.caption(
                        "La dona separa los 5 mayores holdings, el resto del Top 10 y el resto del ETF. "
                        "A mayor concentración, mayor sensibilidad a shocks específicos de esos emisores."
                    )

        with tab_memo:
            render_section_heading(f"Memo preliminar: {memo_ticker}")
            memo_row = analysis.loc[analysis["ticker"] == memo_ticker].iloc[0]
            render_memo_decision_summary(memo_row)
            try:
                memo_text = generate_due_diligence_memo(memo_ticker, analysis, red_flags)
            except Exception:
                memo_text = bundle.memos.get(memo_ticker, "Memo no disponible para este ETF.")
            with st.expander("Ver borrador completo generado por reglas", expanded=False):
                st.markdown(memo_text)


if __name__ == "__main__":
    main()
