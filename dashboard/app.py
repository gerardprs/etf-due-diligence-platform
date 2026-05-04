"""Dashboard ejecutivo en Streamlit para selección de ETFs/fondos."""

from __future__ import annotations

from html import escape
import sys
from pathlib import Path

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
)
from fund_selection.pipeline import (  # noqa: E402
    build_selection_analysis,
    build_selection_analysis_from_processed,
)


PROCESSED_DIR = ROOT / "data" / "processed"
RAW_UNIVERSE_PATH = ROOT / "data" / "raw" / "fund_universe.csv"

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
    """Parse a comma/space separated ETF list for custom screening."""

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
    """Use the predownloaded curated universe when all required data is present."""

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
    """Build the full due diligence pipeline for a selected mandate."""

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
            --ink: #20242a;
            --muted: #626a73;
            --line: #d8dde3;
            --teal: #0b6f69;
            --gold: #a66f00;
            --red: #9f2f2f;
        }
        .stApp {
            background: var(--bg);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        h1 {
            font-size: 2rem;
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
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
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
        }
        section[data-testid="stSidebar"] {
            background: #eef1f4;
            border-right: 1px solid var(--line);
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
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(range=[0, 100], title="Fund Selection Score"),
        yaxis=dict(title=""),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
    )
    return fig


def cumulative_chart(
    cumulative: pd.DataFrame,
    selected_tickers: list[str],
    benchmark_map: dict[str, str],
) -> go.Figure:
    columns = []
    for ticker in selected_tickers:
        columns.append(ticker)
        benchmark = benchmark_map.get(ticker)
        if benchmark:
            columns.append(benchmark)

    columns = [column for index, column in enumerate(columns) if column in cumulative.columns and column not in columns[:index]]

    fig = go.Figure()
    for column in columns:
        dash = "dash" if column not in selected_tickers else "solid"
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative[column] * 100.0,
                mode="lines",
                name=column,
                line=dict(width=2.4 if dash == "solid" else 1.6, dash=dash),
                hovertemplate=f"<b>{column}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=10, b=25),
        yaxis_title="Retorno Acumulado",
        xaxis_title="",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


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
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
    )
    return fig


def metric_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("CAGR", _format_pct(row.get("cagr"))),
            ("Volatilidad", _format_pct(row.get("volatility"))),
            ("Sharpe Ratio", f"{pd.to_numeric(row.get('sharpe_ratio'), errors='coerce'):.2f}"),
            ("Sortino Ratio", f"{pd.to_numeric(row.get('sortino_ratio'), errors='coerce'):.2f}"),
            ("Max Drawdown", _format_pct(row.get("max_drawdown"))),
            ("Tracking Error", _format_pct(row.get("tracking_error"))),
            ("Information Ratio", f"{pd.to_numeric(row.get('information_ratio'), errors='coerce'):.2f}"),
            ("Expense Ratio", _format_pct_points(row.get("expense_ratio_pct"))),
            ("AUM", _format_money(row.get("total_assets"))),
            ("Volumen Promedio USD", _format_money(row.get("average_dollar_volume"))),
        ],
        columns=["Métrica", "Valor"],
    )


def enrich_with_source_links(analysis: pd.DataFrame) -> pd.DataFrame:
    """Add source URLs for recruiter-friendly verification."""

    enriched = analysis.copy()
    enriched["yahoo_finance_url"] = enriched["ticker"].map(
        lambda ticker: f"https://finance.yahoo.com/quote/{ticker}"
    )
    enriched["issuer_url"] = enriched["ticker"].map(
        lambda ticker: ISSUER_LINKS.get(str(ticker).upper(), "")
    )
    return enriched


def render_external_link(label: str, url: object) -> None:
    """Render an external link that works reliably in embedded browsers."""

    if pd.isna(url) or not str(url).strip():
        st.caption("Fuente oficial no mapeada para este ETF.")
        return

    safe_url = escape(str(url), quote=True)
    safe_label = escape(label)
    st.markdown(
        f"""
        <a href="{safe_url}" target="_self" style="
            display: block;
            width: 100%;
            padding: 0.7rem 0.9rem;
            border: 1px solid #d8dde3;
            border-radius: 8px;
            background: #ffffff;
            color: #0b6f69;
            text-align: center;
            text-decoration: none;
            font-weight: 650;
        ">{safe_label}</a>
        """,
        unsafe_allow_html=True,
    )
    st.caption(str(url))


def render_score_explanation() -> None:
    """Explain the Fund Selection Score in recruiter-friendly language."""

    st.subheader("Cómo se interpreta el Fund Selection Score")
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
            ("Benchmark fit", "20%", "Tracking error, R², information ratio y beta vs benchmark."),
            ("Liquidez", "15%", "AUM y volumen promedio en dólares."),
            ("Costo", "15%", "Expense ratio relativo dentro del peer group."),
            ("Penalizaciones", "Hasta -30 pts", "Alertas por drawdown, tracking error, costo, liquidez o data faltante."),
        ],
        columns=["Componente", "Peso", "Qué captura"],
    )
    st.dataframe(weights, hide_index=True, use_container_width=True)

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
              35% * Score(menor Tracking Error)
            + 25% * Score(R²)
            + 25% * Score(Information Ratio)
            + 15% * Score(Beta cercano a 1)
            ```

            Un ETF core debería tener tracking error bajo, alta relación con su benchmark y beta razonablemente cercana a 1.
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

            Ejemplos: tracking error alto, drawdown elevado, CAGR negativo, baja liquidez, AUM bajo, costo alto o metadata faltante.
            """
        )

    with st.expander("+ Fórmula final"):
        st.markdown(
            """
            ```text
            Fund Selection Score =
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


def render_intro() -> None:
    """Render the public-facing framing before the analytics workflow."""

    st.title("Plataforma de Selección de ETFs")
    st.caption("Hecho por Gerardo | Data Analytics + Investment Analytics")

    st.markdown(
        """
        Esta herramienta simula un flujo de **preliminary due diligence** para comparar ETFs desde una perspectiva de portfolio advisory. El objetivo es ordenar alternativas por performance, riesgo, ajuste al benchmark, liquidez, costo y alertas cuantitativas antes de pasar a una revisión cualitativa.

        **No constituye recomendación de inversión.** Los resultados son educativos y dependen de datos públicos. Antes de invertir, se debe revisar la ficha oficial del emisor, metodología del índice, holdings, spreads, impuestos, suitability y restricciones del cliente.
        """
    )

    with st.expander("Base teórica del análisis"):
        st.markdown(
            """
            - **Performance:** mide crecimiento histórico mediante CAGR y retornos acumulados.
            - **Riesgo:** evalúa volatilidad, drawdown, downside risk, VaR y CVaR.
            - **Benchmark fit:** primero evalúa cada ETF contra el benchmark que corresponde a su mandato. Luego compara los ETFs entre sí dentro del peer group.
            - **Liquidez y costo:** incorpora AUM, volumen promedio, volumen en dólares y expense ratio.
            - **Red flags:** penaliza señales que un analyst revisaría antes de recomendar: baja liquidez, AUM bajo, tracking error elevado, drawdown alto, costo alto o metadata faltante.
            - **Fund Selection Score:** combina los bloques anteriores en un ranking explicable. No es una caja negra ni una recomendación automática.
            """
        )

    with st.expander("Fuente de datos y confiabilidad"):
        st.markdown(
            """
            La app usa `yfinance`, que consulta datos públicos de Yahoo Finance, para precios, volumen y metadata disponible. Para una demo de portafolio es suficiente, pero no reemplaza fuentes institucionales como Bloomberg, FactSet, Morningstar Direct o datos oficiales del emisor.

            En la tabla se incluyen links a **Yahoo Finance** y, cuando está mapeado, a la **página oficial del emisor / factsheet**. En un workflow real, el analyst validaría expense ratio, AUM, benchmark, holdings y metodología directamente contra esa fuente oficial.
            """
        )


def render_screening_controls() -> tuple[str, list[str]]:
    """Render the guided ETF screening controls in the main page."""

    st.subheader("Construye el universo de comparación")
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
    st.dataframe(benchmark_preview, hide_index=True, use_container_width=True)

    return mandate, selected_tickers


def setup_step(risk_free_rate: float, start_date: str) -> None:
    """Render the configuration step and store the selected analysis universe."""

    mandate_label, selected_tickers = render_screening_controls()

    st.markdown("---")
    c1, c2 = st.columns([0.32, 0.68])
    with c1:
        continue_clicked = st.button(
            "Continuar al análisis",
            type="primary",
            use_container_width=True,
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
        st.rerun()


def render_analysis_header(config: dict[str, object]) -> None:
    """Show a compact summary of the selected universe with navigation."""

    left, right = st.columns([0.72, 0.28])
    with left:
        st.subheader("Análisis generado")
        st.caption(
            f"Mandato: {config['mandate_label']} | Benchmark: asignado por ETF | "
            f"ETFs: {', '.join(config['selected_tickers'])}"
        )
    with right:
        if st.button("Retroceder y ajustar selección", use_container_width=True):
            st.session_state["analysis_ready"] = False
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Plataforma de Selección de ETFs",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    st.session_state.setdefault("analysis_ready", False)
    st.session_state.setdefault("analysis_config", None)
    render_intro()

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
        setup_step(risk_free_rate=risk_free_rate, start_date=start_date)
        return

    config = st.session_state["analysis_config"]
    if not config or "mandate_label" not in config:
        st.session_state["analysis_ready"] = False
        st.session_state["analysis_config"] = None
        st.rerun()

    selected_tickers = config["selected_tickers"]
    render_analysis_header(config)
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

    with st.sidebar:
        memo_ticker = st.selectbox(
            "ETF para memo",
            options=scope_tickers,
            index=0,
        )

    filtered = analysis_scope.loc[analysis_scope["fund_selection_score"] >= min_score].copy()

    top = analysis_scope.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ETF líder", top["ticker"])
    col2.metric("Recomendación", top["recommendation"])
    col3.metric("Score líder", f"{top['fund_selection_score']:.1f}")
    col4.metric("ETFs revisados", f"{len(analysis_scope)}")

    tab_overview, tab_detail, tab_memo = st.tabs(
        ["Resumen Ejecutivo", "Análisis por ETF", "Memo de Due Diligence"]
    )

    with tab_overview:
        left, right = st.columns([1.05, 1.35])
        with left:
            st.subheader("Ranking de Selección")
            st.plotly_chart(score_bar_chart(filtered), use_container_width=True)

        with right:
            st.subheader("Comparación de Performance")
            st.plotly_chart(
                cumulative_chart(cumulative, selected_tickers, benchmark_map),
                use_container_width=True,
            )

        with st.expander("Ver metodología del score", expanded=True):
            render_score_explanation()

        ranking_columns = [
            "ticker",
            "name",
            "asset_class",
            "benchmark_ticker",
            "recommendation",
            "fund_selection_score",
            "cagr",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "tracking_error",
            "expense_ratio_pct",
            "total_assets",
            "red_flag_count",
            "yahoo_finance_url",
            "issuer_url",
        ]
        st.subheader("Tabla Ejecutiva de Ranking")
        st.dataframe(
            filtered[ranking_columns],
            hide_index=True,
            use_container_width=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "name": st.column_config.TextColumn("Nombre"),
                "asset_class": st.column_config.TextColumn("Clase de activo"),
                "benchmark_ticker": st.column_config.TextColumn("Benchmark"),
                "recommendation": st.column_config.TextColumn("Recomendación"),
                "fund_selection_score": st.column_config.NumberColumn("Score", format="%.1f"),
                "cagr": st.column_config.NumberColumn("CAGR", format="%.1%"),
                "volatility": st.column_config.NumberColumn("Volatilidad", format="%.1%"),
                "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "max_drawdown": st.column_config.NumberColumn("Max DD", format="%.1%"),
                "tracking_error": st.column_config.NumberColumn("Tracking Error", format="%.1%"),
                "expense_ratio_pct": st.column_config.NumberColumn("Expense Ratio", format="%.2f%%"),
                "total_assets": st.column_config.NumberColumn("AUM", format="$%.0f"),
                "red_flag_count": st.column_config.NumberColumn("Alertas", format="%d"),
                "yahoo_finance_url": st.column_config.LinkColumn(
                    "Yahoo Finance",
                    display_text="Ver data",
                ),
                "issuer_url": st.column_config.LinkColumn(
                    "Emisor / factsheet",
                    display_text="Ver fuente",
                ),
            },
        )

        st.subheader("Alertas")
        if red_flags.empty:
            st.success("No se detectaron alertas.")
        else:
            st.dataframe(
                red_flags,
                hide_index=True,
                use_container_width=True,
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
            index=scope_tickers.index(memo_ticker),
        )
        detail_row = analysis.loc[analysis["ticker"] == selected_detail].iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Score", f"{detail_row['fund_selection_score']:.1f}")
        c2.metric("CAGR", _format_pct(detail_row.get("cagr")))
        c3.metric("Sharpe", f"{pd.to_numeric(detail_row.get('sharpe_ratio'), errors='coerce'):.2f}")
        c4.metric("Costo", _format_pct_points(detail_row.get("expense_ratio_pct")))
        c5.metric("AUM", _format_money(detail_row.get("total_assets")))

        source_cols = st.columns(2)
        with source_cols[0]:
            render_external_link("Abrir Yahoo Finance", detail_row["yahoo_finance_url"])
        with source_cols[1]:
            render_external_link(
                "Abrir página del emisor / factsheet",
                detail_row["issuer_url"],
            )

        chart_left, chart_right = st.columns([1.35, 1])
        with chart_left:
            st.subheader("ETF vs Benchmark")
            st.plotly_chart(
                cumulative_chart(cumulative, [selected_detail], benchmark_map),
                use_container_width=True,
            )
        with chart_right:
            st.subheader("Resumen de Métricas")
            st.dataframe(metric_table(detail_row), hide_index=True, use_container_width=True)

        st.subheader("Drawdown")
        st.plotly_chart(drawdown_chart(drawdowns, selected_detail), use_container_width=True)

    with tab_memo:
        st.subheader(f"Memo preliminar: {memo_ticker}")
        st.markdown(bundle.memos[memo_ticker])


if __name__ == "__main__":
    main()
