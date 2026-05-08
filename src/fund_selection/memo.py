"""Generación de memos preliminares de revisión."""

from __future__ import annotations

import pandas as pd


def _pct(value: object, decimals: int = 1) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric * 100:.{decimals}f}%"


def _pct_points(value: object, decimals: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}%"


def _money(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    if numeric >= 1_000_000_000:
        return f"${numeric / 1_000_000_000:,.1f}B"
    if numeric >= 1_000_000:
        return f"${numeric / 1_000_000:,.1f}M"
    return f"${numeric:,.0f}"


def _number(value: object, decimals: int = 2) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}"


def _numeric(value: object) -> float:
    """Convierte un valor a float o NaN para reglas del memo."""

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _multiple(value: object, decimals: int = 1) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}x"


def _strengths(row: pd.Series) -> list[str]:
    strengths: list[str] = []

    if row.get("fund_selection_score", 0) >= 80:
        strengths.append("Puntaje general sólido en performance, riesgo, costo y liquidez.")
    if pd.to_numeric(row.get("expense_ratio_pct"), errors="coerce") <= 0.05:
        strengths.append("Perfil de costos muy competitivo para una asignación sensible a comisiones.")
    if pd.to_numeric(row.get("total_assets"), errors="coerce") >= 50_000_000_000:
        strengths.append("Base de activos amplia, favorable para uso institucional y estabilidad del producto.")
    if pd.to_numeric(row.get("average_dollar_volume"), errors="coerce") >= 100_000_000:
        strengths.append("Actividad de negociación saludable, favorable para una implementación eficiente.")
    if pd.to_numeric(row.get("tracking_error"), errors="coerce") <= 0.02:
        strengths.append("Comportamiento cercano al benchmark, útil para una exposición limpia en cartera.")
    if pd.to_numeric(row.get("sharpe_ratio"), errors="coerce") >= 0.75:
        strengths.append("Retorno ajustado por riesgo aceptable durante la ventana analizada.")
    if pd.to_numeric(row.get("top_10_concentration"), errors="coerce") <= 0.30:
        strengths.append("Concentración Top 10 moderada frente a un filtro preliminar.")

    return strengths or ["No se detecta una fortaleza cuantitativa sobresaliente; requiere contexto cualitativo adicional."]


def _risks(row: pd.Series, ticker_flags: pd.DataFrame) -> list[str]:
    if not ticker_flags.empty:
        return ticker_flags["rationale"].dropna().astype(str).tolist()

    risks: list[str] = []
    if pd.to_numeric(row.get("max_drawdown"), errors="coerce") < -0.20:
        risks.append("El drawdown histórico es material y debe considerarse en el tamaño de la posición.")
    if pd.to_numeric(row.get("tracking_error"), errors="coerce") > 0.03:
        risks.append("La desviación frente al benchmark debe revisarse antes de usarlo como exposición core.")

    return risks or ["No se detecta una alerta cuantitativa material en este filtro preliminar."]


def _decision_view(row: pd.Series) -> str:
    """Construye una lectura breve del primer filtro."""

    ticker = row.get("ticker")
    committee_status = row.get("committee_status", "Revisión manual requerida")
    recommendation = row.get("recommendation", "n/a")
    score = _number(row.get("fund_selection_score"), 1)
    return (
        f"{ticker} queda en estado **{committee_status}** con lectura cuantitativa "
        f"**{recommendation}** y score de **{score}/100**. El resultado debe interpretarse "
        "como priorización de revisión, no como orden de compra."
    )


def _committee_questions(row: pd.Series) -> list[str]:
    """Genera checks cualitativos pendientes después del filtro cuantitativo."""

    ticker = str(row.get("ticker", "")).upper()
    benchmark = row.get("benchmark_ticker", "benchmark asignado")
    questions = [
        f"Confirmar que la metodología de {ticker} replica correctamente la exposición buscada frente a {benchmark}.",
        "Validar TER, AUM, volumen, bid-ask spread y riesgo de cierre contra el factsheet oficial del emisor.",
        "Revisar holdings, concentración Top 10 y overlap con posiciones existentes del portafolio.",
        "Evaluar tratamiento tributario, versión UCITS/offshore disponible y restricciones de suitability del cliente.",
    ]
    tracking_error = _numeric(row.get("tracking_error"))
    top_10 = _numeric(row.get("top_10_concentration"))
    if pd.notna(tracking_error) and tracking_error > 0.03:
        questions.append("Explicar la fuente del tracking error antes de usarlo como exposición core.")
    if pd.notna(top_10) and top_10 > 0.40:
        questions.append("Preparar lectura de concentración por issuer y sensibilidad a mega caps.")
    return questions


def generate_due_diligence_memo(
    ticker: str,
    scored_analysis: pd.DataFrame,
    red_flags: pd.DataFrame,
) -> str:
    """Genera un memo preliminar para un fondo."""

    clean_ticker = ticker.upper()
    row_match = scored_analysis.loc[scored_analysis["ticker"] == clean_ticker]
    if row_match.empty:
        raise ValueError(f"Ticker not found in scored analysis: {clean_ticker}")

    row = row_match.iloc[0]
    ticker_flags = red_flags.loc[red_flags["ticker"] == clean_ticker] if not red_flags.empty else red_flags

    strengths = "\n".join(f"- {item}" for item in _strengths(row))
    risks = "\n".join(f"- {item}" for item in _risks(row, ticker_flags))
    questions = "\n".join(f"- {item}" for item in _committee_questions(row))
    decision_view = _decision_view(row)

    memo = f"""# Memo Preliminar de Due Diligence

## 1. Decisión Preliminar

{decision_view}

## 2. Resumen del Fondo

**Ticker:** {row.get("ticker")}
**Nombre:** {row.get("name", "n/a")}
**Clase de Activo:** {row.get("asset_class", "n/a")}
**Benchmark:** {row.get("benchmark_ticker", "n/a")}
**Recomendación:** {row.get("recommendation", "n/a")}
**Siguiente paso:** {row.get("committee_status", "n/a")}
**Score de priorización:** {_number(row.get("fund_selection_score"), 1)} / 100

## 3. Resumen Cuantitativo

| Métrica | Valor |
|---|---:|
| CAGR | {_pct(row.get("cagr"))} |
| Alpha | {_pct(row.get("alpha"))} |
| Sortino Ratio | {_number(row.get("sortino_ratio"))} |
| Max Drawdown | {_pct(row.get("max_drawdown"))} |
| Tracking Error | {_pct(row.get("tracking_error"))} |
| P/E | {_multiple(row.get("valuation_pe"))} |
| ROE | {_pct(row.get("return_on_equity"))} |
| Top 10 Concentration | {_pct(row.get("top_10_concentration"))} |
| Expense Ratio | {_pct_points(row.get("expense_ratio_pct"))} |
| AUM | {_money(row.get("total_assets"))} |
| Volumen Promedio en USD | {_money(row.get("average_dollar_volume"))} |

## 4. Racional de Inversión

{row.get("ticker")} está clasificado como exposición de {row.get("asset_class", "n/a")} y se evalúa contra {row.get("benchmark_ticker", "n/a")}. El filtro actual asigna una recomendación de {row.get("recommendation", "n/a")} con base en performance, control de riesgo, ajuste al benchmark, liquidez, eficiencia de costos y penalizaciones por alertas.

## 5. Fortalezas Clave

{strengths}

## 6. Riesgos / Alertas

{risks}

## 7. Checks Cualitativos Pendientes

{questions}

## 8. Vista Preliminar

Este memo automatiza la primera lectura que normalmente se haría en Excel: ranking relativo, benchmark fit, riesgo de pérdida, liquidez, costo, concentración y alertas. Antes de tomar una decisión de inversión, el analista debe validar metodología del índice, composición, overlap con cartera, consideraciones tributarias, comportamiento de spreads, política de securities lending y restricciones de suitability del cliente.
"""

    return memo


def generate_all_memos(scored_analysis: pd.DataFrame, red_flags: pd.DataFrame) -> dict[str, str]:
    """Genera memos para todos los fondos evaluados."""

    return {
        ticker: generate_due_diligence_memo(ticker, scored_analysis, red_flags)
        for ticker in scored_analysis["ticker"].tolist()
    }
