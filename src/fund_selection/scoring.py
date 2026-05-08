"""Explainable Fund Selection Score."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_SCORE_WEIGHTS = {
    "performance_score": 0.25,
    "risk_score": 0.25,
    "benchmark_fit_score": 0.20,
    "liquidity_score": 0.15,
    "cost_score": 0.15,
}


def _score_series(
    values: pd.Series,
    higher_is_better: bool = True,
    missing_score: float = 35.0,
) -> pd.Series:
    """Convert a metric into a 0-100 cross-sectional score."""

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    score = pd.Series(missing_score, index=values.index, dtype=float)

    valid = numeric.dropna()
    if valid.empty:
        return score

    min_value = valid.min()
    max_value = valid.max()
    if min_value == max_value:
        score.loc[valid.index] = 70.0
        return score

    scaled = (valid - min_value) / (max_value - min_value)
    if not higher_is_better:
        scaled = 1.0 - scaled

    score.loc[valid.index] = (40.0 + scaled * 60.0).clip(0.0, 100.0)
    return score


def _log_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Score scale-sensitive fields such as AUM and dollar volume."""

    numeric = pd.to_numeric(values, errors="coerce")
    transformed = np.log10(numeric.where(numeric > 0))
    return _score_series(transformed, higher_is_better=higher_is_better)


def calculate_component_scores(analysis: pd.DataFrame) -> pd.DataFrame:
    """Calculate transparent component scores used by the final ranking."""

    scored = analysis.copy()

    scored["performance_score"] = (
        0.65 * _score_series(scored.get("cagr"), higher_is_better=True)
        + 0.35 * _score_series(scored.get("sharpe_ratio"), higher_is_better=True)
    )

    scored["risk_score"] = (
        0.35 * _score_series(scored.get("volatility"), higher_is_better=False)
        + 0.25 * _score_series(scored.get("max_drawdown"), higher_is_better=True)
        + 0.20 * _score_series(scored.get("sortino_ratio"), higher_is_better=True)
        + 0.20 * _score_series(scored.get("historical_cvar_95"), higher_is_better=False)
    )

    scored["benchmark_fit_score"] = (
        0.45 * _score_series(scored.get("tracking_error"), higher_is_better=False)
        + 0.30 * _score_series(scored.get("r_squared"), higher_is_better=True)
        + 0.25 * _score_series(scored.get("information_ratio"), higher_is_better=True)
    )

    scored["liquidity_score"] = (
        0.55 * _log_score(scored.get("total_assets"), higher_is_better=True)
        + 0.45 * _log_score(scored.get("average_dollar_volume"), higher_is_better=True)
    )

    scored["cost_score"] = _score_series(
        scored.get("expense_ratio_pct"),
        higher_is_better=False,
    )

    return scored


def recommendation_from_score(score: float) -> str:
    """Translate final score into an investment-office style status."""

    if pd.isna(score):
        return "Requiere revisión"
    if score >= 80:
        return "Preferido"
    if score >= 65:
        return "Aprobado"
    if score >= 40:
        return "En observación"
    return "No prioritario"


def committee_status_from_row(row: pd.Series) -> str:
    """Translate score and flags into a next-step workflow status.

    The score is useful for ranking, but an analyst still needs a plain-language
    output: which funds deserve review, which stay in the watchlist, and which
    are not worth prioritizing without a qualitative reason.
    """

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


def score_funds(
    analysis: pd.DataFrame,
    red_flags: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build final Fund Selection Score after red-flag penalties."""

    score_weights = {**DEFAULT_SCORE_WEIGHTS, **(weights or {})}
    total_weight = sum(score_weights.values())
    if not np.isclose(total_weight, 1.0):
        score_weights = {key: value / total_weight for key, value in score_weights.items()}

    scored = calculate_component_scores(analysis)

    if red_flags.empty:
        penalties = pd.DataFrame(
            {
                "ticker": scored["ticker"],
                "red_flag_count": 0,
                "red_flag_penalty": 0.0,
            }
        )
    else:
        penalties = (
            red_flags.groupby("ticker", as_index=False)
            .agg(red_flag_count=("flag", "count"), red_flag_penalty=("penalty", "sum"))
        )

    scored = scored.merge(penalties, on="ticker", how="left")
    scored["red_flag_count"] = scored["red_flag_count"].fillna(0).astype(int)
    scored["red_flag_penalty"] = scored["red_flag_penalty"].fillna(0.0).clip(upper=30.0)

    weighted_score = sum(scored[column] * weight for column, weight in score_weights.items())
    scored["fund_selection_score"] = (weighted_score - scored["red_flag_penalty"]).clip(0.0, 100.0)
    scored["recommendation"] = scored["fund_selection_score"].map(recommendation_from_score)
    scored["committee_status"] = scored.apply(committee_status_from_row, axis=1)

    columns_first = [
        "ticker",
        "name",
        "asset_class",
        "benchmark_ticker",
        "recommendation",
        "committee_status",
        "fund_selection_score",
        "performance_score",
        "risk_score",
        "benchmark_fit_score",
        "liquidity_score",
        "cost_score",
        "red_flag_count",
        "red_flag_penalty",
    ]
    remaining = [column for column in scored.columns if column not in columns_first]
    return scored[[*columns_first, *remaining]].sort_values(
        "fund_selection_score",
        ascending=False,
    ).reset_index(drop=True)
