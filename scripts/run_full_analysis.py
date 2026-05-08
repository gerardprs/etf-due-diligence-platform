"""Ejecuta el flujo completo de análisis y selección de ETFs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fund_selection.pipeline import build_selection_analysis_from_processed  # noqa: E402


def main() -> None:
    """Genera ranking, alertas y memos preliminares."""

    processed_dir = ROOT / "data" / "processed"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_selection_analysis_from_processed(processed_dir)

    bundle.analysis.to_csv(
        output_dir / "fund_selection_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle.red_flags.to_csv(
        output_dir / "red_flags.csv",
        index=False,
        encoding="utf-8-sig",
    )

    memo_path = output_dir / "due_diligence_memos.md"
    memo_path.write_text(
        "\n\n---\n\n".join(bundle.memos.values()),
        encoding="utf-8",
    )

    top = bundle.analysis.iloc[0]
    print("Análisis completo finalizado.")
    print(f"Fondos analizados: {len(bundle.analysis)}")
    print(f"Fondo líder: {top['ticker']} ({top['recommendation']})")
    print(f"Score líder: {top['fund_selection_score']:.1f}")
    print(f"Carpeta de salida: {output_dir}")


if __name__ == "__main__":
    main()
