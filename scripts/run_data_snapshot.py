"""Crea un snapshot local de datos para la plataforma de selección de ETFs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fund_selection import build_data_snapshot  # noqa: E402


def main() -> None:
    """Descarga, valida y guarda el snapshot inicial de datos de mercado."""

    bundle = build_data_snapshot(
        universe_path=ROOT / "data" / "raw" / "fund_universe.csv",
        start="2021-01-01",
        include_metadata=True,
    )

    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle.universe.to_csv(output_dir / "fund_universe_validated.csv", index=False)
    bundle.prices.to_csv(output_dir / "prices.csv", index_label="date")
    bundle.returns.to_csv(output_dir / "daily_returns.csv", index_label="date")
    bundle.metadata.to_csv(output_dir / "fund_metadata.csv", index=False)
    bundle.quality_report.to_frame().to_csv(
        output_dir / "data_quality_summary.csv",
        index=False,
    )

    with (output_dir / "missing_ratio_by_ticker.json").open("w", encoding="utf-8") as file:
        json.dump(bundle.quality_report.missing_ratio_by_ticker, file, indent=2)

    print("Snapshot de datos creado correctamente.")
    print(f"Filas: {bundle.quality_report.observations}")
    print(f"Tickers disponibles: {', '.join(bundle.quality_report.available_tickers)}")
    print(f"Tickers faltantes: {', '.join(bundle.quality_report.missing_tickers) or 'Ninguno'}")
    print(f"Carpeta de salida: {output_dir}")


if __name__ == "__main__":
    main()
