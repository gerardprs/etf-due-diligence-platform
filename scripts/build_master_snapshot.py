"""Build a stable local snapshot for the curated ETF universe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fund_selection.curated_universe import MANDATE_PRESETS  # noqa: E402
from fund_selection.data_loader import (  # noqa: E402
    build_quality_report,
    calculate_daily_returns,
    download_price_history,
    fetch_fund_metadata,
)


def build_master_universe() -> pd.DataFrame:
    """Create a long-form table of curated ETFs and assigned benchmarks."""

    rows: list[dict[str, str]] = []
    for mandate, preset in MANDATE_PRESETS.items():
        for ticker in preset["tickers"]:
            rows.append(
                {
                    "mandate": mandate,
                    "ticker": ticker,
                    "name": "",
                    "asset_class": preset["asset_class"],
                    "benchmark_ticker": preset["benchmarks"][ticker],
                    "category": preset["category"],
                    "role": f"Mandato - {mandate}",
                }
            )

    universe = pd.DataFrame(rows).drop_duplicates(["mandate", "ticker"])
    return universe.sort_values(["mandate", "ticker"]).reset_index(drop=True)


def main() -> None:
    """Download prices, returns, and metadata for every curated ETF/benchmark."""

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    universe = build_master_universe()
    all_tickers = sorted(
        set(universe["ticker"].tolist()) | set(universe["benchmark_ticker"].tolist())
    )
    fund_tickers = sorted(set(universe["ticker"].tolist()))

    prices = download_price_history(all_tickers, start="2021-01-01")
    returns = calculate_daily_returns(prices)
    metadata = fetch_fund_metadata(fund_tickers)
    quality_report = build_quality_report(all_tickers, prices)

    universe.to_csv(raw_dir / "etf_universe_master.csv", index=False, encoding="utf-8-sig")
    prices.to_csv(processed_dir / "prices_master.csv", index_label="date")
    returns.to_csv(processed_dir / "daily_returns_master.csv", index_label="date")
    metadata.to_csv(processed_dir / "fund_metadata_master.csv", index=False, encoding="utf-8-sig")
    quality_report.to_frame().to_csv(
        processed_dir / "master_data_quality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (processed_dir / "master_missing_ratio_by_ticker.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(quality_report.missing_ratio_by_ticker, file, indent=2)

    print("Master snapshot created successfully.")
    print(f"Curated fund entries: {len(universe)}")
    print(f"Unique requested tickers: {len(all_tickers)}")
    print(f"Available tickers: {len(quality_report.available_tickers)}")
    print(f"Missing tickers: {', '.join(quality_report.missing_tickers) or 'None'}")
    print(f"Output directory: {processed_dir}")


if __name__ == "__main__":
    main()
