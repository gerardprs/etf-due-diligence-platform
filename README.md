# ETF & Fund Selection Platform for Portfolio Advisory

Institutional-style analytics platform for ETF and fund screening, built for wealth management, family office, asset management, and investment advisory workflows.

## Portfolio Positioning

This project is designed to show investment judgement first and Python second. It automates the first-pass workflow a fund analyst would otherwise build manually in Excel: define the mandate, build a comparable peer group, assign benchmarks, calculate risk/return/liquidity/cost metrics, flag implementation risks, and produce a preliminary memo for PM review.

**Core message for recruiters:** the candidate understands fund selection as an investment process, and uses Python automation to make that process faster, more consistent, and more auditable.

## Objective

Create a Streamlit tool that supports preliminary ETF/fund due diligence by combining:

- performance analytics;
- risk metrics;
- benchmark fit;
- liquidity and cost checks;
- red flag detection;
- an explainable Fund Selection Score;
- investment committee status;
- a preliminary due diligence memo.

This is not a price prediction project. The goal is to simulate the type of decision-support tool an investment analyst could use before recommending a fund to a portfolio manager, advisor, or investment committee.

## Business Question

> Which ETF or fund is suitable for a client portfolio, considering return quality, risk control, benchmark behavior, liquidity, cost efficiency, and implementation red flags?

## Current Architecture

```text
.
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── fund_universe.csv
│   └── processed/
├── outputs/
│   └── due_diligence_memos.md
├── scripts/
│   ├── run_data_snapshot.py
│   └── run_full_analysis.py
├── src/
│   └── fund_selection/
│       ├── benchmark.py
│       ├── data_loader.py
│       ├── liquidity_cost.py
│       ├── memo.py
│       ├── performance.py
│       ├── pipeline.py
│       ├── red_flags.py
│       ├── risk.py
│       └── scoring.py
├── requirements.txt
└── README.md
```

## Workflow

```text
Fund universe
→ price and metadata download
→ data quality checks
→ performance and risk analytics
→ benchmark comparison
→ liquidity and cost review
→ red flag detection
→ Fund Selection Score
→ due diligence memo
→ Streamlit dashboard
```

## Data Sources

The platform currently uses:

- `yfinance` for ETF prices, volume, AUM, expense ratio, category, exchange, and issuer metadata when available;
- manually defined benchmark mapping in `data/raw/fund_universe.csv`;
- internally calculated returns, risk metrics, tracking metrics, scores, and flags.

Public vendor metadata can be incomplete. The pipeline treats missing metadata as a review item instead of silently assuming the fund is suitable.

## Metrics

Performance:

- total return;
- CAGR;
- annualized arithmetic return;
- best/worst month;
- positive months ratio.

Risk:

- annualized volatility;
- Sharpe ratio;
- Sortino ratio;
- downside deviation;
- max drawdown;
- historical VaR;
- historical CVaR.

Benchmark fit:

- beta;
- alpha;
- tracking error;
- information ratio;
- correlation;
- R-squared;
- annualized excess return.

Implementation checks:

- expense ratio;
- AUM;
- average volume;
- average dollar volume;
- red flag penalties.

## Fund Selection Score

The score is explainable and component-based:

```text
Fund Selection Score =
  Performance Score
+ Risk Score
+ Benchmark Fit Score
+ Liquidity Score
+ Cost Score
- Red Flag Penalties
```

The dashboard currently presents one of four preliminary review statuses:

- `Preferido`;
- `Aprobado`;
- `En observación`;
- `No prioritario`.

## Run

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Create or refresh the data snapshot:

```powershell
py scripts\run_data_snapshot.py
```

Run the full analytics workflow:

```powershell
py scripts\run_full_analysis.py
```

Launch the dashboard:

```powershell
py -m streamlit run dashboard\app.py
```

## Portfolio Positioning

This project demonstrates a hybrid profile:

- data analytics and automation;
- Python-based investment analytics;
- fund screening and benchmark comparison;
- investment memo generation;
- dashboard delivery for non-technical stakeholders.

Interview framing:

> I built a fund selection platform that automates preliminary ETF due diligence by combining performance, risk, benchmark fit, liquidity, cost, and red-flag checks into an explainable score and an investment-style memo.
