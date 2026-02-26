# fmcg-demand-forecast

> Portfolio-ready FMCG demand & sales forecasting pipeline with Indonesian market calendar features.

## Overview

Dual-pipeline deep learning system for Fast-Moving Consumer Goods (FMCG) inventory planning.
Extracted and sanitized from a production system serving an Indonesian distributor operating 6 darkstores.
All database dependencies replaced with synthetic data generation — fully runnable out of the box.

**Pipeline 1 (Demand):** Unit-level demand forecasting using a BiLSTM + Additive Attention model
with quantile loss (probabilistic outputs), per-product-warehouse embeddings, and STL decomposition.

**Pipeline 2 (Sales):** Revenue and COGS forecasting using per-region LSTM with sinusoidal seasonal
features (individual) and aggregated LSTM with day-of-week one-hots (merged), followed by a
purchase recommendation engine that redistributes non-working-day orders.

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  DATA LAYER                                                             │
  │                                                                         │
  │  scripts/generate_data.py ──▶ data/synthetic/                          │
  │    ├── products.csv  (80 SKUs, 5 categories)                           │
  │    ├── orders.csv    (daily orders, payday spikes, seasonal patterns)   │
  │    ├── sales.csv     (aggregated sales + COGS with realistic noise)     │
  │    └── calendar.csv  (Indonesian FMCG features, 13+ columns)           │
  └─────────────────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
  ┌───────────────────────┐    ┌──────────────────────────────┐
  │  PIPELINE 1: DEMAND   │    │  PIPELINE 2: SALES           │
  │                       │    │                              │
  │  preprocess_demand    │    │  preprocess_raw_data()       │
  │    ├─ OOS detection   │    │    └─ date adjustment        │
  │    ├─ outlier removal │    │       (Sundays/holidays)     │
  │    ├─ lag features    │    │                              │
  │    └─ Lebaran signal  │    │  ┌─────────┬──────────────┐ │
  │          │            │    │  │Individual│  Merged      │ │
  │  TimeSeriesForecaster │    │  │(per R-G) │  (aggregate) │ │
  │    ├─ STL decomp.     │    │  │SeasonalF.│  FinancialF. │ │
  │    ├─ MinMaxScaler    │    │  │LSTM+MHA  │  Plain LSTM  │ │
  │    ├─ TimeSeriesSplit │    │  │sin/cos   │  DOW one-hot │ │
  │    └─ BiLSTM+Attn     │    │  └────┬────┴──────┬───────┘ │
  │          │            │    │       │ sales+COGS│          │
  │  demand_forecast.csv  │    │       ▼           ▼          │
  └───────────────────────┘    │  recommendations.py          │
                                │    └─ 60/40 redistribution   │
                                │  sales_forecast/ + rec_buy/  │
                                └──────────────────────────────┘
```

## Model Architecture

```
  BiLSTM + Additive Attention (Demand Pipeline)
  ─────────────────────────────────────────────

  Input: [continuous features | product_embed | gudang_embed]
         (batch, seq_len=21, input_dim=24)
              │
              ▼
  ┌──────────────────────────┐
  │  Bidirectional LSTM      │  hidden_dim=24, num_layers=4
  │  (2 × hidden_dim output) │  dropout=0.25
  └──────────────────────────┘
              │
              ▼
  ┌──────────────────────────┐
  │  Additive Attention      │  Bahdanau-style
  │  (weights over seq_len)  │  v ∈ R^hidden_dim
  └──────────────────────────┘
              │ context vector (batch, 2×hidden_dim)
              ▼
  ┌──────────────────────────┐
  │  Linear + Reshape        │  output_dim × num_quantiles
  └──────────────────────────┘
              │
  Output: (batch, horizon=60, quantiles=1)
```

## Key Features

- **Indonesian FMCG calendar** — Ramadan/Lebaran proximity signal, payday periods (25th-5th), rainy season, Idul Adha, long weekends, back-to-school, THR payout week (2018-2026)
- **Probabilistic demand forecasting** — quantile loss (pinball) for uncertainty estimation
- **STL decomposition** — separates trend/seasonal/residual before LSTM training
- **Purchase recommendations** — non-working-day (Sunday + holidays) order redistribution using 60/40 split to preceding business days
- **Synthetic data generator** — realistic FMCG patterns (payday spikes, Ramadan surge, rainy-season dip, stock-outs, outliers)
- **GPU-ready Docker** — NVIDIA CUDA 12.4.1 runtime, one-command deployment

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
uv venv && uv sync

# 2. Generate synthetic data (80 SKUs, 5 warehouses, 4 years)
uv run python scripts/generate_data.py

# 3. Run demand forecasting pipeline
uv run python scripts/run_demand.py

# 4. Run sales forecasting pipeline (individual + merged + recommendations)
uv run python scripts/run_sales.py

# 5. Or run everything at once
uv run python scripts/run_all.py
```

Output files written to `data/`:
```
data/
├── synthetic/      # generated input data
├── demand_forecast/
├── sales_forecast/
└── sales_rec_buy/
```

## Indonesian Calendar Features

| Feature | Description |
|---------|-------------|
| `is_ramadan_month` | 1 during Ramadan month |
| `is_lebaran_peak_week` | 1 during Eid al-Fitr week |
| `is_thr_payout_week` | 1 during THR bonus payout week |
| `days_until_lebaran` | Days until next Eid (0 during Eid) |
| `days_after_lebaran` | Days since last Eid |
| `lebaran_proximity_signal` | 0–1 ramp signal starting 60 days before |
| `is_idul_adha_week` | 1 during Eid al-Adha week |
| `is_payday_period` | 1 from 25th to 5th of next month |
| `is_rainy_season` | 1 Oct–Apr (Indonesian rainy season) |
| `is_back_to_school` | 1 during school term start |
| `is_long_weekend` | 1 if part of a 3+ day long weekend |
| `is_cny_week` | 1 during Chinese New Year week |
| `is_year_end_holiday_period` | 1 during Dec 24–Jan 2 |

## Project Structure

```
fmcg-demand-forecast/
├── config/
│   └── default.yaml          # default hyperparameters for all pipelines
├── docker/
│   ├── Dockerfile            # NVIDIA CUDA 12.4.1 + Python 3.11
│   └── docker-compose.yml    # GPU-enabled service with volume mounts
├── notebooks/
│   └── demo.ipynb            # end-to-end walkthrough
├── scripts/
│   ├── generate_data.py      # synthetic data generation
│   ├── run_demand.py         # demand pipeline entry point
│   ├── run_sales.py          # sales pipeline entry point
│   └── run_all.py            # full pipeline orchestrator
├── src/fmcg_forecast/
│   ├── config.py             # DemandConfig, SalesConfig (Pydantic Settings)
│   ├── data/
│   │   ├── calendar.py       # Indonesian FMCG calendar features
│   │   ├── generator.py      # synthetic data generator
│   │   └── loader.py         # CSV loader utility
│   ├── demand/
│   │   ├── forecaster.py     # TimeSeriesForecaster (BiLSTM + Attention)
│   │   └── preprocessing.py  # OOS detection, outlier removal, features
│   ├── models/
│   │   └── lstm_attention.py # QuantileLoss, Attention, LSTM model classes
│   ├── sales/
│   │   ├── fetch.py          # sales CSV loader
│   │   ├── individual.py     # SeasonalFinancialForecaster
│   │   ├── merged.py         # FinancialForecaster (aggregated)
│   │   └── recommendations.py # purchase recommendation engine
│   └── utils/
│       ├── logging.py        # centralized logger setup
│       └── paths.py          # project directory management
└── tests/                    # 48 tests, 92% coverage
```

## Configuration

Edit `config/default.yaml` or use environment variables (prefix `FMCG_FORECAST_` for demand, `FMCG_SALES_` for sales):

```yaml
demand:
  input_window: 21       # lookback window (days)
  forecast_horizon: 60   # forecast horizon (days)
  epochs: 500
  batch_size: 32

sales:
  input_window: 30
  forecast_horizon: 90
  epochs: 200
  batch_size: 64
```

## Docker

```bash
# Build and run with GPU support
cd docker
docker-compose up --build

# CPU-only (modify Dockerfile base image to python:3.11-slim)
docker build -t fmcg-forecast -f docker/Dockerfile ..
docker run -v $(pwd)/data:/app/data fmcg-forecast
```

## Testing

```bash
# Run all tests
uv run pytest

# Run unit tests only (fast)
uv run pytest -m unit

# Run with coverage report
uv run pytest --cov=src --cov-report=term-missing
```

## License

MIT — see `LICENSE`.
