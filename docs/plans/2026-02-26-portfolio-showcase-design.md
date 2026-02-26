# Design: fmcg-demand-forecast Portfolio Showcase

**Date:** 2026-02-26
**Author:** Johannes Baptista Adiatmaja Pambudi
**Status:** Approved

---

## Problem

Showcase the `yokulak-forecasting` production pipeline as a clean, open-source
portfolio project. The original is tightly coupled to ClickHouse/MySQL databases
and contains company-specific references. The portfolio version must be fully
generic, fully runnable with synthetic data, and demonstrate production-grade
ML engineering.

## Target Audience

- Recruiters / Hiring managers (clean code, production readiness)
- Technical peers / Data Scientists (model architecture, feature engineering)
- Potential clients (business impact, end-to-end capability)

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Repo name | `fmcg-demand-forecast` | Domain-specific, professional, searchable |
| Scope | Fully generic | No company references, synthetic data only |
| Runnability | Fully runnable | Clone -> install -> generate -> train -> forecast |
| Data | Synthetic generator | Realistic FMCG patterns, no external DB needed |
| Pipelines | Both (demand + sales) | Demonstrates breadth of the system |

---

## Dual Pipeline Architecture

The project contains **two distinct forecasting pipelines** that share common
infrastructure (data generation, calendar features, LSTM-Attention model):

### Original Pipeline Flow

**Pipeline 1 — Demand Forecasting** (operational: unit demand prediction):
```
fetch/fetch.py -> fetch/preprocessing.py -> pytorch/forecast.py -> pytorch/insert.py
```

**Pipeline 2 — Sales Forecasting** (financial: revenue/COGS + buying recommendations):
```
sales/fetch.py -> sales/individual/forecasting.py -> sales/merged/forecasting.py
-> sales/recommended_buy/individual/rec_buy.py -> sales/recommended_buy/merged/rec_buy.py
-> sales/insert/postprocess.py -> sales/insert/insert.py
```

### New Portfolio Architecture

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  SHARED: DATA GENERATION                                             │
  │                                                                      │
  │  scripts/generate_data.py ──▶ data/synthetic/*.csv                  │
  │  (50-100 SKUs, 3-5 warehouses, 2+ years daily sales + revenue)     │
  └──────────────────────┬──────────────────────────┬────────────────────┘
                         │                          │
         ┌───────────────▼───────────┐   ┌─────────▼────────────────────┐
         │  PIPELINE 1: DEMAND       │   │  PIPELINE 2: SALES           │
         │                           │   │                              │
         │  ┌─────────────────────┐  │   │  ┌────────────────────────┐  │
         │  │ demand/             │  │   │  │ sales/                 │  │
         │  │  preprocessing.py   │  │   │  │  fetch.py              │  │
         │  │  (OOS detect,      │  │   │  │  (sales data loading)  │  │
         │  │   cleaning,        │  │   │  └───────────┬────────────┘  │
         │  │   features)        │  │   │              │               │
         │  └──────────┬──────────┘  │   │  ┌──────────▼────────────┐  │
         │             │             │   │  │ individual.py          │  │
         │  ┌──────────▼──────────┐  │   │  │ (per-product LSTM)    │  │
         │  │ demand/             │  │   │  └──────────┬────────────┘  │
         │  │  forecaster.py     │  │   │              │               │
         │  │  (BiLSTM-Attention │  │   │  ┌──────────▼────────────┐  │
         │  │   + quantile loss) │  │   │  │ merged.py             │  │
         │  └──────────┬──────────┘  │   │  │ (consolidated LSTM)   │  │
         │             │             │   │  └──────────┬────────────┘  │
         │             ▼             │   │              │               │
         │  demand predictions      │   │  ┌──────────▼────────────┐  │
         │  (units, with intervals) │   │  │ recommendations.py    │  │
         │                           │   │  │ (individual + merged  │  │
         └───────────────────────────┘   │  │  purchase quantities) │  │
                                         │  └──────────┬────────────┘  │
                                         │              │               │
                                         │              ▼               │
                                         │  sales & COGS forecasts     │
                                         │  + buying recommendations   │
                                         └──────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │  SHARED INFRASTRUCTURE                                               │
  │                                                                      │
  │  data/calendar.py ── 13+ Indonesian FMCG temporal features          │
  │  models/lstm_attention.py ── Shared BiLSTM-Attention architecture   │
  │  config.py ── Pydantic Settings + YAML loader                       │
  │  utils/ ── Logging, path management                                 │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
fmcg-demand-forecast/
├── src/
│   └── fmcg_forecast/
│       ├── __init__.py
│       ├── config.py                        # Pydantic Settings + YAML loader
│       ├── data/                            # Shared data modules
│       │   ├── __init__.py
│       │   ├── generator.py                 # Synthetic FMCG data generator
│       │   ├── loader.py                    # Generic CSV/Parquet loader
│       │   └── calendar.py                  # 13+ Indonesian FMCG temporal features
│       ├── models/                          # Shared model architectures
│       │   ├── __init__.py
│       │   └── lstm_attention.py            # BiLSTM-Attention with quantile loss
│       ├── demand/                          # Pipeline 1: Demand Forecasting
│       │   ├── __init__.py
│       │   ├── preprocessing.py             # OOS detection, cleaning, feature eng.
│       │   └── forecaster.py                # Demand model training + inference
│       ├── sales/                           # Pipeline 2: Sales Forecasting
│       │   ├── __init__.py
│       │   ├── fetch.py                     # Sales data loading + aggregation
│       │   ├── individual.py                # Per-product sales LSTM
│       │   ├── merged.py                    # Consolidated sales + COGS LSTM
│       │   └── recommendations.py           # Purchase quantity recommendations
│       └── utils/
│           ├── __init__.py
│           ├── logging.py                   # Centralized logger setup
│           └── paths.py                     # Path management
├── tests/
│   ├── conftest.py
│   ├── test_data/                           # Small fixture CSVs
│   ├── test_calendar.py
│   ├── test_demand_preprocessing.py
│   ├── test_demand_forecaster.py
│   ├── test_sales_individual.py
│   ├── test_sales_merged.py
│   ├── test_recommendations.py
│   └── test_model.py
├── notebooks/
│   └── demo.ipynb                           # End-to-end walkthrough (both pipelines)
├── config/
│   └── default.yaml                         # Default configuration
├── docker/
│   ├── Dockerfile                           # NVIDIA CUDA GPU-ready
│   └── docker-compose.yml
├── scripts/
│   ├── generate_data.py                     # CLI: generate synthetic FMCG data
│   ├── run_demand.py                        # CLI: run demand forecasting pipeline
│   ├── run_sales.py                         # CLI: run sales forecasting pipeline
│   └── run_all.py                           # CLI: run both pipelines end-to-end
├── docs/
│   └── plans/
├── .env.example
├── .python-version                          # 3.11
├── pyproject.toml                           # Single source of truth
├── CLAUDE.md
├── README.md                                # Portfolio-grade README with ASCII diagrams
├── LICENSE                                  # MIT
└── .gitignore
```

---

## Module Mapping (Original -> New)

### Pipeline 1: Demand Forecasting

| Original | New | Changes |
|----------|-----|---------|
| `fetch/fetch.py` | `data/generator.py` + `data/loader.py` | Replace ClickHouse/MySQL with synthetic data + generic loader |
| `fetch/holidays.py` | `data/calendar.py` | Remove DB dep, accept date range input, keep all 13+ features |
| `fetch/preprocessing.py` (496 lines) | `demand/preprocessing.py` | Rewrite cleanly: OOS detection + cleaning + feature engineering |
| `pytorch/forecast.py` | `models/lstm_attention.py` + `demand/forecaster.py` | Split architecture from training/inference |
| `pytorch/insert.py` | **REMOVED** | DB insertion not needed — output to CSV |

### Pipeline 2: Sales Forecasting

| Original | New | Changes |
|----------|-----|---------|
| `sales/fetch.py` | `sales/fetch.py` | Rewrite to load from synthetic CSVs instead of ClickHouse |
| `sales/individual/forecasting.py` | `sales/individual.py` | Clean rewrite, reuse shared LSTM-Attention model |
| `sales/merged/forecasting.py` | `sales/merged.py` | Clean rewrite, consolidated sales + COGS |
| `sales/recommended_buy/individual/rec_buy.py` | `sales/recommendations.py` | Consolidate individual + merged into one module |
| `sales/recommended_buy/merged/rec_buy.py` | `sales/recommendations.py` | (merged into above) |
| `sales/insert/postprocess.py` | **REMOVED** | Output to CSV instead of DB |
| `sales/insert/insert.py` | **REMOVED** | DB insertion not needed |

### Shared

| Original | New | Changes |
|----------|-----|---------|
| `utils/paths.py` | `utils/paths.py` | Simplify, remove MySQL dep |
| (MySQL logging) | `utils/logging.py` | Replace with standard Python logging |
| (scattered config) | `config.py` | Centralize with Pydantic Settings |

---

## Synthetic Data Generator

Generates realistic FMCG data for **both pipelines**:

**Demand data** (for Pipeline 1):
- 50-100 SKUs across 5 categories (Beverages, Snacks, Personal Care, Household, Dairy)
- 3-5 warehouses (anonymized as Warehouse A/B/C...)
- 2+ years daily order records with realistic patterns:
  - Weekly seasonality (weekend dips)
  - Monthly payday spikes (25th-5th)
  - Ramadan/Lebaran demand surges
  - Rainy season effects (Oct-Apr)
  - Random stock-out events (OOS periods)
  - Replenishment events after stock-outs
  - Occasional outliers
- Product attributes: brand, category, unit_price, is_promo

**Sales data** (for Pipeline 2):
- Daily revenue and COGS records per product
- Aggregated sales metrics for merged forecasting
- Pricing variations and promotional periods

**Output**: CSVs in `data/synthetic/` matching both pipelines' input schemas.

---

## Tech Stack

- **Python 3.11** with `uv` package manager
- **PyTorch** (LSTM-Attention model, shared across both pipelines)
- **pandas / numpy** (data manipulation)
- **scikit-learn** (MinMaxScaler, StandardScaler, metrics)
- **statsmodels** (STL seasonal decomposition)
- **Pydantic Settings** (typed configuration)
- **ruff** (formatting + linting, replaces black + isort)
- **mypy** (type checking in CI)
- **pytest** (testing, 80%+ coverage target)
- **Docker** (NVIDIA CUDA GPU-ready container)

---

## CLI Entry Points

```bash
# Generate synthetic FMCG data
uv run python scripts/generate_data.py --skus 100 --warehouses 5 --years 2

# Run demand forecasting pipeline
uv run python scripts/run_demand.py --config config/default.yaml

# Run sales forecasting pipeline
uv run python scripts/run_sales.py --config config/default.yaml

# Run both pipelines end-to-end
uv run python scripts/run_all.py --config config/default.yaml
```

---

## README Structure

1. Title + one-line description
2. Overview (3-5 sentences — mention both pipelines)
3. Architecture (ASCII dual-pipeline diagram)
4. Key Features (bullet list highlighting both pipelines)
5. Quick Start (5-6 commands: install, generate, run demand, run sales)
6. Model Architecture (ASCII diagram of shared LSTM-Attention)
7. Results (sample forecast plots from both pipelines)
8. Project Structure (directory tree)
9. Configuration (YAML + env vars)
10. Docker (GPU training)
11. License (MIT)
