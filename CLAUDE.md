# CLAUDE.md — fmcg-demand-forecast

FMCG demand & sales forecasting pipeline. Portfolio project extracted from Yokulak production system
with all database dependencies replaced by file-based I/O and synthetic data.

---

## Architecture

**Dual-pipeline system:**
- **Pipeline 1 (Demand):** unit-level daily sales forecasting, per product-warehouse pair
- **Pipeline 2 (Sales):** revenue + COGS forecasting (individual per-region and aggregated/merged)

Both pipelines share: synthetic data generator, Indonesian FMCG calendar, centralized config/logging.

## Key Files

| File | Purpose |
|------|---------|
| `src/fmcg_forecast/config.py` | `DemandConfig`, `SalesConfig` (Pydantic BaseSettings) |
| `src/fmcg_forecast/data/calendar.py` | Indonesian FMCG calendar, `FMCG_HOLIDAYS`, `create_feature_calendar()` |
| `src/fmcg_forecast/data/generator.py` | `generate_products()`, `generate_orders()`, `generate_sales_data()` |
| `src/fmcg_forecast/demand/preprocessing.py` | `detect_oos_periods()`, `remove_outliers()`, `preprocess_demand_data()` |
| `src/fmcg_forecast/demand/forecaster.py` | `TimeSeriesForecaster` (BiLSTM + Attention, STL, TimeSeriesSplit CV) |
| `src/fmcg_forecast/models/lstm_attention.py` | `QuantileLoss`, `Attention`, `TimeSeriesLSTMAttentionModel`, `SeasonalFinancialLSTMModel`, `FinancialLSTMModel` |
| `src/fmcg_forecast/sales/individual.py` | `SeasonalFinancialForecaster` (per region-gudang, sinusoidal features) |
| `src/fmcg_forecast/sales/merged.py` | `FinancialForecaster` (aggregated, DOW one-hot, COGS depends on sales) |
| `src/fmcg_forecast/sales/recommendations.py` | `calculate_recommendations()` (60/40 redistribution to preceding working days) |
| `config/default.yaml` | Default hyperparameters for all pipelines |
| `scripts/run_all.py` | Orchestrates full pipeline: generate → demand → sales |

## Common Commands

```bash
# Setup
uv venv && uv sync

# Generate synthetic data
uv run python scripts/generate_data.py

# Run pipelines
uv run python scripts/run_demand.py --config config/default.yaml
uv run python scripts/run_sales.py --config config/default.yaml --mode both
uv run python scripts/run_all.py

# Tests
uv run pytest -m unit          # fast unit tests only
uv run pytest                  # all 48 tests
uv run pytest --cov=src --cov-report=term-missing  # with coverage (92%)

# Code quality
uv run ruff fmt .
uv run ruff check --fix .
uv run mypy src/ --ignore-missing-imports
```

## Test Markers

- `@pytest.mark.unit` — fast, no model training (< 5s)
- `@pytest.mark.slow` — includes model training, may take 10-60s

## Key Design Decisions

1. **No database dependencies** — all MySQL/ClickHouse removed. File I/O via `data/loader.py`.
2. **`model_params` field in SalesConfig** — Pydantic v2 reserves `model_config` as a class-level attribute; use `model_params: SalesModelConfig` instead.
3. **`patience` on SalesConfig** — added as a field (not in the original plan), needed by `SeasonalFinancialForecaster.train_model()`.
4. **Recommendations: no `shift(-1)`** — using `rec_buy = cogs` (same-day) preserves total conservation; non-working-day redistribution handles the "order before use" semantics.
5. **STL decomposition in demand forecaster** — separates trend from residual before MinMaxScaler; trend is extrapolated linearly at predict time.
6. **Product/warehouse LabelEncoder** — maps string IDs to integer indices for embedding layers.

## Domain Context

**Yokulak** = Indonesian FMCG distributor, 6 darkstores (darkstore = urban micro-fulfillment center).
Indonesian FMCG has pronounced seasonality: Ramadan/Lebaran, payday periods (25th–5th), rainy season (Oct–Apr), Chinese New Year.

**Lebaran proximity signal:** ramps from 0 to 1 over the 60 days before Eid al-Fitr. Used as a continuous feature.

**Non-working days:** Sundays + Indonesian public holidays. Orders for non-working days are split 60% to 2 days prior and 40% to 1 day prior (the preceding working days).

## Source Reference

Original production code: `C:\working-space\yokulak-forecasting\`
- `pytorch/forecast.py` → `demand/forecaster.py` + `models/lstm_attention.py`
- `fetch/holidays.py` → `data/calendar.py`
- `fetch/preprocessing.py` → `demand/preprocessing.py`
- `sales/individual/forecasting.py` → `sales/individual.py`
- `sales/merged/forecasting.py` → `sales/merged.py`
- `sales/recommended_buy/*/rec_buy.py` → `sales/recommendations.py`
