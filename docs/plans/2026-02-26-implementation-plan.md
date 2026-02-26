# fmcg-demand-forecast Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a portfolio-ready, fully runnable FMCG demand & sales forecasting pipeline from the existing yokulak-forecasting codebase.

**Architecture:** Dual-pipeline system — Pipeline 1 (Demand: unit forecasting with BiLSTM-Attention + quantile loss) and Pipeline 2 (Sales: revenue/COGS forecasting with LSTM + buying recommendations). Shared infrastructure includes synthetic data generation, Indonesian FMCG calendar features, and centralized config/logging. All database dependencies (ClickHouse, MySQL) replaced with file-based I/O and synthetic data.

**Tech Stack:** Python 3.11, PyTorch, pandas, numpy, scikit-learn, statsmodels, Pydantic Settings, ruff, mypy, pytest, Docker (NVIDIA CUDA)

**Source reference:** Original code at `C:\working-space\yokulak-forecasting\`

---

## Phase 1: Project Scaffolding

### Task 1: Initialize project with uv and create directory structure

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `LICENSE`
- Create: all `__init__.py` files

**Step 1: Initialize git repo and create `.python-version`**

```bash
cd C:/working-space/fmcg-demand-forecast
git init
```

Create `.python-version`:
```
3.11
```

**Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "fmcg-demand-forecast"
version = "0.1.0"
description = "LSTM-Attention demand forecasting pipeline for FMCG retail with Indonesian market calendar features"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Johannes Baptista Adiatmaja Pambudi"}
]
dependencies = [
    "torch>=2.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "statsmodels>=0.14",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "matplotlib>=3.7",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-mock>=3.0",
    "ruff>=0.4",
    "mypy>=1.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "DTZ", "EM", "ISC", "PIE", "PT", "Q", "RET", "SIM", "TID", "ARG"]
ignore = ["E501", "B008", "ARG001", "ARG002"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]

[tool.ruff.format]
quote-style = "double"

[tool.ruff.lint.isort]
lines-after-imports = 2

[tool.mypy]
python_version = "3.11"
disallow_untyped_defs = true
disallow_incomplete_defs = true
warn_return_any = true
warn_redundant_casts = true
warn_unused_ignores = true
strict_equality = true
no_implicit_optional = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: fast, isolated unit tests",
    "integration: tests that touch external systems",
    "slow: tests that take more than a few seconds",
]
addopts = "-v --tb=short"
```

**Step 3: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp

# Environment
.env

# Data artifacts (generated)
data/synthetic/
data/demand/
data/sales/
logs/
*.pt
*.png

# OS
.DS_Store
Thumbs.db

# Coverage
htmlcov/
.coverage
coverage.xml
```

**Step 4: Create `.env.example`**

```env
# Model configuration (optional overrides)
FMCG_FORECAST_INPUT_WINDOW=21
FMCG_FORECAST_HORIZON=60
FMCG_FORECAST_EPOCHS=500
FMCG_FORECAST_BATCH_SIZE=128
FMCG_FORECAST_LEARNING_RATE=0.0001

# Sales model configuration
FMCG_SALES_INPUT_WINDOW=30
FMCG_SALES_FORECAST_HORIZON=90
FMCG_SALES_EPOCHS=200
```

**Step 5: Create `LICENSE` (MIT)**

Standard MIT license with author name.

**Step 6: Create all directory stubs with `__init__.py`**

```bash
mkdir -p src/fmcg_forecast/{data,models,demand,sales,utils}
mkdir -p tests/test_data
mkdir -p notebooks scripts config docker docs/plans
```

Create empty `__init__.py` in each package:
- `src/fmcg_forecast/__init__.py`
- `src/fmcg_forecast/data/__init__.py`
- `src/fmcg_forecast/models/__init__.py`
- `src/fmcg_forecast/demand/__init__.py`
- `src/fmcg_forecast/sales/__init__.py`
- `src/fmcg_forecast/utils/__init__.py`

**Step 7: Install dependencies**

```bash
uv venv
uv pip install -e ".[dev]"
```

**Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold project structure with pyproject.toml and directories"
```

---

## Phase 2: Shared Utilities

### Task 2: Create centralized logging utility

**Files:**
- Create: `src/fmcg_forecast/utils/logging.py`
- Test: `tests/test_logging.py`

**Step 1: Write the failing test**

```python
# tests/test_logging.py
import logging

import pytest


@pytest.mark.unit
def test_setup_logger_returns_configured_logger():
    from fmcg_forecast.utils.logging import setup_logger

    logger = setup_logger("test_logger", level=logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG


@pytest.mark.unit
def test_setup_logger_has_console_handler():
    from fmcg_forecast.utils.logging import setup_logger

    logger = setup_logger("test_console", level=logging.INFO)
    handler_types = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in handler_types


@pytest.mark.unit
def test_setup_logger_with_file_handler(tmp_path):
    from fmcg_forecast.utils.logging import setup_logger

    log_file = tmp_path / "test.log"
    logger = setup_logger("test_file", log_file=str(log_file))
    logger.info("test message")
    assert log_file.exists()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logging.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/fmcg_forecast/utils/logging.py
"""Centralized logging configuration."""
import logging
import sys


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Create a configured logger with console and optional file output.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_logging.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/fmcg_forecast/utils/logging.py tests/test_logging.py
git commit -m "feat: add centralized logging utility"
```

---

### Task 3: Create path management utility

**Files:**
- Create: `src/fmcg_forecast/utils/paths.py`
- Test: `tests/test_paths.py`

**Step 1: Write the failing test**

```python
# tests/test_paths.py
import pytest
from pathlib import Path


@pytest.mark.unit
def test_get_data_dirs_returns_expected_keys():
    from fmcg_forecast.utils.paths import get_data_dirs

    dirs = get_data_dirs()
    expected_keys = {
        "data", "logs",
        "demand_raw", "demand_preprocessed", "demand_forecast",
        "sales_raw", "sales_forecast", "sales_rec_buy",
        "synthetic",
    }
    assert set(dirs.keys()) == expected_keys


@pytest.mark.unit
def test_get_data_dirs_with_custom_root(tmp_path):
    from fmcg_forecast.utils.paths import get_data_dirs

    dirs = get_data_dirs(project_root=tmp_path)
    assert dirs["data"] == tmp_path / "data"
    assert dirs["synthetic"] == tmp_path / "data" / "synthetic"


@pytest.mark.unit
def test_ensure_dirs_creates_directories(tmp_path):
    from fmcg_forecast.utils.paths import get_data_dirs, ensure_dirs

    dirs = get_data_dirs(project_root=tmp_path)
    ensure_dirs(dirs)
    for path in dirs.values():
        assert path.exists()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_paths.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/fmcg_forecast/utils/paths.py
"""Centralized path management for the forecasting pipeline."""
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_data_dirs(project_root: Path | None = None) -> dict[str, Path]:
    """Return a dict of all standard data directories.

    Args:
        project_root: Override project root. Uses auto-detected root if None.

    Returns:
        Dictionary mapping directory names to Path objects.
    """
    root = project_root or get_project_root()
    data = root / "data"

    return {
        "data": data,
        "logs": root / "logs",
        "synthetic": data / "synthetic",
        "demand_raw": data / "demand" / "01_raw",
        "demand_preprocessed": data / "demand" / "02_preprocessed",
        "demand_forecast": data / "demand" / "03_forecast",
        "sales_raw": data / "sales" / "01_raw",
        "sales_forecast": data / "sales" / "02_forecast",
        "sales_rec_buy": data / "sales" / "03_recommended_buy",
    }


def ensure_dirs(dirs: dict[str, Path]) -> None:
    """Create all directories in the dict if they don't exist."""
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
```

**Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_paths.py -v`

**Step 5: Commit**

```bash
git add src/fmcg_forecast/utils/paths.py tests/test_paths.py
git commit -m "feat: add centralized path management"
```

---

### Task 4: Create configuration module with Pydantic Settings

**Files:**
- Create: `src/fmcg_forecast/config.py`
- Create: `config/default.yaml`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest


@pytest.mark.unit
def test_demand_config_has_defaults():
    from fmcg_forecast.config import DemandConfig

    cfg = DemandConfig()
    assert cfg.input_window == 21
    assert cfg.forecast_horizon == 60
    assert cfg.epochs == 500
    assert cfg.batch_size == 128
    assert cfg.quantiles == [0.5]


@pytest.mark.unit
def test_sales_config_has_defaults():
    from fmcg_forecast.config import SalesConfig

    cfg = SalesConfig()
    assert cfg.input_window == 30
    assert cfg.forecast_horizon == 90
    assert cfg.epochs == 200


@pytest.mark.unit
def test_load_yaml_config(tmp_path):
    from fmcg_forecast.config import load_yaml_config

    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("demand:\n  input_window: 14\n  epochs: 100\n")
    cfg = load_yaml_config(str(yaml_file))
    assert cfg["demand"]["input_window"] == 14
    assert cfg["demand"]["epochs"] == 100


@pytest.mark.unit
def test_model_config_has_defaults():
    from fmcg_forecast.config import ModelConfig

    cfg = ModelConfig()
    assert cfg.hidden_dim == 24
    assert cfg.num_layers == 4
    assert cfg.learning_rate == 0.0001
    assert cfg.dropout == 0.25
    assert cfg.product_embedding_dim == 16
    assert cfg.gudang_embedding_dim == 8
    assert cfg.early_stopping_patience == 25
```

**Step 2: Run test to verify it fails**

**Step 3: Write implementation**

```python
# src/fmcg_forecast/config.py
"""Configuration management using Pydantic Settings and YAML."""
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """LSTM-Attention model hyperparameters."""

    hidden_dim: int = 24
    num_layers: int = 4
    learning_rate: float = 0.0001
    dropout: float = 0.25
    weight_decay: float = 0.0001
    product_embedding_dim: int = 16
    gudang_embedding_dim: int = 8
    early_stopping_patience: int = 25


class SalesModelConfig(BaseModel):
    """Sales LSTM model hyperparameters."""

    hidden_dim: int = 72
    num_layers: int = 3
    learning_rate: float = 0.001
    dropout: float = 0.3
    patience: int = 20


class DemandConfig(BaseSettings):
    """Demand forecasting pipeline configuration."""

    model_config_cls: ModelConfig = ModelConfig()
    input_window: int = 21
    forecast_horizon: int = 60
    epochs: int = 500
    batch_size: int = 128
    cv_splits: int = 5
    quantiles: list[float] = [0.5]

    model_config = {"env_prefix": "FMCG_FORECAST_"}


class SalesConfig(BaseSettings):
    """Sales forecasting pipeline configuration."""

    model_config_cls: SalesModelConfig = SalesModelConfig()
    input_window: int = 30
    forecast_horizon: int = 90
    epochs: int = 200
    batch_size: int = 64

    model_config = {"env_prefix": "FMCG_SALES_"}


class GeneratorConfig(BaseModel):
    """Synthetic data generator configuration."""

    num_skus: int = 80
    num_warehouses: int = 5
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"
    seed: int = 42
    categories: list[str] = [
        "Beverages",
        "Snacks",
        "Personal Care",
        "Household",
        "Dairy",
    ]


def load_yaml_config(path: str) -> dict:
    """Load configuration from a YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    with open(path) as f:
        return yaml.safe_load(f)
```

Create `config/default.yaml`:

```yaml
demand:
  input_window: 21
  forecast_horizon: 60
  epochs: 500
  batch_size: 128
  cv_splits: 5
  quantiles: [0.5]
  model:
    hidden_dim: 24
    num_layers: 4
    learning_rate: 0.0001
    dropout: 0.25
    product_embedding_dim: 16
    gudang_embedding_dim: 8
    early_stopping_patience: 25

sales:
  input_window: 30
  forecast_horizon: 90
  epochs: 200
  batch_size: 64
  model:
    hidden_dim: 72
    num_layers: 3
    learning_rate: 0.001
    dropout: 0.3
    patience: 20

generator:
  num_skus: 80
  num_warehouses: 5
  start_date: "2022-01-01"
  end_date: "2025-12-31"
  seed: 42
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add src/fmcg_forecast/config.py config/default.yaml tests/test_config.py
git commit -m "feat: add Pydantic Settings configuration with YAML support"
```

---

## Phase 3: Data Layer

### Task 5: Create Indonesian FMCG calendar feature engine

**Source reference:** `yokulak-forecasting/fetch/holidays.py`

This is the showcase module — 13+ temporal features engineered for Indonesian FMCG seasonality.

**Files:**
- Create: `src/fmcg_forecast/data/calendar.py`
- Test: `tests/test_calendar.py`

**Step 1: Write the failing test**

```python
# tests/test_calendar.py
import pandas as pd
import pytest


@pytest.mark.unit
def test_create_calendar_returns_dataframe():
    from fmcg_forecast.data.calendar import create_feature_calendar

    df = create_feature_calendar("2024-01-01", "2024-01-31")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 31


@pytest.mark.unit
def test_calendar_has_all_base_features():
    from fmcg_forecast.data.calendar import create_feature_calendar

    df = create_feature_calendar("2024-01-01", "2024-12-31")
    expected_features = [
        "date", "is_ramadan_month", "is_lebaran_peak_week",
        "is_thr_payout_week", "days_until_lebaran", "days_after_lebaran",
        "is_idul_adha_week", "is_year_end_holiday_period",
        "is_independence_day_week", "is_cny_week", "is_rainy_season",
        "is_payday_period", "is_back_to_school", "is_long_weekend",
    ]
    for feat in expected_features:
        assert feat in df.columns, f"Missing feature: {feat}"


@pytest.mark.unit
def test_calendar_interaction_features():
    from fmcg_forecast.data.calendar import create_feature_calendar

    df = create_feature_calendar("2024-01-01", "2024-12-31")
    assert "interaction_payday_lebaran" in df.columns
    assert "interaction_payday_year_end" in df.columns
    assert "interaction_payday_long_weekend" in df.columns


@pytest.mark.unit
def test_calendar_sequential_features():
    from fmcg_forecast.data.calendar import create_feature_calendar

    df = create_feature_calendar("2024-01-01", "2024-12-31")
    assert "long_weekend_day_number" in df.columns


@pytest.mark.unit
def test_rainy_season_oct_to_apr():
    from fmcg_forecast.data.calendar import create_feature_calendar

    df = create_feature_calendar("2024-01-01", "2024-12-31")
    jan_row = df[df["date"] == "2024-01-15"].iloc[0]
    jul_row = df[df["date"] == "2024-07-15"].iloc[0]
    assert jan_row["is_rainy_season"] == 1
    assert jul_row["is_rainy_season"] == 0


@pytest.mark.unit
def test_payday_period_25th_to_5th():
    from fmcg_forecast.data.calendar import create_feature_calendar

    df = create_feature_calendar("2024-01-01", "2024-01-31")
    day_3 = df[df["date"] == "2024-01-03"].iloc[0]
    day_15 = df[df["date"] == "2024-01-15"].iloc[0]
    day_26 = df[df["date"] == "2024-01-26"].iloc[0]
    assert day_3["is_payday_period"] == 1
    assert day_15["is_payday_period"] == 0
    assert day_26["is_payday_period"] == 1


@pytest.mark.unit
def test_ramadan_2024_is_flagged():
    from fmcg_forecast.data.calendar import create_feature_calendar

    df = create_feature_calendar("2024-03-01", "2024-04-30")
    # Ramadan 2024 starts March 12, Lebaran April 10
    mar_20 = df[df["date"] == "2024-03-20"].iloc[0]
    assert mar_20["is_ramadan_month"] == 1
```

**Step 2: Run tests to verify failure**

**Step 3: Write implementation**

Rewrite from `yokulak-forecasting/fetch/holidays.py`, removing all MySQL logging and ClickHouse dependencies. Keep the exact same holiday dates and feature engineering logic.

```python
# src/fmcg_forecast/data/calendar.py
"""Indonesian FMCG calendar feature engineering.

Generates 13+ temporal features tailored to Indonesian FMCG retail patterns:
Ramadan, Lebaran, payday cycles, rainy season, Chinese New Year, and more.
"""
import logging
from datetime import date, timedelta

import pandas as pd


logger = logging.getLogger(__name__)

# --- Indonesian holiday dates ---
FMCG_HOLIDAYS: dict[str, dict[int, date]] = {
    "idul_fitri": {
        2018: date(2018, 6, 15), 2019: date(2019, 6, 5),
        2020: date(2020, 5, 24), 2021: date(2021, 5, 13),
        2022: date(2022, 5, 2), 2023: date(2023, 4, 22),
        2024: date(2024, 4, 10), 2025: date(2025, 3, 31),
        2026: date(2026, 3, 20),
    },
    "ramadan_start": {
        2018: date(2018, 5, 17), 2019: date(2019, 5, 6),
        2020: date(2020, 4, 24), 2021: date(2021, 4, 13),
        2022: date(2022, 4, 3), 2023: date(2023, 3, 23),
        2024: date(2024, 3, 12), 2025: date(2025, 3, 1),
        2026: date(2026, 2, 18),
    },
    "idul_adha": {
        2018: date(2018, 8, 22), 2019: date(2019, 8, 11),
        2020: date(2020, 7, 31), 2021: date(2021, 7, 20),
        2022: date(2022, 7, 10), 2023: date(2023, 6, 29),
        2024: date(2024, 6, 17), 2025: date(2025, 6, 7),
        2026: date(2026, 5, 28),
    },
    "chinese_new_year": {
        2018: date(2018, 2, 16), 2019: date(2019, 2, 5),
        2020: date(2020, 1, 25), 2021: date(2021, 2, 12),
        2022: date(2022, 2, 1), 2023: date(2023, 1, 22),
        2024: date(2024, 2, 10), 2025: date(2025, 1, 29),
        2026: date(2026, 2, 17),
    },
}

# Comprehensive Indonesian public holidays (2018-2026)
ALL_PUBLIC_HOLIDAYS: set[date] = {
    # (exact same set from yokulak-forecasting/fetch/holidays.py lines 66-91)
    date(2018, 1, 1), date(2018, 2, 16), date(2018, 3, 17),
    # ... (full set to be copied from original)
}


def create_feature_calendar(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Generate an FMCG calendar with 13+ temporal features.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with date column and all engineered features.
    """
    logger.info("Generating feature calendar: %s to %s", start_date, end_date)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = pd.DataFrame({"date": pd.date_range(start=start, end=end)})

    # Initialize base features to 0
    base_features = [
        "is_ramadan_month", "is_lebaran_peak_week", "is_thr_payout_week",
        "days_until_lebaran", "days_after_lebaran", "is_idul_adha_week",
        "is_year_end_holiday_period", "is_independence_day_week",
        "is_cny_week", "is_rainy_season", "is_payday_period",
        "is_back_to_school", "is_long_weekend",
    ]
    for col in base_features:
        df[col] = 0

    # Static seasonal features
    df["is_rainy_season"] = df["date"].dt.month.isin(
        [10, 11, 12, 1, 2, 3, 4]
    ).astype(int)
    df["is_payday_period"] = (
        (df["date"].dt.day >= 25) | (df["date"].dt.day <= 5)
    ).astype(int)
    df["is_back_to_school"] = (
        df["date"].dt.month.isin([7])
        | ((df["date"].dt.month == 1) & (df["date"].dt.day <= 15))
    ).astype(int)

    # Long weekend detection
    for holiday in ALL_PUBLIC_HOLIDAYS:
        dates: list[date] = []
        if holiday.weekday() in [0, 1]:  # Mon/Tue
            dates = [holiday - timedelta(d) for d in range(4)]
        elif holiday.weekday() in [3, 4]:  # Thu/Fri
            dates = [holiday + timedelta(d) for d in range(4)]
        if dates:
            pd_dates = pd.to_datetime(dates)
            df.loc[df["date"].isin(pd_dates), "is_long_weekend"] = 1

    # Event-based features per year
    for year in range(start.year, end.year + 1):
        if year not in FMCG_HOLIDAYS["idul_fitri"]:
            continue

        lebaran = pd.to_datetime(FMCG_HOLIDAYS["idul_fitri"][year])
        ramadan_start = pd.to_datetime(FMCG_HOLIDAYS["ramadan_start"][year])

        mask_ramadan = (df["date"] >= ramadan_start) & (df["date"] < lebaran)
        df.loc[mask_ramadan, "is_ramadan_month"] = 1

        mask_peak = (
            (df["date"] >= lebaran - timedelta(days=7))
            & (df["date"] < lebaran)
        )
        df.loc[mask_peak, "is_lebaran_peak_week"] = 1

        mask_thr = (
            (df["date"] >= lebaran - timedelta(days=14))
            & (df["date"] <= lebaran - timedelta(days=8))
        )
        df.loc[mask_thr, "is_thr_payout_week"] = 1

        mask_until = (
            (df["date"] >= lebaran - timedelta(days=30))
            & (df["date"] <= lebaran)
        )
        df.loc[mask_until, "days_until_lebaran"] = (
            df["date"] - lebaran
        ).dt.days

        mask_after = (
            (df["date"] > lebaran)
            & (df["date"] <= lebaran + timedelta(days=14))
        )
        df.loc[mask_after, "days_after_lebaran"] = (
            df["date"] - lebaran
        ).dt.days

        adha = pd.to_datetime(FMCG_HOLIDAYS["idul_adha"][year])
        mask_adha = (
            (df["date"] >= adha - timedelta(days=3))
            & (df["date"] <= adha + timedelta(days=3))
        )
        df.loc[mask_adha, "is_idul_adha_week"] = 1

        cny = pd.to_datetime(FMCG_HOLIDAYS["chinese_new_year"][year])
        mask_cny = (
            (df["date"] >= cny - timedelta(days=3))
            & (df["date"] <= cny + timedelta(days=3))
        )
        df.loc[mask_cny, "is_cny_week"] = 1

    # Year-end and Independence Day
    df.loc[
        (df["date"].dt.month == 12) & (df["date"].dt.day >= 20)
        | (df["date"].dt.month == 1) & (df["date"].dt.day <= 2),
        "is_year_end_holiday_period",
    ] = 1
    df.loc[
        (df["date"].dt.month == 8)
        & (df["date"].dt.day >= 14)
        & (df["date"].dt.day <= 20),
        "is_independence_day_week",
    ] = 1

    # Interaction features
    df["interaction_payday_lebaran"] = (
        df["is_payday_period"] * df["is_lebaran_peak_week"]
    )
    df["interaction_payday_year_end"] = (
        df["is_payday_period"] * df["is_year_end_holiday_period"]
    )
    df["interaction_payday_long_weekend"] = (
        df["is_payday_period"] * df["is_long_weekend"]
    )

    # Sequential features
    weekend_blocks = (df["is_long_weekend"].diff() != 0).cumsum()
    df["long_weekend_day_number"] = df.groupby(weekend_blocks).cumcount() + 1
    df.loc[df["is_long_weekend"] == 0, "long_weekend_day_number"] = 0

    # Format date as string for CSV compatibility
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    logger.info(
        "Calendar generated: %d days, %d features",
        len(df),
        len(df.columns) - 1,
    )
    return df
```

**NOTE:** Copy the full `ALL_PUBLIC_HOLIDAYS` set from the original `holidays.py` lines 66-91.

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add src/fmcg_forecast/data/calendar.py tests/test_calendar.py
git commit -m "feat: add Indonesian FMCG calendar with 13+ temporal features"
```

---

### Task 6: Create synthetic FMCG data generator

**Files:**
- Create: `src/fmcg_forecast/data/generator.py`
- Test: `tests/test_generator.py`

**Step 1: Write the failing test**

```python
# tests/test_generator.py
import pandas as pd
import pytest


@pytest.mark.unit
def test_generate_products_returns_correct_count():
    from fmcg_forecast.data.generator import generate_products

    products = generate_products(num_skus=10, seed=42)
    assert len(products) == 10
    assert "product_name" in products.columns
    assert "category" in products.columns
    assert "unit_price" in products.columns


@pytest.mark.unit
def test_generate_orders_returns_daily_records():
    from fmcg_forecast.data.generator import generate_products, generate_orders

    products = generate_products(num_skus=5, seed=42)
    orders = generate_orders(
        products,
        num_warehouses=2,
        start_date="2024-01-01",
        end_date="2024-01-31",
        seed=42,
    )
    assert isinstance(orders, pd.DataFrame)
    assert "date" in orders.columns
    assert "product_name" in orders.columns
    assert "id_gudang" in orders.columns
    assert "main_product_sales" in orders.columns
    assert len(orders) > 0


@pytest.mark.unit
def test_generate_orders_has_seasonal_patterns():
    from fmcg_forecast.data.generator import generate_products, generate_orders

    products = generate_products(num_skus=5, seed=42)
    orders = generate_orders(
        products,
        num_warehouses=2,
        start_date="2024-01-01",
        end_date="2024-12-31",
        seed=42,
    )
    # Payday periods (25th-5th) should have higher average sales
    orders["day"] = pd.to_datetime(orders["date"]).dt.day
    payday = orders[(orders["day"] >= 25) | (orders["day"] <= 5)]
    non_payday = orders[(orders["day"] > 5) & (orders["day"] < 25)]
    assert payday["main_product_sales"].mean() > non_payday["main_product_sales"].mean()


@pytest.mark.unit
def test_generate_sales_data_returns_revenue_and_cogs():
    from fmcg_forecast.data.generator import (
        generate_products,
        generate_orders,
        generate_sales_data,
    )

    products = generate_products(num_skus=5, seed=42)
    orders = generate_orders(
        products,
        num_warehouses=2,
        start_date="2024-01-01",
        end_date="2024-03-31",
        seed=42,
    )
    sales = generate_sales_data(orders, products, seed=42)
    assert "sales" in sales.columns
    assert "cogs" in sales.columns
    # COGS should be less than sales (margin > 0)
    assert (sales["cogs"] <= sales["sales"]).all()
```

**Step 2: Run tests to verify failure**

**Step 3: Write implementation**

The generator creates realistic FMCG data with:
- Weekly seasonality (weekend dips)
- Monthly payday spikes
- Ramadan/Lebaran surges
- Rainy season effects
- Random stock-out events
- Occasional outliers

```python
# src/fmcg_forecast/data/generator.py
"""Synthetic FMCG data generator.

Generates realistic sales data for Indonesian FMCG retail with
seasonal patterns, stock-out events, and promotional effects.
"""
import logging

import numpy as np
import pandas as pd

from fmcg_forecast.data.calendar import FMCG_HOLIDAYS


logger = logging.getLogger(__name__)

BRAND_POOL: dict[str, list[str]] = {
    "Beverages": ["AquaClear", "TeaMax", "JuiceFresh", "SodaPop"],
    "Snacks": ["CrunchBite", "NuttyBar", "ChipKing", "WaferLite"],
    "Personal Care": ["CleanWash", "FreshGlow", "SilkCare", "PureWhite"],
    "Household": ["SparkleClean", "PowerWash", "FreshAir", "ShineMore"],
    "Dairy": ["MilkPure", "YoguFresh", "CheesyBite", "CreamDelight"],
}


def generate_products(
    num_skus: int = 80,
    categories: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a product master table.

    Args:
        num_skus: Number of SKUs to generate.
        categories: Product category names.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with product_name, category, brand, unit_price, is_promo.
    """
    rng = np.random.default_rng(seed)
    if categories is None:
        categories = list(BRAND_POOL.keys())

    products = []
    for i in range(num_skus):
        cat = categories[i % len(categories)]
        brands = BRAND_POOL.get(cat, ["GenericBrand"])
        brand = brands[i % len(brands)]
        variant = f"{i // len(categories) + 1:02d}"
        products.append({
            "product_id": f"SKU-{i + 1:04d}",
            "product_name": f"{brand} {cat[:3].upper()}-{variant}",
            "category": cat,
            "brand": brand,
            "unit_price": int(rng.uniform(5_000, 50_000)),
            "is_promo": int(rng.random() < 0.15),
        })

    logger.info("Generated %d synthetic products", num_skus)
    return pd.DataFrame(products)


def generate_orders(
    products: pd.DataFrame,
    num_warehouses: int = 5,
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily order records with realistic FMCG patterns.

    Args:
        products: Product master DataFrame.
        num_warehouses: Number of warehouses.
        start_date: Start date string.
        end_date: End date string.
        seed: Random seed.

    Returns:
        DataFrame with daily order records per product per warehouse.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date)
    warehouses = [f"WH-{chr(65 + i)}" for i in range(num_warehouses)]

    records = []
    for _, product in products.iterrows():
        base_demand = rng.uniform(5, 80)
        for wh in warehouses:
            wh_factor = rng.uniform(0.6, 1.4)

            for dt in dates:
                demand = base_demand * wh_factor

                # Weekend dip (Saturday half, Sunday zero)
                if dt.weekday() == 5:
                    demand *= 0.5
                elif dt.weekday() == 6:
                    demand = 0
                    continue

                # Payday spike (25th-5th)
                if dt.day >= 25 or dt.day <= 5:
                    demand *= rng.uniform(1.2, 1.6)

                # Rainy season boost (Oct-Apr)
                if dt.month in [10, 11, 12, 1, 2, 3, 4]:
                    demand *= 1.08

                # Ramadan/Lebaran surge
                for year, lebaran in FMCG_HOLIDAYS["idul_fitri"].items():
                    lebaran_dt = pd.Timestamp(lebaran)
                    days_to = (lebaran_dt - dt).days
                    if 0 < days_to <= 30:
                        demand *= 1.0 + (0.5 * (1 - days_to / 30))
                    elif -7 <= days_to <= 0:
                        demand *= 0.3  # Post-Lebaran dip

                # Random stock-out (2% chance per product-warehouse-day)
                if rng.random() < 0.02:
                    demand = 0

                # Occasional outlier (0.5% chance)
                if rng.random() < 0.005:
                    demand *= rng.uniform(3, 6)

                # Add noise
                demand = max(0, demand + rng.normal(0, base_demand * 0.15))

                records.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "product_name": product["product_name"],
                    "product_id": product["product_id"],
                    "category": product["category"],
                    "id_gudang": wh,
                    "main_product_sales": round(demand),
                    "promo_sales": round(demand * 0.1) if product["is_promo"] else 0,
                    "prioritas_sales": round(demand * rng.uniform(0, 0.05)),
                    "is_payday_period": int(dt.day >= 25 or dt.day <= 5),
                })

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d order records (%d products x %d warehouses x %d days)",
        len(df), len(products), num_warehouses, len(dates),
    )
    return df


def generate_sales_data(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate revenue and COGS data from order records.

    Args:
        orders: Order records DataFrame.
        products: Product master DataFrame.
        seed: Random seed.

    Returns:
        DataFrame with date, id_gudang, id_region, sales, cogs columns.
    """
    rng = np.random.default_rng(seed)
    merged = orders.merge(products[["product_name", "unit_price"]], on="product_name")
    merged["sales"] = merged["main_product_sales"] * merged["unit_price"]
    merged["cogs"] = merged["sales"] * rng.uniform(0.55, 0.75, size=len(merged))
    merged["id_region"] = merged["id_gudang"].map(
        {f"WH-{chr(65 + i)}": f"R-{i // 2 + 1}" for i in range(10)}
    )

    daily = (
        merged.groupby(["date", "id_region", "id_gudang"])
        .agg({"sales": "sum", "cogs": "sum"})
        .reset_index()
    )
    logger.info("Generated %d daily sales records", len(daily))
    return daily
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add src/fmcg_forecast/data/generator.py tests/test_generator.py
git commit -m "feat: add synthetic FMCG data generator with Indonesian seasonal patterns"
```

---

### Task 7: Create data loader utility

**Files:**
- Create: `src/fmcg_forecast/data/loader.py`

Brief module — loads CSVs into DataFrames with date parsing and validation. Keep it minimal (YAGNI).

```python
# src/fmcg_forecast/data/loader.py
"""Generic data loading utilities."""
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def load_csv(
    path: str | Path,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to CSV file.
        parse_dates: Column names to parse as dates.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, parse_dates=parse_dates)
    logger.info("Loaded %d records from %s", len(df), path.name)
    return df
```

Commit with tests.

---

## Phase 4: Demand Forecasting Pipeline (Pipeline 1)

### Task 8: Create demand preprocessing module

**Source reference:** `yokulak-forecasting/fetch/preprocessing.py`

**Files:**
- Create: `src/fmcg_forecast/demand/preprocessing.py`
- Test: `tests/test_demand_preprocessing.py`

Rewrite the 496-line preprocessing.py into a clean module. Key logic:
1. OOS (out-of-stock) detection — identify periods with zero sales followed by replenishment
2. Data cleaning — remove outliers (z-score >= 3), filter products with min 30 days data
3. Feature engineering — merge calendar features, create moving averages, brand extraction

The test should verify:
- OOS detection marks correct date ranges
- Outlier removal catches z-score >= 3
- Calendar features merge correctly
- Moving averages are calculated

**Step 1: Write failing tests**

```python
# tests/test_demand_preprocessing.py
import pandas as pd
import numpy as np
import pytest


@pytest.mark.unit
def test_detect_oos_periods():
    from fmcg_forecast.demand.preprocessing import detect_oos_periods

    # Create data with a stock-out period
    dates = pd.date_range("2024-01-01", "2024-01-10")
    sales = [10, 15, 0, 0, 0, 20, 25, 30, 15, 10]
    df = pd.DataFrame({"date": dates, "main_product_sales": sales})
    oos = detect_oos_periods(df, min_zero_days=2)
    assert len(oos) == 1
    assert oos.iloc[0]["oos_start"] == pd.Timestamp("2024-01-03")


@pytest.mark.unit
def test_remove_outliers_zscore():
    from fmcg_forecast.demand.preprocessing import remove_outliers

    rng = np.random.default_rng(42)
    sales = rng.normal(100, 10, size=100).tolist()
    sales[50] = 500  # Obvious outlier
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "main_product_sales": sales,
    })
    cleaned = remove_outliers(df, column="main_product_sales", z_threshold=3.0)
    assert len(cleaned) < len(df)


@pytest.mark.unit
def test_preprocess_demand_data_adds_features():
    from fmcg_forecast.demand.preprocessing import preprocess_demand_data

    dates = pd.date_range("2024-01-01", "2024-06-30")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date": dates,
        "product_name": "TestProduct",
        "id_gudang": "WH-A",
        "main_product_sales": rng.integers(5, 50, size=len(dates)),
        "promo_sales": rng.integers(0, 5, size=len(dates)),
        "prioritas_sales": rng.integers(0, 3, size=len(dates)),
        "is_payday_period": ((dates.day >= 25) | (dates.day <= 5)).astype(int),
    })
    result = preprocess_demand_data(df)
    assert "is_ramadan_month" in result.columns
    assert "sales_lag_1" in result.columns or len(result) > 0
```

**Step 3: Implementation** — Rewrite from original preprocessing.py, keeping:
- OOS detection logic
- Outlier removal (z-score >= 3)
- Calendar feature merge
- Moving average calculations
- Brand extraction from product names
- But removing: MySQL logging, ClickHouse queries, multi-threading complexity

Commit after tests pass.

---

### Task 9: Create LSTM-Attention model architecture

**Source reference:** `yokulak-forecasting/pytorch/forecast.py` (classes: `QuantileLoss`, `Attention`, `TimeSeriesLSTMAttentionModel`)

**Files:**
- Create: `src/fmcg_forecast/models/lstm_attention.py`
- Test: `tests/test_model.py`

**Step 1: Write failing tests**

```python
# tests/test_model.py
import pytest
import torch


@pytest.mark.unit
def test_quantile_loss_forward():
    from fmcg_forecast.models.lstm_attention import QuantileLoss

    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    preds = torch.randn(4, 10, 3)  # batch=4, horizon=10, quantiles=3
    target = torch.randn(4, 10)
    loss = loss_fn(preds, target)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0


@pytest.mark.unit
def test_attention_module():
    from fmcg_forecast.models.lstm_attention import Attention

    attn = Attention(hidden_dim=32)
    hidden = torch.randn(4, 21, 32)  # batch=4, seq=21, hidden=32
    weights = attn(hidden)
    assert weights.shape == (4, 21)
    # Weights should sum to ~1 per batch
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


@pytest.mark.unit
def test_lstm_attention_model_forward():
    from fmcg_forecast.models.lstm_attention import TimeSeriesLSTMAttentionModel

    model = TimeSeriesLSTMAttentionModel(
        input_dim=12, hidden_dim=24, num_layers=2,
        output_dim=60, dropout_prob=0.25, num_quantiles=1,
        num_product_embeddings=10, product_embedding_dim=16,
        num_gudang_embeddings=5, gudang_embedding_dim=8,
    )
    x_cont = torch.randn(4, 21, 12)
    x_cat_prod = torch.randint(0, 10, (4, 1))
    x_cat_gudang = torch.randint(0, 5, (4, 1))
    output = model(x_cont, x_cat_prod, x_cat_gudang)
    assert output.shape == (4, 60, 1)


@pytest.mark.unit
def test_sales_lstm_model_forward():
    from fmcg_forecast.models.lstm_attention import SeasonalFinancialLSTMModel

    model = SeasonalFinancialLSTMModel(
        input_dim=1, seasonal_dim=6, hidden_dim=72,
        num_layers=3, output_dim=1, dropout_rate=0.2,
    )
    x = torch.randn(4, 30, 1)
    seasonal = torch.randn(4, 6)
    output = model(x, seasonal)
    assert output.shape == (4, 1)


@pytest.mark.unit
def test_merged_lstm_model_forward():
    from fmcg_forecast.models.lstm_attention import FinancialLSTMModel

    model = FinancialLSTMModel(
        input_dim=8, hidden_dim=72, num_layers=3, output_dim=1,
    )
    x = torch.randn(4, 30, 8)
    output = model(x)
    assert output.shape == (4, 1)
```

**Step 3: Implementation**

Copy the three model classes (`QuantileLoss`, `Attention`, `TimeSeriesLSTMAttentionModel`) from `pytorch/forecast.py` lines 68-128, plus the sales models (`SeasonalFinancialLSTMModel` from `sales/individual/forecasting.py` lines 100-122, and `FinancialLSTMModel` from `sales/merged/forecasting.py` lines 91-105).

All in one file with proper type annotations. No MySQL dependencies.

Commit after tests pass.

---

### Task 10: Create demand forecaster (training + inference)

**Source reference:** `yokulak-forecasting/pytorch/forecast.py` (class: `TimeSeriesForecaster`)

**Files:**
- Create: `src/fmcg_forecast/demand/forecaster.py`
- Test: `tests/test_demand_forecaster.py`

The `TimeSeriesForecaster` class handles:
1. Feature engineering (`_create_features`)
2. Data preparation with STL decomposition (`prepare_data_for_series`)
3. Training with time series cross-validation (`train_global_model`)
4. Prediction with trend extrapolation and spike factors (`predict_future`)
5. Model state save/load

Rewrite cleanly with:
- Type annotations on all methods
- `logging` module instead of MySQL
- Configurable via `DemandConfig`
- Remove ClickHouse/MySQL imports

Test should verify:
- Feature creation adds expected columns (lags, rolling stats, lebaran_proximity_signal)
- Model trains without error on small synthetic data
- Prediction returns expected shape

Commit after tests pass.

---

## Phase 5: Sales Forecasting Pipeline (Pipeline 2)

### Task 11: Create sales data fetcher

**Source reference:** `yokulak-forecasting/sales/fetch.py` (not in the reads but referenced)

**Files:**
- Create: `src/fmcg_forecast/sales/fetch.py`

Simple module that loads sales CSV data generated by the synthetic generator. Replaces the ClickHouse fetcher.

```python
# src/fmcg_forecast/sales/fetch.py
"""Sales data loading and initial aggregation."""
import logging
from pathlib import Path

import pandas as pd

from fmcg_forecast.data.loader import load_csv


logger = logging.getLogger(__name__)


def load_sales_data(raw_data_path: str | Path) -> pd.DataFrame:
    """Load raw sales data and return with parsed dates.

    Args:
        raw_data_path: Path to fetched_sales.csv or equivalent.

    Returns:
        DataFrame with date, id_region, id_gudang, sales, cogs columns.
    """
    df = load_csv(raw_data_path, parse_dates=["date"])
    logger.info("Loaded %d sales records", len(df))
    return df
```

Commit.

---

### Task 12: Create individual sales forecaster

**Source reference:** `yokulak-forecasting/sales/individual/forecasting.py`

**Files:**
- Create: `src/fmcg_forecast/sales/individual.py`
- Test: `tests/test_sales_individual.py`

Rewrite `SeasonalFinancialForecaster` class. Key methods:
- `preprocess_raw_data()` — adjusts dates for Sundays/holidays, aggregates daily by region/gudang
- `prepare_data_with_seasonality()` — creates sinusoidal seasonal features, scales data
- `train_model()` — per-product-warehouse LSTM training with early stopping
- `generate_seasonal_forecast()` — autoregressive multi-step forecast
- `run_forecasting()` — orchestrates training and forecasting for all region-warehouse pairs

Test should verify the full pipeline on tiny synthetic data (3 products, 2 warehouses, 60 days).

Commit after tests pass.

---

### Task 13: Create merged sales/COGS forecaster

**Source reference:** `yokulak-forecasting/sales/merged/forecasting.py`

**Files:**
- Create: `src/fmcg_forecast/sales/merged.py`
- Test: `tests/test_sales_merged.py`

Rewrite `FinancialForecaster` class. Key differences from individual:
- Aggregates all products into single daily time series
- Trains one LSTM for sales, one for COGS
- COGS depends on sales forecast (sequential dependency)
- Uses day-of-week one-hot features instead of sinusoidal

Test should verify:
- Sales model trains and produces positive forecast values
- COGS model uses sales forecast as input feature
- Summary generation works

Commit after tests pass.

---

### Task 14: Create purchase recommendation engine

**Source reference:** `yokulak-forecasting/sales/recommended_buy/individual/rec_buy.py` and `merged/rec_buy.py`

**Files:**
- Create: `src/fmcg_forecast/sales/recommendations.py`
- Test: `tests/test_recommendations.py`

Consolidate both rec_buy modules into one with a `mode` parameter.

Core business logic (preserved exactly):
1. Recommended Buy for Day D-1 = forecasted COGS of Day D (`shift(-1)`)
2. Non-working day amounts redistributed to previous working days (60/40 split)
3. Data integrity verification (total COGS == total recommended buy)

**Step 1: Write failing tests**

```python
# tests/test_recommendations.py
import pandas as pd
import numpy as np
import pytest


@pytest.mark.unit
def test_calculate_recommendations_individual():
    from fmcg_forecast.sales.recommendations import calculate_recommendations

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),  # Mon-Wed start
        "id_region": "R-1",
        "id_gudang": "WH-A",
        "sales": [100] * 10,
        "cogs": [60] * 10,
    })
    holidays = pd.DataFrame({"date": pd.to_datetime([])})
    result = calculate_recommendations(df, holidays, mode="individual")
    assert "recommended_buy" in result.columns
    assert "is_working_day" in result.columns


@pytest.mark.unit
def test_recommendations_redistributes_weekend_amounts():
    from fmcg_forecast.sales.recommendations import calculate_recommendations

    # Create data spanning a weekend (Sat=5, Sun=6)
    dates = pd.date_range("2024-01-01", periods=14)  # Mon to Sun+
    df = pd.DataFrame({
        "date": dates,
        "sales": [100] * 14,
        "cogs": [60] * 14,
    })
    holidays = pd.DataFrame({"date": pd.to_datetime([])})
    result = calculate_recommendations(df, holidays, mode="merged")
    # Weekend days should have 0 recommended buy
    weekend_rows = result[~result["is_working_day"]]
    assert (weekend_rows["recommended_buy"] == 0).all()


@pytest.mark.unit
def test_recommendations_total_equals_cogs():
    from fmcg_forecast.sales.recommendations import calculate_recommendations

    dates = pd.date_range("2024-01-01", periods=30)
    df = pd.DataFrame({
        "date": dates,
        "sales": np.random.default_rng(42).integers(50, 200, size=30),
        "cogs": np.random.default_rng(42).integers(30, 120, size=30),
    })
    holidays = pd.DataFrame({"date": pd.to_datetime([])})
    result = calculate_recommendations(df, holidays, mode="merged")
    assert np.isclose(
        result["cogs"].sum(),
        result["recommended_buy"].sum(),
        rtol=0.01,
    )
```

**Step 3: Implementation**

```python
# src/fmcg_forecast/sales/recommendations.py
"""Purchase quantity recommendation engine.

Calculates recommended buy amounts based on forecasted COGS,
redistributing non-working-day purchases to previous business days.
"""
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _process_group(
    group_df: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Apply recommendation logic to a single region-warehouse group."""
    df = group_df.copy()
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    df["is_working_day"] = ~df.index.dayofweek.isin([5, 6])
    df.loc[df.index.isin(holiday_dates), "is_working_day"] = False
    df["recommended_buy"] = df["cogs"].shift(-1).fillna(0)

    working_day_indices = df.index[df["is_working_day"]]
    if working_day_indices.empty:
        return df.reset_index()

    for current_date in reversed(df.index):
        if not df.loc[current_date, "is_working_day"]:
            amount = df.loc[current_date, "recommended_buy"]
            if amount > 0:
                loc = working_day_indices.searchsorted(current_date)
                buy_day_1 = working_day_indices[loc - 1] if loc > 0 else None
                buy_day_2 = working_day_indices[loc - 2] if loc > 1 else None

                if buy_day_1 and buy_day_2:
                    df.loc[buy_day_2, "recommended_buy"] += amount * 0.6
                    df.loc[buy_day_1, "recommended_buy"] += amount * 0.4
                elif buy_day_1:
                    df.loc[buy_day_1, "recommended_buy"] += amount
            df.loc[current_date, "recommended_buy"] = 0

    return df.reset_index()


def calculate_recommendations(
    forecast_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    mode: str = "merged",
) -> pd.DataFrame:
    """Calculate recommended purchase quantities.

    Args:
        forecast_df: Forecast data with date, sales, cogs columns.
        holidays_df: Holiday dates DataFrame.
        mode: "individual" (per region-gudang) or "merged" (aggregate).

    Returns:
        DataFrame with recommended_buy and is_working_day columns added.
    """
    logger.info("Calculating recommendations (mode=%s)", mode)

    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    holiday_dates = pd.to_datetime(holidays_df["date"]) if len(holidays_df) > 0 else pd.DatetimeIndex([])

    if mode == "individual" and "id_gudang" in df.columns:
        grouped = df.groupby(["id_region", "id_gudang"])
        processed = [_process_group(group, holiday_dates) for _, group in grouped]
        result = pd.concat(processed, ignore_index=True)
    else:
        result = _process_group(df, holiday_dates)

    result["recommended_buy"] = result["recommended_buy"].round(2)

    # Verification
    total_cogs = result["cogs"].sum()
    total_rec = result["recommended_buy"].sum()
    if np.isclose(total_cogs, total_rec):
        logger.info("Verification passed: COGS=%.2f, Rec Buy=%.2f", total_cogs, total_rec)
    else:
        logger.warning(
            "Verification warning: COGS=%.2f, Rec Buy=%.2f, diff=%.2f",
            total_cogs, total_rec, total_cogs - total_rec,
        )

    return result
```

Commit after tests pass.

---

## Phase 6: CLI Scripts

### Task 15: Create CLI scripts for pipeline execution

**Files:**
- Create: `scripts/generate_data.py`
- Create: `scripts/run_demand.py`
- Create: `scripts/run_sales.py`
- Create: `scripts/run_all.py`

Each script uses `argparse` for CLI arguments and orchestrates the appropriate pipeline.

**`scripts/generate_data.py`:**
```python
"""Generate synthetic FMCG data for demo and testing."""
import argparse
import logging

from fmcg_forecast.data.generator import generate_products, generate_orders, generate_sales_data
from fmcg_forecast.data.calendar import create_feature_calendar
from fmcg_forecast.utils.paths import get_data_dirs, ensure_dirs
from fmcg_forecast.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FMCG data")
    parser.add_argument("--skus", type=int, default=80)
    parser.add_argument("--warehouses", type=int, default=5)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logger("fmcg_forecast", level=logging.INFO)
    logger = logging.getLogger(__name__)

    dirs = get_data_dirs()
    ensure_dirs(dirs)

    logger.info("Generating %d products...", args.skus)
    products = generate_products(num_skus=args.skus, seed=args.seed)
    products.to_csv(dirs["synthetic"] / "products.csv", index=False)

    logger.info("Generating order history...")
    orders = generate_orders(products, args.warehouses, args.start_date, args.end_date, args.seed)
    orders.to_csv(dirs["synthetic"] / "orders.csv", index=False)

    logger.info("Generating sales/COGS data...")
    sales = generate_sales_data(orders, products, seed=args.seed)
    sales.to_csv(dirs["synthetic"] / "sales.csv", index=False)

    logger.info("Generating FMCG calendar...")
    calendar = create_feature_calendar(args.start_date, args.end_date)
    calendar.to_csv(dirs["synthetic"] / "calendar.csv", index=False)

    logger.info("Data generation complete. Files saved to %s", dirs["synthetic"])


if __name__ == "__main__":
    main()
```

Similar scripts for `run_demand.py`, `run_sales.py`, and `run_all.py` that:
1. Load config from YAML
2. Load synthetic data
3. Run the appropriate pipeline
4. Save outputs to the data directories

Commit all scripts.

---

## Phase 7: Docker

### Task 16: Create Docker configuration

**Source reference:** `yokulak-forecasting/Dockerfile` and `docker-compose.yml`

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`

Simplified from original — no ClickHouse service, no Rust dependency, just GPU-ready Python.

```dockerfile
# docker/Dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip3 install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cu124

COPY . .
RUN mkdir -p data logs

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "scripts.run_all"]
```

```yaml
# docker/docker-compose.yml
services:
  forecast:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../data:/app/data
      - ../logs:/app/logs
      - ../config:/app/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - PYTHONPATH=/app/src
    command: python3 scripts/run_all.py --config config/default.yaml
```

Commit.

---

## Phase 8: Testing & Quality

### Task 17: Create conftest.py with shared fixtures

**Files:**
- Create: `tests/conftest.py`

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_products():
    """Small product master for testing."""
    from fmcg_forecast.data.generator import generate_products
    return generate_products(num_skus=5, seed=42)


@pytest.fixture
def sample_orders(sample_products):
    """Small order dataset for testing."""
    from fmcg_forecast.data.generator import generate_orders
    return generate_orders(
        sample_products,
        num_warehouses=2,
        start_date="2024-01-01",
        end_date="2024-03-31",
        seed=42,
    )


@pytest.fixture
def sample_sales(sample_orders, sample_products):
    """Small sales dataset for testing."""
    from fmcg_forecast.data.generator import generate_sales_data
    return generate_sales_data(sample_orders, sample_products, seed=42)


@pytest.fixture
def sample_calendar():
    """Small calendar for testing."""
    from fmcg_forecast.data.calendar import create_feature_calendar
    return create_feature_calendar("2024-01-01", "2024-12-31")


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output
```

Commit.

---

### Task 18: Run full test suite and verify 80%+ coverage

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80
```

Fix any failing tests or coverage gaps. Add additional tests where needed.

Commit.

---

## Phase 9: Documentation

### Task 19: Create README.md with ASCII diagrams

**Files:**
- Create: `README.md`

Portfolio-grade README following the documentation guidelines with ASCII box diagrams.

Structure:
1. Title + tagline
2. Overview (problem, solution, results)
3. Architecture (dual-pipeline ASCII diagram)
4. Key Features
5. Quick Start (5 commands)
6. Model Architecture (LSTM-Attention ASCII)
7. Indonesian Calendar Features table
8. Project Structure
9. Configuration
10. Docker
11. License

Commit.

---

### Task 20: Create CLAUDE.md for the project

**Files:**
- Create: `CLAUDE.md`

Project instructions for Claude Code. Include key patterns, file paths, test commands, and domain context.

Commit.

---

## Phase 10: Demo Notebook

### Task 21: Create end-to-end demo notebook

**Files:**
- Create: `notebooks/demo.ipynb`

Jupyter notebook with cells:
1. Introduction (markdown)
2. Generate synthetic data
3. Visualize calendar features
4. Run demand preprocessing
5. Train demand model (small config for fast demo)
6. Plot demand forecasts
7. Run sales forecasting
8. Plot sales + COGS forecasts
9. Generate buying recommendations
10. Summary table

Use small parameters for fast execution:
- 10 SKUs, 2 warehouses, 6 months data
- 50 epochs, input_window=7

Commit.

---

## Phase 11: Final Polish

### Task 22: Update root CLAUDE.md to include fmcg-demand-forecast

**Files:**
- Modify: `C:\working-space\CLAUDE.md`

Add `fmcg-demand-forecast` to the project index table.

Commit (in the root workspace, not the project repo).

---

### Task 23: Final review and cleanup

- Run `ruff fmt .` and `ruff check --fix .`
- Run `mypy src/` and fix type errors
- Run full test suite one more time
- Verify the Quick Start commands work end-to-end
- Review README for accuracy

Final commit.

---

## Execution Order & Dependencies

```
Phase 1 (Task 1)
    │
Phase 2 (Tasks 2-4) ── shared utils, must come first
    │
Phase 3 (Tasks 5-7) ── data layer depends on utils
    │
    ├── Phase 4 (Tasks 8-10) ── demand pipeline depends on data layer
    │
    └── Phase 5 (Tasks 11-14) ── sales pipeline depends on data layer
         │
Phase 6 (Task 15) ── CLI scripts depend on both pipelines
    │
Phase 7 (Task 16) ── Docker depends on project structure
    │
Phase 8 (Tasks 17-18) ── testing depends on all code
    │
Phase 9 (Tasks 19-20) ── docs depend on understanding full system
    │
Phase 10 (Task 21) ── notebook depends on working pipeline
    │
Phase 11 (Tasks 22-23) ── final polish
```

**Total: 23 tasks across 11 phases.**
**Estimated commits: ~25 (one per task plus fixes).**
