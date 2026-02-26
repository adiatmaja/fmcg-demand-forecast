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

    model_config = {"env_prefix": "FMCG_FORECAST_"}

    model_params: ModelConfig = ModelConfig()
    input_window: int = 21
    forecast_horizon: int = 60
    epochs: int = 500
    batch_size: int = 128
    cv_splits: int = 5
    quantiles: list[float] = [0.5]


class SalesConfig(BaseSettings):
    """Sales forecasting pipeline configuration."""

    model_config = {"env_prefix": "FMCG_SALES_"}

    model_params: SalesModelConfig = SalesModelConfig()
    input_window: int = 30
    forecast_horizon: int = 90
    epochs: int = 200
    batch_size: int = 64


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
