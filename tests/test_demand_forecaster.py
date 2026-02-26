import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_demand_df():
    """Minimal demand DataFrame for fast testing (1 product, 1 warehouse, 120 days)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=120)
    return pd.DataFrame(
        {
            "date": dates,
            "product_name": "TestProd BEV-01",
            "id_gudang": "WH-A",
            "main_product_sales": rng.integers(10, 60, size=120).astype(float),
            "promo_sales": rng.integers(0, 5, size=120).astype(float),
            "prioritas_sales": rng.integers(0, 3, size=120).astype(float),
            "is_payday_period": ((dates.day >= 25) | (dates.day <= 5)).astype(int),
        }
    )


@pytest.mark.unit
def test_create_features_adds_lag_columns(small_demand_df):
    from fmcg_forecast.demand.forecaster import TimeSeriesForecaster
    from fmcg_forecast.config import DemandConfig

    cfg = DemandConfig(epochs=2, input_window=7, forecast_horizon=10, batch_size=4)
    forecaster = TimeSeriesForecaster(cfg)
    result = forecaster._create_features(small_demand_df)
    assert "sales_lag_1" in result.columns
    assert "sales_lag_7" in result.columns
    assert "lebaran_proximity_signal" in result.columns
    assert "sales_rolling_mean_7" in result.columns


@pytest.mark.unit
def test_prepare_data_returns_tensors(small_demand_df):
    from fmcg_forecast.demand.forecaster import TimeSeriesForecaster
    from fmcg_forecast.config import DemandConfig
    import torch

    cfg = DemandConfig(epochs=2, input_window=7, forecast_horizon=10, batch_size=4)
    forecaster = TimeSeriesForecaster(cfg)
    X, y, _ = forecaster.prepare_data_for_series(small_demand_df, "TestProd_WH-A")
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[1] == cfg.input_window
    assert y.shape[1] == cfg.forecast_horizon


@pytest.mark.slow
def test_train_global_model_runs(small_demand_df):
    from fmcg_forecast.demand.forecaster import TimeSeriesForecaster
    from fmcg_forecast.config import DemandConfig

    cfg = DemandConfig(epochs=2, input_window=7, forecast_horizon=10, batch_size=4, cv_splits=2)
    forecaster = TimeSeriesForecaster(cfg)
    forecaster.train_global_model(small_demand_df)
    assert forecaster.global_model is not None


@pytest.mark.slow
def test_predict_future_returns_dataframe(small_demand_df):
    from fmcg_forecast.demand.forecaster import TimeSeriesForecaster
    from fmcg_forecast.config import DemandConfig

    cfg = DemandConfig(epochs=2, input_window=7, forecast_horizon=10, batch_size=4, cv_splits=2)
    forecaster = TimeSeriesForecaster(cfg)
    forecaster.train_global_model(small_demand_df)
    result = forecaster.predict_future(small_demand_df, "2024-05-01")
    assert not result.empty
    assert "forecast_value" in result.columns
    assert "forecast_date" in result.columns
