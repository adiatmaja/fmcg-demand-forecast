"""Tests for FinancialForecaster (merged/aggregated sales and COGS)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_merged_df():
    """60 days of daily aggregated sales + COGS."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=60)
    return pd.DataFrame(
        {
            "date": dates,
            "sales": rng.integers(5000, 20000, size=60).astype(float),
            "cogs": rng.integers(3000, 12000, size=60).astype(float),
        }
    )


@pytest.mark.unit
def test_prepare_data_sales_returns_tensors(small_merged_df):
    import torch

    from fmcg_forecast.sales.merged import FinancialForecaster
    from fmcg_forecast.config import SalesConfig

    cfg = SalesConfig(epochs=2, input_window=7, forecast_horizon=5, batch_size=4)
    forecaster = FinancialForecaster(cfg, holiday_dates=set())
    X, y, processed = forecaster.prepare_data_sales(small_merged_df)
    assert X is not None
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[1] == cfg.input_window


@pytest.mark.unit
def test_prepare_data_cogs_uses_sales_feature(small_merged_df):
    import torch

    from fmcg_forecast.sales.merged import FinancialForecaster
    from fmcg_forecast.config import SalesConfig

    cfg = SalesConfig(epochs=2, input_window=7, forecast_horizon=5, batch_size=4)
    forecaster = FinancialForecaster(cfg, holiday_dates=set())
    forecaster.prepare_data_sales(small_merged_df)  # fit sales scaler first
    X, y, _ = forecaster.prepare_data_cogs(small_merged_df, small_merged_df)
    assert X is not None
    assert isinstance(X, torch.Tensor)
    # COGS input features = sales + cogs + 7 day-of-week one-hots = 9
    assert X.shape[2] == 9


@pytest.mark.slow
def test_run_forecasting_returns_both_forecasts(small_merged_df):
    from fmcg_forecast.sales.merged import FinancialForecaster
    from fmcg_forecast.config import SalesConfig

    cfg = SalesConfig(epochs=2, input_window=7, forecast_horizon=5, batch_size=4)
    forecaster = FinancialForecaster(cfg, holiday_dates=set())
    results = forecaster.run_forecasting(small_merged_df, small_merged_df)
    assert "sales" in results
    assert "cogs" in results
    assert not results["sales"]["forecast_df"].empty
    assert not results["cogs"]["forecast_df"].empty


@pytest.mark.slow
def test_sales_forecast_is_positive(small_merged_df):
    from fmcg_forecast.sales.merged import FinancialForecaster
    from fmcg_forecast.config import SalesConfig

    cfg = SalesConfig(epochs=2, input_window=7, forecast_horizon=5, batch_size=4)
    forecaster = FinancialForecaster(cfg, holiday_dates=set())
    results = forecaster.run_forecasting(small_merged_df, small_merged_df)
    sales_fc = results["sales"]["forecast_df"]
    assert (sales_fc["sales"] >= 0).all()
