"""Tests for SeasonalFinancialForecaster (individual sales per region-warehouse)."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_sales_df():
    """Minimal sales DataFrame: 3 products, 2 warehouses, 60 days."""
    rng = np.random.default_rng(42)
    records = []
    dates = pd.date_range("2024-01-01", periods=60)
    for region in ["R-1", "R-2"]:
        for gudang in ["WH-A", "WH-B"]:
            for date in dates:
                records.append(
                    {
                        "date": date,
                        "id_region": region,
                        "id_gudang": gudang,
                        "sales": float(rng.integers(500, 2000)),
                        "cogs": float(rng.integers(300, 1200)),
                    }
                )
    return pd.DataFrame(records)


@pytest.fixture
def empty_holidays():
    return pd.DataFrame({"date": pd.to_datetime([])})


@pytest.mark.unit
def test_create_seasonal_features_shape(small_sales_df):
    from fmcg_forecast.config import SalesConfig
    from fmcg_forecast.sales.individual import SeasonalFinancialForecaster

    cfg = SalesConfig(epochs=2, input_window=7, forecast_horizon=5, batch_size=4)
    forecaster = SeasonalFinancialForecaster(cfg, holiday_dates=set())
    dates = small_sales_df["date"].unique()[:10]
    features = forecaster.create_seasonal_features(list(dates))
    assert features.shape == (10, 6)


@pytest.mark.unit
def test_prepare_data_returns_tensors(small_sales_df):
    import torch

    from fmcg_forecast.config import SalesConfig
    from fmcg_forecast.sales.individual import SeasonalFinancialForecaster

    cfg = SalesConfig(epochs=2, input_window=7, forecast_horizon=5, batch_size=4)
    forecaster = SeasonalFinancialForecaster(cfg, holiday_dates=set())
    df_rg = small_sales_df[
        (small_sales_df["id_region"] == "R-1") & (small_sales_df["id_gudang"] == "WH-A")
    ].copy()
    X, y, seasonal, _ = forecaster.prepare_data_with_seasonality(
        df_rg, "sales", "R-1", "WH-A"
    )
    assert X is not None
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[1] == cfg.input_window


@pytest.mark.slow
def test_run_forecasting_returns_results(small_sales_df, empty_holidays):
    from fmcg_forecast.config import SalesConfig
    from fmcg_forecast.sales.individual import (
        SeasonalFinancialForecaster,
        preprocess_raw_data,
    )

    cfg = SalesConfig(epochs=2, input_window=7, forecast_horizon=5, batch_size=4)
    data_dict = preprocess_raw_data(small_sales_df, holiday_dates=set())
    forecaster = SeasonalFinancialForecaster(cfg, holiday_dates=set())
    results = forecaster.run_forecasting(data_dict)
    assert len(results) > 0


@pytest.mark.slow
def test_forecast_output_has_expected_columns(small_sales_df):
    from fmcg_forecast.config import SalesConfig
    from fmcg_forecast.sales.individual import (
        SeasonalFinancialForecaster,
        preprocess_raw_data,
    )

    cfg = SalesConfig(epochs=2, input_window=7, forecast_horizon=5, batch_size=4)
    data_dict = preprocess_raw_data(small_sales_df, holiday_dates=set())
    forecaster = SeasonalFinancialForecaster(cfg, holiday_dates=set())
    results = forecaster.run_forecasting(data_dict)
    for result in results.values():
        assert "forecast_df" in result
        assert not result["forecast_df"].empty
