import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
def test_detect_oos_periods():
    from fmcg_forecast.demand.preprocessing import detect_oos_periods

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
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100),
            "main_product_sales": sales,
        }
    )
    cleaned = remove_outliers(df, column="main_product_sales", z_threshold=3.0)
    assert len(cleaned) < len(df)
    assert 500 not in cleaned["main_product_sales"].values


@pytest.mark.unit
def test_remove_outliers_constant_series():
    from fmcg_forecast.demand.preprocessing import remove_outliers

    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "main_product_sales": [100.0] * 10,
        }
    )
    cleaned = remove_outliers(df, column="main_product_sales", z_threshold=3.0)
    # Constant series — std=0, nothing removed
    assert len(cleaned) == len(df)


@pytest.mark.unit
def test_preprocess_demand_data_adds_calendar_features():
    from fmcg_forecast.demand.preprocessing import preprocess_demand_data

    dates = pd.date_range("2024-01-01", "2024-06-30")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "product_name": "TestProduct BEV-01",
            "id_gudang": "WH-A",
            "main_product_sales": rng.integers(5, 50, size=len(dates)).astype(float),
            "promo_sales": rng.integers(0, 5, size=len(dates)).astype(float),
            "prioritas_sales": rng.integers(0, 3, size=len(dates)).astype(float),
            "is_payday_period": ((dates.day >= 25) | (dates.day <= 5)).astype(int),
        }
    )
    result = preprocess_demand_data(df)
    assert "is_ramadan_month" in result.columns
    assert "sales_lag_1" in result.columns
    assert "lebaran_proximity_signal" in result.columns
    assert len(result) > 0
