"""Tests for the purchase recommendation engine."""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
def test_calculate_recommendations_individual():
    from fmcg_forecast.sales.recommendations import calculate_recommendations

    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "id_region": "R-1",
            "id_gudang": "WH-A",
            "sales": [100.0] * 10,
            "cogs": [60.0] * 10,
        }
    )
    holidays = pd.DataFrame({"date": pd.to_datetime([])})
    result = calculate_recommendations(df, holidays, mode="individual")
    assert "recommended_buy" in result.columns
    assert "is_working_day" in result.columns


@pytest.mark.unit
def test_recommendations_redistributes_weekend_amounts():
    from fmcg_forecast.sales.recommendations import calculate_recommendations

    dates = pd.date_range("2024-01-01", periods=14)  # Mon Jan 1 to Sun Jan 14
    df = pd.DataFrame(
        {
            "date": dates,
            "sales": [100.0] * 14,
            "cogs": [60.0] * 14,
        }
    )
    holidays = pd.DataFrame({"date": pd.to_datetime([])})
    result = calculate_recommendations(df, holidays, mode="merged")
    weekend_rows = result[~result["is_working_day"]]
    assert (weekend_rows["recommended_buy"] == 0).all()


@pytest.mark.unit
def test_recommendations_total_equals_cogs():
    from fmcg_forecast.sales.recommendations import calculate_recommendations

    dates = pd.date_range("2024-01-01", periods=30)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "sales": rng.integers(50, 200, size=30).astype(float),
            "cogs": rng.integers(30, 120, size=30).astype(float),
        }
    )
    holidays = pd.DataFrame({"date": pd.to_datetime([])})
    result = calculate_recommendations(df, holidays, mode="merged")
    assert np.isclose(
        result["cogs"].sum(),
        result["recommended_buy"].sum(),
        rtol=0.01,
    )
