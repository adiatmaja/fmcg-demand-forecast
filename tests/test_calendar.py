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
        "date",
        "is_ramadan_month",
        "is_lebaran_peak_week",
        "is_thr_payout_week",
        "days_until_lebaran",
        "days_after_lebaran",
        "is_idul_adha_week",
        "is_year_end_holiday_period",
        "is_independence_day_week",
        "is_cny_week",
        "is_rainy_season",
        "is_payday_period",
        "is_back_to_school",
        "is_long_weekend",
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
