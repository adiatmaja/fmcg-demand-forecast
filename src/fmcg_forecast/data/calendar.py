"""Indonesian FMCG calendar feature engineering.

Generates 13+ temporal features tailored to Indonesian FMCG retail patterns:
Ramadan, Lebaran, payday cycles, rainy season, Chinese New Year, and more.
"""
import logging
from datetime import date, timedelta

import pandas as pd


logger = logging.getLogger(__name__)

# --- Indonesian holiday dates (2018-2026) ---
FMCG_HOLIDAYS: dict[str, dict[int, date]] = {
    "idul_fitri": {
        2018: date(2018, 6, 15),
        2019: date(2019, 6, 5),
        2020: date(2020, 5, 24),
        2021: date(2021, 5, 13),
        2022: date(2022, 5, 2),
        2023: date(2023, 4, 22),
        2024: date(2024, 4, 10),
        2025: date(2025, 3, 31),
        2026: date(2026, 3, 20),
    },
    "ramadan_start": {
        2018: date(2018, 5, 17),
        2019: date(2019, 5, 6),
        2020: date(2020, 4, 24),
        2021: date(2021, 4, 13),
        2022: date(2022, 4, 3),
        2023: date(2023, 3, 23),
        2024: date(2024, 3, 12),
        2025: date(2025, 3, 1),
        2026: date(2026, 2, 18),
    },
    "idul_adha": {
        2018: date(2018, 8, 22),
        2019: date(2019, 8, 11),
        2020: date(2020, 7, 31),
        2021: date(2021, 7, 20),
        2022: date(2022, 7, 10),
        2023: date(2023, 6, 29),
        2024: date(2024, 6, 17),
        2025: date(2025, 6, 7),
        2026: date(2026, 5, 28),
    },
    "chinese_new_year": {
        2018: date(2018, 2, 16),
        2019: date(2019, 2, 5),
        2020: date(2020, 1, 25),
        2021: date(2021, 2, 12),
        2022: date(2022, 2, 1),
        2023: date(2023, 1, 22),
        2024: date(2024, 2, 10),
        2025: date(2025, 1, 29),
        2026: date(2026, 2, 17),
    },
}

# Comprehensive Indonesian public holidays (2018-2026)
ALL_PUBLIC_HOLIDAYS: set[date] = {
    date(2018, 1, 1), date(2018, 2, 16), date(2018, 3, 17), date(2018, 3, 30),
    date(2018, 4, 14), date(2018, 5, 1), date(2018, 5, 10), date(2018, 5, 29),
    date(2018, 6, 1), date(2018, 6, 15), date(2018, 6, 16), date(2018, 8, 17),
    date(2018, 8, 22), date(2018, 9, 11), date(2018, 11, 20), date(2018, 12, 25),
    date(2019, 1, 1), date(2019, 2, 5), date(2019, 3, 7), date(2019, 4, 3),
    date(2019, 4, 19), date(2019, 5, 1), date(2019, 5, 19), date(2019, 5, 30),
    date(2019, 6, 1), date(2019, 6, 5), date(2019, 6, 6), date(2019, 8, 11),
    date(2019, 8, 17), date(2019, 9, 1), date(2019, 11, 9), date(2019, 12, 25),
    date(2020, 1, 1), date(2020, 1, 25), date(2020, 3, 22), date(2020, 3, 25),
    date(2020, 4, 10), date(2020, 5, 1), date(2020, 5, 7), date(2020, 5, 21),
    date(2020, 5, 24), date(2020, 5, 25), date(2020, 6, 1), date(2020, 7, 31),
    date(2020, 8, 17), date(2020, 8, 20), date(2020, 10, 29), date(2020, 12, 25),
    date(2021, 1, 1), date(2021, 2, 12), date(2021, 3, 11), date(2021, 3, 14),
    date(2021, 4, 2), date(2021, 5, 1), date(2021, 5, 13), date(2021, 5, 14),
    date(2021, 5, 26), date(2021, 6, 1), date(2021, 7, 20), date(2021, 8, 10),
    date(2021, 8, 17), date(2021, 10, 19), date(2021, 12, 25),
    date(2022, 1, 1), date(2022, 2, 1), date(2022, 2, 28), date(2022, 3, 3),
    date(2022, 4, 15), date(2022, 5, 1), date(2022, 5, 2), date(2022, 5, 3),
    date(2022, 5, 16), date(2022, 5, 26), date(2022, 6, 1), date(2022, 7, 10),
    date(2022, 7, 30), date(2022, 8, 17), date(2022, 10, 8), date(2022, 12, 25),
    date(2023, 1, 1), date(2023, 1, 22), date(2023, 2, 18), date(2023, 3, 22),
    date(2023, 4, 7), date(2023, 4, 22), date(2023, 4, 23), date(2023, 5, 1),
    date(2023, 5, 18), date(2023, 6, 1), date(2023, 6, 4), date(2023, 6, 29),
    date(2023, 7, 19), date(2023, 8, 17), date(2023, 9, 28), date(2023, 12, 25),
    date(2024, 1, 1), date(2024, 2, 8), date(2024, 2, 10), date(2024, 3, 11),
    date(2024, 3, 29), date(2024, 4, 10), date(2024, 4, 11), date(2024, 5, 1),
    date(2024, 5, 9), date(2024, 5, 23), date(2024, 6, 1), date(2024, 6, 17),
    date(2024, 7, 7), date(2024, 8, 17), date(2024, 9, 16), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 29), date(2025, 3, 3), date(2025, 3, 31),
    date(2025, 4, 1), date(2025, 4, 18), date(2025, 5, 1), date(2025, 5, 12),
    date(2025, 5, 29), date(2025, 6, 1), date(2025, 6, 7), date(2025, 6, 26),
    date(2025, 8, 17), date(2025, 9, 5), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 2, 17), date(2026, 2, 27), date(2026, 3, 19),
    date(2026, 3, 20), date(2026, 3, 21), date(2026, 4, 3), date(2026, 5, 1),
    date(2026, 5, 14), date(2026, 5, 28), date(2026, 6, 1), date(2026, 6, 2),
    date(2026, 6, 18), date(2026, 8, 17), date(2026, 8, 27), date(2026, 12, 25),
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
    for col in base_features:
        df[col] = 0

    # Static seasonal features
    df["is_rainy_season"] = df["date"].dt.month.isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    df["is_payday_period"] = ((df["date"].dt.day >= 25) | (df["date"].dt.day <= 5)).astype(int)
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

        df.loc[(df["date"] >= ramadan_start) & (df["date"] < lebaran), "is_ramadan_month"] = 1
        df.loc[
            (df["date"] >= lebaran - timedelta(days=7)) & (df["date"] < lebaran),
            "is_lebaran_peak_week",
        ] = 1
        df.loc[
            (df["date"] >= lebaran - timedelta(days=14))
            & (df["date"] <= lebaran - timedelta(days=8)),
            "is_thr_payout_week",
        ] = 1
        df.loc[
            (df["date"] >= lebaran - timedelta(days=30)) & (df["date"] <= lebaran),
            "days_until_lebaran",
        ] = (df["date"] - lebaran).dt.days
        df.loc[
            (df["date"] > lebaran) & (df["date"] <= lebaran + timedelta(days=14)),
            "days_after_lebaran",
        ] = (df["date"] - lebaran).dt.days

        adha = pd.to_datetime(FMCG_HOLIDAYS["idul_adha"][year])
        df.loc[
            (df["date"] >= adha - timedelta(days=3)) & (df["date"] <= adha + timedelta(days=3)),
            "is_idul_adha_week",
        ] = 1

        cny = pd.to_datetime(FMCG_HOLIDAYS["chinese_new_year"][year])
        df.loc[
            (df["date"] >= cny - timedelta(days=3)) & (df["date"] <= cny + timedelta(days=3)),
            "is_cny_week",
        ] = 1

    # Year-end and Independence Day
    df.loc[
        (df["date"].dt.month == 12) & (df["date"].dt.day >= 20)
        | (df["date"].dt.month == 1) & (df["date"].dt.day <= 2),
        "is_year_end_holiday_period",
    ] = 1
    df.loc[
        (df["date"].dt.month == 8) & (df["date"].dt.day >= 14) & (df["date"].dt.day <= 20),
        "is_independence_day_week",
    ] = 1

    # Interaction features
    df["interaction_payday_lebaran"] = df["is_payday_period"] * df["is_lebaran_peak_week"]
    df["interaction_payday_year_end"] = df["is_payday_period"] * df["is_year_end_holiday_period"]
    df["interaction_payday_long_weekend"] = df["is_payday_period"] * df["is_long_weekend"]

    # Sequential features
    weekend_blocks = (df["is_long_weekend"].diff() != 0).cumsum()
    df["long_weekend_day_number"] = df.groupby(weekend_blocks).cumcount() + 1
    df.loc[df["is_long_weekend"] == 0, "long_weekend_day_number"] = 0

    # Format date as string for CSV compatibility
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    logger.info("Calendar generated: %d days, %d features", len(df), len(df.columns) - 1)
    return df
