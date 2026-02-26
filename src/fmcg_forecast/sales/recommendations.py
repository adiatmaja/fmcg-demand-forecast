"""Purchase quantity recommendation engine.

Calculates recommended buy amounts based on forecasted COGS,
redistributing non-working-day purchases to previous business days.

Business logic (preserved exactly from yokulak-forecasting):
1. Recommended Buy for Day D-1 = forecasted COGS of Day D (shift(-1))
2. Non-working day amounts redistributed to previous working days (60/40 split)
3. Data integrity: total COGS == total recommended buy
"""

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _process_group(
    group_df: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Apply recommendation logic to a single region-warehouse group.

    Args:
        group_df: DataFrame with date, sales, cogs columns for one group.
        holiday_dates: DatetimeIndex of public holidays.

    Returns:
        DataFrame with is_working_day and recommended_buy columns added.
    """
    df = group_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    df["is_working_day"] = ~df.index.dayofweek.isin([5, 6])
    if len(holiday_dates) > 0:
        df.loc[df.index.isin(holiday_dates), "is_working_day"] = False

    # Each day's recommended buy = that day's COGS.
    # Non-working-day amounts get redistributed to preceding working days,
    # so working days that precede a non-working day also cover it.
    df["recommended_buy"] = df["cogs"].copy()

    working_day_indices = df.index[df["is_working_day"]]
    if working_day_indices.empty:
        return df.reset_index()

    # Redistribute non-working-day amounts to preceding business days (60/40)
    for current_date in reversed(df.index):
        if not df.loc[current_date, "is_working_day"]:
            amount = df.loc[current_date, "recommended_buy"]
            if amount > 0:
                loc = working_day_indices.searchsorted(current_date)
                buy_day_1 = working_day_indices[loc - 1] if loc > 0 else None
                buy_day_2 = working_day_indices[loc - 2] if loc > 1 else None

                if buy_day_1 is not None and buy_day_2 is not None:
                    df.loc[buy_day_2, "recommended_buy"] += amount * 0.6
                    df.loc[buy_day_1, "recommended_buy"] += amount * 0.4
                elif buy_day_1 is not None:
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
        holidays_df: DataFrame with a 'date' column of public holidays.
        mode: "individual" (per region-gudang) or "merged" (aggregate).

    Returns:
        DataFrame with recommended_buy and is_working_day columns added.
    """
    logger.info("Calculating recommendations (mode=%s)", mode)

    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if len(holidays_df) > 0:
        holiday_dates = pd.DatetimeIndex(pd.to_datetime(holidays_df["date"]))
    else:
        holiday_dates = pd.DatetimeIndex([])

    if mode == "individual" and "id_gudang" in df.columns and "id_region" in df.columns:
        processed = [
            _process_group(group, holiday_dates)
            for _, group in df.groupby(["id_region", "id_gudang"])
        ]
        result = pd.concat(processed, ignore_index=True)
    else:
        result = _process_group(df, holiday_dates)

    result["recommended_buy"] = result["recommended_buy"].round(2)

    # Integrity check
    total_cogs = result["cogs"].sum()
    total_rec = result["recommended_buy"].sum()
    if np.isclose(total_cogs, total_rec, rtol=0.01):
        logger.info(
            "Verification passed: COGS=%.2f, Rec Buy=%.2f", total_cogs, total_rec
        )
    else:
        logger.warning(
            "Verification warning: COGS=%.2f, Rec Buy=%.2f, diff=%.2f",
            total_cogs,
            total_rec,
            total_cogs - total_rec,
        )

    return result
