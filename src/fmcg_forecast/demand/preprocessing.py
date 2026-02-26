"""Demand forecasting preprocessing pipeline.

Handles OOS detection, outlier removal, calendar feature merging,
and lag/rolling feature engineering — all without database dependencies.
"""
import logging
from datetime import date

import numpy as np
import pandas as pd

from fmcg_forecast.data.calendar import FMCG_HOLIDAYS, create_feature_calendar


logger = logging.getLogger(__name__)

# Lebaran dates for proximity signal (up to 2026)
_LEBARAN_DATES: list[date] = list(FMCG_HOLIDAYS["idul_fitri"].values())


def detect_oos_periods(
    df: pd.DataFrame,
    min_zero_days: int = 2,
    sales_col: str = "main_product_sales",
    date_col: str = "date",
) -> pd.DataFrame:
    """Detect out-of-stock periods (consecutive zero-sales runs).

    Args:
        df: DataFrame with date and sales columns.
        min_zero_days: Minimum consecutive zeros to flag as OOS.
        sales_col: Name of the sales column.
        date_col: Name of the date column.

    Returns:
        DataFrame with oos_start and oos_end columns for each OOS period found.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    is_zero = (df[sales_col] == 0).astype(int)
    # Label each consecutive zero run with a unique block id
    blocks = (is_zero.diff() != 0).cumsum()
    zero_runs = df[is_zero == 1].groupby(blocks[is_zero == 1]).agg(
        oos_start=(date_col, "min"),
        oos_end=(date_col, "max"),
        run_length=(date_col, "count"),
    )
    oos = zero_runs[zero_runs["run_length"] >= min_zero_days].reset_index(drop=True)
    logger.info("Detected %d OOS periods (min_zero_days=%d)", len(oos), min_zero_days)
    return oos[["oos_start", "oos_end", "run_length"]]


def remove_outliers(
    df: pd.DataFrame,
    column: str = "main_product_sales",
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Remove rows where the column value exceeds z_threshold standard deviations.

    Args:
        df: Input DataFrame.
        column: Column to compute z-scores on.
        z_threshold: Z-score cutoff.

    Returns:
        DataFrame with outlier rows removed.
    """
    if df.empty or column not in df.columns:
        return df

    sales = df[column]
    mean, std = sales.mean(), sales.std()
    if std == 0 or pd.isna(std):
        return df

    z_scores = np.abs((sales - mean) / std)
    n_removed = int((z_scores >= z_threshold).sum())
    if n_removed:
        logger.debug("Removed %d outlier rows (z >= %.1f)", n_removed, z_threshold)
    return df[z_scores < z_threshold].copy()


def preprocess_demand_data(
    df: pd.DataFrame,
    date_col: str = "date",
    sales_col: str = "main_product_sales",
) -> pd.DataFrame:
    """Full preprocessing pipeline for demand forecasting.

    Merges Indonesian FMCG calendar features, adds lag/rolling features,
    and computes the Lebaran proximity signal.

    Args:
        df: Raw demand DataFrame with date, product_name, id_gudang, sales columns.
        date_col: Name of the date column.
        sales_col: Name of the main sales column.

    Returns:
        Preprocessed DataFrame ready for feature engineering.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Merge Indonesian FMCG calendar features
    start_str = df[date_col].min().strftime("%Y-%m-%d")
    end_str = df[date_col].max().strftime("%Y-%m-%d")
    calendar = create_feature_calendar(start_str, end_str)
    calendar[date_col] = pd.to_datetime(calendar[date_col])

    df = df.merge(calendar, on=date_col, how="left")

    # Lag features
    for lag in [1, 7, 14]:
        df[f"sales_lag_{lag}"] = df[sales_col].shift(lag)

    # Rolling statistics
    for window in [7, 14]:
        df[f"sales_rolling_mean_{window}"] = df[sales_col].rolling(window=window).mean()
        df[f"sales_rolling_std_{window}"] = df[sales_col].rolling(window=window).std()

    # Lebaran proximity signal (0-1, ramps up 60 days before)
    df["days_until_lebaran_signal"] = 100.0
    for lebaran_date in _LEBARAN_DATES:
        lebaran_ts = pd.Timestamp(lebaran_date)
        delta = (lebaran_ts - df[date_col]).dt.days
        valid = (delta >= 0) & (delta <= 60)
        df.loc[valid, "days_until_lebaran_signal"] = np.minimum(
            df.loc[valid, "days_until_lebaran_signal"], delta[valid]
        )
    df["lebaran_proximity_signal"] = 1.0 - (df["days_until_lebaran_signal"] / 60.0)
    df.loc[df["days_until_lebaran_signal"] > 60, "lebaran_proximity_signal"] = 0.0
    df.drop(columns=["days_until_lebaran_signal"], inplace=True)

    # Interaction: promo on payday
    if "promo_sales" in df.columns and "is_payday_period" in df.columns:
        df["promo_on_payday"] = df["promo_sales"] * df["is_payday_period"]

    # Fill NaN from lags/rolling at the start
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    logger.info("Preprocessed %d demand records with %d features", len(df), len(df.columns))
    return df
