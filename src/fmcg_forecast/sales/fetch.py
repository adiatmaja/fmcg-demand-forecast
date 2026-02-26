"""Sales data loading and initial aggregation."""
import logging
from pathlib import Path

import pandas as pd

from fmcg_forecast.data.loader import load_csv


logger = logging.getLogger(__name__)


def load_sales_data(raw_data_path: str | Path) -> pd.DataFrame:
    """Load raw sales data and return with parsed dates.

    Args:
        raw_data_path: Path to fetched_sales.csv or equivalent.

    Returns:
        DataFrame with date, id_region, id_gudang, sales, cogs columns.
    """
    df = load_csv(raw_data_path, parse_dates=["date"])
    logger.info("Loaded %d sales records", len(df))
    return df
