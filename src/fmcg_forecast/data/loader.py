"""Generic data loading utilities."""
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def load_csv(
    path: str | Path,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to CSV file.
        parse_dates: Column names to parse as dates.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, parse_dates=parse_dates)
    logger.info("Loaded %d records from %s", len(df), path.name)
    return df
