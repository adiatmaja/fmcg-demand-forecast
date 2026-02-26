"""Centralized path management for the forecasting pipeline."""
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_data_dirs(project_root: Path | None = None) -> dict[str, Path]:
    """Return a dict of all standard data directories.

    Args:
        project_root: Override project root. Uses auto-detected root if None.

    Returns:
        Dictionary mapping directory names to Path objects.
    """
    root = project_root or get_project_root()
    data = root / "data"

    return {
        "data": data,
        "logs": root / "logs",
        "synthetic": data / "synthetic",
        "demand_raw": data / "demand" / "01_raw",
        "demand_preprocessed": data / "demand" / "02_preprocessed",
        "demand_forecast": data / "demand" / "03_forecast",
        "sales_raw": data / "sales" / "01_raw",
        "sales_forecast": data / "sales" / "02_forecast",
        "sales_rec_buy": data / "sales" / "03_recommended_buy",
    }


def ensure_dirs(dirs: dict[str, Path]) -> None:
    """Create all directories in the dict if they don't exist."""
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
