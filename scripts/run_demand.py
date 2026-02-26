"""Run the demand forecasting pipeline."""
import argparse
import logging

import pandas as pd

from fmcg_forecast.config import DemandConfig, load_yaml_config
from fmcg_forecast.demand.forecaster import TimeSeriesForecaster
from fmcg_forecast.demand.preprocessing import preprocess_demand_data
from fmcg_forecast.utils.logging import setup_logger
from fmcg_forecast.utils.paths import ensure_dirs, get_data_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FMCG demand forecasting pipeline")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--data", default=None, help="Path to input CSV (default: data/synthetic/sales.csv)"
    )
    parser.add_argument(
        "--forecast-start", default=None, help="Forecast start date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    setup_logger("fmcg_forecast", level=logging.INFO)
    logger = logging.getLogger(__name__)

    dirs = get_data_dirs()
    ensure_dirs(dirs)

    # Load config
    yaml_cfg = load_yaml_config(args.config)
    demand_yaml = yaml_cfg.get("demand", {})
    cfg = DemandConfig(**demand_yaml)

    # Load data
    data_path = args.data or str(dirs["synthetic"] / "sales.csv")
    logger.info("Loading demand data from %s", data_path)
    df = pd.read_csv(data_path, parse_dates=["date"])

    # Preprocess
    logger.info("Preprocessing...")
    df = preprocess_demand_data(df)

    # Train
    logger.info("Training global demand model...")
    forecaster = TimeSeriesForecaster(cfg)
    forecaster.train_global_model(df)

    # Forecast
    forecast_start = args.forecast_start or str(df["date"].max().date())
    logger.info("Generating forecast from %s...", forecast_start)
    forecast_df = forecaster.predict_future(df, forecast_start)

    # Save
    out_path = dirs["demand_forecast"] / "demand_forecast.csv"
    forecast_df.to_csv(out_path, index=False)
    logger.info("Demand forecast saved to %s (%d rows)", out_path, len(forecast_df))


if __name__ == "__main__":
    main()
