"""Run the sales and COGS forecasting pipeline (individual + merged + recommendations)."""
import argparse
import logging

import pandas as pd

from fmcg_forecast.config import SalesConfig, load_yaml_config
from fmcg_forecast.data.calendar import ALL_PUBLIC_HOLIDAYS
from fmcg_forecast.sales.fetch import load_sales_data
from fmcg_forecast.sales.individual import SeasonalFinancialForecaster, preprocess_raw_data
from fmcg_forecast.sales.merged import FinancialForecaster
from fmcg_forecast.sales.recommendations import calculate_recommendations
from fmcg_forecast.utils.logging import setup_logger
from fmcg_forecast.utils.paths import ensure_dirs, get_data_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FMCG sales forecasting pipeline")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--data", default=None, help="Path to input CSV (default: data/synthetic/sales.csv)"
    )
    parser.add_argument(
        "--mode",
        choices=["individual", "merged", "both"],
        default="both",
        help="Forecasting mode",
    )
    args = parser.parse_args()

    setup_logger("fmcg_forecast", level=logging.INFO)
    logger = logging.getLogger(__name__)

    dirs = get_data_dirs()
    ensure_dirs(dirs)

    yaml_cfg = load_yaml_config(args.config)
    sales_yaml = yaml_cfg.get("sales", {})
    cfg = SalesConfig(**sales_yaml)

    data_path = args.data or str(dirs["synthetic"] / "sales.csv")
    logger.info("Loading sales data from %s", data_path)
    raw_df = load_sales_data(data_path)

    # Build holiday set from calendar module
    holiday_dates = {d for d in ALL_PUBLIC_HOLIDAYS}
    holidays_df = pd.DataFrame({"date": sorted(ALL_PUBLIC_HOLIDAYS)})

    if args.mode in ("individual", "both"):
        logger.info("Running individual (per region-gudang) forecasting...")
        data_dict = preprocess_raw_data(raw_df, holiday_dates=holiday_dates)
        ind_forecaster = SeasonalFinancialForecaster(cfg, holiday_dates=holiday_dates)
        ind_results = ind_forecaster.run_forecasting(data_dict)

        # Merge individual forecasts and save
        sales_parts = [v["forecast_df"] for v in ind_results.values() if v["metric"] == "sales"]
        cogs_parts = [v["forecast_df"] for v in ind_results.values() if v["metric"] == "cogs"]
        if sales_parts and cogs_parts:
            ind_forecast = pd.merge(
                pd.concat(sales_parts),
                pd.concat(cogs_parts),
                on=["date", "id_region", "id_gudang"],
                how="outer",
            )
            out = dirs["sales_forecast"] / "individual_forecast.csv"
            ind_forecast.to_csv(out, index=False)
            logger.info("Individual forecast saved to %s", out)

            # Recommendations
            rec = calculate_recommendations(ind_forecast, holidays_df, mode="individual")
            rec_out = dirs["sales_rec_buy"] / "individual_rec_buy.csv"
            rec.to_csv(rec_out, index=False)
            logger.info("Individual recommendations saved to %s", rec_out)

    if args.mode in ("merged", "both"):
        logger.info("Running merged (aggregated) forecasting...")
        # Aggregate
        daily = (
            raw_df.groupby("date")
            .agg({"sales": "sum", "cogs": "sum"})
            .reset_index()
        )
        merged_forecaster = FinancialForecaster(cfg, holiday_dates=holiday_dates)
        merged_results = merged_forecaster.run_forecasting(daily, daily)

        if "sales" in merged_results and "cogs" in merged_results:
            merged_forecast = pd.merge(
                merged_results["sales"]["forecast_df"],
                merged_results["cogs"]["forecast_df"],
                on="date",
            )
            out = dirs["sales_forecast"] / "merged_forecast.csv"
            merged_forecast.to_csv(out, index=False)
            logger.info("Merged forecast saved to %s", out)

            # Summary
            summary = merged_forecaster.generate_summary(merged_results)
            summary.to_csv(dirs["sales_forecast"] / "merged_forecast_summary.csv", index=False)

            # Recommendations
            rec = calculate_recommendations(merged_forecast, holidays_df, mode="merged")
            rec_out = dirs["sales_rec_buy"] / "merged_rec_buy.csv"
            rec.to_csv(rec_out, index=False)
            logger.info("Merged recommendations saved to %s", rec_out)


if __name__ == "__main__":
    main()
