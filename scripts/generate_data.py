"""Generate synthetic FMCG data for demo and testing."""
import argparse
import logging

from fmcg_forecast.data.calendar import create_feature_calendar
from fmcg_forecast.data.generator import generate_orders, generate_products, generate_sales_data
from fmcg_forecast.utils.logging import setup_logger
from fmcg_forecast.utils.paths import ensure_dirs, get_data_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FMCG data")
    parser.add_argument("--skus", type=int, default=80)
    parser.add_argument("--warehouses", type=int, default=5)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logger("fmcg_forecast", level=logging.INFO)
    logger = logging.getLogger(__name__)

    dirs = get_data_dirs()
    ensure_dirs(dirs)

    logger.info("Generating %d products...", args.skus)
    products = generate_products(num_skus=args.skus, seed=args.seed)
    products.to_csv(dirs["synthetic"] / "products.csv", index=False)

    logger.info("Generating order history...")
    orders = generate_orders(
        products, args.warehouses, args.start_date, args.end_date, args.seed
    )
    orders.to_csv(dirs["synthetic"] / "orders.csv", index=False)

    logger.info("Generating sales/COGS data...")
    sales = generate_sales_data(orders, products, seed=args.seed)
    sales.to_csv(dirs["synthetic"] / "sales.csv", index=False)

    logger.info("Generating FMCG calendar...")
    calendar = create_feature_calendar(args.start_date, args.end_date)
    calendar.to_csv(dirs["synthetic"] / "calendar.csv", index=False)

    logger.info("Data generation complete. Files saved to %s", dirs["synthetic"])


if __name__ == "__main__":
    main()
