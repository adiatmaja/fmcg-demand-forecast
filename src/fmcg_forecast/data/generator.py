"""Synthetic FMCG data generator.

Generates realistic sales data for Indonesian FMCG retail with
seasonal patterns, stock-out events, and promotional effects.
"""

import logging

import numpy as np
import pandas as pd

from fmcg_forecast.data.calendar import FMCG_HOLIDAYS


logger = logging.getLogger(__name__)

BRAND_POOL: dict[str, list[str]] = {
    "Beverages": ["AquaClear", "TeaMax", "JuiceFresh", "SodaPop"],
    "Snacks": ["CrunchBite", "NuttyBar", "ChipKing", "WaferLite"],
    "Personal Care": ["CleanWash", "FreshGlow", "SilkCare", "PureWhite"],
    "Household": ["SparkleClean", "PowerWash", "FreshAir", "ShineMore"],
    "Dairy": ["MilkPure", "YoguFresh", "CheesyBite", "CreamDelight"],
}


def generate_products(
    num_skus: int = 80,
    categories: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a product master table.

    Args:
        num_skus: Number of SKUs to generate.
        categories: Product category names.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with product_name, category, brand, unit_price, is_promo.
    """
    rng = np.random.default_rng(seed)
    if categories is None:
        categories = list(BRAND_POOL.keys())

    products = []
    for i in range(num_skus):
        cat = categories[i % len(categories)]
        brands = BRAND_POOL.get(cat, ["GenericBrand"])
        brand = brands[i % len(brands)]
        variant = f"{i // len(categories) + 1:02d}"
        products.append(
            {
                "product_id": f"SKU-{i + 1:04d}",
                "product_name": f"{brand} {cat[:3].upper()}-{variant}",
                "category": cat,
                "brand": brand,
                "unit_price": int(rng.uniform(5_000, 50_000)),
                "is_promo": int(rng.random() < 0.15),
            }
        )

    logger.info("Generated %d synthetic products", num_skus)
    return pd.DataFrame(products)


def generate_orders(
    products: pd.DataFrame,
    num_warehouses: int = 5,
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily order records with realistic FMCG patterns.

    Args:
        products: Product master DataFrame.
        num_warehouses: Number of warehouses.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        seed: Random seed.

    Returns:
        DataFrame with daily order records per product per warehouse.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date)
    warehouses = [f"WH-{chr(65 + i)}" for i in range(num_warehouses)]

    # Pre-build lebaran lookup for fast access in inner loop
    lebaran_lookup = [pd.Timestamp(dt) for dt in FMCG_HOLIDAYS["idul_fitri"].values()]

    records = []
    for _, product in products.iterrows():
        base_demand = rng.uniform(5, 80)
        for wh in warehouses:
            wh_factor = rng.uniform(0.6, 1.4)

            for dt in dates:
                demand = base_demand * wh_factor

                # Weekend dip (Saturday half, Sunday zero)
                if dt.weekday() == 5:
                    demand *= 0.5
                elif dt.weekday() == 6:
                    continue  # No orders on Sunday

                # Payday spike (25th-5th)
                if dt.day >= 25 or dt.day <= 5:
                    demand *= rng.uniform(1.2, 1.6)

                # Rainy season boost (Oct-Apr)
                if dt.month in [10, 11, 12, 1, 2, 3, 4]:
                    demand *= 1.08

                # Ramadan/Lebaran surge
                for lebaran_dt in lebaran_lookup:
                    days_to = (lebaran_dt - dt).days
                    if 0 < days_to <= 30:
                        demand *= 1.0 + (0.5 * (1 - days_to / 30))
                    elif -7 <= days_to <= 0:
                        demand *= 0.3  # Post-Lebaran dip

                # Random stock-out (2% chance per product-warehouse-day)
                if rng.random() < 0.02:
                    demand = 0

                # Occasional outlier (0.5% chance)
                if rng.random() < 0.005:
                    demand *= rng.uniform(3, 6)

                # Add noise
                demand = max(0, demand + rng.normal(0, base_demand * 0.15))

                records.append(
                    {
                        "date": dt.strftime("%Y-%m-%d"),
                        "product_name": product["product_name"],
                        "product_id": product["product_id"],
                        "category": product["category"],
                        "id_gudang": wh,
                        "main_product_sales": round(demand),
                        "promo_sales": round(demand * 0.1)
                        if product["is_promo"]
                        else 0,
                        "prioritas_sales": round(demand * rng.uniform(0, 0.05)),
                        "is_payday_period": int(dt.day >= 25 or dt.day <= 5),
                    }
                )

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d order records (%d products x %d warehouses)",
        len(df),
        len(products),
        num_warehouses,
    )
    return df


def generate_sales_data(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate revenue and COGS data from order records.

    Args:
        orders: Order records DataFrame.
        products: Product master DataFrame.
        seed: Random seed.

    Returns:
        DataFrame with date, id_gudang, id_region, sales, cogs columns.
    """
    rng = np.random.default_rng(seed)
    merged = orders.merge(products[["product_name", "unit_price"]], on="product_name")
    merged["sales"] = merged["main_product_sales"] * merged["unit_price"]
    merged["cogs"] = merged["sales"] * rng.uniform(0.55, 0.75, size=len(merged))
    merged["id_region"] = merged["id_gudang"].map(
        {f"WH-{chr(65 + i)}": f"R-{i // 2 + 1}" for i in range(26)}
    )

    daily = (
        merged.groupby(["date", "id_region", "id_gudang"])
        .agg({"sales": "sum", "cogs": "sum"})
        .reset_index()
    )
    logger.info("Generated %d daily sales records", len(daily))
    return daily
