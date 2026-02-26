import pandas as pd
import pytest


@pytest.mark.unit
def test_generate_products_returns_correct_count():
    from fmcg_forecast.data.generator import generate_products

    products = generate_products(num_skus=10, seed=42)
    assert len(products) == 10
    assert "product_name" in products.columns
    assert "category" in products.columns
    assert "unit_price" in products.columns


@pytest.mark.unit
def test_generate_orders_returns_daily_records():
    from fmcg_forecast.data.generator import generate_orders, generate_products

    products = generate_products(num_skus=5, seed=42)
    orders = generate_orders(
        products,
        num_warehouses=2,
        start_date="2024-01-01",
        end_date="2024-01-31",
        seed=42,
    )
    assert isinstance(orders, pd.DataFrame)
    assert "date" in orders.columns
    assert "product_name" in orders.columns
    assert "id_gudang" in orders.columns
    assert "main_product_sales" in orders.columns
    assert len(orders) > 0


@pytest.mark.unit
def test_generate_orders_has_seasonal_patterns():
    from fmcg_forecast.data.generator import generate_orders, generate_products

    products = generate_products(num_skus=5, seed=42)
    orders = generate_orders(
        products,
        num_warehouses=2,
        start_date="2024-01-01",
        end_date="2024-12-31",
        seed=42,
    )
    # Payday periods (25th-5th) should have higher average sales
    orders["day"] = pd.to_datetime(orders["date"]).dt.day
    payday = orders[(orders["day"] >= 25) | (orders["day"] <= 5)]
    non_payday = orders[(orders["day"] > 5) & (orders["day"] < 25)]
    assert payday["main_product_sales"].mean() > non_payday["main_product_sales"].mean()


@pytest.mark.unit
def test_generate_sales_data_returns_revenue_and_cogs():
    from fmcg_forecast.data.generator import (
        generate_orders,
        generate_products,
        generate_sales_data,
    )

    products = generate_products(num_skus=5, seed=42)
    orders = generate_orders(
        products,
        num_warehouses=2,
        start_date="2024-01-01",
        end_date="2024-03-31",
        seed=42,
    )
    sales = generate_sales_data(orders, products, seed=42)
    assert "sales" in sales.columns
    assert "cogs" in sales.columns
    # COGS should be less than or equal to sales (margin >= 0)
    assert (sales["cogs"] <= sales["sales"]).all()
