"""Shared fixtures for fmcg-demand-forecast tests."""

import pytest


@pytest.fixture
def sample_products():
    """Small product master for testing."""
    from fmcg_forecast.data.generator import generate_products

    return generate_products(num_skus=5, seed=42)


@pytest.fixture
def sample_orders(sample_products):
    """Small order dataset for testing."""
    from fmcg_forecast.data.generator import generate_orders

    return generate_orders(
        sample_products,
        num_warehouses=2,
        start_date="2024-01-01",
        end_date="2024-03-31",
        seed=42,
    )


@pytest.fixture
def sample_sales(sample_orders, sample_products):
    """Small sales dataset for testing."""
    from fmcg_forecast.data.generator import generate_sales_data

    return generate_sales_data(sample_orders, sample_products, seed=42)


@pytest.fixture
def sample_calendar():
    """Small calendar for testing."""
    from fmcg_forecast.data.calendar import create_feature_calendar

    return create_feature_calendar("2024-01-01", "2024-12-31")


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output
