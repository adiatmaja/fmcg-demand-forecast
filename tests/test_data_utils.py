"""Tests for data loading utilities and sales fetch."""
import pandas as pd
import pytest


@pytest.mark.unit
def test_load_csv_reads_file(tmp_path):
    from fmcg_forecast.data.loader import load_csv

    csv = tmp_path / "test.csv"
    csv.write_text("date,value\n2024-01-01,100\n2024-01-02,200\n")
    df = load_csv(csv, parse_dates=["date"])
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


@pytest.mark.unit
def test_load_csv_no_parse_dates(tmp_path):
    from fmcg_forecast.data.loader import load_csv

    csv = tmp_path / "test.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    df = load_csv(csv)
    assert list(df.columns) == ["a", "b"]


@pytest.mark.unit
def test_load_sales_data(tmp_path):
    from fmcg_forecast.sales.fetch import load_sales_data

    csv = tmp_path / "sales.csv"
    csv.write_text("date,id_region,id_gudang,sales,cogs\n2024-01-01,R-1,WH-A,1000,600\n")
    df = load_sales_data(csv)
    assert len(df) == 1
    assert "sales" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
