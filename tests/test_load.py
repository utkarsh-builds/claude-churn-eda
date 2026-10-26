"""Tests for eda/load.py — each loader against the real data files."""
import os
import pandas as pd
import pytest

from eda.load import load_customers, load_transactions, load_monthly_agg

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_load_customers():
    df = load_customers(os.path.join(DATA_DIR, "customers.csv"))
    assert df.shape == (1500, 5), f"Expected (1500, 5), got {df.shape}"
    expected_cols = {"customer_id", "signup_date", "acquisition_channel", "region", "churned"}
    assert set(df.columns) == expected_cols, f"Columns mismatch: {set(df.columns)}"
    assert df["customer_id"].dtype == "int64"
    assert df["churned"].dtype == "int64"
    assert df["signup_date"].dtype == "datetime64[us]" or df["signup_date"].dtype == "datetime64[ns]"
    assert df["customer_id"].iloc[0] == 1
    assert df["customer_id"].iloc[-1] == 1500


def test_load_transactions():
    df = load_transactions(os.path.join(DATA_DIR, "transactions.db"))
    assert df.shape == (11742, 6), f"Expected (11742, 6), got {df.shape}"
    expected_cols = {"customer_id", "txn_timestamp", "category", "channel", "amount", "is_returned"}
    assert set(df.columns) == expected_cols, f"Columns mismatch: {set(df.columns)}"
    assert df["customer_id"].dtype == "int64"
    assert df["amount"].dtype == "float64"
    assert df["is_returned"].dtype == "int64"
    assert df["txn_timestamp"].dtype == "datetime64[ns]"


def test_load_monthly_agg():
    df = load_monthly_agg(os.path.join(DATA_DIR, "transactions_monthly_agg.parquet"))
    assert df.shape == (8086, 5), f"Expected (8086, 5), got {df.shape}"
    expected_cols = {"customer_id", "txn_month", "txn_count", "total_amount", "return_count"}
    assert set(df.columns) == expected_cols, f"Columns mismatch: {set(df.columns)}"
    assert df["customer_id"].dtype == "int64"
    assert df["txn_count"].dtype == "int64"
    assert df["total_amount"].dtype == "float64"
    assert df["txn_month"].dtype == "datetime64[us]"