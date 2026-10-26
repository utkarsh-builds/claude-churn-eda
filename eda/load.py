"""Loaders for the three raw data sources. Pure I/O, no analysis logic."""
import sqlite3
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_customers(path=None):
    """Load customers.csv. Returns DataFrame."""
    if path is None:
        path = os.path.join(DATA_DIR, "customers.csv")
    df = pd.read_csv(path)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


def load_transactions(path=None):
    """Load transactions.db (SQLite, table 'transactions'). Returns DataFrame."""
    if path is None:
        path = os.path.join(DATA_DIR, "transactions.db")
    conn = sqlite3.connect(path)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
    return df


def load_monthly_agg(path=None):
    """Load transactions_monthly_agg.parquet. Returns DataFrame."""
    if path is None:
        path = os.path.join(DATA_DIR, "transactions_monthly_agg.parquet")
    return pd.read_parquet(path)