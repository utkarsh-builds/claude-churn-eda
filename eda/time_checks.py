"""Time-based checks: monthly volume trend and multi-source agreement."""
import pandas as pd
import numpy as np


def monthly_volume_trend(transactions_df):
    """
    Compute monthly transaction counts and active-customer counts from row-level data.
    Returns DataFrame with columns: month, txn_count, unique_customers, total_amount.
    """
    df = transactions_df.copy()
    df["month"] = df["txn_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month").agg(
        txn_count=("amount", "count"),
        unique_customers=("customer_id", "nunique"),
        total_amount=("amount", "sum"),
    ).reset_index().sort_values("month")
    monthly["month"] = pd.to_datetime(monthly["month"])
    return monthly


def trend_direction(transactions_df):
    """
    Characterize the monthly volume trend.
    Returns dict with first-half avg, second-half avg, and growth ratio.
    """
    mv = monthly_volume_trend(transactions_df)
    n = len(mv)
    first_half = mv.iloc[: n // 2]
    second_half = mv.iloc[n // 2 :]
    first_avg = first_half["txn_count"].mean()
    second_avg = second_half["txn_count"].mean()
    return {
        "first_half_avg_txn_per_month": round(first_avg, 1),
        "second_half_avg_txn_per_month": round(second_avg, 1),
        "growth_ratio": round(second_avg / first_avg, 2) if first_avg > 0 else None,
        "total_months": n,
        "date_range": (mv["month"].min(), mv["month"].max()),
    }


def compare_row_level_vs_monthly_agg(transactions_df, agg_df):
    """
    Reaggregate the row-level transactions.db to customer-month and compare
    against transactions_monthly_agg.parquet month by month.
    Returns dict with disagreement stats and per-month comparison DataFrame.
    """
    # Reaggregate row-level
    df = transactions_df.copy()
    df["txn_month"] = df["txn_timestamp"].dt.to_period("M").dt.to_timestamp()
    reaggregated = df.groupby(["customer_id", "txn_month"]).agg(
        txn_count=("amount", "count"),
        total_amount=("amount", "sum"),
        return_count=("is_returned", "sum"),
    ).reset_index()

    # Prepare agg_df
    agg = agg_df.copy()
    agg["txn_month"] = pd.to_datetime(agg["txn_month"])

    # Merge
    reaggregated["txn_month"] = pd.to_datetime(reaggregated["txn_month"])
    merged = reaggregated.merge(
        agg,
        on=["customer_id", "txn_month"],
        how="outer",
        suffixes=("_reagg", "_agg"),
        indicator=True,
    )

    only_in_reagg = merged[merged["_merge"] == "left_only"]
    only_in_agg = merged[merged["_merge"] == "right_only"]

    # Numeric differences where both sides exist
    both = merged[merged["_merge"] == "both"]
    txn_count_diff = (both["txn_count_reagg"] - both["txn_count_agg"]).abs().sum()
    total_amount_diff = (both["total_amount_reagg"] - both["total_amount_agg"]).abs().sum()
    return_count_diff = (both["return_count_reagg"] - both["return_count_agg"]).abs().sum()

    # Per-month comparison
    monthly_reagg = reaggregated.groupby("txn_month").agg(
        txn_count=("txn_count", "sum"),
        total_amount=("total_amount", "sum"),
    ).reset_index().sort_values("txn_month")

    monthly_agg = agg.groupby("txn_month").agg(
        txn_count=("txn_count", "sum"),
        total_amount=("total_amount", "sum"),
    ).reset_index().sort_values("txn_month")

    comparison = monthly_reagg.merge(
        monthly_agg, on="txn_month", suffixes=("_reagg", "_agg"), how="outer"
    ).fillna(0)

    return {
        "total_rows_reaggregated": len(reaggregated),
        "total_rows_agg": len(agg),
        "rows_only_in_reaggregated": len(only_in_reagg),
        "rows_only_in_agg": len(only_in_agg),
        "rows_in_both": len(both),
        "abs_diff_txn_count_sum": int(txn_count_diff),
        "abs_diff_total_amount_sum": round(total_amount_diff, 2),
        "abs_diff_return_count_sum": int(return_count_diff),
        "monthly_comparison": comparison,
    }