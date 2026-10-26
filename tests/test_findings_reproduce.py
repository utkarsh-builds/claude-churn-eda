"""
Critical test: proves every numeric claim in EDA_FINDINGS.md actually reproduces
from the raw data. If this test fails, the corresponding claim in EDA_FINDINGS.md
is not verified — fix EDA_FINDINGS.md, not this test.
"""

import os
import pandas as pd
import pytest
import sqlite3
import numpy as np

from eda.load import load_customers, load_transactions, load_monthly_agg
from eda.profile import (
    count_outliers_mad,
    count_zero_activity_customers,
    profile_distribution,
)
from eda.hypotheses import (
    hypothesis_acquisition_channel,
    hypothesis_region,
    hypothesis_txn_channel,
    hypothesis_category,
    hypothesis_avg_monthly_spend,
    hypothesis_total_amount,
    hypothesis_return_count,
    hypothesis_days_since_last_txn,
)
from eda.time_checks import trend_direction, compare_row_level_vs_monthly_agg
from eda.leakage_checks import check_no_post_as_of_rows_used, flag_label_circular_drivers

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="session")
def data():
    cust = load_customers(os.path.join(DATA_DIR, "customers.csv"))
    txn = load_transactions(os.path.join(DATA_DIR, "transactions.db"))
    agg = load_monthly_agg(os.path.join(DATA_DIR, "transactions_monthly_agg.parquet"))
    return cust, txn, agg


# ---------------------------------------------------------------------------
# Customer profile claims
# ---------------------------------------------------------------------------


class TestCustomerClaims:
    def test_total_customers(self, data):
        cust, _, _ = data
        assert len(cust) == 1500

    def test_churned_count(self, data):
        cust, _, _ = data
        assert int(cust["churned"].sum()) == 202

    def test_churn_rate(self, data):
        cust, _, _ = data
        assert cust["churned"].mean() == pytest.approx(0.1347, abs=0.001)

    def test_columns(self, data):
        cust, _, _ = data
        assert set(cust.columns) == {"customer_id", "signup_date", "acquisition_channel", "region", "churned"}

    def test_channels(self, data):
        cust, _, _ = data
        ch = cust["acquisition_channel"].value_counts()
        assert ch["organic"] == 609
        assert ch["paid_search"] == 431
        assert ch["referral"] == 295
        assert ch["social"] == 165

    def test_regions(self, data):
        cust, _, _ = data
        r = cust["region"].value_counts()
        assert r["east"] == 385 and r["north"] == 381 and r["west"] == 380 and r["south"] == 354


# ---------------------------------------------------------------------------
# Zero-activity claims
# ---------------------------------------------------------------------------


class TestZeroActivityClaims:
    def test_zero_activity_count(self, data):
        cust, txn, _ = data
        count, ids = count_zero_activity_customers(cust, txn)
        assert count == 81, f"Expected 81 zero-activity, got {count}"

    def test_zero_activity_all_churned(self, data):
        cust, txn, _ = data
        _, ids = count_zero_activity_customers(cust, txn)
        df_zero = cust[cust["customer_id"].isin(ids)]
        assert df_zero["churned"].sum() == len(df_zero), "Not all zero-activity customers are churned"


# ---------------------------------------------------------------------------
# Transaction claims
# ---------------------------------------------------------------------------


class TestTransactionClaims:
    def test_total_transactions(self, data):
        _, txn, _ = data
        assert len(txn) == 11742

    def test_amount_mean(self, data):
        _, txn, _ = data
        assert txn["amount"].mean() == pytest.approx(49.72, abs=0.1)

    def test_amount_median(self, data):
        _, txn, _ = data
        assert txn["amount"].median() == pytest.approx(33.82, abs=0.1)

    def test_amount_max(self, data):
        _, txn, _ = data
        assert txn["amount"].max() == pytest.approx(1684.21, abs=0.1)

    def test_amount_skew(self, data):
        _, txn, _ = data
        assert txn["amount"].skew() > 5, "Amount should be heavily right-skewed"

    def test_amount_outliers_mad(self, data):
        _, txn, _ = data
        n = count_outliers_mad(txn["amount"], threshold=3.0)
        assert n >= 1500, f"Expected >=1500 MAD outliers, got {n}"

    def test_unique_customers(self, data):
        _, txn, _ = data
        assert txn["customer_id"].nunique() == 1419


# ---------------------------------------------------------------------------
# Monthly agg claims
# ---------------------------------------------------------------------------


class TestMonthlyAggClaims:
    def test_total_rows(self, data):
        _, _, agg = data
        assert len(agg) == 8086

    def test_txn_count_mean(self, data):
        _, _, agg = data
        assert agg["txn_count"].mean() == pytest.approx(1.42, abs=0.05)

    def test_total_amount_mean(self, data):
        _, _, agg = data
        assert agg["total_amount"].mean() == pytest.approx(70.72, abs=0.5)

    def test_return_count_mean(self, data):
        _, _, agg = data
        assert agg["return_count"].mean() == pytest.approx(0.071, abs=0.01)


# ---------------------------------------------------------------------------
# Hypothesis test claims
# ---------------------------------------------------------------------------


class TestHypothesisClaims:
    def test_acquisition_channel_not_significant(self, data):
        cust, _, _ = data
        r = hypothesis_acquisition_channel(cust)
        assert r["p_value"] > 0.05, f"acquisition_channel should NOT be significant (p={r['p_value']})"

    def test_region_not_significant(self, data):
        cust, _, _ = data
        r = hypothesis_region(cust)
        assert r["p_value"] > 0.05, f"region should NOT be significant (p={r['p_value']})"

    def test_txn_channel_significant(self, data):
        cust, txn, _ = data
        r = hypothesis_txn_channel(cust, txn)
        assert r["p_value"] < 0.05, f"txn_channel should be significant (p={r['p_value']})"

    def test_category_significant(self, data):
        cust, txn, _ = data
        r = hypothesis_category(cust, txn)
        assert r["p_value"] < 0.05, f"category should be significant (p={r['p_value']})"

    def test_avg_monthly_spend_significant(self, data):
        cust, txn, _ = data
        r = hypothesis_avg_monthly_spend(cust, txn)
        assert r["p_value"] < 0.05, f"avg_monthly_spend should be significant (p={r['p_value']})"

    def test_total_amount_significant(self, data):
        cust, txn, _ = data
        r = hypothesis_total_amount(cust, txn)
        assert r["p_value"] < 0.05, f"total_amount should be significant (p={r['p_value']})"

    def test_return_count_significant(self, data):
        cust, txn, _ = data
        r = hypothesis_return_count(cust, txn)
        assert r["p_value"] < 0.05, f"return_count should be significant (p={r['p_value']})"

    def test_days_since_last_txn_significant_leakage(self, data):
        cust, txn, _ = data
        r = hypothesis_days_since_last_txn(cust, txn)
        assert r["p_value"] < 0.001, "days_since_last_txn should be highly significant (leakage)"
        assert "LEAKAGE" in r.get("note", "")
        assert r["group_higher"] == "churned"


# ---------------------------------------------------------------------------
# Trend claims
# ---------------------------------------------------------------------------


class TestTrendClaims:
    def test_total_months(self, data):
        _, txn, _ = data
        td = trend_direction(txn)
        assert td["total_months"] == 30

    def test_growth_ratio(self, data):
        _, txn, _ = data
        td = trend_direction(txn)
        assert td["growth_ratio"] >= 7.0, f"Growth ratio should be ~8x, got {td['growth_ratio']}"


# ---------------------------------------------------------------------------
# Multi-source claims
# ---------------------------------------------------------------------------


class TestMultiSourceClaims:
    def test_reaggregated_row_count(self, data):
        _, txn, agg = data
        ms = compare_row_level_vs_monthly_agg(txn, agg)
        assert ms["total_rows_reaggregated"] >= 8086
        assert ms["total_rows_agg"] == 8086

    def test_disagreement_exists(self, data):
        _, txn, agg = data
        ms = compare_row_level_vs_monthly_agg(txn, agg)
        assert ms["abs_diff_txn_count_sum"] > 0, "Multi-source should show some disagreement"


# ---------------------------------------------------------------------------
# Leakage claims
# ---------------------------------------------------------------------------


class TestLeakageClaims:
    def test_no_post_as_of_data(self, data):
        _, txn, _ = data
        result = check_no_post_as_of_rows_used(txn)
        assert result["pass_as_of_check"] is True
        assert result["rows_after_as_of_date"] == 0

    def test_leakage_window_size(self, data):
        _, txn, _ = data
        result = check_no_post_as_of_rows_used(txn)
        assert result["rows_within_trailing_90d_window"] > 3000, "Should have 3000+ rows in leakage window"

    def test_leakage_tainted_drivers(self, data):
        ld = flag_label_circular_drivers()
        assert "days_since_last_txn" in ld["leakage_tainted_drivers"]
