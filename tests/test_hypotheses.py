"""Tests for eda/hypotheses.py — constructed inputs with known answers."""
import pandas as pd
import numpy as np
import pytest

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


@pytest.fixture
def two_channel_customers():
    """10 customers: 5 organic (4 churned=80%), 5 paid_search (1 churned=20%)."""
    return pd.DataFrame({
        "customer_id": range(1, 11),
        "acquisition_channel": ["organic"] * 5 + ["paid_search"] * 5,
        "region": ["east"] * 10,
        "churned": [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],  # 5 total
    })


class TestAcquisitionChannel:
    def test_significant_difference(self, two_channel_customers):
        result = hypothesis_acquisition_channel(two_channel_customers)
        assert result["driver"] == "acquisition_channel"
        assert result["test"] == "Chi-square"
        assert result["statistic"] is not None
        assert result["p_value"] is not None

    def test_no_difference(self):
        df = pd.DataFrame({
            "customer_id": range(1, 11),
            "acquisition_channel": ["organic"] * 5 + ["paid_search"] * 5,
            "region": ["east"] * 10,
            "churned": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 50% each
        })
        result = hypothesis_acquisition_channel(df)
        # Both channels have same rate, so p should be high
        assert result["p_value"] >= 0.05


class TestRegion:
    def test_basic(self):
        df = pd.DataFrame({
            "customer_id": range(1, 11),
            "acquisition_channel": ["organic"] * 10,
            "region": ["east"] * 5 + ["west"] * 5,
            "churned": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        })
        result = hypothesis_region(df)
        assert result["driver"] == "region"
        assert result["test"] == "Chi-square"
        assert result["group_higher"] == "east"


class TestTxnChannel:
    def test_with_data(self):
        cust = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "churned": [1, 0, 0, 1],
        })
        txn = pd.DataFrame({
            "customer_id": [1, 1, 1, 2, 2, 3, 4, 4],
            "channel": ["web", "web", "web", "mobile_app", "mobile_app", "store", "mobile_app", "mobile_app"],
            "amount": [10] * 8,
            "txn_timestamp": pd.to_datetime(["2025-01-01"] * 8),
        })
        result = hypothesis_txn_channel(cust, txn)
        assert result["driver"] == "txn_channel"
        assert result["test"] == "Chi-square"


class TestCategory:
    def test_with_data(self):
        cust = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "churned": [1, 1, 0, 0],
        })
        txn = pd.DataFrame({
            "customer_id": [1, 1, 2, 3, 3, 4, 4, 4],
            "category": ["electronics", "electronics", "grocery", "grocery", "grocery", "apparel", "apparel", "apparel"],
            "amount": [10] * 8,
            "txn_timestamp": pd.to_datetime(["2025-06-01"] * 8),
        })
        result = hypothesis_category(cust, txn)
        assert result["driver"] == "category"
        assert result["test"] == "Chi-square"


class TestAvgMonthlySpend:
    def test_with_difference(self):
        cust = pd.DataFrame({
            "customer_id": [1, 2, 3, 4, 5, 6],
            "churned": [1, 1, 1, 0, 0, 0],
        })
        # Higher spend for not_churned (outside leakage window)
        txn = pd.DataFrame({
            "customer_id": [1, 2, 3, 4, 5, 6],
            "amount": [10.0, 10.0, 10.0, 200.0, 200.0, 200.0],
            "txn_timestamp": pd.to_datetime(["2025-01-01"] * 6),
        })
        result = hypothesis_avg_monthly_spend(cust, txn)
        assert result["driver"] == "avg_monthly_spend"
        assert result["test"] == "Mann-Whitney U"
        assert result["statistic"] is not None


class TestTotalAmount:
    def test_with_difference(self):
        cust = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "churned": [1, 1, 0, 0],
        })
        txn = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "amount": [50.0, 50.0, 500.0, 500.0],
            "txn_timestamp": pd.to_datetime(["2025-01-01"] * 4),
        })
        result = hypothesis_total_amount(cust, txn)
        assert result["driver"] == "total_amount"
        assert result["test"] == "Mann-Whitney U"


class TestReturnCount:
    def test_with_difference(self):
        cust = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "churned": [1, 1, 0, 0],
        })
        txn = pd.DataFrame({
            "customer_id": [1, 1, 2, 2, 3, 4],
            "is_returned": [1, 0, 1, 0, 0, 0],
            "amount": [10] * 6,
        })
        result = hypothesis_return_count(cust, txn)
        assert result["driver"] == "return_count"
        assert result["test"] == "Mann-Whitney U"


class TestDaysSinceLastTxn:
    def test_leakage_check(self):
        cust = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "churned": [1, 1, 0, 0],
        })
        # Churned: last txn 200 days ago; not churned: last txn 10 days ago
        as_of = pd.Timestamp("2026-08-01")
        txn = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "amount": [10] * 4,
            "txn_timestamp": pd.to_datetime([
                "2026-01-01", "2026-01-15", "2026-07-20", "2026-07-25"
            ]),
        })
        result = hypothesis_days_since_last_txn(cust, txn, as_of_date=as_of)
        assert result["driver"] == "days_since_last_txn"
        assert result["test"] == "Mann-Whitney U"
        assert "LEAKAGE" in result.get("note", "")
        # Churned should have higher days_since
        assert result["group_higher"] == "churned"