"""Tests for eda/time_checks.py and eda/leakage_checks.py."""
import pandas as pd
import pytest
import numpy as np

from eda.time_checks import monthly_volume_trend, trend_direction, compare_row_level_vs_monthly_agg
from eda.leakage_checks import check_no_post_as_of_rows_used, flag_label_circular_drivers, get_leakage_cutoff


class TestMonthlyVolumeTrend:
    def test_monotonic_increase(self):
        # 6 months, increasing counts
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        txn = pd.DataFrame({
            "customer_id": [1] * 100,
            "amount": [10.0] * 100,
            "txn_timestamp": dates,
        })
        mv = monthly_volume_trend(txn)
        assert len(mv) > 0
        assert list(mv.columns) == ["month", "txn_count", "unique_customers", "total_amount"]

    def test_flat_series(self):
        dates = pd.date_range("2025-01-01", periods=30, freq="D")
        txn = pd.DataFrame({
            "customer_id": [1] * 30,
            "amount": [10.0] * 30,
            "txn_timestamp": dates,
        })
        mv = monthly_volume_trend(txn)
        assert len(mv) >= 1


class TestTrendDirection:
    def test_growth_ratio(self):
        # Construct 4 months: 10, 20, 30, 40 txn
        txn = pd.DataFrame({
            "customer_id": [1] * 100,
            "amount": [10.0] * 100,
            "txn_timestamp": list(pd.date_range("2025-01-01", periods=10, freq="D")) +
                             list(pd.date_range("2025-02-01", periods=20, freq="D")) +
                             list(pd.date_range("2025-03-01", periods=30, freq="D")) +
                             list(pd.date_range("2025-04-01", periods=40, freq="D")),
        })
        td = trend_direction(txn)
        assert td["total_months"] >= 3
        assert td["growth_ratio"] is None or td["growth_ratio"] > 0


class TestCompareRowLevelVsMonthlyAgg:
    def test_matching_data(self):
        # Identical reagg and agg
        txn = pd.DataFrame({
            "customer_id": [1, 1, 2],
            "amount": [10.0, 20.0, 30.0],
            "is_returned": [0, 1, 0],
            "txn_timestamp": pd.to_datetime(["2025-01-15", "2025-01-20", "2025-02-10"]),
        })
        # Manually construct the agg version
        agg = pd.DataFrame({
            "customer_id": [1, 2],
            "txn_month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "txn_count": [2, 1],
            "total_amount": [30.0, 30.0],
            "return_count": [1, 0],
        })
        ms = compare_row_level_vs_monthly_agg(txn, agg)
        assert ms["rows_only_in_reaggregated"] == 0
        assert ms["rows_only_in_agg"] == 0

    def test_mismatched_data(self):
        txn = pd.DataFrame({
            "customer_id": [1, 1],
            "amount": [10.0, 20.0],
            "is_returned": [0, 0],
            "txn_timestamp": pd.to_datetime(["2025-01-15", "2025-01-20"]),
        })
        # Missing one row compared to DB
        agg = pd.DataFrame({
            "customer_id": [1],
            "txn_month": pd.to_datetime(["2025-01-01"]),
            "txn_count": [1],
            "total_amount": [10.0],
            "return_count": [0],
        })
        ms = compare_row_level_vs_monthly_agg(txn, agg)
        # Reaggregated has 1 row, agg has 1 — but values disagree
        assert ms["abs_diff_txn_count_sum"] >= 1


class TestCheckNoPostAsOfRowsUsed:
    def test_all_before_as_of(self):
        txn = pd.DataFrame({
            "txn_timestamp": pd.to_datetime(["2026-06-01", "2026-07-01"]),
            "customer_id": [1, 2],
            "amount": [10.0, 20.0],
        })
        result = check_no_post_as_of_rows_used(txn, as_of_date=pd.Timestamp("2026-08-01"))
        assert result["pass_as_of_check"] is True

    def test_with_post_as_of_rows(self):
        txn = pd.DataFrame({
            "txn_timestamp": pd.to_datetime(["2026-08-15", "2026-09-01"]),
            "customer_id": [1, 2],
            "amount": [10.0, 20.0],
        })
        result = check_no_post_as_of_rows_used(txn, as_of_date=pd.Timestamp("2026-08-01"))
        assert result["rows_after_as_of_date"] > 0


class TestFlagLabelCircularDrivers:
    def test_tainted_list(self):
        result = flag_label_circular_drivers()
        assert "days_since_last_txn" in result["leakage_tainted_drivers"]
        assert "total_spend_trailing_90d" in result["leakage_tainted_drivers"]
        assert result["trailing_window_days"] == 90


class TestGetLeakageCutoff:
    def test_cutoff_value(self):
        cutoff = get_leakage_cutoff()
        assert cutoff == pd.Timestamp("2026-05-03")