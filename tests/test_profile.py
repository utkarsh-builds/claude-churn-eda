"""Tests for eda/profile.py — profiling functions on constructed inputs."""
import pandas as pd
import numpy as np
import pytest

from eda.profile import (
    profile_missingness,
    profile_distribution,
    count_outliers_mad,
    count_zero_activity_customers,
)


class TestProfileMissingness:
    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        m = profile_missingness(df)
        assert m == {"a": 0, "b": 0}

    def test_with_missing(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, "y", None]})
        m = profile_missingness(df)
        assert m["a"] == 1
        assert m["b"] == 2


class TestProfileDistribution:
    def test_basic_stats(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        d = profile_distribution(s)
        assert d["count"] == 5
        assert d["mean"] == pytest.approx(3.0)
        assert d["min"] == pytest.approx(1.0)
        assert d["max"] == pytest.approx(5.0)
        assert d["median"] == pytest.approx(3.0)
        assert d["q25"] == pytest.approx(2.0)
        assert d["q75"] == pytest.approx(4.0)

    def test_skew_detection(self):
        # Right-skewed: mean > median
        s = pd.Series([1, 1, 1, 1, 1, 10, 20, 30])
        d = profile_distribution(s)
        assert d["mean"] > d["median"]
        assert d["skew"] > 0


class TestCountOutliersMAD:
    def test_no_outliers(self):
        s = pd.Series([10.0, 11.0, 10.5, 11.5, 10.2, 11.3])
        assert count_outliers_mad(s, threshold=3.0) == 0

    def test_with_outlier(self):
        # Values clustered at ~10-12 (spread = 2), with one extreme at 100
        # Median ~10.5, deviations ~0.5-1.5 for the cluster + ~89.5 for outlier
        # MAD > 0 because the deviations have spread, so MAD rule works
        s = pd.Series([10.0, 10.5, 11.0, 11.5, 12.0] * 10 + [100.0])
        n = count_outliers_mad(s, threshold=3.0)
        assert n >= 1, f"Expected at least 1 outlier, got {n}"

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        assert count_outliers_mad(s) == 0

    def test_identical_values(self):
        s = pd.Series([5.0] * 10)
        assert count_outliers_mad(s) == 0


class TestCountZeroActivityCustomers:
    def test_basic(self):
        cust = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5]})
        txn = pd.DataFrame({"customer_id": [1, 3, 5]})
        count, ids = count_zero_activity_customers(cust, txn)
        assert count == 2
        assert ids == [2, 4]

    def test_none_zero(self):
        cust = pd.DataFrame({"customer_id": [1, 2, 3]})
        txn = pd.DataFrame({"customer_id": [1, 2, 3, 3, 1]})
        count, ids = count_zero_activity_customers(cust, txn)
        assert count == 0
        assert ids == []

    def test_all_zero(self):
        cust = pd.DataFrame({"customer_id": [1, 2]})
        txn = pd.DataFrame({"customer_id": [3, 4]})
        count, ids = count_zero_activity_customers(cust, txn)
        assert count == 2
        assert ids == [1, 2]