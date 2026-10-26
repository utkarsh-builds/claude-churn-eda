"""Profiling functions: schema, missingness, distributions, outliers."""
import numpy as np
import pandas as pd


def profile_schema(df, name=""):
    """Return a dict of column name -> dtype for the given DataFrame."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def profile_missingness(df):
    """Return a dict of column name -> count of missing (null/NaN) values."""
    return df.isnull().sum().to_dict()


def profile_distribution(series):
    """Return summary stats: count, mean, std, min, 25%, 50%, 75%, max, skew."""
    s = series.dropna()
    return {
        "count": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4),
        "q25": round(float(s.quantile(0.25)), 4),
        "median": round(float(s.median()), 4),
        "q75": round(float(s.quantile(0.75)), 4),
        "max": round(float(s.max()), 4),
        "skew": round(float(s.skew()), 4),
    }


def count_outliers_mad(series, threshold=3.0):
    """
    Count outliers using a MAD-based rule.
    MAD = median(|x_i - median(x)|).
    A value is an outlier if |x_i - median| / MAD > threshold.
    Falls back to IQR if MAD is zero (all identical values).
    """
    s = series.dropna()
    if len(s) == 0:
        return 0
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad == 0:
        # All values identical or MAD collapsed
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return int(((s < lower) | (s > upper)).sum())
    deviations = np.abs(s - med) / mad
    return int((deviations > threshold).sum())


def count_zero_activity_customers(customers_df, transactions_df):
    """
    Count customers in customers_df who have zero rows in transactions_df.
    Returns (count, list_of_ids).
    """
    cust_ids = set(customers_df["customer_id"])
    txn_ids = set(transactions_df["customer_id"])
    zero_ids = sorted(cust_ids - txn_ids)
    return len(zero_ids), zero_ids