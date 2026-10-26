"""Hypothesis tests for churn-driver candidates. Each function is pure and returns a dict."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def _safe_mannwhitneyu(x, y, alternative="two-sided"):
    """Run Mann-Whitney U with safe handling of degenerate cases."""
    if len(x) == 0 or len(y) == 0:
        return {"statistic": None, "p_value": None, "error": "empty group"}
    if len(set(x)) == 1 and len(set(y)) == 1 and x.iloc[0] == y.iloc[0]:
        return {"statistic": None, "p_value": 1.0, "note": "identical distributions"}
    result = mannwhitneyu(x, y, alternative=alternative)
    return {"statistic": round(float(result.statistic), 4), "p_value": round(float(result.pvalue), 6)}


def _safe_chi2(df, cat_col, target_col="churned"):
    """Run chi-square test of independence between cat_col and target_col."""
    ct = pd.crosstab(df[cat_col], df[target_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"statistic": None, "p_value": None, "error": "contingency table too small"}
    result = chi2_contingency(ct)
    return {"statistic": round(float(result.statistic), 4), "p_value": round(float(result.pvalue), 6)}


def hypothesis_acquisition_channel(customers_df):
    """Chi-square: acquisition_channel vs churned."""
    cs = _safe_chi2(customers_df, "acquisition_channel")
    rates = customers_df.groupby("acquisition_channel")["churned"].mean()
    cs.update({
        "driver": "acquisition_channel",
        "test": "Chi-square",
        "group_higher": rates.idxmax(),
        "effect_size": f"Churn rates: {rates.sort_values().to_dict()}",
    })
    return cs


def hypothesis_region(customers_df):
    """Chi-square: region vs churned."""
    cs = _safe_chi2(customers_df, "region")
    rates = customers_df.groupby("region")["churned"].mean()
    cs.update({
        "driver": "region",
        "test": "Chi-square",
        "group_higher": rates.idxmax(),
        "effect_size": f"Churn rates: {rates.sort_values().to_dict()}",
    })
    return cs


def hypothesis_txn_channel(customers_df, transactions_df):
    """Chi-square: preferred transaction channel vs churned."""
    # Most-used channel per customer
    top_channel = (
        transactions_df.groupby(["customer_id", "channel"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .groupby("customer_id")
        .first()
        .reset_index()[["customer_id", "channel"]]
    )
    merged = customers_df[["customer_id", "churned"]].merge(top_channel, on="customer_id", how="left")
    merged["channel"] = merged["channel"].fillna("none")
    cs = _safe_chi2(merged, "channel")
    rates = merged.groupby("channel")["churned"].mean()
    cs.update({
        "driver": "txn_channel",
        "test": "Chi-square",
        "group_higher": rates.idxmax(),
        "effect_size": f"Churn by preferred channel: {rates.sort_values().to_dict()}",
    })
    return cs


def hypothesis_category(customers_df, transactions_df):
    """Chi-square: most-purchased category vs churned."""
    top_cat = (
        transactions_df.groupby(["customer_id", "category"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .groupby("customer_id")
        .first()
        .reset_index()[["customer_id", "category"]]
    )
    merged = customers_df[["customer_id", "churned"]].merge(top_cat, on="customer_id", how="left")
    merged["category"] = merged["category"].fillna("none")
    cs = _safe_chi2(merged, "category")
    rates = merged.groupby("category")["churned"].mean()
    cs.update({
        "driver": "category",
        "test": "Chi-square",
        "group_higher": rates.idxmax(),
        "effect_size": f"Churn by top category: {rates.sort_values().to_dict()}",
    })
    return cs


def hypothesis_avg_monthly_spend(customers_df, transactions_df):
    """Mann-Whitney U: average monthly spend vs churned."""
    as_of = pd.Timestamp("2026-08-01")
    cutoff = as_of - pd.DateOffset(days=90)
    txn = transactions_df[transactions_df["txn_timestamp"] < cutoff].copy()
    txn["month"] = txn["txn_timestamp"].dt.to_period("M")
    monthly = txn.groupby(["customer_id", "month"])["amount"].sum().reset_index()
    avg_spend = monthly.groupby("customer_id")["amount"].mean().reset_index()
    avg_spend.columns = ["customer_id", "avg_monthly_spend"]

    merged = customers_df[["customer_id", "churned"]].merge(avg_spend, on="customer_id", how="left")
    merged["avg_monthly_spend"] = merged["avg_monthly_spend"].fillna(0)

    churned = merged[merged["churned"] == 1]["avg_monthly_spend"]
    not_churned = merged[merged["churned"] == 0]["avg_monthly_spend"]
    mw = _safe_mannwhitneyu(churned, not_churned, alternative="two-sided")
    mean_c, mean_nc = churned.mean(), not_churned.mean()
    mw.update({
        "driver": "avg_monthly_spend",
        "test": "Mann-Whitney U",
        "group_higher": "churned" if mean_c > mean_nc else "not_churned",
        "effect_size": f"Mean churned=${mean_c:.2f}, not_churned=${mean_nc:.2f} (diff=${abs(mean_c - mean_nc):.2f})",
    })
    return mw


def hypothesis_total_amount(customers_df, transactions_df):
    """Mann-Whitney U: total lifetime amount (pre-leakage-window) vs churned."""
    as_of = pd.Timestamp("2026-08-01")
    cutoff = as_of - pd.DateOffset(days=90)
    txn = transactions_df[transactions_df["txn_timestamp"] < cutoff].copy()
    totals = txn.groupby("customer_id")["amount"].sum().reset_index()
    totals.columns = ["customer_id", "total_amount"]

    merged = customers_df[["customer_id", "churned"]].merge(totals, on="customer_id", how="left")
    merged["total_amount"] = merged["total_amount"].fillna(0)

    churned = merged[merged["churned"] == 1]["total_amount"]
    not_churned = merged[merged["churned"] == 0]["total_amount"]
    mw = _safe_mannwhitneyu(churned, not_churned, alternative="two-sided")
    mean_c, mean_nc = churned.mean(), not_churned.mean()
    mw.update({
        "driver": "total_amount",
        "test": "Mann-Whitney U",
        "group_higher": "churned" if mean_c > mean_nc else "not_churned",
        "effect_size": f"Mean churned=${mean_c:.2f}, not_churned=${mean_nc:.2f} (diff=${abs(mean_c - mean_nc):.2f})",
    })
    return mw


def hypothesis_return_count(customers_df, transactions_df):
    """Mann-Whitney U: return count per customer vs churned."""
    returns = transactions_df.groupby("customer_id")["is_returned"].sum().reset_index()
    returns.columns = ["customer_id", "return_count"]
    merged = customers_df[["customer_id", "churned"]].merge(returns, on="customer_id", how="left")
    merged["return_count"] = merged["return_count"].fillna(0)

    churned = merged[merged["churned"] == 1]["return_count"]
    not_churned = merged[merged["churned"] == 0]["return_count"]
    mw = _safe_mannwhitneyu(churned, not_churned, alternative="two-sided")
    mean_c, mean_nc = churned.mean(), not_churned.mean()
    mw.update({
        "driver": "return_count",
        "test": "Mann-Whitney U",
        "group_higher": "churned" if mean_c > mean_nc else "not_churned",
        "effect_size": f"Mean churned={mean_c:.4f}, not_churned={mean_nc:.4f} (diff={abs(mean_c - mean_nc):.4f})",
    })
    return mw


def hypothesis_days_since_last_txn(customers_df, transactions_df, as_of_date=None):
    """Mann-Whitney U: days since last transaction vs churned. LEAKAGE CHECK."""
    if as_of_date is None:
        as_of_date = pd.Timestamp("2026-08-01")
    else:
        as_of_date = pd.Timestamp(as_of_date)

    last_txn = transactions_df.groupby("customer_id")["txn_timestamp"].max().reset_index()
    last_txn["days_since"] = (as_of_date - last_txn["txn_timestamp"]).dt.days
    merged = customers_df[["customer_id", "churned"]].merge(last_txn, on="customer_id", how="left")
    merged["days_since"] = merged["days_since"].fillna(9999)  # zero-activity → large recency

    churned = merged[merged["churned"] == 1]["days_since"]
    not_churned = merged[merged["churned"] == 0]["days_since"]
    mw = _safe_mannwhitneyu(churned, not_churned, alternative="two-sided")
    median_c, median_nc = churned.median(), not_churned.median()
    mw.update({
        "driver": "days_since_last_txn",
        "test": "Mann-Whitney U",
        "note": "LEAKAGE CHECK — this variable is circular with the label definition",
        "group_higher": "churned" if median_c > median_nc else "not_churned",
        "effect_size": f"Median churned={median_c:.0f}d, not_churned={median_nc:.0f}d (diff={abs(median_c - median_nc):.0f}d)",
    })
    return mw