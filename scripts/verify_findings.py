"""
Verify every numeric claim in EDA_FINDINGS.md from raw data.
Exits with code 0 if all claims reproduce, 1 otherwise.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eda.load import load_customers, load_transactions, load_monthly_agg
from eda.profile import (
    profile_distribution,
    count_outliers_mad,
    count_zero_activity_customers,
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
from eda.leakage_checks import check_no_post_as_of_rows_used


def verify(label, computed, expected, tol=0.0):
    """Verify computed matches expected within tolerance. Returns True/False."""
    if isinstance(expected, bool):
        ok = computed == expected
    elif tol > 0:
        ok = abs(computed - expected) <= tol * max(1.0, abs(expected))
    else:
        ok = computed == expected
    status = "PASS" if ok else "FAIL"
    if not ok:
        print(f"  [{status}] {label}: computed={computed}, expected={expected}")
    else:
        print(f"  [{status}] {label}: {computed}")
    return ok


def main():
    print("Loading raw data...")
    cust = load_customers()
    txn = load_transactions()
    agg = load_monthly_agg()

    all_ok = True

    # === 1. Customer counts ===
    print("\n=== 1. Customer Counts ===")
    all_ok &= verify("Total customers", len(cust), 1500)
    n_churned = int(cust["churned"].sum())
    all_ok &= verify("Churned count", n_churned, 202)
    churn_rate = round(float(cust["churned"].mean()), 4)
    all_ok &= verify("Churn rate", churn_rate, 0.1347, tol=0.001)

    # === 2. Zero-activity customers ===
    print("\n=== 2. Zero-Activity Customers ===")
    n_zero, zero_ids = count_zero_activity_customers(cust, txn)
    all_ok &= verify("Zero-activity count", n_zero, 81)
    df_zero = cust[cust["customer_id"].isin(zero_ids)]
    zero_churned = int(df_zero["churned"].sum())
    all_ok &= verify("Zero-activity all churned", zero_churned, 81)

    # === 3. Transaction amount distribution ===
    print("\n=== 3. Transaction Amount Distribution ===")
    d = profile_distribution(txn["amount"])
    all_ok &= verify("Amount mean", d["mean"], 49.7202, tol=0.01)
    all_ok &= verify("Amount median", d["median"], 33.82, tol=0.01)
    all_ok &= verify("Amount max", d["max"], 1684.21, tol=0.01)
    all_ok &= verify("Amount skew > 5", d["skew"] > 5, True)

    # === 4. Outliers (MAD-based) ===
    print("\n=== 4. Outliers ===")
    n_outliers = count_outliers_mad(txn["amount"], threshold=3.0)
    all_ok &= verify("Amount MAD outliers (thresh=3)", n_outliers, 1612, tol=0.02)

    # === 5. Parquet distributions ===
    print("\n=== 5. Parquet Distributions ===")
    d_txn = profile_distribution(agg["txn_count"])
    all_ok &= verify("txn_count mean", d_txn["mean"], 1.4228, tol=0.005)
    d_amt = profile_distribution(agg["total_amount"])
    all_ok &= verify("total_amount mean", d_amt["mean"], 70.7206, tol=0.01)
    d_ret = profile_distribution(agg["return_count"])
    all_ok &= verify("return_count mean", d_ret["mean"], 0.0710, tol=0.01)

    # === 6. Hypothesis tests ===
    print("\n=== 6. Hypothesis Tests ===")
    results = {
        "acquisition_channel": hypothesis_acquisition_channel(cust),
        "region": hypothesis_region(cust),
        "txn_channel": hypothesis_txn_channel(cust, txn),
        "category": hypothesis_category(cust, txn),
        "avg_monthly_spend": hypothesis_avg_monthly_spend(cust, txn),
        "total_amount": hypothesis_total_amount(cust, txn),
        "return_count": hypothesis_return_count(cust, txn),
        "days_since_last_txn": hypothesis_days_since_last_txn(cust, txn),
    }

    # acquisition_channel ns (p > 0.05)
    all_ok &= verify("acquisition_channel p > 0.05", results["acquisition_channel"]["p_value"] > 0.05, True)
    # region ns (p > 0.05)
    all_ok &= verify("region p > 0.05", results["region"]["p_value"] > 0.05, True)
    # txn_channel sig (p < 0.001)
    all_ok &= verify("txn_channel p < 0.001", results["txn_channel"]["p_value"] < 0.001, True)
    # category sig (p < 0.001)
    all_ok &= verify("category p < 0.001", results["category"]["p_value"] < 0.001, True)
    # avg_monthly_spend sig
    all_ok &= verify("avg_monthly_spend p < 0.001", results["avg_monthly_spend"]["p_value"] < 0.001, True)
    # total_amount sig
    all_ok &= verify("total_amount p < 0.001", results["total_amount"]["p_value"] < 0.001, True)
    # return_count sig (p < 0.01)
    all_ok &= verify("return_count p < 0.01", results["return_count"]["p_value"] < 0.01, True)
    # days_since_last_txn sig (leakage check)
    all_ok &= verify("days_since_last_txn p < 0.001", results["days_since_last_txn"]["p_value"] < 0.001, True)

    # === 7. Monthly volume trend ===
    print("\n=== 7. Monthly Volume Trend ===")
    td = trend_direction(txn)
    all_ok &= verify("Total months", td["total_months"], 30)
    all_ok &= verify("Growth ratio", td["growth_ratio"], 8.11, tol=0.02)

    # === 8. Multi-source agreement ===
    print("\n=== 8. Multi-Source Agreement ===")
    ms = compare_row_level_vs_monthly_agg(txn, agg)
    all_ok &= verify("Rows only in reaggregated", ms["rows_only_in_reaggregated"], 102)
    all_ok &= verify("Rows in both", ms["rows_in_both"], 8086)

    # === 9. Leakage checks ===
    print("\n=== 9. Leakage Checks ===")
    lc = check_no_post_as_of_rows_used(txn)
    all_ok &= verify("No rows after as-of date", lc["pass_as_of_check"], True)

    # === Summary ===
    print("\n" + "=" * 50)
    if all_ok:
        print("ALL CLAIMS VERIFIED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("SOME CLAIMS FAILED VERIFICATION")
        sys.exit(1)


if __name__ == "__main__":
    main()
