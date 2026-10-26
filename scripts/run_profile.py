"""Run profiling across all three sources, print and save a profile summary."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eda.load import load_customers, load_transactions, load_monthly_agg
from eda.profile import (
    profile_schema, profile_missingness, profile_distribution,
    count_outliers_mad, count_zero_activity_customers,
)


def run_profile():
    """Run full profile and save results."""
    print("Loading data...")
    cust = load_customers()
    txn = load_transactions()
    agg = load_monthly_agg()

    report = {}

    # Customers
    n_zero, zero_ids = count_zero_activity_customers(cust, txn)
    df_zero = cust[cust["customer_id"].isin(zero_ids)]
    report["customers"] = {
        "shape": list(cust.shape),
        "columns": list(cust.columns),
        "schema": profile_schema(cust),
        "missingness": profile_missingness(cust),
        "churn_rate": round(float(cust["churned"].mean()), 4),
        "churn_counts": cust["churned"].value_counts().to_dict(),
        "acquisition_channels": cust["acquisition_channel"].value_counts().to_dict(),
        "regions": cust["region"].value_counts().to_dict(),
    }

    # Transactions
    report["transactions"] = {
        "shape": list(txn.shape),
        "columns": list(txn.columns),
        "schema": profile_schema(txn),
        "missingness": profile_missingness(txn),
        "amount_distribution": profile_distribution(txn["amount"]),
        "amount_outliers_mad": count_outliers_mad(txn["amount"], threshold=3.0),
        "date_range": [str(txn["txn_timestamp"].min()), str(txn["txn_timestamp"].max())],
        "category_counts": txn["category"].value_counts().to_dict(),
        "channel_counts": txn["channel"].value_counts().to_dict(),
    }

    # Monthly Agg
    report["monthly_agg"] = {
        "shape": list(agg.shape),
        "columns": list(agg.columns),
        "schema": profile_schema(agg),
        "missingness": profile_missingness(agg),
        "txn_count_distribution": profile_distribution(agg["txn_count"]),
        "total_amount_distribution": profile_distribution(agg["total_amount"]),
        "return_count_distribution": profile_distribution(agg["return_count"]),
    }

    # Zero-activity
    report["zero_activity"] = {
        "count": n_zero,
        "churned_count": int(df_zero["churned"].sum()),
        "by_channel": df_zero["acquisition_channel"].value_counts().to_dict(),
        "by_region": df_zero["region"].value_counts().to_dict(),
    }

    print(json.dumps(report, indent=2, default=str))

    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "profile_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nProfile saved to {out_path}")


if __name__ == "__main__":
    run_profile()