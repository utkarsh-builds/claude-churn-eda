"""Run time checks and leakage checks, print and save results."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eda.load import load_transactions, load_monthly_agg
from eda.time_checks import trend_direction, compare_row_level_vs_monthly_agg
from eda.leakage_checks import check_no_post_as_of_rows_used, flag_label_circular_drivers


def run_time_and_leakage_checks():
    """Run all time and leakage checks."""
    print("Loading data...")
    txn = load_transactions()
    agg = load_monthly_agg()

    print("\n=== TIME CHECKS ===")
    td = trend_direction(txn)
    print(f"Total months: {td['total_months']}")
    print(f"Date range: {td['date_range'][0].date()} to {td['date_range'][1].date()}")
    print(f"First-half avg txn/month: {td['first_half_avg_txn_per_month']}")
    print(f"Second-half avg txn/month: {td['second_half_avg_txn_per_month']}")
    print(f"Growth ratio: {td['growth_ratio']}x")

    print("\n=== MULTI-SOURCE AGREEMENT ===")
    ms = compare_row_level_vs_monthly_agg(txn, agg)
    print(f"Rows in reaggregated: {ms['total_rows_reaggregated']}")
    print(f"Rows in agg file:      {ms['total_rows_agg']}")
    print(f"Rows in both:          {ms['rows_in_both']}")
    print(f"Rows only in reagg:    {ms['rows_only_in_reaggregated']}")
    print(f"Rows only in agg:      {ms['rows_only_in_agg']}")
    print(f"Abs diff txn_count:     {ms['abs_diff_txn_count_sum']}")
    print(f"Abs diff total_amount:  {ms['abs_diff_total_amount_sum']}")
    print(f"Abs diff return_count:  {ms['abs_diff_return_count_sum']}")

    print("\n=== MULTI-SOURCE MONTHLY COMPARISON (first 5, last 5 months) ===")
    comp = ms["monthly_comparison"]
    print(comp.head(5).to_string(index=False))
    print("...")
    print(comp.tail(5).to_string(index=False))

    print("\n=== LEAKAGE CHECKS ===")
    lc = check_no_post_as_of_rows_used(txn)
    print(f"As-of date: {lc['as_of_date'].date()}")
    print(f"Leakage cutoff: {lc['leakage_cutoff'].date()}")
    print(f"Max txn timestamp: {lc['max_txn_timestamp_in_data']}")
    print(f"Rows after as-of date: {lc['rows_after_as_of_date']}")
    print(f"Rows within trailing 90d window: {lc['rows_within_trailing_90d_window']}")
    print(f"Pass as-of check: {lc['pass_as_of_check']}")

    ld = flag_label_circular_drivers()
    print(f"\nLeakage-tainted drivers: {ld['leakage_tainted_drivers']}")
    print(f"Reasoning: {ld['reasoning'][:200]}...")

    out = {
        "trend": td,
        "multi_source": {k: v for k, v in ms.items() if k != "monthly_comparison"},
        "leakage": lc,
        "leakage_drivers": ld,
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "time_and_leakage_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_time_and_leakage_checks()