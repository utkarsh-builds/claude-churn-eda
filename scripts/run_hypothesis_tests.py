"""Run every hypothesis test in eda/hypotheses.py and save a results table."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eda.load import load_customers, load_transactions
from eda.hypotheses import (
    hypothesis_acquisition_channel, hypothesis_region, hypothesis_txn_channel, hypothesis_category,
    hypothesis_avg_monthly_spend, hypothesis_total_amount, hypothesis_return_count,
    hypothesis_days_since_last_txn,
)


def run_hypothesis_tests():
    """Run all hypothesis tests and print/save results."""
    print("Loading data...")
    cust = load_customers()
    txn = load_transactions()

    tests = [
        hypothesis_acquisition_channel(cust),
        hypothesis_region(cust),
        hypothesis_txn_channel(cust, txn),
        hypothesis_category(cust, txn),
        hypothesis_avg_monthly_spend(cust, txn),
        hypothesis_total_amount(cust, txn),
        hypothesis_return_count(cust, txn),
        hypothesis_days_since_last_txn(cust, txn),
    ]

    # Clean output for readability
    results = []
    for t in tests:
        row = {
            "driver": t.get("driver"),
            "test": t.get("test"),
            "statistic": t.get("statistic"),
            "p_value": t.get("p_value"),
            "group_higher": t.get("group_higher"),
            "effect_size": t.get("effect_size"),
            "error": t.get("error"),
            "note": t.get("note"),
        }
        results.append(row)

    print("\n=== HYPOTHESIS TEST RESULTS ===\n")
    for r in results:
        sig = "SIGNIFICANT" if r["p_value"] is not None and r["p_value"] < 0.05 else "NOT SIGNIFICANT"
        p_str = f"{r['p_value']:.6f}" if r["p_value"] is not None else "N/A"
        stat_str = f"{r['statistic']:.4f}" if r["statistic"] is not None else "N/A"
        note_str = f" [{r['note']}]" if r.get("note") else ""
        print(f"{r['driver']:30s} | {r['test']:20s} | stat={stat_str:12s} | p={p_str:12s} | {sig:20s}{note_str}")
        print(f"  Effect: {r['effect_size']}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "hypothesis_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_hypothesis_tests()