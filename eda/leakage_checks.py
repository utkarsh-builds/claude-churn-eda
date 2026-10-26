"""Leakage checks: as-of-date discipline and label-circularity detection."""
import pandas as pd


AS_OF_DATE = pd.Timestamp("2026-08-01")
TRAILING_WINDOW_DAYS = 90
LEAKAGE_CUTOFF = AS_OF_DATE - pd.DateOffset(days=TRAILING_WINDOW_DAYS)


def check_no_post_as_of_rows_used(transactions_df, as_of_date=None):
    """
    Confirm that no transaction row with txn_timestamp after the as-of date
    was used in any computation. Returns a dict with pass/fail and details.
    """
    if as_of_date is None:
        as_of_date = AS_OF_DATE
    else:
        as_of_date = pd.Timestamp(as_of_date)

    max_ts = transactions_df["txn_timestamp"].max()
    n_post = (transactions_df["txn_timestamp"] > as_of_date).sum()
    # Also check: any data after the leakage cutoff was used in training features
    n_within_window = (transactions_df["txn_timestamp"] > LEAKAGE_CUTOFF).sum()

    return {
        "as_of_date": as_of_date,
        "leakage_cutoff": LEAKAGE_CUTOFF,
        "max_txn_timestamp_in_data": max_ts,
        "rows_after_as_of_date": int(n_post),
        "rows_within_trailing_90d_window": int(n_within_window),
        "pass_as_of_check": bool(n_post == 0),
        "note": f"All valid features must use data before {LEAKAGE_CUTOFF.date()}. "
        f"{n_within_window} rows ({n_within_window/len(transactions_df)*100:.1f}%) fall in the leakage window.",
    }


def flag_label_circular_drivers():
    """
    Return the set of candidate drivers that are definitionally circular
    with a 90-day-window churn label.
    """
    leakage_tainted = [
        "days_since_last_txn",
        "total_spend_trailing_90d",
        "txn_count_trailing_90d",
        "recency",
    ]
    return {
        "leakage_tainted_drivers": leakage_tainted,
        "reasoning": (
            f"The churn label is defined as 'no purchase in the {TRAILING_WINDOW_DAYS} days "
            f"before the as-of date ({AS_OF_DATE.date()}). Any driver computed from transactions "
            f"within this window is definitionally circular — it is restating the label, not predicting it. "
            f"These drivers must be strictly excluded from modeling."
        ),
        "as_of_date": AS_OF_DATE,
        "trailing_window_days": TRAILING_WINDOW_DAYS,
        "leakage_cutoff": LEAKAGE_CUTOFF,
    }


def get_leakage_cutoff():
    """Return the timestamp before which data is safe for feature computation."""
    return LEAKAGE_CUTOFF