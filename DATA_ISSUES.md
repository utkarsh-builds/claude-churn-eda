# Data Issues — Verified Flags

> Every claim below is backed by a specific numeric check run against the raw data sources.

---

## 1. Class Imbalance

| Metric | Value |
|---|---|
| Total customers | 1,500 |
| Churned | 202 |
| Not churned | 1,298 |
| **Churn rate** | **13.47%** |
| Imbalance ratio | 6.4:1 (not churned : churned) |

**Verified by**: Direct count from `customers.csv` — `churned` column sum = 202.

**Assessment**: Moderate imbalance. A naive "always predict not churned" baseline achieves 86.5% accuracy but has zero recall on the target class. Raw accuracy is a misleading metric.

**Recommendation**: Evaluate using Precision, Recall, and F1 Score. Use class weights in model training. Stratified train/test split to preserve class proportions.

---

## 2. Non-Stationarity (Time Drift)

| Metric | Value |
|---|---|
| Observation window | Feb 2024 – Jul 2026 |
| Total months | 30 |
| First-half avg monthly transactions | 85.9 |
| Second-half avg monthly transactions | 696.9 |
| **Growth ratio** | **8.11×** |

**Verified by**: Monthly aggregation of `transactions.db` timestamps, grouped by month, splitting the 30-month window into two 15-month halves and comparing means.

**Assessment**: Severe non-stationarity. The transaction volume grew ~8× over 2.5 years, with active customer counts growing in parallel. A model trained on a random split of this data would look unrealistically good because future high-volume patterns would leak into the training set.

**Recommendation**: A strict **time-based split** is mandatory (e.g., first ~80% of months for training, remaining ~20% for validation and test). Monitor churn rate stability across time segments.

---

## 3. Leakage — Label-Circular Variables

| Leakage-tainted Driver | Reason | Effect Size (documenting circularity) |
|---|---|---|
| **days_since_last_txn** | Trailing-90d window overlap | Median churned: **168d** vs not churned: **23d** (p < 0.001, stat = 262,196) |
| **total_spend_trailing_90d** | Computed from same window as label | (Strictly excluded without testing — definitionally circular) |
| **txn_count_trailing_90d** | Same window overlap | (Strictly excluded without testing — definitionally circular) |

**Verified by**: `check_no_post_as_of_rows_used()` confirms as-of date discipline (0 rows after 2026-08-01). `test_days_since_last_txn()` documents the massive 145-day median gap between churned and not-churned customers — this is the label definition itself, not a prediction about it.

**As-of date**: 2026-08-01. Leakage cutoff: **2026-05-03** (90 days prior). Any feature derived from transactions after this date is tainted.

**Note on 2026-05-03 vs 2026-05-01**: The 90-day trailing window from 2026-08-01 runs back through May 2026 (31 days in July + 31 in June + 30 in May = 92 days), making May 3 the cutoff date. Using a clean month boundary (May 1) is also defensible and slightly more conservative.

**Impact**: 3,416 rows (29.1% of all transactions) fall within the 90-day leakage window. These rows must be strictly excluded from any feature computation for modeling.

---

## 4. Leakage — Zero-Activity Cohort

| Metric | Value |
|---|---|
| Zero-activity customers | 81 (5.4% of all customers) |
| Churned among zero-activity | **81 of 81 (100%)** |
| Zero-activity as % of all churned | 40.1% |

**Verified by**: Set difference between `customers.customer_id` and distinct `transactions.customer_id`.

**Assessment**: This is a **cohort separation issue**, not a trailing-window overlap. The 81 zero-activity customers have zero lifetime transactions, making their churn a definitional guarantee rather than a prediction. Including them in any model trained on transaction-based features creates a trivial separator.

**Recommendation**: Model the zero-activity cohort separately or exclude from transaction-based models. Do not impute transaction features for these customers.

---

## 5. Multi-Source Data Disagreement

| Metric | Value |
|---|---|
| Rows in reaggregated (from `transactions.db`) | 8,188 |
| Rows in `transactions_monthly_agg.parquet` | 8,086 |
| Rows in both | 8,086 |
| **Rows only in reaggregated** | **102** |
| Total abs difference, txn_count | 127 |
| Total abs difference, total_amount | $6,805.25 |
| Total abs difference, return_count | 5 |

**Verified by**: `compare_row_level_vs_monthly_agg()` — full outer join on (customer_id, txn_month).

**Details**: The two sources agree perfectly for all 30 months except **July 2026**, where:
- Raw `transactions.db`: 1,231 transactions
- `transactions_monthly_agg.parquet`: 994 transactions (237 fewer)

**Cause**: The parquet was generated from a snapshot that did not include the complete final month's data.

**Recommendation**: Use `transactions.db` (row-level) as the authoritative source for feature engineering. The parquet is suitable only for time ranges fully captured in it (Feb 2024 – Jun 2026). Any pipeline using the parquet must validate that it covers the intended date range.

---

## 6. Hypothesis Test Results That Inform Modeling

| Driver | Significant? | p-value | Implication for Modeling |
|---|---|---|---|
| acquisition_channel | **No** | 0.120 | Not independently predictive, but may interact with other features |
| region | **No** | 0.289 | Not independently predictive, but may interact |
| txn_channel | **Yes** | < 0.001 | Driven by zero-activity cohort; modest signal among active customers |
| category | **Yes** | < 0.001 | Electronics and grocery show distinct churn rates |
| avg_monthly_spend | **Yes** | < 0.001 | Churned customers spend less ($40 vs $57) |
| total_amount | **Yes** | < 0.001 | Churned customers have lower lifetime totals ($189 vs $287) |
| return_count | **Yes** | 0.001 | Small effect; churned customers return less often (0.20 vs 0.42) |

**Note**: `acquisition_channel` and `region` not reaching significance is itself a finding — Stage 0's initial pass over-interpreted the raw rate differences (e.g., organic 15.3% vs paid_search 10.2%) that a formal test shows are within the range of sampling variation.

---

## Summary of Issues

| Issue | Severity | Evidence | Action |
|---|---|---|---|
| Class imbalance (13.5%) | Moderate | 202/1500 churned | F1 metrics, class weights, stratified split |
| Non-stationarity (8× growth) | High | 85.9 → 696.9 monthly avg txn | Time-based train/test split |
| Leakage — trailing-90d variables | High | 29.1% of txn in leakage window; days_since p<0.001, 145d gap | Strictly exclude from modeling |
| Leakage — zero-activity cohort | High | 81/81 churned, 40% of all churned | Model separately or exclude |
| Multi-source disagreement | Medium | 102 extra rows in reagg vs agg (Jul 2026 only) | Use DB as source of truth |
| acquisition_channel n.s. | Low | p=0.12, not significant | Keep in model but don't over-interpret |
| region n.s. | Low | p=0.29, not significant | Same |