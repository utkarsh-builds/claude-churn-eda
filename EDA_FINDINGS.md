# EDA Findings — Verified Report

> Every numeric claim in this document is reproducible from the raw data sources via `scripts/verify_findings.py`.

---

## Data Sources Overview

| Source | Rows | Columns | Missing values |
|---|---|---|---|
| `customers.csv` | 1,500 | 5 | 0 |
| `transactions.db` (SQLite) | 11,742 | 6 | 0 |
| `transactions_monthly_agg.parquet` | 8,086 | 5 | 0 |

All three sources have **zero missing values** across all columns. No sentinel-value defects were found.

---

## 1. Schema & Dtypes

| Source | Columns & Types |
|---|---|
| **customers.csv** | `customer_id` (int64), `signup_date` (datetime64), `acquisition_channel` (object), `region` (object), `churned` (int64) |
| **transactions.db** | `customer_id` (int64), `txn_timestamp` (datetime64), `category` (object), `channel` (object), `amount` (float64), `is_returned` (int64) |
| **monthly_agg.parquet** | `customer_id` (int64), `txn_month` (datetime64[us]), `txn_count` (int64), `total_amount` (float64), `return_count` (int64) |

---

## 2. Missingness

Zero missing values in all three sources confirmed. All string categorical columns contain populated values (no empty strings or "unknown" sentinels).

---

## 3. Zero-Activity Customers

**81 customers** have zero rows in `transactions.db`.

- **All 81 are churned (churned = 1).** This represents 40.1% of all churned customers (81 of 202).
- These customers are treated as a **separate behavioral cohort**, not imputed. Their zero-activity status is a definitional property, not a predictor — every zero-activity customer in this dataset is trivially churned because they never had the opportunity to be otherwise.
- **Recommendation**: This cohort must be modeled separately or excluded from any model that relies on transaction-based features, because the zero-activity condition perfectly predicts churn for this subset but has zero predictive value for the 1,419 customers who have at least one transaction.

---

## 4. Distributions — Numeric Columns

### Transaction Amount (transactions.db)

| Metric | Value |
|---|---|
| Count | 11,742 |
| Mean | $49.72 |
| Std | $55.24 |
| Min | $3.00 |
| Q25 | $18.36 |
| Median | $33.82 |
| Q75 | $61.39 |
| Max | $1,684.21 |
| Skewness | 6.02 |

**Severe right skew (log-normal shape).** The mean ($49.72) is well above the median ($33.82), and the maximum ($1,684.21) is over 27× the 75th percentile. These are natural high-value purchases, not data errors.

**MAD-based outlier count (threshold = 3): 1,612 outliers (13.7%).** Standard IQR rules incorrectly flag ~6-7% of this positively-skewed data as errors; a MAD-based rule is the correct approach here. These represent legitimate high-value transactions, not anomalies.

### Transaction Count (per customer-month, parquet)

| Metric | Value |
|---|---|
| Count | 8,086 |
| Mean | 1.42 |
| Median | 1.0 |
| Max | 6 |
| Skew | 1.47 |

Right-skewed. 75% of customer-months have ≤2 transactions. Most customers transact infrequently.

### Total Amount (per customer-month, parquet)

| Metric | Value |
|---|---|
| Count | 8,086 |
| Mean | $70.72 |
| Median | $48.32 |
| Max | $1,753.23 |
| Skew | 4.09 |

Consistent with the per-transaction skew.

### Return Count (per customer-month, parquet)

| Metric | Value |
|---|---|
| Count | 8,086 |
| Mean | 0.07 |
| Median | 0 |
| Max | 3 |

Heavily zero-inflated: ~93% of customer-months have zero returns.

---

## 5. Categorical Distributions

### Acquisition Channel (customers.csv)

| Channel | Count | % |
|---|---|---|
| organic | 609 | 40.6% |
| paid_search | 431 | 28.7% |
| referral | 295 | 19.7% |
| social | 165 | 11.0% |

### Region (customers.csv)

| Region | Count | % |
|---|---|---|
| east | 385 | 25.7% |
| north | 381 | 25.4% |
| west | 380 | 25.3% |
| south | 354 | 23.6% |

Regions are nearly balanced.

### Transaction Category (transactions.db)

Balanced across six categories (1,900–2,000 each): home, grocery, sports, beauty, apparel, electronics.

### Transaction Channel (transactions.db)

| Channel | Count | % |
|---|---|---|
| web | 5,278 | 45.0% |
| mobile_app | 4,146 | 35.3% |
| store | 2,318 | 19.7% |

---

## 6. Hypothesis Test Results

| Driver | Test | Statistic | p-value | Significant? | Effect |
|---|---|---|---|---|---|
| acquisition_channel | Chi-square | 5.84 | 0.120 | **No** | organic 15.3%, paid_search 10.2%, referral 13.9%, social 14.5% |
| region | Chi-square | 3.76 | 0.289 | **No** | east 16.4%, south 12.7%, west 12.4%, north 12.3% |
| txn_channel | Chi-square | 551.40 | < 0.001 | **Yes** | no-txn customers 100% churn; store 10.9%, mobile 9.1%, web 7.8% |
| category | Chi-square | 558.16 | < 0.001 | **Yes** | electronics 12.7%, apparel 9.6%, beauty 8.9%, sports 7.9%, home 7.6%, grocery 4.0%, no-txn 100% |
| avg_monthly_spend | Mann-Whitney U | 101,979.0 | < 0.001 | **Yes** | Churned mean: $40.23; Not churned mean: $57.19 (diff: $16.96) |
| total_amount | Mann-Whitney U | 98,830.5 | < 0.001 | **Yes** | Churned mean: $188.73; Not churned mean: $286.91 (diff: $98.18) |
| return_count | Mann-Whitney U | 108,409.5 | < 0.001 | **Yes** | Churned mean: 0.20; Not churned mean: 0.42 (diff: 0.23) |
| days_since_last_txn | Mann-Whitney U | 262,196.0 | < 0.001 | **Yes** | Median churned: 168d; Not churned: 23d (diff: 145d) |

### Key findings from hypothesis tests:

1. **acquisition_channel and region are NOT statistically significant drivers of churn** (p > 0.05). Despite observable rate differences in the raw data, they do not independently predict churn after accounting for sample variation.

2. **txn_channel and category are significant**, but the effect is driven primarily by the zero-activity (no-txn) cohort having 100% churn. Among customers with transactions, channel and category differences are modest.

3. **Churned customers have lower avg_monthly_spend and lower total_amount** than non-churned customers. This is intuitive: customers who leave the platform naturally generate less revenue.

4. **return_count** is significant but counterintuitive — churned customers have *fewer* returns on average. The returns feature is unlikely to be a strong actionable predictor.

5. **days_since_last_txn** shows a massive effect (median 168d churned vs 23d not churned) because it is definitionally circular with the churn label (see Leakage section).

---

## 7. Monthly Transaction-Volume Trend

| Metric | Value |
|---|---|
| Observation period | Feb 2024 – Jul 2026 (30 months) |
| First-half avg txn/month | 85.9 |
| Second-half avg txn/month | 696.9 |
| Growth ratio | **8.11×** |

The monthly transaction volume grew approximately 8× over the 30-month observation window. Active customer counts show a parallel growth trajectory (not driven solely by per-customer increases). This severe non-stationarity means a **time-based train/test split is mandatory** for any modeling effort.

---

## 8. Multi-Source Agreement

| Metric | Value |
|---|---|
| Rows in reaggregated (from DB) | 8,188 |
| Rows in parquet | 8,086 |
| Rows in both | 8,086 |
| Rows only in reaggregated | 102 |
| Total abs diff in txn_count | 127 |
| Total abs diff in total_amount | $6,805.25 |
| Total abs diff in return_count | 5 |

The two sources **agree perfectly for months Feb 2024 through Jun 2026**. The disagreement is concentrated entirely in **July 2026**:
- Raw DB: 1,231 transactions
- Parquet: 994 transactions

This suggests the parquet file was generated from an earlier snapshot that did not include the full final month's data. For modeling, the row-level `transactions.db` should be used as the source of truth, and the parquet should be used only after verifying it covers the intended date range.