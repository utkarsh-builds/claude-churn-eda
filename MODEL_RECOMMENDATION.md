# Model-Family Recommendation

> **Verified recommendation**, backed by the statistical hypothesis tests and checks in `EDA_FINDINGS.md`. Leakage-tainted variables are explicitly excluded from the reasoning.

---

## Primary Recommendation: Gradient-Boosted Trees (XGBoost / LightGBM)

### Justification from the checked data characteristics

1. **Severely skewed numeric features**: Transaction amount has a skew of 6.02 (log-normal shape) with a max over 27× the 75th percentile. Tree-based models are invariant to monotonic transformations — no log-transforms or scaling needed.

2. **Mixed categorical variables**: `acquisition_channel` (4 levels), `region` (4), `txn_channel` (3), `category` (6). Tree models handle these natively without one-hot encoding across 17+ dummy variables (which would be problematic for a dataset of 1,500 customers).

3. **Interaction effects**: Cross-tab analysis shows that channel × region combinations have varying churn rates (e.g., organic-east at 22.9% vs. paid_search-south at 8.8%). Tree ensembles automatically capture these interactions.

4. **Class imbalance (13.5% churn)**: Gradient boosting implementations (XGBoost `scale_pos_weight`, LightGBM `class_weight`) handle imbalance naturally without resampling.

5. **Non-significant main effects with potential interactions**: `acquisition_channel` (p=0.12) and `region` (p=0.29) are not independently significant, but their interactions may still be informative — trees can discover these without manual specification.

### Alternative Considered: Regularized Logistic Regression

**Rejected as top pick because:**
- Would require extensive manual feature engineering (log-transforms for amount skew, explicit interaction terms for channel × region)
- Would need careful handling of the severe right-skew (log or Box-Cox transforms)
- The dataset has only 202 churned examples for 4+ categorical variables — sparse dummy encoding could be problematic
- No built-in mechanism for modeling the non-linear patterns the data almost certainly contains

**Use case**: As a baseline or if regulatory interpretability is required. Regularized (L1 or L2) logistic regression with log-transformed amount features would be the appropriate implementation.

### Not Recommended: Deep Neural Networks

- Overfit risk is high with only 1,500 labeled examples (202 churned)
- Requires extensive hyperparameter tuning for marginal gain over gradient boosting
- Less auditable and harder to explain to business stakeholders

---

## Variables: Usable vs. Excluded

### Usable variables (properly computed outside the leakage window)

| Variable | Source | Type |
|---|---|---|
| acquisition_channel | customers.csv | Categorical |
| region | customers.csv | Categorical |
| txn_channel (preferred) | transactions.db | Categorical |
| category (preferred) | transactions.db | Categorical |
| avg_monthly_spend | transactions.db | Continuous (pre-leakage-window) |
| total_amount | transactions.db | Continuous (pre-leakage-window) |
| return_count | transactions.db | Count |
| avg_transaction_value | transactions.db | Continuous |
| tenure (days since signup) | customers.csv | Continuous |
| monthly_txn_frequency | transactions.db | Continuous |

### Excluded variables (definitionally leakage-tainted)

| Variable | Exclusion Reason |
|---|---|
| **days_since_last_txn** | Circular with label: median 168d (churned) vs 23d (not churned) — a 145d gap that matches the 90d trailing-window definition. This variable is restating the label, not predicting it. |
| **total_spend_trailing_90d** | Computed from the same 90-day window as the label definition. |
| **txn_count_trailing_90d** | Same overlap issue. |
| **recency (any form)** | Every recency metric overlapping the trailing 90 days is definitionally circular. |

### Special case: zero-activity cohort

The 81 zero-activity customers (40.1% of all churned) are a **separate behavioral cohort** that should be excluded from any model trained on transaction-based features. Modeling them as a distinct segment (is_zero_activity = 1) is acceptable, but including them in a model that also uses transaction features creates a trivial separator with no real predictive value.

---

## Evaluation Strategy

| Concern | Mitigation |
|---|---|
| **Class imbalance (13.5%)** | Use **Precision, Recall, F1** as primary metrics. Raw accuracy (~86.5% by always predicting "not churned") is misleading. |
| **Non-stationarity (8× growth)** | Strict **time-based train/validation/test split** (first 80% of time for training, last 20% for evaluation). Random splits leak future patterns. |
| **Leakage** | All trailing-90-day features explicitly excluded. Zero-activity cohort flagged and separated. |
| **Small churn class (202 customers)** | Use cross-validation with stratified folds. Consider bootstrapped confidence intervals for performance metrics. |