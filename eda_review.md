# eda_review.md — Churn EDA & Model-Family Recommendation

<<LEARNER: fill in the SKELETON section below yourself, against skill.md's Domain Context
section and Stage 0's generated first-pass notebook/draft findings — correct that draft,
don't restate it. Do this before opening Claude Code for the build stage. See the EXAMPLE
section below (and Helper Guide.docx) for what a filled-in answer looks like — the example
uses a return-rate-by-channel hypothesis, which is NOT necessarily one of this dataset's real
findings once you check it, so it can't be copied in as your own answer. Delete the EXAMPLE
section and every bracketed prompt in the SKELETON before you submit.>>

---

## SKELETON 

## Distribution / missingness / outlier corrections
- **Correction**: Stage 0's draft applied standard IQR outlier detection to transaction amounts, flagging 6.7% of the data as outliers. I verified that the transaction amounts are heavily right-skewed (log-normal). Standard IQR is the wrong rule here; these are natural high-value purchases, not data errors. The correct approach for the build is a MAD-based rule (with threshold 3) or a percentile-based cap at the 99th percentile.
- **Correction**: Stage 0 flagged the 81 zero-activity customers as a "powerful signal" for churn. I verified this is a definitional artifact. However, this is a cohort separation issue, not a trailing-window overlap. Because they have zero lifetime transactions, they trivially guarantee churn = 1, creating perfect prediction in the training set that can never apply to active customers. This cohort must be excluded or modeled entirely separately, not imputed as missing data.

## Hypothesis tests to run

### 1. Categorical Driver
- **Driver**: acquisition_channel
- **Test**: Chi-square
- **Why this test**: Both the driver (channel) and the outcome (churned) are categorical variables, requiring a test of independence.

### 2. Categorical Driver
- **Driver**: region
- **Test**: Chi-square
- **Why this test**: Similar to acquisition channel, region is a categorical variable being tested against a binary outcome.

### 3. Categorical Driver
- **Driver**: txn_channel
- **Test**: Chi-square
- **Why this test**: The platform used (web, mobile_app, store) is a categorical variable that may show distinct churn behaviors.

### 4. Categorical Driver
- **Driver**: category
- **Test**: Chi-square
- **Why this test**: Product purchase category (home, grocery, sports, etc.) is a categorical variable; checking its association with binary churn requires a test of independence.

### 5. Continuous Driver
- **Driver**: avg_monthly_spend
- **Test**: Mann-Whitney U
- **Why this test**: Transaction amounts are strictly positive and heavily right-skewed. This violates the normality assumption required for a standard t-test, making a non-parametric rank-sum test appropriate.

### 6. Continuous Driver
- **Driver**: total_amount (from monthly aggregate)
- **Test**: Mann-Whitney U
- **Why this test**: Total aggregated monthly spend is continuous and right-skewed, requiring a non-parametric test rather than a standard t-test.

### 7. Behavioral Driver
- **Driver**: return_count
- **Test**: Mann-Whitney U
- **Why this test**: Return counts are heavily zero-inflated and right-skewed. A non-parametric test is required to see if return frequency correlates with churn.

### 8. Leakage Check
- **Driver**: days_since_last_txn (Leakage Check)
- **Test**: Mann-Whitney U
- **Why this test**: Recency is continuous and skewed. We are explicitly testing this to document its artificially massive effect size, proving it is circular with the label before formally excluding it.

## Leakage / imbalance / non-stationarity / multi-source flags
- **Leakage**: "Days since last transaction", "total spend in trailing 90 days", and "transaction count in trailing 90 days" are strictly excluded. The as-of date is 2026-08-01; any feature computed from transactions after 2026-05-03 (90 days prior) is tainted. Any variable overlapping this 90-day window is definitionally circular with the label.
- **Imbalance**: The baseline churn rate is roughly 13.5%. This moderate imbalance means raw accuracy is a misleading evaluation metric; precision, recall, and F1 score must be used to properly evaluate model performance on the minority class.
- **Non-stationarity**: Monthly transaction volume grew roughly 8x over the observation window. A random train/test split will leak future high-volume patterns into the training set, meaning a strict time-based split is required for robust validation.
- **Multi-source**: `transactions.db` must be explicitly re-aggregated to the customer-month grain and compared against `transactions_monthly_agg.parquet` month by month to detect any edge disagreements or dropped records, rather than assuming they perfectly align.

## Model-family judgment
- **Candidate family**: Gradient Boosted Trees (e.g., XGBoost / LightGBM)
- **Why, tied to this dataset's checked characteristics**: The data contains heavily skewed numeric values (which trees handle without log-transformations), mixed categorical features, and requires class weights/scaling to handle the 13.5% class imbalance. Tree ensembles natively capture interaction effects (like channel × region) highlighted in the data.
- **Alternative considered**: Regularized Logistic Regression — **Why not the top pick**: It would require extensive manual feature engineering, such as log-transforms for the severe amount skew and explicit interaction terms.
- **Drivers assumed usable / excluded**: Categorical drivers (`acquisition_channel`, `region`, `txn_channel`, `category`) and historical baselines outside the 90-day window (`avg_monthly_spend`, `total_amount`, `return_count`) are usable. Any trailing 90-day feature (like recency or 90-day spend/counts) is strictly excluded due to definitional leakage. The 81 zero-activity customers are excluded as a separate cohort.