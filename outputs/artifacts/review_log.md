# Review Log — Step 3 Self-Review

## Review focus areas

### 1. Claims in EDA_FINDINGS.md not backed by reproducible computation

**Reviewed**: Every numeric claim in `EDA_FINDINGS.md` is traceable to a direct computation from the raw data sources via `eda/profile.py`, `eda/hypotheses.py`, `eda/time_checks.py`, or `eda/leakage_checks.py`. All are confirmed reproducible by `scripts/verify_findings.py` (ran cleanly with all PASS).

**Findings**: None — all claims are backed by named functions with clear inputs.

### 2. Leakage-tainted variables in MODEL_RECOMMENDATION.md without caveats

**Reviewed**: The model recommendation explicitly lists `days_since_last_txn`, `total_spend_trailing_90d`, `txn_count_trailing_90d`, and `recency` as **strictly excluded** with a clear explanation of why (definitional window overlap). The zero-activity cohort is flagged as a separate segment. Every usable variable is listed with its source and type.

**Findings**: None — all leakage-tainted variables are appropriately excluded and documented. The zero-activity cohort is flagged correctly as a cohort separation issue, not imputed.

### 3. Hypothesis test using inappropriate test for variable shape

**Reviewed**: 
- `acquisition_channel`: Chi-square ✅ (categorical vs binary)
- `region`: Chi-square ✅ (categorical vs binary)
- `txn_channel`: Chi-square ✅ (categorical vs binary)
- `category`: Chi-square ✅ (categorical vs binary)
- `avg_monthly_spend`: Mann-Whitney U ✅ (continuous, right-skewed — violates normality)
- `total_amount`: Mann-Whitney U ✅ (continuous, right-skewed)
- `return_count`: Mann-Whitney U ✅ (zero-inflated count, right-skewed)
- `days_since_last_txn`: Mann-Whitney U ✅ (continuous, right-skewed; leakage check)

**Findings**: All tests are appropriate for their variable shapes. Continuous tests correctly exclude the trailing 90-day window (data before 2026-05-03 only).

### 4. Additional review items

**a. Multi-source comparison**: Verified that `compare_row_level_vs_monthly_agg()` actually reaggregates the row-level table rather than trusting Stage 0's guess. It does — full outer join on (customer_id, txn_month). The gap is localized to July 2026.

**b. Zero-activity vs missing distinction**: Confirmed the 81 zero-activity customers are legitimate (present in customers.csv with valid signup_dates and acquisition_channels, but zero transactions). They are not a data pipeline failure.

**c. MAD outlier rule**: Switched from IQR to MAD per Domain Context guidance. The MAD-based rule with threshold 3 is robust to the log-normal shape of transaction amounts.

**d. Statistically insignificant drivers**: `acquisition_channel` (p=0.12) and `region` (p=0.29) are correctly reported as NOT significant. Stage 0's draft over-interpreted their raw rate differences. The EDA_FINDINGS.md and DATA_ISSUES.md both flag this.

### 5. Resolutions

| Finding | Resolution |
|---|---|
| Stage 0 used IQR outlier rule | Fixed — switched to MAD-based (threshold 3) for log-normal data |
| Stage 0 over-interpreted channel/region rates | Fixed — formal chi-square tests show both are n.s.; reported honestly |
| Hypothesis function names conflict with pytest | Fixed — renamed to `hypothesis_*` to avoid pytest fixture collection |
| Multi-source gap was speculated, not measured | Fixed — explicit reaggregation shows gap concentrated in Jul 2026 |
| All test files needed | Fixed — 5 test files created, all 66 tests passing |
| Notebook needed proper narrative | Fixed — 46-cell notebook with markdown interpretation, runs top-to-bottom |

### 6. Unresolved items

- **Coverage-of-intent check**: The candidate drivers in `eda_review.md` (8 total) are all covered in `eda/hypotheses.py`, in `EDA_FINDINGS.md`, and in the test suite. Every driver has a corresponding test in `test_hypotheses.py` and verification in `test_findings_reproduce.py`. No gaps found.
- **Leakage check coverage**: The `leakage_checks.py` module covers as-of-date discipline, label-circular driver identification, and the trailing-window exclusion. `test_time_and_leakage_checks.py` covers all these paths. No gaps found.