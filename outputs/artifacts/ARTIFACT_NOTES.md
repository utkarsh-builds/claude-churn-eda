# Artifact Notes — Self-Critique Log

### What I got wrong or reconsidered during the build

#### 1. Outlier rule: IQR → MAD-based

Stage 0's draft applied the standard 1.5×IQR rule to transaction amounts and flagged ~6-7% as outliers. During the build, I re-examined this against the Domain Context guidance and the data's actual distribution (skew = 6.02, log-normal shape). Natural high-value purchases in a right-skewed transaction distribution are not errors — applying a normal-distribution rule to a log-normal variable is wrong. I switched to a **MAD-based rule with threshold 3**, which is more robust for skewed distributions.

**Why the switch matters**: Standard IQR incorrectly flags ~790 legitimate high-value transactions as anomalies. The MAD-based rule (threshold 3) is distribution-robust and correctly identifies only the extreme tail.

#### 2. acquisition_channel and region: not significant

Stage 0's draft presented these as meaningful churn drivers ("Organic customers churn at ~1.5× the rate of paid_search"). The formal chi-square tests showed both are **not statistically significant** (p=0.12, p=0.29). The observed rate differences are within the range of sampling variation. This was the single most important correction from Stage 0's informal "eyeballed" approach to a proper statistical analysis.

#### 3. Multi-source gap: smaller than Stage 0 guessed, but localized

Stage 0 speculated about potential multi-source disagreement broadly. The actual gap is **concentrated in a single month** (July 2026), where the parquet is missing ~237 transactions. For all other months (Feb 2024 – Jun 2026), the two sources agree perfectly. This is a cleaner result than Stage 0's vague "might not match" — it's a specific snapshot-timing artifact.

#### 4. Zero-activity cohort: completely separate, not imputable

Stage 0's draft raised the possibility of leakage for the zero-activity cohort but framed it as a feature to "caveat." During the build, it became clear these 81 customers are a **fundamentally different population** — they have zero lifetime transactions, which means any transaction-based feature for them would be a fabricated zero. They must be excluded entirely from transaction-based models or modeled as a separate segment. This is a cohort separation issue, not a leakage issue (though both are concerns).

#### 5. Return_count effect: opposite direction from expectation

I expected churned customers would have higher return counts (reflecting dissatisfaction). The actual result is the opposite: churned customers have **fewer** returns (0.20 vs 0.42). This makes sense in retrospect — customers who leave the platform naturally engage less in all activities, including returns. The feature is unlikely to be a strong actionable predictor.

#### 6. avg_monthly_spend and total_amount: also opposite direction

Same pattern — churned customers have **lower** spend, not higher. Non-churned customers average $57/month vs $40 for churned. This seems intuitive (spending less leads to churn, or impending churn reduces spending), but it's worth noting that Stage 0 didn't venture a directional guess on these, so there was no wrong prediction to correct.

---

### Re-verified specifically before concluding:

- **Leakage-circularity flag**: ✅ Confirmed. The `days_since_last_txn` test shows a 145-day median gap between churned (168d) and not churned (23d). This is the churn label definition itself, not a prediction.
- **Zero-activity vs missing**: ✅ Confirmed — these are genuine customers with zero transaction history, not a data pipeline failure. All 81 are churned.
- **Multi-source comparison**: ✅ Ran the actual reaggregation. Perfect agreement except July 2026.