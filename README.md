# Churn EDA — Verified Analysis

Exploratory Data Analysis of an e-commerce customer churn dataset across three heterogeneous data sources (CSV, SQLite, Parquet).

## Project Structure

```
repo-root/
  eda/                    # Analysis modules (load, profile, hypotheses, time_checks, leakage_checks)
  scripts/                # Runner scripts for each analysis module
  notebooks/              # Jupyter notebook with full narrative + executable code cells
  tests/                  # pytest test suite (66 tests, all passing)
  data/                   # Raw data sources (not tracked in git)
  outputs/                # Generated JSON summaries + artifact documents
    artifacts/            #   ARTIFACT_NOTES.md, review_log.md
  EDA_FINDINGS.md         # Verified findings report
  MODEL_RECOMMENDATION.md # Model-family recommendation with justification
  DATA_ISSUES.md          # Documented data issues found during analysis
  pytest.ini              # pytest configuration
  requirements.txt        # Python dependencies
  .env.example            # Placeholder (no environment variables needed)
```

## Prerequisites

- **Python 3.11+**
- **pip** (Python package installer)

## Environment Setup

### 1. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify setup

```bash
python -c "import pandas, numpy, scipy, pyarrow, matplotlib, jupyter, pytest; print('All dependencies installed')"
```

## Running the Notebook

The main analysis notebook is `notebooks/eda_churn.ipynb` (52 cells: 16 code + 36 markdown).

### Option A: VS Code

1. Open the project root in VS Code
2. Open `notebooks/eda_churn.ipynb`
3. If prompted, select the `.venv` kernel (from the virtual environment created above)
4. Click **Run All** or run cells individually

### Option B: Jupyter Lab / Notebook

```bash
# From the project root (where your virtual env is active):
jupyter notebook notebooks/eda_churn.ipynb
# or
jupyter lab notebooks/eda_churn.ipynb
```

### Option C: Headless execution (terminal)

```bash
# From the project root with the virtual env active:
python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 --output eda_churn_executed notebooks/eda_churn.ipynb
```

This produces `notebooks/eda_churn_executed.ipynb` with all outputs populated.

> **Note:** The notebook imports from the `eda/` modules and loads data from `data/`. Both `eda/` and `data/` are siblings of `notebooks/`, so the notebook calls `os.chdir()` to the project root at startup.

## Running the Runner Scripts

All four runner scripts live in `scripts/` and can be invoked from the project root:

```bash
# From the project root with the virtual env active:

# 1. Profile — schema, missingness, distributions, outliers, zero-activity customers
python scripts/run_profile.py

# 2. Hypothesis tests — all 8 statistical tests (chi-square / Mann-Whitney U)
python scripts/run_hypothesis_tests.py

# 3. Time and leakage checks — monthly trends, multi-source agreement, as-of-date validation
python scripts/run_time_and_leakage_checks.py

# 4. Verify all findings — proves every numeric claim in EDA_FINDINGS.md reproduces from raw data
python scripts/verify_findings.py
```

Each script prints its results to the terminal and saves a JSON summary to `outputs/`.

## Running the Test Suite

There are 66 tests across 5 test files. **Always use the `pytest` command** (not `python` directly):

```bash
# From the project root with the virtual env active:

# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_load.py -v
pytest tests/test_profile.py -v
pytest tests/test_hypotheses.py -v
pytest tests/test_time_and_leakage_checks.py -v
pytest tests/test_findings_reproduce.py -v

# Run a specific test class
pytest tests/test_findings_reproduce.py::TestCustomerClaims -v

# Run a specific test
pytest tests/test_findings_reproduce.py::TestCustomerClaims::test_churn_rate -v

# Stop on first failure
pytest -x

# Show print() output during tests (no-capture mode)
pytest -v -s
```

### Troubleshooting test imports

If `pytest` reports `ModuleNotFoundError: No module named 'eda'`:
1. Make sure you're running from the **project root** (the directory containing `eda/`, `tests/`, and `pytest.ini`)
2. Verify your virtual environment is active
3. Check that `pytest.ini` exists at the project root with:
   ```ini
   [pytest]
   pythonpath = .
   ```

The `pytest.ini` file ensures pytest can find the `eda/` package. **Do not run tests with `python tests/test_*.py`** — the `pytest.ini` `pythonpath` setting is only honored by the `pytest` command.

## Expected Test Output

```
============================= test session starts =============================
...
tests/test_load.py .........                                           [ 13%]
tests/test_profile.py .................                                [ 36%]
tests/test_hypotheses.py ....................                          [ 66%]
tests/test_time_and_leakage_checks.py ...............                  [ 89%]
tests/test_findings_reproduce.py .......                               [100%]

============================== 66 passed in ... ===============================
```

## Expected Runner Output

Each runner script exits cleanly with printed results. `verify_findings.py` exits with code 0 and prints:

```
ALL CLAIMS VERIFIED SUCCESSFULLY
```

## Key Findings Summary

| Finding | Value |
|---|---|
| Churn rate | 13.47% (202 / 1500) |
| Zero-activity customers | 81 (all churned, separate cohort) |
| Acquisition channel significance | **Not significant** (p = 0.12) |
| Region significance | **Not significant** (p = 0.29) |
| Transaction channel significance | **Significant** (p < 0.001) |
| Category significance | **Significant** (p < 0.001) |
| Monthly growth ratio | 8.11× over 30 months (non-stationary) |
| Multi-source disagreement | July 2026 only (~237 missing in parquet) |
| Leakage cutoff | 2026-05-03 (90 days before as-of date 2026-08-01) |

## Model Recommendation

**Gradient Boosted Trees** (XGBoost / LightGBM) based on:
- Heavily right-skewed numeric features (trees handle natively)
- Mixed categorical + numeric features (no one-hot encoding needed)
- Non-linear interaction effects
- Built-in class-weight support for the 13.5% imbalance

**Excluded from modeling:** All trailing-90-day features (definitional leakage), zero-activity cohort treated as separate segment.

## Data Sources

| File | Format | Rows | Columns |
|---|---|---|---|
| `data/customers.csv` | CSV | 1,500 | 5 |
| `data/transactions.db` | SQLite | 11,742 | 6 |
| `data/transactions_monthly_agg.parquet` | Parquet | 8,086 | 5 |