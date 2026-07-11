# LoanGuard — Loan Default Risk Prediction

An end-to-end credit-risk classification pipeline: applicant data → EDA →
feature engineering → model comparison (Logistic Regression vs Random
Forest) → a risk-scoring framework and interactive scoring dashboard.

**[Live Dashboard →](#deployment)** *(add your Streamlit Cloud link here after deploying)*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

---

## What this project does

Predicts the probability that a loan applicant will default, using
demographic, income, and credit-history data, then translates that
probability into a lending-ready decision:

1. **Generate/ingest data** — synthetic applicant dataset (15,000 rows)
   with a default label driven by realistic risk factors plus noise.
2. **Clean & explore** — surface the key risk indicators (credit
   utilization, repayment history, debt-to-income ratio) via correlation
   analysis and default-rate-by-bucket breakdowns.
3. **Engineer features** — derive loan-to-income ratio, estimated
   monthly installment burden, and a composite high-risk flag; encode
   categoricals; split into train/test sets.
4. **Train & compare models** — Logistic Regression (interpretable
   baseline) vs Random Forest (higher-capacity ensemble), evaluated on
   accuracy, precision, recall, F1, and AUC-ROC.
5. **Risk-score the output** — convert model probability into a
   300-900 risk score and a 5-tier risk band (Very Low → Very High)
   with a recommended lending action per band.
6. **Serve it** — an interactive Streamlit dashboard where you enter
   applicant details and get a live risk score, plus a model-performance
   tab with ROC curves and feature importances.

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.658 | 0.317 | **0.664** | 0.430 | **0.720** |
| Random Forest | **0.692** | **0.326** | 0.553 | 0.410 | 0.704 |

Logistic Regression was selected as the production model — it has the
higher AUC-ROC and, critically for a lending use case, a much higher
**recall** (catches 66% of actual defaulters vs 55% for Random Forest).
In credit risk, missing a defaulter (false negative) is typically far
costlier than an extra manual review (false positive), so recall at a
reasonable precision was weighted over raw accuracy when picking the
production model.

**Why not just optimize for accuracy?** The dataset has a ~19% default
rate. A model that predicts "no default" for every applicant would score
~81% accuracy while catching zero real defaulters — which is why
precision, recall, and AUC-ROC are tracked instead of accuracy alone.

### Top risk indicators (correlation with default)

| Indicator | Correlation with default |
|---|---|
| Credit utilization % | 0.189 |
| Late payments (last 12mo) | 0.173 |
| Debt-to-income ratio | 0.129 |
| Credit inquiries (last 6mo) | 0.064 |
| Existing loan count | 0.039 |

Matches underwriting intuition: how much of your available credit
you're using, whether you've missed payments recently, and how much of
your income is already committed to debt are the strongest individual
predictors of default in this dataset.

### Risk band validation

The risk bands were built from model probability thresholds, then
validated against actual (held-out) default outcomes:

| Risk Band | Applicants | Avg Risk Score | Actual Default Rate |
|---|---|---|---|
| Very High Risk | 1,216 | 513 | 31.7% |
| High Risk | 1,080 | 661 | 13.9% |
| Moderate Risk | 614 | 758 | 6.8% |
| Low Risk | 89 | 830 | 3.4% |
| Very Low Risk | 1 | 873 | 0.0% |

Actual default rate decreases monotonically as the risk score increases
— confirmation that the score is doing what it's supposed to.

## Tech stack

| Layer | Tool |
|---|---|
| Data generation & cleaning | Python (pandas, numpy) |
| EDA & visualization | matplotlib, seaborn |
| Modeling | scikit-learn (Logistic Regression, Random Forest) |
| Dashboard | Streamlit + Plotly |
| Model persistence | joblib |

## Project structure

```
loan-default-risk/
├── dashboard.py                      # Streamlit app (entry point)
├── requirements.txt
├── data/
│   ├── loan_applicants_raw.csv
│   ├── loan_applicants_clean.csv
│   ├── X_train.csv / X_test.csv / y_train.csv / y_test.csv
│   ├── scored_applicants.csv         # test-set applicants with risk scores
│   └── risk_indicator_summary.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── scaler.joblib
│   ├── model_comparison.csv
│   ├── risk_band_summary.csv
│   ├── roc_curve_comparison.png
│   ├── feature_importance_rf.png
│   ├── confusion_matrices.png
│   └── eda_*.png
└── src/
    ├── generate_data.py              # synthetic applicant data generator
    ├── data_cleaning_eda.py          # cleaning + risk-indicator EDA
    ├── feature_engineering.py        # derived features, train/test split
    ├── train_models.py               # trains & compares both models
    └── risk_scoring.py               # probability -> score -> band -> action
```

## Risk-scoring framework

| Default Probability | Risk Band | Risk Score Range | Recommended Action |
|---|---|---|---|
| 0–5% | Very Low Risk | ~855-900 | Auto-approve |
| 5–15% | Low Risk | ~735-855 | Approve |
| 15–30% | Moderate Risk | ~600-735 | Approve with conditions (higher rate / lower limit) |
| 30–50% | High Risk | ~450-600 | Manual underwriter review |
| 50%+ | Very High Risk | 300-450 | Decline |

The 300-900 scale mirrors familiar credit-bureau scoring so it's
immediately readable by a lending team without needing to interpret a
raw probability.

## Running locally

```bash
git clone https://github.com/<your-username>/loan-default-risk.git
cd loan-default-risk
pip install -r requirements.txt

# Optional — regenerate everything from scratch (raw/clean data, trained
# models, and plots are already included in the repo, so this is optional)
python src/generate_data.py
python src/data_cleaning_eda.py
python src/feature_engineering.py
python src/train_models.py
python src/risk_scoring.py

# Launch the dashboard
streamlit run dashboard.py
```

## Deployment

To deploy the live dashboard on Streamlit Cloud:
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your
   GitHub account, and select this repo with `dashboard.py` as the entry
   point.
3. Add the deployed link back into this README.

## Possible extensions

- Add SHAP values for per-applicant explainability (why this score?)
- Try gradient boosting (XGBoost/LightGBM) as a third model
- Add threshold tuning UI to let a risk team trade off precision/recall live
- Add model monitoring for feature/label drift over time
