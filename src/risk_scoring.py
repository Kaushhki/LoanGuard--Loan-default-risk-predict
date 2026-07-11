"""
risk_scoring.py
-----------------
Translates raw model default probabilities into a risk-scoring
framework usable by a lending team: a 300-900 risk score (familiar,
credit-bureau-style scale, inverted so higher = safer) and a risk band
with a recommended action. This is the layer that turns "the model
says 34% probability of default" into something a loan officer can
actually act on.

Run:
    python src/risk_scoring.py
Outputs:
    data/scored_applicants.csv
    models/risk_band_summary.csv
"""

import json
import joblib
import numpy as np
import pandas as pd

SCORE_MIN, SCORE_MAX = 300, 900

RISK_BANDS = [
    (0.00, 0.05, "Very Low Risk", "Auto-approve"),
    (0.05, 0.15, "Low Risk", "Approve"),
    (0.15, 0.30, "Moderate Risk", "Approve with conditions (higher rate / lower limit)"),
    (0.30, 0.50, "High Risk", "Manual underwriter review"),
    (0.50, 1.01, "Very High Risk", "Decline"),
]


def prob_to_score(prob):
    """Map default probability to a 300-900 score, higher = safer."""
    safety = 1 - prob
    return (SCORE_MIN + safety * (SCORE_MAX - SCORE_MIN)).round(0).astype(int)


def assign_band(prob):
    for lo, hi, label, action in RISK_BANDS:
        if lo <= prob < hi:
            return label, action
    return "Very High Risk", "Decline"


def main():
    with open("models/best_model.json") as f:
        best_model_name = json.load(f)["best_model"]

    model_file = "logistic_regression.joblib" if best_model_name == "Logistic Regression" else "random_forest.joblib"
    model = joblib.load(f"models/{model_file}")
    scaler = joblib.load("models/scaler.joblib")

    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    applicants = pd.read_csv("data/loan_applicants_clean.csv")

    if best_model_name == "Logistic Regression":
        prob = model.predict_proba(scaler.transform(X_test))[:, 1]
    else:
        prob = model.predict_proba(X_test)[:, 1]

    scored = X_test.copy()
    scored["default_probability"] = prob.round(4)
    scored["actual_default"] = y_test.values
    scored["risk_score"] = prob_to_score(prob)
    bands = [assign_band(p) for p in prob]
    scored["risk_band"] = [b[0] for b in bands]
    scored["recommended_action"] = [b[1] for b in bands]

    scored.to_csv("data/scored_applicants.csv", index=False)

    band_summary = scored.groupby("risk_band").agg(
        applicant_count=("risk_score", "count"),
        avg_risk_score=("risk_score", "mean"),
        actual_default_rate=("actual_default", "mean"),
    ).round(3)
    band_order = [b[2] for b in RISK_BANDS][::-1]
    band_summary = band_summary.reindex(band_order)
    band_summary.to_csv("models/risk_band_summary.csv")

    print(f"Scored {len(scored)} test-set applicants using {best_model_name}")
    print("\n--- Risk Band Summary (validates against actual default rate) ---")
    print(band_summary.to_string())


if __name__ == "__main__":
    main()
