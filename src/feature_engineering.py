"""
feature_engineering.py
------------------------
Prepares the modeling dataset: encodes categoricals, engineers a couple
of derived risk features, and splits into train/test sets. Saves the
fitted preprocessing pipeline so it can be reused by the dashboard for
scoring new applicants.

Run:
    python src/feature_engineering.py
Outputs:
    data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv
    models/feature_columns.json
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["marital_status", "education", "employment_type", "loan_purpose"]

NUMERIC_COLS = [
    "age", "employment_years", "annual_income", "credit_history_length_years",
    "num_late_payments_last_year", "credit_utilization_pct", "num_existing_loans",
    "num_credit_inquiries_last_6m", "loan_amount", "loan_term_months",
    "monthly_debt_obligations", "debt_to_income_ratio",
]


def engineer_features(df):
    df = df.copy()
    # Derived feature: loan amount relative to annual income (leverage)
    df["loan_to_income_ratio"] = (df["loan_amount"] / df["annual_income"]).round(3)
    # Derived feature: estimated monthly installment burden
    df["estimated_monthly_installment"] = (df["loan_amount"] / df["loan_term_months"]).round(0)
    # Derived feature: composite payment stress flag
    df["high_risk_flag"] = (
        (df["debt_to_income_ratio"] > 0.5)
        | (df["credit_utilization_pct"] > 75)
        | (df["num_late_payments_last_year"] >= 3)
    ).astype(int)
    return df


def main():
    df = pd.read_csv("data/loan_applicants_clean.csv")
    df = engineer_features(df)

    numeric_cols = NUMERIC_COLS + [
        "loan_to_income_ratio", "estimated_monthly_installment", "high_risk_flag"
    ]

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    feature_cols = numeric_cols + [
        c for c in df_encoded.columns
        if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_COLS)
    ]

    X = df_encoded[feature_cols]
    y = df_encoded["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    with open("models/feature_columns.json", "w") as f:
        json.dump({
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": CATEGORICAL_COLS,
        }, f, indent=2)

    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"Train default rate: {y_train.mean():.2%}  Test default rate: {y_test.mean():.2%}")


if __name__ == "__main__":
    main()
