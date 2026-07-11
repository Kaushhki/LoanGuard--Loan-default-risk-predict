"""
data_cleaning_eda.py
---------------------
Cleans the raw applicant data and runs exploratory analysis to surface
key risk indicators: credit utilization, repayment history (late
payments), and debt-to-income ratio. Saves the cleaned dataset plus a
correlation/risk-indicator summary for the README and dashboard.

Run:
    python src/data_cleaning_eda.py
Outputs:
    data/loan_applicants_clean.csv
    data/risk_indicator_summary.csv
    models/eda_default_by_dti_bucket.png
    models/eda_default_by_utilization_bucket.png
    models/eda_correlation_heatmap.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")


def clean(df):
    before = len(df)
    df = df.drop_duplicates(subset=["applicant_id"], keep="first")
    print(f"Removed {before - len(df)} duplicate applicant_id rows")

    # Fix invalid ages (data entry errors: negative or absurd values)
    invalid_age = ((df["age"] < 18) | (df["age"] > 90)).sum()
    df.loc[(df["age"] < 18) | (df["age"] > 90), "age"] = df["age"].median()
    print(f"Corrected {invalid_age} invalid age values")

    # Impute numeric nulls with median
    for col in ["employment_years", "credit_utilization_pct", "annual_income",
                "credit_history_length_years"]:
        n_null = df[col].isna().sum()
        df[col] = df[col].fillna(df[col].median())
        print(f"Imputed {n_null} null {col} values with median")

    # Clip credit utilization to valid 0-100 range
    df["credit_utilization_pct"] = df["credit_utilization_pct"].clip(0, 100)

    return df.reset_index(drop=True)


def eda(df):
    # --- Debt-to-income vs default rate ---
    df["dti_bucket"] = pd.cut(
        df["debt_to_income_ratio"], bins=[0, 0.2, 0.35, 0.5, 1.5],
        labels=["Low (<20%)", "Moderate (20-35%)", "High (35-50%)", "Very High (>50%)"]
    )
    dti_summary = df.groupby("dti_bucket", observed=True)["default"].agg(["mean", "count"])
    dti_summary.columns = ["default_rate", "applicant_count"]

    fig, ax = plt.subplots(figsize=(7, 4))
    dti_summary["default_rate"].plot(kind="bar", ax=ax, color="#D64545")
    ax.set_ylabel("Default Rate")
    ax.set_title("Default Rate by Debt-to-Income Bucket")
    plt.tight_layout()
    plt.savefig("models/eda_default_by_dti_bucket.png", dpi=120)
    plt.close()

    # --- Credit utilization vs default rate ---
    df["util_bucket"] = pd.cut(
        df["credit_utilization_pct"], bins=[0, 30, 60, 100],
        labels=["Low (<30%)", "Moderate (30-60%)", "High (>60%)"]
    )
    util_summary = df.groupby("util_bucket", observed=True)["default"].agg(["mean", "count"])
    util_summary.columns = ["default_rate", "applicant_count"]

    fig, ax = plt.subplots(figsize=(7, 4))
    util_summary["default_rate"].plot(kind="bar", ax=ax, color="#4573D6")
    ax.set_ylabel("Default Rate")
    ax.set_title("Default Rate by Credit Utilization Bucket")
    plt.tight_layout()
    plt.savefig("models/eda_default_by_utilization_bucket.png", dpi=120)
    plt.close()

    # --- Late payments vs default rate ---
    late_summary = df.groupby("num_late_payments_last_year")["default"].agg(["mean", "count"])
    late_summary.columns = ["default_rate", "applicant_count"]

    # --- Correlation heatmap of numeric risk indicators ---
    numeric_cols = [
        "debt_to_income_ratio", "credit_utilization_pct", "num_late_payments_last_year",
        "credit_history_length_years", "num_credit_inquiries_last_6m",
        "num_existing_loans", "annual_income", "default"
    ]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Between Risk Indicators and Default")
    plt.tight_layout()
    plt.savefig("models/eda_correlation_heatmap.png", dpi=120)
    plt.close()

    # Combine into one summary CSV
    dti_summary.insert(0, "indicator", "debt_to_income_ratio")
    util_summary.insert(0, "indicator", "credit_utilization_pct")
    late_summary.insert(0, "indicator", "num_late_payments_last_year")
    dti_summary = dti_summary.rename_axis("bucket").reset_index()
    util_summary = util_summary.rename_axis("bucket").reset_index()
    late_summary = late_summary.rename_axis("bucket").reset_index()

    combined = pd.concat([dti_summary, util_summary, late_summary], ignore_index=True)
    combined.to_csv("data/risk_indicator_summary.csv", index=False)

    print("\n--- Correlation with default (top risk indicators) ---")
    print(corr["default"].sort_values(ascending=False).drop("default").to_string())


def main():
    df = pd.read_csv("data/loan_applicants_raw.csv")
    df_clean = clean(df)
    df_clean.to_csv("data/loan_applicants_clean.csv", index=False)
    eda(df_clean)
    print(f"\nCleaned dataset: {len(df_clean)} rows -> data/loan_applicants_clean.csv")


if __name__ == "__main__":
    main()
