"""
generate_data.py
-----------------
Generates a synthetic loan applicant dataset with demographic, income,
and credit-history features, and a binary default label. The default
probability is a function of realistic risk drivers (credit utilization,
repayment history, debt-to-income ratio, credit history length) plus
noise -- so the signal is learnable but not trivial, mirroring a real
underwriting dataset.

Run:
    python src/generate_data.py
Outputs:
    data/loan_applicants_raw.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 15000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    age = np.random.randint(21, 65, N)
    employment_years = np.clip(np.random.exponential(5, N), 0, 40).round(1)
    annual_income = np.clip(np.random.lognormal(mean=10.8, sigma=0.5, size=N), 120000, 5000000).round(0)

    marital_status = np.random.choice(["Single", "Married", "Divorced"], N, p=[0.45, 0.45, 0.10])
    education = np.random.choice(
        ["High School", "Graduate", "Post Graduate"], N, p=[0.3, 0.5, 0.2]
    )
    employment_type = np.random.choice(
        ["Salaried", "Self-Employed", "Business Owner"], N, p=[0.6, 0.25, 0.15]
    )

    credit_history_length = np.clip(np.random.exponential(6, N), 0, 30).round(1)
    num_late_payments_last_year = np.random.poisson(0.8, N)
    credit_utilization_pct = np.clip(np.random.beta(2, 3, N) * 100, 0, 100).round(1)

    loan_amount = np.clip(np.random.lognormal(mean=11.5, sigma=0.6, size=N), 50000, 3000000).round(0)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60], N, p=[0.1, 0.2, 0.35, 0.2, 0.15])
    loan_purpose = np.random.choice(
        ["Personal", "Auto", "Home Improvement", "Education", "Business", "Medical"], N
    )

    monthly_income = annual_income / 12
    monthly_debt_obligations = np.clip(
        monthly_income * np.random.uniform(0.05, 0.55, N), 0, None
    ).round(0)
    debt_to_income_ratio = np.clip(monthly_debt_obligations / monthly_income, 0, 1.2).round(3)

    num_existing_loans = np.random.poisson(1.2, N)
    num_credit_inquiries_last_6m = np.random.poisson(1.0, N)

    # ---------------- default probability model ----------------
    # Higher risk: high DTI, high utilization, late payments, short
    # credit history, more inquiries, lower income, self-employed.
    z = (
        -4.1
        + 2.6 * debt_to_income_ratio
        + 0.028 * credit_utilization_pct
        + 0.55 * num_late_payments_last_year
        - 0.05 * credit_history_length
        + 0.18 * num_credit_inquiries_last_6m
        + 0.12 * num_existing_loans
        - 0.35 * (annual_income / 1_000_000)
        + 0.25 * (employment_type == "Self-Employed").astype(int)
        - 0.15 * (education == "Post Graduate").astype(int)
        + np.random.normal(0, 0.7, N)  # noise
    )
    default_prob = sigmoid(z)
    default = (np.random.uniform(0, 1, N) < default_prob).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"APP{i:06d}" for i in range(1, N + 1)],
        "age": age,
        "marital_status": marital_status,
        "education": education,
        "employment_type": employment_type,
        "employment_years": employment_years,
        "annual_income": annual_income,
        "credit_history_length_years": credit_history_length,
        "num_late_payments_last_year": num_late_payments_last_year,
        "credit_utilization_pct": credit_utilization_pct,
        "num_existing_loans": num_existing_loans,
        "num_credit_inquiries_last_6m": num_credit_inquiries_last_6m,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "loan_purpose": loan_purpose,
        "monthly_debt_obligations": monthly_debt_obligations,
        "debt_to_income_ratio": debt_to_income_ratio,
        "default": default,
    })

    # ---------------- inject realistic messiness ----------------
    # Missing values
    for col, frac in [("employment_years", 0.02), ("credit_utilization_pct", 0.015),
                       ("annual_income", 0.01), ("credit_history_length_years", 0.01)]:
        idx = df.sample(frac=frac, random_state=1).index
        df.loc[idx, col] = np.nan

    # Duplicates
    dupes = df.sample(60, random_state=2)
    df = pd.concat([df, dupes], ignore_index=True)

    # A few outliers / data entry errors
    err_idx = df.sample(10, random_state=3).index
    df.loc[err_idx, "age"] = np.random.choice([-5, 150, 999], size=len(err_idx))

    df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle
    df.to_csv("data/loan_applicants_raw.csv", index=False)
    print(f"Generated {len(df)} rows -> data/loan_applicants_raw.csv")
    print(f"Default rate: {df['default'].mean():.2%}")


if __name__ == "__main__":
    main()
