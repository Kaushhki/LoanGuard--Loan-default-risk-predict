


import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "src"))
from risk_scoring import prob_to_score, assign_band
from feature_engineering import engineer_features, CATEGORICAL_COLS, NUMERIC_COLS

st.set_page_config(page_title="LoanGuard | Default Risk Scoring", layout="wide")


@st.cache_resource
def load_models():
    with open(os.path.join(BASE_DIR, "models", "best_model.json")) as f:
        best_model_name = json.load(f)["best_model"]
    logreg = joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression.joblib"))
    rf = joblib.load(os.path.join(BASE_DIR, "models", "random_forest.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))
    with open(os.path.join(BASE_DIR, "models", "feature_columns.json")) as f:
        cols = json.load(f)
    comparison = pd.read_csv(os.path.join(BASE_DIR, "models", "model_comparison.csv"))
    return best_model_name, logreg, rf, scaler, cols, comparison


best_model_name, logreg, rf, scaler, cols_info, comparison = load_models()
feature_cols = cols_info["feature_cols"]

st.title("🏦 LoanGuard — Loan Default Risk Scoring")
st.caption(
    "Predicts loan default probability from applicant demographic, income, "
    "and credit-history data, and translates it into a lending-ready risk "
    "score and recommended action."
)

tab1, tab2 = st.tabs(["🔍 Score an Applicant", "📈 Model Performance"])


with tab1:
    st.subheader("Applicant Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics**")
        age = st.number_input("Age", 18, 90, 32)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate"])
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business Owner"])
        employment_years = st.number_input("Years Employed", 0.0, 40.0, 5.0)

    with c2:
        st.markdown("**Income & Loan**")
        annual_income = st.number_input("Annual Income (₹)", 100000, 10000000, 800000, step=50000)
        loan_amount = st.number_input("Loan Amount Requested (₹)", 10000, 5000000, 300000, step=10000)
        loan_term_months = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)
        loan_purpose = st.selectbox(
            "Loan Purpose", ["Personal", "Auto", "Home Improvement", "Education", "Business", "Medical"]
        )
        monthly_debt_obligations = st.number_input("Existing Monthly Debt Obligations (₹)", 0, 500000, 15000, step=1000)

    with c3:
        st.markdown("**Credit History**")
        credit_history_length_years = st.number_input("Credit History Length (years)", 0.0, 30.0, 5.0)
        num_late_payments_last_year = st.number_input("Late Payments (last 12 months)", 0, 20, 0)
        credit_utilization_pct = st.slider("Credit Utilization (%)", 0, 100, 35)
        num_existing_loans = st.number_input("Number of Existing Loans", 0, 15, 1)
        num_credit_inquiries_last_6m = st.number_input("Credit Inquiries (last 6 months)", 0, 15, 1)

    if st.button("Score This Applicant", type="primary"):
        monthly_income = annual_income / 12
        debt_to_income_ratio = round(min(monthly_debt_obligations / monthly_income, 1.2), 3)

        raw = pd.DataFrame([{
            "age": age, "marital_status": marital_status, "education": education,
            "employment_type": employment_type, "employment_years": employment_years,
            "annual_income": annual_income, "credit_history_length_years": credit_history_length_years,
            "num_late_payments_last_year": num_late_payments_last_year,
            "credit_utilization_pct": credit_utilization_pct, "num_existing_loans": num_existing_loans,
            "num_credit_inquiries_last_6m": num_credit_inquiries_last_6m, "loan_amount": loan_amount,
            "loan_term_months": loan_term_months, "loan_purpose": loan_purpose,
            "monthly_debt_obligations": monthly_debt_obligations,
            "debt_to_income_ratio": debt_to_income_ratio,
        }])

        engineered = engineer_features(raw)
        encoded = pd.get_dummies(engineered, columns=CATEGORICAL_COLS, drop_first=True)
        
        for c in feature_cols:
            if c not in encoded.columns:
                encoded[c] = 0
        X_input = encoded[feature_cols]

        if best_model_name == "Logistic Regression":
            prob = logreg.predict_proba(scaler.transform(X_input))[0, 1]
        else:
            prob = rf.predict_proba(X_input)[0, 1]

        score = int(prob_to_score(np.array([prob]))[0])
        band, action = assign_band(prob)

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Default Probability", f"{prob:.1%}")
        r2.metric("Risk Score (300-900)", score)
        r3.metric("Risk Band", band)

        color = {"Very Low Risk": "🟢", "Low Risk": "🟢", "Moderate Risk": "🟡",
                 "High Risk": "🟠", "Very High Risk": "🔴"}.get(band, "⚪")
        st.info(f"{color} **Recommended action:** {action}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            gauge={
                "axis": {"range": [300, 900]},
                "bar": {"color": "#262730"},
                "steps": [
                    {"range": [300, 450], "color": "#E57373"},
                    {"range": [450, 600], "color": "#FFB74D"},
                    {"range": [600, 750], "color": "#FFF176"},
                    {"range": [750, 900], "color": "#81C784"},
                ],
            },
            title={"text": "Risk Score"},
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Scored using {best_model_name} (selected as best model by AUC-ROC).")


with tab2:
    st.subheader("Logistic Regression vs Random Forest")
    st.dataframe(comparison.set_index("model").style.format("{:.4f}"), use_container_width=True)
    st.caption(
        f"**{best_model_name}** was selected as the production model based on AUC-ROC — "
        "the metric least sensitive to the class imbalance in a ~19% default-rate dataset."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(BASE_DIR, "models", "roc_curve_comparison.png"),
                  caption="ROC Curve Comparison", use_container_width=True)
    with col2:
        st.image(os.path.join(BASE_DIR, "models", "feature_importance_rf.png"),
                  caption="Top Feature Importances (Random Forest)", use_container_width=True)

    st.image(os.path.join(BASE_DIR, "models", "confusion_matrices.png"),
              caption="Confusion Matrices", use_container_width=True)

    st.markdown("### Why AUC-ROC over accuracy?")
    st.write(
        "With only ~19% of applicants defaulting, a model that predicts "
        "'no default' for everyone would score ~81% accuracy while catching "
        "zero actual defaulters. Precision, recall, and AUC-ROC are tracked "
        "specifically to avoid optimizing for a metric that rewards ignoring "
        "the minority (default) class — the one that actually matters for risk."
    )
