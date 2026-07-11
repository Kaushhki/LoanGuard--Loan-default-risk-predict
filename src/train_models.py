"""
train_models.py
-----------------
Trains and compares two classification algorithms for loan default
prediction: Logistic Regression (interpretable baseline) and Random
Forest (higher-capacity ensemble). Evaluates both on accuracy,
precision, recall, F1, and AUC-ROC, with a specific focus on the
precision/recall trade-off since false positives (rejecting a good
applicant) and false negatives (approving a defaulter) carry very
different business costs in lending.

Run:
    python src/train_models.py
Outputs:
    models/logistic_regression.joblib
    models/random_forest.joblib
    models/scaler.joblib
    models/model_comparison.csv
    models/roc_curve_comparison.png
    models/feature_importance_rf.png
    models/confusion_matrices.png
"""

import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)


def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }
    print(f"\n-- {name} --")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:10s}: {v:.4f}")
    return metrics, y_pred


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Logistic Regression needs scaled features; Random Forest doesn't.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------- Logistic Regression ----------------------
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logreg.fit(X_train_scaled, y_train)
    logreg_prob = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_metrics, logreg_pred = evaluate("Logistic Regression", logreg, X_test, y_test, logreg_prob)

    # ------------------------- Random Forest --------------------------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_metrics, rf_pred = evaluate("Random Forest", rf, X_test, y_test, rf_prob)

    # ---------------------------- Save models ---------------------------
    joblib.dump(logreg, "models/logistic_regression.joblib")
    joblib.dump(rf, "models/random_forest.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    comparison = pd.DataFrame([logreg_metrics, rf_metrics])
    comparison.to_csv("models/model_comparison.csv", index=False)
    print("\n--- Model Comparison ---")
    print(comparison.to_string(index=False))

    best_model = comparison.loc[comparison["auc_roc"].idxmax(), "model"]
    print(f"\nBest model by AUC-ROC: {best_model}")
    with open("models/best_model.json", "w") as f:
        json.dump({"best_model": best_model}, f)

    # ------------------------------- ROC curve -------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, prob in [("Logistic Regression", logreg_prob), ("Random Forest", rf_prob)]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Logistic Regression vs Random Forest")
    ax.legend()
    plt.tight_layout()
    plt.savefig("models/roc_curve_comparison.png", dpi=120)
    plt.close()

    # --------------------------- Feature importance ---------------------------
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", ax=ax, color="#4573D6")
    ax.set_title("Top 12 Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("models/feature_importance_rf.png", dpi=120)
    plt.close()

    # --------------------------- Confusion matrices ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, pred) in zip(axes, [("Logistic Regression", logreg_pred), ("Random Forest", rf_pred)]):
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig("models/confusion_matrices.png", dpi=120)
    plt.close()

    print("\nSaved: models/*.joblib, model_comparison.csv, roc_curve_comparison.png, "
          "feature_importance_rf.png, confusion_matrices.png")


if __name__ == "__main__":
    main()
