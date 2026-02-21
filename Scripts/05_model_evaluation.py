import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# =========================
# Load dataset
# =========================
data = pd.read_csv("data/processed/ml_dataset_selected.csv")

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# Load models
# =========================
baseline_model = joblib.load("models/random_forest_model.pkl")
improved_model = joblib.load("models/random_forest_improved.pkl")

models = {
    "Baseline": baseline_model,
    "Improved": improved_model
}

# =========================
# Evaluate models
# =========================
for name, model in models.items():

    print(f"\n===== {name} Model =====")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # =========================
    # ROC Curve
    # =========================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name} (AUC = {roc_auc:.2f})")
    plt.savefig(f"results/roc_{name.lower()}.png")
    plt.close()

    # =========================
    # Precision-Recall Curve
    # =========================
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {name}")
    plt.savefig(f"results/pr_{name.lower()}.png")
    plt.close()

print("\nEvaluation complete. Plots saved in results/")
