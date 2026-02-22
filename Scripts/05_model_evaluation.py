import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# ==============================
# Load dataset
# ==============================
data = pd.read_csv("data/processed/ml_dataset_selected.csv")

X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Load models
# ==============================
baseline_model = joblib.load("models/random_forest_model.pkl")
improved_model = joblib.load("models/random_forest_improved.pkl")

# ==============================
# Predictions
# ==============================
baseline_probs = baseline_model.predict_proba(X_test)[:, 1]
improved_probs = improved_model.predict_proba(X_test)[:, 1]

baseline_preds = baseline_model.predict(X_test)
improved_preds = improved_model.predict(X_test)

# ==============================
# ROC Curve
# ==============================
fpr_base, tpr_base, _ = roc_curve(y_test, baseline_probs)
fpr_imp, tpr_imp, _ = roc_curve(y_test, improved_probs)

roc_auc_base = auc(fpr_base, tpr_base)
roc_auc_imp = auc(fpr_imp, tpr_imp)

plt.figure()
plt.plot(fpr_base, tpr_base, label=f"Baseline (AUC={roc_auc_base:.2f})")
plt.plot(fpr_imp, tpr_imp, label=f"Improved (AUC={roc_auc_imp:.2f})")
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("results/roc_curve_comparison.png")
plt.close()

# ==============================
# Precision-Recall Curve
# ==============================
prec_base, rec_base, _ = precision_recall_curve(y_test, baseline_probs)
prec_imp, rec_imp, _ = precision_recall_curve(y_test, improved_probs)

plt.figure()
plt.plot(rec_base, prec_base, label="Baseline")
plt.plot(rec_imp, prec_imp, label="Improved")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("results/pr_curve_comparison.png")
plt.close()

# ==============================
# Confusion Matrices
# ==============================
cm_base = confusion_matrix(y_test, baseline_preds)
cm_imp = confusion_matrix(y_test, improved_preds)

disp = ConfusionMatrixDisplay(cm_base)
disp.plot()
plt.title("Confusion Matrix — Baseline Model")
plt.savefig("results/confusion_matrix_baseline.png")
plt.close()

disp = ConfusionMatrixDisplay(cm_imp)
disp.plot()
plt.title("Confusion Matrix — Improved Model")
plt.savefig("results/confusion_matrix_improved.png")
plt.close()

# ==============================
# Metrics Summary
# ==============================
metrics = pd.DataFrame({
    "Model": ["Baseline", "Improved"],
    "Accuracy": [
        accuracy_score(y_test, baseline_preds),
        accuracy_score(y_test, improved_preds)
    ],
    "Precision": [
        precision_score(y_test, baseline_preds),
        precision_score(y_test, improved_preds)
    ],
    "Recall": [
        recall_score(y_test, baseline_preds),
        recall_score(y_test, improved_preds)
    ],
    "F1 Score": [
        f1_score(y_test, baseline_preds),
        f1_score(y_test, improved_preds)
    ]
})

metrics.to_csv("results/model_comparison_metrics.csv", index=False)

# ==============================
# Bar Plot Comparison
# ==============================
metrics.set_index("Model").plot(kind="bar")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.savefig("results/model_comparison_barplot.png")
plt.close()

print("Evaluation complete. All results saved in /results")