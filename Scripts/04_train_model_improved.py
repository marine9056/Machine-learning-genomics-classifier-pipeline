import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Load dataset
# =========================
data = pd.read_csv("data/processed/ml_dataset_selected.csv")

X = data.drop("label", axis=1)
y = data["label"]

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# Improved Random Forest
# =========================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

# =========================
# Cross-validation
# =========================
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print("Cross-validation accuracy:", cv_scores.mean())

# =========================
# Train model
# =========================
rf.fit(X_train, y_train)

# =========================
# Predictions
# =========================
y_pred = rf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL  ✅ THIS IS THE FIX
# =========================
joblib.dump(rf, "models/random_forest_improved.pkl")

print("\nImproved model saved")
