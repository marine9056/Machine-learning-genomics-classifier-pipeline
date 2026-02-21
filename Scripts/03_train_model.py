import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

data = pd.read_csv("data/processed/ml_dataset_selected.csv")

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

joblib.dump(model, "models/random_forest_baseline.pkl")
print("Baseline model saved")
