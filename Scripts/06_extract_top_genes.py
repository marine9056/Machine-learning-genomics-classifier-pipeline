import pandas as pd
import joblib
import os

# Create results folder if missing
os.makedirs("results", exist_ok=True)

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

# Load feature-selected dataset
data = pd.read_csv("data/processed/ml_dataset_selected.csv")

# Separate features and labels
X = data.drop("label", axis=1)

# Get feature importance
importances = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    "gene": X.columns,
    "importance": importances
})

# Sort by importance
importance_df = importance_df.sort_values(by="importance", ascending=False)

# Save top genes
top_genes = importance_df.head(20)
top_genes.to_csv("results/top_predictive_genes.csv", index=False)

print("Top predictive genes saved")
print(top_genes)