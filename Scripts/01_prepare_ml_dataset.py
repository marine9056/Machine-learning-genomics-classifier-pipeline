import pandas as pd

# Load expression data
expr = pd.read_csv("data/raw/log_transformed_expression.csv", index_col=0)

# Load metadata
meta = pd.read_csv("data/raw/metadata.csv", encoding="latin1")


# Keep only required conditions
meta = meta[meta["condition"].isin(["Healthy Control", "Type 1 Diabetes"])]

# Encode labels
meta["label"] = meta["condition"].map({
    "Healthy Control": 0,
    "Type 1 Diabetes": 1
})

# Match expression columns with metadata samples
samples = meta["sample"]
expr = expr[samples]

# Transpose so rows = samples, columns = genes
X = expr.T

# Add labels
X["label"] = meta["label"].values

# Save processed dataset
X.to_csv("data/processed/ml_dataset.csv", index=False)

print("ML dataset saved successfully")
print("Shape:", X.shape)
