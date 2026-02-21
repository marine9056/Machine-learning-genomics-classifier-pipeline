import pandas as pd

data = pd.read_csv("data/processed/ml_dataset.csv")

X = data.drop("label", axis=1)
y = data["label"]

# Select top variable genes
variances = X.var().sort_values(ascending=False)
top_genes = variances.head(1000).index

X_selected = X[top_genes]
X_selected["label"] = y

X_selected.to_csv("data/processed/ml_dataset_selected.csv", index=False)

print("Feature-selected dataset saved")
print("Selected genes:", len(top_genes))
