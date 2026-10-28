
import pandas as pd
from pathlib import Path

# Dataset file paths
train_path = Path("data/raw/training-set.csv")
test_path = Path("data/raw/testing-set.csv")

# Check that both files exist
for path in [train_path, test_path]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Check the filename and location."
        )

# Load the datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Remove spaces around column names
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

# Display dataset information
print("\n========== TRAINING DATASET ==========")
print("Shape:", train.shape)
print("\nColumn names:")
print(train.columns.tolist())

print("\nFirst 5 rows:")
print(train.head())

print("\nLabel distribution:")
print(train["label"].value_counts())

print("\nMissing values:", train.isnull().sum().sum())

print("\n========== TESTING DATASET ==========")
print("Shape:", test.shape)

print("\nLabel distribution:")
print(test["label"].value_counts())  # label = 0 -> normal traffic , label = 1 -> attack traffic

print("\nMissing values:", test.isnull().sum().sum())
