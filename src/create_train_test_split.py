
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "dataset/processed/representative_sample.csv"
TRAIN_OUTPUT = "dataset/processed/train_data.csv"
TEST_OUTPUT = "dataset/processed/test_data.csv"

print("Loading representative dataset...")

data = pd.read_csv(INPUT)

print("Original shape:", data.shape)

# Remove duplicate records
data = data.drop_duplicates().reset_index(drop=True)

print("After removing duplicates:", data.shape)

X = data.drop(columns=["Target"])
y = data["Target"]

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_data = X_train.copy()
train_data["Target"] = y_train

test_data = X_test.copy()
test_data["Target"] = y_test

train_data.to_csv(TRAIN_OUTPUT, index=False)
test_data.to_csv(TEST_OUTPUT, index=False)

print("\nTRAINING DATA:")
print(train_data["Target"].value_counts())
print("Training records:", len(train_data))

print("\nTESTING DATA:")
print(test_data["Target"].value_counts())
print("Testing records:", len(test_data))

print("\nFiles saved successfully!")
