
import pandas as pd
import numpy as np

TRAIN_PATH = "dataset/processed/ids_binary.csv"
TEST_PATH = "dataset/processed/independent_test.csv"

print("Reading training sample...")

train_parts = []
counts = {0: 0, 1: 0}

for chunk in pd.read_csv(TRAIN_PATH, chunksize=10000):
    for label in [0, 1]:
        remaining = 1000 - counts[label]

        if remaining > 0:
            rows = chunk[chunk["Target"] == label]

            if not rows.empty:
                selected = rows.head(remaining)
                train_parts.append(selected)
                counts[label] += len(selected)

    if counts[0] == 1000 and counts[1] == 1000:
        break

train = pd.concat(train_parts, ignore_index=True)
test = pd.read_csv(TEST_PATH)

print("\nTraining class distribution:")
print(train["Target"].value_counts())

print("\nIndependent test class distribution:")
print(test["Target"].value_counts())

features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s"
]

train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

print("\nFeature comparison:")

for feature in features:
    if feature in train.columns and feature in test.columns:

        train_values = pd.to_numeric(
            train[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        test_values = pd.to_numeric(
            test[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        print("\nFeature:", feature)
        print("Training median:", train_values.median())
        print("Test median:", test_values.median())
    else:
        print("\nFeature missing:", feature)
