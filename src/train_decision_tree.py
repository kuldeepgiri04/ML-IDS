
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading dataset...", flush=True)

# Read the same 50,000 balanced sample
samples = {0: [], 1: []}
sample_limit = 25000

for chunk in pd.read_csv(
    "dataset/processed/ids_binary.csv",
    chunksize=10000,
    low_memory=False
):
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)

    for label in [0, 1]:
        current = sum(len(x) for x in samples[label])
        remaining = sample_limit - current

        if remaining > 0:
            rows = chunk[chunk["Target"] == label]

            if not rows.empty:
                samples[label].append(
                    rows.sample(
                        n=min(remaining, len(rows)),
                        random_state=42
                    )
                )

df = pd.concat(
    samples[0] + samples[1],
    ignore_index=True
)

df = df.sample(frac=1, random_state=42)

X = df.drop(columns=["Target"])
y = df["Target"]

X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Decision Tree...", flush=True)

model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "models/decision_tree_ids.pkl", compress=3)

print("\nDecision Tree model saved!")
