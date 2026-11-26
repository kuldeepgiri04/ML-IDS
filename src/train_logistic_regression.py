
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline

DATA_PATH = "dataset/processed/ids_binary.csv"
MODEL_PATH = "models/logistic_regression_ids.pkl"

# Collect a balanced sample without loading the full CSV
samples = []
counts = {0: 0, 1: 0}
MAX_PER_CLASS = 10000

print("Reading dataset in chunks...")

for chunk in pd.read_csv(DATA_PATH, chunksize=10000):
    chunk.columns = chunk.columns.str.strip()

    for label in [0, 1]:
        remaining = MAX_PER_CLASS - counts[label]

        if remaining > 0:
            class_rows = chunk[chunk["Target"] == label]

            if not class_rows.empty:
                selected = class_rows.sample(
                    n=min(remaining, len(class_rows)),
                    random_state=42
                )
                samples.append(selected)
                counts[label] += len(selected)

    if all(counts[label] >= MAX_PER_CLASS for label in [0, 1]):
        break

print("Samples collected:", counts)

data = pd.concat(samples, ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data.drop(columns=["Target"])
y = data["Target"]

# Keep numeric features and replace invalid values
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# Remove constant columns
X = X.loc[:, X.nunique() > 1]

print("Total features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling and model training in one pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ))
])

print("Training Logistic Regression...")
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

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH, compress=3)

print("\nModel saved successfully:", MODEL_PATH)
