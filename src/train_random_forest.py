
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ----------------------------------
# CONFIGURATION
# ----------------------------------

INPUT_FILE = "dataset/processed/ids_binary.csv"
MODEL_FILE = "models/random_forest_ids.pkl"

SAMPLE_PER_CLASS = 25000
CHUNK_SIZE = 10000

print("Starting Random Forest IDS training...", flush=True)

# ----------------------------------
# STEP 1: LOAD BALANCED SAMPLE
# ----------------------------------

samples = {0: [], 1: []}

print("\nReading dataset in chunks...", flush=True)

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    chunk.columns = chunk.columns.str.strip()

    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)

    for label in [0, 1]:
        remaining = SAMPLE_PER_CLASS - sum(
            len(item) for item in samples[label]
        )

        if remaining > 0:
            class_data = chunk[chunk["Target"] == label]

            if not class_data.empty:
                selected = class_data.sample(
                    n=min(remaining, len(class_data)),
                    random_state=42
                )

                samples[label].append(selected)

    print("Processed a dataset chunk...", flush=True)

# Combine selected samples
df = pd.concat(
    samples[0] + samples[1],
    ignore_index=True
)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced sample loaded:", len(df), flush=True)
print(df["Target"].value_counts())

# ----------------------------------
# STEP 2: SEPARATE FEATURES AND LABEL
# ----------------------------------

X = df.drop(columns=["Target"])
y = df["Target"]

# Keep only numeric feature columns
X = X.select_dtypes(include=[np.number])

# ----------------------------------
# STEP 3: TRAIN / TEST SPLIT
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ----------------------------------
# STEP 4: TRAIN RANDOM FOREST
# ----------------------------------

print("\nTraining Random Forest model...", flush=True)

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

print("Model training completed!", flush=True)

# ----------------------------------
# STEP 5: MODEL EVALUATION
# ----------------------------------

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------
# STEP 6: SAVE MODEL
# ----------------------------------

Path("models").mkdir(exist_ok=True)

joblib.dump(model, MODEL_FILE, compress=3)

print("\nModel saved successfully!")
print("Location:", MODEL_FILE)
