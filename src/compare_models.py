
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "dataset/processed/ids_binary.csv"

print("Reading dataset in chunks...")

samples = []
counts = {0: 0, 1: 0}
MAX_PER_CLASS = 10000

for chunk in pd.read_csv(DATA_PATH, chunksize=10000):
    chunk.columns = chunk.columns.str.strip()

    for label in [0, 1]:
        remaining = MAX_PER_CLASS - counts[label]

        if remaining > 0:
            rows = chunk[chunk["Target"] == label]

            if not rows.empty:
                selected = rows.sample(
                    n=min(remaining, len(rows)),
                    random_state=42
                )
                samples.append(selected)
                counts[label] += len(selected)

    if all(counts[label] >= MAX_PER_CLASS for label in [0, 1]):
        break

data = pd.concat(samples, ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data.drop(columns=["Target"])
y = data["Target"]

X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = X.loc[:, X.nunique() > 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ))
    ])
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0)
    })

print("\n========== MODEL COMPARISON ==========")
result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

result_df.to_csv("models/model_comparison.csv", index=False)
print("\nComparison saved to models/model_comparison.csv")
