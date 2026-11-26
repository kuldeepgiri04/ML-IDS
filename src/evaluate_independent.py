

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATA_PATH = "dataset/processed/independent_test.csv"

MODELS = {
    "Random Forest": "models/random_forest_ids.pkl",
    "Decision Tree": "models/decision_tree_ids.pkl",
    "Logistic Regression": "models/logistic_regression_ids.pkl"
}

print("Loading independent test dataset...")

data = pd.read_csv(DATA_PATH)
y_test = data["Target"].astype(int)

X = data.drop(columns=["Target"])
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

results = []

for name, path in MODELS.items():

    print("\nEvaluating:", name)

    model = joblib.load(path)

    expected_features = list(model.feature_names_in_)

    missing = [f for f in expected_features if f not in X.columns]

    if missing:
        print("Missing features:", missing[:10])
        continue

    X_model = X[expected_features]

    predictions = model.predict(X_model)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    ))

if results:
    result_df = pd.DataFrame(results)

    print("\n========== INDEPENDENT MODEL COMPARISON ==========")
    print(result_df.to_string(index=False))

    result_df.to_csv(
        "models/independent_evaluation.csv",
        index=False
    )

    print("\nResults saved to models/independent_evaluation.csv")
else:
    print("No models could be evaluated.")
