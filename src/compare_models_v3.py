import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

TEST_PATH = "dataset/processed/test_data.csv"

MODELS = {
    "Random Forest": "models/random_forest_v3.pkl",
    "Decision Tree": "models/decision_tree_v3.pkl",
    "Logistic Regression": "models/logistic_regression_v3.pkl"
}

print("Loading test dataset...")
test = pd.read_csv(TEST_PATH)

X_test = test.drop(columns=["Target"])
y_test = test["Target"]

results = []

for name, path in MODELS.items():
    print("\nEvaluating:", name)

    model = joblib.load(path)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions) * 100,
        "Precision": precision_score(y_test, predictions) * 100,
        "Recall": recall_score(y_test, predictions) * 100,
        "F1 Score": f1_score(y_test, predictions) * 100
    })

result_df = pd.DataFrame(results).round(2)

print("\n========== V3 MODEL COMPARISON ==========")
print(result_df.to_string(index=False))

result_df.to_csv(
    "models/model_comparison_v3.csv",
    index=False
)

print("\nSaved: models/model_comparison_v3.csv")
