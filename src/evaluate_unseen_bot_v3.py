import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report

test_df = pd.read_csv("dataset/processed/unseen_bot_test.csv")

X_test = test_df.drop(columns=["Target"])
y_test = test_df["Target"]

models = {
    "Random Forest": "models/random_forest_v3.pkl",
    "Decision Tree": "models/decision_tree_v3.pkl",
    "Logistic Regression": "models/logistic_regression_v3.pkl"
}

for name, path in models.items():
    print("\n" + "=" * 50)
    print(name)

    model = joblib.load(path)
    predictions = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nPrediction Distribution:")
    print(pd.Series(predictions).value_counts().rename({
        0: "BENIGN",
        1: "ATTACK"
    }))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    ))
