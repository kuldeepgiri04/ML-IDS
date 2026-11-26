
import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

TRAIN_PATH = "dataset/processed/train_data.csv"
TEST_PATH = "dataset/processed/test_data.csv"

print("Loading datasets...")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X_train = train.drop(columns=["Target"])
y_train = train["Target"]

X_test = test.drop(columns=["Target"])
y_test = test["Target"]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining Logistic Regression...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("Precision:", round(
    precision_score(y_test, predictions) * 100, 2
), "%")

print("Recall:", round(
    recall_score(y_test, predictions) * 100, 2
), "%")

print("F1 Score:", round(
    f1_score(y_test, predictions) * 100, 2
), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/logistic_regression_v2.pkl"
)

print("\nNew model saved: models/logistic_regression_v2.pkl")
