
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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

DATA_DIR = Path("dataset/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading unseen-Bot training and testing datasets...")

train_df = pd.read_csv(DATA_DIR / "unseen_bot_train.csv")
test_df = pd.read_csv(DATA_DIR / "unseen_bot_test.csv")

X_train = train_df.drop(columns=["Target"])
y_train = train_df["Target"]

X_test = test_df.drop(columns=["Target"])
y_test = test_df["Target"]

# Keep exactly the same feature order
X_test = X_test[X_train.columns]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Bot records in training: 0")

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

    print("\n" + "=" * 55)
    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    ))

    filename = name.lower().replace(" ", "_") + "_v3.pkl"
    joblib.dump(model, MODEL_DIR / filename)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    MODEL_DIR / "unseen_bot_comparison_v3.csv",
    index=False
)

print("\n" + "=" * 55)
print("FINAL UNSEEN-BOT MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nAll v3 models and results saved successfully!")
