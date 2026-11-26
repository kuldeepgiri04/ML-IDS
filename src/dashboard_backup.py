
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ML Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Machine Learning Based Intrusion Detection System")
st.caption("Network Traffic Classification | CIC-IDS2017 Dataset")

# ---------------- MODEL PATHS ----------------

MODEL_PATHS = {
    "Random Forest": "models/random_forest_ids.pkl",
    "Decision Tree": "models/decision_tree_ids.pkl",
    "Logistic Regression": "models/logistic_regression_ids.pkl"
}

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ---------------- SIDEBAR ----------------

st.sidebar.header("Detection Settings")

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(MODEL_PATHS.keys())
)

st.sidebar.info(
    "Upload a CSV file containing CIC-IDS2017-compatible "
    "network flow features."
)

# ---------------- MODEL STATUS ----------------

st.subheader("Model Status")

status_cols = st.columns(3)

for col, (name, path) in zip(status_cols, MODEL_PATHS.items()):
    with col:
        if os.path.exists(path):
            st.success(f"{name}: Loaded")
        else:
            st.error(f"{name}: Missing")

# ---------------- MODEL COMPARISON ----------------

st.divider()
st.subheader("Model Performance Comparison")

comparison_path = "models/model_comparison.csv"

if os.path.exists(comparison_path):
    comparison = pd.read_csv(comparison_path)

    display_comparison = comparison.copy()

    metric_columns = [
        "Accuracy", "Precision", "Recall", "F1 Score"
    ]

    display_comparison[metric_columns] = (
        display_comparison[metric_columns] * 100
    ).round(2)

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Accuracy Comparison")

    st.bar_chart(
        comparison.set_index("Model")[["Accuracy"]]
    )

else:
    st.warning("Model comparison file not found.")

# ---------------- FILE UPLOAD ----------------

st.divider()
st.subheader("Network Traffic Detection")

uploaded_file = st.file_uploader(
    "Upload network traffic CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, nrows=10000)
        data.columns = data.columns.str.strip()

        st.write("Uploaded data preview:")
        st.dataframe(data.head(), use_container_width=True)

        actual_labels = None

        if "Target" in data.columns:
            actual_labels = data["Target"].astype(int).copy()
            data = data.drop(columns=["Target"])

        # Keep numeric features only
        X = data.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        if X.empty:
            st.error("No numeric network features found.")

        else:
            model_path = MODEL_PATHS[model_name]

            if not os.path.exists(model_path):
                st.error("Selected model file is missing.")

            else:
                model = load_model(model_path)
                expected_features = list(model.feature_names_in_)

                missing_features = [
                    feature for feature in expected_features
                    if feature not in X.columns
                ]

                extra_features = [
                    feature for feature in X.columns
                    if feature not in expected_features
                ]

                if missing_features:
                    st.error(
                        "Required features are missing: "
                        + ", ".join(missing_features[:10])
                    )

                else:
                    X = X[expected_features]
                    predictions = model.predict(X)

                    # Model evaluation
                    if actual_labels is not None:
                        st.divider()
                        st.subheader("Actual vs Predicted Evaluation")

                        accuracy = accuracy_score(actual_labels, predictions)
                        precision = precision_score(
                            actual_labels, predictions, zero_division=0
                        )
                        recall = recall_score(
                            actual_labels, predictions, zero_division=0
                        )
                        f1 = f1_score(
                            actual_labels, predictions, zero_division=0
                        )

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                        c2.metric("Precision", f"{precision * 100:.2f}%")
                        c3.metric("Recall", f"{recall * 100:.2f}%")
                        c4.metric("F1 Score", f"{f1 * 100:.2f}%")

                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(
                            actual_labels, predictions, labels=[0, 1]
                        )

                        cm_df = pd.DataFrame(
                            cm,
                            index=["Actual BENIGN", "Actual ATTACK"],
                            columns=["Predicted BENIGN", "Predicted ATTACK"]
                        )
                        st.dataframe(cm_df, use_container_width=True)

                        st.subheader("Classification Report")
                        report = classification_report(
                            actual_labels,
                            predictions,
                            labels=[0, 1],
                            target_names=["BENIGN", "ATTACK"],
                            output_dict=True,
                            zero_division=0
                        )
                        st.dataframe(
                            pd.DataFrame(report).transpose(),
                            use_container_width=True
                        )

                    # Detection results
                    results = data.copy()
                    results["Prediction"] = np.where(
                        predictions == 1, "ATTACK", "BENIGN"
                    )

                    attacks = int(np.sum(predictions == 1))
                    benign = int(np.sum(predictions == 0))
                    total = len(predictions)

                    st.subheader("Detection Results")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Flows", total)
                    col2.metric("Benign Flows", benign)
                    col3.metric("Attack Flows", attacks)

                    if attacks > 0:
                        st.error(
                            f"ALERT: {attacks} suspicious network flows detected!"
                        )
                    else:
                        st.success("No attacks detected in this upload.")

                    st.subheader("Prediction Distribution")
                    distribution = pd.Series({
                        "BENIGN": benign,
                        "ATTACK": attacks
                    })
                    st.bar_chart(distribution)

                    st.subheader("Detailed Predictions")
                    st.dataframe(results, use_container_width=True)

                    csv_data = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Detection Results",
                        data=csv_data,
                        file_name="intrusion_detection_results.csv",
                        mime="text/csv"
                    )

                    if extra_features:
                        st.info(
                            f"Ignored {len(extra_features)} extra numeric columns."
                        )

    except Exception as error:
        st.error(f"Error processing file: {error}")

# ---------------- MODEL EVALUATION ----------------

if actual_labels is not None:

    st.divider()
    st.subheader("Actual vs Predicted Evaluation")

    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(
        actual_labels, predictions, zero_division=0
    )
    recall = recall_score(
        actual_labels, predictions, zero_division=0
    )
    f1 = f1_score(
        actual_labels, predictions, zero_division=0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Precision", f"{precision * 100:.2f}%")
    col3.metric("Recall", f"{recall * 100:.2f}%")
    col4.metric("F1 Score", f"{f1 * 100:.2f}%")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        actual_labels, predictions, labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual BENIGN", "Actual ATTACK"],
        columns=["Predicted BENIGN", "Predicted ATTACK"]
    )

    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report")

    report = classification_report(
        actual_labels,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )

                    results = data.copy()
                    results["Prediction"] = np.where(
                        predictions == 1, "ATTACK", "BENIGN"
                    )

                    attacks = int(np.sum(predictions == 1))
                    benign = int(np.sum(predictions == 0))
                    total = len(predictions)

                    st.subheader("Detection Results")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Flows", total)
                    col2.metric("Benign Flows", benign)
                    col3.metric("Attack Flows", attacks)

                    if attacks > 0:
                        st.error(
                            f"ALERT: {attacks} suspicious network flows detected!"
                        )
                    else:
                        st.success("No attacks detected in this upload.")

                    st.subheader("Prediction Distribution")

                    distribution = pd.Series({
                        "BENIGN": benign,
                        "ATTACK": attacks
                    })
                    st.bar_chart(distribution)

                    st.subheader("Detailed Predictions")
                    st.dataframe(results, use_container_width=True)

                    csv_data = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Detection Results",
                        data=csv_data,
                        file_name="intrusion_detection_results.csv",
                        mime="text/csv"
                    )

                    if extra_features:
                        st.info(
                            f"Ignored {len(extra_features)} extra "
                            "numeric columns not used during training."
                        )

    except Exception as error:
        st.error(f"Error processing file: {error}")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Academic Project | Machine Learning Based Intrusion Detection System"
)
