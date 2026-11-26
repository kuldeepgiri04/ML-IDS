# Machine-Learning-Based Intrusion Detection System (ML-IDS)

A B.Tech Cyber Security major project that uses supervised machine-learning models to classify network-flow records as **BENIGN** or **ATTACK**. It includes data-preparation scripts, three trained classifiers, model evaluation utilities, and a Streamlit dashboard for testing CSV traffic-flow data.

> **Project type:** Academic prototype  
> **Dataset:** CIC-IDS2017  
> **Algorithms:** Random Forest, Decision Tree, Logistic Regression  
> **Interface:** Streamlit  
> **Author:** Kuldeep Giri  
> **Programme:** B.Tech, Cyber Security  
> **University:** Uttarakhand Technical University (UTU)

## Contents

- [Overview](#overview)
- [Features](#features)
- [System workflow](#system-workflow)
- [Technology stack](#technology-stack)
- [Repository structure](#repository-structure)
- [Model results](#model-results)
- [Setup](#setup)
- [Run the dashboard](#run-the-dashboard)
- [Train the models](#train-the-models)
- [Test with a CSV file](#test-with-a-csv-file)
- [Dataset and model files](#dataset-and-model-files)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

## Overview

The prototype learns patterns from labelled network-flow data and predicts whether each supplied flow is benign or malicious. The project uses CIC-IDS2017 data and evaluates three classification algorithms. A Streamlit dashboard lets a user select a model, upload a compatible CSV, view predictions and download the resulting CSV.

This is an **offline dataset-based prototype**. It does not currently capture packets from a live network or automatically block traffic.

## Features

- Chunk-based dataset processing to reduce memory use when handling large CSV files.
- Cleaning and validation scripts for selected suspicious negative measurement values.
- Balanced representative sample creation for model experiments.
- Stratified train/test split.
- Three trained binary classifiers: Random Forest, Decision Tree and Logistic Regression.
- Evaluation using accuracy, precision, recall, F1-score, confusion matrix and classification report.
- Streamlit dashboard with model status, model comparison, CSV upload, predictions, attack counts and downloadable results.
- Separate experiment investigating detection of Bot traffic excluded from training.

## System workflow

1. Obtain CIC-IDS2017 flow CSV files.
2. Prepare and inspect the dataset using the scripts in `src/`.
3. Clean selected invalid/suspicious measurement records and validate the resulting data.
4. Create a balanced representative sample and split it into training and testing data.
5. Train the three classifiers and evaluate them on the held-out test split.
6. Load the trained model in the Streamlit dashboard and analyse compatible CSV files.

## Technology stack

- Python
- pandas and NumPy
- scikit-learn
- joblib
- Streamlit
- Matplotlib/Streamlit charting
- Git and GitHub

## Repository structure

```text
ML-IDS/
├── models/
│   ├── random_forest_v3.pkl
│   ├── decision_tree_v3.pkl
│   ├── logistic_regression_v3.pkl
│   ├── model_comparison_v3.csv
│   └── ... evaluation files and earlier model versions
├── src/
│   ├── dashboard.py
│   ├── create_representative_sample.py
│   ├── create_train_test_split.py
│   ├── clean_invalid_records.py
│   ├── validate_cleaned_dataset.py
│   ├── train_random_forest_v3.py
│   ├── train_decision_tree_v3.py
│   ├── train_logistic_regression_v3.py
│   ├── compare_models_v3.py
│   └── ... additional preparation and evaluation scripts
├── dataset/
│   ├── raw/          # Dataset files; not committed
│   └── processed/    # Generated files; not committed
├── .gitignore
└── README.md
```

## Model results

The following results are from the project's **3,997-record balanced held-out test split** created from the cleaned dataset. Values are percentages.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | **99.52%** | **99.80%** | 99.25% | **99.52%** |
| Decision Tree | 99.45% | 99.40% | **99.50%** | 99.45% |
| Logistic Regression | 95.35% | 94.06% | 96.80% | 95.41% |

Random Forest achieved the highest accuracy, precision and F1-score in this experiment. Decision Tree achieved the highest recall. These metrics describe performance on this particular test split and should not be interpreted as a guarantee of performance on live traffic or unseen attack families.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kuldeepgiri04/ML-IDS.git
cd ML-IDS
```

### 2. Create and activate a virtual environment

On Linux/Kali Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn joblib streamlit matplotlib
```

If the repository later includes a pinned `requirements.txt`, prefer installing from that file to reproduce the tested environment.

## Run the dashboard

From the repository root, with the virtual environment activated:

```bash
streamlit run src/dashboard.py
```

Streamlit will print a local URL (normally `http://localhost:8501`). Open it in a browser.

The dashboard uses these v3 model files and comparison data:

- `models/random_forest_v3.pkl`
- `models/decision_tree_v3.pkl`
- `models/logistic_regression_v3.pkl`
- `models/model_comparison_v3.csv`

## Train the models

The training scripts expect the generated training and testing CSV files under `dataset/processed/`.

```bash
python src/create_representative_sample.py
python src/create_train_test_split.py
python src/train_random_forest_v3.py
python src/train_decision_tree_v3.py
python src/train_logistic_regression_v3.py
python src/compare_models_v3.py
```

The representative-sampling script currently expects the cleaned source file at:

```text
dataset/processed/cleaned_ids2017_v2.csv
```

That file is generated by the data-cleaning workflow and is not included in this repository because processed datasets are excluded from Git. Review the relevant preparation scripts and obtain the CIC-IDS2017 source data before attempting to reproduce the full training workflow.

## Test with a CSV file

1. Start the dashboard.
2. Select a model in the sidebar.
3. Upload a CSV containing the numeric CIC-IDS2017-compatible network-flow feature columns expected by the selected model.
4. If the CSV contains a `Target` column (`0` for BENIGN and `1` for ATTACK), the dashboard also displays evaluation metrics. Without `Target`, it displays predictions only.
5. Review the prediction counts and detailed results, then download the output CSV if required.

The dashboard uses the feature names stored in the selected model. Missing required feature columns will be reported. Extra numeric columns are ignored.

## Dataset and model files

The CIC-IDS2017 dataset is not redistributed in this repository. Place the downloaded source CSV files in the expected `dataset/raw/` location and follow the preparation scripts in `src/`.

Generated raw and processed dataset files are excluded through `.gitignore` because of their size. Trained model files and evaluation CSVs are included in `models/` for the dashboard demonstration.

**Security note:** Only load `.pkl` model files from a repository or source you trust. Python pickle files can execute code when loaded.

## Limitations

- The current dashboard analyses uploaded CSV flow records; it does not perform live packet capture, real-time network monitoring or automated traffic blocking.
- Results depend on the CIC-IDS2017 dataset, preprocessing choices, selected features and train/test split.
- The balanced test split is useful for comparing the classifiers but does not represent the natural prevalence of benign and malicious traffic in every production network.
- A separate unseen-Bot experiment showed that generalisation to an attack family excluded from training is a significant challenge. In that experiment, Random Forest and Decision Tree did not detect the Bot test attacks, while Logistic Regression detected only a portion. Therefore, the high scores in the main test split should not be presented as proof of reliable zero-day or unseen-attack detection.
- The data-cleaning workflow removes records with negative values in selected measurement features according to the project's validation rule. This is a project-specific preprocessing decision and should be considered when reproducing or interpreting results.

## Future improvements

- Add live packet capture and network-flow extraction.
- Evaluate on additional datasets and independent network traffic.
- Improve generalisation to unseen attack families and address class imbalance.
- Add threshold tuning, cross-validation and repeated experiments.
- Add model explainability and feature-importance analysis.
- Add authentication, logging and deployment hardening before any operational use.

## Disclaimer

This repository is an academic cybersecurity project intended for learning, research and demonstration. It is not a production-grade network security product. Test it only with data and systems you are authorised to use.
