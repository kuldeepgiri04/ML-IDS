import pandas as pd

files = {
    "Training": "dataset/processed/unseen_bot_train.csv",
    "Unseen Bot Test": "dataset/processed/unseen_bot_test.csv"
}

features = [
    "Flow IAT Min",
    "Flow Duration",
    "Flow IAT Max",
    "Flow IAT Mean",
    "Flow Packets/s",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Fwd Header Length.1",
    "min_seg_size_forward",
    "Bwd Header Length"
]

for name, path in files.items():

    df = pd.read_csv(path)

    print("\n" + "=" * 55)
    print(name)

    for feature in features:

        if feature not in df.columns:
            continue

        negative = df[df[feature] < 0]

        if not negative.empty:
            print(f"\n{feature}")
            print("Negative records:", len(negative))
            print("Negative values:")
            print(negative[feature].value_counts().head(10))
