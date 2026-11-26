import pandas as pd

files = {
    "Training": "dataset/processed/unseen_bot_train.csv",
    "Unseen Bot Test": "dataset/processed/unseen_bot_test.csv"
}

for name, path in files.items():

    df = pd.read_csv(path)
    X = df.drop(columns=["Target"])

    negative_counts = (X < 0).sum()
    negative_counts = negative_counts[negative_counts > 0]
    negative_counts = negative_counts.sort_values(
        ascending=False
    )

    print("\n" + "=" * 50)
    print(name)
    print("\nFeatures containing negative values:")
    print(negative_counts.to_string())

    print("\nTotal affected features:", len(negative_counts))
