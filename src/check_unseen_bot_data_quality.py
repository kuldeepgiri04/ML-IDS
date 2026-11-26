import pandas as pd
import numpy as np

files = {
    "Training": "dataset/processed/unseen_bot_train.csv",
    "Unseen Bot Test": "dataset/processed/unseen_bot_test.csv"
}

for name, path in files.items():

    print("\n" + "=" * 50)
    print(name)

    df = pd.read_csv(path)

    X = df.drop(columns=["Target"])

    numeric = X.select_dtypes(include=np.number)

    print("Shape:", df.shape)

    print("Missing values:", numeric.isna().sum().sum())

    print(
        "Infinite values:",
        np.isinf(numeric.to_numpy()).sum()
    )

    print(
        "Negative values:",
        (numeric < 0).sum().sum()
    )

    print("\nLargest feature maximum values:")

    print(
        numeric.max()
        .sort_values(ascending=False)
        .head(10)
    )

print("\nData quality check completed.")
