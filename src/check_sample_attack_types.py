
import pandas as pd
import numpy as np

SAMPLE = "dataset/processed/representative_sample.csv"
SOURCE = "dataset/processed/cleaned_ids2017.csv"
CHUNK_SIZE = 20000

sample = pd.read_csv(SAMPLE)
sample.columns = sample.columns.str.strip()

features = [c for c in sample.columns if c != "Target"]

sample_hashes = set(
    pd.util.hash_pandas_object(
        sample[features],
        index=False
    ).values
)

label_counts = {}
matched = 0

print("Checking original attack categories...")

for chunk in pd.read_csv(SOURCE, chunksize=CHUNK_SIZE):
    chunk.columns = chunk.columns.str.strip()

    hashes = pd.util.hash_pandas_object(
        chunk[features],
        index=False
    ).values

    mask = np.isin(hashes, list(sample_hashes))

    if mask.any():
        labels = chunk.loc[mask, "Label"].astype(str)
        matched += len(labels)

        for label, count in labels.value_counts().items():
            label_counts[label] = (
                label_counts.get(label, 0) + count
            )

print("\nMatched sample records:", matched)
print("\nOriginal label distribution:")

for label, count in sorted(
    label_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{label}: {count}")
