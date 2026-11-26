import pandas as pd

FILE = "dataset/processed/cleaned_ids2017.csv"

columns = [
    "Flow Duration",
    "Flow Packets/s",
    "Fwd Header Length",
    "Fwd Header Length.1",
    "Bwd Header Length",
    "min_seg_size_forward",
    "Label"
]

print("Checking original cleaned dataset...")

chunks = pd.read_csv(
    FILE,
    usecols=lambda c: c in columns,
    chunksize=20000,
    low_memory=False
)

count = 0

for chunk in chunks:

    for col in columns[:-1]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(
                chunk[col], errors="coerce"
            )

    mask = (
        (chunk["Fwd Header Length"] < 0) |
        (chunk["Bwd Header Length"] < 0) |
        (chunk["min_seg_size_forward"] < 0) |
        (chunk["Flow Duration"] < 0) |
        (chunk["Flow Packets/s"] < 0)
    )

    bad = chunk[mask]

    if not bad.empty:
        print("\nOriginal source records:")
        print(bad.head(10).to_string(index=False))
        count += len(bad)

print("\nTotal suspicious records found:", count)
