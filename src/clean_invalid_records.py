
import pandas as pd
from pathlib import Path

SOURCE = Path("dataset/processed/cleaned_ids2017.csv")
OUTPUT = Path("dataset/processed/cleaned_ids2017_v2.csv")

CHUNK_SIZE = 20000

invalid_columns = [
    "Flow Duration",
    "Flow IAT Min",
    "Flow IAT Max",
    "Flow IAT Mean",
    "Flow Packets/s",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward"
]

first_chunk = True
total_records = 0
removed_records = 0

print("Cleaning suspicious negative measurements...")

for chunk in pd.read_csv(
    SOURCE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    total_records += len(chunk)

    invalid = pd.Series(False, index=chunk.index)

    for column in invalid_columns:
        values = pd.to_numeric(
            chunk[column], errors="coerce"
        )
        invalid |= values < 0

    removed_records += int(invalid.sum())

    cleaned = chunk.loc[~invalid].copy()

    cleaned.to_csv(
        OUTPUT,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

print("\nCleaning completed!")
print("Original records:", total_records)
print("Removed records:", removed_records)
print("Remaining records:", total_records - removed_records)
print("Output file:", OUTPUT)
