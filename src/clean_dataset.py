
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("dataset/processed/combined_ids2017.csv")
OUTPUT_FILE = Path("dataset/processed/cleaned_ids2017.csv")

CHUNK_SIZE = 10000

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

total_rows = 0
kept_rows = 0
removed_missing = 0
removed_duplicates = 0

seen_rows = set()
first_chunk = True

print("Starting dataset cleaning...", flush=True)

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    chunk.columns = chunk.columns.str.strip()

    # Clean attack labels
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    # Convert infinity values to NaN
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

    total_rows += len(chunk)

    # Remove missing values
    before = len(chunk)
    chunk.dropna(inplace=True)
    removed_missing += before - len(chunk)

    # Remove duplicate rows within and across chunks
    row_hashes = pd.util.hash_pandas_object(
        chunk,
        index=False
    )

    unique_mask = ~row_hashes.isin(seen_rows)
    removed_duplicates += len(chunk) - unique_mask.sum()

    chunk = chunk.loc[unique_mask]
    seen_rows.update(row_hashes[unique_mask].tolist())

    if not chunk.empty:
        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )
        first_chunk = False
        kept_rows += len(chunk)

    print(
        f"Processed {total_rows:,} rows | Kept {kept_rows:,}",
        flush=True
    )

print("\nCleaning completed!")
print("Original rows:", total_rows)
print("Removed missing rows:", removed_missing)
print("Removed duplicate rows:", removed_duplicates)
print("Final rows:", kept_rows)
print("Saved file:", OUTPUT_FILE)
