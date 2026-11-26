
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "dataset/processed/cleaned_ids2017.csv"
OUTPUT_FILE = "dataset/processed/ids_binary.csv"

CHUNK_SIZE = 10000

output_path = Path(OUTPUT_FILE)
output_path.parent.mkdir(parents=True, exist_ok=True)

if output_path.exists():
    output_path.unlink()

first_chunk = True
total_rows = 0
normal_rows = 0
attack_rows = 0

print("Preparing binary IDS dataset...", flush=True)

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    chunk.columns = chunk.columns.str.strip()
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    # Binary classification:
    # BENIGN = 0, all attacks = 1
    chunk["Target"] = (chunk["Label"] != "BENIGN").astype("int8")

    # Remove original text label
    chunk.drop(columns=["Label"], inplace=True)

    # Replace infinite values and remove missing rows
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)

    normal_rows += int((chunk["Target"] == 0).sum())
    attack_rows += int((chunk["Target"] == 1).sum())
    total_rows += len(chunk)

    chunk.to_csv(
        output_path,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    print(f"Processed {total_rows:,} rows", flush=True)

print("\nPreparation completed!")
print("Normal traffic:", normal_rows)
print("Attack traffic:", attack_rows)
print("Total saved:", total_rows)
print("Output:", OUTPUT_FILE)
