import pandas as pd

path = "dataset/processed/cleaned_ids2017_v2.csv"

columns = [
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

total_rows = 0
negative_counts = {col: 0 for col in columns}

print("Validating complete cleaned dataset...")

for chunk in pd.read_csv(path, chunksize=20000, low_memory=False):
    total_rows += len(chunk)

    for col in columns:
        negative_counts[col] += int((chunk[col] < 0).sum())

print("\nTotal records checked:", total_rows)
print("\nNegative values by feature:")

for col, count in negative_counts.items():
    print(f"{col}: {count}")

print("\nValidation completed!")
