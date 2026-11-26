import pandas as pd

path = "dataset/processed/cleaned_ids2017.csv"

invalid_count = 0
total_rows = 0
label_counts = {}

for chunk in pd.read_csv(path, chunksize=20000, low_memory=False):

    total_rows += len(chunk)

    # Exclude special TCP window-size fields from this check
    check_columns = [
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

    invalid = pd.Series(False, index=chunk.index)

    for col in check_columns:
        if col in chunk.columns:
            invalid |= pd.to_numeric(
                chunk[col], errors="coerce"
            ) < 0

    bad = chunk[invalid]
    invalid_count += len(bad)

    for label, count in bad["Label"].value_counts().items():
        label_counts[label] = label_counts.get(label, 0) + count

print("Total records:", total_rows)
print("Records with suspicious negative measurements:", invalid_count)
print("Records remaining if removed:", total_rows - invalid_count)

print("\nAffected records by category:")
for label, count in sorted(
    label_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(label, ":", count)
