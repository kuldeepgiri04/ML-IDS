
import pandas as pd
import numpy as np

SOURCE = "dataset/processed/cleaned_ids2017_v2.csv"
OUTPUT = "dataset/processed/representative_sample.csv"

SAMPLES_PER_CLASS = 10000
CHUNK_SIZE = 20000
RANDOM_STATE = 42

def prepare_chunk(chunk):
    chunk.columns = chunk.columns.str.strip()

    # Convert attack categories into binary labels
    chunk["Target"] = (
        chunk["Label"].astype(str).str.strip() != "BENIGN"
    ).astype(int)

    # Keep the same feature structure as before
    chunk = chunk.drop(columns=["Label"])
    return chunk

print("PASS 1: Counting records in each chunk...")

counts = []

for chunk in pd.read_csv(
    SOURCE,
    usecols=["Label"],
    chunksize=CHUNK_SIZE
):
    chunk.columns = chunk.columns.str.strip()
    labels = chunk["Label"].astype(str).str.strip()

    counts.append({
        0: int((labels == "BENIGN").sum()),
        1: int((labels != "BENIGN").sum())
    })

total_benign = sum(c[0] for c in counts)
total_attack = sum(c[1] for c in counts)

print("Total BENIGN:", total_benign)
print("Total ATTACK:", total_attack)

def allocate_quotas(class_id, total_records):
    exact = [
        SAMPLES_PER_CLASS * c[class_id] / total_records
        for c in counts
    ]

    quotas = [int(x) for x in exact]
    remaining = SAMPLES_PER_CLASS - sum(quotas)

    order = sorted(
        range(len(exact)),
        key=lambda i: exact[i] - quotas[i],
        reverse=True
    )

    for i in order[:remaining]:
        quotas[i] += 1

    return quotas

benign_quotas = allocate_quotas(0, total_benign)
attack_quotas = allocate_quotas(1, total_attack)

print("\nPASS 2: Randomly sampling across all chunks...")

samples = []

for i, chunk in enumerate(
    pd.read_csv(SOURCE, chunksize=CHUNK_SIZE)
):
    chunk = prepare_chunk(chunk)

    for label, quota in [
        (0, benign_quotas[i]),
        (1, attack_quotas[i])
    ]:
        if quota == 0:
            continue

        rows = chunk[chunk["Target"] == label]

        selected = rows.sample(
            n=quota,
            random_state=RANDOM_STATE + i
        )

        samples.append(selected)

    if (i + 1) % 20 == 0:
        print("Processed chunks:", i + 1)

result = pd.concat(samples, ignore_index=True)

result = result.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

result.to_csv(OUTPUT, index=False)

print("\nRepresentative sample created!")
print(result["Target"].value_counts())
print("Total records:", len(result))
print("Saved to:", OUTPUT)
