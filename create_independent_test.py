
import pandas as pd

SOURCE = "dataset/processed/ids_binary.csv"
OUTPUT = "dataset/processed/independent_test.csv"

# Start after the early records used by the training scripts
START_ROW = 1800000
SAMPLES_PER_CLASS = 1000

benign = []
attack = []
rows_seen = 0

print("Creating independent test dataset...")

for chunk in pd.read_csv(SOURCE, chunksize=10000):

    chunk_start = rows_seen
    rows_seen += len(chunk)

    # Skip the earlier training-data region
    if rows_seen <= START_ROW:
        continue

    if chunk_start < START_ROW:
        chunk = chunk.iloc[START_ROW - chunk_start:]

    # Collect BENIGN records
    if len(benign) < SAMPLES_PER_CLASS:
        rows = chunk[chunk["Target"] == 0]
        needed = SAMPLES_PER_CLASS - len(benign)
        benign.extend(rows.head(needed).to_dict("records"))

    # Collect ATTACK records
    if len(attack) < SAMPLES_PER_CLASS:
        rows = chunk[chunk["Target"] == 1]
        needed = SAMPLES_PER_CLASS - len(attack)
        attack.extend(rows.head(needed).to_dict("records"))

    if (len(benign) == SAMPLES_PER_CLASS
            and len(attack) == SAMPLES_PER_CLASS):
        break

if len(benign) < SAMPLES_PER_CLASS or len(attack) < SAMPLES_PER_CLASS:
    raise ValueError(
        f"Not enough records. BENIGN: {len(benign)}, "
        f"ATTACK: {len(attack)}"
    )

test_data = pd.DataFrame(benign + attack)
test_data = test_data.sample(frac=1, random_state=42)

test_data.to_csv(OUTPUT, index=False)

print("\nIndependent test dataset created!")
print("\nClass distribution:")
print(test_data["Target"].value_counts())
print("\nTotal records:", len(test_data))
print("Saved to:", OUTPUT)
