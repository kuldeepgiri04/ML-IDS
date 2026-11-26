import pandas as pd
from pathlib import Path

SOURCE = Path("dataset/processed/cleaned_ids2017.csv")
OUTPUT_DIR = Path("dataset/processed")

CHUNK_SIZE = 20000
BOT_TEST_LIMIT = 1951
BENIGN_TEST_LIMIT = 1951

bot_test = []
benign_test = []
training_benign = []
training_attack = []

print("Reading original dataset in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(SOURCE, chunksize=CHUNK_SIZE, low_memory=False)
):
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    bot_rows = chunk[chunk["Label"] == "Bot"]
    benign_rows = chunk[chunk["Label"] == "BENIGN"]
    attack_rows = chunk[
        (~chunk["Label"].isin(["Bot", "BENIGN"]))
    ]

    # Collect Bot records only for the unseen attack test
    if sum(len(x) for x in bot_test) < BOT_TEST_LIMIT:
        remaining = BOT_TEST_LIMIT - sum(len(x) for x in bot_test)
        bot_test.append(bot_rows.head(remaining))

    # Reserve benign test records from the same source chunks
    if sum(len(x) for x in benign_test) < BENIGN_TEST_LIMIT:
        remaining = BENIGN_TEST_LIMIT - sum(len(x) for x in benign_test)
        benign_test.append(benign_rows.head(remaining))

    # Collect training candidates from non-Bot records
    training_benign.append(benign_rows.iloc[BENIGN_TEST_LIMIT:])
    training_attack.append(attack_rows)

    if (chunk_number + 1) % 20 == 0:
        print(f"Processed {chunk_number + 1} chunks...")

print("\nCombining selected records...")

bot_df = pd.concat(bot_test, ignore_index=True)
benign_test_df = pd.concat(benign_test, ignore_index=True)

benign_train_df = pd.concat(training_benign, ignore_index=True)
attack_train_df = pd.concat(training_attack, ignore_index=True)

# Sample training data
benign_train_df = benign_train_df.sample(
    n=10000, random_state=42
)
attack_train_df = attack_train_df.sample(
    n=10000, random_state=42
)

# Convert labels into binary targets
benign_train_df["Target"] = 0
attack_train_df["Target"] = 1

benign_test_df["Target"] = 0
bot_df["Target"] = 1

train_df = pd.concat(
    [benign_train_df, attack_train_df],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

test_df = pd.concat(
    [benign_test_df, bot_df],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# Remove original category column
train_df.drop(columns=["Label"], inplace=True)
test_df.drop(columns=["Label"], inplace=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(
    OUTPUT_DIR / "unseen_bot_train.csv", index=False
)
test_df.to_csv(
    OUTPUT_DIR / "unseen_bot_test.csv", index=False
)

print("\nDataset creation completed!")
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

print("\nTraining class distribution:")
print(train_df["Target"].value_counts())

print("\nUnseen Bot test distribution:")
print(test_df["Target"].value_counts())

print("\nBot records in training:", 0)
print("Bot records in testing:", len(bot_df))
