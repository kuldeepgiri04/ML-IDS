
import pandas as pd

source = "dataset/processed/ids_binary.csv"
output = "dataset/processed/test_traffic.csv"

benign = []
attack = []

for chunk in pd.read_csv(source, chunksize=5000):

    if len(benign) < 50:
        rows = chunk[chunk["Target"] == 0]
        benign.extend(rows.head(50 - len(benign)).to_dict("records"))

    if len(attack) < 50:
        rows = chunk[chunk["Target"] == 1]
        attack.extend(rows.head(50 - len(attack)).to_dict("records"))

    if len(benign) == 50 and len(attack) == 50:
        break

test = pd.DataFrame(benign + attack)
test.to_csv(output, index=False)

print("Test file created successfully!")
print(test["Target"].value_counts())
print("Saved at:", output)
