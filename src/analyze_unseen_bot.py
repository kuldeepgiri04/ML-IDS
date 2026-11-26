import pandas as pd
import numpy as np

train = pd.read_csv(
    "dataset/processed/unseen_bot_train.csv"
)

test = pd.read_csv(
    "dataset/processed/unseen_bot_test.csv"
)

features = train.drop(columns=["Target"]).columns

bot = test[test["Target"] == 1]
benign = test[test["Target"] == 0]
known_attack = train[train["Target"] == 1]

print("\nDATASET SHAPES")
print("Known attack training:", known_attack.shape)
print("Unseen Bot testing:", bot.shape)
print("Benign testing:", benign.shape)

print("\nFEATURE DISTRIBUTION COMPARISON")

comparison = pd.DataFrame({
    "Known_Attack_Median": known_attack[features].median(),
    "Bot_Median": bot[features].median(),
    "Benign_Median": benign[features].median()
})

comparison["Bot_vs_Known_Attack_Difference"] = (
    comparison["Bot_Median"] -
    comparison["Known_Attack_Median"]
).abs()

comparison = comparison.replace(
    [np.inf, -np.inf], np.nan
).dropna()

print(
    comparison.sort_values(
        "Bot_vs_Known_Attack_Difference",
        ascending=False
    ).head(15).to_string()
)

print("\nDIAGNOSTIC COMPLETED")
