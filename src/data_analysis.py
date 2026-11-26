
import pandas as pd

# Load the CIC-IDS2017 dataset
file_path = "dataset/raw/Monday-WorkingHours.pcap_ISCX.csv"

print("Loading dataset...")

df = pd.read_csv(file_path, low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")

# Display dataset information
print("\nDataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# Check attack labels
print("\nTraffic labels:")
print(df["Label"].value_counts())

# Check missing values
print("\nMissing values:", df.isnull().sum().sum())
