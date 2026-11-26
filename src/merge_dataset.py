
import pandas as pd
from pathlib import Path

RAW_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/processed")
OUTPUT_FILE = OUTPUT_DIR / "combined_ids2017.csv"

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start with a fresh output file
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

first_chunk = True

for filename in files:
    file_path = RAW_DIR / filename
    print(f"\nProcessing: {filename}", flush=True)

    for chunk in pd.read_csv(
        file_path,
        chunksize=20000,
        low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()

        # Remove duplicate column names, if any
        chunk = chunk.loc[:, ~chunk.columns.duplicated()]

        # Append each chunk to the output CSV
        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    print(f"Completed: {filename}", flush=True)

print("\nAll datasets merged successfully!")
print("Output:", OUTPUT_FILE)
