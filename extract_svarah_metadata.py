import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/svarah/data")
OUTPUT_DIR = Path("data/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(DATA_DIR.glob("test-*.parquet"))

print(f"Found {len(parquet_files)} Parquet files.")

all_dfs = []

for file in parquet_files:
    print(f"\nReading: {file.name}")

    df = pd.read_parquet(file)

    # Extract only the audio path.
    # DO NOT store the embedded audio bytes.
    def extract_audio_path(x):
        if isinstance(x, dict):
            return x.get("path", "")
        return str(x)

    metadata = pd.DataFrame()

    metadata["audio_path"] = df["audio_filepath"].apply(
        extract_audio_path
    )

    metadata["duration"] = df["duration"]
    metadata["text"] = df["text"]
    metadata["gender"] = df["gender"]
    metadata["age-group"] = df["age-group"]
    metadata["primary_language"] = df["primary_language"]
    metadata["native_place_state"] = df["native_place_state"]
    metadata["native_place_district"] = df["native_place_district"]
    metadata["highest_qualification"] = df["highest_qualification"]
    metadata["job_category"] = df["job_category"]
    metadata["occupation_domain"] = df["occupation_domain"]

    # Keep track of which Parquet shard and row contain the audio.
    metadata["source_shard"] = file.name
    metadata["source_row"] = range(len(metadata))

    all_dfs.append(metadata)

    print(f"Records in shard: {len(metadata)}")

# Combine metadata only — no audio bytes.
metadata = pd.concat(all_dfs, ignore_index=True)

metadata.insert(0, "record_id", range(len(metadata)))

output_file = OUTPUT_DIR / "svarah_metadata.csv"

metadata.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print("\n====================================")
print("SVARAH METADATA EXTRACTION COMPLETE")
print("====================================")

print(f"Total records: {len(metadata)}")
print(f"Columns: {list(metadata.columns)}")
print(f"Output: {output_file}")
print(
    f"File size: "
    f"{output_file.stat().st_size / (1024**2):.2f} MB"
)

print("\nFirst 3 rows:")
print(metadata.head(3).to_string(index=False))

print("\nMissing values:")
print(metadata.isnull().sum())

print("\nState counts:")
print(
    metadata["native_place_state"]
    .value_counts()
)