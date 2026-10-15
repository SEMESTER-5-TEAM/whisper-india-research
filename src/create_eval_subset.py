import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

OUTPUT_DIR = Path("data/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load metadata
# --------------------------------------------------

svarah = pd.read_csv(
    OUTPUT_DIR / "svarah_metadata.csv"
)

lahaja = pd.read_csv(
    OUTPUT_DIR / "lahaja_metadata.csv"
)

# Add dataset identifier
svarah["dataset"] = "svarah"
lahaja["dataset"] = "lahaja"


# --------------------------------------------------
# Stratified sampling by state
# --------------------------------------------------

def stratified_sample(df, region_column, n_per_region=30):

    # pandas GroupBy.sample keeps the grouping column.
    # This avoids the apply() behavior that removed the
    # state column in the previous version.

    sampled = (
        df.groupby(
            region_column,
            group_keys=False
        )
        .sample(
            n=n_per_region,
            random_state=RANDOM_SEED
        )
        .reset_index(drop=True)
    )

    return sampled


# --------------------------------------------------
# Sample Svarah
# --------------------------------------------------

svarah_subset = stratified_sample(
    svarah,
    "native_place_state",
    n_per_region=30
)


# --------------------------------------------------
# Sample LAHAJA
# --------------------------------------------------

lahaja_subset = stratified_sample(
    lahaja,
    "native_state",
    n_per_region=30
)


# --------------------------------------------------
# Combine
# --------------------------------------------------

eval_subset = pd.concat(
    [svarah_subset, lahaja_subset],
    ignore_index=True
)


# --------------------------------------------------
# Add reproducible evaluation ID
# --------------------------------------------------

eval_subset.insert(
    0,
    "eval_id",
    range(len(eval_subset))
)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_file = OUTPUT_DIR / "evaluation_subset.csv"

eval_subset.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# Report
# --------------------------------------------------

print("\n========================================")
print("EVALUATION SUBSET CREATED")
print("========================================")

print(f"Total records: {len(eval_subset)}")

print("\nDataset counts:")
print(
    eval_subset["dataset"].value_counts()
)

print("\nSvarah state counts:")
print(
    eval_subset[
        eval_subset["dataset"] == "svarah"
    ]["native_place_state"]
    .value_counts()
    .sort_index()
)

print("\nLAHAJA state counts:")
print(
    eval_subset[
        eval_subset["dataset"] == "lahaja"
    ]["native_state"]
    .value_counts()
    .sort_index()
)

print("\nSaved to:")
print(output_file.resolve())