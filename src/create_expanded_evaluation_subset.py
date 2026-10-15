import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_FILE = Path(
    "data/metadata/evaluation_subset.csv"
)

SVARAH_FILE = Path(
    "data/metadata/svarah_metadata.csv"
)

LAHAJA_FILE = Path(
    "data/metadata/lahaja_metadata.csv"
)

OUTPUT_FILE = Path(
    "data/metadata/evaluation_subset_60_per_state.csv"
)

TARGET_PER_STATE = 60
RANDOM_SEED = 42


# ============================================================
# LOAD FILES
# ============================================================

current = pd.read_csv(CURRENT_FILE)
svarah = pd.read_csv(SVARAH_FILE)
lahaja = pd.read_csv(LAHAJA_FILE)

print("=" * 70)
print("EXPANDING EVALUATION SET")
print("=" * 70)

print(f"Current evaluation records: {len(current)}")
print()


# ============================================================
# STANDARDIZE METADATA
# ============================================================

svarah["dataset"] = "svarah"
lahaja["dataset"] = "lahaja"

# Common state column for sampling
svarah["state"] = svarah["native_place_state"]
lahaja["state"] = lahaja["native_state"]

# Combine metadata
all_metadata = pd.concat(
    [svarah, lahaja],
    ignore_index=True
)


# ============================================================
# IDENTIFY EXISTING RECORDS
# ============================================================

existing_keys = set(
    zip(
        current["dataset"].astype(str),
        current["source_shard"].astype(str),
        current["source_row"].astype(int)
    )
)

print(f"Existing unique records: {len(existing_keys)}")


# ============================================================
# DETERMINE NEXT eval_id
# ============================================================

next_eval_id = int(
    current["eval_id"].max()
) + 1


# ============================================================
# SAMPLE ADDITIONAL RECORDS
# ============================================================

new_records = []

for (dataset, state), group in all_metadata.groupby(
    ["dataset", "state"]
):

    # --------------------------------------------------------
    # Find how many records from this state already exist
    # --------------------------------------------------------

    if dataset == "svarah":

        existing_group = current[
            (current["dataset"] == dataset) &
            (
                current["native_place_state"] == state
            )
        ]

    else:

        existing_group = current[
            (current["dataset"] == dataset) &
            (
                current["native_state"] == state
            )
        ]

    existing_count = len(existing_group)


    # --------------------------------------------------------
    # If already 60, nothing needs to be added
    # --------------------------------------------------------

    if existing_count >= TARGET_PER_STATE:

        print(
            f"{dataset:8s} | "
            f"{state:20s} | "
            f"already {existing_count}"
        )

        continue


    # --------------------------------------------------------
    # Number of additional recordings needed
    # --------------------------------------------------------

    needed = (
        TARGET_PER_STATE
        - existing_count
    )


    # --------------------------------------------------------
    # Remove records already in current evaluation set
    # --------------------------------------------------------

    candidate_rows = []

    for _, row in group.iterrows():

        key = (
            dataset,
            str(row["source_shard"]),
            int(row["source_row"])
        )

        if key not in existing_keys:

            candidate_rows.append(row)


    candidates = pd.DataFrame(
        candidate_rows
    )


    # --------------------------------------------------------
    # Check whether enough unused records exist
    # --------------------------------------------------------

    if len(candidates) < needed:

        print(
            f"WARNING | {dataset:8s} | "
            f"{state:20s} | "
            f"need={needed} | "
            f"available={len(candidates)} | "
            f"using all available"
        )

        needed = len(candidates)


    # --------------------------------------------------------
    # Randomly select additional recordings
    # --------------------------------------------------------

    sampled = candidates.sample(
        n=needed,
        random_state=RANDOM_SEED
    ).copy()


    # --------------------------------------------------------
    # Assign new eval IDs
    # --------------------------------------------------------

    sampled["eval_id"] = range(
        next_eval_id,
        next_eval_id + len(sampled)
    )

    next_eval_id += len(sampled)


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    new_records.append(sampled)


    print(
        f"{dataset:8s} | "
        f"{state:20s} | "
        f"existing={existing_count:2d} | "
        f"added={needed:2d} | "
        f"final={existing_count + needed:2d}"
    )


# ============================================================
# BUILD NEW RECORD DATAFRAME
# ============================================================

if new_records:

    new_df = pd.concat(
        new_records,
        ignore_index=True
    )

else:

    new_df = pd.DataFrame()


# ============================================================
# MAKE NEW RECORDS MATCH CURRENT SCHEMA
# ============================================================

missing_columns = [
    col
    for col in current.columns
    if col not in new_df.columns
]

for col in missing_columns:

    new_df[col] = None


new_df = new_df[
    current.columns
]


# ============================================================
# COMBINE OLD + NEW
# ============================================================

expanded = pd.concat(
    [current, new_df],
    ignore_index=True
)


# ============================================================
# SORT
# ============================================================

expanded = expanded.sort_values(
    ["dataset", "eval_id"]
).reset_index(
    drop=True
)


# ============================================================
# PRINT FINAL COUNTS
# ============================================================

print()
print("=" * 70)
print("FINAL COUNTS")
print("=" * 70)

print(
    f"Total records: {len(expanded)}"
)


print("\nDataset counts:")

print(
    expanded["dataset"].value_counts()
)


# ============================================================
# STATE COUNTS
# ============================================================

print("\nSvarah state counts:")

svarah_final_counts = (
    expanded[
        expanded["dataset"] == "svarah"
    ]
    ["native_place_state"]
    .value_counts()
    .sort_index()
)

print(
    svarah_final_counts.to_string()
)


print("\nLAHAJA state counts:")

lahaja_final_counts = (
    expanded[
        expanded["dataset"] == "lahaja"
    ]
    ["native_state"]
    .value_counts()
    .sort_index()
)

print(
    lahaja_final_counts.to_string()
)


# ============================================================
# VERIFY FINAL COUNTS
# ============================================================

print(
    "\nMaximum target per state: "
    f"{TARGET_PER_STATE}"
)

print(
    "\nStates below 60 are retained at their "
    "maximum available sample count."
)

print("\nSvarah states below 60:")

print(
    svarah_final_counts[
        svarah_final_counts < TARGET_PER_STATE
    ].to_string()
)

print("\nLAHAJA states below 60:")

print(
    lahaja_final_counts[
        lahaja_final_counts < TARGET_PER_STATE
    ].to_string()
)

print(
    "\nPASS: Evaluation set created without "
    "duplicating or inventing recordings."
)


# ============================================================
# SAVE
# ============================================================

expanded.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


print()
print("Saved:")
print(
    OUTPUT_FILE.resolve()
)