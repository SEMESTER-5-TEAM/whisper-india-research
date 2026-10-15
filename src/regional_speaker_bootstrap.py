
import os
import glob
import numpy as np
import pandas as pd

ROOT = "results/lora_generic_hindi"
PRED_FILE = os.path.join(ROOT, "four_condition_per_record.csv")
DATA_DIR = "data/raw/lahaja/data"
OUTPUT_FILE = os.path.join(ROOT, "regional_speaker_bootstrap.csv")

N_BOOTSTRAP = 2000
SEED = 42

REGIONS = {
    "East": ["West Bengal", "Odisha", "Bihar"],
    "Northeast": ["Assam"],
    "South": [
        "Andhra Pradesh", "Telangana", "Karnataka",
        "Kerala", "Tamil Nadu"
    ],
    "West-Central-North": [
        "Goa", "Gujarat", "Maharashtra", "Madhya Pradesh",
        "Rajasthan", "Punjab", "Jammu Kashmir"
    ],
}

STATE_TO_REGION = {
    state: region
    for region, states in REGIONS.items()
    for state in states
}

CONDITIONS = [
    "baseline_forced_hindi",
    "lora_forced_hindi",
    "baseline_default",
    "lora_default",
]

# Load existing per-record results.
pred = pd.read_csv(PRED_FILE)
pred = pred[pred["condition"].isin(CONDITIONS)].copy()

# Recover state metadata from the original test Parquet shards.
shards = {
    os.path.basename(p): p
    for p in glob.glob(os.path.join(DATA_DIR, "*.parquet"))
}

record_keys = pred[
    ["source_shard", "source_row", "sp_id"]
].drop_duplicates(["source_shard", "source_row"])

parts = []
for shard, group in record_keys.groupby("source_shard"):
    if shard not in shards:
        raise FileNotFoundError(f"Missing shard: {shard}")

    metadata = pd.read_parquet(
        shards[shard], columns=["native_state"]
    )
    group = group.copy()
    group["native_state"] = group["source_row"].map(
        metadata["native_state"]
    )
    parts.append(group)

recordings = pd.concat(parts, ignore_index=True)
recordings["region"] = recordings["native_state"].map(
    STATE_TO_REGION
)

if recordings["region"].isna().any():
    raise ValueError("Some recordings have no regional group.")

# Attach region and speaker IDs to the four conditions.
df = pred.merge(
    recordings[
        ["source_shard", "source_row", "sp_id", "region"]
    ],
    on=["source_shard", "source_row", "sp_id"],
    how="left",
    validate="many_to_one",
)

if df["region"].isna().any():
    raise ValueError("Some predictions could not be assigned a region.")

# Ensure each recording has exactly one row per condition.
counts = df.groupby(
    ["source_shard", "source_row"]
)["condition"].nunique()

if not (counts == 4).all():
    raise ValueError("Expected exactly four conditions per recording.")

rng = np.random.default_rng(SEED)
results = []

for region, group in df.groupby("region"):
    # Collapse recordings to speaker-condition sufficient statistics.
    speaker_stats = (
        group.groupby(["sp_id", "condition"], as_index=False)
        .agg(
            reference_words=("reference_words", "sum"),
            word_errors=("word_errors", "sum"),
        )
    )

    speakers = sorted(speaker_stats["sp_id"].unique())
    n_speakers = len(speakers)

    if n_speakers < 2:
        print(f"WARNING: {region} has only {n_speakers} speaker(s).")

    lookup = {}
    for _, row in speaker_stats.iterrows():
        lookup[(row["sp_id"], row["condition"])] = (
            float(row["reference_words"]),
            float(row["word_errors"]),
        )

    # Store each speaker's counts in a stable array.
    # Last dimension: [reference words, word errors].
    arr = np.zeros((n_speakers, len(CONDITIONS), 2), dtype=float)

    for i, speaker in enumerate(speakers):
        for j, condition in enumerate(CONDITIONS):
            arr[i, j] = lookup[(speaker, condition)]

    # Columns: baseline forced, LoRA forced,
    # baseline default, LoRA default.
    totals = arr.sum(axis=0)
    observed_wers = totals[:, 1] / totals[:, 0]

    observed_forced_reduction = (
        observed_wers[0] - observed_wers[1]
    )
    observed_default_reduction = (
        observed_wers[2] - observed_wers[3]
    )

    forced_diffs = np.empty(N_BOOTSTRAP)
    default_diffs = np.empty(N_BOOTSTRAP)

    for b in range(N_BOOTSTRAP):
        sampled = rng.integers(0, n_speakers, size=n_speakers)
        boot_totals = arr[sampled].sum(axis=0)
        boot_wers = boot_totals[:, 1] / boot_totals[:, 0]

        forced_diffs[b] = boot_wers[0] - boot_wers[1]
        default_diffs[b] = boot_wers[2] - boot_wers[3]

    for mode, observed, samples in [
        ("forced_hindi", observed_forced_reduction, forced_diffs),
        ("default", observed_default_reduction, default_diffs),
    ]:
        low, high = np.percentile(samples, [2.5, 97.5])

        results.append({
            "region": region,
            "decoding": mode,
            "speakers": n_speakers,
            "recordings": group[
                ["source_shard", "source_row"]
            ].drop_duplicates().shape[0],
            "baseline_WER": (
                observed_wers[0] if mode == "forced_hindi"
                else observed_wers[2]
            ),
            "lora_WER": (
                observed_wers[1] if mode == "forced_hindi"
                else observed_wers[3]
            ),
            "WER_reduction_pp": observed * 100,
            "CI95_lower_pp": low * 100,
            "CI95_upper_pp": high * 100,
            "bootstrap_replicates": N_BOOTSTRAP,
            "seed": SEED,
        })

out = pd.DataFrame(results)
out.to_csv(OUTPUT_FILE, index=False)

print("\nREGIONAL SPEAKER-CLUSTER BOOTSTRAP")
print(out.round(3).to_string(index=False))
print("\nSaved:", OUTPUT_FILE)
print(
    "\nInterpretation: positive WER reduction means LoRA improved WER. "
    "Intervals are percentile bootstrap intervals over speakers."
)
print(
    "Caution: regional intervals are exploratory, especially for "
    "regions represented by very few speakers."
)
