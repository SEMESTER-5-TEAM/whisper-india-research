
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "results/lora_generic_hindi"
PRED_FILE = os.path.join(ROOT, "four_condition_per_record.csv")
DATA_DIR = "data/raw/lahaja/data"

# Load unique recordings from the existing evaluation results.
pred = pd.read_csv(PRED_FILE)
keys = pred[["source_shard", "source_row", "sp_id"]].drop_duplicates(
    ["source_shard", "source_row"]
)

# Recover the state metadata from the original test shards.
shards = {
    os.path.basename(path): path
    for path in glob.glob(os.path.join(DATA_DIR, "*.parquet"))
}

parts = []
for shard, group in keys.groupby("source_shard"):
    if shard not in shards:
        raise FileNotFoundError(f"Missing test shard: {shard}")

    metadata = pd.read_parquet(shards[shard], columns=["native_state"])
    group = group.copy()
    group["native_state"] = group["source_row"].map(
        metadata["native_state"]
    )
    parts.append(group)

recordings = pd.concat(parts, ignore_index=True)

if recordings["native_state"].isna().any():
    raise ValueError("Some test recordings could not be mapped to a state.")

# Exploratory geographic groupings.
regions = {
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

state_to_region = {
    state: region
    for region, states in regions.items()
    for state in states
}

recordings["region"] = recordings["native_state"].map(state_to_region)

if recordings["region"].isna().any():
    missing = recordings.loc[
        recordings["region"].isna(), "native_state"
    ].unique()
    raise ValueError(f"States without regional groups: {missing}")

# Join the region labels to the existing four-condition per-record metrics.
df = pred.merge(
    recordings[["source_shard", "source_row", "region", "native_state"]],
    on=["source_shard", "source_row"],
    how="left",
    validate="many_to_one"
)

if df["region"].isna().any():
    raise ValueError("Some predictions did not receive a regional label.")

# Pool S, D, I and reference words to calculate corpus WER per region.
summary = (
    df.groupby(["region", "condition"], as_index=False)
      .agg(
          recordings=("source_row", "size"),
          speakers=("sp_id", "nunique"),
          reference_words=("reference_words", "sum"),
          substitutions=("S", "sum"),
          deletions=("D", "sum"),
          insertions=("I", "sum"),
          word_errors=("word_errors", "sum"),
      )
)

summary["WER"] = summary["word_errors"] / summary["reference_words"]

# Count unique speakers and recordings in each region.
coverage = (
    recordings.groupby("region", as_index=False)
    .agg(
        unique_recordings=("source_row", "size"),
        unique_speakers=("sp_id", "nunique"),
    )
)

summary = summary.merge(coverage, on="region", validate="many_to_one")

# Compare baseline and LoRA within each decoding setting.
wide = summary.pivot(
    index="region", columns="condition", values="WER"
)

wide["forced_WER_reduction_pp"] = 100 * (
    wide["baseline_forced_hindi"] - wide["lora_forced_hindi"]
)
wide["default_WER_reduction_pp"] = 100 * (
    wide["baseline_default"] - wide["lora_default"]
)

summary.to_csv(
    os.path.join(ROOT, "regional_wer_summary.csv"), index=False
)
wide.reset_index().to_csv(
    os.path.join(ROOT, "regional_wer_comparison.csv"), index=False
)
coverage.to_csv(
    os.path.join(ROOT, "regional_coverage.csv"), index=False
)

# Plot baseline and LoRA WER for both decoding settings.
plot_data = summary.pivot(
    index="region", columns="condition", values="WER"
)
ax = (100 * plot_data).plot(kind="bar", figsize=(12, 6))
ax.set_ylabel("Corpus WER (%)")
ax.set_xlabel("Exploratory geographic group")
ax.set_title("Whisper-small and Hindi LoRA: WER by region")
ax.legend(title="Condition")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(ROOT, "regional_wer_comparison.png"), dpi=200
)
plt.close()

print("\nREGIONAL COVERAGE")
print(coverage.to_string(index=False))
print("\nREGIONAL WER COMPARISON")
print(wide.round(4).to_string())
print("\nSaved regional tables and chart in:", ROOT)
