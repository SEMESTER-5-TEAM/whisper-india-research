
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lora_generic_hindi"

INPUT = RESULTS / "read_extempore_per_record.csv"
OUTPUT = RESULTS / "read_extempore_summary.csv"
COUNTS_OUTPUT = RESULTS / "read_extempore_recording_counts.csv"

KEY = ["source_shard", "source_row"]

d = pd.read_csv(INPUT)

required = KEY + [
    "scenario", "condition", "sp_id", "reference_words",
    "S", "D", "I", "word_errors"
]
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# One recording key must map to exactly one scenario and speaker.
recordings = d.drop_duplicates(KEY).copy()

if recordings.duplicated(KEY).any():
    raise ValueError("Duplicate recording keys found.")

if recordings["scenario"].isna().any():
    raise ValueError("Some recordings have no scenario.")

if recordings.groupby(KEY)["sp_id"].nunique().gt(1).any():
    raise ValueError("A recording key maps to multiple speakers.")

# Count unique recordings, not row numbers across shards.
counts = (
    recordings.groupby("scenario")
    .agg(
        recordings=("source_row", "size"),
        speakers=("sp_id", "nunique")
    )
    .reset_index()
)
counts.to_csv(COUNTS_OUTPUT, index=False)

# Each recording should have one row per condition.
per_condition = d.drop_duplicates(KEY + ["condition"]).copy()

condition_counts = per_condition.groupby(KEY)["condition"].nunique()
if not condition_counts.eq(4).all():
    raise ValueError(
        "Expected exactly four conditions per recording. "
        "Check duplicate or missing recording-condition rows."
    )

# Aggregate the already computed S, D, I and word errors.
summary = (
    per_condition.groupby(["scenario", "condition"], dropna=False)
    .agg(
        recordings=("source_row", "size"),
        speakers=("sp_id", "nunique"),
        reference_words=("reference_words", "sum"),
        S=("S", "sum"),
        D=("D", "sum"),
        I=("I", "sum"),
        word_errors=("word_errors", "sum"),
    )
    .reset_index()
)

summary["corpus_WER"] = (
    summary["word_errors"] / summary["reference_words"]
)

# Internal consistency: word errors must equal S + D + I.
assert (
    summary["word_errors"] == summary["S"] + summary["D"] + summary["I"]
).all(), "S + D + I does not equal word_errors."

summary.to_csv(OUTPUT, index=False)

print("\nUNIQUE TEST RECORDINGS")
print("Total recordings:", len(recordings))
print("Total speakers:", recordings["sp_id"].nunique())
print(counts.to_string(index=False))

print("\nCORRECTED READ VS EXTEMPORE RESULTS")
print(summary.to_string(index=False))

print("\nSaved:", OUTPUT)
print("Saved:", COUNTS_OUTPUT)

assert len(recordings) == 1313, (
    f"Expected 1313 unique recordings, found {len(recordings)}"
)
assert summary.groupby("condition")["recordings"].sum().eq(1313).all(), (
    "Scenario counts do not sum to 1313 per condition."
)

print("\nPASS: all 1,313 recordings are accounted for in all four conditions.")
