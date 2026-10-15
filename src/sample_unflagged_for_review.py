import pandas as pd
from pathlib import Path

BASE = Path("results/baseline")

PREDICTIONS = BASE / "whisper_small_baseline_expanded.csv"
CANDIDATES = BASE / "hallucination_candidates_expanded.csv"
COMPLETED_REVIEW = BASE / "hallucination_review_expanded_completed.csv"
OUTPUT = BASE / "unflagged_random_sample_100.csv"

SEED = 42
SAMPLE_SIZE = 100

predictions = pd.read_csv(PREDICTIONS)
candidates = pd.read_csv(CANDIDATES)
reviewed = pd.read_csv(COMPLETED_REVIEW)

# Use eval_id to identify recordings already flagged by the detector.
flagged_ids = set(candidates["eval_id"].dropna().astype(int))
reviewed_ids = set(reviewed["eval_id"].dropna().astype(int))

# Restrict to successful predictions and exclude every flagged recording.
eligible = predictions[
    predictions["status"].eq("success")
    & ~predictions["eval_id"].isin(flagged_ids)
].copy()

if eligible["eval_id"].duplicated().any():
    raise ValueError("Duplicate eval_id values found in the eligible recordings.")

sample_n = min(SAMPLE_SIZE, len(eligible))
sample = eligible.sample(n=sample_n, random_state=SEED).copy()

sample = sample[
    [
        "eval_id",
        "dataset",
        "state",
        "district",
        "detected_language",
        "audio_duration",
        "reference",
        "prediction",
    ]
].copy()

sample["human_label"] = ""
sample["reviewer_notes"] = ""

sample.to_csv(OUTPUT, index=False, encoding="utf-8")

print("Eligible unflagged recordings:", len(eligible))
print("Random sample size:", len(sample))
print("Random seed:", SEED)
print("\nSample by dataset:")
print(sample["dataset"].value_counts().to_string())
print("\nSaved to:", OUTPUT.resolve())
print("\nNo existing candidate or completed-review file was modified.")