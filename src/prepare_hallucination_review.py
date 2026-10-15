import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "results/baseline/hallucination_candidates.csv"
)

OUTPUT_FILE = Path(
    "results/baseline/hallucination_review.csv"
)

df = pd.read_csv(INPUT_FILE)

# Keep the information needed for manual validation
review = df[
    [
        "eval_id",
        "dataset",
        "state",
        "district",
        "detected_language",
        "audio_duration",
        "reference",
        "prediction",
        "reference_tokens",
        "prediction_tokens",
        "length_ratio",
        "bigram_repetition",
        "longest_repeated_run",
        "candidate_reason",
    ]
].copy()

# ------------------------------------------------------------
# Human-validation columns
# ------------------------------------------------------------

review["human_label"] = ""
review["reviewer_notes"] = ""

# Allowed labels:
# HALLUCINATION
# ORDINARY_ASR_ERROR
# WRONG_LANGUAGE
# VALID
# UNCERTAIN

review.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("=" * 60)
print("HALLUCINATION HUMAN-REVIEW FILE")
print("=" * 60)

print(f"Candidates: {len(review)}")

print("\nColumns:")
print(review.columns.tolist())

print("\nSaved to:")
print(OUTPUT_FILE.resolve())

print("\nAllowed human labels:")
print("HALLUCINATION")
print("ORDINARY_ASR_ERROR")
print("WRONG_LANGUAGE")
print("VALID")
print("UNCERTAIN")